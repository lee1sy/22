import os, h5py, torch, pickle, faiss, gc
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam, lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

class GaussianUDGLoss(nn.Module):
    def __init__(self, margin=0.5, lambda_gcl=0.1, lambda_pml=0.1, device='cuda'):
        super().__init__()
        self.margin, self.lambda_gcl, self.lambda_pml = margin, lambda_gcl, lambda_pml
        self.device = device
        self.geo_proj = nn.Linear(128, 6).to(device)

    def forward(self, global_des, batch_dict, output_dict, nNeg):
        loss_wtl = self._compute_lazy_triplet(global_des, nNeg)
        gaussians = batch_dict['gaussians']
        fused_feat = output_dict.get('fused_feat')
        valid_mask = output_dict.get('valid_mask')
        loss_gcl = self._compute_gcl(fused_feat, gaussians, valid_mask) if fused_feat is not None else 0
        sampled_feats = output_dict.get('sampled_visual_feats')
        loss_pml = self._compute_pml(sampled_feats, gaussians, valid_mask) if sampled_feats is not None else 0
        return loss_wtl + self.lambda_gcl * loss_gcl + self.lambda_pml * loss_pml

    def _safe_distance(self, x1, x2):
        diff = x1 - x2
        return torch.sqrt(torch.sum(diff * diff, dim=-1) + 1e-8)

    def _compute_lazy_triplet(self, global_des, nNeg):
        neg_des, pos_des, query_des = torch.split(global_des, [nNeg, 1, 1], dim=0)
        d_pos = self._safe_distance(query_des.expand(nNeg, -1), pos_des.expand(nNeg, -1))
        d_neg = self._safe_distance(query_des.expand(nNeg, -1), neg_des)
        loss = torch.clamp(d_pos - d_neg + self.margin, min=0.0)
        mask = (loss > 0).float()
        return (loss * mask).sum() / (mask.sum() + 1e-6)

    def _compute_gcl(self, fused_feat, gaussians, mask):
        geo_gt_input = torch.cat([gaussians[..., :3], gaussians[..., 4:7]], dim=-1)
        geo_gt_norm = torch.norm(geo_gt_input, dim=-1, keepdim=True)
        # 如果 norm 为 0，会产生 NaN，需要处理
        if (geo_gt_norm < 1e-8).any():
            # 对于全零向量，使用单位向量
            geo_gt = geo_gt_input / (geo_gt_norm + 1e-8)
            # 将全零位置设为 [1,0,0,0,0,0]
            zero_mask = (geo_gt_norm.squeeze(-1) < 1e-8)
            geo_gt[zero_mask, 0] = 1.0
            geo_gt[zero_mask, 1:] = 0.0
        else:
            geo_gt = geo_gt_input / geo_gt_norm
        
        feat_proj_input = self.geo_proj(fused_feat)
        feat_proj_norm = torch.norm(feat_proj_input, dim=-1, keepdim=True)
        if (feat_proj_norm < 1e-8).any():
            feat_proj = feat_proj_input / (feat_proj_norm + 1e-8)
            zero_mask = (feat_proj_norm.squeeze(-1) < 1e-8)
            feat_proj[zero_mask, 0] = 1.0
            feat_proj[zero_mask, 1:] = 0.0
        else:
            feat_proj = feat_proj_input / feat_proj_norm
        
        dist = 1 - torch.clamp(F.cosine_similarity(feat_proj, geo_gt, dim=-1), -1.0 + 1e-7, 1.0 - 1e-7)
        return (dist * mask.float()).sum() / (mask.sum() + 1e-6)

    def _compute_pml(self, sampled_feats, gaussians, mask):
        weight = (gaussians[..., 11] * torch.norm(gaussians[..., 4:7], dim=-1)).detach()
        vis_norm = torch.norm(sampled_feats, dim=-1)
        return (vis_norm * weight * mask.float()).sum() / (mask.sum() + 1e-6)

class Trainer:
    def __init__(self, model, train_loader, whole_train_loader, whole_val_set, whole_val_loader,
                 device, num_epochs, ckpt_dir, cache_dir, log_dir, lr, step_size, gamma, margin, freeze_visual=True):
        self.model, self.device = model, device
        self.train_loader, self.whole_train_loader = train_loader, whole_train_loader
        self.whole_val_loader, self.whole_val_set = whole_val_loader, whole_val_set
        self.ckpt_dir, self.cache_dir, self.num_epochs = ckpt_dir, cache_dir, num_epochs
        self.criterion = GaussianUDGLoss(margin=margin, device=device).to(device)
        
        # 为不同模块设置不同的学习率
        # 视觉相关（proj_layer 融合视觉和LiDAR）：较小学习率
        # LiDAR相关（spconv_enc, uncert_net）：较大学习率
        # 聚合层（vlad）：中等学习率
        visual_params = []
        lidar_params = []
        fusion_params = []
        vlad_params = []
        
        # 处理视觉编码器冻结/解冻
        if not freeze_visual:
            # 解冻视觉编码器，用于微调
            for name, param in model.named_parameters():
                if 'visual_enc' in name:
                    param.requires_grad = True
            print("⚠️  视觉编码器已解冻，将用极小学习率微调")
        else:
            # 确保视觉编码器冻结
            for name, param in model.named_parameters():
                if 'visual_enc' in name:
                    param.requires_grad = False
            print("✅ 视觉编码器已冻结（使用预训练特征）")
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if 'visual_enc' in name:
                # 视觉编码器：极小学习率（如果解冻）
                visual_params.append(param)
            elif 'proj_layer' in name:
                # 融合层：较小学习率（视觉特征已经预训练好）
                fusion_params.append(param)
            elif 'spconv_enc' in name or 'uncert_net' in name:
                # LiDAR相关：较大学习率
                lidar_params.append(param)
            elif 'vlad' in name:
                # 聚合层：中等学习率
                vlad_params.append(param)
            else:
                # 其他参数：默认学习率
                lidar_params.append(param)
        
        # 设置不同学习率：LiDAR > VLAD > Fusion > Visual
        # 建议比例：LiDAR:VLAD:Fusion:Visual = 1:0.5:0.1:0.01
        optimizer_params = [
            {'params': lidar_params, 'lr': lr, 'name': 'lidar'},  # LiDAR: 1x
            {'params': vlad_params, 'lr': lr * 0.5, 'name': 'vlad'},  # VLAD: 0.5x
            {'params': fusion_params, 'lr': lr * 0.1, 'name': 'fusion'},  # Fusion: 0.1x
        ]
        
        # 如果视觉编码器解冻，添加极小学习率
        if len(visual_params) > 0:
            optimizer_params.append({'params': visual_params, 'lr': lr * 0.01, 'name': 'visual'})  # Visual: 0.01x
        
        # 过滤掉空列表
        optimizer_params = [p for p in optimizer_params if len(p['params']) > 0]
        
        self.optimizer = Adam(optimizer_params, lr=lr)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        
        # 打印学习率设置
        print("\n" + "="*80)
        print("学习率设置:")
        for param_group in self.optimizer.param_groups:
            print(f"  {param_group.get('name', 'unknown')}: lr={param_group['lr']:.6f}, 参数数量={len(param_group['params'])}")
        print("="*80 + "\n")
        self.writer = SummaryWriter(log_dir)

    def train(self):
        if torch.cuda.device_count() > 1: self.model = nn.DataParallel(self.model)
        self.model.to(self.device)
        for epoch in range(1, self.num_epochs + 1):
            self.build_cache()
            self.model.train()
            epoch_losses = []
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
            for i, (input_dict, nNeg) in enumerate(pbar):
                if input_dict is None: continue
                batch = {k: v.squeeze(0).to(self.device) if isinstance(v, torch.Tensor) else v for k, v in input_dict.items()}
                self.optimizer.zero_grad()
                out = self.model(batch)
                emb = out['embedding']
                
                # 详细检查 NaN 来源
                if torch.isnan(emb).any() or torch.isinf(emb).any():
                    print(f"\n❌ NaN/Inf in embedding at iteration {i}")
                    print(f"  检查模型输出...")
                    if out.get('fused_feat') is not None and torch.isnan(out['fused_feat']).any():
                        print(f"  ❌ fused_feat 包含 NaN")
                        # 检查 fused_feat 的来源
                        print(f"    检查输入...")
                        if torch.isnan(batch['gaussians']).any():
                            print(f"    ❌ gaussians 输入包含 NaN")
                    if out.get('sampled_visual_feats') is not None and torch.isnan(out['sampled_visual_feats']).any():
                        print(f"  ❌ sampled_visual_feats 包含 NaN")
                    if out.get('offsets') is not None and torch.isnan(out['offsets']).any():
                        print(f"  ❌ offsets 包含 NaN")
                        # 检查 offsets 的来源
                        print(f"    检查 UncertaintyNet 输入...")
                        if torch.isnan(batch['gaussians']).any():
                            print(f"    ❌ gaussians 输入包含 NaN")
                        # 检查模型参数
                        for name, param in self.model.named_parameters():
                            if 'uncert_net' in name and torch.isnan(param).any():
                                print(f"    ❌ UncertaintyNet 参数 {name} 包含 NaN（可能是梯度爆炸）")
                    print(f"  跳过此迭代，建议检查模型参数是否包含 NaN")
                    continue
                
                # 检查 embedding norm
                emb_norm = torch.norm(emb, dim=1, keepdim=True)
                if (emb_norm < 1e-8).any():
                    print(f"\n⚠️  WARNING: embedding norm 接近 0 at iteration {i}, norm_min={emb_norm.min().item():.6f}")
                
                # 安全归一化
                emb_normalized = emb / (emb_norm + 1e-8)
                
                # 再次检查
                if torch.isnan(emb_normalized).any() or torch.isinf(emb_normalized).any():
                    print(f"\n❌ NaN/Inf after normalization at iteration {i}")
                    print(f"  跳过此迭代")
                    continue
                
                loss = self.criterion(emb_normalized, batch, out, nNeg[0])
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"\n❌ NaN/Inf loss at iteration {i}")
                    # 检查损失函数的各个组成部分
                    gaussians = batch['gaussians']
                    fused_feat = out.get('fused_feat')
                    valid_mask = out.get('valid_mask')
                    
                    # 检查 _compute_gcl 的输入
                    if fused_feat is not None:
                        geo_gt_input = torch.cat([gaussians[..., :3], gaussians[..., 4:7]], dim=-1)
                        geo_gt_norm = torch.norm(geo_gt_input, dim=-1)
                        if (geo_gt_norm < 1e-8).any():
                            print(f"  ❌ geo_gt_input 包含全零向量，norm_min={geo_gt_norm.min().item():.6f}")
                        if torch.isnan(geo_gt_input).any():
                            print(f"  ❌ geo_gt_input 包含 NaN")
                    
                    print(f"  跳过此迭代")
                    continue
                loss.backward()
                
                # 详细的梯度追溯
                has_nan_grad = False
                max_grad_norm = 0.0
                grad_info = {}  # 按模块分组统计
                
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            print(f"\n{'='*80}")
                            print(f"🔍 梯度追溯 - NaN/Inf 梯度")
                            print(f"{'='*80}")
                            print(f"参数: {name}")
                            print(f"形状: {param.shape}")
                            nan_count = torch.isnan(param.grad).sum().item() if torch.isnan(param.grad).any() else 0
                            inf_count = torch.isinf(param.grad).sum().item() if torch.isinf(param.grad).any() else 0
                            print(f"NaN数量: {nan_count} || Inf数量: {inf_count}")
                            print(f"{'='*80}\n")
                            has_nan_grad = True
                            param.grad.zero_()
                        else:
                            param_grad_norm = param.grad.data.norm(2).item()
                            max_grad_norm = max(max_grad_norm, param_grad_norm)
                            
                            # 按模块分组统计
                            module_name = name.split('.')[0] if '.' in name else name
                            if module_name not in grad_info:
                                grad_info[module_name] = {'count': 0, 'total_norm': 0.0, 'max_norm': 0.0, 'params': []}
                            grad_info[module_name]['count'] += 1
                            grad_info[module_name]['total_norm'] += param_grad_norm
                            grad_info[module_name]['max_norm'] = max(grad_info[module_name]['max_norm'], param_grad_norm)
                            grad_info[module_name]['params'].append((name, param_grad_norm))
                            
                            # 如果单个参数的梯度过大，也清零
                            if param_grad_norm > 100.0:
                                print(f"\n{'='*80}")
                                print(f"🔍 梯度追溯 - 单个参数梯度过大")
                                print(f"{'='*80}")
                                print(f"参数: {name}")
                                print(f"形状: {param.shape}")
                                print(f"梯度范数: {param_grad_norm:.2f}")
                                print(f"参数值范围: [{param.data.min().item():.6f}, {param.data.max().item():.6f}]")
                                print(f"梯度值范围: [{param.grad.data.min().item():.6f}, {param.grad.data.max().item():.6f}]")
                                print(f"{'='*80}\n")
                                param.grad.zero_()
                                has_nan_grad = True
                
                if has_nan_grad:
                    continue
                
                # 计算总梯度范数（不裁剪，只观察）
                # 注意：梯度范数本身不是问题，关键是实际参数更新量 = lr * grad_norm
                # 当前 lr = 1e-5，所以 grad_norm = 12.28 时，实际更新量 = 0.0001228，这是安全的
                # 但如果梯度呈指数增长（1 → 10 → 100 → 1000），那就是梯度爆炸
                total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=float('inf'))
                
                # 计算实际参数更新量（更合理的判断标准）
                actual_update = self.optimizer.param_groups[0]['lr'] * total_norm
                
                # 只在梯度范数很大（> 50）或实际更新量很大（> 0.01）时打印警告
                if total_norm > 50.0 or actual_update > 0.01:
                    print(f"\n{'='*80}")
                    print(f"⚠️  梯度观察 - 总梯度范数较大")
                    print(f"{'='*80}")
                    print(f"总梯度范数: {total_norm:.2f}")
                    print(f"最大单参数梯度范数: {max_grad_norm:.2f}")
                    print(f"学习率: {self.optimizer.param_groups[0]['lr']:.6f}")
                    print(f"实际参数更新量: {actual_update:.6f} (lr * grad_norm)")
                    print(f"说明: 实际更新量 < 0.001 通常是安全的，> 0.01 需要关注")
                    print(f"\n按模块统计:")
                    # 按总梯度范数排序
                    sorted_modules = sorted(grad_info.items(), key=lambda x: x[1]['total_norm'], reverse=True)
                    for module_name, info in sorted_modules[:10]:  # 只显示前10个
                        print(f"  {module_name}:")
                        print(f"    参数数量: {info['count']}")
                        print(f"    总梯度范数: {info['total_norm']:.2f}")
                        print(f"    最大单参数梯度范数: {info['max_norm']:.2f}")
                        # 显示该模块中梯度最大的参数
                        top_params = sorted(info['params'], key=lambda x: x[1], reverse=True)[:3]
                        for param_name, param_norm in top_params:
                            print(f"      - {param_name}: {param_norm:.2f}")
                    print(f"{'='*80}\n")
                
                self.optimizer.step()
                
                # 检查参数是否变成 NaN
                for name, param in self.model.named_parameters():
                    if torch.isnan(param).any():
                        print(f"ERROR: 参数 {name} 变成 NaN，训练可能已损坏")
                        # 尝试恢复：将 NaN 参数设为 0
                        with torch.no_grad():
                            param.data = torch.nan_to_num(param.data, nan=0.0, posinf=0.0, neginf=0.0)
                epoch_losses.append(loss.item())
                pbar.set_postfix(loss=f"{loss.item():.4f}")
            self.scheduler.step()
            self.save_ckpt(epoch)
            self.validate(epoch)

    def build_cache(self):
        self.model.eval()
        path = os.path.join(self.cache_dir, "feat_cache.hdf5")
        nan_count = 0
        with torch.no_grad(), h5py.File(path, 'w') as h5:
            ds_len = len(self.whole_train_loader.dataset)
            feat_set = h5.create_dataset('features', [ds_len, 256], dtype=np.float32)
            ptr = 0
            for batch in tqdm(self.whole_train_loader, desc="Building Cache"):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                emb = self.model(batch)['embedding']
                
                # 检查 NaN/Inf
                if torch.isnan(emb).any() or torch.isinf(emb).any():
                    nan_count += 1
                    emb = torch.zeros_like(emb)
                
                # 安全归一化
                emb_norm = torch.norm(emb, dim=1, keepdim=True)
                emb = emb / (emb_norm + 1e-8)
                
                # 再次检查
                if torch.isnan(emb).any() or torch.isinf(emb).any():
                    emb = torch.zeros_like(emb)
                
                bs = emb.shape[0]
                feat_set[ptr : ptr + bs] = emb.cpu().numpy()
                ptr += bs
        if nan_count > 0:
            print(f"WARNING: Found NaN/Inf in {nan_count} batches during build_cache")
        gc.collect(); torch.cuda.empty_cache()

    @torch.no_grad()
    def validate(self, epoch):
        self.model.eval()
        all_feats = []
        nan_count = 0
        for batch in tqdm(self.whole_val_loader, desc="Validating"):
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            emb = self.model(batch)['embedding']
            
            # 检查 NaN/Inf
            if torch.isnan(emb).any() or torch.isinf(emb).any():
                nan_count += 1
                emb = torch.zeros_like(emb)
            
            # 安全归一化
            emb_norm = torch.norm(emb, dim=1, keepdim=True)
            emb = emb / (emb_norm + 1e-8)
            
            # 再次检查
            if torch.isnan(emb).any() or torch.isinf(emb).any():
                emb = torch.zeros_like(emb)
            
            all_feats.append(emb.cpu().numpy())
        
        feats = np.concatenate(all_feats, axis=0)
        
        # 检查 feats 中的 NaN
        if np.isnan(feats).any() or np.isinf(feats).any():
            print(f"WARNING: Found NaN/Inf in feats, replacing with zeros")
            feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        
        if nan_count > 0:
            print(f"WARNING: Found NaN/Inf in {nan_count} batches during validate")
        
        db_f, q_f = feats[:self.whole_val_set.num_db], feats[self.whole_val_set.num_db:]
        if db_f.shape[0] == 0 or q_f.shape[0] == 0:
            print(f"ERROR: Empty database or query features")
            return
        index = faiss.IndexFlatL2(256)
        index.add(db_f.astype('float32'))
        _, preds = index.search(q_f.astype('float32'), 20)
        gt = self.whole_val_set.getPositives()
        for k in [1, 5, 10]:
            if len(q_f) == 0:
                acc = 0.0
            else:
                acc = sum(np.any(np.in1d(preds[i, :k], gt[i])) for i in range(len(q_f))) / len(q_f) * 100
            print(f"Recall@{k}: {acc:.2f}%")
            self.writer.add_scalar(f"Val/Recall@{k}", acc, epoch)

    def save_ckpt(self, epoch):
        sd = self.model.module.state_dict() if isinstance(self.model, nn.DataParallel) else self.model.state_dict()
        torch.save({'epoch': epoch, 'net': sd}, os.path.join(self.ckpt_dir, f"GS_PR_epoch_{epoch}.pth.tar"))

class Evaluator:
    def __init__(self, model, test_loader, test_set, device):
        self.model, self.test_loader, self.test_set, self.device = model, test_loader, test_set, device

    @torch.no_grad()
    def full_evaluation(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu')
        sd = ckpt['net']
        if list(sd.keys())[0].startswith('module.'): sd = {k[7:]: v for k, v in sd.items()}
        self.model.load_state_dict(sd)
        self.model.to(self.device).eval()
        all_feats = []
        for batch in tqdm(self.test_loader, desc="Testing"):
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            all_feats.append(F.normalize(self.model(batch)['embedding'], p=2, dim=1).cpu().numpy())
        feats = np.concatenate(all_feats, axis=0)
        db_f, q_f = feats[:self.test_set.num_db], feats[self.test_set.num_db:]
        index = faiss.IndexFlatL2(256)
        index.add(db_f.astype('float32'))
        dists, preds = index.search(q_f.astype('float32'), 50)
        gt = self.test_set.getPositives()
        for k in [1, 5, 10]:
            acc = sum(np.any(np.in1d(preds[i, :k], gt[i])) for i in range(len(q_f))) / len(q_f) * 100
            print(f"Test Recall@{k}: {acc:.2f}%")
        self._compute_pr_and_f1(dists, preds, gt)

    def _compute_pr_and_f1(self, dists, preds, gt):
        min_dists = dists[:, 0]
        thresholds = np.unique(np.sort(min_dists))[::len(min_dists)//200]
        precisions, recalls = [], []
        total_gt = len([g for g in gt if len(g) > 0])
        for th in tqdm(thresholds, desc="PR Curve"):
            tp, fp = 0, 0
            for i in range(len(min_dists)):
                if min_dists[i] < th:
                    if np.any(np.in1d(preds[i, 0], gt[i])): tp += 1
                    else: fp += 1
            p = tp / (tp + fp) if (tp + fp) > 0 else 1.0
            r = tp / total_gt if total_gt > 0 else 0
            precisions.append(p)
            recalls.append(r)
        precisions, recalls = np.array(precisions), np.array(recalls)
        f1 = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
        print(f"Max F1-score: {np.max(f1):.4f}")
        return precisions, recalls, np.max(f1)