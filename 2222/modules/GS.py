import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import spconv.pytorch as spconv
import random
import numpy as np
import math
from typing import Optional, Dict, Tuple, List

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class UncertaintyNet(nn.Module):
    def __init__(self, input_dim=14):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.GELU(),
            nn.Linear(32, 2),
            nn.Tanh()
        )
        # 使用更小的权重初始化，避免梯度爆炸
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
    def forward(self, x):
        # 检查输入（追溯来源，不修复）
        if torch.isnan(x).any() or torch.isinf(x).any():
            nan_count = torch.isnan(x).sum().item() if torch.isnan(x).any() else 0
            inf_count = torch.isinf(x).sum().item() if torch.isinf(x).any() else 0
            print(f"\n{'='*80}")
            print(f"🔍 NaN/Inf 来源追溯 - UncertaintyNet 输入")
            print(f"{'='*80}")
            print(f"UncertaintyNet 输入包含 NaN/Inf")
            print(f"NaN数量: {nan_count} || Inf数量: {inf_count} || shape: {x.shape}")
            # 找出 NaN 位置
            if torch.isnan(x).any():
                nan_indices = torch.nonzero(torch.isnan(x), as_tuple=False)
                print(f"NaN 位置 (前20个): {nan_indices[:20].tolist()}")
            print(f"{'='*80}\n")
        
        # 限制输入范围，避免异常值
        x = torch.clamp(x, min=-100.0, max=100.0)
        
        # 逐步检查每个层（追溯来源，不修复）
        for i, layer in enumerate(self.mlp):
            x_prev = x
            x = layer(x)
            if torch.isnan(x).any() or torch.isinf(x).any():
                layer_name = type(layer).__name__
                nan_count = torch.isnan(x).sum().item() if torch.isnan(x).any() else 0
                inf_count = torch.isinf(x).sum().item() if torch.isinf(x).any() else 0
                print(f"\n{'='*80}")
                print(f"🔍 NaN/Inf 来源追溯 - UncertaintyNet 第{i}层({layer_name})")
                print(f"{'='*80}")
                print(f"输出包含 NaN/Inf || NaN数量={nan_count} || Inf数量={inf_count}")
                print(f"输入: shape={x_prev.shape} || min={x_prev.min().item():.6f} || max={x_prev.max().item():.6f} || mean={x_prev.mean().item():.6f}")
                if isinstance(layer, nn.Linear):
                    print(f"权重: shape={layer.weight.shape} || min={layer.weight.min().item():.6f} || max={layer.weight.max().item():.6f} || mean={layer.weight.mean().item():.6f}")
                    if layer.bias is not None:
                        print(f"偏置: shape={layer.bias.shape} || min={layer.bias.min().item():.6f} || max={layer.bias.max().item():.6f}")
                    # 检查权重是否包含 NaN
                    if torch.isnan(layer.weight).any():
                        print(f"❌ 权重包含 NaN！这可能是梯度爆炸导致的")
                    if torch.isinf(layer.weight).any():
                        print(f"❌ 权重包含 Inf！这可能是梯度爆炸导致的")
                print(f"输出: shape={x.shape} || min={x.min().item():.6f} || max={x.max().item():.6f} || mean={x.mean().item():.6f}")
                print(f"{'='*80}\n")
        
        out = x * 0.1
        # 检查最终输出（追溯来源，不修复）
        if torch.isnan(out).any() or torch.isinf(out).any():
            nan_count = torch.isnan(out).sum().item() if torch.isnan(out).any() else 0
            inf_count = torch.isinf(out).sum().item() if torch.isinf(out).any() else 0
            print(f"\n{'='*80}")
            print(f"🔍 NaN/Inf 来源追溯 - UncertaintyNet 最终输出")
            print(f"{'='*80}")
            print(f"最终输出包含 NaN/Inf || NaN数量={nan_count} || Inf数量={inf_count}")
            print(f"{'='*80}\n")
        # 限制输出范围
        out = torch.clamp(out, min=-1.0, max=1.0)
        return out

class VisualEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model('vit_small_patch14_dinov2.lvd142m', pretrained=True, num_classes=0, dynamic_img_size=True)
        for p in self.backbone.parameters():
            p.requires_grad = False
    def forward(self, x):
        B, C, H, W = x.shape
        feat = self.backbone.forward_features(x)
        patch_tokens = feat[:, self.backbone.num_prefix_tokens:, :]
        h, w = H // 14, W // 14
        return patch_tokens.permute(0, 2, 1).reshape(B, 384, h, w)

class PhysicsGatedProjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.bottleneck = nn.Sequential(nn.Linear(384, 64), nn.GELU(), nn.Linear(64, 32), nn.LayerNorm(32))
        self.gate_mlp = nn.Sequential(nn.Linear(4, 16), nn.GELU(), nn.Linear(16, 1), nn.Sigmoid())
    def forward(self, xyz, g14d, v_feats, ext, intr, offsets, mask):
        B, N, _ = xyz.shape
        BV, C, H, W = v_feats.shape
        V = BV // B
        v_feats = v_feats.view(B, V, C, H, W)
        scale = torch.clamp(g14d[:, :, 4:7], 0.01, 10.0)
        opacity = torch.clamp(g14d[:, :, 11:12], 0.0, 1.0)
        gate = self.gate_mlp(torch.cat([scale, opacity], dim=-1))
        pts_homo = torch.cat([xyz, torch.ones(B, N, 1, device=xyz.device)], dim=-1)
        sum_f, sum_m = torch.zeros(B, N, 32, device=xyz.device), torch.zeros(B, N, 1, device=xyz.device)
        for v in range(V):
            cam_pts = torch.bmm(ext[:, v], pts_homo.transpose(1, 2)).transpose(1, 2)
            uv_h = torch.bmm(intr[:, v], cam_pts[:, :, :3].transpose(1, 2)).transpose(1, 2)
            depth = uv_h[:, :, 2:3].clamp(min=1e-6)
            uv = uv_h[:, :, :2] / depth
            grid = torch.stack([2.0 * uv[:, :, 0] / (W * 14.0) - 1.0, 2.0 * uv[:, :, 1] / (H * 14.0) - 1.0], dim=-1)
            grid = (grid + offsets).unsqueeze(2)
            sampled = F.grid_sample(v_feats[:, v], grid, align_corners=False).squeeze(-1).transpose(1, 2)
            v_32 = self.bottleneck(sampled)
            valid_mask = (depth > 0.1) & (grid.abs() <= 1.0).all(dim=-1) & mask.unsqueeze(-1)
            sum_f += (v_32 * gate * valid_mask.float())
            sum_m += valid_mask.float()
        return sum_f / (sum_m + 1e-6)

class SpconvBackbone(nn.Module):
    def __init__(self, in_c, out_c, voxel_size=0.4):
        super().__init__()
        self.vs = voxel_size
        self.net = spconv.SparseSequential(
            spconv.SubMConv3d(in_c, 32, 3, padding=1), nn.BatchNorm1d(32), nn.ReLU(),
            spconv.SubMConv3d(32, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
            spconv.SubMConv3d(64, out_c, 3, padding=1), nn.BatchNorm1d(out_c), nn.ReLU()
        )
    def forward(self, xyz, feat, mask):
        B, N, C = feat.shape
        # 检查输入（追溯来源，不修复）
        if torch.isnan(xyz).any() or torch.isinf(xyz).any():
            nan_count = torch.isnan(xyz).sum().item() if torch.isnan(xyz).any() else 0
            inf_count = torch.isinf(xyz).sum().item() if torch.isinf(xyz).any() else 0
            print(f"\n{'='*80}")
            print(f"🔍 NaN/Inf 来源追溯 - SpconvBackbone xyz 输入")
            print(f"{'='*80}")
            print(f"xyz 输入包含 NaN/Inf || NaN数量={nan_count} || Inf数量={inf_count} || shape={xyz.shape}")
            if torch.isnan(xyz).any():
                nan_indices = torch.nonzero(torch.isnan(xyz), as_tuple=False)
                print(f"NaN 位置 (前20个): {nan_indices[:20].tolist()}")
            print(f"{'='*80}\n")
        if torch.isnan(feat).any() or torch.isinf(feat).any():
            nan_count = torch.isnan(feat).sum().item() if torch.isnan(feat).any() else 0
            inf_count = torch.isinf(feat).sum().item() if torch.isinf(feat).any() else 0
            print(f"\n{'='*80}")
            print(f"🔍 NaN/Inf 来源追溯 - SpconvBackbone feat 输入")
            print(f"{'='*80}")
            print(f"feat 输入包含 NaN/Inf || NaN数量={nan_count} || Inf数量={inf_count} || shape={feat.shape}")
            if torch.isnan(feat).any():
                nan_indices = torch.nonzero(torch.isnan(feat), as_tuple=False)
                print(f"NaN 位置 (前20个): {nan_indices[:20].tolist()}")
                # 检查哪些列包含 NaN
                nan_cols = torch.unique(nan_indices[:, 2]) if len(nan_indices) > 0 else torch.tensor([])
                print(f"包含NaN的列索引: {nan_cols.tolist()}")
            print(f"{'='*80}\n")
        
        origin = xyz.view(-1, 3).min(0)[0]
        v_coo = ((xyz - origin) / self.vs).long().clamp(min=0)
        grid_s = (v_coo.view(-1, 3).max(0)[0] + 1).tolist()[::-1]
        l_feats, l_coos = [], []
        for b in range(B):
            lin = v_coo[b, :, 0] + 1000 * (v_coo[b, :, 1] + 1000 * v_coo[b, :, 2])
            unq, inv = torch.unique(lin, return_inverse=True)
            n_v = unq.shape[0]
            c_b = torch.stack([torch.full((n_v,), b, device=xyz.device, dtype=torch.int32), (unq // 1000000), (unq // 1000 % 1000), (unq % 1000)], dim=1).to(torch.int32)
            f_b = torch.zeros(n_v, C, device=xyz.device).scatter_add_(0, inv.unsqueeze(1).expand(-1, C), feat[b] * mask[b].unsqueeze(-1).float())
            cnt = torch.bincount(inv, weights=mask[b].float(), minlength=n_v).clamp(min=1.0).unsqueeze(1)
            l_feats.append(f_b / cnt)
            l_coos.append(c_b)
        
        # 检查聚合后的特征（追溯来源，不修复）
        l_feats_cat = torch.cat(l_feats)
        if torch.isnan(l_feats_cat).any() or torch.isinf(l_feats_cat).any():
            nan_count = torch.isnan(l_feats_cat).sum().item() if torch.isnan(l_feats_cat).any() else 0
            inf_count = torch.isinf(l_feats_cat).sum().item() if torch.isinf(l_feats_cat).any() else 0
            print(f"\n{'='*80}")
            print(f"🔍 NaN/Inf 来源追溯 - SpconvBackbone 聚合特征")
            print(f"{'='*80}")
            print(f"聚合特征包含 NaN/Inf || NaN数量={nan_count} || Inf数量={inf_count} || shape={l_feats_cat.shape}")
            print(f"输入feat统计: min={feat.min().item():.6f} || max={feat.max().item():.6f} || mean={feat.mean().item():.6f}")
            if torch.isnan(l_feats_cat).any():
                nan_indices = torch.nonzero(torch.isnan(l_feats_cat), as_tuple=False)
                print(f"NaN 位置 (前20个): {nan_indices[:20].tolist()}")
            print(f"{'='*80}\n")
        
        s_tensor = spconv.SparseConvTensor(torch.cat(l_feats), torch.cat(l_coos), grid_s, B)
        
        # 直接前向传播，不逐层检查（避免 BatchNorm1d 的兼容性问题）
        # 如果出现 NaN，会在最终输出时检查
        out_s = self.net(s_tensor)
        
        # 检查最终输出（追溯来源，不修复）
        if hasattr(out_s, 'features') and (torch.isnan(out_s.features).any() or torch.isinf(out_s.features).any()):
            nan_count = torch.isnan(out_s.features).sum().item() if torch.isnan(out_s.features).any() else 0
            inf_count = torch.isinf(out_s.features).sum().item() if torch.isinf(out_s.features).any() else 0
            print(f"\n{'='*80}")
            print(f"🔍 NaN/Inf 来源追溯 - SpconvBackbone 最终输出")
            print(f"{'='*80}")
            print(f"最终输出包含 NaN/Inf || NaN数量={nan_count} || Inf数量={inf_count} || shape={out_s.features.shape}")
            print(f"输入feat统计: min={feat.min().item():.6f} || max={feat.max().item():.6f} || mean={feat.mean().item():.6f}")
            if torch.isnan(out_s.features).any():
                nan_indices = torch.nonzero(torch.isnan(out_s.features), as_tuple=False)
                print(f"NaN 位置 (前20个): {nan_indices[:20].tolist()}")
            # 检查网络参数
            print(f"\n检查网络参数:")
            for name, module in self.net.named_modules():
                if isinstance(module, (spconv.SubMConv3d, spconv.SparseConv3d)):
                    if hasattr(module, 'weight') and module.weight is not None:
                        if torch.isnan(module.weight).any():
                            print(f"  ❌ {name} 权重包含 NaN！")
                        if torch.isinf(module.weight).any():
                            print(f"  ❌ {name} 权重包含 Inf！")
                elif isinstance(module, nn.BatchNorm1d):
                    if hasattr(module, 'weight') and module.weight is not None:
                        if torch.isnan(module.weight).any():
                            print(f"  ❌ {name} BatchNorm 权重包含 NaN！")
                        if torch.isinf(module.weight).any():
                            print(f"  ❌ {name} BatchNorm 权重包含 Inf！")
            print(f"{'='*80}\n")
        out_f = torch.zeros(B, N, out_s.features.shape[-1], device=xyz.device)
        for b in range(B):
            b_mask = out_s.indices[:, 0] == b
            v_feats, v_indices = out_s.features[b_mask], out_s.indices[b_mask]
            lin_v = v_indices[:, 3].long() + 1000 * (v_indices[:, 2].long() + 1000 * v_indices[:, 1].long())
            lin_p = v_coo[b, :, 0] + 1000 * (v_coo[b, :, 1] + 1000 * v_coo[b, :, 2])
            sort_v, perm = lin_v.sort()
            idx = torch.searchsorted(sort_v, lin_p)
            out_f[b] = v_feats[perm[idx.clamp(max=perm.size(0)-1)]]
        
        # 检查最终输出
        if torch.isnan(out_f).any() or torch.isinf(out_f).any():
            nan_count = torch.isnan(out_f).sum().item() if torch.isnan(out_f).any() else 0
            inf_count = torch.isinf(out_f).sum().item() if torch.isinf(out_f).any() else 0
            print(f"❌ SpconvBackbone 最终输出包含 NaN/Inf || NaN数量={nan_count} || Inf数量={inf_count} || shape={out_f.shape}")
            out_f = torch.nan_to_num(out_f, nan=0.0, posinf=0.0, neginf=0.0)
        
        return out_f * mask.unsqueeze(-1)

class GatingContext(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gating_weights = nn.Parameter(torch.randn(dim, dim) * (1 / math.sqrt(dim)))
        self.norm = nn.LayerNorm(dim)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        gates = self.sigmoid(self.norm(torch.matmul(x, self.gating_weights)))
        return x * gates

class NetVLADLoupe(nn.Module):
    def __init__(self, feature_size=128, cluster_size=64, output_dim=256, gating=True):
        super().__init__()
        self.feature_size = feature_size
        self.cluster_size = cluster_size
        self.output_dim = output_dim
        self.cluster_weights = nn.Parameter(torch.randn(feature_size, cluster_size) * (1 / math.sqrt(feature_size)))
        self.cluster_weights2 = nn.Parameter(torch.randn(1, feature_size, cluster_size) * (1 / math.sqrt(feature_size)))
        self.hidden1_weights = nn.Parameter(torch.randn(cluster_size * feature_size, output_dim) * (1 / math.sqrt(feature_size)))
        self.norm1 = nn.LayerNorm(cluster_size)
        self.gating = GatingContext(output_dim) if gating else None
    def forward(self, x, mask=None):
        B, N, D = x.shape
        activation = torch.matmul(x, self.cluster_weights)
        activation = self.norm1(activation)
        activation = F.softmax(activation, dim=-1)
        if mask is not None:
            activation = activation * mask.unsqueeze(-1).to(x.dtype)
        a_sum = activation.sum(-2, keepdim=True)
        a = a_sum * self.cluster_weights2
        vlad = torch.matmul(activation.transpose(1, 2), x)
        vlad = vlad.transpose(1, 2) - a
        vlad = F.normalize(vlad, dim=1, p=2)
        vlad = vlad.reshape(B, -1)
        vlad = F.normalize(vlad, dim=1, p=2)
        vlad = torch.matmul(vlad, self.hidden1_weights)
        if self.gating:
            vlad = self.gating(vlad)
        return vlad

class GaussianFusionNet(nn.Module):
    def __init__(self, hidden_dim=128, output_dim=256):
        super().__init__()
        self.visual_enc = VisualEncoder()
        self.uncert_net = UncertaintyNet()
        self.proj_layer = PhysicsGatedProjection()
        self.spconv_enc = SpconvBackbone(46, hidden_dim)
        self.vlad = NetVLADLoupe(feature_size=hidden_dim, output_dim=output_dim)
    def forward(self, batch):
        g, img, ext, intr = batch['gaussians'], batch['images'], batch['extrinsics'], batch['intrinsics']
        
        # 检查 NaN/Inf 来源（不修复，只追溯）
        if torch.isnan(g).any() or torch.isinf(g).any():
            nan_count = torch.isnan(g).sum().item() if torch.isnan(g).any() else 0
            inf_count = torch.isinf(g).sum().item() if torch.isinf(g).any() else 0
            print(f"\n{'='*80}")
            print(f"🔍 NaN/Inf 来源追溯 - 模型入口")
            print(f"{'='*80}")
            print(f"GaussianFusionNet 输入 gaussians 包含 NaN/Inf")
            print(f"NaN数量: {nan_count} || Inf数量: {inf_count} || shape: {g.shape}")
            
            # 找出 NaN/Inf 的位置
            if torch.isnan(g).any():
                nan_mask = torch.isnan(g)
                nan_indices = torch.nonzero(nan_mask, as_tuple=False)
                print(f"\nNaN 位置分析 (前20个):")
                for idx in nan_indices[:20]:
                    print(f"  位置 [{idx[0].item()}, {idx[1].item()}, {idx[2].item()}]")
                # 检查哪些列包含 NaN
                nan_cols = torch.unique(nan_indices[:, 2]) if len(nan_indices) > 0 else torch.tensor([])
                print(f"  包含NaN的列索引: {nan_cols.tolist()}")
            
            if torch.isinf(g).any():
                inf_mask = torch.isinf(g)
                inf_indices = torch.nonzero(inf_mask, as_tuple=False)
                print(f"\nInf 位置分析 (前20个):")
                for idx in inf_indices[:20]:
                    print(f"  位置 [{idx[0].item()}, {idx[1].item()}, {idx[2].item()}]")
                inf_cols = torch.unique(inf_indices[:, 2]) if len(inf_indices) > 0 else torch.tensor([])
                print(f"  包含Inf的列索引: {inf_cols.tolist()}")
            
            print(f"{'='*80}\n")
        
        B, V, C, H, W = img.shape
        v_feats = self.visual_enc(img.view(B * V, C, H, W))
        offsets = self.uncert_net(g)
        mask = g.abs().sum(-1) > 1e-5
        v_32 = self.proj_layer(g[:, :, :3], g, v_feats, ext, intr, offsets, mask)
        f_46 = torch.cat([g, v_32], dim=-1)
        f_sp = self.spconv_enc(g[:, :, :3], f_46, mask)
        embedding = self.vlad(f_sp, mask=mask)
        return {
            'embedding': embedding,
            'fused_feat': f_sp,
            'sampled_visual_feats': v_32,
            'valid_mask': mask,
            'offsets': offsets
        }