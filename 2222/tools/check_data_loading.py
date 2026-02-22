"""
数据加载检查脚本
检查 info 文件和高斯点云是否正确加载
"""
import os
import torch
import numpy as np
import pickle
from dataset.NuScenesDataset import DatabaseQueryDataset
from torchvision.transforms import transforms
from tools.utils import load_config

def check_data_loading():
    print("=" * 60)
    print("数据加载检查")
    print("=" * 60)
    
    cfg = load_config('config/config.yaml')
    
    # 1. 检查配置文件路径
    print("\n[1] 检查配置文件路径...")
    info_path = cfg['data']['info_path']
    gaussian_path = cfg['data'].get('gaussian_path', None)
    data_root_dir = cfg['data']['data_root_dir']
    
    print(f"  info_path: {info_path}")
    print(f"  gaussian_path: {gaussian_path}")
    print(f"  data_root_dir: {data_root_dir}")
    
    # 2. 检查 info 文件
    print("\n[2] 检查 info 文件...")
    if not os.path.exists(info_path):
        print(f"  ❌ ERROR: info 文件不存在: {info_path}")
        return
    else:
        print(f"  ✅ info 文件存在")
        try:
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
            print(f"  ✅ info 文件成功加载，包含 {len(infos)} 个样本")
            
            # 检查第一个样本的结构
            if len(infos) > 0:
                first_info = infos[0]
                print(f"  ✅ 第一个样本的键: {list(first_info.keys())}")
                if 'lidar_infos' in first_info:
                    print(f"  ✅ 包含 lidar_infos")
                    if 'LIDAR_TOP' in first_info['lidar_infos']:
                        lidar_info = first_info['lidar_infos']['LIDAR_TOP']
                        print(f"  ✅ 包含 LIDAR_TOP，filename: {lidar_info.get('filename', 'N/A')}")
                    else:
                        print(f"  ❌ ERROR: lidar_infos 中没有 LIDAR_TOP")
                else:
                    print(f"  ❌ ERROR: info 中没有 lidar_infos")
                    
                if 'camera_infos' in first_info:
                    print(f"  ✅ 包含 camera_infos，相机数量: {len(first_info['camera_infos'])}")
                else:
                    print(f"  ❌ ERROR: info 中没有 camera_infos")
        except Exception as e:
            print(f"  ❌ ERROR: 加载 info 文件失败: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # 3. 检查 gaussian_path
    print("\n[3] 检查 gaussian_path...")
    if gaussian_path is None or gaussian_path == "":
        print(f"  ❌ ERROR: gaussian_path 未配置或为空")
        return
    else:
        print(f"  ✅ gaussian_path 已配置: {gaussian_path}")
        if not os.path.exists(gaussian_path):
            print(f"  ❌ ERROR: gaussian_path 目录不存在: {gaussian_path}")
            return
        else:
            print(f"  ✅ gaussian_path 目录存在")
            files = os.listdir(gaussian_path)
            npy_files = [f for f in files if f.endswith('.npy')]
            print(f"  ✅ 目录下包含 {len(npy_files)} 个 .npy 文件")
            if len(npy_files) == 0:
                print(f"  ❌ ERROR: 目录下没有 .npy 文件")
                return
    
    # 4. 尝试加载数据集
    print("\n[4] 尝试加载数据集...")
    try:
        img_transforms = transforms.Compose([
            transforms.Resize(cfg['runner']['resize']),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        val_query_path = cfg['data']['val_query_path']
        dataset = DatabaseQueryDataset(
            data_root_dir=data_root_dir,
            database_path=cfg['data']['database_path'],
            query_path=val_query_path,
            info_path=info_path,
            transforms=img_transforms,
            nonTrivPosDistThres=cfg['runner']['nonTrivPosDistThres'],
            gaussian_path=gaussian_path,
            resize=cfg['runner']['resize']
        )
        print(f"  ✅ 数据集成功创建，总样本数: {len(dataset)}")
        print(f"  ✅ 数据库样本数: {dataset.num_db}, 查询样本数: {dataset.num_query}")
    except Exception as e:
        print(f"  ❌ ERROR: 创建数据集失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 5. 检查实际数据加载
    print("\n[5] 检查实际数据加载...")
    try:
        # 检查前几个样本
        check_indices = [0, min(10, len(dataset)-1), min(100, len(dataset)-1)]
        for idx in check_indices:
            print(f"\n  检查样本 {idx}...")
            try:
                data = dataset[idx]
                
                # 检查 gaussians
                gaussians = data['gaussians']
                print(f"    gaussians shape: {gaussians.shape}")
                print(f"    gaussians dtype: {gaussians.dtype}")
                print(f"    gaussians min/max: {gaussians.min().item():.4f} / {gaussians.max().item():.4f}")
                print(f"    gaussians mean: {gaussians.mean().item():.4f}")
                
                # 检查是否全为 0
                if gaussians.abs().sum() < 1e-6:
                    print(f"    ❌ ERROR: gaussians 全为 0！")
                else:
                    print(f"    ✅ gaussians 包含有效数据")
                    
                # 检查前 3 维（xyz）
                xyz = gaussians[:, :3]
                xyz_sum = xyz.abs().sum()
                print(f"    xyz (前3维) abs sum: {xyz_sum.item():.4f}")
                if xyz_sum < 1e-6:
                    print(f"    ❌ ERROR: xyz 坐标全为 0！")
                else:
                    print(f"    ✅ xyz 坐标有效")
                
                # 检查 mask
                mask = gaussians.abs().sum(-1) > 1e-5
                valid_points = mask.sum().item()
                print(f"    有效点数 (mask): {valid_points} / {gaussians.shape[0]}")
                if valid_points == 0:
                    print(f"    ❌ ERROR: 没有有效点！")
                else:
                    print(f"    ✅ 有 {valid_points} 个有效点")
                
                # 检查 images
                images = data['images']
                print(f"    images shape: {images.shape}")
                print(f"    images dtype: {images.dtype}")
                print(f"    images min/max: {images.min().item():.4f} / {images.max().item():.4f}")
                
                # 检查 extrinsics
                extrinsics = data['extrinsics']
                print(f"    extrinsics shape: {extrinsics.shape}")
                
                # 检查 intrinsics
                intrinsics = data['intrinsics']
                print(f"    intrinsics shape: {intrinsics.shape}")
                
            except Exception as e:
                print(f"    ❌ ERROR: 加载样本 {idx} 失败: {e}")
                import traceback
                traceback.print_exc()
                
    except Exception as e:
        print(f"  ❌ ERROR: 检查数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 6. 检查模型输入格式
    print("\n[6] 检查模型输入格式...")
    try:
        # 获取一个样本
        sample = dataset[0]
        batch = {
            'gaussians': sample['gaussians'].unsqueeze(0),
            'images': sample['images'].unsqueeze(0),
            'extrinsics': sample['extrinsics'].unsqueeze(0),
            'intrinsics': sample['intrinsics'].unsqueeze(0)
        }
        
        print(f"  batch['gaussians'] shape: {batch['gaussians'].shape}")
        print(f"  batch['images'] shape: {batch['images'].shape}")
        print(f"  batch['extrinsics'] shape: {batch['extrinsics'].shape}")
        print(f"  batch['intrinsics'] shape: {batch['intrinsics'].shape}")
        
        # 检查 gaussians 是否全为 0
        g_sum = batch['gaussians'].abs().sum()
        print(f"  gaussians abs sum: {g_sum.item():.4f}")
        if g_sum < 1e-6:
            print(f"  ❌ ERROR: batch 中的 gaussians 全为 0！")
        else:
            print(f"  ✅ batch 中的 gaussians 有效")
        
        # 检查 mask
        mask = batch['gaussians'].abs().sum(-1) > 1e-5
        valid_points = mask.sum().item()
        print(f"  有效点数 (mask): {valid_points} / {batch['gaussians'].shape[1]}")
        if valid_points == 0:
            print(f"  ❌ ERROR: 没有有效点！这会导致模型输出全为 0 或 NaN")
        else:
            print(f"  ✅ 有 {valid_points} 个有效点")
        
        # 尝试前向传播（需要 CUDA）
        print(f"\n  尝试模型前向传播...")
        if not torch.cuda.is_available():
            print(f"  ⚠️  WARNING: CUDA 不可用，跳过模型前向传播（spconv 需要 CUDA）")
            print(f"  ⚠️  请在有 GPU 的环境中运行训练来检查模型输出")
        else:
            try:
                from modules.GS import GaussianFusionNet
                device = torch.device('cuda')
                model = GaussianFusionNet().to(device)
                model.eval()
                
                # 将 batch 移到 GPU
                batch_gpu = {
                    'gaussians': batch['gaussians'].to(device),
                    'images': batch['images'].to(device),
                    'extrinsics': batch['extrinsics'].to(device),
                    'intrinsics': batch['intrinsics'].to(device)
                }
                
                with torch.no_grad():
                    output = model(batch_gpu)
                    embedding = output['embedding']
                    print(f"    embedding shape: {embedding.shape}")
                    print(f"    embedding dtype: {embedding.dtype}")
                    print(f"    embedding min/max: {embedding.min().item():.4f} / {embedding.max().item():.4f}")
                    print(f"    embedding mean: {embedding.mean().item():.4f}")
                    print(f"    embedding norm: {torch.norm(embedding, dim=1).item():.4f}")
                    
                    # 检查是否有 NaN
                    if torch.isnan(embedding).any():
                        print(f"    ❌ ERROR: embedding 包含 NaN！")
                        nan_count = torch.isnan(embedding).sum().item()
                        print(f"    NaN 数量: {nan_count} / {embedding.numel()}")
                    else:
                        print(f"    ✅ embedding 没有 NaN")
                    
                    # 检查是否有 Inf
                    if torch.isinf(embedding).any():
                        print(f"    ❌ ERROR: embedding 包含 Inf！")
                        inf_count = torch.isinf(embedding).sum().item()
                        print(f"    Inf 数量: {inf_count} / {embedding.numel()}")
                    else:
                        print(f"    ✅ embedding 没有 Inf")
                    
                    # 检查是否全为 0
                    if embedding.abs().sum() < 1e-6:
                        print(f"    ❌ ERROR: embedding 全为 0！这会导致 recall 为 0")
                    else:
                        print(f"    ✅ embedding 有效")
                    
                    # 检查 fused_feat
                    fused_feat = output.get('fused_feat')
                    if fused_feat is not None:
                        print(f"    fused_feat shape: {fused_feat.shape}")
                        print(f"    fused_feat abs sum: {fused_feat.abs().sum().item():.4f}")
                        if fused_feat.abs().sum() < 1e-6:
                            print(f"    ❌ WARNING: fused_feat 全为 0")
                        else:
                            print(f"    ✅ fused_feat 有效")
                    
                    # 检查 sampled_visual_feats
                    sampled_feats = output.get('sampled_visual_feats')
                    if sampled_feats is not None:
                        print(f"    sampled_visual_feats shape: {sampled_feats.shape}")
                        print(f"    sampled_visual_feats abs sum: {sampled_feats.abs().sum().item():.4f}")
                        if sampled_feats.abs().sum() < 1e-6:
                            print(f"    ❌ WARNING: sampled_visual_feats 全为 0")
                        else:
                            print(f"    ✅ sampled_visual_feats 有效")
                    
            except Exception as e:
                print(f"    ❌ ERROR: 模型前向传播失败: {e}")
                import traceback
                traceback.print_exc()
                print(f"    ⚠️  这可能是 spconv 需要 CUDA 的问题，但数据检查已完成")
                
    except Exception as e:
        print(f"  ❌ ERROR: 检查模型输入失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 60)
    print("数据加载检查完成")
    print("=" * 60)

if __name__ == '__main__':
    check_data_loading()
