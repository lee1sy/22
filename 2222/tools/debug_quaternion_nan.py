"""
调试四元数 NaN 来源
检查 rotation_matrix_to_quaternion_torch 和特征值分解
"""
import torch
import numpy as np

def rotation_matrix_to_quaternion_torch(R):
    """批量将旋转矩阵转为四元数 (B, 3, 3) -> (B, 4)"""
    B = R.shape[0]
    w = torch.sqrt(1.0 + R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2] + 1e-8) * 0.5
    w_inv = 0.25 / (w + 1e-8)
    x = (R[:, 2, 1] - R[:, 1, 2]) * w_inv
    y = (R[:, 0, 2] - R[:, 2, 0]) * w_inv
    z = (R[:, 1, 0] - R[:, 0, 1]) * w_inv
    q = torch.stack([w, x, y, z], dim=-1)
    return q / (torch.norm(q, dim=-1, keepdim=True) + 1e-8)

def debug_quaternion_nan():
    """调试四元数 NaN 的来源"""
    print("=" * 80)
    print("调试四元数 NaN 来源")
    print("=" * 80)
    
    # 模拟 compute_local_features_gpu 的过程
    K_NEIGHBORS = 20
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建一个可能有问题的点云（共线或共面的点）
    points_torch = torch.randn(100, 3).to(DEVICE)
    
    # 计算协方差矩阵
    mean_p = points_torch.mean(dim=0, keepdim=True)
    centered = points_torch - mean_p
    cov = torch.matmul(centered.t(), centered) / (points_torch.shape[0] - 1)
    
    print(f"\n协方差矩阵检查:")
    print(f"  cov shape: {cov.shape}")
    print(f"  cov min/max: {cov.min().item():.6f} / {cov.max().item():.6f}")
    print(f"  cov 是否包含 NaN: {torch.isnan(cov).any().item()}")
    print(f"  cov 是否包含 Inf: {torch.isinf(cov).any().item()}")
    
    # 特征值分解
    try:
        vals, vecs = torch.linalg.eigh(cov)
        print(f"\n特征值分解检查:")
        print(f"  vals: {vals}")
        print(f"  vals 是否包含 NaN: {torch.isnan(vals).any().item()}")
        print(f"  vals 是否包含 Inf: {torch.isinf(vals).any().item()}")
        print(f"  vecs shape: {vecs.shape}")
        print(f"  vecs 是否包含 NaN: {torch.isnan(vecs).any().item()}")
        print(f"  vecs 是否包含 Inf: {torch.isinf(vecs).any().item()}")
        
        # 检查 vecs 是否是有效的旋转矩阵
        # 旋转矩阵应该是正交的：R @ R.T = I
        identity_check = torch.matmul(vecs, vecs.transpose(-2, -1))
        print(f"\n旋转矩阵正交性检查:")
        print(f"  vecs @ vecs.T 应该接近单位矩阵")
        print(f"  对角线元素: {torch.diagonal(identity_check)}")
        print(f"  是否接近单位矩阵: {torch.allclose(identity_check, torch.eye(3).to(DEVICE), atol=1e-3)}")
        
        # 转为降序
        vals = torch.flip(vals, dims=[-1])
        vecs = torch.flip(vecs, dims=[-1])
        
        # 转换为四元数
        # 需要扩展维度以匹配批量处理
        vecs_batch = vecs.unsqueeze(0)  # (1, 3, 3)
        quat = rotation_matrix_to_quaternion_torch(vecs_batch)
        print(f"\n四元数转换检查:")
        print(f"  quat: {quat}")
        print(f"  quat 是否包含 NaN: {torch.isnan(quat).any().item()}")
        print(f"  quat 是否包含 Inf: {torch.isinf(quat).any().item()}")
        
        # 检查 rotation_matrix_to_quaternion_torch 的中间步骤
        print(f"\n四元数转换中间步骤:")
        trace = 1.0 + vecs[0, 0, 0] + vecs[0, 1, 1] + vecs[0, 2, 2]
        print(f"  trace = 1 + R[0,0] + R[1,1] + R[2,2] = {trace.item():.6f}")
        if trace < 0:
            print(f"  ⚠️  trace < 0，sqrt 会产生 NaN！")
        sqrt_val = torch.sqrt(trace + 1e-8)
        print(f"  sqrt(trace + 1e-8) = {sqrt_val.item():.6f}")
        w = sqrt_val * 0.5
        print(f"  w = {w.item():.6f}")
        
    except Exception as e:
        print(f"❌ 特征值分解失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试边界情况：共线点
    print(f"\n{'='*80}")
    print("测试边界情况：共线点")
    print(f"{'='*80}")
    colinear_points = torch.stack([
        torch.tensor([0.0, 0.0, 0.0]),
        torch.tensor([1.0, 0.0, 0.0]),
        torch.tensor([2.0, 0.0, 0.0]),
        torch.tensor([3.0, 0.0, 0.0]),
    ]).to(DEVICE)
    
    mean_p = colinear_points.mean(dim=0, keepdim=True)
    centered = colinear_points - mean_p
    cov = torch.matmul(centered.t(), centered) / (colinear_points.shape[0] - 1)
    
    print(f"共线点协方差矩阵:")
    print(f"  cov:\n{cov}")
    print(f"  cov 的行列式: {torch.det(cov).item():.6f}")
    print(f"  cov 是否奇异: {torch.det(cov).abs() < 1e-6}")
    
    try:
        vals, vecs = torch.linalg.eigh(cov)
        print(f"  特征值: {vals}")
        print(f"  特征值是否包含 NaN: {torch.isnan(vals).any().item()}")
        print(f"  特征向量是否包含 NaN: {torch.isnan(vecs).any().item()}")
        
        vals = torch.flip(vals, dims=[-1])
        vecs = torch.flip(vecs, dims=[-1])
        
        vecs_batch = vecs.unsqueeze(0)
        quat = rotation_matrix_to_quaternion_torch(vecs_batch)
        print(f"  四元数: {quat}")
        print(f"  四元数是否包含 NaN: {torch.isnan(quat).any().item()}")
        
    except Exception as e:
        print(f"❌ 共线点特征值分解失败: {e}")

if __name__ == '__main__':
    debug_quaternion_nan()
