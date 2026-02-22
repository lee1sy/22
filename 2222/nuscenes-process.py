import os
import pickle
import numpy as np
import torch
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# --- 配置 ---
INFO_PATH = '/home/james/LSY/GSNET/nuscenes/nuscenes_infos-bs.pkl'
DATAROOT = '/mnt/nuscenes/'
SAVE_ROOT = '/home/james/LSY/11/nuscenes/'
TARGET_POINTS = 4096
K_NEIGHBORS = 20
VOXEL_SIZE = 0.5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def robust_rotation_matrix_to_quaternion_torch(R):
    """鲁棒的批量旋转矩阵转四元数 (B, 3, 3) -> (B, 4)"""
    # 确保是纯旋转矩阵 (det = 1)，修复 eigh 返回反射矩阵 (det = -1) 的问题
    det = torch.linalg.det(R)
    R_safe = R.clone() 
    R_safe[:, :, 2] *= det.unsqueeze(-1) 
    
    m00, m01, m02 = R_safe[:, 0, 0], R_safe[:, 0, 1], R_safe[:, 0, 2]
    m10, m11, m12 = R_safe[:, 1, 0], R_safe[:, 1, 1], R_safe[:, 1, 2]
    m20, m21, m22 = R_safe[:, 2, 0], R_safe[:, 2, 1], R_safe[:, 2, 2]

    def _copysign(a, b):
        signs_differ = (a < 0) != (b < 0)
        return torch.where(signs_differ, -a, a)

    # 钳制最小值防止精度浮动导致 sqrt(负数)
    q_abs = torch.stack([
        1.0 + m00 + m11 + m22,
        1.0 + m00 - m11 - m22,
        1.0 - m00 + m11 - m22,
        1.0 - m00 - m11 + m22,
    ], dim=-1)
    q_abs = torch.sqrt(torch.clamp(q_abs, min=0.0))

    q = torch.stack([
        q_abs[:, 0],
        _copysign(q_abs[:, 1], m21 - m12),
        _copysign(q_abs[:, 2], m02 - m20),
        _copysign(q_abs[:, 3], m10 - m01),
    ], dim=-1)
    
    return q / (torch.norm(q, dim=-1, keepdim=True) + 1e-8)

@torch.no_grad()
def compute_local_features_gpu(points_torch):
    """全向量化 GPU 计算：KNN + Covariance + Eigen"""
    N = points_torch.shape[0]
    # 1. GPU 极速 KNN
    dist_mat = torch.cdist(points_torch, points_torch) # (N, N)
    dist, indices = torch.topk(dist_mat, k=K_NEIGHBORS + 1, largest=False, sorted=True)
    
    # 2. 准备邻域点 (N, K, 3)
    neighbor_indices = indices[:, 1:] # 去掉点自身
    neighbor_points = points_torch[neighbor_indices] # (N, K, 3)
    
    # 3. 批量计算协方差矩阵
    mean_p = neighbor_points.mean(dim=1, keepdim=True) # (N, 1, 3)
    centered = neighbor_points - mean_p
    cov = torch.matmul(centered.transpose(1, 2), centered) / (K_NEIGHBORS - 1)
    
    # 增加极小的正则化，防止共面、共线或重叠点导致的协方差矩阵退化
    cov = cov + torch.eye(3, device=DEVICE).unsqueeze(0) * 1e-6 
    
    # 4. 批量特征值分解
    vals, vecs = torch.linalg.eigh(cov)
    vals = torch.clamp(vals, min=1e-6)
    
    # 转为降序
    vals = torch.flip(vals, dims=[-1])
    vecs = torch.flip(vecs, dims=[-1])

    # 5. 计算各项指标
    scales = torch.sqrt(vals)
    rotations = robust_rotation_matrix_to_quaternion_torch(vecs)
    linearity = (vals[:, 0] - vals[:, 1]) / vals[:, 0]
    planarity = (vals[:, 1] - vals[:, 2]) / vals[:, 0]
    
    # 6. 计算 Opacity
    mean_dist = dist[:, 1:].mean(dim=1)
    opacities = torch.clamp(torch.exp(-mean_dist), 0.01, 1.0)
    
    return scales, rotations, linearity, planarity, opacities

def process_frame(bin_path, save_path):
    if os.path.exists(save_path): return
    try:
        # CPU 加载与 Voxel 下采样
        raw_data = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 5)
        pts_np, intensity_np = raw_data[:, :3], raw_data[:, 3] / 255.0
        
        coords = np.floor(pts_np / VOXEL_SIZE).astype(np.int32)
        v_ids = coords[:, 0] * 1000000 + coords[:, 1] * 1000 + coords[:, 2]
        _, unq_idx = np.unique(v_ids, return_index=True)
        pts_np, intensity_np = pts_np[unq_idx], intensity_np[unq_idx]
        
        # 多退少补到 4096
        N = pts_np.shape[0]
        if N > TARGET_POINTS:
            idx = np.random.choice(N, TARGET_POINTS, replace=False)
            pts_np, intensity_np = pts_np[idx], intensity_np[idx]
        else:
            idx = np.random.choice(N, TARGET_POINTS - N, replace=True)
            pts_np = np.vstack([pts_np, pts_np[idx]])
            intensity_np = np.concatenate([intensity_np, intensity_np[idx]])

        # --- 搬运到 GPU ---
        pts_gpu = torch.from_numpy(pts_np).to(DEVICE)
        s, r, l, p, o = compute_local_features_gpu(pts_gpu)
        
        # --- 组装并存回 CPU ---
        res = np.zeros((TARGET_POINTS, 14), dtype=np.float32)
        res[:, 0:3] = pts_np
        res[:, 3] = intensity_np
        res[:, 4:7] = s.cpu().numpy()
        res[:, 7:11] = r.cpu().numpy()
        res[:, 11] = o.cpu().numpy()
        res[:, 12] = l.cpu().numpy()
        res[:, 13] = p.cpu().numpy()
        
        np.save(save_path, res)
    except Exception as e:
        # 建议在实际跑的时候把 print 加上，方便排查由于文件损坏导致的问题
        # print(f"Error processing {bin_path}: {e}")
        pass

def main():
    if not os.path.exists(SAVE_ROOT): os.makedirs(SAVE_ROOT)
    with open(INFO_PATH, 'rb') as f: infos = pickle.load(f)
    
    for info in tqdm(infos):
        raw_p = info['lidar_infos']['LIDAR_TOP']['filename']
        bin_p = raw_p if raw_p.startswith('/') else os.path.join(DATAROOT, raw_p)
        save_p = os.path.join(SAVE_ROOT, os.path.basename(bin_p).replace('.bin', '.npy'))
        process_frame(bin_p, save_p)

if __name__ == '__main__':
    main()