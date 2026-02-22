"""
针对包含 NaN 的文件重新生成，查看详细的追溯信息
直接调用 nuscenes-process.py 中的函数
"""
import os
import sys
import pickle
import numpy as np
import torch

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 直接导入 nuscenes-process.py（需要重命名文件或使用 importlib）
# 由于文件名包含连字符，使用 importlib
import importlib.util
spec = importlib.util.spec_from_file_location("nuscenes_process", 
    os.path.join(project_root, "nuscenes-process.py"))
nuscenes_process = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nuscenes_process)

# 使用导入的模块
compute_local_features_gpu = nuscenes_process.compute_local_features_gpu
rotation_matrix_to_quaternion_torch = nuscenes_process.rotation_matrix_to_quaternion_torch
TARGET_POINTS = nuscenes_process.TARGET_POINTS
K_NEIGHBORS = nuscenes_process.K_NEIGHBORS
VOXEL_SIZE = nuscenes_process.VOXEL_SIZE
DEVICE = nuscenes_process.DEVICE
DATAROOT = nuscenes_process.DATAROOT
INFO_PATH = nuscenes_process.INFO_PATH

def debug_specific_file(index, bin_path):
    """调试单个文件，查看 NaN 产生的详细过程"""
    print(f"\n{'='*80}")
    print(f"调试文件: 索引 {index}")
    print(f"文件路径: {bin_path}")
    print(f"{'='*80}")
    
    # 加载原始点云
    raw_data = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 5)
    pts_np, intensity_np = raw_data[:, :3], raw_data[:, 3] / 255.0
    
    # Voxel 下采样
    coords = np.floor(pts_np / VOXEL_SIZE).astype(np.int32)
    v_ids = coords[:, 0] * 1000000 + coords[:, 1] * 1000 + coords[:, 2]
    _, unq_idx = np.unique(v_ids, return_index=True)
    pts_np, intensity_np = pts_np[unq_idx], intensity_np[unq_idx]
    
    # 多退少补到 4096
    N = pts_np.shape[0]
    if N > TARGET_POINTS:
        idx = np.random.choice(N, TARGET_POINTS, replace=False)
        pts_np, intensity_np = pts_np[idx], intensity_np[idx]
    elif N < TARGET_POINTS:
        idx = np.random.choice(N, TARGET_POINTS - N, replace=True)
        pts_np = np.vstack([pts_np, pts_np[idx]])
        intensity_np = np.concatenate([intensity_np, intensity_np[idx]])
    
    # 搬运到 GPU
    pts_gpu = torch.from_numpy(pts_np).to(DEVICE)
    
    # 调用 compute_local_features_gpu（会打印详细的追溯信息）
    s, r, l, p, o = compute_local_features_gpu(pts_gpu)
    
    # 检查结果
    if torch.isnan(r).any():
        nan_indices = torch.where(torch.isnan(r).any(dim=-1))[0]
        print(f"\n✅ 确认: 发现 {len(nan_indices)} 个点的四元数包含 NaN")
        for idx in nan_indices[:5]:  # 只显示前5个
            print(f"\n点 {idx.item()}:")
            print(f"  位置: {pts_np[idx]}")
            print(f"  四元数: {r[idx]}")
    
    return s, r, l, p, o

def main():
    # 读取包含 NaN 的文件列表
    nan_file = "nan_files_list.txt"
    if not os.path.exists(nan_file):
        print(f"❌ 文件不存在: {nan_file}")
        print("请先运行 tools/check_nan_files.py")
        return
    
    with open(nan_file, 'r') as f:
        lines = f.readlines()[1:]  # 跳过标题行
        nan_files = []
        for line in lines:
            if line.strip():
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    nan_files.append((int(parts[0]), parts[1]))
    
    print(f"找到 {len(nan_files)} 个包含 NaN 的文件")
    print(f"将调试前 3 个文件...\n")
    
    # 加载 info
    with open(INFO_PATH, 'rb') as f:
        infos = pickle.load(f)
    
    # 调试前几个文件
    for idx, fname in nan_files[:3]:
        info = infos[idx]
        raw_p = info['lidar_infos']['LIDAR_TOP']['filename']
        bin_p = raw_p if raw_p.startswith('/') else os.path.join(DATAROOT, raw_p)
        
        if not os.path.exists(bin_p):
            print(f"❌ 文件不存在: {bin_p}")
            continue
        
        debug_specific_file(idx, bin_p)
        print("\n" + "="*80 + "\n")

if __name__ == '__main__':
    main()
