"""
检查已生成的 .npy 文件，找出包含 NaN 的文件
然后可以针对这些文件重新生成并查看追溯信息
"""
import os
import numpy as np
import pickle
from tqdm import tqdm

INFO_PATH = '/home/james/LSY/GSNET/nuscenes/nuscenes_infos-bs.pkl'
GAUSSIAN_PATH = '/home/james/LSY/11/nuscenes/'

def check_nan_files():
    print("=" * 80)
    print("检查 .npy 文件中的 NaN")
    print("=" * 80)
    
    # 加载 info 文件
    with open(INFO_PATH, 'rb') as f:
        infos = pickle.load(f)
    
    print(f"总样本数: {len(infos)}")
    print(f"检查目录: {GAUSSIAN_PATH}\n")
    
    nan_files = []
    nan_details = []
    
    for idx, info in enumerate(tqdm(infos, desc="检查文件")):
        lidar_info = info['lidar_infos']['LIDAR_TOP']
        fname = os.path.basename(lidar_info['filename']).replace('.bin', '.npy')
        path = os.path.join(GAUSSIAN_PATH, fname)
        
        if not os.path.exists(path):
            continue
        
        try:
            data = np.load(path)
            if np.isnan(data).any() or np.isinf(data).any():
                nan_count = np.isnan(data).sum() if np.isnan(data).any() else 0
                inf_count = np.isinf(data).sum() if np.isinf(data).any() else 0
                nan_mask = np.isnan(data)
                nan_indices = np.where(nan_mask)
                
                nan_files.append((idx, fname, path))
                nan_details.append({
                    'index': idx,
                    'filename': fname,
                    'path': path,
                    'nan_count': nan_count,
                    'inf_count': inf_count,
                    'nan_positions': list(zip(nan_indices[0][:10], nan_indices[1][:10])) if len(nan_indices[0]) > 0 else [],
                    'nan_cols': np.unique(nan_indices[1]).tolist() if len(nan_indices[1]) > 0 else []
                })
        except Exception as e:
            print(f"❌ 读取文件失败 {fname}: {e}")
    
    print(f"\n{'='*80}")
    print(f"检查结果")
    print(f"{'='*80}")
    print(f"包含 NaN/Inf 的文件数: {len(nan_files)}")
    print(f"总文件数: {len(infos)}")
    print(f"NaN 文件比例: {len(nan_files)/len(infos)*100:.2f}%")
    
    if len(nan_files) > 0:
        print(f"\n前20个包含 NaN 的文件:")
        for i, (idx, fname, path) in enumerate(nan_files[:20]):
            detail = nan_details[i]
            print(f"  [{i+1}] 索引 {idx} || 文件: {fname}")
            print(f"      NaN数量: {detail['nan_count']} || Inf数量: {detail['inf_count']}")
            print(f"      包含NaN的列: {detail['nan_cols']}")
            print(f"      NaN位置(前5个): {detail['nan_positions'][:5]}")
        
        # 保存到文件
        output_file = "nan_files_list.txt"
        with open(output_file, 'w') as f:
            f.write(f"包含 NaN/Inf 的文件列表 (共 {len(nan_files)} 个)\n")
            f.write("=" * 80 + "\n")
            for idx, fname, path in nan_files:
                f.write(f"{idx}\t{fname}\t{path}\n")
        print(f"\n✅ 文件列表已保存到: {output_file}")
        
        # 统计哪些列包含 NaN
        all_nan_cols = set()
        for detail in nan_details:
            all_nan_cols.update(detail['nan_cols'])
        print(f"\n包含 NaN 的列索引: {sorted(all_nan_cols)}")
        print(f"  列 7-10 是四元数 (rotations)")
    
    print(f"\n{'='*80}")
    print("检查完成")
    print(f"{'='*80}")

if __name__ == '__main__':
    check_nan_files()
