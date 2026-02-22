import os, h5py, torch, pickle, numpy as np
from PIL import Image, ImageFile
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import Dataset, default_collate
from pyquaternion import Quaternion

ImageFile.LOAD_TRUNCATED_IMAGES = True

class BaseDataset(Dataset):
    def __init__(self, data_root_dir, info_path, gaussian_path=None, resize=None):
        super().__init__()
        with open(info_path, 'rb') as f: self.infos = pickle.load(f)
        self.dataroot, self.gaussian_path, self.resize = data_root_dir, gaussian_path, resize
        self.num_points = 4096
        self.img_transforms = None

    def get_pose_matrix(self, translation, rotation):
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = Quaternion(rotation).rotation_matrix
        T[:3, 3] = np.array(translation)
        return T

    def load_lidar_data(self, index):
        lidar_info = self.infos[index]['lidar_infos']['LIDAR_TOP']
        fname = os.path.basename(lidar_info['filename']).replace('.bin', '.npy')
        path = os.path.join(self.gaussian_path, fname)
        if not os.path.exists(path): raise FileNotFoundError(f"Missing 3DGS: {path}")
        data = np.load(path)
        
        # 检查 NaN/Inf 来源
        if np.isnan(data).any() or np.isinf(data).any():
            nan_count = np.isnan(data).sum() if np.isnan(data).any() else 0
            inf_count = np.isinf(data).sum() if np.isinf(data).any() else 0
            nan_mask = np.isnan(data)
            inf_mask = np.isinf(data)
            
            # 找出 NaN/Inf 的位置
            nan_indices = np.where(nan_mask)
            inf_indices = np.where(inf_mask)
            
            print(f"\n{'='*80}")
            print(f"🔍 NaN/Inf 来源追溯 - 数据加载阶段")
            print(f"{'='*80}")
            print(f"文件路径: {path}")
            print(f"文件名: {fname}")
            print(f"样本索引: {index}")
            print(f"数据形状: {data.shape}")
            print(f"NaN 数量: {nan_count} || Inf 数量: {inf_count}")
            
            if nan_count > 0:
                print(f"\nNaN 位置分析:")
                print(f"  前10个NaN位置: {list(zip(nan_indices[0][:10], nan_indices[1][:10]))}")
                # 检查哪些列包含 NaN
                nan_cols = np.unique(nan_indices[1]) if len(nan_indices[1]) > 0 else []
                print(f"  包含NaN的列索引: {nan_cols.tolist()}")
                # 检查这些列的数据范围
                for col in nan_cols[:5]:  # 只显示前5列
                    col_data = data[:, col]
                    valid_data = col_data[~np.isnan(col_data)]
                    if len(valid_data) > 0:
                        print(f"    列{col}: 有效数据范围 [{valid_data.min():.6f}, {valid_data.max():.6f}], 均值={valid_data.mean():.6f}")
                    else:
                        print(f"    列{col}: 全部为NaN")
            
            if inf_count > 0:
                print(f"\nInf 位置分析:")
                print(f"  前10个Inf位置: {list(zip(inf_indices[0][:10], inf_indices[1][:10]))}")
                inf_cols = np.unique(inf_indices[1]) if len(inf_indices[1]) > 0 else []
                print(f"  包含Inf的列索引: {inf_cols.tolist()}")
            
            # 检查数据统计
            print(f"\n数据统计:")
            print(f"  整体: min={np.nanmin(data):.6f} || max={np.nanmax(data):.6f} || mean={np.nanmean(data):.6f}")
            for col in range(min(14, data.shape[1])):
                col_data = data[:, col]
                valid_data = col_data[~np.isnan(col_data) & ~np.isinf(col_data)]
                if len(valid_data) > 0:
                    print(f"  列{col}: min={valid_data.min():.6f} || max={valid_data.max():.6f} || mean={valid_data.mean():.6f} || NaN数={np.isnan(col_data).sum()}")
            print(f"{'='*80}\n")
        
        N = data.shape[0]
        if N > self.num_points:
            data = data[np.random.permutation(N)[:self.num_points]]
        elif N < self.num_points:
            data = np.concatenate([data, np.zeros((self.num_points - N, 14))], axis=0)
        result = torch.from_numpy(data).float()
        
        # 检查转换后是否仍有 NaN
        if torch.isnan(result).any() or torch.isinf(result).any():
            print(f"⚠️  样本 {index} 转换后仍包含 NaN/Inf")
            print(f"  原始数据NaN数: {nan_count if 'nan_count' in locals() else 0}")
            print(f"  转换后NaN数: {torch.isnan(result).sum().item()}")
        
        return result

    def load_data_with_matrices(self, index):
        channels = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']
        info = self.infos[index]
        gaussians = self.load_lidar_data(index)
        imgs = []
        for ch in channels:
            cam_info = info['camera_infos'][ch]
            img = Image.open(os.path.join(self.dataroot, cam_info['filename']))
            if self.resize: img = img.resize((self.resize[1], self.resize[0]), Image.BILINEAR)
            imgs.append(self.img_transforms(img))
        l_calib = info['lidar_infos']['LIDAR_TOP']['calibrated_sensor']
        T_l2e = self.get_pose_matrix(l_calib['translation'], l_calib['rotation'])
        exts, ints = [], []
        for ch in channels:
            c_calib = info['camera_infos'][ch]['calibrated_sensor']
            T_c2e = self.get_pose_matrix(c_calib['translation'], c_calib['rotation'])
            exts.append(np.linalg.inv(T_c2e) @ T_l2e)
            K = np.array(c_calib['camera_intrinsic'], dtype=np.float32)
            if self.resize:
                sx, sy = self.resize[1]/1600.0, self.resize[0]/900.0
                K[0,0] *= sx; K[0,2] *= sx; K[1,1] *= sy; K[1,2] *= sy
            ints.append(K)
        return {'images': torch.stack(imgs), 'gaussians': gaussians, 'extrinsics': torch.from_numpy(np.stack(exts)).float(), 'intrinsics': torch.from_numpy(np.stack(ints)).float()}

class TripletDataset(BaseDataset):
    def __init__(self, data_root_dir, database_path, query_path, info_path, cache_dir, img_transforms, nNeg, nNegSample, nonTrivPosDistThres, posDistThr, margin, gaussian_path=None, resize=None):
        super().__init__(data_root_dir, info_path, gaussian_path, resize)
        self.data_base, self.queries = np.load(database_path), np.load(query_path)
        self.nNeg, self.nNegSample, self.margin = nNeg, nNegSample, margin
        self.nonTrivPosDistThres, self.posDistThr = nonTrivPosDistThres, posDistThr
        self.img_transforms = img_transforms
        self.cache = os.path.join(cache_dir, 'feat_cache.hdf5')
        self.negCache = [np.empty((0,)) for _ in range(len(self.queries))]
        knn = NearestNeighbors().fit(self.data_base[:, 1:])
        self.nontrivial_positives = list(knn.radius_neighbors(self.queries[:, 1:], radius=nonTrivPosDistThres, return_distance=False))
        potential_positives = list(knn.radius_neighbors(self.queries[:, 1:], radius=posDistThr, return_distance=False))
        self.potential_negatives = [np.setdiff1d(np.arange(len(self.data_base)), p, assume_unique=True) for p in potential_positives]

    def _get_feat_safe(self, h5_feat, indices):
        idx_list = indices.tolist() if isinstance(indices, np.ndarray) else list(indices)
        sort_idx = np.argsort(idx_list)
        sorted_indices = np.array(idx_list)[sort_idx].tolist()
        data = h5_feat[sorted_indices]
        rev_idx = np.argsort(sort_idx)
        return torch.from_numpy(data[rev_idx]).float()

    def __getitem__(self, index):
        if len(self.potential_negatives[index]) == 0: return None
        with h5py.File(self.cache, mode='r') as h5:
            feat = h5.get('features')
            q_f = torch.tensor(feat[index + len(self.data_base)])
            pos_indices = self.nontrivial_positives[index]
            if len(pos_indices) == 0: return None
            p_f_all = self._get_feat_safe(feat, pos_indices)
            dists = torch.norm(q_f - p_f_all, dim=1)
            pos_idx_in_db = pos_indices[torch.argmax(dists).item()]
            pos_dist = dists.max()
            neg_pool = np.random.choice(self.potential_negatives[index], min(self.nNegSample, len(self.potential_negatives[index])), replace=False)
            neg_pool = np.unique(np.concatenate([self.negCache[index], neg_pool]).astype(int))
            n_f_all = self._get_feat_safe(feat, neg_pool)
            n_dists = torch.norm(q_f - n_f_all, dim=1)
            violating = n_dists < pos_dist + self.margin
            selected_neg_indices = neg_pool[torch.topk(n_dists, self.nNeg, largest=False).indices] if violating.sum() < 1 else neg_pool[n_dists.argsort()[:self.nNeg]]
            self.negCache[index] = selected_neg_indices
        q_data = self.load_data_with_matrices(int(self.queries[index][0]))
        p_data = self.load_data_with_matrices(int(self.data_base[pos_idx_in_db][0]))
        n_datas = [self.load_data_with_matrices(int(self.data_base[i][0])) for i in selected_neg_indices]
        return {'images': torch.stack([n['images'] for n in n_datas] + [p_data['images'], q_data['images']]), 'gaussians': torch.stack([n['gaussians'] for n in n_datas] + [p_data['gaussians'], q_data['gaussians']]), 'extrinsics': torch.stack([n['extrinsics'] for n in n_datas] + [p_data['extrinsics'], q_data['extrinsics']]), 'intrinsics': torch.stack([n['intrinsics'] for n in n_datas] + [p_data['intrinsics'], q_data['intrinsics']])}, len(selected_neg_indices)

    def __len__(self): return len(self.queries)

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch: return None, None
    return default_collate([b[0] for b in batch]), [b[1] for b in batch]

class DatabaseQueryDataset(BaseDataset):
    def __init__(self, data_root_dir, database_path, query_path, info_path, transforms, nonTrivPosDistThres, gaussian_path=None, resize=None):
        super().__init__(data_root_dir, info_path, gaussian_path, resize)
        db, q = np.load(database_path), np.load(query_path)
        self.dataset = np.concatenate([db, q], axis=0)
        self.num_db, self.num_query = len(db), len(q)
        self.img_transforms, self.positives = transforms, None
        self.nonTrivPosDistThres = nonTrivPosDistThres
    def __getitem__(self, item): return self.load_data_with_matrices(int(self.dataset[item][0]))
    def __len__(self): return len(self.dataset)
    def getPositives(self):
        if self.positives is None:
            knn = NearestNeighbors(n_jobs=-1).fit(self.dataset[:self.num_db, 1:])
            self.positives = list(knn.radius_neighbors(self.dataset[self.num_db:, 1:], radius=self.nonTrivPosDistThres, return_distance=False))
        return self.positives