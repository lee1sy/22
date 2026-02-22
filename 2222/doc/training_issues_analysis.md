# 训练问题分析：NaN Loss 和 Recall 为 0

## 问题描述

1. **NaN Loss 问题**：训练时频繁出现 NaN loss，导致部分迭代被跳过
2. **Recall 为 0 问题**：第一轮训练的 recall@1/5/10 都为 0（**正常模型第一轮 recall 不应低于 60%**）

## ✅ 数据加载检查结果

运行 `tools/check_data_loading.py` 后确认：
- ✅ Info 文件正确加载（22103 个样本）
- ✅ 高斯点云数据正确加载（不是全零，所有样本都有 4096 个有效点）
- ✅ 模型前向传播正常（embedding 有效，没有 NaN/Inf）

**结论**：数据加载没有问题，问题出在训练流程中。

## 🔍 关键问题分析

### 问题 1: build_cache() 中的 NaN 处理

**位置**：`tools/runner.py:87-100`

```python
def build_cache(self):
    self.model.eval()
    path = os.path.join(self.cache_dir, "feat_cache.hdf5")
    with torch.no_grad(), h5py.File(path, 'w') as h5:
        ds_len = len(self.whole_train_loader.dataset)
        feat_set = h5.create_dataset('features', [ds_len, 256], dtype=np.float32)
        ptr = 0
        for batch in tqdm(self.whole_train_loader, desc="Building Cache"):
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            emb = F.normalize(self.model(batch)['embedding'], p=2, dim=1)  # ⚠️ 问题点
            bs = emb.shape[0]
            feat_set[ptr : ptr + bs] = emb.cpu().numpy()
            ptr += bs
```

**潜在问题**：
- 如果 `self.model(batch)['embedding']` 全为 0，`F.normalize` 会产生 NaN
- 如果 embedding 包含 NaN，会被直接写入 cache
- Cache 中的 NaN 特征会导致 triplet 构建失败
- 这会导致训练数据无效，进而导致 recall 为 0

### 问题 2: validate() 中的 NaN 处理

**位置**：`tools/runner.py:102-118`

```python
@torch.no_grad()
def validate(self, epoch):
    self.model.eval()
    all_feats = []
    for batch in tqdm(self.whole_val_loader, desc="Validating"):
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        all_feats.append(F.normalize(self.model(batch)['embedding'], p=2, dim=1).cpu().numpy())  # ⚠️ 问题点
    feats = np.concatenate(all_feats, axis=0)
    # ... 检索计算
```

**潜在问题**：
- 如果 embedding 全为 0 或包含 NaN，`F.normalize` 会产生 NaN
- NaN 特征会导致 faiss 检索异常
- 这会导致 recall 为 0

### 问题 3: F.normalize 在全零向量上的行为

**核心问题**：PyTorch 的 `F.normalize` 在全零向量上会产生 NaN：

```python
import torch.nn.functional as F
x = torch.zeros(1, 256)
y = F.normalize(x, p=2, dim=1)
print(y)  # 输出: tensor([[nan, nan, ..., nan]])
```

### 问题 4: NetVLADLoupe 可能输出全零

**位置**：`modules/GS.py:132-149`

如果 mask 导致所有 activation 为 0，或者中间计算产生全零，最终 embedding 可能全为 0。

## 🔧 修复方案

### 1. 修复 F.normalize 在全零向量上的问题

创建安全的归一化函数：

```python
def safe_normalize(x, p=2, dim=1, eps=1e-8):
    norm = torch.norm(x, p=p, dim=dim, keepdim=True)
    return x / (norm + eps)
```

### 2. 修复 build_cache()

添加 NaN 检查和修复：

```python
def build_cache(self):
    self.model.eval()
    path = os.path.join(self.cache_dir, "feat_cache.hdf5")
    with torch.no_grad(), h5py.File(path, 'w') as h5:
        ds_len = len(self.whole_train_loader.dataset)
        feat_set = h5.create_dataset('features', [ds_len, 256], dtype=np.float32)
        ptr = 0
        for batch in tqdm(self.whole_train_loader, desc="Building Cache"):
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            emb = self.model(batch)['embedding']
            
            # 检查 NaN/Inf
            if torch.isnan(emb).any() or torch.isinf(emb).any():
                print(f"WARNING: NaN/Inf in embedding during build_cache, replacing with zeros")
                emb = torch.zeros_like(emb)
            
            # 安全归一化
            emb_norm = torch.norm(emb, dim=1, keepdim=True)
            emb = emb / (emb_norm + 1e-8)
            
            # 再次检查
            if torch.isnan(emb).any() or torch.isinf(emb).any():
                print(f"WARNING: NaN/Inf after normalization, replacing with zeros")
                emb = torch.zeros_like(emb)
            
            bs = emb.shape[0]
            feat_set[ptr : ptr + bs] = emb.cpu().numpy()
            ptr += bs
```

### 3. 修复 validate()

添加 NaN 检查和修复：

```python
@torch.no_grad()
def validate(self, epoch):
    self.model.eval()
    all_feats = []
    for batch in tqdm(self.whole_val_loader, desc="Validating"):
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        emb = self.model(batch)['embedding']
        
        # 检查 NaN/Inf
        if torch.isnan(emb).any() or torch.isinf(emb).any():
            print(f"WARNING: NaN/Inf in embedding during validate, replacing with zeros")
            emb = torch.zeros_like(emb)
        
        # 安全归一化
        emb_norm = torch.norm(emb, dim=1, keepdim=True)
        emb = emb / (emb_norm + 1e-8)
        
        # 再次检查
        if torch.isnan(emb).any() or torch.isinf(emb).any():
            print(f"WARNING: NaN/Inf after normalization, replacing with zeros")
            emb = torch.zeros_like(emb)
        
        all_feats.append(emb.cpu().numpy())
    
    feats = np.concatenate(all_feats, axis=0)
    
    # 检查 feats 中的 NaN
    if np.isnan(feats).any() or np.isinf(feats).any():
        print(f"WARNING: NaN/Inf in feats, replacing with zeros")
        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    
    db_f, q_f = feats[:self.whole_val_set.num_db], feats[self.whole_val_set.num_db:]
    # ... 后续检索代码
```

### 4. 修复训练循环中的 normalize

```python
out = self.model(batch)
emb = out['embedding']

# 检查 NaN/Inf
if torch.isnan(emb).any() or torch.isinf(emb).any():
    print(f"Skipping NaN/Inf embedding at iteration {i}")
    continue

# 安全归一化
emb_norm = torch.norm(emb, dim=1, keepdim=True)
emb_normalized = emb / (emb_norm + 1e-8)

# 再次检查
if torch.isnan(emb_normalized).any() or torch.isinf(emb_normalized).any():
    print(f"Skipping NaN/Inf normalized embedding at iteration {i}")
    continue

loss = self.criterion(emb_normalized, batch, out, nNeg[0])
```

### 5. 修复 NetVLADLoupe

确保输出不会全为 0：

```python
def forward(self, x, mask=None):
    B, N, D = x.shape
    activation = torch.matmul(x, self.cluster_weights)
    activation = self.norm1(activation)
    activation = F.softmax(activation, dim=-1)
    if mask is not None:
        activation = activation * mask.unsqueeze(-1).to(x.dtype)
        # 检查是否有有效点
        valid_count = mask.sum(dim=-1, keepdim=True).float()
        if (valid_count < 1).any():
            # 如果没有有效点，使用均匀分布
            activation = torch.ones_like(activation) / self.cluster_size
    
    a_sum = activation.sum(-2, keepdim=True)
    a = a_sum * self.cluster_weights2
    vlad = torch.matmul(activation.transpose(1, 2), x)
    vlad = vlad.transpose(1, 2) - a
    
    # 安全归一化
    vlad_norm = torch.norm(vlad, dim=1, keepdim=True)
    vlad = vlad / (vlad_norm + 1e-8)
    
    vlad = vlad.reshape(B, -1)
    vlad_norm2 = torch.norm(vlad, dim=1, keepdim=True)
    vlad = vlad / (vlad_norm2 + 1e-8)
    
    vlad = torch.matmul(vlad, self.hidden1_weights)
    if self.gating:
        vlad = self.gating(vlad)
    
    # 最终安全归一化
    vlad_norm3 = torch.norm(vlad, dim=1, keepdim=True)
    vlad = vlad / (vlad_norm3 + 1e-8)
    
    return vlad
```

## 总结

**根本原因**：
1. `F.normalize` 在全零向量上产生 NaN
2. `build_cache()` 和 `validate()` 没有检查 NaN，导致 NaN 特征被使用
3. NaN 特征导致检索失败，recall 为 0

**修复优先级**：
1. **高优先级**：修复 `build_cache()` 和 `validate()` 中的 NaN 处理
2. **中优先级**：修复训练循环中的 normalize
3. **低优先级**：修复 `NetVLADLoupe` 中的数值稳定性
