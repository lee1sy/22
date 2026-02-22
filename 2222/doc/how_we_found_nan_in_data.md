# 如何发现高斯点云数据包含 NaN

## 发现过程

### 1. 最初的问题
训练时频繁出现 NaN loss：
```
Epoch 1: 0%| | 1/8118 [00:05<13:11:59, 5.85s/it, loss=0.7008]
Skipping NaN loss at iteration 1
```

### 2. 添加检查代码

我们在多个位置添加了 NaN 检查：

#### 2.1 数据加载阶段 (`dataset/NuScenesDataset.py`)

在 `load_lidar_data()` 方法中，加载 `.npy` 文件后立即检查：

```python
def load_lidar_data(self, index):
    data = np.load(path)  # 从磁盘加载 .npy 文件
    
    # 检查 NaN/Inf 来源
    if np.isnan(data).any() or np.isinf(data).any():
        # 打印详细的追溯信息
        print(f"🔍 NaN/Inf 来源追溯 - 数据加载阶段")
        print(f"文件路径: {path}")
        print(f"NaN 数量: {nan_count}")
        # ...
```

**关键点**：在数据加载时（从 `.npy` 文件读取后），立即检查数据是否包含 NaN。

#### 2.2 模型入口检查 (`modules/GS.py`)

在 `GaussianFusionNet.forward()` 入口处检查：

```python
def forward(self, batch):
    g = batch['gaussians']  # 从数据加载器获取
    
    # 检查 NaN/Inf 来源（不修复，只追溯）
    if torch.isnan(g).any() or torch.isinf(g).any():
        print(f"🔍 NaN/Inf 来源追溯 - 模型入口")
        # ...
```

**关键点**：在模型接收数据时检查，确认 NaN 来自数据加载器。

### 3. 训练时的输出

运行训练时，我们看到了这样的输出：

```
================================================================================
🔍 NaN/Inf 来源追溯 - 数据加载阶段
================================================================================
文件路径: /home/james/LSY/11/nuscenes/n008-2018-08-01-15-52-19-0400__LIDAR_TOP__1533153441196688.pcd.npy
文件名: n008-2018-08-01-15-52-19-0400__LIDAR_TOP__1533153441196688.pcd.npy
样本索引: 219
数据形状: (4096, 14)
NaN 数量: 4 || Inf 数量: 0

NaN 位置分析:
  前10个NaN位置: [(1975, 7), (1975, 8), (1975, 9), (1975, 10)]
  包含NaN的列索引: [7, 8, 9, 10]
```

**这证明了**：
- NaN 在数据加载阶段就已经存在
- NaN 来自 `.npy` 文件本身
- 不是模型计算产生的

### 4. 批量检查确认

运行 `tools/check_nan_files.py` 批量检查所有文件：

```bash
python tools/check_nan_files.py
```

结果：
- 176 个文件包含 NaN（0.80%）
- 所有 NaN 都在列 7-10（四元数）
- 每个文件只有 4 个 NaN（正好是四元数的 4 个维度）

**这进一步确认**：NaN 是在数据生成阶段（`nuscenes-process.py`）产生的，而不是训练时产生的。

## 检查代码的位置

### 数据加载检查
- **文件**：`dataset/NuScenesDataset.py`
- **方法**：`load_lidar_data()`
- **检查时机**：从 `.npy` 文件加载后，转换为 tensor 前
- **检查内容**：`np.isnan(data).any()` 和 `np.isinf(data).any()`

### 模型入口检查
- **文件**：`modules/GS.py`
- **方法**：`GaussianFusionNet.forward()`
- **检查时机**：模型接收 batch 数据时
- **检查内容**：`torch.isnan(g).any()` 和 `torch.isinf(g).any()`

## 如何验证

如果你想验证数据是否包含 NaN，可以：

1. **直接检查 .npy 文件**：
```python
import numpy as np
data = np.load('path/to/file.npy')
print(f"包含 NaN: {np.isnan(data).any()}")
print(f"NaN 数量: {np.isnan(data).sum()}")
```

2. **运行检查脚本**：
```bash
python tools/check_nan_files.py
```

3. **运行训练**（会打印追溯信息）：
```bash
python bev_nclt.py
```

## 总结

我们发现 NaN 的流程：
1. ✅ 训练时出现 NaN loss
2. ✅ 添加数据加载阶段的检查
3. ✅ 训练时看到"数据加载阶段"的追溯输出
4. ✅ 确认 NaN 来自 `.npy` 文件本身
5. ✅ 批量检查确认 176 个文件包含 NaN
6. ✅ 定位到 NaN 在列 7-10（四元数），是在数据生成时产生的
