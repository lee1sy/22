# 数据格式检查：14维高斯点云数据

## nuscenes-process.py 生成的数据格式

```python
res = np.zeros((TARGET_POINTS, 14), dtype=np.float32)
res[:, 0:3] = pts_np           # xyz 坐标
res[:, 3] = intensity_np       # intensity
res[:, 4:7] = s.cpu().numpy()  # scales (3维)
res[:, 7:11] = r.cpu().numpy() # rotations (quaternion, 4维)
res[:, 11] = o.cpu().numpy()   # opacities
res[:, 12] = l.cpu().numpy()   # linearity
res[:, 13] = p.cpu().numpy()   # planarity
```

**数据顺序**：
- 维度 0-2: xyz 坐标
- 维度 3: intensity
- 维度 4-6: scales (3维)
- 维度 7-10: rotations (quaternion, 4维) ⚠️ **NaN 出现在这里**
- 维度 11: opacities
- 维度 12: linearity
- 维度 13: planarity

## 模型使用情况

### modules/GS.py

1. **PhysicsGatedProjection**:
   ```python
   scale = torch.clamp(g14d[:, :, 4:7], 0.01, 10.0)      # ✓ 对应 res[:, 4:7]
   opacity = torch.clamp(g14d[:, :, 11:12], 0.0, 1.0)     # ✓ 对应 res[:, 11]
   ```

2. **GaussianFusionNet.forward**:
   ```python
   g[:, :, :3]  # xyz ✓ 对应 res[:, 0:3]
   ```

3. **SpconvBackbone**:
   ```python
   g[:, :, :3]  # xyz ✓
   ```

### tools/runner.py

1. **_compute_gcl**:
   ```python
   geo_gt_input = torch.cat([gaussians[..., :3], gaussians[..., 4:7]], dim=-1)
   # 使用 xyz (0:3) 和 scales (4:7) ✓
   ```

2. **_compute_pml**:
   ```python
   weight = (gaussians[..., 11] * torch.norm(gaussians[..., 4:7], dim=-1)).detach()
   # 使用 opacity (11) 和 scales (4:7) ✓
   ```

## 格式匹配检查

✅ **匹配的维度**：
- 0-2: xyz ✓
- 4-6: scales ✓
- 11: opacity ✓

⚠️ **未使用的维度**：
- 3: intensity - 模型中没有使用
- 7-10: rotations (quaternion) - 模型中没有使用，但包含 NaN
- 12: linearity - 模型中没有使用
- 13: planarity - 模型中没有使用

## NaN 来源分析

NaN 出现在维度 7-10（四元数），这些维度在模型中**没有被使用**，但会导致：
1. 数据加载时检测到 NaN
2. 如果后续代码使用这些维度，会产生问题

**可能的原因**：
1. `torch.linalg.eigh(cov)` 在某些情况下可能产生 NaN（奇异矩阵、数值不稳定）
2. `rotation_matrix_to_quaternion_torch` 转换可能产生 NaN（无效的旋转矩阵）

## 建议

虽然这些维度没有被使用，但为了数据完整性，应该在数据生成时修复 NaN。
