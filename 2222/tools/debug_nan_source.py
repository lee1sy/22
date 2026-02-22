"""
调试 NaN 来源的脚本
在训练时添加详细的检查，找出 NaN 产生的具体位置
"""
import torch
import torch.nn.functional as F

def check_tensor(name, tensor, check_nan=True, check_inf=True, check_zero=False):
    """检查 tensor 的状态"""
    if tensor is None:
        print(f"  {name}: None")
        return
    
    if check_nan and torch.isnan(tensor).any():
        nan_count = torch.isnan(tensor).sum().item()
        print(f"  ❌ {name}: 包含 {nan_count} 个 NaN")
        return True
    
    if check_inf and torch.isinf(tensor).any():
        inf_count = torch.isinf(tensor).sum().item()
        print(f"  ❌ {name}: 包含 {inf_count} 个 Inf")
        return True
    
    if check_zero:
        zero_count = (tensor.abs() < 1e-8).sum().item()
        print(f"  {name}: 包含 {zero_count} 个接近 0 的值")
    
    print(f"  ✅ {name}: shape={tensor.shape}, min={tensor.min().item():.6f}, max={tensor.max().item():.6f}, mean={tensor.mean().item():.6f}")
    return False

def debug_model_forward(model, batch):
    """调试模型前向传播，找出 NaN 来源"""
    print("=" * 60)
    print("调试模型前向传播")
    print("=" * 60)
    
    g, img, ext, intr = batch['gaussians'], batch['images'], batch['extrinsics'], batch['intrinsics']
    
    # 检查输入
    print("\n[1] 检查输入数据...")
    check_tensor("gaussians", g)
    check_tensor("images", img)
    check_tensor("extrinsics", ext)
    check_tensor("intrinsics", intr)
    
    # VisualEncoder
    print("\n[2] VisualEncoder...")
    B, V, C, H, W = img.shape
    v_feats = model.visual_enc(img.view(B * V, C, H, W))
    if check_tensor("v_feats", v_feats):
        return None
    
    # UncertaintyNet
    print("\n[3] UncertaintyNet...")
    offsets = model.uncert_net(g)
    if check_tensor("offsets", offsets):
        print("  ❌ offsets 包含 NaN/Inf，这会导致 grid 包含 NaN")
        return None
    
    # Mask
    print("\n[4] Mask...")
    mask = g.abs().sum(-1) > 1e-5
    valid_points = mask.sum().item()
    print(f"  有效点数: {valid_points} / {g.shape[1]}")
    if valid_points == 0:
        print("  ❌ 没有有效点！")
        return None
    
    # PhysicsGatedProjection
    print("\n[5] PhysicsGatedProjection...")
    v_feats = v_feats.view(B, V, C, H, W)
    scale = torch.clamp(g[:, :, 4:7], 0.01, 10.0)
    opacity = torch.clamp(g[:, :, 11:12], 0.0, 1.0)
    gate = model.proj_layer.gate_mlp(torch.cat([scale, opacity], dim=-1))
    if check_tensor("gate", gate):
        return None
    
    pts_homo = torch.cat([g[:, :, :3], torch.ones(B, g.shape[1], 1, device=g.device)], dim=-1)
    sum_f, sum_m = torch.zeros(B, g.shape[1], 32, device=g.device), torch.zeros(B, g.shape[1], 1, device=g.device)
    
    for v in range(V):
        print(f"\n  [5.{v}] 相机 {v}...")
        cam_pts = torch.bmm(ext[:, v], pts_homo.transpose(1, 2)).transpose(1, 2)
        if check_tensor(f"cam_pts_{v}", cam_pts):
            return None
        
        uv_h = torch.bmm(intr[:, v], cam_pts[:, :, :3].transpose(1, 2)).transpose(1, 2)
        if check_tensor(f"uv_h_{v}", uv_h):
            return None
        
        depth = uv_h[:, :, 2:3].clamp(min=1e-6)
        if check_tensor(f"depth_{v}", depth):
            return None
        
        uv = uv_h[:, :, :2] / depth
        if check_tensor(f"uv_{v}", uv):
            print("  ❌ uv 包含 NaN/Inf，可能是 depth 为 0 或很小")
            return None
        
        grid = torch.stack([2.0 * uv[:, :, 0] / (W * 14.0) - 1.0, 2.0 * uv[:, :, 1] / (H * 14.0) - 1.0], dim=-1)
        if check_tensor(f"grid_before_offsets_{v}", grid):
            return None
        
        grid = (grid + offsets).unsqueeze(2)
        if check_tensor(f"grid_after_offsets_{v}", grid):
            print("  ❌ grid 包含 NaN/Inf，可能是 offsets 包含 NaN/Inf")
            return None
        
        sampled = F.grid_sample(v_feats[:, v], grid, align_corners=False).squeeze(-1).transpose(1, 2)
        if check_tensor(f"sampled_{v}", sampled):
            print("  ❌ sampled 包含 NaN/Inf，可能是 grid_sample 的问题")
            return None
        
        v_32 = model.proj_layer.bottleneck(sampled)
        if check_tensor(f"v_32_{v}", v_32):
            return None
        
        valid_mask = (depth > 0.1) & (grid.abs() <= 1.0).all(dim=-1) & mask.unsqueeze(-1)
        sum_f += (v_32 * gate * valid_mask.float())
        sum_m += valid_mask.float()
    
    v_32_final = sum_f / (sum_m + 1e-6)
    if check_tensor("v_32_final", v_32_final):
        return None
    
    # SpconvBackbone
    print("\n[6] SpconvBackbone...")
    f_46 = torch.cat([g, v_32_final], dim=-1)
    if check_tensor("f_46", f_46):
        return None
    
    f_sp = model.spconv_enc(g[:, :, :3], f_46, mask)
    if check_tensor("f_sp", f_sp):
        return None
    
    # NetVLADLoupe
    print("\n[7] NetVLADLoupe...")
    embedding = model.vlad(f_sp, mask=mask)
    if check_tensor("embedding", embedding):
        return None
    
    # 检查归一化
    print("\n[8] 归一化检查...")
    emb_norm = torch.norm(embedding, dim=1, keepdim=True)
    print(f"  embedding norm: min={emb_norm.min().item():.6f}, max={emb_norm.max().item():.6f}, mean={emb_norm.mean().item():.6f}")
    
    if emb_norm.min() < 1e-8:
        print(f"  ⚠️  WARNING: 有 embedding 的 norm 接近 0，归一化会产生 NaN")
        print(f"  接近 0 的数量: {(emb_norm < 1e-8).sum().item()}")
    
    emb_normalized = F.normalize(embedding, p=2, dim=1)
    if check_tensor("emb_normalized", emb_normalized):
        print("  ❌ 归一化后产生 NaN，可能是 embedding 全为 0")
        return None
    
    print("\n" + "=" * 60)
    print("✅ 模型前向传播正常，没有发现 NaN/Inf")
    print("=" * 60)
    
    return {
        'embedding': embedding,
        'fused_feat': f_sp,
        'sampled_visual_feats': v_32_final,
        'valid_mask': mask,
        'offsets': offsets
    }
