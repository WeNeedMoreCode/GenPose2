import torch

# -------------------------- 工具函数（内部使用，不对外暴露） --------------------------
def _furthest_point_sampling(xyz, npoints):
    """纯PyTorch实现FPS核心逻辑"""
    B, N, _ = xyz.shape
    device = xyz.device
    idx = torch.zeros((B, npoints), dtype=torch.int64, device=device)
    distance = torch.ones((B, N), device=device, dtype=torch.float32) * 1e10
    batch_indices = torch.arange(B, device=device)
    farthest = torch.zeros((B,), dtype=torch.int64, device=device)
    idx[:, 0] = farthest

    for i in range(1, npoints):
        print(f'npoints{npoints}')
        print(f'iiiiiiiiiiiiiiiiiiii{i}')
        print('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
        print(f'[DEBUG] xyz.device={xyz.device}, xyz.shape={xyz.shape}')
        print(f'[DEBUG] batch_indices.device={batch_indices.device}, batch_indices.shape={batch_indices.shape}')
        print(f'[DEBUG] farthest.device={farthest.device}, farthest.shape={farthest.shape}, farthest[:5]={farthest[:5]}')
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        print('bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb')
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        distance = torch.min(distance, dist)
        
        farthest = torch.argmax(distance, dim=-1)
        print('cccccccccccccccccccccccccccccccccccccccc')
        idx[:, i] = farthest
    return idx.to(torch.int32)

def _gather_points(features, idx):
    """纯PyTorch实现Gather核心逻辑"""
    B, C, N = features.shape
    M = idx.shape[1]
    idx = idx.unsqueeze(1).expand(-1, C, -1)
    output = torch.gather(features, dim=-1, index=idx)
    return output

def _gather_points_grad(grad_out, idx, N):
    """纯PyTorch实现Gather反向梯度"""
    B, C, M = grad_out.shape
    # 创建梯度容器 (B, C, N)
    grad_points = torch.zeros((B, C, N), device=grad_out.device, dtype=grad_out.dtype)
    # 将索引扩展到所有通道 (B, M) -> (B, C, M)
    idx_expanded = idx.unsqueeze(1).expand(-1, C, -1)
    # 使用 scatter_add_ 累加梯度（关键：同一个点可能被多次采样，需要累加）
    grad_points.scatter_add_(dim=-1, index=idx_expanded, src=grad_out)
    return grad_points

def _ball_query(new_xyz, xyz, radius, nsample):
    """纯PyTorch实现球形邻域查询核心逻辑"""
    B, N, _ = xyz.shape
    M = new_xyz.shape[1]
    dists = torch.cdist(new_xyz, xyz, p=2)
    radius2 = radius * radius
    mask = dists < radius2
    idx = torch.arange(N, device=xyz.device).unsqueeze(0).unsqueeze(0).repeat(B, M, 1)
    idx = torch.where(mask, idx, torch.zeros_like(idx))
    dists_sorted, idx_sorted = torch.sort(dists, dim=-1)
    idx = torch.gather(idx, dim=-1, index=idx_sorted)
    idx = idx[:, :, :nsample]
    mask_empty = (dists_sorted[:, :, :nsample] >= radius2).all(dim=-1, keepdim=True)
    idx = torch.where(mask_empty, idx[:, :, 0:1].repeat(1, 1, nsample), idx)
    return idx.to(torch.int32)

def _group_points(points, idx):
    """纯PyTorch实现点特征分组核心逻辑"""
    B, C, N = points.shape
    M, nsample = idx.shape[1], idx.shape[2]
    idx = idx.unsqueeze(1).expand(-1, C, -1, -1)
    points = points.unsqueeze(2).expand(-1, -1, M, -1)
    out = torch.gather(points, dim=-1, index=idx)
    return out

def _group_points_grad(grad_out, idx, N):
    """纯PyTorch实现group_points反向梯度"""
    B, C, M, nsample = grad_out.shape
    idx = idx.unsqueeze(1).expand(-1, C, -1, -1)
    grad_points = torch.zeros((B, C, N), device=grad_out.device, dtype=grad_out.dtype)
    grad_points.scatter_add_(dim=-1, index=idx, src=grad_out)
    return grad_points

def _three_nn(unknown, known):
    """纯PyTorch实现三邻域查询核心逻辑"""
    dist = torch.cdist(unknown, known, p=2)
    dist2, idx = torch.topk(dist, k=3, dim=-1, largest=False)
    dist2 = dist2 ** 2
    return dist2, idx.to(torch.int32)

def _three_interpolate(points, idx, weight):
    """纯PyTorch实现三邻域插值核心逻辑"""
    B, C, M = points.shape
    N = idx.shape[1]
    idx = idx.unsqueeze(1).expand(-1, C, -1, -1)
    points = points.unsqueeze(2).expand(-1, -1, N, -1)
    neighbor_feat = torch.gather(points, dim=-1, index=idx)
    weight = weight.unsqueeze(1).expand(-1, C, -1, -1)
    out = torch.sum(neighbor_feat * weight, dim=-1)
    return out

def _three_interpolate_grad(grad_out, idx, weight, M):
    """纯PyTorch实现三邻域插值反向梯度"""
    B, C, N = grad_out.shape
    grad_out = grad_out.unsqueeze(-1)
    weight = weight.unsqueeze(1).expand(-1, C, -1, -1)
    grad_neighbor = grad_out * weight
    idx = idx.unsqueeze(1).expand(-1, C, -1, -1)
    grad_points = torch.zeros((B, C, M), device=grad_out.device, dtype=grad_out.dtype)
    grad_points.scatter_add_(dim=-1, index=idx, src=grad_neighbor)
    return grad_points

# -------------------------- 对外暴露：与原pointnet2_cuda完全一致的wrapper接口 --------------------------
def furthest_point_sampling_wrapper(B, N, m, points_tensor, temp_tensor, idx_tensor):
    """
    与原CUDA版本接口完全一致的FPS wrapper
    参数：B(批次), N(总点数), m(采样点数), points_tensor(点云BxNx3), temp_tensor(临时张量), idx_tensor(输出索引)
    作用：直接修改idx_tensor的值（与原CUDA算子行为一致）
    """
    # 调用纯PyTorch FPS核心逻辑
    idx = _furthest_point_sampling(points_tensor, m)
    # 把结果写入传入的idx_tensor（模拟原CUDA算子的in-place修改）
    idx_tensor.copy_(idx)
    return 1  # 原CUDA版本返回1，保持一致

def gather_points_wrapper(B, C, N, npoints, points_tensor, idx_tensor, out_tensor):
    """与原CUDA版本接口完全一致的Gather wrapper"""
    print(f"[DEBUG-8] pointnet2_torch.py gather_points_wrapper, points shape: {points_tensor.shape}, idx shape: {idx_tensor.shape}")
    out = _gather_points(points_tensor, idx_tensor)
    print(f"[DEBUG-8] pointnet2_torch.py gather_points_wrapper 返回")
    out_tensor.copy_(out)
    return 1

def gather_points_grad_wrapper(B, C, N, npoints, grad_out_tensor, idx_tensor, grad_points_tensor):
    """与原CUDA版本接口完全一致的Gather反向wrapper"""
    grad_points = _gather_points_grad(grad_out_tensor, idx_tensor, N)
    grad_points_tensor.copy_(grad_points)
    return 1

def ball_query_wrapper(B, n, m, radius, nsample, new_xyz_tensor, xyz_tensor, idx_tensor):
    """与原CUDA版本接口完全一致的球形邻域查询wrapper"""
    print(f"[DEBUG-9] pointnet2_torch.py ball_query_wrapper, new_xyz shape: {new_xyz_tensor.shape}, xyz shape: {xyz_tensor.shape}, radius: {radius}, nsample: {nsample}")
    idx = _ball_query(new_xyz_tensor, xyz_tensor, radius, nsample)
    print(f"[DEBUG-9] pointnet2_torch.py ball_query_wrapper 返回")
    idx_tensor.copy_(idx)
    return 1

# 兼容原ball_query_wrapper_fast（避免调用别名遗漏）
ball_query_wrapper_fast = ball_query_wrapper

def group_points_wrapper(B, c, n, npoints, nsample, points_tensor, idx_tensor, out_tensor):
    """与原CUDA版本接口完全一致的点分组wrapper"""
    print(f"[DEBUG-10] pointnet2_torch.py group_points_wrapper, points shape: {points_tensor.shape}, idx shape: {idx_tensor.shape}")
    out = _group_points(points_tensor, idx_tensor)
    print(f"[DEBUG-10] pointnet2_torch.py group_points_wrapper 返回")
    out_tensor.copy_(out)
    return 1

# 兼容原group_points_wrapper_fast
group_points_wrapper_fast = group_points_wrapper

def group_points_grad_wrapper(B, c, n, npoints, nsample, grad_out_tensor, idx_tensor, grad_points_tensor):
    """与原CUDA版本接口完全一致的点分组反向wrapper"""
    grad_points = _group_points_grad(grad_out_tensor, idx_tensor, n)
    grad_points_tensor.copy_(grad_points)
    return 1

# 兼容原group_points_grad_wrapper_fast
group_points_grad_wrapper_fast = group_points_grad_wrapper

def three_nn_wrapper(B, n, m, unknown_tensor, known_tensor, dist2_tensor, idx_tensor):
    """与原CUDA版本接口完全一致的三邻域查询wrapper"""
    print(f"[DEBUG-11] pointnet2_torch.py three_nn_wrapper, unknown shape: {unknown_tensor.shape}, known shape: {known_tensor.shape}")
    dist2, idx = _three_nn(unknown_tensor, known_tensor)
    print(f"[DEBUG-11] pointnet2_torch.py three_nn_wrapper 返回")
    dist2_tensor.copy_(dist2)
    idx_tensor.copy_(idx)

# 兼容原three_nn_wrapper_fast
three_nn_wrapper_fast = three_nn_wrapper

def three_interpolate_wrapper(B, c, m, n, points_tensor, idx_tensor, weight_tensor, out_tensor):
    """与原CUDA版本接口完全一致的三邻域插值wrapper"""
    print(f"[DEBUG-12] pointnet2_torch.py three_interpolate_wrapper, points shape: {points_tensor.shape}, idx shape: {idx_tensor.shape}")
    out = _three_interpolate(points_tensor, idx_tensor, weight_tensor)
    print(f"[DEBUG-12] pointnet2_torch.py three_interpolate_wrapper 返回")
    out_tensor.copy_(out)

# 兼容原three_interpolate_wrapper_fast
three_interpolate_wrapper_fast = three_interpolate_wrapper

def three_interpolate_grad_wrapper(B, c, n, m, grad_out_tensor, idx_tensor, weight_tensor, grad_points_tensor):
    """与原CUDA版本接口完全一致的三邻域插值反向wrapper"""
    grad_points = _three_interpolate_grad(grad_out_tensor, idx_tensor, weight_tensor, m)
    grad_points_tensor.copy_(grad_points)

# 兼容原three_interpolate_grad_wrapper_fast
three_interpolate_grad_wrapper_fast = three_interpolate_grad_wrapper