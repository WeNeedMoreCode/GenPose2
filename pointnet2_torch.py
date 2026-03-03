import torch

# -------------------------- 工具函数（内部使用，不对外暴露） --------------------------
def _furthest_point_sampling(xyz, npoints):
    B, N, _ = xyz.shape
    device = xyz.device
    idx = torch.zeros((B, npoints), dtype=torch.int64, device=device)
    distance = torch.ones((B, N), device=device, dtype=torch.float32) * 1e10
    batch_indices = torch.arange(B, device=device)
    farthest = torch.zeros((B,), dtype=torch.int64, device=device)
    idx[:, 0] = farthest

    # 1. 必须和CUDA的block_size一致（关键！）
    block_size = 512  # 你的CUDA版本n_threads=512

    for i in range(1, npoints):
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        distance = torch.min(distance, dist)
        
        # 2. 模拟CUDA的线程分片：每个tid处理k=tid, tid+block_size...
        # 步骤1：创建分片索引（模拟每个线程的处理范围）
        tid_indices = []
        for tid in range(block_size):
            # 每个tid处理的点索引：tid, tid+block_size, tid+2*block_size...
            k_list = torch.arange(tid, N, block_size, device=device, dtype=torch.int64)
            tid_indices.append(k_list)
        
        # 步骤2：每个分片找最大值（模拟单个线程的besti）
        tid_max_vals = []
        tid_max_idxs = []
        for tid in range(block_size):
            k_list = tid_indices[tid]
            if len(k_list) == 0:
                tid_max_vals.append(torch.tensor(-1.0, device=device))
                tid_max_idxs.append(torch.tensor(0, device=device))
                continue
            # 取当前tid处理的点的距离
            tid_dist = distance[:, k_list]
            # 每个tid找自己分片内的最大值
            max_val, max_idx = torch.max(tid_dist, dim=-1)
            # 转换回全局索引
            global_idx = k_list[max_idx]
            tid_max_vals.append(max_val)
            tid_max_idxs.append(global_idx)
        
        # 步骤3：归约所有tid的结果（模拟__update，相等时保留tid更小的）
        tid_max_vals = torch.stack(tid_max_vals, dim=1)  # (B, block_size)
        tid_max_idxs = torch.stack(tid_max_idxs, dim=1)  # (B, block_size)
        # 找所有tid中最大值的索引（相等时保留第一个tid，即tid更小的）
        global_max_val, global_tid = torch.max(tid_max_vals, dim=1)
        # 按tid取最终的全局索引
        farthest = torch.gather(tid_max_idxs, 1, global_tid.unsqueeze(1)).squeeze(1)

        # 保留你的debug逻辑
        if i == 1:
            print(f"[FPS NPU DEBUG] Iteration 1 (first batch):")
            print(f"  Selected idx: {farthest[0].item()}")
            print(f"  Distance at selected idx: {distance[0, farthest[0]].item():.10f}")
            top5_dist, top5_idx = torch.topk(distance[0], k=5)
            print(f"  Top 5 distances:")
            for j in range(5):
                print(f"    idx={top5_idx[j].item()}, dist={top5_dist[j].item():.10f}")

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
    # [NPU DEBUG] Before calling FPS
    print(f"[FPS NPU Python] B={B}, N={N}, npoint={m}")
    print(f"[FPS NPU Python] xyz sample first 3 points: {points_tensor[0, :3].tolist()}")

    # [NPU DEBUG] 手动计算点536和点197到点0的距离（与CUDA版本对比）
    point0 = points_tensor[0, 0, :]
    point536 = points_tensor[0, 536, :]
    point197 = points_tensor[0, 197, :]
    dist_to_536 = torch.sum((point536 - point0) ** 2)
    dist_to_197 = torch.sum((point197 - point0) ** 2)
    print(f"[FPS NPU Python] Manual distance check:")
    print(f"  Point 0: {point0.tolist()}")
    print(f"  Point 536: {point536.tolist()}, dist to 0: {dist_to_536.item():.10f}")
    print(f"  Point 197: {point197.tolist()}, dist to 0: {dist_to_197.item():.10f}")

    # 调用纯PyTorch FPS核心逻辑
    idx = _furthest_point_sampling(points_tensor, m)

    # 把结果写入传入的idx_tensor（模拟原CUDA算子的in-place修改）
    idx_tensor.copy_(idx)

    # [NPU DEBUG] After calling FPS
    print(f"[FPS NPU Python] Result idx (first batch, first 10): {idx_tensor[0, :10].tolist()}")
    print(f"[FPS NPU Python] Result idx (first batch, last 10): {idx_tensor[0, -10:].tolist()}")

    return 1  # 原CUDA版本返回1，保持一致

def gather_points_wrapper(B, C, N, npoints, points_tensor, idx_tensor, out_tensor):
    """与原CUDA版本接口完全一致的Gather wrapper"""
    out = _gather_points(points_tensor, idx_tensor)
    out_tensor.copy_(out)
    return 1

def gather_points_grad_wrapper(B, C, N, npoints, grad_out_tensor, idx_tensor, grad_points_tensor):
    """与原CUDA版本接口完全一致的Gather反向wrapper"""
    grad_points = _gather_points_grad(grad_out_tensor, idx_tensor, N)
    grad_points_tensor.copy_(grad_points)
    return 1

def ball_query_wrapper(B, n, m, radius, nsample, new_xyz_tensor, xyz_tensor, idx_tensor):
    """与原CUDA版本接口完全一致的球形邻域查询wrapper"""
    idx = _ball_query(new_xyz_tensor, xyz_tensor, radius, nsample)
    idx_tensor.copy_(idx)
    return 1

# 兼容原ball_query_wrapper_fast（避免调用别名遗漏）
ball_query_wrapper_fast = ball_query_wrapper

def group_points_wrapper(B, c, n, npoints, nsample, points_tensor, idx_tensor, out_tensor):
    """与原CUDA版本接口完全一致的点分组wrapper"""
    out = _group_points(points_tensor, idx_tensor)
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
    dist2, idx = _three_nn(unknown_tensor, known_tensor)
    dist2_tensor.copy_(dist2)
    idx_tensor.copy_(idx)

# 兼容原three_nn_wrapper_fast
three_nn_wrapper_fast = three_nn_wrapper

def three_interpolate_wrapper(B, c, m, n, points_tensor, idx_tensor, weight_tensor, out_tensor):
    """与原CUDA版本接口完全一致的三邻域插值wrapper"""
    out = _three_interpolate(points_tensor, idx_tensor, weight_tensor)
    out_tensor.copy_(out)

# 兼容原three_interpolate_wrapper_fast
three_interpolate_wrapper_fast = three_interpolate_wrapper

def three_interpolate_grad_wrapper(B, c, n, m, grad_out_tensor, idx_tensor, weight_tensor, grad_points_tensor):
    """与原CUDA版本接口完全一致的三邻域插值反向wrapper"""
    grad_points = _three_interpolate_grad(grad_out_tensor, idx_tensor, weight_tensor, m)
    grad_points_tensor.copy_(grad_points)

# 兼容原three_interpolate_grad_wrapper_fast
three_interpolate_grad_wrapper_fast = three_interpolate_grad_wrapper