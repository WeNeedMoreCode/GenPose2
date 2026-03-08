import torch

def _furthest_point_sampling(xyz, npoints):
    """Implemetation of PyTorch Furthest Point Sampling (FPS) algorithm."""
    B, N, _ = xyz.shape
    device = xyz.device
    idx = torch.zeros((B, npoints), dtype=torch.int64, device=device)
    distance = torch.ones((B, N), device=device, dtype=torch.float32) * 1e10
    batch_indices = torch.arange(B, device=device)
    farthest = torch.zeros((B,), dtype=torch.int64, device=device)
    idx[:, 0] = farthest

    for i in range(1, npoints):
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        distance = torch.min(distance, dist)
        farthest = torch.argmax(distance, dim=-1)
        idx[:, i] = farthest
    return idx.to(torch.int32)

def _gather_points(features, idx):
    B, C, N = features.shape
    M = idx.shape[1]
    idx = idx.unsqueeze(1).expand(-1, C, -1)
    output = torch.gather(features, dim=-1, index=idx)
    return output

def _gather_points_grad(grad_out, idx, N):
    B, C, M = grad_out.shape
    grad_points = torch.zeros((B, C, N), device=grad_out.device, dtype=grad_out.dtype)
    idx_expanded = idx.unsqueeze(1).expand(-1, C, -1)
    grad_points.scatter_add_(dim=-1, index=idx_expanded, src=grad_out)
    return grad_points

def _ball_query(new_xyz, xyz, radius, nsample):
    B, N, _ = xyz.shape
    M = new_xyz.shape[1]
    radius2 = radius * radius

    dists = torch.cdist(new_xyz, xyz, p=2) ** 2
    mask = dists < radius2
    indices = torch.arange(N, device=xyz.device).unsqueeze(0).unsqueeze(1).expand(B, M, -1)
    idx = torch.zeros((B, M, nsample), dtype=torch.int32, device=xyz.device)

    for b in range(B):
        for m in range(M):
            valid_indices = indices[b, m][mask[b, m]]  

            if len(valid_indices) == 0:
                continue
            elif len(valid_indices) < nsample:
                first_idx = valid_indices[0]
                idx[b, m, :len(valid_indices)] = valid_indices
                idx[b, m, len(valid_indices):] = first_idx
            else:
                idx[b, m, :] = valid_indices[:nsample]

    return idx.to(torch.int32)

def _group_points(points, idx):
    B, C, N = points.shape
    M, _ = idx.shape[1], idx.shape[2]
    idx = idx.unsqueeze(1).expand(-1, C, -1, -1)
    points = points.unsqueeze(2).expand(-1, -1, M, -1)
    out = torch.gather(points, dim=-1, index=idx)
    return out

def _group_points_grad(grad_out, idx, N):
    B, C, M, nsample = grad_out.shape
    idx = idx.unsqueeze(1).expand(-1, C, -1, -1)
    grad_points = torch.zeros((B, C, N), device=grad_out.device, dtype=grad_out.dtype)
    grad_points.scatter_add_(dim=-1, index=idx, src=grad_out)
    return grad_points


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
    B, C, _ = grad_out.shape
    grad_out = grad_out.unsqueeze(-1)
    weight = weight.unsqueeze(1).expand(-1, C, -1, -1)
    grad_neighbor = grad_out * weight
    idx = idx.unsqueeze(1).expand(-1, C, -1, -1)
    grad_points = torch.zeros((B, C, M), device=grad_out.device, dtype=grad_out.dtype)
    grad_points.scatter_add_(dim=-1, index=idx, src=grad_neighbor)
    return grad_points

def furthest_point_sampling_wrapper(B, N, m, points_tensor, temp_tensor, idx_tensor):
    idx = _furthest_point_sampling(points_tensor, m)
    idx_tensor.copy_(idx)
    return 1

def gather_points_wrapper(B, C, N, npoints, points_tensor, idx_tensor, out_tensor):
    out = _gather_points(points_tensor, idx_tensor)
    out_tensor.copy_(out)
    return 1

def gather_points_grad_wrapper(B, C, N, npoints, grad_out_tensor, idx_tensor, grad_points_tensor):
    grad_points = _gather_points_grad(grad_out_tensor, idx_tensor, N)
    grad_points_tensor.copy_(grad_points)
    return 1

def ball_query_wrapper(B, n, m, radius, nsample, new_xyz_tensor, xyz_tensor, idx_tensor):
    idx = _ball_query(new_xyz_tensor, xyz_tensor, radius, nsample)
    idx_tensor.copy_(idx)
    return 1

ball_query_wrapper_fast = ball_query_wrapper



def three_nn_wrapper(B, n, m, unknown_tensor, known_tensor, dist2_tensor, idx_tensor):
    dist2, idx = _three_nn(unknown_tensor, known_tensor)
    dist2_tensor.copy_(dist2)
    idx_tensor.copy_(idx)

three_nn_wrapper_fast = three_nn_wrapper

def three_interpolate_wrapper(B, c, m, n, points_tensor, idx_tensor, weight_tensor, out_tensor):
    out = _three_interpolate(points_tensor, idx_tensor, weight_tensor)
    out_tensor.copy_(out)

three_interpolate_wrapper_fast = three_interpolate_wrapper

def three_interpolate_grad_wrapper(B, c, n, m, grad_out_tensor, idx_tensor, weight_tensor, grad_points_tensor):
    grad_points = _three_interpolate_grad(grad_out_tensor, idx_tensor, weight_tensor, m)
    grad_points_tensor.copy_(grad_points)

three_interpolate_grad_wrapper_fast = three_interpolate_grad_wrapper