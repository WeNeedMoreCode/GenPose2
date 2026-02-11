import torch
from torch.autograd import Variable
from torch.autograd import Function
import torch.nn as nn
from typing import Tuple
import sys

# import pointnet2_cuda as pointnet2
# import pointnet2_torch as pointnet2


# class FurthestPointSampling(Function):
#     @staticmethod
#     def forward(ctx, xyz: torch.Tensor, npoint: int) -> torch.Tensor:
#         """
#         Uses iterative furthest point sampling to select a set of npoint features that have the largest
#         minimum distance
#         :param ctx:
#         :param xyz: (B, N, 3) where N > npoint
#         :param npoint: int, number of features in the sampled set
#         :return:
#              output: (B, npoint) tensor containing the set
#         """
#         assert xyz.is_contiguous()

#         B, N, _ = xyz.size()
#                 # import ipdb;ipdb.set_trace()
#         # output = torch.cuda.IntTensor(B, npoint)
#         # output = torch.npu.IntTensor(B, npoint)
#         output = torch.empty((B, npoint), dtype=torch.int32, device='npu:0')
#         # temp = torch.cuda.FloatTensor(B, N).fill_(1e10)
#         # temp = torch.npu.FloatTensor(B, N).fill_(1e10)
#         temp = torch.empty((B, N), dtype=torch.float32, device='npu:0')
#         temp.fill_(1e10)

#         print(f"[DEBUG-5] pointnet2_utils.py:  准备调用 furthest_point_sampling_wrapper, xyz shape: {xyz.shape}, npoint: {npoint}")
#         pointnet2.furthest_point_sampling_wrapper(B, N, npoint, xyz, temp, output)
#         print(f"[DEBUG-5] pointnet2_utils.py:  furthest_point_sampling_wrapper 返回")
#         return output

#     @staticmethod
#     def backward(xyz, a=None):
#         return None, None


# furthest_point_sample = FurthestPointSampling.apply

def furthest_point_sample(points: torch.Tensor, num_samples: int) -> torch.Tensor:
    # points: (B, N, 3) 输入点云
    # num_samples: M 采样点数量
    # 返回: (B, M) 采样点索引
    # 假返回：跳过FPS计算，测试后续流程
    B, N, _ = points.shape
    return torch.zeros(B, num_samples, dtype=torch.int32, device=points.device)

    B, N, _ = points.shape
    idxs = torch.zeros(B, num_samples, dtype=torch.long, device=points.device)
    dists = torch.ones(B, N, device=points.device) * 1e10  # 初始距离设为很大值
    
    # 选择第一个点（索引0）
    idxs[:, 0] = 0
    batch_indices = torch.arange(B, device=points.device)
    
    for i in range(1, num_samples):
        # 获取当前选中点的坐标
        current_points = points[batch_indices, idxs[:, i-1], :]
        # 计算所有点到当前选中点的欧氏距离平方
        new_dists = torch.sum((points - current_points.unsqueeze(1)) ** 2, dim=-1)
        # 更新距离矩阵：保留每个点到已选点集的最小距离
        dists = torch.min(dists, new_dists)
        # 选择距离最大的点作为下一个采样点
        max_dist, idx = torch.max(dists, dim=1)
        idxs[:, i] = idx
    idxs = idxs.to(torch.int32)
    return idxs


# class GatherOperation(Function):

#     @staticmethod
#     def forward(ctx, features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
#         """
#         :param ctx:
#         :param features: (B, C, N)
#         :param idx: (B, npoint) index tensor of the features to gather
#         :return:
#             output: (B, C, npoint)
#         """
#         assert features.is_contiguous()
#         assert idx.is_contiguous()

#         B, npoint = idx.size()
#         _, C, N = features.size()
#         # output = torch.cuda.FloatTensor(B, C, npoint)
#         output = torch.empty((B, C, npoint), dtype=torch.float32, device='npu:0')

#         print(f"[DEBUG-6] pointnet2_utils.py:  准备调用 gather_points_wrapper, features shape: {features.shape}, idx shape: {idx.shape}")
#         pointnet2.gather_points_wrapper(B, C, N, npoint, features, idx, output)
#         print(f"[DEBUG-6] pointnet2_utils.py:  gather_points_wrapper 返回")

#         ctx.for_backwards = (idx, C, N)
#         return output

    # @staticmethod
    # def backward(ctx, grad_out):
    #     idx, C, N = ctx.for_backwards
    #     B, npoint = idx.size()

    #     grad_features = Variable(torch.cuda.FloatTensor(B, C, N).zero_())
    #     grad_out_data = grad_out.data.contiguous()
    #     pointnet2.gather_points_grad_wrapper(B, C, N, npoint, grad_out_data, idx, grad_features.data)
    #     return grad_features, None


# gather_operation = GatherOperation.apply
def gather_operation(features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    # features: (B, C, N) 输入特征
    # idx: (B, M) 采样点索引
    # 输出: (B, C, M) 收集的特征
    B, C, N = features.size()
    M = idx.size(1)
    
    # 将索引扩展为 (B, C, M) 以匹配特征维度
    idx_expanded = idx.unsqueeze(1).expand(-1, C, -1).long()
    # 沿第2维度(点维度)收集特征
    return torch.gather(features, dim=2, index=idx_expanded)


# class ThreeNN(Function):

#     @staticmethod
#     def forward(ctx, unknown: torch.Tensor, known: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         # """
#         # Find the three nearest neighbors of unknown in known
#         # :param ctx:
#         # :param unknown: (B, N, 3)
#         # :param known: (B, M, 3)
#         # :return:
#         #     dist: (B, N, 3) l2 distance to the three nearest neighbors
#         #     idx: (B, N, 3) index of 3 nearest neighbors
#         # """
#         # assert unknown.is_contiguous()
#         # assert known.is_contiguous()
def three_nn(unknown: torch.Tensor, known: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Find the three nearest neighbors of unknown in known
    :param unknown: (B, N, 3) unknown points
    :param known: (B, M, 3) known points
    :return:
        dist: (B, N, 3) l2 distance to the three nearest neighbors
        idx: (B, N, 3) index of 3 nearest neighbors
    """
    assert unknown.is_contiguous(), "Input unknown must be contiguous"
    assert known.is_contiguous(), "Input known must be contiguous"

    B, N, _ = unknown.size()
    M = known.size(1)

    # Calculate squared Euclidean distance matrix (B, N, M)
    dist2 = torch.cdist(unknown, known) ** 2
    
    # Get three nearest neighbors (smallest distances)
    dist2, idx = torch.topk(dist2, k=3, dim=-1, largest=False)
    
    return torch.sqrt(dist2), idx.long()


# class ThreeInterpolate(Function):

#     @staticmethod
#     def forward(ctx, features: torch.Tensor, idx: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
#         """
#         Performs weight linear interpolation on 3 features
#         :param ctx:
#         :param features: (B, C, M) Features descriptors to be interpolated from
#         :param idx: (B, n, 3) three nearest neighbors of the target features in features
#         :param weight: (B, n, 3) weights
#         :return:
#             output: (B, C, N) tensor of the interpolated features
#         """
#         assert features.is_contiguous()
#         assert idx.is_contiguous()
#         assert weight.is_contiguous()

#         B, c, m = features.size()
#         n = idx.size(1)
#         ctx.three_interpolate_for_backward = (idx, weight, m)
#         output = torch.cuda.FloatTensor(B, c, n)

#         pointnet2.three_interpolate_wrapper(B, c, m, n, features, idx, weight, output)
#         return output

#     @staticmethod
#     def backward(ctx, grad_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         """
#         :param ctx:
#         :param grad_out: (B, C, N) tensor with gradients of outputs
#         :return:
#             grad_features: (B, C, M) tensor with gradients of features
#             None:
#             None:
#         """
#         idx, weight, m = ctx.three_interpolate_for_backward
#         B, c, n = grad_out.size()

#         grad_features = Variable(torch.cuda.FloatTensor(B, c, m).zero_())
#         grad_out_data = grad_out.data.contiguous()

#         pointnet2.three_interpolate_grad_wrapper(B, c, n, m, grad_out_data, idx, weight, grad_features.data)
#         return grad_features, None, None


# three_interpolate = ThreeInterpolate.apply

def three_interpolate(features: torch.Tensor, idx: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Performs weight linear interpolation on 3 features
    :param features: (B, C, M) Features descriptors to be interpolated from
    :param idx: (B, N, 3) three nearest neighbors indices
    :param weight: (B, N, 3) interpolation weights
    :return:
        output: (B, C, N) interpolated features
    """
    assert features.is_contiguous(), "Input features must be contiguous"
    assert idx.is_contiguous(), "Input idx must be contiguous"
    assert weight.is_contiguous(), "Input weight must be contiguous"

    B, C, M = features.shape
    N = idx.size(1)

    # Expand indices to (B, C, N, 3) and convert to long
    idx = idx.long().unsqueeze(1).expand(B, C, N, 3)
    
    # Gather features and apply weights
    gathered_features = torch.gather(features.unsqueeze(2), dim=3, index=idx)
    interpolated = torch.sum(gathered_features * weight.unsqueeze(1), dim=-1)
    
    return interpolated


# class GroupingOperation(Function):

#     @staticmethod
#     def forward(ctx, features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
#         """
#         :param ctx:
#         :param features: (B, C, N) tensor of features to group
#         :param idx: (B, npoint, nsample) tensor containing the indicies of features to group with
#         :return:
#             output: (B, C, npoint, nsample) tensor
#         """
#         assert features.is_contiguous()
#         assert idx.is_contiguous()

#         B, nfeatures, nsample = idx.size()
#         _, C, N = features.size()
#         output = torch.cuda.FloatTensor(B, C, nfeatures, nsample)

#         pointnet2.group_points_wrapper(B, C, N, nfeatures, nsample, features, idx, output)

#         ctx.for_backwards = (idx, N)
#         return output

#     @staticmethod
#     def backward(ctx, grad_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         :param ctx:
#         :param grad_out: (B, C, npoint, nsample) tensor of the gradients of the output from forward
#         :return:
#             grad_features: (B, C, N) gradient of the features
#         """
#         idx, N = ctx.for_backwards

#         B, C, npoint, nsample = grad_out.size()
#         grad_features = Variable(torch.cuda.FloatTensor(B, C, N).zero_())

#         grad_out_data = grad_out.data.contiguous()
#         pointnet2.group_points_grad_wrapper(B, C, N, npoint, nsample, grad_out_data, idx, grad_features.data)
#         return grad_features, None


# grouping_operation = GroupingOperation.apply

def grouping_operation(features, idx):
    B, C, N = features.shape
    _, npoint, nsample = idx.shape
    
    # 将索引张量重塑为 [B, npoint*nsample] 以匹配3D特征张量
    idx_reshaped = idx.reshape(B, npoint * nsample)  # (B, npoint*nsample)
    
    # 扩展索引张量以匹配特征通道数C
    idx_expanded = idx_reshaped.unsqueeze(1).expand(B, C, npoint * nsample).long()  # (B, C, npoint*nsample)
    
    # 沿点云维度(N)收集特征，输入和索引张量均为3D
    grouped_features = torch.gather(features, dim=2, index=idx_expanded)  # (B, C, npoint*nsample)
    
    # 将结果重塑为目标形状 [B, C, npoint, nsample]
    grouped_features = grouped_features.reshape(B, C, npoint, nsample)  # (B, C, npoint, nsample)
    
    return grouped_features


# class BallQuery(Function):

#     @staticmethod
#     def forward(ctx, radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
#         """
#         :param ctx:
#         :param radius: float, radius of the balls
#         :param nsample: int, maximum number of features in the balls
#         :param xyz: (B, N, 3) xyz coordinates of the features
#         :param new_xyz: (B, npoint, 3) centers of the ball query
#         :return:
#             idx: (B, npoint, nsample) tensor with the indicies of the features that form the query balls
#         """
#         assert new_xyz.is_contiguous()
#         assert xyz.is_contiguous()

#         B, N, _ = xyz.size()
#         npoint = new_xyz.size(1)
#         idx = torch.cuda.IntTensor(B, npoint, nsample).zero_()

#         pointnet2.ball_query_wrapper(B, N, npoint, radius, nsample, new_xyz, xyz, idx)
#         return idx

#     @staticmethod
#     def backward(ctx, a=None):
#         return None, None, None, None


# ball_query = BallQuery.apply

def ball_query(radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
    """
    纯PyTorch实现的ball_query函数，与CUDA版本功能完全一致
    :param radius: 球查询半径
    :param nsample: 每个查询点返回的最大样本数
    :param xyz: (B, N, 3) 原始点云坐标
    :param new_xyz: (B, M, 3) 查询点坐标
    :return: (B, M, nsample) 每个查询点的近邻索引
    """
    B, N, _ = xyz.size()
    M = new_xyz.size(1)
    idx = torch.zeros(B, M, nsample, dtype=torch.long, device=xyz.device)

    # 计算所有点对之间的欧氏距离平方 (B, M, N)
    dist_sq = torch.cdist(new_xyz, xyz, p=2) ** 2
    radius_sq = radius ** 2

    for b in range(B):
        for m in range(M):
            # 获取当前查询点的距离掩码
            mask = dist_sq[b, m] < radius_sq
            valid_indices = torch.nonzero(mask, as_tuple=False).squeeze(1)
            K = valid_indices.size(0)

            if K == 0:
                # 无有效点时保持初始0值
                continue
            elif K > nsample:
                # 超过nsample时取前nsample个(保持原始顺序)
                idx[b, m] = valid_indices[:nsample]
            else:
                # 不足nsample时用第一个有效点填充剩余位置
                idx[b, m, :K] = valid_indices
                idx[b, m, K:] = valid_indices[0]

    return idx

class QueryAndGroup(nn.Module):
    def __init__(self, radius: float, nsample: int, use_xyz: bool = True):
        """
        :param radius: float, radius of ball
        :param nsample: int, maximum number of features to gather in the ball
        :param use_xyz:
        """
        super().__init__()
        self.radius, self.nsample, self.use_xyz = radius, nsample, use_xyz

    def forward(self, xyz: torch.Tensor, new_xyz: torch.Tensor, features: torch.Tensor = None) -> Tuple[torch.Tensor]:
        """
        :param xyz: (B, N, 3) xyz coordinates of the features
        :param new_xyz: (B, npoint, 3) centroids
        :param features: (B, C, N) descriptors of the features
        :return:
            new_features: (B, 3 + C, npoint, nsample)
        """
        print(f"[DEBUG-7] pointnet2_utils.py:  QueryAndGroup.forward, xyz shape: {xyz.shape}, new_xyz shape: {new_xyz.shape}")
        idx = ball_query(self.radius, self.nsample, xyz, new_xyz)
        print(f"[DEBUG-7] pointnet2_utils.py:  ball_query 返回, idx shape: {idx.shape}")
        xyz_trans = xyz.transpose(1, 2).contiguous()
        grouped_xyz = grouping_operation(xyz_trans, idx)  # (B, 3, npoint, nsample)
        print(f"[DEBUG-7] pointnet2_utils.py:  grouping_operation(xyz_trans) 返回, shape: {grouped_xyz.shape}")
        grouped_xyz -= new_xyz.transpose(1, 2).unsqueeze(-1)

        if features is not None:
            grouped_features = grouping_operation(features, idx)
            if self.use_xyz:
                new_features = torch.cat([grouped_xyz, grouped_features], dim=1)  # (B, C + 3, npoint, nsample)
            else:
                new_features = grouped_features
        else:
            assert self.use_xyz, "Cannot have not features and not use xyz as a feature!"
            new_features = grouped_xyz

        return new_features


class GroupAll(nn.Module):
    def __init__(self, use_xyz: bool = True):
        super().__init__()
        self.use_xyz = use_xyz

    def forward(self, xyz: torch.Tensor, new_xyz: torch.Tensor, features: torch.Tensor = None):
        """
        :param xyz: (B, N, 3) xyz coordinates of the features
        :param new_xyz: ignored
        :param features: (B, C, N) descriptors of the features
        :return:
            new_features: (B, C + 3, 1, N)
        """
        grouped_xyz = xyz.transpose(1, 2).unsqueeze(2)
        if features is not None:
            grouped_features = features.unsqueeze(2)
            if self.use_xyz:
                new_features = torch.cat([grouped_xyz, grouped_features], dim=1)  # (B, 3 + C, 1, N)
            else:
                new_features = grouped_features
        else:
            new_features = grouped_xyz

        return new_features
