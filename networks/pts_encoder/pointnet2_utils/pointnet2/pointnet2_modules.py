import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from . import pointnet2_utils
from . import pytorch_utils as pt_utils
from typing import List

# Setup logger for NPU version
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class _PointnetSAModuleBase(nn.Module):

    def __init__(self):
        super().__init__()
        self.npoint = None
        self.groupers = None
        self.mlps = None
        self.pool_method = 'max_pool'

    def forward(self, xyz: torch.Tensor, features: torch.Tensor = None, new_xyz=None, return_idx=False) \
        -> "(torch.Tensor, torch.Tensor) | (torch.Tensor, torch.Tensor, torch.Tensor)":
        """
        :param xyz: (B, N, 3) tensor of the xyz coordinates of the features
        :param features: (B, N, C) tensor of the descriptors of the the features
        :param new_xyz:
        :return:
            new_xyz: (B, npoint, 3) tensor of the new features' xyz
            new_features: (B, npoint, \sum_k(mlps[k][-1])) tensor of the new_features descriptors
            new_idx: (B, npoint) tensor of indices
        """
        new_features_list = []

        xyz_flipped = xyz.transpose(1, 2).contiguous()
        idx = None
        if new_xyz is None:
            if self.npoint is not None:
                # ===== 监测1: furthest_point_sample =====
                logger.warning(f"[NPU OP MONITOR] Before furthest_point_sample: xyz shape={xyz.shape}, dtype={xyz.dtype}, min={xyz.min():.6f}, max={xyz.max():.6f}, mean={xyz.mean():.6f}")
                idx = pointnet2_utils.furthest_point_sample(xyz, self.npoint)
                logger.warning(f"[NPU OP MONITOR] After furthest_point_sample: idx shape={idx.shape}, dtype={idx.dtype}, min={idx.min()}, max={idx.max()}")

                # ===== 监测2: gather_operation =====
                logger.warning(f"[NPU OP MONITOR] Before gather_operation: xyz_flipped shape={xyz_flipped.shape}, idx shape={idx.shape}")
                new_xyz = pointnet2_utils.gather_operation(
                    xyz_flipped,
                    idx
                ).transpose(1, 2).contiguous()
                logger.warning(f"[NPU OP MONITOR] After gather_operation: new_xyz shape={new_xyz.shape}, dtype={new_xyz.dtype}, min={new_xyz.min():.6f}, max={new_xyz.max():.6f}, mean={new_xyz.mean():.6f}")
            else:
                new_xyz = None

        for i in range(len(self.groupers)):
            # ===== 监测3+4: grouper (ball_query + grouping_operation) =====
            logger.warning(f"[NPU OP MONITOR] Before grouper[{i}]: xyz shape={xyz.shape}, new_xyz shape={new_xyz.shape if new_xyz is not None else None}, features shape={features.shape if features is not None else None}")
            new_features = self.groupers[i](xyz, new_xyz, features)  # (B, C, npoint, nsample)
            logger.warning(f"[NPU OP MONITOR] After grouper[{i}]: new_features shape={new_features.shape}, dtype={new_features.dtype}, min={new_features.min():.6f}, max={new_features.max():.6f}, mean={new_features.mean():.6f}, is_contiguous={new_features.is_contiguous()}")

            # [DEBUG NPU] grouper 输出检查
            if not hasattr(self, '_debug_sa_base_count'):
                self._debug_sa_base_count = 0
            if self._debug_sa_base_count < 1:
                print(f"[NPU DEBUG SA_Base] grouper {i} output: shape={new_features.shape}, min={new_features.min():.6f}, max={new_features.max():.6f}, mean={new_features.mean():.6f}")

            new_features = self.mlps[i](new_features)  # (B, mlp[-1], npoint, nsample)

            # [DEBUG NPU] MLP 输出检查
            if not hasattr(self, '_debug_sa_base_count'):
                self._debug_sa_base_count = 0
            if self._debug_sa_base_count < 1:
                print(f"[NPU DEBUG SA_Base] MLP {i} output: shape={new_features.shape}, is_contiguous={new_features.is_contiguous()}, min={new_features.min():.6f}, max={new_features.max():.6f}, mean={new_features.mean():.6f}")

            if self.pool_method == 'max_pool':
                # [DEBUG NPU] max_pool2d 前检查
                if hasattr(self, '_debug_sa_base_count') and self._debug_sa_base_count < 1:
                    print(f"[NPU DEBUG SA_Base] Before max_pool2d: shape={new_features.shape}, is_contiguous={new_features.is_contiguous()}, min={new_features.min():.6f}, max={new_features.max():.6f}")

                new_features = F.max_pool2d(
                    new_features.to('cpu'), kernel_size=[1, new_features.size(3)]
                ).to(new_features.device)  # (B, mlp[-1], npoint, 1)

                # [DEBUG NPU] max_pool2d 后检查
                if hasattr(self, '_debug_sa_base_count') and self._debug_sa_base_count < 1:
                    print(f"[NPU DEBUG SA_Base] After max_pool2d: shape={new_features.shape}, is_contiguous={new_features.is_contiguous()}, min={new_features.min():.6f}, max={new_features.max():.6f}")
            elif self.pool_method == 'avg_pool':
                new_features = F.avg_pool2d(
                    new_features, kernel_size=[1, new_features.size(3)]
                )  # (B, mlp[-1], npoint, 1)
            else:
                raise NotImplementedError

            # [DEBUG NPU] squeeze 前后检查
            if hasattr(self, '_debug_sa_base_count') and self._debug_sa_base_count < 1:
                print(f"[NPU DEBUG SA_Base] Before squeeze: shape={new_features.shape}, min={new_features.min():.6f}, max={new_features.max():.6f}")

            new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint)

            if hasattr(self, '_debug_sa_base_count') and self._debug_sa_base_count < 1:
                print(f"[NPU DEBUG SA_Base] After squeeze: shape={new_features.shape}, min={new_features.min():.6f}, max={new_features.max():.6f}")

            new_features_list.append(new_features)

        # [DEBUG NPU] 拼接前检查
        if hasattr(self, '_debug_sa_base_count') and self._debug_sa_base_count < 1:
            print(f"[NPU DEBUG SA_Base] Before concat:")
            for j, nf in enumerate(new_features_list):
                print(f"  new_features_list[{j}]: shape={nf.shape}, min={nf.min():.6f}, max={nf.max():.6f}, mean={nf.mean():.6f}")
            concat_result = torch.cat(new_features_list, dim=1)
            print(f"  concat_result: shape={concat_result.shape}, min={concat_result.min():.6f}, max={concat_result.max():.6f}, mean={concat_result.mean():.6f}")
            self._debug_sa_base_count += 1

        if return_idx:
            return new_xyz, torch.cat(new_features_list, dim=1), idx
        return new_xyz, torch.cat(new_features_list, dim=1)


class PointnetSAModuleMSG(_PointnetSAModuleBase):
    """Pointnet set abstraction layer with multiscale grouping"""

    def __init__(self, *, npoint: int, radii: List[float], nsamples: List[int], mlps: List[List[int]], bn: bool = True,
                 use_xyz: bool = True, pool_method='max_pool', instance_norm=False):
        """
        :param npoint: int
        :param radii: list of float, list of radii to group with
        :param nsamples: list of int, number of samples in each ball query
        :param mlps: list of list of int, spec of the pointnet before the global pooling for each scale
        :param bn: whether to use batchnorm
        :param use_xyz:
        :param pool_method: max_pool / avg_pool
        :param instance_norm: whether to use instance_norm
        """
        super().__init__()

        assert len(radii) == len(nsamples) == len(mlps)

        self.npoint = npoint
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            self.groupers.append(
                pointnet2_utils.QueryAndGroup(radius, nsample, use_xyz=use_xyz)
                if npoint is not None else pointnet2_utils.GroupAll(use_xyz)
            )
            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3

            self.mlps.append(pt_utils.SharedMLP(mlp_spec, bn=bn, instance_norm=instance_norm))
        self.pool_method = pool_method


class PointnetSAModule(PointnetSAModuleMSG):
    """Pointnet set abstraction layer"""

    def __init__(self, *, mlp: List[int], npoint: int = None, radius: float = None, nsample: int = None,
                 bn: bool = True, use_xyz: bool = True, pool_method='max_pool', instance_norm=False):
        """
        :param mlp: list of int, spec of the pointnet before the global max_pool
        :param npoint: int, number of features
        :param radius: float, radius of ball
        :param nsample: int, number of samples in the ball query
        :param bn: whether to use batchnorm
        :param use_xyz:
        :param pool_method: max_pool / avg_pool
        :param instance_norm: whether to use instance_norm
        """
        super().__init__(
            mlps=[mlp], npoint=npoint, radii=[radius], nsamples=[nsample], bn=bn, use_xyz=use_xyz,
            pool_method=pool_method, instance_norm=instance_norm
        )


class PointnetFPModule(nn.Module):
    r"""Propigates the features of one set to another"""

    def __init__(self, *, mlp: List[int], bn: bool = True):
        """
        :param mlp: list of int
        :param bn: whether to use batchnorm
        """
        super().__init__()
        self.mlp = pt_utils.SharedMLP(mlp, bn=bn)

    def forward(
            self, unknown: torch.Tensor, known: torch.Tensor, unknow_feats: torch.Tensor, known_feats: torch.Tensor
    ) -> torch.Tensor:
        """
        :param unknown: (B, n, 3) tensor of the xyz positions of the unknown features
        :param known: (B, m, 3) tensor of the xyz positions of the known features
        :param unknow_feats: (B, C1, n) tensor of the features to be propigated to
        :param known_feats: (B, C2, m) tensor of features to be propigated
        :return:
            new_features: (B, mlp[-1], n) tensor of the features of the unknown features
        """
        if known is not None:
            dist, idx = pointnet2_utils.three_nn(unknown, known)
            dist_recip = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm

            interpolated_feats = pointnet2_utils.three_interpolate(known_feats, idx, weight)
        else:
            interpolated_feats = known_feats.expand(*known_feats.size()[0:2], unknown.size(1))

        if unknow_feats is not None:
            new_features = torch.cat([interpolated_feats, unknow_feats], dim=1)  # (B, C2 + C1, n)
        else:
            new_features = interpolated_feats

        new_features = new_features.unsqueeze(-1)

        new_features = self.mlp(new_features)

        return new_features.squeeze(-1)


if __name__ == "__main__":
    pass
