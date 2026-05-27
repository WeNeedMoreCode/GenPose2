"""
AscendC GroupPoints kernel Python wrapper via ctypes (dynamic shape).

Usage:
    import torch
    import torch_npu

    from group_points_ascendc import GroupPointsAscendC

    gp = GroupPointsAscendC(num_cores=8)
    points = torch.randn(1, 390, 1024, device='npu:0', dtype=torch.float32)
    idx = torch.zeros(1, 512, 32, dtype=torch.int32, device='npu:0')
    out = gp(points, idx)  # [1, 390, 512, 32]

Monkey-patch into PointNet2:
    from group_points_ascendc import patch_group_points
    patch_group_points(num_cores=8)
"""

import ctypes
import os
import torch


def _load_lib(name):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    lib_path = os.path.join(script_dir, "out", "lib", name)
    if not os.path.exists(lib_path):
        lib_path = os.path.join(script_dir, name)
    if not os.path.exists(lib_path):
        raise FileNotFoundError(
            f"{name} not found. Run 'bash run.sh -r npu' first. "
            f"Searched: {script_dir}/out/lib/ and {script_dir}/"
        )
    # Ensure the build output directory is on LD_LIBRARY_PATH
    out_lib = os.path.join(script_dir, "out", "lib")
    ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    if out_lib not in ld_path.split(os.pathsep):
        os.environ["LD_LIBRARY_PATH"] = out_lib + os.pathsep + ld_path
    return ctypes.CDLL(lib_path)


def _get_npu_stream():
    """Get raw aclrtStream pointer from torch.npu."""
    s = torch.npu.current_stream()
    for attr in ['npu_stream', 'stream', '_stream', '_cstream']:
        val = getattr(s, attr, None)
        if callable(val):
            val = val()
        if isinstance(val, int) and val != 0:
            return val
    raise RuntimeError(
        f"Cannot get NPU stream pointer from {type(s)}. "
        f"Available attrs: {[a for a in dir(s) if not a.startswith('__')]}. "
    )


class GroupPointsAscendC:
    """Multi-core AscendC GroupPoints kernel with dynamic shape support."""

    def __init__(self, num_cores=8):
        self.num_cores = num_cores
        self.lib = _load_lib("libgroup_points_host_dynamic.so")
        self.lib.group_points_run_dynamic.restype = ctypes.c_int
        self.lib.group_points_run_dynamic.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
            ctypes.c_int32, ctypes.c_int32,
            ctypes.c_int32, ctypes.c_void_p,
        ]

    @torch.no_grad()
    def __call__(self, points, idx):
        """
        points: [B, C, N] float32 on NPU
        idx:    [B, npoint, nsample] int32 on NPU
        returns: [B, C, npoint, nsample] float32
        """
        assert points.is_npu and points.dtype == torch.float32 and points.dim() == 3
        assert idx.is_npu and idx.dtype == torch.int32 and idx.dim() == 3

        B, C, N = points.shape
        npoint = idx.shape[1]
        nsample = idx.shape[2]

        out = torch.zeros(B, C, npoint, nsample, dtype=torch.float32, device=points.device)
        stream_ptr = _get_npu_stream()

        ret = self.lib.group_points_run_dynamic(
            ctypes.c_void_p(points.data_ptr()),
            ctypes.c_void_p(idx.data_ptr()),
            ctypes.c_void_p(out.data_ptr()),
            ctypes.c_int32(B),
            ctypes.c_int32(C),
            ctypes.c_int32(N),
            ctypes.c_int32(npoint),
            ctypes.c_int32(nsample),
            ctypes.c_int32(self.num_cores),
            ctypes.c_void_p(stream_ptr),
        )
        assert ret == 0, f"group_points_run_dynamic failed: ret={ret}"
        return out


_patched = False

def patch_group_points(num_cores=8):
    """Replace pointnet2_utils.grouping_operation with AscendC kernel."""
    global _patched
    if _patched:
        return
    kernel = GroupPointsAscendC(num_cores=num_cores)

    from networks.pts_encoder.pointnet2_utils.pointnet2 import pointnet2_utils

    # grouping_operation expects (points, idx) where points is [B,C,N], idx is [B,npoint,nsample]
    # The autograd Function wrapper calls .apply(features, idx)
    # We need to handle both cases: direct call and .apply() wrapper
    _orig_fn = pointnet2_utils.grouping_operation

    def _grouping_replacement(features, idx):
        # features: [B, C, N], idx: [B, npoint, nsample]
        # idx might be int64 from autograd wrapper, convert to int32
        if idx.dtype != torch.int32:
            idx = idx.to(torch.int32)
        return kernel(features, idx)

    pointnet2_utils.grouping_operation = _grouping_replacement
    _patched = True
