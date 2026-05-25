"""
AscendC Ball Query kernel Python wrapper via ctypes (dynamic shape).

Usage:
    import torch
    import torch_npu

    from ball_query_ascendc import BallQueryAscendC

    bq = BallQueryAscendC(num_cores=8)
    xyz = torch.randn(1, 1024, 3, device='npu:0', dtype=torch.float32)
    new_xyz = torch.randn(1, 512, 3, device='npu:0', dtype=torch.float32)
    idx = bq(0.02, 32, xyz, new_xyz)  # [1, 512, 32]

Monkey-patch into PointNet2:
    from ball_query_ascendc import patch_ball_query
    patch_ball_query(num_cores=8)
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


class BallQueryAscendC:
    """Multi-core AscendC Ball Query kernel with dynamic shape support."""

    def __init__(self, num_cores=8):
        self.num_cores = num_cores
        self.lib = _load_lib("libball_query_host_dynamic.so")
        self.lib.ball_query_run_dynamic.restype = ctypes.c_int
        self.lib.ball_query_run_dynamic.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
            ctypes.c_int32, ctypes.c_float,
            ctypes.c_int32, ctypes.c_void_p,
        ]

    @torch.no_grad()
    def __call__(self, radius, nsample, xyz, new_xyz):
        """
        radius: float
        nsample: int
        xyz:      [B, N, 3] float32 on NPU
        new_xyz:  [B, M, 3] float32 on NPU
        returns:  [B, M, nsample] int32
        """
        assert xyz.is_npu and xyz.dtype == torch.float32 and xyz.dim() == 3
        assert new_xyz.is_npu and new_xyz.dtype == torch.float32 and new_xyz.dim() == 3

        B, N, _ = xyz.shape
        M = new_xyz.shape[1]

        idx = torch.zeros(B, M, nsample, dtype=torch.int32, device=xyz.device)
        stream_ptr = _get_npu_stream()

        ret = self.lib.ball_query_run_dynamic(
            ctypes.c_void_p(xyz.data_ptr()),
            ctypes.c_void_p(new_xyz.data_ptr()),
            ctypes.c_void_p(idx.data_ptr()),
            ctypes.c_int32(B),
            ctypes.c_int32(N),
            ctypes.c_int32(M),
            ctypes.c_int32(nsample),
            ctypes.c_float(radius),
            ctypes.c_int32(self.num_cores),
            ctypes.c_void_p(stream_ptr),
        )
        assert ret == 0, f"ball_query_run_dynamic failed: ret={ret}"
        return idx


_patched = False


def patch_ball_query(num_cores=8):
    """Replace pointnet2_utils.ball_query with AscendC kernel."""
    global _patched
    if _patched:
        return
    kernel = BallQueryAscendC(num_cores=num_cores)

    from networks.pts_encoder.pointnet2_utils.pointnet2 import pointnet2_utils

    _orig_fn = pointnet2_utils.ball_query

    def _ball_query_replacement(radius, nsample, xyz, new_xyz):
        # xyz and new_xyz are tensors on NPU
        # The original signature is ball_query(radius, nsample, xyz, new_xyz)
        # where xyz is [B, N, 3], new_xyz is [B, M, 3]
        return kernel(radius, nsample, xyz, new_xyz)

    pointnet2_utils.ball_query = _ball_query_replacement
    _patched = True
