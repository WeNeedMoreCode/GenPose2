"""
AscendC FPS kernel Python wrapper via ctypes (dynamic shape).

Usage:
    import torch
    import torch_npu

    from fps_ascendc import FurthestPointSamplingAscendC

    fps = FurthestPointSamplingAscendC(num_cores=8)
    xyz = torch.randn(1, 1024, 3, device='npu:0', dtype=torch.float32)
    idx = fps(xyz, 512)  # [1, 512] int64

Supports any N divisible by 64 and npoints <= N.

Monkey-patch into PointNet2:
    from fps_ascendc import patch_pointnet2_fps
    patch_pointnet2_fps(num_cores=8)
    # After this, pointnet2_utils.furthest_point_sample uses AscendC kernel
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
    stream = torch.npu.current_stream()
    # stream.stream is a property returning int (raw pointer), or method
    raw = getattr(stream, 'stream', None)
    if callable(raw):
        raw = raw()
    if raw is None or raw == 0:
        return None
    print(f"[fps_ascendc] NPU stream pointer: {hex(raw)}")
    return raw


class FurthestPointSamplingAscendC:
    """Multi-core AscendC FPS kernel with dynamic shape support."""

    def __init__(self, num_cores=8):
        self.num_cores = num_cores
        self.lib = _load_lib("libfps_host_dynamic.so")
        self.lib.fps_run_dynamic.restype = ctypes.c_int
        self.lib.fps_run_dynamic.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
            ctypes.c_void_p,
        ]

    @torch.no_grad()
    def __call__(self, xyz, npoints):
        assert xyz.is_npu and xyz.dtype == torch.float32 and xyz.dim() == 3
        B, N, C = xyz.shape
        assert C == 3

        xyz_t = xyz.permute(0, 2, 1).contiguous().reshape(B, -1)
        idx = torch.zeros(B, npoints, dtype=torch.int32, device=xyz.device)

        stream_ptr = _get_npu_stream()

        for b in range(B):
            ret = self.lib.fps_run_dynamic(
                ctypes.c_void_p(xyz_t[b].data_ptr()),
                ctypes.c_void_p(idx[b].data_ptr()),
                ctypes.c_int32(N),
                ctypes.c_int32(npoints),
                ctypes.c_int32(self.num_cores),
                ctypes.c_void_p(stream_ptr) if stream_ptr else ctypes.c_void_p(0),
            )
            assert ret == 0, f"fps_run_dynamic failed for batch {b}: ret={ret}"

        return idx.long()


_patched = False

def patch_pointnet2_fps(num_cores=8):
    """Replace pointnet2_utils.furthest_point_sample with AscendC kernel."""
    global _patched
    if _patched:
        return
    kernel = FurthestPointSamplingAscendC(num_cores=num_cores)

    # Quick self-test
    print("[fps_ascendc] Running self-test...")
    test_xyz = torch.randn(1, 1024, 3, device='npu:0', dtype=torch.float32)
    test_idx = kernel(test_xyz, 512)
    assert test_idx.shape == (1, 512), f"Self-test failed: shape={test_idx.shape}"
    print(f"[fps_ascendc] Self-test passed: {test_idx.shape}")

    from networks.pts_encoder.pointnet2_utils.pointnet2 import pointnet2_utils
    pointnet2_utils.furthest_point_sample = kernel
    _patched = True
    print(f"[fps_ascendc] Patched furthest_point_sample -> AscendC {num_cores}-core kernel")
