"""
AscendC FPS kernel Python wrapper via ctypes.

Usage:
    import torch
    import torch_npu

    from fps_ascendc import FurthestPointSamplingAscendC, FurthestPointSamplingMultiCore

    fps_8c = FurthestPointSamplingMultiCore()
    xyz = torch.randn(1, 1024, 3, device='npu:0', dtype=torch.float32)
    idx = fps_8c(xyz, 512)  # [1, 512] int64
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


class FurthestPointSamplingAscendC:
    """Single-core AscendC FPS kernel."""

    def __init__(self):
        self.lib = _load_lib("libfps_host.so")
        self.lib.fps_run_device.restype = ctypes.c_int
        self.lib.fps_run_device.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self._warmup_done = False

    def _warmup(self):
        if self._warmup_done:
            return
        dummy_xyz = torch.zeros(3 * 1024, device='npu:0', dtype=torch.float32)
        dummy_idx = torch.zeros(512, device='npu:0', dtype=torch.int32)
        torch.npu.synchronize()
        self.lib.fps_run_device(
            ctypes.c_void_p(dummy_xyz.data_ptr()),
            ctypes.c_void_p(dummy_idx.data_ptr()),
        )
        self._warmup_done = True

    @torch.no_grad()
    def __call__(self, xyz, npoints):
        assert xyz.is_npu and xyz.dtype == torch.float32 and xyz.dim() == 3
        B, N, C = xyz.shape
        assert C == 3 and N == 1024 and npoints == 512

        self._warmup()

        xyz_t = xyz.permute(0, 2, 1).contiguous().reshape(B, -1)
        idx = torch.zeros(B, npoints, dtype=torch.int32, device=xyz.device)
        torch.npu.synchronize()

        for b in range(B):
            ret = self.lib.fps_run_device(
                ctypes.c_void_p(xyz_t[b].data_ptr()),
                ctypes.c_void_p(idx[b].data_ptr()),
            )
            assert ret == 0, f"fps_run_device failed for batch {b}"

        return idx.long()


class FurthestPointSamplingMultiCore:
    """8-core AscendC FPS kernel (v3dbg final: GetValue→UB, SyncAll 3-arg, stageBuf)."""

    def __init__(self):
        self.lib = _load_lib("libfps_host_mc_v3_dbg.so")
        self.lib.fps_run_mc_v3_dbg.restype = ctypes.c_int
        self.lib.fps_run_mc_v3_dbg.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int,
        ]
        self._warmup_done = False

    def _warmup(self):
        if self._warmup_done:
            return
        dummy_xyz = torch.zeros(3 * 1024, device='npu:0', dtype=torch.float32)
        dummy_idx = torch.zeros(512, device='npu:0', dtype=torch.int32)
        torch.npu.synchronize()
        self.lib.fps_run_mc_v3_dbg(
            ctypes.c_void_p(dummy_xyz.data_ptr()),
            ctypes.c_void_p(dummy_idx.data_ptr()),
            None, 0,
        )
        self._warmup_done = True

    @torch.no_grad()
    def __call__(self, xyz, npoints):
        assert xyz.is_npu and xyz.dtype == torch.float32 and xyz.dim() == 3
        B, N, C = xyz.shape
        assert C == 3 and N == 1024 and npoints == 512

        self._warmup()

        xyz_t = xyz.permute(0, 2, 1).contiguous().reshape(B, -1)
        idx = torch.zeros(B, npoints, dtype=torch.int32, device=xyz.device)
        torch.npu.synchronize()

        for b in range(B):
            ret = self.lib.fps_run_mc_v3_dbg(
                ctypes.c_void_p(xyz_t[b].data_ptr()),
                ctypes.c_void_p(idx[b].data_ptr()),
                None, 0,
            )
            assert ret == 0, f"fps_run_mc_v3_dbg failed for batch {b}"

        return idx.long()
