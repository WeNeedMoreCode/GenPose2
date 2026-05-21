"""
AscendC FPS kernel Python wrapper via ctypes.

Usage:
    import torch
    import torch_npu

    from fps_wrapper import FurthestPointSamplingAscendC, FurthestPointSamplingMultiCore

    fps_1c = FurthestPointSamplingAscendC()
    fps_8c = FurthestPointSamplingMultiCore()
    xyz = torch.randn(1, 1024, 3, device='npu:0', dtype=torch.float32)
    idx = fps_8c(xyz, 512)  # [1, 512] int64
"""

import ctypes
import os
import time
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


def run_bench(name, fn, xyz, npoints, warmup=5, repeats=100):
    """Generic benchmark: fn(xyz, npoints) -> idx, returns (avg_ms, idx)."""
    for _ in range(warmup):
        fn(xyz, npoints)
    torch.npu.synchronize()

    times = []
    idx = None
    for _ in range(repeats):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        idx = fn(xyz, npoints)
        torch.npu.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    avg_ms = sum(times) / len(times)
    min_ms = min(times)
    max_ms = max(times)
    print(f"  {name}: avg={avg_ms:.3f}ms, min={min_ms:.3f}ms, max={max_ms:.3f}ms")
    return avg_ms, idx


if __name__ == "__main__":
    import torch_npu  # noqa: F401
    import pointnet2_ops

    N = 1024
    npoints = 512
    B = 1
    xyz = torch.randn(B, N, 3, device='npu:0', dtype=torch.float32)

    print(f"=== FPS Performance Benchmark ===")
    print(f"B={B}, N={N}, npoints={npoints}, device=npu:0\n")

    # --- 1. PyTorch FPS (pointnet2_ops, baseline) ---
    def bench_pytorch_fps(xyz, np):
        return pointnet2_ops._furthest_point_sampling(xyz, np)

    py_ms, py_idx = run_bench("PyTorch FPS", bench_pytorch_fps, xyz, npoints)

    # --- 2. AscendC 1-core ---
    try:
        fps_1c = FurthestPointSamplingAscendC()
        ac_ms, ac_idx = run_bench("AscendC (1 core)", fps_1c, xyz, npoints)
    except Exception as e:
        print(f"  AscendC (1 core): SKIP ({e})")
        ac_ms, ac_idx = None, None

    # --- 3. AscendC 8-core ---
    try:
        fps_8c = FurthestPointSamplingMultiCore()
        mc_ms, mc_idx = run_bench("AscendC (8 cores)", fps_8c, xyz, npoints)
    except Exception as e:
        print(f"  AscendC (8 cores): SKIP ({e})")
        mc_ms, mc_idx = None, None

    # --- Summary ---
    print(f"\n=== Summary ===")
    print(f"  PyTorch FPS:         {py_ms:.3f}ms  (baseline)")
    if ac_ms is not None:
        print(f"  AscendC (1 core):    {ac_ms:.3f}ms  ({py_ms/ac_ms:.1f}x faster)")
    if mc_ms is not None:
        print(f"  AscendC (8 cores):   {mc_ms:.3f}ms  ({py_ms/mc_ms:.1f}x faster)")

    # --- Correctness ---
    print(f"\n=== Correctness (baseline: pointnet2_ops) ===")
    print(f"  Baseline[:10]: {py_idx[0, :10].tolist()}")
    if ac_idx is not None:
        match = torch.equal(py_idx, ac_idx)
        print(f"  vs AscendC (1 core): {'MATCH' if match else 'MISMATCH'}")
        if not match:
            print(f"    Mismatches: {(py_idx != ac_idx).sum().item()}/{npoints}")
    if mc_idx is not None:
        match = torch.equal(py_idx, mc_idx)
        print(f"  vs AscendC (8 cores): {'MATCH' if match else 'MISMATCH'}")
        if not match:
            print(f"    Mismatches: {(py_idx != mc_idx).sum().item()}/{npoints}")
