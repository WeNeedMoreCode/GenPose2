"""
AscendC FPS kernel Python wrapper via ctypes.

Usage:
    import torch
    import torch_npu

    from fps_wrapper import FurthestPointSamplingAscendC

    fps = FurthestPointSamplingAscendC()
    xyz = torch.randn(1, 1024, 3, device='npu:0', dtype=torch.float32)
    idx = fps(xyz, 512)  # [1, 512] int64
"""

import ctypes
import os
import time
import torch

_LIB = None

def _get_lib():
    global _LIB
    if _LIB is None:
        # libfps_host.so is in the same directory or out/lib/
        script_dir = os.path.dirname(os.path.abspath(__file__))
        lib_path = os.path.join(script_dir, "out", "lib", "libfps_host.so")
        if not os.path.exists(lib_path):
            lib_path = os.path.join(script_dir, "libfps_host.so")
        if not os.path.exists(lib_path):
            raise FileNotFoundError(
                f"libfps_host.so not found. Run 'bash run.sh -r npu' first. "
                f"Searched: {script_dir}/out/lib/ and {script_dir}/"
            )
        _LIB = ctypes.CDLL(lib_path)
        _LIB.fps_run_device.restype = ctypes.c_int
        _LIB.fps_run_device.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    return _LIB


class FurthestPointSamplingAscendC:
    def __init__(self):
        self.lib = _get_lib()
        self._warmup_done = False

    def _warmup(self):
        """First call may be slow due to stream creation overhead."""
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
        """
        Args:
            xyz: [B, N, 3] float32 tensor on NPU
            npoints: int, number of points to sample (must be 512 for current kernel)
        Returns:
            idx: [B, npoints] int64 tensor on NPU
        """
        assert xyz.is_npu, "xyz must be on NPU"
        assert xyz.dtype == torch.float32, "xyz must be float32"
        assert xyz.dim() == 3, "xyz must be [B, N, 3]"

        B, N, C = xyz.shape
        assert C == 3
        assert N == 1024, f"Current kernel is compiled for N=1024, got N={N}"
        assert npoints == 512, f"Current kernel is compiled for npoints=512, got {npoints}"

        self._warmup()

        # Transpose to [3, N] layout: x[0:N], y[N:2N], z[2N:3N]
        xyz_t = xyz.permute(0, 2, 1).contiguous()  # [B, 3, N]
        xyz_flat = xyz_t.reshape(B, -1)  # [B, 3*N]

        idx = torch.zeros(B, npoints, dtype=torch.int32, device=xyz.device)

        # Sync torch_npu ops before kernel launch
        torch.npu.synchronize()

        for b in range(B):
            ret = self.lib.fps_run_device(
                ctypes.c_void_p(xyz_flat[b].data_ptr()),
                ctypes.c_void_p(idx[b].data_ptr()),
            )
            assert ret == 0, f"fps_run_device failed for batch {b}"

        return idx.long()


_LIB_MC = None

def _get_mc_lib():
    global _LIB_MC
    if _LIB_MC is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        lib_path = os.path.join(script_dir, "out", "lib", "libfps_host_multicore.so")
        if not os.path.exists(lib_path):
            lib_path = os.path.join(script_dir, "libfps_host_multicore.so")
        if not os.path.exists(lib_path):
            raise FileNotFoundError(
                f"libfps_host_multicore.so not found. Run 'bash run.sh -r npu' first. "
                f"Searched: {script_dir}/out/lib/ and {script_dir}/"
            )
        _LIB_MC = ctypes.CDLL(lib_path)
        _LIB_MC.fps_run_multicore.restype = ctypes.c_int
        _LIB_MC.fps_run_multicore.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    return _LIB_MC


class FurthestPointSamplingMultiCore:
    """Multi-core AscendC FPS using 8 AI Cores."""

    def __init__(self):
        self.lib = _get_mc_lib()
        self._warmup_done = False

    def _warmup(self):
        if self._warmup_done:
            return
        dummy_xyz = torch.zeros(3 * 1024, device='npu:0', dtype=torch.float32)
        dummy_idx = torch.zeros(512, device='npu:0', dtype=torch.int32)
        torch.npu.synchronize()
        self.lib.fps_run_multicore(
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
            ret = self.lib.fps_run_multicore(
                ctypes.c_void_p(xyz_t[b].data_ptr()),
                ctypes.c_void_p(idx[b].data_ptr()),
            )
            assert ret == 0, f"fps_run_multicore failed for batch {b}"

        return idx.long()


_LIB_MC_V2 = None

def _get_mc_v2_lib():
    global _LIB_MC_V2
    if _LIB_MC_V2 is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        lib_path = os.path.join(script_dir, "out", "lib", "libfps_host_multicore_v2.so")
        if not os.path.exists(lib_path):
            lib_path = os.path.join(script_dir, "libfps_host_multicore_v2.so")
        if not os.path.exists(lib_path):
            raise FileNotFoundError(
                f"libfps_host_multicore_v2.so not found. Run 'bash run.sh -r npu' first. "
                f"Searched: {script_dir}/out/lib/ and {script_dir}/"
            )
        _LIB_MC_V2 = ctypes.CDLL(lib_path)
        _LIB_MC_V2.fps_run_multicore_v2.restype = ctypes.c_int
        _LIB_MC_V2.fps_run_multicore_v2.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    return _LIB_MC_V2


class FurthestPointSamplingMultiCoreV2:
    """Multi-core AscendC FPS v2: DataCopy-based cross-core communication."""

    def __init__(self):
        self.lib = _get_mc_v2_lib()
        self._warmup_done = False

    def _warmup(self):
        if self._warmup_done:
            return
        dummy_xyz = torch.zeros(3 * 1024, device='npu:0', dtype=torch.float32)
        dummy_idx = torch.zeros(512, device='npu:0', dtype=torch.int32)
        torch.npu.synchronize()
        self.lib.fps_run_multicore_v2(
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
            ret = self.lib.fps_run_multicore_v2(
                ctypes.c_void_p(xyz_t[b].data_ptr()),
                ctypes.c_void_p(idx[b].data_ptr()),
            )
            assert ret == 0, f"fps_run_multicore_v2 failed for batch {b}"

        return idx.long()


_LIB_MC_V3 = None

def _get_mc_v3_lib():
    global _LIB_MC_V3
    if _LIB_MC_V3 is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        lib_path = os.path.join(script_dir, "out", "lib", "libfps_host_multicore_v3.so")
        if not os.path.exists(lib_path):
            lib_path = os.path.join(script_dir, "libfps_host_multicore_v3.so")
        if not os.path.exists(lib_path):
            raise FileNotFoundError(
                f"libfps_host_multicore_v3.so not found. Run 'bash run.sh -r npu' first. "
                f"Searched: {script_dir}/out/lib/ and {script_dir}/"
            )
        _LIB_MC_V3 = ctypes.CDLL(lib_path)
        _LIB_MC_V3.fps_run_multicore_v3.restype = ctypes.c_int
        _LIB_MC_V3.fps_run_multicore_v3.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    return _LIB_MC_V3


class FurthestPointSamplingMultiCoreV3:
    """Multi-core AscendC FPS v3: pure vector+DMA, no scalar SetValue for cross-core."""

    def __init__(self):
        self.lib = _get_mc_v3_lib()
        self._warmup_done = False

    def _warmup(self):
        if self._warmup_done:
            return
        dummy_xyz = torch.zeros(3 * 1024, device='npu:0', dtype=torch.float32)
        dummy_idx = torch.zeros(512, device='npu:0', dtype=torch.int32)
        torch.npu.synchronize()
        self.lib.fps_run_multicore_v3(
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
            ret = self.lib.fps_run_multicore_v3(
                ctypes.c_void_p(xyz_t[b].data_ptr()),
                ctypes.c_void_p(idx[b].data_ptr()),
            )
            assert ret == 0, f"fps_run_multicore_v3 failed for batch {b}"

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

    # --- 2. AscendC kernel (1 core) ---
    try:
        fps_ascendc = FurthestPointSamplingAscendC()
        def bench_ascendc_fps(xyz, np):
            return fps_ascendc(xyz, np)

        ac_ms, ac_idx = run_bench("AscendC (1 core)", bench_ascendc_fps, xyz, npoints)
    except Exception as e:
        print(f"  AscendC (1 core): SKIP ({e})")
        ac_ms, ac_idx = None, None

    # --- 3. AscendC kernel (8 cores, v1 SetValue) ---
    try:
        fps_mc = FurthestPointSamplingMultiCore()
        def bench_mc_fps(xyz, np):
            return fps_mc(xyz, np)

        mc_ms, mc_idx = run_bench("AscendC (8 cores v1)", bench_mc_fps, xyz, npoints)
    except Exception as e:
        print(f"  AscendC (8 cores v1): SKIP ({e})")
        mc_ms, mc_idx = None, None

    # --- 4. AscendC kernel (8 cores, v2 DataCopy) ---
    try:
        fps_mc_v2 = FurthestPointSamplingMultiCoreV2()
        def bench_mc_v2_fps(xyz, np):
            return fps_mc_v2(xyz, np)

        mc_v2_ms, mc_v2_idx = run_bench("AscendC (8 cores v2)", bench_mc_v2_fps, xyz, npoints)
    except Exception as e:
        print(f"  AscendC (8 cores v2): SKIP ({e})")
        mc_v2_ms, mc_v2_idx = None, None

    # --- 5. AscendC kernel (8 cores, v3 pure vector+DMA) ---
    try:
        fps_mc_v3 = FurthestPointSamplingMultiCoreV3()
        def bench_mc_v3_fps(xyz, np):
            return fps_mc_v3(xyz, np)

        mc_v3_ms, mc_v3_idx = run_bench("AscendC (8 cores v3)", bench_mc_v3_fps, xyz, npoints)
    except Exception as e:
        print(f"  AscendC (8 cores v3): SKIP ({e})")
        mc_v3_ms, mc_v3_idx = None, None

    # --- Summary ---
    print(f"\n=== Summary ===")
    print(f"  PyTorch FPS:         {py_ms:.3f}ms  (baseline)")
    if ac_ms is not None:
        print(f"  AscendC (1 core):    {ac_ms:.3f}ms  ({py_ms/ac_ms:.1f}x faster)")
    if mc_ms is not None:
        print(f"  AscendC (8c v1):     {mc_ms:.3f}ms  ({py_ms/mc_ms:.1f}x faster)")
    if mc_v2_ms is not None:
        print(f"  AscendC (8c v2):     {mc_v2_ms:.3f}ms  ({py_ms/mc_v2_ms:.1f}x faster)")
    if mc_v3_ms is not None:
        print(f"  AscendC (8c v3):     {mc_v3_ms:.3f}ms  ({py_ms/mc_v3_ms:.1f}x faster)")

    # --- Correctness (pointnet2_ops as baseline) ---
    print(f"\n=== Correctness (baseline: pointnet2_ops) ===")
    print(f"  Baseline[:10]: {py_idx[0, :10].tolist()}")
    if ac_idx is not None:
        match = torch.equal(py_idx, ac_idx)
        print(f"  Baseline vs AscendC (1 core): {'MATCH' if match else 'MISMATCH'}")
        if not match:
            diff = (py_idx != ac_idx).sum().item()
            print(f"    Mismatches: {diff}/{npoints}")
            print(f"    Ascend1c[:10]: {ac_idx[0, :10].tolist()}")
    if mc_idx is not None:
        match = torch.equal(py_idx, mc_idx)
        print(f"  Baseline vs AscendC (8c v1): {'MATCH' if match else 'MISMATCH'}")
        if not match:
            diff = (py_idx != mc_idx).sum().item()
            print(f"    Mismatches: {diff}/{npoints}")
            print(f"    Ascend8c_v1[:10]: {mc_idx[0, :10].tolist()}")
    if mc_v2_idx is not None:
        match = torch.equal(py_idx, mc_v2_idx)
        print(f"  Baseline vs AscendC (8c v2): {'MATCH' if match else 'MISMATCH'}")
        if not match:
            diff = (py_idx != mc_v2_idx).sum().item()
            print(f"    Mismatches: {diff}/{npoints}")
            print(f"    Ascend8c_v2[:10]: {mc_v2_idx[0, :10].tolist()}")
    if mc_v3_idx is not None:
        match = torch.equal(py_idx, mc_v3_idx)
        print(f"  Baseline vs AscendC (8c v3): {'MATCH' if match else 'MISMATCH'}")
        if not match:
            diff = (py_idx != mc_v3_idx).sum().item()
            print(f"    Mismatches: {diff}/{npoints}")
            print(f"    Ascend8c_v3[:10]: {mc_v3_idx[0, :10].tolist()}")
