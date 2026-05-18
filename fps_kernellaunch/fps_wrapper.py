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

    def benchmark(self, xyz, npoints, warmup=5, repeats=100):
        """
        Benchmark FPS kernel performance.

        Returns:
            avg_ms: average time in milliseconds
        """
        assert xyz.is_npu
        B, N, C = xyz.shape
        xyz_t = xyz.permute(0, 2, 1).contiguous().reshape(B, -1)

        self._warmup()

        # Warmup
        idx = torch.zeros(B, npoints, dtype=torch.int32, device=xyz.device)
        for _ in range(warmup):
            torch.npu.synchronize()
            for b in range(B):
                self.lib.fps_run_device(
                    ctypes.c_void_p(xyz_t[b].data_ptr()),
                    ctypes.c_void_p(idx[b].data_ptr()),
                )
            torch.npu.synchronize()

        # Measure
        times = []
        for _ in range(repeats):
            torch.npu.synchronize()
            t0 = time.perf_counter()
            for b in range(B):
                self.lib.fps_run_device(
                    ctypes.c_void_p(xyz_t[b].data_ptr()),
                    ctypes.c_void_p(idx[b].data_ptr()),
                )
            torch.npu.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

        avg_ms = sum(times) / len(times)
        min_ms = min(times)
        max_ms = max(times)
        print(f"AscendC FPS: B={B}, N={N}, npoints={npoints}")
        print(f"  avg={avg_ms:.3f}ms, min={min_ms:.3f}ms, max={max_ms:.3f}ms")
        return avg_ms


def pytorch_fps(xyz, npoints):
    """Pure PyTorch FPS for comparison."""
    device = xyz.device
    B, N, _ = xyz.shape
    xyz_t = xyz.permute(0, 2, 1)  # [B, 3, N]

    farthest = torch.zeros(B, dtype=torch.long, device=device)
    distances = torch.full((B, N), 1e10, dtype=torch.float32, device=device)
    indices = torch.zeros(B, npoints, dtype=torch.long, device=device)

    for i in range(npoints):
        centroid = xyz_t.gather(2, farthest.reshape(B, 1, 1).expand(B, 3, 1)).squeeze(2)  # [B, 3]
        dist = torch.sum((xyz_t - centroid.unsqueeze(2)) ** 2, dim=1)  # [B, N]
        distances = torch.min(distances, dist)
        farthest = torch.argmax(distances, dim=1)  # [B]
        indices[:, i] = farthest

    return indices


def run_bench(name, fn, xyz, npoints, warmup=5, repeats=100):
    """Generic benchmark: fn(xyz, npoints) -> idx, returns (avg_ms, idx)."""
    # Warmup
    for _ in range(warmup):
        fn(xyz, npoints)
    torch.npu.synchronize()

    # Measure
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

    N = 1024
    npoints = 512
    B = 1
    xyz = torch.randn(B, N, 3, device='npu:0', dtype=torch.float32)

    print(f"=== FPS Performance Benchmark ===")
    print(f"B={B}, N={N}, npoints={npoints}, device=npu:0\n")

    # --- 1. PyTorch pure (Python for loop) ---
    def bench_pytorch_fps(xyz, np):
        return pytorch_fps(xyz, np)

    py_ms, py_idx = run_bench("PyTorch (pure)", bench_pytorch_fps, xyz, npoints)

    # --- 2. pointnet2_ops NPU ---
    try:
        import pointnet2_ops

        def bench_pn2_ops_fps(xyz, np):
            return pointnet2_ops._furthest_point_sampling(xyz, np)

        ops_ms, ops_idx = run_bench("pointnet2_ops", bench_pn2_ops_fps, xyz, npoints)
    except Exception as e:
        print(f"  pointnet2_ops: SKIP ({e})")
        ops_ms, ops_idx = None, None

    # --- 3. AscendC kernel ---
    try:
        fps_ascendc = FurthestPointSamplingAscendC()
        # Warmup done inside benchmark()
        def bench_ascendc_fps(xyz, np):
            return fps_ascendc(xyz, np)

        ac_ms, ac_idx = run_bench("AscendC kernel", bench_ascendc_fps, xyz, npoints)
    except Exception as e:
        print(f"  AscendC kernel: SKIP ({e})")
        ac_ms, ac_idx = None, None

    # --- Summary ---
    print(f"\n=== Summary ===")
    print(f"  PyTorch (pure): {py_ms:.3f}ms  (baseline)")
    if ops_ms is not None:
        print(f"  pointnet2_ops:  {ops_ms:.3f}ms  ({py_ms/ops_ms:.1f}x faster than PyTorch)")
    if ac_ms is not None:
        print(f"  AscendC kernel: {ac_ms:.3f}ms  ({py_ms/ac_ms:.1f}x faster than PyTorch)")

    # --- Correctness ---
    print(f"\n=== Correctness ===")
    if ac_idx is not None:
        match = torch.equal(py_idx, ac_idx)
        print(f"  PyTorch vs AscendC: {'MATCH' if match else 'MISMATCH'}")
        if not match:
            diff = (py_idx != ac_idx).sum().item()
            print(f"    Mismatches: {diff}/{npoints}")
            print(f"    PyTorch[:10]: {py_idx[0, :10].tolist()}")
            print(f"    AscendC[:10]: {ac_idx[0, :10].tolist()}")
    if ops_idx is not None:
        match = torch.equal(py_idx, ops_idx)
        print(f"  PyTorch vs pn2_ops: {'MATCH' if match else 'MISMATCH'}")
        if not match:
            diff = (py_idx != ops_idx).sum().item()
            print(f"    Mismatches: {diff}/{npoints}")
