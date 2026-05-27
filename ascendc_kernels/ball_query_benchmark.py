"""
Ball Query Performance Benchmark: PyTorch vs AscendC kernel.

Tests shapes matching PointNet2 SA layers.

Usage:
    cd GenPose2/fps_kernellaunch
    bash run.sh -r npu
    python ball_query_benchmark.py
"""

import ctypes
import time
import torch
import torch_npu  # noqa: F401

torch.npu.set_device(0)
torch_npu.npu.set_compile_mode(jit_compile=False)

import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, '..'))
from pointnet2_ops import _ball_query
from ball_query_ascendc import BallQueryAscendC, _get_npu_stream

WARMUP = 5
REPEATS = 50

# Shapes matching PointNet2 SA layers
# (B, N, M, nsample, radius)
SHAPES = [
    (1, 1024, 512, 32, 0.02),   # SA0
    (1, 512, 256, 32, 0.04),    # SA1
    (1, 256, 128, 32, 0.08),    # SA2
]


def run_bench(name, fn, radius, nsample, xyz, new_xyz):
    for _ in range(WARMUP):
        fn(radius, nsample, xyz, new_xyz)
    torch.npu.synchronize()

    times = []
    for _ in range(REPEATS):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        fn(radius, nsample, xyz, new_xyz)
        torch.npu.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    avg_ms = sum(times) / len(times)
    min_ms = min(times)
    print(f"  {name}: avg={avg_ms:.3f}ms, min={min_ms:.3f}ms")
    return avg_ms


def _py_ball_query_wrapper(radius, nsample, xyz, new_xyz):
    """Wrapper to match AscendC call signature: (radius, nsample, xyz, new_xyz)."""
    return _ball_query(new_xyz, xyz, radius, nsample)


def _load_debug_lib():
    """Load the debug kernel host wrapper .so (same Python-level signature)."""
    lib_path = os.path.join(SCRIPT_DIR, "out", "lib", "libball_query_debug.so")
    if not os.path.exists(lib_path):
        lib_path = os.path.join(SCRIPT_DIR, "libball_query_debug.so")
    if not os.path.exists(lib_path):
        raise FileNotFoundError(f"{lib_path} not found")
    lib = ctypes.CDLL(lib_path)
    lib.ball_query_debug_run.restype = ctypes.c_int
    lib.ball_query_debug_run.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
        ctypes.c_int32, ctypes.c_float,
        ctypes.c_int32, ctypes.c_void_p,
    ]
    return lib


def correctness_test():
    """Verify production kernel output matches PyTorch."""
    print("=== Correctness Test (production kernel) ===\n")
    torch.manual_seed(42)
    bq_kernel = BallQueryAscendC(num_cores=8)

    for B, N, M, nsample, radius in SHAPES:
        xyz = torch.randn(B, N, 3, device='npu:0', dtype=torch.float32)
        new_xyz = torch.randn(B, M, 3, device='npu:0', dtype=torch.float32)

        pt_out = _ball_query(new_xyz, xyz, radius, nsample)
        ac_out = bq_kernel(radius, nsample, xyz, new_xyz)

        max_diff = (pt_out - ac_out).abs().max().item()
        match = max_diff < nsample
        status = "PASS" if match else "FAIL"
        print(f"  B={B}, N={N}, M={M}, nsample={nsample}, r={radius}: {status} (max_diff={max_diff:.2e})")
    print()


def correctness_test_debug():
    """Verify debug kernel output matches PyTorch."""
    print("=== Correctness Test (debug kernel) ===\n")
    torch.manual_seed(42)
    lib = _load_debug_lib()
    num_cores = 8

    for B, N, M, nsample, radius in SHAPES:
        xyz = torch.randn(B, N, 3, device='npu:0', dtype=torch.float32)
        new_xyz = torch.randn(B, M, 3, device='npu:0', dtype=torch.float32)

        pt_out = _ball_query(new_xyz, xyz, radius, nsample)

        idx_ac = torch.zeros(B, M, nsample, dtype=torch.int32, device='npu:0')
        stream_ptr = _get_npu_stream()
        ret = lib.ball_query_debug_run(
            ctypes.c_void_p(xyz.data_ptr()),
            ctypes.c_void_p(new_xyz.data_ptr()),
            ctypes.c_void_p(idx_ac.data_ptr()),
            ctypes.c_int32(B), ctypes.c_int32(N), ctypes.c_int32(M),
            ctypes.c_int32(nsample), ctypes.c_float(radius),
            ctypes.c_int32(num_cores), ctypes.c_void_p(stream_ptr),
        )
        ac_out = idx_ac.to(torch.int64)
        assert ret == 0, f"debug kernel failed: ret={ret}"

        max_diff = (pt_out - ac_out).abs().max().item()
        match = max_diff < nsample
        status = "PASS" if match else "FAIL"
        print(f"  B={B}, N={N}, M={M}, nsample={nsample}, r={radius}: {status} (max_diff={max_diff:.2e})")
    print()


def main():
    bq_1core = BallQueryAscendC(num_cores=1)
    bq_8core = BallQueryAscendC(num_cores=8)

    correctness_test()
    correctness_test_debug()

    print("=== Ball Query Performance Benchmark ===")
    print(f"warmup={WARMUP}, repeats={REPEATS}\n")

    # B=1 shapes
    for B, N, M, nsample, radius in SHAPES:
        xyz = torch.randn(B, N, 3, device='npu:0', dtype=torch.float32)
        new_xyz = torch.randn(B, M, 3, device='npu:0', dtype=torch.float32)

        print(f"--- B={B}, N={N}, M={M}, nsample={nsample}, r={radius} ---")
        py_ms = run_bench("PyTorch", _py_ball_query_wrapper, radius, nsample, xyz, new_xyz)
        ac1_ms = run_bench("AscendC (1-core)", bq_1core, radius, nsample, xyz, new_xyz)
        ac8_ms = run_bench("AscendC (8-core)", bq_8core, radius, nsample, xyz, new_xyz)
        print(f"  Speedup: 1-core={py_ms/ac1_ms:.1f}x, 8-core={py_ms/ac8_ms:.1f}x\n")

    # Multi-batch tests
    for B in [8, 16, 32]:
        N, M, nsample, radius = 1024, 512, 32, 0.02
        xyz = torch.randn(B, N, 3, device='npu:0', dtype=torch.float32)
        new_xyz = torch.randn(B, M, 3, device='npu:0', dtype=torch.float32)
        print(f"--- Batch B={B}, N={N}, M={M}, nsample={nsample}, r={radius} ---")
        py_ms = run_bench("PyTorch", _py_ball_query_wrapper, radius, nsample, xyz, new_xyz)
        ac1_ms = run_bench("AscendC (1-core)", bq_1core, radius, nsample, xyz, new_xyz)
        ac8_ms = run_bench("AscendC (8-core)", bq_8core, radius, nsample, xyz, new_xyz)
        print(f"  Speedup: 1-core={py_ms/ac1_ms:.1f}x, 8-core={py_ms/ac8_ms:.1f}x\n")


if __name__ == "__main__":
    main()
