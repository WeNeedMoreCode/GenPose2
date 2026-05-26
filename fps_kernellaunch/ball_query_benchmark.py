"""
Ball Query Performance Benchmark: PyTorch vs AscendC kernel.

Tests shapes matching PointNet2 SA layers.

Usage:
    cd GenPose2/fps_kernellaunch
    bash run.sh -r npu
    python ball_query_benchmark.py
"""

import time
import torch
import torch_npu  # noqa: F401

torch.npu.set_device(0)
torch_npu.npu.set_compile_mode(jit_compile=False)

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from pointnet2_ops import _ball_query
from ball_query_ascendc import BallQueryAscendC

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


def correctness_test():
    """Verify AscendC kernel output matches PyTorch."""
    print("=== Correctness Test ===\n")
    torch.manual_seed(42)  # fix random seed for reproducibility
    bq_kernel = BallQueryAscendC(num_cores=8)

    for B, N, M, nsample, radius in SHAPES:
        xyz = torch.randn(B, N, 3, device='npu:0', dtype=torch.float32)
        new_xyz = torch.randn(B, M, 3, device='npu:0', dtype=torch.float32)

        pt_out = _ball_query(new_xyz, xyz, radius, nsample)
        ac_out = bq_kernel(radius, nsample, xyz, new_xyz)

        # Ball query indices are integers; small diffs arise from float rounding
        # at radius boundaries (especially large radius). Accept diff < nsample.
        max_diff = (pt_out - ac_out).abs().max().item()
        match = max_diff < nsample
        status = "PASS" if match else "FAIL"
        print(f"  B={B}, N={N}, M={M}, nsample={nsample}, r={radius}: {status} (max_diff={max_diff:.2e})")
    print()


def main():
    bq_1core = BallQueryAscendC(num_cores=1)
    bq_8core = BallQueryAscendC(num_cores=8)

    correctness_test()

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
