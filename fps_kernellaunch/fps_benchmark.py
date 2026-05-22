"""
FPS Performance Benchmark: PyTorch vs 1-core AscendC vs 8-core AscendC.

Usage:
    bash run.sh -r npu
    export LD_LIBRARY_PATH=$(pwd)/out/lib:$LD_LIBRARY_PATH
    python fps_benchmark.py
"""

import time
import torch
import torch_npu  # noqa: F401
import pointnet2_ops

from fps_ascendc import FurthestPointSamplingAscendC, FurthestPointSamplingMultiCore

N = 1024
NPOINTS = 512
WARMUP = 5
REPEATS = 100


def run_bench(name, fn, xyz, npoints):
    for _ in range(WARMUP):
        fn(xyz, npoints)
    torch.npu.synchronize()

    times = []
    idx = None
    for _ in range(REPEATS):
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


def main():
    B = 1
    xyz = torch.randn(B, N, 3, device='npu:0', dtype=torch.float32)

    print(f"=== FPS Performance Benchmark ===")
    print(f"B={B}, N={N}, npoints={NPOINTS}, device=npu:0")
    print(f"warmup={WARMUP}, repeats={REPEATS}\n")

    # --- PyTorch ---
    py_ms, py_idx = run_bench(
        "PyTorch FPS",
        lambda x, np: pointnet2_ops._furthest_point_sampling(x, np),
        xyz, NPOINTS,
    )

    # --- AscendC 1-core ---
    try:
        fps_1c = FurthestPointSamplingAscendC()
        ac_ms, ac_idx = run_bench("AscendC (1 core)", fps_1c, xyz, NPOINTS)
    except Exception as e:
        print(f"  AscendC (1 core): SKIP ({e})")
        ac_ms, ac_idx = None, None

    # --- AscendC 8-core ---
    try:
        fps_8c = FurthestPointSamplingMultiCore()
        mc_ms, mc_idx = run_bench("AscendC (8 cores)", fps_8c, xyz, NPOINTS)
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
            print(f"    Mismatches: {(py_idx != ac_idx).sum().item()}/{NPOINTS}")
    if mc_idx is not None:
        match = torch.equal(py_idx, mc_idx)
        print(f"  vs AscendC (8 cores): {'MATCH' if match else 'MISMATCH'}")
        if not match:
            print(f"    Mismatches: {(py_idx != mc_idx).sum().item()}/{NPOINTS}")


if __name__ == "__main__":
    main()
