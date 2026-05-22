"""
FPS Performance Benchmark: PyTorch vs AscendC dynamic kernel.

Tests 3 SA layer shapes: 1024→512, 512→256, 256→128.

Usage:
    cd GenPose2/fps_kernellaunch
    bash run.sh -r npu
    python fps_benchmark.py
"""

import time
import torch
import torch_npu  # noqa: F401

torch.npu.set_device(0)

from fps_ascendc import FurthestPointSamplingAscendC

WARMUP = 5
REPEATS = 100
SHAPES = [
    (1024, 512),
    (512, 256),
    (256, 128),
]


def pytorch_fps(xyz, npoints):
    """PyTorch argmax loop implementation."""
    B, N, _ = xyz.shape
    device = xyz.device
    dtype = xyz.dtype
    dist = torch.zeros(B, N, device=device, dtype=dtype)
    idx = torch.zeros(B, npoints, device=device, dtype=torch.int64)
    farthest = torch.randint(0, N, (B,), device=device)
    batch_idx = torch.arange(B, device=device).unsqueeze(1).repeat(1, N)
    for i in range(npoints):
        idx[:, i] = farthest
        centroid = xyz[batch_idx, farthest, :].view(B, 1, 3)
        dist = torch.max(dist, (xyz - centroid).norm(dim=2))
        farthest = dist.argmax(dim=1)
        dist.scatter_(1, farthest.unsqueeze(1), 0)
    return idx


def run_bench(name, fn, xyz, npoints):
    for _ in range(WARMUP):
        fn(xyz, npoints)
    torch.npu.synchronize()

    times = []
    for _ in range(REPEATS):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        fn(xyz, npoints)
        torch.npu.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    avg_ms = sum(times) / len(times)
    min_ms = min(times)
    print(f"  {name}: avg={avg_ms:.3f}ms, min={min_ms:.3f}ms")
    return avg_ms


def main():
    fps_kernel = FurthestPointSamplingAscendC(num_cores=8)

    print("=== FPS Performance Benchmark ===")
    print(f"warmup={WARMUP}, repeats={REPEATS}\n")

    for N, npoints in SHAPES:
        xyz = torch.randn(1, N, 3, device='npu:0', dtype=torch.float32)
        print(f"--- N={N}, npoints={npoints} ---")

        py_ms = run_bench("PyTorch", pytorch_fps, xyz, npoints)
        ac_ms = run_bench("AscendC (8-core)", fps_kernel, xyz, npoints)

        print(f"  Speedup: {py_ms / ac_ms:.1f}x\n")

    # Multi-batch test
    B = 16
    N = 1024
    npoints = 512
    xyz = torch.randn(B, N, 3, device='npu:0', dtype=torch.float32)
    print(f"--- Batch B={B}, N={N}, npoints={npoints} ---")
    ac_ms = run_bench("AscendC (8-core)", fps_kernel, xyz, npoints)


if __name__ == "__main__":
    main()
