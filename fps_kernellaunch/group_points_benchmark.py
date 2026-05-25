"""
GroupPoints Performance Benchmark: PyTorch vs AscendC kernel.

Tests shapes matching PointNet2 SA layers (with DINOv2 fusion, C=390 for SA0).

Usage:
    cd GenPose2/fps_kernellaunch
    bash run.sh -r npu
    python group_points_benchmark.py
"""

import time
import torch
import torch_npu  # noqa: F401

torch.npu.set_device(0)

from group_points_ascendc import GroupPointsAscendC

WARMUP = 5
REPEATS = 50

# Shapes matching PointNet2 SA layers (ClsMSG_CFG_Light with input_channels=387+3)
SHAPES = [
    # (B, C, N, npoint, nsample)
    (1, 390, 1024, 512, 32),   # SA0 (largest)
    (1, 131, 512, 256, 32),    # SA1
    (1, 387, 256, 128, 32),    # SA2
]


def pytorch_group_points(points, idx):
    """PyTorch expand+gather implementation (pointnet2_ops._group_points)."""
    B, C, N = points.shape
    M = idx.shape[1]
    nsample = idx.shape[2]
    idx = idx.unsqueeze(1).expand(-1, C, -1, -1)
    points = points.unsqueeze(2).expand(-1, -1, M, -1)
    out = torch.gather(points, dim=-1, index=idx)
    return out


def run_bench(name, fn, points, idx):
    for _ in range(WARMUP):
        fn(points, idx)
    torch.npu.synchronize()

    times = []
    for _ in range(REPEATS):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        fn(points, idx)
        torch.npu.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    avg_ms = sum(times) / len(times)
    min_ms = min(times)
    print(f"  {name}: avg={avg_ms:.3f}ms, min={min_ms:.3f}ms")
    return avg_ms


def correctness_test():
    """Verify AscendC kernel output matches PyTorch."""
    print("=== Correctness Test ===\n")
    gp_kernel = GroupPointsAscendC(num_cores=8)

    for B, C, N, npoint, nsample in SHAPES:
        points = torch.randn(B, C, N, device='npu:0', dtype=torch.float32)
        idx = torch.randint(0, N, (B, npoint, nsample), device='npu:0', dtype=torch.int32)

        pt_out = pytorch_group_points(points, idx)
        ac_out = gp_kernel(points, idx)

        match = torch.allclose(pt_out, ac_out, atol=1e-5)
        max_diff = (pt_out - ac_out).abs().max().item()
        status = "PASS" if match else "FAIL"
        print(f"  B={B}, C={C}, N={N}, npoint={npoint}, nsample={nsample}: {status} (max_diff={max_diff:.2e})")
    print()


def main():
    gp_kernel = GroupPointsAscendC(num_cores=8)

    correctness_test()

    print("=== GroupPoints Performance Benchmark ===")
    print(f"warmup={WARMUP}, repeats={REPEATS}\n")

    for B, C, N, npoint, nsample in SHAPES:
        points = torch.randn(B, C, N, device='npu:0', dtype=torch.float32)
        idx = torch.randint(0, N, (B, npoint, nsample), device='npu:0', dtype=torch.int32)

        print(f"--- B={B}, C={C}, N={N}, npoint={npoint}, nsample={nsample} ---")
        py_ms = run_bench("PyTorch", pytorch_group_points, points, idx)
        ac_ms = run_bench("AscendC (8-core)", gp_kernel, points, idx)
        print(f"  Speedup: {py_ms / ac_ms:.1f}x\n")

    # Multi-batch test
    B = 16
    C, N, npoint, nsample = 390, 1024, 512, 32
    points = torch.randn(B, C, N, device='npu:0', dtype=torch.float32)
    idx = torch.randint(0, N, (B, npoint, nsample), device='npu:0', dtype=torch.int32)
    print(f"--- Batch B={B}, C={C}, N={N}, npoint={npoint}, nsample={nsample} ---")
    py_ms = run_bench("PyTorch", pytorch_group_points, points, idx)
    ac_ms = run_bench("AscendC (8-core)", gp_kernel, points, idx)
    print(f"  Speedup: {py_ms / ac_ms:.1f}x\n")


if __name__ == "__main__":
    main()
