"""
Ball Query debug compare: run AscendC debug kernel + PyTorch baseline,
print side-by-side intermediate values for query (b=0, m=0).

Usage (on NPU server):
    cd fps_kernellaunch
    bash run.sh -r npu
    python ball_query_debug_compare.py
"""
import ctypes
import os
import sys
import torch
import torch_npu  # noqa

torch.npu.set_device(0)
torch_npu.npu.config.set_compile_mode(jit_compile=False)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from pointnet2_ops import _ball_query

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _get_npu_stream():
    s = torch.npu.current_stream()
    for attr in ['npu_stream', 'stream', '_stream', '_cstream']:
        val = getattr(s, attr, None)
        if callable(val):
            val = val()
        if isinstance(val, int) and val != 0:
            return val
    raise RuntimeError("Cannot get NPU stream pointer")


def load_debug_lib():
    lib_path = os.path.join(SCRIPT_DIR, "out", "lib", "libball_query_debug.so")
    if not os.path.exists(lib_path):
        lib_path = os.path.join(SCRIPT_DIR, "libball_query_debug.so")
    if not os.path.exists(lib_path):
        raise FileNotFoundError(f"{lib_path} not found. Build first: bash run.sh -r npu")
    lib = ctypes.CDLL(lib_path)
    lib.ball_query_debug_run.restype = ctypes.c_int
    lib.ball_query_debug_run.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
        ctypes.c_int32, ctypes.c_float,
        ctypes.c_int32, ctypes.c_void_p,
    ]
    return lib


def run_debug():
    SHAPES = [
        (1, 1024, 512, 32, 0.02),
        (1, 512, 256, 32, 0.04),
        (1, 256, 128, 32, 0.08),
    ]

    lib = load_debug_lib()
    num_cores = 8

    for B, N, M, nsample, radius in SHAPES:
        print(f"\n{'='*70}")
        print(f"Shape: B={B}, N={N}, M={M}, nsample={nsample}, radius={radius}")
        print(f"{'='*70}")

        torch.manual_seed(42)
        xyz = torch.randn(B, N, 3, device='npu:0', dtype=torch.float32)
        new_xyz = torch.randn(B, M, 3, device='npu:0', dtype=torch.float32)

        # --- PyTorch baseline for query (0, 0) ---
        print(f"\n--- PyTorch baseline (query b=0, m=0) ---")
        nx_pt = new_xyz[0, 0, 0].item()
        ny_pt = new_xyz[0, 0, 1].item()
        nz_pt = new_xyz[0, 0, 2].item()
        print(f"  new_xyz = ({nx_pt:.8f}, {ny_pt:.8f}, {nz_pt:.8f})")
        print(f"  radius2 = {radius*radius:.10f}")

        radius2 = radius * radius
        # Compute first 8 distances manually
        print(f"  First 8 squared distances (direct computation):")
        for k in range(8):
            x = xyz[0, k, 0].item()
            y = xyz[0, k, 1].item()
            z = xyz[0, k, 2].item()
            dx, dy, dz = nx_pt - x, ny_pt - y, nz_pt - z
            d2 = dx*dx + dy*dy + dz*dz
            print(f"    k={k}: xyz=({x:.8f}, {y:.8f}, {z:.8f})  d2={d2:.12f}  pass={d2 < radius2}")

        # Also check via torch.cdist (what baseline actually uses)
        dists_cdist = torch.cdist(new_xyz[0:1, 0:1, :], xyz[0:1, :, :], p=2) ** 2
        print(f"  First 8 cdist^2 distances:")
        for k in range(8):
            d2_cdist = dists_cdist[0, 0, k].item()
            print(f"    k={k}: cdist^2={d2_cdist:.12f}  pass={d2_cdist < radius2}")

        # Run full baseline
        pt_out = _ball_query(new_xyz, xyz, radius, nsample)
        print(f"  Baseline idx[0,0,:8] = {pt_out[0, 0, :8].tolist()}")

        # Count how many points baseline found for this query
        baseline_mask = dists_cdist[0, 0] < radius2
        baseline_cnt = baseline_mask.sum().item()
        print(f"  Baseline cnt (points within radius) = {baseline_cnt}")

        # --- AscendC debug kernel ---
        idx_ac = torch.zeros(B, M, nsample, dtype=torch.int32, device='npu:0')
        stream_ptr = _get_npu_stream()

        print(f"\n--- AscendC debug kernel ---")
        ret = lib.ball_query_debug_run(
            ctypes.c_void_p(xyz.data_ptr()),
            ctypes.c_void_p(new_xyz.data_ptr()),
            ctypes.c_void_p(idx_ac.data_ptr()),
            ctypes.c_int32(B), ctypes.c_int32(N), ctypes.c_int32(M),
            ctypes.c_int32(nsample), ctypes.c_float(radius),
            ctypes.c_int32(num_cores), ctypes.c_void_p(stream_ptr),
        )
        print(f"  Kernel return: {ret}")
        if ret != 0:
            print(f"  ERROR: kernel failed!")
            continue

        print(f"  AscendC idx[0,0,:8] = {idx_ac[0, 0, :8].tolist()}")

        # Compare outputs
        diff = (pt_out - idx_ac.to(torch.int64)).abs()
        max_diff = diff.max().item()
        print(f"\n  max_diff = {max_diff}")

        # Check if the AscendC kernel's stderr diagnostics appeared
        # (they come from the host wrapper's fprintf(stderr, ...))


if __name__ == "__main__":
    run_debug()
