"""
Debug test for 8-core FPS kernel.
Compares intermediate values between 1-core and 8-core to identify where
the precision mismatch occurs.
"""
import ctypes
import os
import numpy as np
import torch
import torch_npu  # noqa: F401
import pointnet2_ops

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Load debug library ---
lib_path = os.path.join(SCRIPT_DIR, "out", "lib", "libfps_host_mc_debug.so")
if not os.path.exists(lib_path):
    lib_path = os.path.join(SCRIPT_DIR, "libfps_host_mc_debug.so")
if not os.path.exists(lib_path):
    raise FileNotFoundError(f"libfps_host_mc_debug.so not found. Build first.")

lib_dbg = ctypes.CDLL(lib_path)
lib_dbg.fps_run_mc_debug.restype = ctypes.c_int
lib_dbg.fps_run_mc_debug.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int
]

# --- Also load 1-core lib for comparison ---
from fps_wrapper import FurthestPointSamplingAscendC
fps_1c = FurthestPointSamplingAscendC()

# --- Also load 8-core lib ---
from fps_wrapper import FurthestPointSamplingMultiCore
fps_8c = FurthestPointSamplingMultiCore()

NUM_CORES = 8
DEBUG_ITERS = 3
DEBUG_SIZE = NUM_CORES * 4 * DEBUG_ITERS  # 96 floats

N = 1024
NPOINTS = 512

def run_debug():
    # Use same random data for all tests
    torch.manual_seed(42)
    xyz = torch.randn(1, N, 3, device='npu:0', dtype=torch.float32)

    # 1-core result (known correct)
    idx_1c = fps_1c(xyz, NPOINTS)
    print(f"1-core idx[:15]: {idx_1c[0, :15].tolist()}")

    # 8-core result (known buggy)
    idx_8c = fps_8c(xyz, NPOINTS)
    print(f"8-core idx[:15]: {idx_8c[0, :15].tolist()}")

    # PyTorch baseline
    idx_py = pointnet2_ops._furthest_point_sampling(xyz, NPOINTS)
    print(f"PyTorch idx[:15]: {idx_py[0, :15].tolist()}")

    # Run debug kernel
    xyz_t = xyz.permute(0, 2, 1).contiguous().reshape(1, -1)
    idx_dbg = torch.zeros(1, NPOINTS, dtype=torch.int32, device='npu:0')

    debug_host = (ctypes.c_float * DEBUG_SIZE)()

    torch.npu.synchronize()
    ret = lib_dbg.fps_run_mc_debug(
        ctypes.c_void_p(xyz_t[0].data_ptr()),
        ctypes.c_void_p(idx_dbg[0].data_ptr()),
        debug_host,
        DEBUG_SIZE,
    )
    assert ret == 0, f"fps_run_mc_debug failed: {ret}"

    # Copy debug output to device for comparison
    idx_dbg_np = idx_dbg[0].cpu().numpy()
    print(f"\nDebug kernel idx[:15]: {idx_dbg_np[:15].tolist()}")

    # Parse debug buffer
    # Layout: [NUM_CORES * 4 * DEBUG_ITERS]
    # For each (core, iter): [localBestVal, localBestIdx, globalBestVal, globalBestIdx]
    print(f"\n=== Debug: Per-core intermediate values (first {DEBUG_ITERS} iterations) ===")
    vals = np.array([debug_host[i] for i in range(DEBUG_SIZE)], dtype=np.float32)

    for iteration in range(DEBUG_ITERS):
        print(f"\n--- Iteration {iteration + 1} (j={iteration + 1}) ---")
        for core in range(NUM_CORES):
            off = core * 4 * DEBUG_ITERS + iteration * 4
            local_val = vals[off + 0]
            local_idx_raw = vals[off + 1]
            local_idx = int(np.frombuffer(np.array([local_idx_raw], dtype=np.float32).tobytes(), dtype=np.uint32)[0])
            global_val = vals[off + 2]
            global_idx_raw = vals[off + 3]
            global_idx = int(np.frombuffer(np.array([global_idx_raw], dtype=np.float32).tobytes(), dtype=np.uint32)[0])
            print(f"  Core {core}: local(val={local_val:.4f}, idx={local_idx})"
                  f"  global(val={global_val:.4f}, idx={global_idx})")

        # Check: all cores should see the same global result after SyncAll
        global_vals = []
        global_idxs = []
        for core in range(NUM_CORES):
            off = core * 4 * DEBUG_ITERS + iteration * 4
            global_vals.append(vals[off + 2])
            raw = vals[off + 3]
            global_idxs.append(int(np.frombuffer(np.array([raw], dtype=np.float32).tobytes(), dtype=np.uint32)[0]))

        all_same_val = all(v == global_vals[0] for v in global_vals)
        all_same_idx = all(i == global_idxs[0] for i in global_idxs)
        print(f"  Consistency: globalVals same={all_same_val}, globalIdxs same={all_same_idx}")
        if not all_same_val or not all_same_idx:
            print(f"    WARNING: Cores disagree! vals={global_vals}, idxs={global_idxs}")

    # Compare with expected
    print(f"\n=== Expected vs Actual ===")
    expected = idx_py[0].cpu().numpy()
    for i in range(min(15, DEBUG_ITERS)):
        print(f"  j={i}: expected={expected[i]}, debug_kernel={idx_dbg_np[i]}")


if __name__ == "__main__":
    run_debug()
