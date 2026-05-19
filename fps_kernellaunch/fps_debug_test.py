"""
Debug test for 8-core FPS kernels (v1, v2, v3).
Compares intermediate values to identify where precision mismatch occurs.
"""
import ctypes
import os
import numpy as np
import torch
import torch_npu  # noqa: F401
import pointnet2_ops

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NUM_CORES = 8
DEBUG_ITERS = 3
DIAG_FIELDS = 6
DEBUG_SIZE = NUM_CORES * DIAG_FIELDS * DEBUG_ITERS
N = 1024
NPOINTS = 512


def _load_lib(name, func_name):
    lib_path = os.path.join(SCRIPT_DIR, "out", "lib", name)
    if not os.path.exists(lib_path):
        lib_path = os.path.join(SCRIPT_DIR, name)
    if not os.path.exists(lib_path):
        raise FileNotFoundError(f"{name} not found. Build first.")
    lib = ctypes.CDLL(lib_path)
    lib[func_name].restype = ctypes.c_int
    lib[func_name].argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int
    ]
    return lib


def float_to_idx(f):
    return int(np.frombuffer(np.array([f], dtype=np.float32).tobytes(), dtype=np.uint32)[0])


def parse_debug_v3(debug_host):
    """Parse v3 6-field diagnostic: dup_ok, cksum, reduceVal, reduceIdx, localBestVal, localBestIdx"""
    vals = np.array([debug_host[i] for i in range(DEBUG_SIZE)], dtype=np.float32)
    results = []
    for it in range(DEBUG_ITERS):
        iter_data = []
        for core in range(NUM_CORES):
            off = core * DIAG_FIELDS * DEBUG_ITERS + it * DIAG_FIELDS
            iter_data.append({
                'dist0': vals[off + 0],
                'cksum': vals[off + 1],
                'reduce_val': vals[off + 2],
                'reduce_idx': float_to_idx(vals[off + 3]),
                'local_val': vals[off + 4],
                'local_idx': float_to_idx(vals[off + 5]),
            })
        results.append(iter_data)
    return results


def print_diag_v3(debug_data):
    for it, iter_data in enumerate(debug_data):
        print(f"\n--- Iteration {it + 1} ---")
        for core, d in enumerate(iter_data):
            dup_ok = "OK" if d['dist0'] > 1e9 or d['dist0'] > 0 else "ZERO!"
            ck_ok = "OK" if d['cksum'] > 0 else "ZERO!"
            print(f"  Core {core}: dist0={d['dist0']:.4f}({dup_ok})"
                  f"  cksum={d['cksum']:.4f}({ck_ok})"
                  f"  reduce(val={d['reduce_val']:.4f},idx={d['reduce_idx']})"
                  f"  local(val={d['local_val']:.4f},idx={d['local_idx']})")


def parse_debug_v12(debug_host):
    """Parse v1/v2 4-field format: localVal, localIdx, globalVal, globalIdx"""
    vals = np.array([debug_host[i] for i in range(NUM_CORES * 4 * DEBUG_ITERS)], dtype=np.float32)
    results = []
    for it in range(DEBUG_ITERS):
        iter_data = []
        for core in range(NUM_CORES):
            off = core * 4 * DEBUG_ITERS + it * 4
            iter_data.append({
                'local_val': vals[off + 0],
                'local_idx': float_to_idx(vals[off + 1]),
                'global_val': vals[off + 2],
                'global_idx': float_to_idx(vals[off + 3]),
            })
        results.append(iter_data)
    return results


def print_debug_v12(label, debug_data):
    print(f"\n=== {label} ===")
    for it, iter_data in enumerate(debug_data):
        print(f"\n--- Iteration {it + 1} ---")
        for core, d in enumerate(iter_data):
            print(f"  Core {core}: local(val={d['local_val']:.4f},idx={d['local_idx']})"
                  f"  global(val={d['global_val']:.4f},idx={d['global_idx']})")
        global_vals = [d['global_val'] for d in iter_data]
        same = all(v == global_vals[0] for v in global_vals)
        print(f"  Cores {'AGREE' if same else 'DISAGREE'} on global: val={global_vals[0]:.4f}")


def run_debug():
    torch.manual_seed(42)
    xyz = torch.randn(1, N, 3, device='npu:0', dtype=torch.float32)

    from fps_wrapper import FurthestPointSamplingAscendC
    fps_1c = FurthestPointSamplingAscendC()
    idx_1c = fps_1c(xyz, NPOINTS)
    idx_py = pointnet2_ops._furthest_point_sampling(xyz, NPOINTS)

    print(f"PyTorch  idx[:15]: {idx_py[0, :15].tolist()}")
    print(f"1-core   idx[:15]: {idx_1c[0, :15].tolist()}")

    xyz_t = xyz.permute(0, 2, 1).contiguous().reshape(1, -1)

    # v3 diag (6-field)
    try:
        lib_v3 = _load_lib("libfps_host_mc_v3_dbg.so", "fps_run_mc_v3_dbg")
        idx_dbg = torch.zeros(1, NPOINTS, dtype=torch.int32, device='npu:0')
        debug_host = (ctypes.c_float * DEBUG_SIZE)()
        torch.npu.synchronize()
        ret = lib_v3.fps_run_mc_v3_dbg(
            ctypes.c_void_p(xyz_t[0].data_ptr()),
            ctypes.c_void_p(idx_dbg[0].data_ptr()),
            debug_host,
            DEBUG_SIZE,
        )
        assert ret == 0
        diag_data = parse_debug_v3(debug_host)
        print(f"\n=== v3 Diagnostics ===")
        print_diag_v3(diag_data)
        idx_v3 = idx_dbg[0].cpu().numpy()
        match = "MATCH" if np.array_equal(idx_py[0].cpu().numpy(), idx_v3) else "MISMATCH"
        print(f"\n  v3 idx[:15]: {idx_v3[:15].tolist()}  ({match})")
    except Exception as e:
        print(f"v3 diag: SKIP ({e})")


if __name__ == "__main__":
    run_debug()
