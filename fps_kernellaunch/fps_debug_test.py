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
DEBUG_SIZE = NUM_CORES * 4 * DEBUG_ITERS
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


def parse_debug(debug_host):
    vals = np.array([debug_host[i] for i in range(DEBUG_SIZE)], dtype=np.float32)
    results = []
    for iteration in range(DEBUG_ITERS):
        iter_data = []
        for core in range(NUM_CORES):
            off = core * 4 * DEBUG_ITERS + iteration * 4
            local_val = vals[off + 0]
            local_idx = int(np.frombuffer(
                np.array([vals[off + 1]], dtype=np.float32).tobytes(), dtype=np.uint32)[0])
            global_val = vals[off + 2]
            global_idx = int(np.frombuffer(
                np.array([vals[off + 3]], dtype=np.float32).tobytes(), dtype=np.uint32)[0])
            iter_data.append({
                'local_val': local_val, 'local_idx': local_idx,
                'global_val': global_val, 'global_idx': global_idx,
            })
        results.append(iter_data)
    return results


def print_debug(label, debug_data):
    print(f"\n=== {label}: Per-core intermediate values ===")
    for it, iter_data in enumerate(debug_data):
        print(f"\n--- Iteration {it + 1} ---")
        for core, d in enumerate(iter_data):
            print(f"  Core {core}: local(val={d['local_val']:.4f}, idx={d['local_idx']})"
                  f"  global(val={d['global_val']:.4f}, idx={d['global_idx']})")

        global_vals = [d['global_val'] for d in iter_data]
        global_idxs = [d['global_idx'] for d in iter_data]
        same_val = all(v == global_vals[0] for v in global_vals)
        same_idx = all(i == global_idxs[0] for i in global_idxs)
        if same_val and same_idx:
            print(f"  All cores AGREE on global: val={global_vals[0]:.4f}, idx={global_idxs[0]}")
        else:
            print(f"  WARNING: Cores DISAGREE!")
            print(f"    vals: {[f'{v:.4f}' for v in global_vals]}")
            print(f"    idxs: {global_idxs}")


def run_one_debug(lib, func_name, xyz_t, label):
    idx_dbg = torch.zeros(1, NPOINTS, dtype=torch.int32, device='npu:0')
    debug_host = (ctypes.c_float * DEBUG_SIZE)()
    torch.npu.synchronize()
    ret = lib[func_name](
        ctypes.c_void_p(xyz_t[0].data_ptr()),
        ctypes.c_void_p(idx_dbg[0].data_ptr()),
        debug_host,
        DEBUG_SIZE,
    )
    assert ret == 0, f"{func_name} failed: {ret}"
    debug_data = parse_debug(debug_host)
    idx_np = idx_dbg[0].cpu().numpy()
    print_debug(label, debug_data)
    return idx_np


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

    results = {}

    # v1 debug (SetValue)
    try:
        lib_v1 = _load_lib("libfps_host_mc_debug.so", "fps_run_mc_debug")
        results['v1'] = run_one_debug(lib_v1, "fps_run_mc_debug", xyz_t, "v1 (SetValue)")
    except Exception as e:
        print(f"v1 debug: SKIP ({e})")

    # v2 debug (DataCopy + SetValue on UB)
    try:
        lib_v2 = _load_lib("libfps_host_mc_v2_dbg.so", "fps_run_mc_v2_dbg")
        results['v2'] = run_one_debug(lib_v2, "fps_run_mc_v2_dbg", xyz_t, "v2 (DataCopy+SetValueUB)")
    except Exception as e:
        print(f"v2 debug: SKIP ({e})")

    # v3 debug (pure vector Duplicate + DataCopy)
    try:
        lib_v3 = _load_lib("libfps_host_mc_v3_dbg.so", "fps_run_mc_v3_dbg")
        results['v3'] = run_one_debug(lib_v3, "fps_run_mc_v3_dbg", xyz_t, "v3 (Duplicate+DataCopy)")
    except Exception as e:
        print(f"v3 debug: SKIP ({e})")

    # Final comparison
    expected = idx_py[0].cpu().numpy()
    print(f"\n=== Final Comparison ===")
    print(f"  PyTorch: {expected[:15].tolist()}")
    print(f"  1-core:  {idx_1c[0, :15].tolist()}")
    for name, idx in results.items():
        match = "MATCH" if np.array_equal(expected, idx) else "MISMATCH"
        print(f"  {name}:     {idx[:15].tolist()}  ({match})")


if __name__ == "__main__":
    run_debug()
