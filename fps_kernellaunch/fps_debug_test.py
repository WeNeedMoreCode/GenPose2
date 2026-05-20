"""
Debug test for v3 multi-core FPS kernel (overlap-fixed, Duplicate+DataCopy version).
Compares 10-field diagnostics across cores and validates against PyTorch.
"""
import ctypes
import os
import numpy as np
import torch
import torch_npu  # noqa: F401
import pointnet2_ops

torch.npu.set_device(0)  # activate device context

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NUM_CORES = 8
ACTUAL_FIELDS = 10
DIAG_FIELDS = 16  # stride (DataCopy alignment)
DEBUG_ITERS = 3
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
    """Parse v3 10-field diagnostic (stride 16)."""
    vals = np.array([debug_host[i] for i in range(DEBUG_SIZE)], dtype=np.float32)
    results = []
    for it in range(DEBUG_ITERS):
        iter_data = []
        for core in range(NUM_CORES):
            off = core * DIAG_FIELDS * DEBUG_ITERS + it * DIAG_FIELDS
            iter_data.append({
                'dist_init': vals[off + 0],
                'dist_post': vals[off + 1],
                'cksum': vals[off + 2],
                'red0_val': vals[off + 3],
                'red0_idx': float_to_idx(vals[off + 4]),
                'red1_val': vals[off + 5],
                'red1_idx': float_to_idx(vals[off + 6]),
                'local_val': vals[off + 7],
                'global_val': vals[off + 8],
                'global_idx': float_to_idx(vals[off + 9]),
            })
        results.append(iter_data)
    return results


def print_diag_v3(debug_data):
    for it, iter_data in enumerate(debug_data):
        print(f"\n--- Iteration {it + 1} ---")
        for core, d in enumerate(iter_data):
            init_ok = "OK" if d['dist_init'] > 1e9 else "FAIL"
            post_ok = "OK" if d['dist_post'] > 0 or d['cksum'] > 0 else "ZERO"
            local_ok = "OK" if d['local_val'] > 0 else "ZERO"
            global_ok = "OK" if d['global_val'] > 0 else "ZERO"
            print(f"  Core {core}: init={d['dist_init']:.1f}({init_ok})"
                  f"  post={d['dist_post']:.4f}({post_ok})"
                  f"  cksum={d['cksum']:.4f}"
                  f"  red0(v={d['red0_val']:.4f},i={d['red0_idx']})"
                  f"  red1(v={d['red1_val']:.4f},i={d['red1_idx']})"
                  f"  local={d['local_val']:.4f}({local_ok})"
                  f"  global(v={d['global_val']:.4f},i={d['global_idx']})({global_ok})")


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
        print(f"\n=== v3 Diagnostics (overlap-fixed, Duplicate+DataCopy) ===")
        print_diag_v3(diag_data)
        idx_v3 = idx_dbg[0].cpu().numpy()
        match = "MATCH" if np.array_equal(idx_py[0].cpu().numpy(), idx_v3) else "MISMATCH"
        print(f"\n  v3 idx[:15]: {idx_v3[:15].tolist()}  ({match})")
    except Exception as e:
        print(f"v3 diag: SKIP ({e})")


if __name__ == "__main__":
    run_debug()
