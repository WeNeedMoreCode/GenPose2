"""
Minimal multi-core Duplicate test.
Each core does Duplicate(dist, 1e10) then SetValue(debugGm, val).
Host reads first 8 floats from debug buffer.
"""
import ctypes
import os
import numpy as np
import torch
import torch_npu  # noqa: F401
import pointnet2_ops

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NUM_CORES = 8
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


def run_minimal_test():
    # Need dummy tensors for xyz and idx (kernel doesn't use them but host passes pointers)
    xyz = torch.randn(1, 3, N, device='npu:0', dtype=torch.float32)
    xyz_t = xyz.reshape(1, -1)
    idx_dummy = torch.zeros(1, NPOINTS, dtype=torch.int32, device='npu:0')

    try:
        lib = _load_lib("libfps_host_mc_v3_dbg.so", "fps_run_mc_v3_dbg")
        debug_host = (ctypes.c_float * (NUM_CORES * 8 * 3))()  # allocate same as host expects
        torch.npu.synchronize()
        ret = lib.fps_run_mc_v3_dbg(
            ctypes.c_void_p(xyz_t[0].data_ptr()),
            ctypes.c_void_p(idx_dummy[0].data_ptr()),
            debug_host,
            NUM_CORES * 8 * 3,
        )
        assert ret == 0

        print("=== Minimal Duplicate Test (v2: alive marker) ===")
        print("Each core: SetValue(alive=-1) → Duplicate(dist, 1e10) → GetValue → SetValue")
        print()
        any_alive_no_dup = False
        all_ok = True
        for c in range(NUM_CORES):
            alive = debug_host[c]
            dup_val = debug_host[NUM_CORES + c]
            alive_ok = alive == -1.0
            dup_ok = dup_val > 1e9
            if not dup_ok:
                all_ok = False

            alive_str = "ALIVE" if alive_ok else "DEAD"
            dup_str = "OK" if dup_ok else "FAIL"

            if alive_ok and not dup_ok:
                any_alive_no_dup = True

            print(f"  Core {c}: alive={alive:.1f}({alive_str})  dup={dup_val:.1f}({dup_str})")

        print()
        if all_ok:
            print("All cores OK — basic Duplicate works, problem is elsewhere")
        elif any_alive_no_dup:
            print("Core ran (alive=-1) but Duplicate result wrong → scalar/vector sync issue")
        else:
            print("Core didn't run (alive=0) → SPMD core dispatch or SetValue issue")

    except Exception as e:
        print(f"Test failed: {e}")


if __name__ == "__main__":
    run_minimal_test()
