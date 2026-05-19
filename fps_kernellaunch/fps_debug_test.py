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

        print("=== Minimal Duplicate Test ===")
        print("Each core: Duplicate(dist, 1e10, 128) → GetValue(0) → SetValue(debugGm)")
        all_ok = True
        for c in range(NUM_CORES):
            val = debug_host[c]
            status = "OK" if val > 1e9 else "FAIL"
            if status == "FAIL":
                all_ok = False
            print(f"  Core {c}: {val:.1f} ({status})")

        if all_ok:
            print("\nAll cores OK — Duplicate works in isolation, problem is buffer interaction")
        else:
            print("\nSome cores FAIL — AscendC SPMD Duplicate has fundamental issue")

    except Exception as e:
        print(f"Test failed: {e}")


if __name__ == "__main__":
    run_minimal_test()
