"""
Test: does GetValue on GlobalTensor work on all 8 cores?
Sub-test A: GetValue from host-written GM (aclrtMemcpy H2D → GetValue)
Sub-test B: DataCopy(UB→scratch GM) → GetValue from scratch GM
Results written via Duplicate(count=8)+DataCopy (proven reliable).
"""
import ctypes
import os
import numpy as np
import torch
import torch_npu  # noqa: F401

torch.npu.set_device(0)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NUM_CORES = 8
DEBUG_SIZE = NUM_CORES * 16  # 8 per sub-test × 2


def run_getval_check():
    lib_path = os.path.join(SCRIPT_DIR, "out", "lib", "libfps_host_getval_check.so")
    if not os.path.exists(lib_path):
        raise FileNotFoundError(f"{lib_path} not found. Build first.")
    lib = ctypes.CDLL(lib_path)
    lib.fps_run_getval_check.restype = ctypes.c_int
    lib.fps_run_getval_check.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int]

    debug_host = (ctypes.c_float * DEBUG_SIZE)()
    ret = lib.fps_run_getval_check(debug_host, DEBUG_SIZE)
    assert ret == 0, f"Kernel returned {ret}"

    vals = np.array([debug_host[i] for i in range(DEBUG_SIZE)], dtype=np.float32)
    print("=== GetValue on GlobalTensor ===\n")
    print(f"{'Core':>4}  {'A: hostGM':>12}  {'expect':>12}  {'B: scratch':>12}  {'expect':>12}  {'A':>4}  {'B':>4}")
    all_ok = True
    for core in range(NUM_CORES):
        off = core * 16
        got_a = vals[off]          # GetValue from host-written GM
        exp_a = (core + 1) * 10.0
        got_b = vals[off + 8]      # GetValue from kernel-written scratch GM
        exp_b = (core + 1) * 100.0
        ok_a = "OK" if abs(got_a - exp_a) < 0.01 else "FAIL"
        ok_b = "OK" if abs(got_b - exp_b) < 0.01 else "FAIL"
        if ok_a != "OK" or ok_b != "OK":
            all_ok = False
        print(f"{core:>4}  {got_a:>12.1f}  {exp_a:>12.1f}  {got_b:>12.1f}  {exp_b:>12.1f}  {ok_a:>4}  {ok_b:>4}")

    print(f"\nResult: {'ALL PASS' if all_ok else 'SOME FAILED'}")
    return all_ok


if __name__ == "__main__":
    run_getval_check()
