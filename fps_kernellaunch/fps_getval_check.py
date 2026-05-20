"""
Test: GetValue on GM + DataCopy(GM→UB) readback.
Three sub-tests per core:
  A: GetValue from host-written GM (aclrtMemcpy H2D)
  B: GetValue from kernel-written scratch GM
  C: DataCopy(scratch GM→UB) readback — bypasses GetValue, uses DMA
"""
import ctypes
import os
import numpy as np
import torch
import torch_npu  # noqa: F401

torch.npu.set_device(0)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NUM_CORES = 8
PER_CORE_DEBUG = 24  # 3 sub-tests × 8
DEBUG_SIZE = NUM_CORES * PER_CORE_DEBUG


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
    print("=== GetValue on GM + DataCopy readback ===\n")
    header = (f"{'Core':>4}  {'A:GetVal(host)':>14}  {'expect':>8}"
              f"  {'B:GetVal(scr)':>14}  {'expect':>8}"
              f"  {'C:DCopy(scr)':>13}  {'expect':>8}")
    print(header)
    all_ok = True
    for core in range(NUM_CORES):
        off = core * PER_CORE_DEBUG
        got_a = vals[off]
        exp_a = (core + 1) * 10.0
        got_b = vals[off + 8]
        exp_b = (core + 1) * 100.0
        got_c = vals[off + 16]
        exp_c = (core + 1) * 100.0  # same as B expect
        ok_a = "OK" if abs(got_a - exp_a) < 0.01 else "FAIL"
        ok_b = "OK" if abs(got_b - exp_b) < 0.01 else "FAIL"
        ok_c = "OK" if abs(got_c - exp_c) < 0.01 else "FAIL"
        if ok_a != "OK" or ok_b != "OK" or ok_c != "OK":
            all_ok = False
        print(f"{core:>4}  {got_a:>14.1f}{ok_a:>4}({exp_a:>6.0f})"
              f"  {got_b:>14.1f}{ok_b:>4}({exp_b:>6.0f})"
              f"  {got_c:>13.1f}{ok_c:>4}({exp_c:>6.0f})")

    print(f"\nResult: {'ALL PASS' if all_ok else 'SOME FAILED'}")

    # Diagnosis summary
    print("\n--- Diagnosis ---")
    b_pass = sum(1 for c in range(NUM_CORES) if abs(vals[c * PER_CORE_DEBUG + 8] - (c + 1) * 100.0) < 0.01)
    c_pass = sum(1 for c in range(NUM_CORES) if abs(vals[c * PER_CORE_DEBUG + 16] - (c + 1) * 100.0) < 0.01)
    if c_pass == NUM_CORES and b_pass < NUM_CORES:
        print("DataCopy readback OK but GetValue FAIL → scratch GM data correct, GetValue API is the problem")
    elif c_pass < NUM_CORES:
        print("DataCopy readback also FAIL → scratch GM data is wrong, issue is in DataCopy(UB→GM)")
    else:
        print("Both GetValue and DataCopy readback OK → no issue detected")

    return all_ok


if __name__ == "__main__":
    run_getval_check()
