"""
Multi-core alive check: each core does Duplicate + DataCopy.
Core 0 -> 100.0, Core 1 -> 200.0, ..., Core 7 -> 800.0
"""
import ctypes
import os
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NUM_CORES = 8
PER_CORE = 8
DEBUG_SIZE = NUM_CORES * PER_CORE


def run_alive_check():
    lib_path = os.path.join(SCRIPT_DIR, "out", "lib", "libfps_host_alive_check.so")
    if not os.path.exists(lib_path):
        raise FileNotFoundError(f"{lib_path} not found. Build first.")
    lib = ctypes.CDLL(lib_path)
    lib.fps_run_alive_check.restype = ctypes.c_int
    lib.fps_run_alive_check.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.c_int
    ]

    debug_host = (ctypes.c_float * DEBUG_SIZE)()
    ret = lib.fps_run_alive_check(debug_host, DEBUG_SIZE)
    assert ret == 0, f"Kernel returned {ret}"

    vals = np.array([debug_host[i] for i in range(DEBUG_SIZE)], dtype=np.float32)
    all_ok = True
    for core in range(NUM_CORES):
        off = core * PER_CORE
        expected = (core + 1) * 100.0
        actual = vals[off:off + PER_CORE]
        match = all(np.isclose(a, expected, rtol=1e-3) for a in actual)
        status = "OK" if match else "FAIL"
        if not match:
            all_ok = False
        print(f"  Core {core}: {actual.tolist()[:4]}...  expected={expected:.0f}  ({status})")

    print(f"\nResult: {'ALL PASS' if all_ok else 'SOME FAILED'}")
    return all_ok


if __name__ == "__main__":
    run_alive_check()
