"""
Minimal test: verify 8-core aclrtlaunch dispatch and SetValue.
Each core should write [1e10, 3.14, -1.0, 42.0] to its own GM offset.
"""
import ctypes
import os
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NUM_CORES = 8
FIELDS_PER_CORE = 4
DEBUG_SIZE = NUM_CORES * FIELDS_PER_CORE

EXPECTED = [1e10, 3.14, -1.0, 42.0]


def run_minimal():
    lib_path = os.path.join(SCRIPT_DIR, "out", "lib", "libfps_host_minimal_test.so")
    if not os.path.exists(lib_path):
        raise FileNotFoundError(f"{lib_path} not found. Build first.")
    lib = ctypes.CDLL(lib_path)
    lib.fps_run_minimal_test.restype = ctypes.c_int
    lib.fps_run_minimal_test.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.c_int
    ]

    debug_host = (ctypes.c_float * DEBUG_SIZE)()
    ret = lib.fps_run_minimal_test(debug_host, DEBUG_SIZE)
    assert ret == 0, f"Kernel returned {ret}"

    vals = np.array([debug_host[i] for i in range(DEBUG_SIZE)], dtype=np.float32)
    all_ok = True
    for core in range(NUM_CORES):
        off = core * FIELDS_PER_CORE
        actual = vals[off:off + FIELDS_PER_CORE].tolist()
        match = all(np.isclose(a, e, rtol=1e-3) for a, e in zip(actual, EXPECTED))
        status = "OK" if match else "FAIL"
        if not match:
            all_ok = False
        print(f"  Core {core}: {actual}  ({status})")

    print(f"\nResult: {'ALL PASS' if all_ok else 'SOME FAILED'}")
    return all_ok


if __name__ == "__main__":
    run_minimal()
