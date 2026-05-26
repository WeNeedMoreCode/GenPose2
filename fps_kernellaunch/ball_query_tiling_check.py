"""
Tiling readback check: verify kernel reads correct tiling data.
Run 3 shapes sequentially to detect cross-call buffer race.

Usage:
    cd fps_kernellaunch
    bash run.sh -r npu
    python ball_query_tiling_check.py
"""
import ctypes
import os
import torch
import torch_npu

torch.npu.set_device(0)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _get_npu_stream():
    s = torch.npu.current_stream()
    for attr in ['npu_stream', 'stream', '_stream', '_cstream']:
        val = getattr(s, attr, None)
        if callable(val):
            val = val()
        if isinstance(val, int) and val != 0:
            return val
    raise RuntimeError("Cannot get NPU stream pointer")


def run_check():
    lib_path = os.path.join(SCRIPT_DIR, "out", "lib", "libball_query_tiling_check.so")
    if not os.path.exists(lib_path):
        lib_path = os.path.join(SCRIPT_DIR, "libball_query_tiling_check.so")
    if not os.path.exists(lib_path):
        raise FileNotFoundError(f"{lib_path} not found. Build first: bash run.sh -r npu")
    lib = ctypes.CDLL(lib_path)
    lib.ball_query_tiling_check_run.restype = ctypes.c_int
    lib.ball_query_tiling_check_run.argtypes = [
        ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
        ctypes.c_int32, ctypes.c_float,
        ctypes.c_int32, ctypes.c_void_p,
    ]

    stream_ptr = _get_npu_stream()

    SHAPES = [
        (1, 1024, 512, 32, 0.02),
        (1, 512, 256, 32, 0.04),
        (1, 256, 128, 32, 0.08),
    ]

    all_ok = True
    for B, N, M, nsample, radius in SHAPES:
        ret = lib.ball_query_tiling_check_run(
            ctypes.c_int32(B), ctypes.c_int32(N), ctypes.c_int32(M),
            ctypes.c_int32(nsample), ctypes.c_float(radius),
            ctypes.c_int32(8), ctypes.c_void_p(stream_ptr),
        )
        if (ret != 0):
            all_ok = False
            print(f"[CHECK] Shape (N={N}) FAILED with ret={ret}")
        else:
            print(f"[CHECK] Shape (N={N}) PASSED")

    print(f"\n{'='*60}")
    print(f"Overall: {'ALL PASS' if all_ok else 'SOME FAILED'}")
    print(f"{'='*60}")
    return all_ok


if __name__ == "__main__":
    run_check()
