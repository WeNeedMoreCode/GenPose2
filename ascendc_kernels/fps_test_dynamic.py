"""
Test dynamic-shape FPS kernel: verify all three SA layer shapes against PyTorch.

Usage:
    bash run.sh -r npu
    export LD_LIBRARY_PATH=$(pwd)/out/lib:$LD_LIBRARY_PATH
    python fps_test_dynamic.py
"""
import ctypes
import os
import numpy as np
import torch
import torch_npu  # noqa: F401
import pointnet2_ops

torch.npu.set_device(0)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NUM_CORES = 8

# Three SA layer shapes: (input_N, output_npoints)
SHAPES = [
    (1024, 512),
    (512, 256),
    (256, 128),
]


def _load_lib():
    lib_path = os.path.join(SCRIPT_DIR, "out", "lib", "libfps_host_dynamic.so")
    if not os.path.exists(lib_path):
        lib_path = os.path.join(SCRIPT_DIR, "libfps_host_dynamic.so")
    if not os.path.exists(lib_path):
        raise FileNotFoundError(f"libfps_host_dynamic.so not found. Build first.")
    lib = ctypes.CDLL(lib_path)
    lib.fps_run_dynamic.restype = ctypes.c_int
    lib.fps_run_dynamic.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
    ]
    return lib


def run_dynamic_fps(lib, xyz, total_n, npoints):
    """Run dynamic FPS kernel. xyz is [1, N, 3] on NPU."""
    xyz_t = xyz.permute(0, 2, 1).contiguous().reshape(-1)
    idx = torch.zeros(npoints, device='npu:0', dtype=torch.int32)
    torch.npu.synchronize()

    ret = lib.fps_run_dynamic(
        ctypes.c_void_p(xyz_t.data_ptr()),
        ctypes.c_void_p(idx.data_ptr()),
        total_n, npoints, NUM_CORES,
    )
    assert ret == 0, f"fps_run_dynamic failed: ret={ret}"
    return idx.cpu().numpy().astype(np.int64)


def run_pytorch_fps(xyz, npoints):
    return pointnet2_ops._furthest_point_sampling(xyz, npoints)[0].cpu().numpy().astype(np.int64)


def main():
    lib = _load_lib()

    all_pass = True

    for total_n, npoints in SHAPES:
        print(f"\n{'='*60}")
        print(f"  Shape: {total_n} → {npoints}")
        print(f"{'='*60}")

        seeds = [42, 0, 7, 123, 999]
        shape_pass = True

        for seed in seeds:
            torch.manual_seed(seed)
            xyz = torch.randn(1, total_n, 3, device='npu:0', dtype=torch.float32)

            py_idx = run_pytorch_fps(xyz, npoints)
            dyn_idx = run_dynamic_fps(lib, xyz, total_n, npoints)

            match = np.array_equal(py_idx, dyn_idx)
            unique = len(np.unique(dyn_idx))
            valid = dyn_idx.min() >= 0 and dyn_idx.max() < total_n

            status = "MATCH" if match else f"FAIL ({int(np.sum(py_idx != dyn_idx))} diff)"
            if not match:
                shape_pass = False
                all_pass = False

            print(f"    seed={seed:<6} {status:<20} unique={unique}/{npoints}  "
                  f"range=[{dyn_idx.min()},{dyn_idx.max()}] {'OK' if valid else 'INVALID'}")

            if not match:
                i = int(np.argmax(py_idx != dyn_idx))
                print(f"      first diff at pos {i}: py={py_idx[i]} got={dyn_idx[i]}")

        tag = "PASS" if shape_pass else "FAIL"
        print(f"  → {total_n}→{npoints}: {tag}")

    print(f"\n{'='*60}")
    print(f"  Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
