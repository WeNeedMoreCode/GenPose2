"""
Debug test for v3 multi-core FPS kernel.
Computes ground truth by simulating kernel behavior in Python, then compares
field-by-field with actual kernel diagnostic output.
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
ACTUAL_FIELDS = 10
DIAG_FIELDS = 16
DEBUG_ITERS = 3
DEBUG_SIZE = NUM_CORES * DIAG_FIELDS * DEBUG_ITERS
N = 1024
NPOINTS = 512
CHUNK = N // NUM_CORES
BLOCK_SIZE = 64


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
                'local_idx': float_to_idx(vals[off + 9]),  # reuse global_idx for local
                'global_val': vals[off + 8],
                'global_idx': float_to_idx(vals[off + 9]),
            })
        results.append(iter_data)
    return results


def compute_ground_truth(xyz_flat):
    """
    Simulate multi-core FPS kernel behavior for first DEBUG_ITERS iterations.
    xyz_flat: [3*N] float32, transposed layout (x[0:N], y[N:2N], z[2N:3N]).
    Mirrors kernel exactly, including dist[0]/dist[1] reset after each iter.
    """
    dists = [np.full(CHUNK, 1e10, dtype=np.float32) for _ in range(NUM_CORES)]
    old = 0
    all_iters = []

    for j in range(1, DEBUG_ITERS + 1):
        x1, y1, z1 = xyz_flat[old], xyz_flat[N + old], xyz_flat[2 * N + old]

        iter_cores = []
        for core in range(NUM_CORES):
            d = dists[core]
            off = core * CHUNK

            # Distance update: d[i] = min(d[i], ||point_i - ref||^2)
            for i in range(CHUNK):
                gi = off + i
                dx = xyz_flat[gi] - x1
                dy = xyz_flat[N + gi] - y1
                dz = xyz_flat[2 * N + gi] - z1
                d[i] = min(float(d[i]), float(dx * dx + dy * dy + dz * dz))

            dist_post = float(d[0])
            cksum = float(d[0] + d[1] + d[2] + d[3])

            # Block-level argmax (simulating WholeReduceMax, 2 blocks of 64)
            b0 = d[:BLOCK_SIZE]
            b1 = d[BLOCK_SIZE:]
            red0_val, red0_idx = float(np.max(b0)), int(np.argmax(b0))
            red1_val, red1_idx = float(np.max(b1)), int(np.argmax(b1))

            # Local argmax
            local_val = float(np.max(d))
            local_idx = off + int(np.argmax(d))

            iter_cores.append({
                'dist_init': 1e10 if j == 1 else dist_post,
                'dist_post': dist_post,
                'cksum': cksum,
                'red0_val': red0_val, 'red0_idx': red0_idx,
                'red1_val': red1_val, 'red1_idx': red1_idx,
                'local_val': local_val, 'local_idx': local_idx,
            })

        # Cross-core argmax
        best = max(range(NUM_CORES), key=lambda c: iter_cores[c]['local_val'])
        gval = iter_cores[best]['local_val']
        gidx = iter_cores[best]['local_idx']
        for c in range(NUM_CORES):
            iter_cores[c]['global_val'] = gval
            iter_cores[c]['global_idx'] = gidx

        all_iters.append(iter_cores)

        # Reset dist[0] and dist[1] per core (kernel behavior after communication)
        for c in range(NUM_CORES):
            dists[c][0] = 1e10
            dists[c][1] = 1e10

        old = gidx

    return all_iters


def compare_diag(kernel_data, gt_data):
    """Compare kernel diagnostics with ground truth, field by field."""
    RTOL = 0.01
    float_fields = ['dist_init', 'dist_post', 'cksum', 'red0_val', 'red1_val',
                    'local_val', 'global_val']
    idx_fields = ['red0_idx', 'red1_idx', 'local_idx', 'global_idx']

    total_fields = 0
    total_mismatch = 0

    for it in range(DEBUG_ITERS):
        print(f"\n{'=' * 72}")
        print(f"  Iteration {it + 1}")
        print(f"{'=' * 72}")
        for core in range(NUM_CORES):
            kd = kernel_data[it][core]
            gd = gt_data[it][core]

            mismatches = []
            for f in float_fields:
                total_fields += 1
                kv, gv = kd[f], gd[f]
                if abs(gv) < 1e-6:
                    match = abs(kv) < 1e-6
                else:
                    match = abs(kv - gv) / max(abs(gv), 1e-10) < RTOL
                if not match:
                    total_mismatch += 1
                    mismatches.append(f"{f}: got={kv:.6f} expect={gv:.6f}")

            for f in idx_fields:
                total_fields += 1
                kv, gv = kd[f], gd[f]
                if kv != gv:
                    total_mismatch += 1
                    mismatches.append(f"{f}: got={kv} expect={gv}")

            if mismatches:
                print(f"  Core {core}: MISMATCH ({len(mismatches)} fields)")
                for m in mismatches:
                    print(f"    {m}")
            else:
                print(f"  Core {core}: OK  "
                      f"local={kd['local_val']:.4f}(i={kd['local_idx']})  "
                      f"global={kd['global_val']:.4f}(i={kd['global_idx']})")

    print(f"\n  Summary: {total_fields - total_mismatch}/{total_fields} fields match")
    return total_mismatch == 0


def run_debug():
    torch.manual_seed(42)
    xyz = torch.randn(1, N, 3, device='npu:0', dtype=torch.float32)

    # Ground truth FPS
    idx_py = pointnet2_ops._furthest_point_sampling(xyz, NPOINTS)
    print(f"PyTorch idx[:15]: {idx_py[0, :15].tolist()}")

    # Transpose for kernel: [1,3,N] -> [1, 3*N]
    xyz_t = xyz.permute(0, 2, 1).contiguous().reshape(1, -1)

    # Compute expected diagnostics from the same input
    xyz_flat = xyz_t[0].cpu().numpy()
    gt_data = compute_ground_truth(xyz_flat)

    # Run multi-core kernel
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
        print(f"\n=== v3 Diagnostics vs Ground Truth ===")
        all_ok = compare_diag(diag_data, gt_data)

        idx_v3 = idx_dbg[0].cpu().numpy()
        match = "MATCH" if np.array_equal(idx_py[0].cpu().numpy(), idx_v3) else "MISMATCH"
        print(f"\n  v3 idx[:15]: {idx_v3[:15].tolist()}  ({match})")
    except Exception as e:
        print(f"v3 diag: SKIP ({e})")


if __name__ == "__main__":
    run_debug()
