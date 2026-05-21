"""
Full comparison: PyTorch vs 1-core AscendC vs 8-core AscendC.
Multiple verification dimensions, raw value display at the end.
"""
import numpy as np
import torch
import torch_npu  # noqa: F401
import pointnet2_ops

from fps_wrapper import FurthestPointSamplingAscendC, FurthestPointSamplingMultiCore

torch.npu.set_device(0)

N = 1024
NPOINTS = 512


def run_pytorch_fps(xyz):
    return pointnet2_ops._furthest_point_sampling(xyz, NPOINTS)[0].cpu().numpy().astype(np.int64)


def run_1core_fps(xyz_t):
    fps = FurthestPointSamplingAscendC()
    xyz_4d = xyz_t.reshape(1, -1)  # wrapper expects [B, 3*N]
    # wrapper does permute internally, but xyz_t is already transposed, so pass directly
    # actually wrapper expects [B, N, 3], so reconstruct
    xyz_in = xyz_t.reshape(1, 3, N).permute(0, 2, 1).contiguous()  # back to [1, N, 3]
    idx = fps(xyz_in, NPOINTS)
    return idx[0].cpu().numpy().astype(np.int64)


def run_8core_fps(xyz_t):
    fps = FurthestPointSamplingMultiCore()
    xyz_in = xyz_t.reshape(1, 3, N).permute(0, 2, 1).contiguous()  # back to [1, N, 3]
    idx = fps(xyz_in, NPOINTS)
    return idx[0].cpu().numpy().astype(np.int64)


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    xyz = torch.randn(1, N, 3, device='npu:0', dtype=torch.float32)
    xyz_t = xyz.permute(0, 2, 1).contiguous().reshape(-1)
    xyz_np = xyz[0].cpu().numpy()

    print(f"Input: {N} points, select {NPOINTS}, seed=42\n")

    py = run_pytorch_fps(xyz)
    sc = run_1core_fps(xyz_t)
    mc = run_8core_fps(xyz_t)

    # === [1] Full array exact match ===
    sc_ok = np.array_equal(py, sc)
    mc_ok = np.array_equal(py, mc)
    print(f"[1] Full {NPOINTS}-element exact match")
    print(f"    PyTorch vs 1-core: {'MATCH' if sc_ok else f'FAIL ({int(np.sum(py!=sc))} diff)'}")
    print(f"    PyTorch vs 8-core: {'MATCH' if mc_ok else f'FAIL ({int(np.sum(py!=mc))} diff)'}")

    # === [2] Unique count ===
    print(f"\n[2] Unique index count (expect {NPOINTS}, no repeats)")
    print(f"    PyTorch: {len(np.unique(py))}  1-core: {len(np.unique(sc))}  8-core: {len(np.unique(mc))}")

    # === [3] Range validity ===
    print(f"\n[3] Index range (expect [0, {N}))")
    print(f"    PyTorch: [{py.min()}, {py.max()}]")
    print(f"    1-core:  [{sc.min()}, {sc.max()}] {'OK' if sc.min()>=0 and sc.max()<N else 'INVALID'}")
    print(f"    8-core:  [{mc.min()}, {mc.max()}] {'OK' if mc.min()>=0 and mc.max()<N else 'INVALID'}")

    # === [4] First mismatch position ===
    if not sc_ok:
        i = int(np.argmax(py != sc))
        print(f"\n[4a] 1-core first diff at pos {i}: py={py[i]} got={sc[i]}")
    if not mc_ok:
        i = int(np.argmax(py != mc))
        print(f"[4b] 8-core first diff at pos {i}: py={py[i]} got={mc[i]}")

    # === [5] Multi-seed robustness ===
    print(f"\n[5] Multi-seed check")
    seeds = [0, 1, 7, 42, 123, 999, 2024, 31415]
    for seed in seeds:
        torch.manual_seed(seed)
        x = torch.randn(1, N, 3, device='npu:0', dtype=torch.float32)
        xt = x.permute(0, 2, 1).contiguous().reshape(-1)
        p = run_pytorch_fps(x)
        s = run_1core_fps(xt)
        m = run_8core_fps(xt)
        t1 = "OK" if np.array_equal(p, s) else f"FAIL({int(np.sum(p!=s))})"
        t8 = "OK" if np.array_equal(p, m) else f"FAIL({int(np.sum(p!=m))})"
        print(f"    seed={seed:<6} 1-core:{t1:<14} 8-core:{t8}")

    # === [6] Raw values for visual inspection ===
    print(f"\n{'='*70}")
    print(f"  RAW VALUES — visual inspection")
    print(f"{'='*70}")

    positions = [0, slice(0, 20), slice(100, 120), slice(250, 270), slice(-20, None)]
    labels = ["idx[0]", "idx[0:20]", "idx[100:120]", "idx[250:270]", "idx[-20:] (492:512)"]
    for label, pos in zip(labels, positions):
        p = py[pos].tolist()
        s = sc[pos].tolist()
        m = mc[pos].tolist()
        print(f"\n  {label}")
        print(f"    PyTorch: {p}")
        print(f"    1-core:  {s} {'✓' if s==p else '✗'}")
        print(f"    8-core:  {m} {'✓' if m==p else '✗'}")

    # Show coordinates of selected points at a few positions
    print(f"\n  Selected point coordinates (x,y,z) at key positions:")
    for i in [0, 1, 10, 50, 100, 200, 300, 400, 511]:
        pi = py[i]
        si = sc[i]
        mi = mc[i]
        pc = xyz_np[pi].tolist()
        sc_c = xyz_np[si].tolist()
        mc_c = xyz_np[mi].tolist()
        match_s = "✓" if si == pi else "✗"
        match_m = "✓" if mi == pi else "✗"
        print(f"    pos {i:>3}: py[{pi:>4}]={pc}  1c[{si:>4}]{match_s}={sc_c}  8c[{mi:>4}]{match_m}={mc_c}")


if __name__ == "__main__":
    main()
