"""
TIK 算子：Furthest Point Sampling (FPS)

功能：从 N 个点中迭代选择 npoints 个最远点
  idx[0] = 0
  for j = 1..npoints-1:
    计算所有点到当前选中点的距离
    更新最小距离
    选择最远点作为下一个采样点

固定参数：B=1, N=1024, npoints=512
输入格式：xyz [3*N] float32（转置布局：x连续, y连续, z连续）
输出格式：idx [npoints] int32

参考：GenPose2-cuda/.../sampling_gpu.cu（CUDA FPS kernel）
"""

from tbe import tik
import tbe.common.platform as tbe_platform
import numpy as np

# 固定参数（tikdb 验证用小参数，NPU 性能测试改回大参数）
B = 1
N = 1024
NPOINTS = 512
DTYPE = "float32"
BLOCK_SIZE = 32  # bytes
VEC_WIDTH = 64   # float32 向量宽度
N_BLOCKS = N // VEC_WIDTH  # 128//64=2 或 1024//64=16
BLOCKS_OF_IDX = NPOINTS // 8  # 32//8=4 或 512//8=64


def fps_forward():
    tbe_platform.set_current_compile_soc_info("Ascend910A")
    tik_inst = tik.Tik(disable_debug=False)

    # ---- GM tensors ----
    xyz_gm = tik_inst.Tensor(DTYPE, (3 * N,), name="xyz_gm", scope=tik.scope_gm)
    idx_gm = tik_inst.Tensor("int32", (NPOINTS,), name="idx_gm", scope=tik.scope_gm)

    # ---- UB tensors ----
    xyz_ub = tik_inst.Tensor(DTYPE, (3 * N,), name="xyz_ub", scope=tik.scope_ubuf)
    distance_ub = tik_inst.Tensor(DTYPE, (N,), name="distance_ub", scope=tik.scope_ubuf)
    idx_ub = tik_inst.Tensor("int32", (NPOINTS,), name="idx_ub", scope=tik.scope_ubuf)
    temp_ub = tik_inst.Tensor(DTYPE, (VEC_WIDTH,), name="temp_ub", scope=tik.scope_ubuf)
    dist_ub = tik_inst.Tensor(DTYPE, (VEC_WIDTH,), name="dist_ub", scope=tik.scope_ubuf)

    # ---- Step 1: 搬入 xyz (GM → UB) ----
    xyz_burst = 3 * N * 4 // BLOCK_SIZE  # 384 blocks
    tik_inst.data_move(xyz_ub, xyz_gm, 0, 1, xyz_burst, 0, 0)

    # ---- Step 2: 初始化 distance = 1e10 ----
    tik_inst.vec_dup(VEC_WIDTH, distance_ub[0], 1e10, N // VEC_WIDTH, 8)

    # ---- Step 3: 初始化 idx[0] = 0, old = 0 ----
    old = tik_inst.Scalar("int32", name="old", init_value=0)
    zero_scalar = tik_inst.Scalar("int32", name="zero_s", init_value=0)
    tik_inst.vec_dup([0, 1], idx_ub[0], zero_scalar, 1, 1)

    # ---- Step 4: 主循环 ----
    # 外层按 8 个一组循环（保证 idx 写入基址对齐）
    # 内层 Python for loop 用 bit-mode mask 写入
    with tik_inst.for_range(0, BLOCKS_OF_IDX, name="block_j") as block_j:
        for s in range(8):
            # j = block_j * 8 + s
            # 跳过 j=0（block_j=0, s=0）
            if s == 0:
                with tik_inst.if_scope(block_j == 0):
                    # j=0: 已初始化，跳过
                    pass
                with tik_inst.else_scope():
                    # j = block_j * 8 > 0: 正常 FPS 迭代
                    _fps_iteration(tik_inst, xyz_ub, distance_ub, temp_ub, dist_ub,
                                   old, N, VEC_WIDTH, N_BLOCKS)
                    # 写入 idx: 基址 block_j*8（对齐），bit mask 选第 0 个元素
                    _write_idx(tik_inst, idx_ub, block_j, s, old)
            else:
                # s > 0: 正常 FPS 迭代
                _fps_iteration(tik_inst, xyz_ub, distance_ub, temp_ub, dist_ub,
                               old, N, VEC_WIDTH, N_BLOCKS)
                # 写入 idx
                _write_idx(tik_inst, idx_ub, block_j, s, old)

    # ---- Step 5: 搬出 idx (UB → GM) ----
    idx_burst = NPOINTS * 4 // BLOCK_SIZE  # 64 blocks
    tik_inst.data_move(idx_gm, idx_ub, 0, 1, idx_burst, 0, 0)

    # ---- Step 6: 编译 ----
    tik_inst.BuildCCE(kernel_name="fps_forward",
                       inputs=[xyz_gm],
                       outputs=[idx_gm])
    return tik_inst


def _fps_iteration(tik_inst, xyz_ub, distance_ub, temp_ub, dist_ub,
                   old, N, VEC_WIDTH, N_BLOCKS):
    """一次 FPS 迭代：读质心 → 算距离 → 更新最小距离 → argmax → 更新 old"""
    # 读质心
    x1 = tik_inst.Scalar(DTYPE, name="x1")
    y1 = tik_inst.Scalar(DTYPE, name="y1")
    z1 = tik_inst.Scalar(DTYPE, name="z1")
    x1.set_as(xyz_ub[old])
    y1.set_as(xyz_ub[N + old])
    z1.set_as(xyz_ub[2 * N + old])

    # 距离计算：16 个 block，每 block 64 个点
    for block in range(N_BLOCKS):
        base = block * VEC_WIDTH

        # dx²
        tik_inst.vec_dup(VEC_WIDTH, temp_ub[0], x1, 1, 1)
        tik_inst.vec_sub(VEC_WIDTH, dist_ub[0], xyz_ub[base], temp_ub[0], 1, 8, 8, 8)
        tik_inst.vec_mul(VEC_WIDTH, dist_ub[0], dist_ub[0], dist_ub[0], 1, 8, 8, 8)

        # dy² 累加
        tik_inst.vec_dup(VEC_WIDTH, temp_ub[0], y1, 1, 1)
        tik_inst.vec_sub(VEC_WIDTH, temp_ub[0], xyz_ub[N + base], temp_ub[0], 1, 8, 8, 8)
        tik_inst.vec_mul(VEC_WIDTH, temp_ub[0], temp_ub[0], temp_ub[0], 1, 8, 8, 8)
        tik_inst.vec_add(VEC_WIDTH, dist_ub[0], dist_ub[0], temp_ub[0], 1, 8, 8, 8)

        # dz² 累加
        tik_inst.vec_dup(VEC_WIDTH, temp_ub[0], z1, 1, 1)
        tik_inst.vec_sub(VEC_WIDTH, temp_ub[0], xyz_ub[2 * N + base], temp_ub[0], 1, 8, 8, 8)
        tik_inst.vec_mul(VEC_WIDTH, temp_ub[0], temp_ub[0], temp_ub[0], 1, 8, 8, 8)
        tik_inst.vec_add(VEC_WIDTH, dist_ub[0], dist_ub[0], temp_ub[0], 1, 8, 8, 8)

        # distance = min(distance, d)
        tik_inst.vec_min(VEC_WIDTH, distance_ub[base], distance_ub[base], dist_ub[0], 1, 8, 8, 8)

    # argmax：TIK for_range 硬件循环 + Scalar 比较
    best_val = tik_inst.Scalar(DTYPE, name="best_val")
    best_val.set_as(-1.0)
    best_idx = tik_inst.Scalar("int32", name="best_idx")
    best_idx.set_as(0)

    with tik_inst.for_range(0, N, name="k") as k:
        val = tik_inst.Scalar(DTYPE, name="val")
        val.set_as(distance_ub[k])
        with tik_inst.if_scope(val > best_val):
            best_val.set_as(val)
            best_idx.set_as(k)

    old.set_as(best_idx)


def _write_idx(tik_inst, idx_ub, block_j, s, old):
    """用 bit-mode mask 写入 idx_ub[block_j*8 + s]"""
    base = block_j * 8  # 对齐基址（8 的倍数）
    idx_scalar = tik_inst.Scalar("int32", name="idx_s")
    idx_scalar.set_as(old)
    tik_inst.vec_dup([0, 1 << s], idx_ub[base], idx_scalar, 1, 1)


def pytorch_fps(xyz, npoints):
    """参考实现：纯 numpy FPS 算法"""
    B, N, _ = xyz.shape
    idx = np.zeros((B, npoints), dtype=np.int32)
    distance = np.ones((B, N), dtype=np.float32) * 1e10
    farthest = np.zeros((B,), dtype=np.int32)
    idx[:, 0] = farthest

    for b in range(B):
        for i in range(1, npoints):
            centroid = xyz[b, farthest[b], :]
            dist = np.sum((xyz[b] - centroid) ** 2, axis=-1)
            distance[b] = np.minimum(distance[b], dist)
            farthest[b] = np.argmax(distance[b])
            idx[b, i] = farthest[b]

    return idx


if __name__ == "__main__":
    tik_inst = fps_forward()

    # 生成测试数据
    np.random.seed(42)
    xyz = np.random.randn(B, N, 3).astype(np.float32)

    # 转置输入：[N, 3] → [3, N] = x连续, y连续, z连续
    xyz_t = xyz.transpose(0, 2, 1).reshape(-1)

    # 期望输出
    expected_idx = pytorch_fps(xyz, NPOINTS)

    feed_dict = {"xyz_gm": xyz_t}

    print("启动 TIK 调试...")
    out_result, = tik_inst.tikdb.start_debug(feed_dict=feed_dict, interactive=False)

    print("\n期望输出 (前 20 个):")
    print(expected_idx[0, :20])
    print("\nTIK 输出 (前 20 个):")
    print(out_result[:20])
    print("\n结果:", "PASS" if np.array_equal(out_result, expected_idx[0]) else "FAIL")
