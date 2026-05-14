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

# 固定参数
B = 1
N = 1024
NPOINTS = 512
DTYPE = "float32"
DTYPE_SIZE = 4
BLOCK_SIZE = 32
ELEMS_PER_BLOCK = BLOCK_SIZE // DTYPE_SIZE  # 8 float32 per block
VEC_WIDTH = 64  # float32 向量宽度
N_BLOCKS = N // VEC_WIDTH  # 16 blocks


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
    xyz_burst = 3 * N * DTYPE_SIZE // BLOCK_SIZE  # 384 blocks
    tik_inst.data_move(xyz_ub, xyz_gm, 0, 1, xyz_burst, 0, 0)

    # ---- Step 2: 初始化 distance = 1e10 ----
    init_repeats = N // VEC_WIDTH  # 16
    tik_inst.vec_dup(VEC_WIDTH, distance_ub[0], 1e10, init_repeats, 8)

    # ---- Step 3: 初始化 idx[0] = 0, old = 0 ----
    old = tik_inst.Scalar("int32", name="old", init_value=0)
    first_val = tik_inst.Scalar("int32", name="first_val", init_value=0)
    tik_inst.vec_dup([0, 1], idx_ub[0], first_val, 1, 1)  # bit-mode mask 写 idx[0]

    # ---- Step 4: 主循环 ----
    with tik_inst.for_range(1, NPOINTS, name="j") as j:
        # 4a. 读质心坐标
        x1 = tik_inst.Scalar(DTYPE, name="x1")
        y1 = tik_inst.Scalar(DTYPE, name="y1")
        z1 = tik_inst.Scalar(DTYPE, name="z1")
        x1.set_as(xyz_ub[old])
        y1.set_as(xyz_ub[N + old])
        z1.set_as(xyz_ub[2 * N + old])

        # 4b. 距离计算：16 个 block，每 block 64 个点
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

            # distance = min(distance, dist)
            tik_inst.vec_min(VEC_WIDTH, distance_ub[base], distance_ub[base], dist_ub[0], 1, 8, 8, 8)

        # 4c. argmax：Scalar 扫描找最大距离及其索引
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

        # 4d. 写入 idx[j] 并更新 old
        # idx 是 int32，每个元素占 4 字节
        # 用 Scalar 直接写入（需验证是否支持 int32 tensor 的 Scalar 写入）
        # 安全方式：通过 data_move 写回 GM 或用 vec_dup
        old.set_as(best_idx)
        # 将 best_idx 写入 idx_ub[j]
        # j 是 TIK Expr，不能直接 idx_ub[j] = best_idx
        # 用 vec_dup + bit-mode mask 写入对齐块内的正确位置
        # j 的范围是 1..511，需要计算 j 所在的对齐块基址和块内偏移
        # 简化：用 Scalar 索引写入（如果支持的话）
        idx_scalar = tik_inst.Scalar("int32", name="idx_scalar")
        idx_scalar.set_as(best_idx)
        # TIK 不支持 Scalar 索引写入 tensor，需要用其他方式
        # 方案：先写到 temp Scalar，再用 data_move 或 vec_dup 写入
        # 这里先用 tik_inst 的 tensor_scalar_set 如果有的话
        # 临时方案：写入 GM 然后搬回来（太慢）
        # 最佳方案：用 vec_dup 写入对齐块
        # idx 是 int32，对齐要求是 8 个 int32 = 32 bytes
        # 但 bit-mode mask 是按 float32 设计的，int32 也适用吗？
        # 先尝试用 Scalar 赋值
        tik_inst.vec_dup([0, 1], idx_ub[j], idx_scalar, 1, 1)

    # ---- Step 5: 搬出 idx (UB → GM) ----
    idx_burst = NPOINTS * 4 // BLOCK_SIZE  # 256 blocks (512 * 4 / 32 = 64)
    tik_inst.data_move(idx_gm, idx_ub, 0, 1, idx_burst, 0, 0)

    # ---- Step 6: 编译 ----
    tik_inst.BuildCCE(kernel_name="fps_forward",
                       inputs=[xyz_gm],
                       outputs=[idx_gm])
    return tik_inst


def pytorch_fps(xyz, npoints):
    """参考实现：纯 Python FPS 算法"""
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
    xyz_t = xyz.transpose(0, 2, 1).reshape(-1)  # [3*N]

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
