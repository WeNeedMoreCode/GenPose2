"""
TIK 算子：Group Points Forward（点特征分组）

功能：按索引从 points 中 gather 特征到 out
  out[b, c, pt, s] = points[b, c, idx[b, pt, s]]

简化版固定参数：B=1, C=8, N=128, npoints=32, nsample=8
参考：vec_add.py + CANN算子开发学习文档.md

数据流：
  GM(points, idx) → UB → 计算(gather) → UB(out) → GM(out)
"""

from tbe import tik
import tbe.common.platform as tbe_platform
import numpy as np

# 固定参数
B = 1
C = 8
N = 128
NPOINTS = 32
NSAMPLE = 8

BLOCK_SIZE = 32  # bytes per block (data_move 最小单位)
DTYPE = "float32"
DTYPE_SIZE = 4   # bytes per float32
ELEMS_PER_BLOCK = BLOCK_SIZE // DTYPE_SIZE  # 8 float32 per block


def group_points_forward():
    tbe_platform.set_current_compile_soc_info("Ascend910A")
    tik_inst = tik.Tik(disable_debug=False)

    # ---- GM tensors ----
    # points: [B, C, N] 展平为一维
    points_gm = tik_inst.Tensor(DTYPE, (B * C * N,),
                                name="points_gm", scope=tik.scope_gm)
    # idx: [B, npoints, nsample] 展平为一维
    idx_gm = tik_inst.Tensor("int32", (B * NPOINTS * NSAMPLE,),
                              name="idx_gm", scope=tik.scope_gm)
    # out: [B, C, npoints, nsample] 展平为一维
    out_gm = tik_inst.Tensor(DTYPE, (B * C * NPOINTS * NSAMPLE,),
                              name="out_gm", scope=tik.scope_gm)

    # ---- UB tensors ----
    # 全部数据可以一次性放入 UB（~248KB）
    points_ub = tik_inst.Tensor(DTYPE, (C * N,),
                                 name="points_ub", scope=tik.scope_ubuf)
    idx_ub = tik_inst.Tensor("int32", (NPOINTS * NSAMPLE,),
                              name="idx_ub", scope=tik.scope_ubuf)
    out_ub = tik_inst.Tensor(DTYPE, (C * NPOINTS * NSAMPLE,),
                              name="out_ub", scope=tik.scope_ubuf)

    # ---- Step 1: 搬入 idx (GM → UB) ----
    # NPOINTS * NSAMPLE 个 int32 = 256 * 4 = 1024 bytes = 32 blocks
    idx_total_bytes = NPOINTS * NSAMPLE * 4
    idx_burst = idx_total_bytes // BLOCK_SIZE  # 32
    tik_inst.data_move(idx_ub, idx_gm, 0, 1, idx_burst, 0, 0)

    # ---- Step 2: 搬入 points (GM → UB) ----
    # C * N 个 float32 = 1024 * 4 = 4096 bytes = 128 blocks
    points_total_bytes = C * N * DTYPE_SIZE
    points_burst = points_total_bytes // BLOCK_SIZE  # 128
    tik_inst.data_move(points_ub, points_gm, 0, 1, points_burst, 0, 0)

    # ---- Step 3: 初始化 out_ub 为 0 ----
    out_elems = C * NPOINTS * NSAMPLE  # 2048
    # vec_dup: 用 0 填充，mask=64 (float32 最大 mask), repeat 直到覆盖全部
    full_repeats = out_elems // 64  # 32
    tik_inst.vec_dup(64, out_ub[0], 0, full_repeats, 8)

    # ---- Step 4: Gather 循环 ----
    # out[c * NPOINTS * NSAMPLE + pt * NSAMPLE + s] = points[c * N + idx[pt * NSAMPLE + s]]
    # 关键：out_base = c*NPOINTS*NSAMPLE + pt*NSAMPLE 始终是 8 的倍数（对齐）
    # 用逐 bit 模式 mask 选择块内第 s 个元素写入，其余元素保持不变
    with tik_inst.for_range(0, C, name="c") as c:
        with tik_inst.for_range(0, NPOINTS, name="pt") as pt:
            for s in range(NSAMPLE):  # Python 循环展开，s 是 Python int
                idx_scalar = tik_inst.Scalar("int32", name="idx_val")
                idx_scalar.set_as(idx_ub[pt * NSAMPLE + s])

                val_scalar = tik_inst.Scalar(DTYPE, name="val")
                val_scalar.set_as(points_ub[c * N + idx_scalar])

                # bit-mode mask: [高64位, 低64位]，float32 只用低64位
                out_base = c * NPOINTS * NSAMPLE + pt * NSAMPLE
                tik_inst.vec_dup([0, 1 << s], out_ub[out_base], val_scalar, 1, 1)

    # ---- Step 5: 搬出 out (UB → GM) ----
    out_total_bytes = C * NPOINTS * NSAMPLE * DTYPE_SIZE
    out_burst = out_total_bytes // BLOCK_SIZE  # 256
    tik_inst.data_move(out_gm, out_ub, 0, 1, out_burst, 0, 0)

    # ---- Step 6: 编译 ----
    tik_inst.BuildCCE(kernel_name="group_points_forward",
                       inputs=[points_gm, idx_gm],
                       outputs=[out_gm])
    return tik_inst


if __name__ == "__main__":
    tik_inst = group_points_forward()

    # 生成测试数据
    np.random.seed(42)
    points = np.random.randn(B * C * N).astype(np.float32)
    idx = np.random.randint(0, N, (B * NPOINTS * NSAMPLE,)).astype(np.int32)

    # 计算期望输出
    points_4d = points.reshape(B, C, N)
    idx_3d = idx.reshape(B, NPOINTS, NSAMPLE)
    expected = np.zeros((B, C, NPOINTS, NSAMPLE), dtype=np.float32)
    for b in range(B):
        for c in range(C):
            for pt in range(NPOINTS):
                for s in range(NSAMPLE):
                    expected[b, c, pt, s] = points_4d[b, c, idx_3d[b, pt, s]]
    expected_flat = expected.flatten()

    feed_dict = {"points_gm": points, "idx_gm": idx}

    print("启动 TIK 调试...")
    out_result, = tik_inst.tikdb.start_debug(feed_dict=feed_dict, interactive=True)

    print("\n期望输出 (前 20 个):")
    print(expected_flat[:20])
    print("\nTIK 输出 (前 20 个):")
    print(out_result[:20])
    print("\n最大误差:", np.max(np.abs(out_result - expected_flat)))
    print("结果:", "PASS" if np.allclose(out_result, expected_flat, atol=1e-6) else "FAIL")
