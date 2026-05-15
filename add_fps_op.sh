#!/bin/bash
# FPS 算子文件添加到 TIK 样例工程
# 用法: bash add_fps_op.sh <样例工程路径>
# 例如: bash add_fps_op.sh /usr/local/Ascend/ascend-toolkit/8.3.RC1/tools/msopgen/template/custom_operator_sample/TIK/PyTorch

BASE=${1:-.}

# 文件 1: op_proto/furthest_point_sampling.h
cat > $BASE/op_proto/furthest_point_sampling.h << 'EOF'
#ifndef GE_OP_FURTHEST_POINT_SAMPLING_H
#define GE_OP_FURTHEST_POINT_SAMPLING_H

#include "graph/operator_reg.h"
#include "graph/operator.h"

namespace ge {
REG_OP(FurthestPointSampling)
    .INPUT(xyz, TensorType({DT_FLOAT}))
    .OUTPUT(idx, TensorType({DT_INT32}))
    .REQUIRED_ATTR(npoints, Int)
    .OP_END_FACTORY_REG(FurthestPointSampling)
}  // namespace ge

#endif
EOF

# 文件 2: op_proto/furthest_point_sampling.cc
cat > $BASE/op_proto/furthest_point_sampling.cc << 'EOF'
#include "furthest_point_sampling.h"
#include "util/util.h"

namespace ge {
IMPLEMT_INFERFUNC(FurthestPointSampling, FurthestPointSamplingInferShape) {
    auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
    auto xyz_desc = op_desc->MutableInputDesc(0);
    int npoints = 0;
    (void)op.GetAttr("npoints", npoints);
    ge::Shape idx_shape({npoints});
    auto idx_desc = op_desc->MutableOutputDesc(0);
    idx_desc->SetShape(idx_shape);
    idx_desc->SetDataType(DT_INT32);
    return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(FurthestPointSampling, FurthestPointSamplingVerify) {
    return GRAPH_SUCCESS;
}

INFER_FUNC_REG(FurthestPointSampling, FurthestPointSamplingInferShape);
VERIFY_FUNC_REG(FurthestPointSampling, FurthestPointSamplingVerify);
}  // namespace ge
EOF

# 文件 3: tbe/impl/furthest_point_sampling.py
cat > $BASE/tbe/impl/furthest_point_sampling.py << 'PYEOF'
# -*- coding: utf-8 -*-
from tbe import tik
from tbe.common.register import register_op_compute

BLOCK_SIZE = 32
VEC_WIDTH = 64


def _fps_iteration(tik_inst, xyz_ub, distance_ub, temp_ub, dist_ub,
                   old, N, N_BLOCKS):
    x1 = tik_inst.Scalar("float32", name="x1")
    y1 = tik_inst.Scalar("float32", name="y1")
    z1 = tik_inst.Scalar("float32", name="z1")
    x1.set_as(xyz_ub[old])
    y1.set_as(xyz_ub[N + old])
    z1.set_as(xyz_ub[2 * N + old])

    for block in range(N_BLOCKS):
        base = block * VEC_WIDTH
        tik_inst.vec_dup(VEC_WIDTH, temp_ub[0], x1, 1, 1)
        tik_inst.vec_sub(VEC_WIDTH, dist_ub[0], xyz_ub[base], temp_ub[0], 1, 8, 8, 8)
        tik_inst.vec_mul(VEC_WIDTH, dist_ub[0], dist_ub[0], dist_ub[0], 1, 8, 8, 8)
        tik_inst.vec_dup(VEC_WIDTH, temp_ub[0], y1, 1, 1)
        tik_inst.vec_sub(VEC_WIDTH, temp_ub[0], xyz_ub[N + base], temp_ub[0], 1, 8, 8, 8)
        tik_inst.vec_mul(VEC_WIDTH, temp_ub[0], temp_ub[0], temp_ub[0], 1, 8, 8, 8)
        tik_inst.vec_add(VEC_WIDTH, dist_ub[0], dist_ub[0], temp_ub[0], 1, 8, 8, 8)
        tik_inst.vec_dup(VEC_WIDTH, temp_ub[0], z1, 1, 1)
        tik_inst.vec_sub(VEC_WIDTH, temp_ub[0], xyz_ub[2 * N + base], temp_ub[0], 1, 8, 8, 8)
        tik_inst.vec_mul(VEC_WIDTH, temp_ub[0], temp_ub[0], temp_ub[0], 1, 8, 8, 8)
        tik_inst.vec_add(VEC_WIDTH, dist_ub[0], dist_ub[0], temp_ub[0], 1, 8, 8, 8)
        tik_inst.vec_min(VEC_WIDTH, distance_ub[base], distance_ub[base], dist_ub[0], 1, 8, 8, 8)

    best_val = tik_inst.Scalar("float32", name="best_val")
    best_val.set_as(-1.0)
    best_idx = tik_inst.Scalar("int32", name="best_idx")
    best_idx.set_as(0)
    with tik_inst.for_range(0, N, name="k") as k:
        val = tik_inst.Scalar("float32", name="val")
        val.set_as(distance_ub[k])
        with tik_inst.if_scope(val > best_val):
            best_val.set_as(val)
            best_idx.set_as(k)
    old.set_as(best_idx)


def _write_idx(tik_inst, idx_ub, block_j, s, old):
    base = block_j * 8
    idx_scalar = tik_inst.Scalar("int32", name="idx_s")
    idx_scalar.set_as(old)
    tik_inst.vec_dup([0, 1 << s], idx_ub[base], idx_scalar, 1, 1)


@register_op_compute("FurthestPointSampling")
def furthest_point_sampling_compute(xyz, idx, npoints, kernel_name="furthest_point_sampling"):
    tik_inst = tik.Tik()

    N = 1024
    npoints_val = npoints
    N_BLOCKS = N // VEC_WIDTH
    BLOCKS_OF_IDX = npoints_val // 8

    xyz_gm = tik_inst.Tensor("float32", (3 * N,), name="xyz_gm", scope=tik.scope_gm)
    idx_gm = tik_inst.Tensor("int32", (npoints_val,), name="idx_gm", scope=tik.scope_gm)

    xyz_ub = tik_inst.Tensor("float32", (3 * N,), name="xyz_ub", scope=tik.scope_ubuf)
    distance_ub = tik_inst.Tensor("float32", (N,), name="distance_ub", scope=tik.scope_ubuf)
    idx_ub = tik_inst.Tensor("int32", (npoints_val,), name="idx_ub", scope=tik.scope_ubuf)
    temp_ub = tik_inst.Tensor("float32", (VEC_WIDTH,), name="temp_ub", scope=tik.scope_ubuf)
    dist_ub = tik_inst.Tensor("float32", (VEC_WIDTH,), name="dist_ub", scope=tik.scope_ubuf)

    xyz_burst = 3 * N * 4 // BLOCK_SIZE
    tik_inst.data_move(xyz_ub, xyz_gm, 0, 1, xyz_burst, 0, 0)

    tik_inst.vec_dup(VEC_WIDTH, distance_ub[0], 1e10, N // VEC_WIDTH, 8)

    old = tik_inst.Scalar("int32", name="old", init_value=0)
    zero_scalar = tik_inst.Scalar("int32", name="zero_s", init_value=0)
    tik_inst.vec_dup([0, 1], idx_ub[0], zero_scalar, 1, 1)

    with tik_inst.for_range(0, BLOCKS_OF_IDX, name="block_j") as block_j:
        for s in range(8):
            if s == 0:
                with tik_inst.if_scope(block_j == 0):
                    pass
                with tik_inst.else_scope():
                    _fps_iteration(tik_inst, xyz_ub, distance_ub, temp_ub, dist_ub,
                                   old, N, N_BLOCKS)
                    _write_idx(tik_inst, idx_ub, block_j, s, old)
            else:
                _fps_iteration(tik_inst, xyz_ub, distance_ub, temp_ub, dist_ub,
                               old, N, N_BLOCKS)
                _write_idx(tik_inst, idx_ub, block_j, s, old)

    idx_burst = npoints_val * 4 // BLOCK_SIZE
    tik_inst.data_move(idx_gm, idx_ub, 0, 1, idx_burst, 0, 0)

    tik_inst.BuildCCE(kernel_name=kernel_name,
                       inputs=[xyz_gm],
                       outputs=[idx_gm])
    return tik_inst
PYEOF

# 文件 4: tbe/op_info_cfg/ai_core/ascend910/furthest_point_sampling.ini
cat > $BASE/tbe/op_info_cfg/ai_core/ascend910/furthest_point_sampling.ini << 'EOF'
[FurthestPointSampling]
input0.xyz=required,float32,ND
output0.idx=required,int32,ND
attr0.npoints=required,int
EOF

echo "Done. 4 files created:"
echo "  $BASE/op_proto/furthest_point_sampling.h"
echo "  $BASE/op_proto/furthest_point_sampling.cc"
echo "  $BASE/tbe/impl/furthest_point_sampling.py"
echo "  $BASE/tbe/op_info_cfg/ai_core/ascend910/furthest_point_sampling.ini"
