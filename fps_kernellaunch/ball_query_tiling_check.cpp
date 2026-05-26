/**
 * Minimal debug kernel: read tiling data and write back to debug GM.
 * Each core writes 8 floats: [B, N, M, nsample, radius, numCores, queriesPerCore, coreId]
 */
#include "kernel_operator.h"
#include "ball_query_dynamic.h"

class KernelBallQueryTilingCheck {
public:
    __aicore__ inline void Init(GM_ADDR tiling, GM_ADDR debug)
    {
        // 1. Read tiling into UB
        pipe.InitBuffer(tilingBuf, 8 * sizeof(int32_t));
        AscendC::GlobalTensor<int32_t> tilingGm;
        tilingGm.SetGlobalBuffer((__gm__ int32_t *)tiling, 8);
        auto tilingLocal = tilingBuf.Get<int32_t>();
        AscendC::DataCopy(tilingLocal, tilingGm, 8);
        pipe_barrier(PIPE_V);

        B = tilingLocal.GetValue(0);
        N = tilingLocal.GetValue(1);
        M = tilingLocal.GetValue(2);
        nsample = tilingLocal.GetValue(3);
        union { int32_t i; float f; } c;
        c.i = tilingLocal.GetValue(4);
        radius = c.f;
        numCores = tilingLocal.GetValue(5);
        queriesPerCore = tilingLocal.GetValue(6);
        coreId = AscendC::GetBlockIdx();

        // 2. Debug output GM
        debugGm.SetGlobalBuffer((__gm__ float *)debug, 8 * 8); // 8 cores * 8 floats
        pipe.InitBuffer(debugBuf, 8 * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        auto out = debugBuf.Get<float>();
        out.SetValue(0, (float)B);
        out.SetValue(1, (float)N);
        out.SetValue(2, (float)M);
        out.SetValue(3, (float)nsample);
        out.SetValue(4, radius);
        out.SetValue(5, (float)numCores);
        out.SetValue(6, (float)queriesPerCore);
        out.SetValue(7, (float)coreId);
        pipe_barrier(PIPE_V);

        AscendC::DataCopy(debugGm[coreId * 8], out, 8);
        pipe_barrier(PIPE_V);
    }

private:
    int32_t B, N, M, nsample, numCores, queriesPerCore, coreId;
    float radius;
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECIN> tilingBuf;
    AscendC::TBuf<AscendC::TPosition::VECIN> debugBuf;
    AscendC::GlobalTensor<float> debugGm;
};

extern "C" __global__ __aicore__ void ball_query_tiling_check(
    GM_ADDR tiling, GM_ADDR debug)
{
    KernelBallQueryTilingCheck op;
    op.Init(tiling, debug);
    op.Process();
}
