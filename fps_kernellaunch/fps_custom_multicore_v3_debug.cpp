/**
 * Minimal multi-core Duplicate test.
 * Each core: InitBuffer(distBuf) → Duplicate(dist, 1e10) → GetValue → SetValue(debugGm)
 * Only distBuf and debugGm, no other buffers.
 * Goal: determine if Duplicate fails in isolation on some cores.
 */
#include "kernel_operator.h"

constexpr int32_t N = 1024;
constexpr int32_t NUM_CORES = 8;
constexpr int32_t CHUNK = N / NUM_CORES;

class KernelFpsMultiCoreV3Debug {
public:
    __aicore__ inline void Init(GM_ADDR xyz, GM_ADDR idx, GM_ADDR results, GM_ADDR debug)
    {
        coreId = AscendC::GetBlockIdx();
        // debug layout: [alive_marker(8), dup_result(8)]
        debugGm.SetGlobalBuffer((__gm__ float *)debug, NUM_CORES * 2);
        pipe.InitBuffer(distBuf, CHUNK * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        // CP0: "I'm alive" marker — if host sees this but not dup_result, core ran but Duplicate failed
        debugGm.SetValue(coreId, -1.0f);

        auto dist = distBuf.Get<float>();

        AscendC::Duplicate(dist, 1e10f, CHUNK);
        pipe_barrier(PIPE_V);

        float val = dist.GetValue(0);
        debugGm.SetValue(NUM_CORES + coreId, val);
    }

private:
    int32_t coreId;
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECIN> distBuf;
    AscendC::GlobalTensor<float> debugGm;
};

extern "C" __global__ __aicore__ void fps_custom_mc_v3_dbg(GM_ADDR xyz, GM_ADDR idx, GM_ADDR results, GM_ADDR debug)
{
    KernelFpsMultiCoreV3Debug op;
    op.Init(xyz, idx, results, debug);
    op.Process();
}

#ifndef ASCENDC_CPU_DEBUG
void fps_custom_mc_v3_dbg_do(uint32_t blockDim, void *stream, uint8_t *xyz, uint8_t *idx, uint8_t *results, uint8_t *debug)
{
    fps_custom_mc_v3_dbg<<<blockDim, nullptr, stream>>>(xyz, idx, results, debug);
}
#endif
