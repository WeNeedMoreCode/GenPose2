/**
 * Test Duplicate(count=1) with SEPARATE buffers — no buf reuse, no DMA race.
 * buf_a only used for sub-test A, buf_b only for sub-test B.
 * If count=1 works: debug[coreId*16 + 0] = val_a
 * If count=1 broken: debug[coreId*16 + 0] = 0 (from clear)
 */
#include "kernel_operator.h"

constexpr int32_t NUM_CORES = 8;

class KernelDup1Check {
public:
    __aicore__ inline void Init(GM_ADDR xyz, GM_ADDR idx, GM_ADDR results,
                                GM_ADDR debug, GM_ADDR scratch)
    {
        coreId = AscendC::GetBlockIdx();
        debugGm.SetGlobalBuffer((__gm__ float *)debug, NUM_CORES * 16);
        pipe.InitBuffer(bufA, 8 * sizeof(float));
        pipe.InitBuffer(bufB, 8 * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        auto ba = bufA.Get<float>();
        auto bb = bufB.Get<float>();

        // Sub-test A: Duplicate count=1 → writes to bufA (NEVER reused)
        AscendC::Duplicate(ba, 0.0f, 8);  // clear to 0
        pipe_barrier(PIPE_V);
        float val_a = (float)(coreId + 1) * 100.0f;
        AscendC::Duplicate(ba, val_a, 1);
        pipe_barrier(PIPE_V);
        AscendC::DataCopy(debugGm[coreId * 16], ba, 8);
        pipe_barrier(PIPE_V);

        // Sub-test B: Duplicate count=8 → writes to bufB (SEPARATE buffer)
        AscendC::Duplicate(bb, 0.0f, 8);
        pipe_barrier(PIPE_V);
        float val_b = (float)(coreId + 1) * 200.0f;
        AscendC::Duplicate(bb, val_b, 8);
        pipe_barrier(PIPE_V);
        AscendC::DataCopy(debugGm[coreId * 16 + 8], bb, 8);
        pipe_barrier(PIPE_V);
    }

private:
    int32_t coreId;
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECIN> bufA;
    AscendC::TBuf<AscendC::TPosition::VECIN> bufB;
    AscendC::GlobalTensor<float> debugGm;
};

extern "C" __global__ __aicore__ void fps_custom_dup1_check(
    GM_ADDR xyz, GM_ADDR idx, GM_ADDR results, GM_ADDR debug, GM_ADDR scratch)
{
    KernelDup1Check op;
    op.Init(xyz, idx, results, debug, scratch);
    op.Process();
}
