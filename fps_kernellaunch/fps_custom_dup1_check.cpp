/**
 * Minimal test: does Duplicate(buf, val, 1) work on all 8 cores?
 * Each core writes 2 values:
 *   - val_a using Duplicate(count=1) → output[coreId*16 + 0..7]
 *   - val_b using Duplicate(count=8) → output[coreId*16 + 8..15]
 * Host checks: val_a and val_b both correct? If count=1 fails, val_a=0.
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
        pipe.InitBuffer(buf, 8 * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        auto b = buf.Get<float>();

        // Sub-test A: Duplicate count=1
        AscendC::Duplicate(b, 0.0f, 8);  // clear
        pipe_barrier(PIPE_V);
        float val_a = (float)(coreId + 1) * 100.0f;
        AscendC::Duplicate(b, val_a, 1);
        pipe_barrier(PIPE_V);
        AscendC::DataCopy(debugGm[coreId * 16], b, 8);
        pipe_barrier(PIPE_V);

        // Sub-test B: Duplicate count=8
        AscendC::Duplicate(b, 0.0f, 8);  // clear
        pipe_barrier(PIPE_V);
        float val_b = (float)(coreId + 1) * 200.0f;
        AscendC::Duplicate(b, val_b, 8);
        pipe_barrier(PIPE_V);
        AscendC::DataCopy(debugGm[coreId * 16 + 8], b, 8);
        pipe_barrier(PIPE_V);
    }

private:
    int32_t coreId;
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECIN> buf;
    AscendC::GlobalTensor<float> debugGm;
};

extern "C" __global__ __aicore__ void fps_custom_dup1_check(
    GM_ADDR xyz, GM_ADDR idx, GM_ADDR results, GM_ADDR debug, GM_ADDR scratch)
{
    KernelDup1Check op;
    op.Init(xyz, idx, results, debug, scratch);
    op.Process();
}
