/**
 * Minimal test: does GetValue on GlobalTensor work on all 8 cores?
 * Two sub-tests per core:
 *   A: GetValue from host-written GM (via aclrtMemcpy H2D)
 *   B: DataCopy(UB→scratch GM) then GetValue from scratch GM
 * Results written via Duplicate(count=8)+DataCopy (proven reliable).
 */
#include "kernel_operator.h"

constexpr int32_t NUM_CORES = 8;
constexpr int32_t PER_CORE_INPUT = 8;

class KernelGetValCheck {
public:
    __aicore__ inline void Init(GM_ADDR xyz, GM_ADDR idx, GM_ADDR results,
                                GM_ADDR debug, GM_ADDR scratch)
    {
        coreId = AscendC::GetBlockIdx();
        inputGm.SetGlobalBuffer((__gm__ float *)xyz, NUM_CORES * PER_CORE_INPUT);
        scratchGm.SetGlobalBuffer((__gm__ float *)scratch, NUM_CORES * PER_CORE_INPUT);
        debugGm.SetGlobalBuffer((__gm__ float *)debug, NUM_CORES * 16);
        pipe.InitBuffer(buf, 8 * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        auto b = buf.Get<float>();

        // Sub-test A: GetValue from host-written GM
        float val_a = inputGm.GetValue(coreId * PER_CORE_INPUT);
        AscendC::Duplicate(b, val_a, 8);
        pipe_barrier(PIPE_V);
        AscendC::DataCopy(debugGm[coreId * 16], b, 8);
        pipe_barrier(PIPE_MTE3);  // wait DMA finish before reusing buf

        // Sub-test B: DataCopy(UB→scratch GM) then GetValue from scratch
        float known_b = (float)(coreId + 1) * 100.0f;
        AscendC::Duplicate(b, known_b, 8);
        pipe_barrier(PIPE_V);
        AscendC::DataCopy(scratchGm[coreId * PER_CORE_INPUT], b, 8);
        pipe_barrier(PIPE_MTE3);  // wait DMA finish before GetValue

        float val_b = scratchGm.GetValue(coreId * PER_CORE_INPUT);
        AscendC::Duplicate(b, val_b, 8);
        pipe_barrier(PIPE_V);
        AscendC::DataCopy(debugGm[coreId * 16 + 8], b, 8);
        pipe_barrier(PIPE_MTE3);
    }

private:
    int32_t coreId;
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECIN> buf;
    AscendC::GlobalTensor<float> inputGm;
    AscendC::GlobalTensor<float> scratchGm;
    AscendC::GlobalTensor<float> debugGm;
};

extern "C" __global__ __aicore__ void fps_custom_getval_check(
    GM_ADDR xyz, GM_ADDR idx, GM_ADDR results, GM_ADDR debug, GM_ADDR scratch)
{
    KernelGetValCheck op;
    op.Init(xyz, idx, results, debug, scratch);
    op.Process();
}
