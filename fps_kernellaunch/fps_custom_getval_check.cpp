/**
 * Test GetValue on GM with SEPARATE buffers — no buf reuse, no DMA race.
 * buf_src for scratch GM write, buf_out for debug output.
 * Sub-test A: GetValue from host GM, write result to buf_out → debug
 * Sub-test B: Duplicate(buf_src) → DataCopy(scratch) → GetValue(scratch) → Duplicate(buf_out) → DataCopy(debug)
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
        pipe.InitBuffer(bufSrc, 8 * sizeof(float));
        pipe.InitBuffer(bufOut, 8 * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        auto bSrc = bufSrc.Get<float>();
        auto bOut = bufOut.Get<float>();

        // Sub-test A: GetValue from host-written GM → write to bufOut (separate from bSrc)
        float val_a = inputGm.GetValue(coreId * PER_CORE_INPUT);
        AscendC::Duplicate(bOut, val_a, 8);
        pipe_barrier(PIPE_V);
        AscendC::DataCopy(debugGm[coreId * 16], bOut, 8);
        pipe_barrier(PIPE_V);

        // Sub-test B: DataCopy(bufSrc → scratch GM) then GetValue(scratch)
        // bufSrc is NEVER used for debug output, no DMA race
        float known_b = (float)(coreId + 1) * 100.0f;
        AscendC::Duplicate(bSrc, known_b, 8);
        pipe_barrier(PIPE_V);
        AscendC::DataCopy(scratchGm[coreId * PER_CORE_INPUT], bSrc, 8);
        pipe_barrier(PIPE_V);

        float val_b = scratchGm.GetValue(coreId * PER_CORE_INPUT);

        // Write val_b to debug via bufOut (reused, but previous DataCopy already sent)
        AscendC::Duplicate(bOut, val_b, 8);
        pipe_barrier(PIPE_V);
        AscendC::DataCopy(debugGm[coreId * 16 + 8], bOut, 8);
        pipe_barrier(PIPE_V);
    }

private:
    int32_t coreId;
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECIN> bufSrc;
    AscendC::TBuf<AscendC::TPosition::VECIN> bufOut;
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
