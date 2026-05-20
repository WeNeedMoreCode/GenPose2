/**
 * Test GetValue on GM + DataCopy(GM→UB) readback — WITH SyncAll.
 * Three sub-tests per core, each 8 floats in debug output (total 24 per core):
 *   A: GetValue from host-written GM
 *   B: GetValue from kernel-written scratch GM (after SyncAll)
 *   C: DataCopy(scratch GM→UB) readback (after SyncAll)
 *
 * Key change vs previous version: added SyncAll between write and read,
 * removed harmful pipe_barrier(PIPE_MTE3).
 * Pattern from official reduce_min sample: SyncAll(syncGM, syncLocal).
 */
#include "kernel_operator.h"

constexpr int32_t NUM_CORES = 8;
constexpr int32_t PER_CORE_INPUT = 8;
constexpr int32_t PER_CORE_DEBUG = 24;  // 3 sub-tests × 8 floats
constexpr int32_t SYNCALL_PER_CORE = 8; // SyncAll needs 8 int32_t per core

class KernelGetValCheck {
public:
    __aicore__ inline void Init(GM_ADDR xyz, GM_ADDR idx, GM_ADDR results,
                                GM_ADDR debug, GM_ADDR scratch, GM_ADDR sync)
    {
        coreId = AscendC::GetBlockIdx();
        inputGm.SetGlobalBuffer((__gm__ float *)xyz, NUM_CORES * PER_CORE_INPUT);
        scratchGm.SetGlobalBuffer((__gm__ float *)scratch, NUM_CORES * PER_CORE_INPUT);
        debugGm.SetGlobalBuffer((__gm__ float *)debug, NUM_CORES * PER_CORE_DEBUG);
        syncGm.SetGlobalBuffer((__gm__ int32_t *)sync, NUM_CORES * SYNCALL_PER_CORE);

        pipe.InitBuffer(bufA, 8 * sizeof(float));
        pipe.InitBuffer(bufB, 8 * sizeof(float));
        pipe.InitBuffer(bufC, 8 * sizeof(float));
        pipe.InitBuffer(syncBuf, NUM_CORES * SYNCALL_PER_CORE * sizeof(int32_t));
    }

    __aicore__ inline void Process()
    {
        auto bA = bufA.Get<float>();
        auto bB = bufB.Get<float>();
        auto bC = bufC.Get<float>();
        auto syncLocal = syncBuf.Get<int32_t>();
        int32_t base = coreId * PER_CORE_DEBUG;

        // --- Sub-test A: GetValue from host-written GM ---
        float val_a = inputGm.GetValue(coreId * PER_CORE_INPUT);
        AscendC::Duplicate(bA, val_a, 8);
        pipe_barrier(PIPE_V);
        AscendC::DataCopy(debugGm[base], bA, 8);
        pipe_barrier(PIPE_V);

        // --- Write known values to scratch GM ---
        float known = (float)(coreId + 1) * 100.0f;
        AscendC::Duplicate(bB, known, 8);
        pipe_barrier(PIPE_V);
        AscendC::DataCopy(scratchGm[coreId * PER_CORE_INPUT], bB, 8);
        pipe_barrier(PIPE_V);

        // --- SyncAll: ensure all cores' writes are visible ---
        AscendC::SyncAll(syncGm, syncLocal);

        // --- Sub-test B: GetValue from scratch GM (after SyncAll) ---
        float val_b = scratchGm.GetValue(coreId * PER_CORE_INPUT);
        AscendC::Duplicate(bA, val_b, 8);
        pipe_barrier(PIPE_V);
        AscendC::DataCopy(debugGm[base + 8], bA, 8);
        pipe_barrier(PIPE_V);

        // --- Sub-test C: DataCopy(scratch GM→UB) readback (after SyncAll) ---
        AscendC::DataCopy(bC, scratchGm[coreId * PER_CORE_INPUT], 8);
        pipe_barrier(PIPE_V);
        AscendC::DataCopy(debugGm[base + 16], bC, 8);
        pipe_barrier(PIPE_V);
    }

private:
    int32_t coreId;
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECIN> bufA;
    AscendC::TBuf<AscendC::TPosition::VECIN> bufB;
    AscendC::TBuf<AscendC::TPosition::VECIN> bufC;
    AscendC::TBuf<AscendC::TPosition::VECIN> syncBuf;
    AscendC::GlobalTensor<float> inputGm;
    AscendC::GlobalTensor<float> scratchGm;
    AscendC::GlobalTensor<float> debugGm;
    AscendC::GlobalTensor<int32_t> syncGm;
};

extern "C" __global__ __aicore__ void fps_custom_getval_check(
    GM_ADDR xyz, GM_ADDR idx, GM_ADDR results, GM_ADDR debug, GM_ADDR scratch, GM_ADDR sync)
{
    KernelGetValCheck op;
    op.Init(xyz, idx, results, debug, scratch, sync);
    op.Process();
}
