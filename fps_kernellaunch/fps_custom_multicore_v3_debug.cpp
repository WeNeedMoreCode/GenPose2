/**
 * Debug v3 multi-core FPS kernel — overlap bug fixed, SetValue replaced.
 *
 * Changes from previous version:
 *   - Overlap bug fixed: single 10-field write per iter (stride 16 for DataCopy)
 *   - All debugGm.SetValue replaced with Duplicate + DataCopy (proven reliable)
 *   - Added globalBestVal/globalBestIdx fields (CP8, CP9)
 *
 * 10 checkpoints per (core, iter), stride 16:
 *   CP0: distInit    — 1e10 (j==1) or distPostVal (j>1)
 *   CP1: distPost    — dist[0] after distance update
 *   CP2: cksum       — dist[0..3] sum
 *   CP3: red0_val    — WholeReduceMax block-0 value
 *   CP4: red0_idx    — WholeReduceMax block-0 index
 *   CP5: red1_val    — WholeReduceMax block-1 value
 *   CP6: red1_idx    — WholeReduceMax block-1 index
 *   CP7: localBest   — scalar scan result
 *   CP8: globalBest  — cross-core argmax value
 *   CP9: globalIdx   — cross-core argmax index (as float)
 */
#include "kernel_operator.h"

constexpr int32_t N = 1024;
constexpr int32_t NPOINTS = 512;
constexpr int32_t BLOCK_SIZE = 64;
constexpr int32_t NUM_CORES = 8;
constexpr int32_t CHUNK = N / NUM_CORES;
constexpr int32_t BLOCKS_PER_CORE = CHUNK / BLOCK_SIZE;
constexpr int32_t DEBUG_ITERS = 3;
constexpr int32_t ACTUAL_FIELDS = 10;
constexpr int32_t DIAG_FIELDS = 16;  // DataCopy needs multiple of 8
constexpr int32_t SCRATCH_PER_CORE = 64;

class KernelFpsMultiCoreV3Debug {
public:
    __aicore__ inline void Init(GM_ADDR xyz, GM_ADDR idx, GM_ADDR results,
                                GM_ADDR debug, GM_ADDR scratch)
    {
        coreId = AscendC::GetBlockIdx();
        pointOffset = coreId * CHUNK;
        scratchOff = coreId * SCRATCH_PER_CORE;

        xyzGm.SetGlobalBuffer((__gm__ float *)xyz, 3 * N);
        idxGm.SetGlobalBuffer((__gm__ int32_t *)idx, NPOINTS);
        resultsGm.SetGlobalBuffer((__gm__ float *)results, NUM_CORES * CHUNK);
        debugGm.SetGlobalBuffer((__gm__ float *)debug, NUM_CORES * DIAG_FIELDS * DEBUG_ITERS);
        scratchGm.SetGlobalBuffer((__gm__ float *)scratch, NUM_CORES * SCRATCH_PER_CORE);

        pipe.InitBuffer(xyzBuf, 3 * N * sizeof(float));
        pipe.InitBuffer(distBuf, CHUNK * sizeof(float));
        pipe.InitBuffer(idxBuf, NPOINTS * sizeof(int32_t));
        pipe.InitBuffer(tmpBuf, BLOCK_SIZE * sizeof(float));
        pipe.InitBuffer(blkBuf, BLOCK_SIZE * sizeof(float));
        pipe.InitBuffer(redBuf, BLOCK_SIZE * sizeof(float));
        pipe.InitBuffer(dbgBuf, DIAG_FIELDS * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        auto xyz = xyzBuf.Get<float>();
        auto dist = distBuf.Get<float>();
        auto idxLocal = idxBuf.Get<int32_t>();
        auto tmp = tmpBuf.Get<float>();
        auto blk = blkBuf.Get<float>();
        auto red = redBuf.Get<float>();

        AscendC::DataCopy(xyz, xyzGm, 3 * N);
        pipe_barrier(PIPE_V);

        AscendC::Duplicate(dist, 1e10f, CHUNK);
        if (coreId == 0) {
            AscendC::Duplicate(idxLocal, (int32_t)0, NPOINTS);
        }
        pipe_barrier(PIPE_V);

        int32_t old = 0;

        for (int32_t j = 1; j < NPOINTS; j++) {
            float x1 = xyzGm.GetValue(old);
            float y1 = xyzGm.GetValue(N + old);
            float z1 = xyzGm.GetValue(2 * N + old);

            for (int32_t b = 0; b < BLOCKS_PER_CORE; b++) {
                int32_t localBase = b * BLOCK_SIZE;
                int32_t globalBase = pointOffset + localBase;

                AscendC::Duplicate(tmp, x1, BLOCK_SIZE);
                AscendC::Sub(blk, xyz[globalBase], tmp, BLOCK_SIZE);
                AscendC::Mul(blk, blk, blk, BLOCK_SIZE);

                AscendC::Duplicate(tmp, y1, BLOCK_SIZE);
                AscendC::Sub(tmp, xyz[N + globalBase], tmp, BLOCK_SIZE);
                AscendC::Mul(tmp, tmp, tmp, BLOCK_SIZE);
                AscendC::Add(blk, blk, tmp, BLOCK_SIZE);

                AscendC::Duplicate(tmp, z1, BLOCK_SIZE);
                AscendC::Sub(tmp, xyz[2 * N + globalBase], tmp, BLOCK_SIZE);
                AscendC::Mul(tmp, tmp, tmp, BLOCK_SIZE);
                AscendC::Add(blk, blk, tmp, BLOCK_SIZE);

                AscendC::Min(dist[localBase], dist[localBase], blk, BLOCK_SIZE);
            }
            pipe_barrier(PIPE_V);

            // dist → scratch GM for scalar reads
            AscendC::DataCopy(scratchGm[scratchOff], dist, 8);
            pipe_barrier(PIPE_V);

            float distPostVal = scratchGm.GetValue(scratchOff);
            float cksum = scratchGm.GetValue(scratchOff)
                        + scratchGm.GetValue(scratchOff + 1)
                        + scratchGm.GetValue(scratchOff + 2)
                        + scratchGm.GetValue(scratchOff + 3);

            AscendC::WholeReduceMax<float>(red, dist, 64, BLOCKS_PER_CORE, 1, 1, 8);
            pipe_barrier(PIPE_V);

            // reduce → scratch GM for scalar reads
            AscendC::DataCopy(scratchGm[scratchOff], red, 8);
            pipe_barrier(PIPE_V);

            float red0Val = scratchGm.GetValue(scratchOff);
            float red0Idx = scratchGm.GetValue(scratchOff + 1);
            float red1Val = scratchGm.GetValue(scratchOff + 2);
            float red1Idx = scratchGm.GetValue(scratchOff + 3);

            // Local argmax scan from scratch GM
            float localBestVal = -1.0f;
            int32_t localBestIdx = 0;
            for (int32_t b = 0; b < BLOCKS_PER_CORE; b++) {
                float val = scratchGm.GetValue(scratchOff + b * 2);
                if (val > localBestVal) {
                    localBestVal = val;
                    float idxFloat = scratchGm.GetValue(scratchOff + b * 2 + 1);
                    int32_t idxInBlock = *reinterpret_cast<uint32_t *>(&idxFloat);
                    localBestIdx = pointOffset + b * BLOCK_SIZE + idxInBlock;
                }
            }

            // Write local result to results GM for cross-core argmax
            AscendC::Duplicate(dist, localBestVal, 1);
            AscendC::Duplicate(dist[1], *reinterpret_cast<float *>(&localBestIdx), 1);
            pipe_barrier(PIPE_V);
            AscendC::DataCopy(resultsGm[coreId * CHUNK], dist, CHUNK);
            pipe_barrier(PIPE_V);

            AscendC::SyncAll();

            // Cross-core argmax from resultsGm
            float globalBestVal = -1.0f;
            int32_t globalBestIdx = 0;
            for (int32_t c = 0; c < NUM_CORES; c++) {
                float val = resultsGm.GetValue(c * CHUNK);
                if (val > globalBestVal) {
                    globalBestVal = val;
                    float idxFloat = resultsGm.GetValue(c * CHUNK + 1);
                    globalBestIdx = *reinterpret_cast<uint32_t *>(&idxFloat);
                }
            }

            // Debug output: Duplicate + DataCopy (no SetValue on GM, no overlap)
            if (j <= DEBUG_ITERS) {
                auto dbg = dbgBuf.Get<float>();
                AscendC::Duplicate(dbg[0], (j == 1) ? 1e10f : distPostVal, 1);
                AscendC::Duplicate(dbg[1], distPostVal, 1);
                AscendC::Duplicate(dbg[2], cksum, 1);
                AscendC::Duplicate(dbg[3], red0Val, 1);
                AscendC::Duplicate(dbg[4], red0Idx, 1);
                AscendC::Duplicate(dbg[5], red1Val, 1);
                AscendC::Duplicate(dbg[6], red1Idx, 1);
                AscendC::Duplicate(dbg[7], localBestVal, 1);
                AscendC::Duplicate(dbg[8], globalBestVal, 1);
                AscendC::Duplicate(dbg[9], *reinterpret_cast<float *>(&globalBestIdx), 1);
                pipe_barrier(PIPE_V);

                int32_t dOff = coreId * DIAG_FIELDS * DEBUG_ITERS + (j - 1) * DIAG_FIELDS;
                AscendC::DataCopy(debugGm[dOff], dbg, DIAG_FIELDS);
                pipe_barrier(PIPE_V);
            }

            if (coreId == 0) {
                idxLocal.SetValue(j, globalBestIdx);
            }
            old = globalBestIdx;

            AscendC::Duplicate(dist, 1e10f, 1);
            AscendC::Duplicate(dist[1], 1e10f, 1);
        }

        pipe_barrier(PIPE_V);
        if (coreId == 0) {
            AscendC::DataCopy(idxGm, idxLocal, NPOINTS);
        }
        pipe_barrier(PIPE_V);
    }

private:
    int32_t coreId;
    int32_t pointOffset;
    int32_t scratchOff;
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECIN> xyzBuf;
    AscendC::TBuf<AscendC::TPosition::VECIN> distBuf;
    AscendC::TBuf<AscendC::TPosition::VECOUT> idxBuf;
    AscendC::TBuf<AscendC::TPosition::VECIN> tmpBuf;
    AscendC::TBuf<AscendC::TPosition::VECIN> blkBuf;
    AscendC::TBuf<AscendC::TPosition::VECIN> redBuf;
    AscendC::TBuf<AscendC::TPosition::VECIN> dbgBuf;
    AscendC::GlobalTensor<float> xyzGm;
    AscendC::GlobalTensor<int32_t> idxGm;
    AscendC::GlobalTensor<float> resultsGm;
    AscendC::GlobalTensor<float> debugGm;
    AscendC::GlobalTensor<float> scratchGm;
};

extern "C" __global__ __aicore__ void fps_custom_mc_v3_dbg(
    GM_ADDR xyz, GM_ADDR idx, GM_ADDR results, GM_ADDR debug, GM_ADDR scratch)
{
    KernelFpsMultiCoreV3Debug op;
    op.Init(xyz, idx, results, debug, scratch);
    op.Process();
}
