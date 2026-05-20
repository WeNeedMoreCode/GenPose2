/**
 * Debug v3 multi-core FPS kernel — GetValue replaced with UB reads.
 *
 * Key changes from previous version:
 *   - All GlobalTensor::GetValue replaced with LocalTensor::GetValue (UB reads)
 *   - xyzGm.GetValue → xyz.GetValue (already in UB from initial DataCopy)
 *   - scratchGm.GetValue → dist.GetValue / red.GetValue (already in UB)
 *   - Cross-core argmax: resultsGm.GetValue → DataCopy(GM→UB) + UB scan
 *   - SyncAll() no-args → SyncAll(syncGm, syncLocal, NUM_CORES)
 *   - scratch GM writes removed (no longer needed for local reads)
 *
 * Based on verified findings:
 *   - GetValue(GlobalTensor) is unreliable on 310P3 multi-core
 *   - GetValue(LocalTensor) reads from UB (on-chip) — reliable
 *   - DataCopy(GM→UB) + SyncAll is reliable (8/8 PASS in getval test)
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
constexpr int32_t SYNCALL_PER_CORE = 8;

class KernelFpsMultiCoreV3Debug {
public:
    __aicore__ inline void Init(GM_ADDR xyz, GM_ADDR idx, GM_ADDR results,
                                GM_ADDR debug, GM_ADDR scratch, GM_ADDR sync)
    {
        coreId = AscendC::GetBlockIdx();
        pointOffset = coreId * CHUNK;

        xyzGm.SetGlobalBuffer((__gm__ float *)xyz, 3 * N);
        idxGm.SetGlobalBuffer((__gm__ int32_t *)idx, NPOINTS);
        resultsGm.SetGlobalBuffer((__gm__ float *)results, NUM_CORES * CHUNK);
        debugGm.SetGlobalBuffer((__gm__ float *)debug, NUM_CORES * DIAG_FIELDS * DEBUG_ITERS);
        syncGm.SetGlobalBuffer((__gm__ int32_t *)sync, NUM_CORES * SYNCALL_PER_CORE);

        pipe.InitBuffer(xyzBuf, 3 * N * sizeof(float));
        pipe.InitBuffer(distBuf, CHUNK * sizeof(float));
        pipe.InitBuffer(idxBuf, NPOINTS * sizeof(int32_t));
        pipe.InitBuffer(tmpBuf, BLOCK_SIZE * sizeof(float));
        pipe.InitBuffer(blkBuf, BLOCK_SIZE * sizeof(float));
        pipe.InitBuffer(redBuf, BLOCK_SIZE * sizeof(float));
        pipe.InitBuffer(dbgBuf, DIAG_FIELDS * sizeof(float));
        pipe.InitBuffer(syncBuf, NUM_CORES * SYNCALL_PER_CORE * sizeof(int32_t));
        pipe.InitBuffer(crossBuf, NUM_CORES * 8 * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        auto xyz = xyzBuf.Get<float>();
        auto dist = distBuf.Get<float>();
        auto idxLocal = idxBuf.Get<int32_t>();
        auto tmp = tmpBuf.Get<float>();
        auto blk = blkBuf.Get<float>();
        auto red = redBuf.Get<float>();
        auto syncLocal = syncBuf.Get<int32_t>();
        auto cross = crossBuf.Get<float>();

        AscendC::DataCopy(xyz, xyzGm, 3 * N);
        pipe_barrier(PIPE_V);

        AscendC::Duplicate(dist, 1e10f, CHUNK);
        if (coreId == 0) {
            AscendC::Duplicate(idxLocal, (int32_t)0, NPOINTS);
        }
        pipe_barrier(PIPE_V);

        int32_t old = 0;

        for (int32_t j = 1; j < NPOINTS; j++) {
            // Read current point coords from UB (already copied at init)
            float x1 = xyz.GetValue(old);
            float y1 = xyz.GetValue(N + old);
            float z1 = xyz.GetValue(2 * N + old);

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

            // Read dist from UB directly (no scratch GM round-trip)
            float distPostVal = dist.GetValue(0);
            float cksum = dist.GetValue(0) + dist.GetValue(1)
                        + dist.GetValue(2) + dist.GetValue(3);

            AscendC::WholeReduceMax<float>(red, dist, 64, BLOCKS_PER_CORE, 1, 1, 8);
            pipe_barrier(PIPE_V);

            // Read reduce results from UB directly
            float red0Val = red.GetValue(0);
            float red0Idx = red.GetValue(1);
            float red1Val = red.GetValue(2);
            float red1Idx = red.GetValue(3);

            // Local argmax scan from UB
            float localBestVal = -1.0f;
            int32_t localBestIdx = 0;
            for (int32_t b = 0; b < BLOCKS_PER_CORE; b++) {
                float val = red.GetValue(b * 2);
                if (val > localBestVal) {
                    localBestVal = val;
                    float idxFloat = red.GetValue(b * 2 + 1);
                    int32_t idxInBlock = *reinterpret_cast<uint32_t *>(&idxFloat);
                    localBestIdx = pointOffset + b * BLOCK_SIZE + idxInBlock;
                }
            }

            // Write local result to results GM
            AscendC::Duplicate(dist, localBestVal, 1);
            AscendC::Duplicate(dist[1], *reinterpret_cast<float *>(&localBestIdx), 1);
            pipe_barrier(PIPE_V);
            AscendC::DataCopy(resultsGm[coreId * CHUNK], dist, CHUNK);
            pipe_barrier(PIPE_V);

            // SyncAll: ensure all cores' writes to resultsGm are visible
            AscendC::SyncAll(syncGm, syncLocal, NUM_CORES);

            // Cross-core argmax: DataCopy(GM→UB) then scan UB
            for (int32_t c = 0; c < NUM_CORES; c++) {
                AscendC::DataCopy(cross + c * 8, resultsGm[c * CHUNK], 8);
            }
            pipe_barrier(PIPE_V);

            float globalBestVal = -1.0f;
            int32_t globalBestIdx = 0;
            for (int32_t c = 0; c < NUM_CORES; c++) {
                float val = cross.GetValue(c * 8);
                if (val > globalBestVal) {
                    globalBestVal = val;
                    float idxFloat = cross.GetValue(c * 8 + 1);
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
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECIN> xyzBuf;
    AscendC::TBuf<AscendC::TPosition::VECIN> distBuf;
    AscendC::TBuf<AscendC::TPosition::VECOUT> idxBuf;
    AscendC::TBuf<AscendC::TPosition::VECIN> tmpBuf;
    AscendC::TBuf<AscendC::TPosition::VECIN> blkBuf;
    AscendC::TBuf<AscendC::TPosition::VECIN> redBuf;
    AscendC::TBuf<AscendC::TPosition::VECIN> dbgBuf;
    AscendC::TBuf<AscendC::TPosition::VECIN> syncBuf;
    AscendC::TBuf<AscendC::TPosition::VECIN> crossBuf;
    AscendC::GlobalTensor<float> xyzGm;
    AscendC::GlobalTensor<int32_t> idxGm;
    AscendC::GlobalTensor<float> resultsGm;
    AscendC::GlobalTensor<float> debugGm;
    AscendC::GlobalTensor<int32_t> syncGm;
};

extern "C" __global__ __aicore__ void fps_custom_mc_v3_dbg(
    GM_ADDR xyz, GM_ADDR idx, GM_ADDR results, GM_ADDR debug, GM_ADDR scratch, GM_ADDR sync)
{
    KernelFpsMultiCoreV3Debug op;
    op.Init(xyz, idx, results, debug, scratch, sync);
    op.Process();
}
