/**
 * Debug version of v3 multi-core FPS kernel — ALL SCALAR READS VIA GM.
 * No GetValue on vector-written UB; all scalar reads go through GlobalMemory.
 * 8 checkpoints per (core, iter):
 *   CP1: dist_init    — set to 1e10 (known constant, no UB read)
 *   CP2: dist_post    — dist[0] after distance update (via scratch GM)
 *   CP3: cksum         — dist[0..3] sum after distance update (via scratch GM)
 *   CP4: red0_val      — WholeReduceMax block-0 value (via scratch GM)
 *   CP5: red0_idx      — WholeReduceMax block-0 index (via scratch GM)
 *   CP6: red1_val      — WholeReduceMax block-1 value (via scratch GM)
 *   CP7: red1_idx      — WholeReduceMax block-1 index (via scratch GM)
 *   CP8: localBestVal  — scalar scan result (via scratch GM)
 */
#include "kernel_operator.h"

constexpr int32_t N = 1024;
constexpr int32_t NPOINTS = 512;
constexpr int32_t BLOCK_SIZE = 64;
constexpr int32_t NUM_CORES = 8;
constexpr int32_t CHUNK = N / NUM_CORES;
constexpr int32_t BLOCKS_PER_CORE = CHUNK / BLOCK_SIZE;
constexpr int32_t DEBUG_ITERS = 3;
constexpr int32_t DIAG_FIELDS = 8;
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

        float distInitVal = 1e10f;

        int32_t old = 0;

        for (int32_t j = 1; j < NPOINTS; j++) {
            // Read reference point coords from GM (not UB)
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

            // CP2/CP3: copy dist to scratch GM, read from GM
            AscendC::DataCopy(scratchGm[scratchOff], dist, 8);
            pipe_barrier(PIPE_V);

            float distPostVal = scratchGm.GetValue(scratchOff);
            float cksum = scratchGm.GetValue(scratchOff)
                        + scratchGm.GetValue(scratchOff + 1)
                        + scratchGm.GetValue(scratchOff + 2)
                        + scratchGm.GetValue(scratchOff + 3);

            AscendC::WholeReduceMax<float>(red, dist, 64, BLOCKS_PER_CORE, 1, 1, 8);
            pipe_barrier(PIPE_V);

            // CP4-7: copy reduce results to scratch GM, read from GM
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

            if (j <= DEBUG_ITERS) {
                int32_t dOff = coreId * DIAG_FIELDS * DEBUG_ITERS + (j - 1) * DIAG_FIELDS;
                debugGm.SetValue(dOff + 0, (j == 1) ? distInitVal : distPostVal);
                debugGm.SetValue(dOff + 1, distPostVal);
                debugGm.SetValue(dOff + 2, cksum);
                debugGm.SetValue(dOff + 3, red0Val);
                debugGm.SetValue(dOff + 4, red0Idx);
                debugGm.SetValue(dOff + 5, red1Val);
                debugGm.SetValue(dOff + 6, red1Idx);
                debugGm.SetValue(dOff + 7, localBestVal);
            }

            AscendC::Duplicate(dist, localBestVal, 1);
            AscendC::Duplicate(dist[1], *reinterpret_cast<float *>(&localBestIdx), 1);
            pipe_barrier(PIPE_V);

            AscendC::DataCopy(resultsGm[coreId * CHUNK], dist, CHUNK);
            pipe_barrier(PIPE_V);

            AscendC::SyncAll();

            // Cross-core argmax: read directly from resultsGm (no inAll UB)
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

            if (j <= DEBUG_ITERS) {
                int32_t dbgOff = coreId * 4 * DEBUG_ITERS + (j - 1) * 4;
                debugGm.SetValue(dbgOff + 2, globalBestVal);
                debugGm.SetValue(dbgOff + 3, *reinterpret_cast<float *>(&globalBestIdx));
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
