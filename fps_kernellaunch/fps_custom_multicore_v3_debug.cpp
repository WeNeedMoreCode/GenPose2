/**
 * Debug version of v3 multi-core FPS kernel.
 * 8 checkpoints per (core, iter):
 *   CP1: dist_init    — dist[0] right after Duplicate(1e10), should be ~1e10
 *   CP2: dist_post    — dist[0] after distance update, before reduce
 *   CP3: cksum         — dist[0..3] sum after distance update
 *   CP4: red0_val      — WholeReduceMax block-0 value
 *   CP5: red0_idx      — WholeReduceMax block-0 index
 *   CP6: red1_val      — WholeReduceMax block-1 value
 *   CP7: red1_idx      — WholeReduceMax block-1 index
 *   CP8: localBestVal  — scalar scan result
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

class KernelFpsMultiCoreV3Debug {
public:
    __aicore__ inline void Init(GM_ADDR xyz, GM_ADDR idx, GM_ADDR results, GM_ADDR debug)
    {
        coreId = AscendC::GetBlockIdx();
        pointOffset = coreId * CHUNK;

        xyzGm.SetGlobalBuffer((__gm__ float *)xyz, 3 * N);
        idxGm.SetGlobalBuffer((__gm__ int32_t *)idx, NPOINTS);
        resultsGm.SetGlobalBuffer((__gm__ float *)results, NUM_CORES * CHUNK);
        debugGm.SetGlobalBuffer((__gm__ float *)debug, NUM_CORES * DIAG_FIELDS * DEBUG_ITERS);

        pipe.InitBuffer(xyzBuf, 3 * N * sizeof(float));
        pipe.InitBuffer(distBuf, CHUNK * sizeof(float));
        pipe.InitBuffer(idxBuf, NPOINTS * sizeof(int32_t));
        pipe.InitBuffer(tmpBuf, BLOCK_SIZE * sizeof(float));
        pipe.InitBuffer(blkBuf, BLOCK_SIZE * sizeof(float));
        pipe.InitBuffer(redBuf, BLOCKS_PER_CORE * 2 * sizeof(float));
        pipe.InitBuffer(inBuf, NUM_CORES * CHUNK * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        auto xyz = xyzBuf.Get<float>();
        auto dist = distBuf.Get<float>();
        auto idxLocal = idxBuf.Get<int32_t>();
        auto tmp = tmpBuf.Get<float>();
        auto blk = blkBuf.Get<float>();
        auto red = redBuf.Get<float>();
        auto inAll = inBuf.Get<float>();

        AscendC::DataCopy(xyz, xyzGm, 3 * N);
        pipe_barrier(PIPE_V);

        AscendC::Duplicate(dist, 1e10f, CHUNK);
        if (coreId == 0) {
            AscendC::Duplicate(idxLocal, (int32_t)0, NPOINTS);
        }
        pipe_barrier(PIPE_V);

        // CP1: verify Duplicate worked — dist[0] should be ~1e10
        float distInitVal = dist.GetValue(0);

        int32_t old = 0;

        for (int32_t j = 1; j < NPOINTS; j++) {
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

            // CP2/CP3: sample dist BEFORE reduce (reduce may clobber dist)
            float distPostVal = dist.GetValue(0);
            float cksum = dist.GetValue(0) + dist.GetValue(1) + dist.GetValue(2) + dist.GetValue(3);

            AscendC::WholeReduceMax<float>(red, dist, 64, BLOCKS_PER_CORE, 1, 1, 8);
            pipe_barrier(PIPE_V);

            // CP4-7: sample all reduce results
            float red0Val = red.GetValue(0);
            float red0Idx = red.GetValue(1);
            float red1Val = red.GetValue(2);
            float red1Idx = red.GetValue(3);

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

            // Write diagnostics (j=1 only for init check, j<=3 for full diag)
            if (j <= DEBUG_ITERS) {
                int32_t dOff = coreId * DIAG_FIELDS * DEBUG_ITERS + (j - 1) * DIAG_FIELDS;
                debugGm.SetValue(dOff + 0, (j == 1) ? distInitVal : distPostVal); // CP1 or CP2
                debugGm.SetValue(dOff + 1, distPostVal);   // CP2
                debugGm.SetValue(dOff + 2, cksum);          // CP3
                debugGm.SetValue(dOff + 3, red0Val);        // CP4
                debugGm.SetValue(dOff + 4, red0Idx);        // CP5
                debugGm.SetValue(dOff + 5, red1Val);        // CP6
                debugGm.SetValue(dOff + 6, red1Idx);        // CP7
                debugGm.SetValue(dOff + 7, localBestVal);   // CP8
            }

            // Encode local result into dist[0..1] using Duplicate (vector op)
            AscendC::Duplicate(dist, localBestVal, 1);
            AscendC::Duplicate(dist[1], *reinterpret_cast<float *>(&localBestIdx), 1);
            pipe_barrier(PIPE_V);

            // DataCopy entire dist chunk to GM (pure DMA)
            AscendC::DataCopy(resultsGm[coreId * CHUNK], dist, CHUNK);
            pipe_barrier(PIPE_V);

            AscendC::SyncAll();

            // DataCopy all cores' chunks from GM (pure DMA read)
            AscendC::DataCopy(inAll, resultsGm, NUM_CORES * CHUNK);
            pipe_barrier(PIPE_V);

            // Find global argmax
            float globalBestVal = -1.0f;
            int32_t globalBestIdx = 0;
            for (int32_t c = 0; c < NUM_CORES; c++) {
                float val = inAll.GetValue(c * CHUNK);
                if (val > globalBestVal) {
                    globalBestVal = val;
                    float idxFloat = inAll.GetValue(c * CHUNK + 1);
                    globalBestIdx = *reinterpret_cast<uint32_t *>(&idxFloat);
                }
            }

            // Debug: save post-sync global values
            if (j <= DEBUG_ITERS) {
                int32_t dbgOff = coreId * 4 * DEBUG_ITERS + (j - 1) * 4;
                debugGm.SetValue(dbgOff + 2, globalBestVal);
                debugGm.SetValue(dbgOff + 3, *reinterpret_cast<float *>(&globalBestIdx));
            }

            if (coreId == 0) {
                idxLocal.SetValue(j, globalBestIdx);
            }
            old = globalBestIdx;

            // Restore dist[0..1] for next iteration
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
    AscendC::TBuf<AscendC::TPosition::VECIN> inBuf;
    AscendC::GlobalTensor<float> xyzGm;
    AscendC::GlobalTensor<int32_t> idxGm;
    AscendC::GlobalTensor<float> resultsGm;
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
