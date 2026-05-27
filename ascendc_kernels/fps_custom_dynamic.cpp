/**
 * Dynamic-shape multi-core FPS kernel.
 * Based on v3 debug (all fixes: GetValue→UB, SyncAll 3-arg, stageBuf).
 * Shape parameters read from TilingData GM buffer, not compile-time constants.
 */
#include "kernel_operator.h"
#include "fps_custom_dynamic.h"

constexpr int32_t BLOCK_SIZE = 64;
constexpr int32_t MAX_N = 1024;
constexpr int32_t MAX_NPOINTS = 512;
constexpr int32_t MAX_CORES = 8;
constexpr int32_t MAX_CHUNK = MAX_N / MAX_CORES;
constexpr int32_t MAX_BLOCKS = MAX_CHUNK / BLOCK_SIZE;

class KernelFpsDynamic {
public:
    __aicore__ inline void Init(GM_ADDR xyz, GM_ADDR idx, GM_ADDR results,
                                GM_ADDR scratch, GM_ADDR sync, GM_ADDR tiling)
    {
        // Step 1: Allocate tiling buffer first (before using it)
        pipe.InitBuffer(tilingTmpBuf, 8 * sizeof(int32_t));

        // Step 2: Read tiling data — DataCopy(GM→UB) then read UB
        AscendC::GlobalTensor<int32_t> tilingGm;
        tilingGm.SetGlobalBuffer((__gm__ int32_t *)tiling, 8);
        auto tilingLocal = tilingTmpBuf.Get<int32_t>();
        AscendC::DataCopy(tilingLocal, tilingGm, 8);
        pipe_barrier(PIPE_V);

        totalN = tilingLocal.GetValue(0);
        npoints = tilingLocal.GetValue(1);
        numCores = tilingLocal.GetValue(2);
        chunk = tilingLocal.GetValue(3);
        blocksPerCore = tilingLocal.GetValue(4);

        coreId = AscendC::GetBlockIdx();
        pointOffset = coreId * chunk;

        xyzGm.SetGlobalBuffer((__gm__ float *)xyz, 3 * totalN);
        idxGm.SetGlobalBuffer((__gm__ int32_t *)idx, npoints);
        resultsGm.SetGlobalBuffer((__gm__ float *)results, numCores * chunk);
        syncGm.SetGlobalBuffer((__gm__ int32_t *)sync, numCores * 8);

        // Step 3: Allocate remaining UB buffers based on tiling
        pipe.InitBuffer(xyzBuf, 3 * totalN * sizeof(float));
        pipe.InitBuffer(distBuf, chunk * sizeof(float));
        pipe.InitBuffer(idxBuf, npoints * sizeof(int32_t));
        pipe.InitBuffer(tmpBuf, BLOCK_SIZE * sizeof(float));
        pipe.InitBuffer(blkBuf, BLOCK_SIZE * sizeof(float));
        pipe.InitBuffer(redBuf, BLOCK_SIZE * sizeof(float));
        pipe.InitBuffer(syncBuf, numCores * 8 * sizeof(int32_t));
        pipe.InitBuffer(crossBuf, numCores * chunk * sizeof(float));
        pipe.InitBuffer(stageBuf, 8 * sizeof(float));
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

        AscendC::DataCopy(xyz, xyzGm, 3 * totalN);
        pipe_barrier(PIPE_V);

        AscendC::Duplicate(dist, 1e10f, chunk);
        if (coreId == 0) {
            AscendC::Duplicate(idxLocal, (int32_t)0, npoints);
        }
        pipe_barrier(PIPE_V);

        int32_t old = 0;

        for (int32_t j = 1; j < npoints; j++) {
            // Read current point coords from UB
            float x1 = xyz.GetValue(old);
            float y1 = xyz.GetValue(totalN + old);
            float z1 = xyz.GetValue(2 * totalN + old);

            for (int32_t b = 0; b < blocksPerCore; b++) {
                int32_t localBase = b * BLOCK_SIZE;
                int32_t globalBase = pointOffset + localBase;

                AscendC::Duplicate(tmp, x1, BLOCK_SIZE);
                AscendC::Sub(blk, xyz[globalBase], tmp, BLOCK_SIZE);
                AscendC::Mul(blk, blk, blk, BLOCK_SIZE);

                AscendC::Duplicate(tmp, y1, BLOCK_SIZE);
                AscendC::Sub(tmp, xyz[totalN + globalBase], tmp, BLOCK_SIZE);
                AscendC::Mul(tmp, tmp, tmp, BLOCK_SIZE);
                AscendC::Add(blk, blk, tmp, BLOCK_SIZE);

                AscendC::Duplicate(tmp, z1, BLOCK_SIZE);
                AscendC::Sub(tmp, xyz[2 * totalN + globalBase], tmp, BLOCK_SIZE);
                AscendC::Mul(tmp, tmp, tmp, BLOCK_SIZE);
                AscendC::Add(blk, blk, tmp, BLOCK_SIZE);

                AscendC::Min(dist[localBase], dist[localBase], blk, BLOCK_SIZE);
            }
            pipe_barrier(PIPE_V);

            // Block-level reduce max
            AscendC::WholeReduceMax<float>(red, dist, BLOCK_SIZE, blocksPerCore, 1, 1, 8);
            pipe_barrier(PIPE_V);

            // Local argmax scan from UB
            float localBestVal = -1.0f;
            int32_t localBestIdx = 0;
            for (int32_t b = 0; b < blocksPerCore; b++) {
                float val = red.GetValue(b * 2);
                if (val > localBestVal) {
                    localBestVal = val;
                    float idxFloat = red.GetValue(b * 2 + 1);
                    int32_t idxInBlock = *reinterpret_cast<uint32_t *>(&idxFloat);
                    localBestIdx = pointOffset + b * BLOCK_SIZE + idxInBlock;
                }
            }

            // Write local result via independent staging buffer
            auto stage = stageBuf.Get<float>();
            AscendC::Duplicate(stage, localBestVal, 1);
            AscendC::Duplicate(stage[1], *reinterpret_cast<float *>(&localBestIdx), 1);
            pipe_barrier(PIPE_V);
            AscendC::DataCopy(resultsGm[coreId * chunk], stage, 8);
            pipe_barrier(PIPE_V);

            // SyncAll (3-arg)
            AscendC::SyncAll(syncGm, syncLocal, numCores);

            // Cross-core argmax: DataCopy all results GM→UB, scan UB
            AscendC::DataCopy(cross, resultsGm, numCores * chunk);
            pipe_barrier(PIPE_V);

            float globalBestVal = -1.0f;
            int32_t globalBestIdx = 0;
            for (int32_t c = 0; c < numCores; c++) {
                float val = cross.GetValue(c * chunk);
                if (val > globalBestVal) {
                    globalBestVal = val;
                    float idxFloat = cross.GetValue(c * chunk + 1);
                    globalBestIdx = *reinterpret_cast<uint32_t *>(&idxFloat);
                }
            }

            if (coreId == 0) {
                idxLocal.SetValue(j, globalBestIdx);
            }
            old = globalBestIdx;
        }

        pipe_barrier(PIPE_V);
        if (coreId == 0) {
            AscendC::DataCopy(idxGm, idxLocal, npoints);
        }
        pipe_barrier(PIPE_V);
    }

private:
    // Tiling parameters (read from GM at runtime)
    int32_t totalN;
    int32_t npoints;
    int32_t numCores;
    int32_t chunk;
    int32_t blocksPerCore;
    int32_t coreId;
    int32_t pointOffset;

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECIN> xyzBuf;
    AscendC::TBuf<AscendC::TPosition::VECIN> distBuf;
    AscendC::TBuf<AscendC::TPosition::VECOUT> idxBuf;
    AscendC::TBuf<AscendC::TPosition::VECIN> tmpBuf;
    AscendC::TBuf<AscendC::TPosition::VECIN> blkBuf;
    AscendC::TBuf<AscendC::TPosition::VECIN> redBuf;
    AscendC::TBuf<AscendC::TPosition::VECIN> syncBuf;
    AscendC::TBuf<AscendC::TPosition::VECIN> crossBuf;
    AscendC::TBuf<AscendC::TPosition::VECIN> stageBuf;
    AscendC::TBuf<AscendC::TPosition::VECIN> tilingTmpBuf;

    AscendC::GlobalTensor<float> xyzGm;
    AscendC::GlobalTensor<int32_t> idxGm;
    AscendC::GlobalTensor<float> resultsGm;
    AscendC::GlobalTensor<int32_t> syncGm;
};

extern "C" __global__ __aicore__ void fps_custom_dynamic(
    GM_ADDR xyz, GM_ADDR idx, GM_ADDR results, GM_ADDR scratch, GM_ADDR sync, GM_ADDR tiling)
{
    KernelFpsDynamic op;
    op.Init(xyz, idx, results, scratch, sync, tiling);
    op.Process();
}
