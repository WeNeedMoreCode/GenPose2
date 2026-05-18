/**
 * AscendC Multi-Core FPS v2: DataCopy-based cross-core communication.
 *
 * Fix: replace SetValue() with DataCopy for cross-core argmax results.
 * SetValue() uses scalar GM writes that may stay in core-local write buffer;
 * DataCopy() uses DMA engine which writes directly to GM, visible to all cores.
 *
 * Each core writes 8 floats (32-byte aligned) to GM via DataCopy.
 * All cores read the full results buffer via DataCopy after SyncAll.
 */
#include "kernel_operator.h"

constexpr int32_t N = 1024;
constexpr int32_t NPOINTS = 512;
constexpr int32_t BLOCK_SIZE = 64;
constexpr int32_t NUM_CORES = 8;
constexpr int32_t CHUNK = N / NUM_CORES;
constexpr int32_t BLOCKS_PER_CORE = CHUNK / BLOCK_SIZE;
constexpr int32_t RES_STRIDE = 8;  // DataCopy alignment: 8 floats = 32 bytes

class KernelFpsMultiCoreV2 {
public:
    __aicore__ inline void Init(GM_ADDR xyz, GM_ADDR idx, GM_ADDR results)
    {
        coreId = AscendC::GetBlockIdx();
        pointOffset = coreId * CHUNK;

        xyzGm.SetGlobalBuffer((__gm__ float *)xyz, 3 * N);
        idxGm.SetGlobalBuffer((__gm__ int32_t *)idx, NPOINTS);
        resultsGm.SetGlobalBuffer((__gm__ float *)results, NUM_CORES * RES_STRIDE);

        pipe.InitBuffer(xyzBuf, 3 * N * sizeof(float));
        pipe.InitBuffer(distBuf, CHUNK * sizeof(float));
        pipe.InitBuffer(idxBuf, NPOINTS * sizeof(int32_t));
        pipe.InitBuffer(tmpBuf, BLOCK_SIZE * sizeof(float));
        pipe.InitBuffer(blkBuf, BLOCK_SIZE * sizeof(float));
        pipe.InitBuffer(redBuf, BLOCKS_PER_CORE * 2 * sizeof(float));
        pipe.InitBuffer(resOutBuf, RES_STRIDE * sizeof(float));
        pipe.InitBuffer(resInBuf, NUM_CORES * RES_STRIDE * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        auto xyz = xyzBuf.Get<float>();
        auto dist = distBuf.Get<float>();
        auto idxLocal = idxBuf.Get<int32_t>();
        auto tmp = tmpBuf.Get<float>();
        auto blk = blkBuf.Get<float>();
        auto red = redBuf.Get<float>();
        auto resOut = resOutBuf.Get<float>();
        auto resIn = resInBuf.Get<float>();

        AscendC::DataCopy(xyz, xyzGm, 3 * N);
        pipe_barrier(PIPE_V);

        AscendC::Duplicate(dist, 1e10f, CHUNK);
        if (coreId == 0) {
            AscendC::Duplicate(idxLocal, (int32_t)0, NPOINTS);
        }
        pipe_barrier(PIPE_V);

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

            AscendC::WholeReduceMax<float>(red, dist, 64, BLOCKS_PER_CORE, 1, 1, 8);
            pipe_barrier(PIPE_V);

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

            // Write local result to GM via DataCopy (DMA, visible to all cores)
            AscendC::Duplicate(resOut, 0.0f, RES_STRIDE);
            resOut.SetValue(0, localBestVal);
            resOut.SetValue(1, *reinterpret_cast<float *>(&localBestIdx));
            AscendC::DataCopy(resultsGm[coreId * RES_STRIDE], resOut, RES_STRIDE);
            pipe_barrier(PIPE_V);

            AscendC::SyncAll();

            // Read all cores' results via DataCopy (DMA read from GM)
            AscendC::DataCopy(resIn, resultsGm, NUM_CORES * RES_STRIDE);
            pipe_barrier(PIPE_V);

            float globalBestVal = -1.0f;
            int32_t globalBestIdx = 0;
            for (int32_t c = 0; c < NUM_CORES; c++) {
                float val = resIn.GetValue(c * RES_STRIDE);
                if (val > globalBestVal) {
                    globalBestVal = val;
                    float idxFloat = resIn.GetValue(c * RES_STRIDE + 1);
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
    AscendC::TBuf<AscendC::TPosition::VECOUT> resOutBuf;
    AscendC::TBuf<AscendC::TPosition::VECIN> resInBuf;

    AscendC::GlobalTensor<float> xyzGm;
    AscendC::GlobalTensor<int32_t> idxGm;
    AscendC::GlobalTensor<float> resultsGm;
};

extern "C" __global__ __aicore__ void fps_custom_multicore_v2(GM_ADDR xyz, GM_ADDR idx, GM_ADDR results)
{
    KernelFpsMultiCoreV2 op;
    op.Init(xyz, idx, results);
    op.Process();
}

#ifndef ASCENDC_CPU_DEBUG
void fps_custom_multicore_v2_do(uint32_t blockDim, void *stream, uint8_t *xyz, uint8_t *idx, uint8_t *results)
{
    fps_custom_multicore_v2<<<blockDim, nullptr, stream>>>(xyz, idx, results);
}
#endif
