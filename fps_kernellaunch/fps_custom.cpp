/**
 * AscendC SIMD FPS (Furthest Point Sampling) kernel
 * Input:  xyz [3*N] float32 (transposed: x[0:N], y[N:2N], z[2N:3N])
 * Output: idx [NPOINTS] int32
 * Fixed params: N=1024, NPOINTS=512, blockDim=1
 */
#include "kernel_operator.h"

constexpr int32_t N = 1024;
constexpr int32_t NPOINTS = 512;
constexpr int32_t BLOCK_SIZE = 64;
constexpr int32_t NUM_BLOCKS = N / BLOCK_SIZE;

class KernelFps {
public:
    __aicore__ inline void Init(GM_ADDR xyz, GM_ADDR idx)
    {
        xyzGm.SetGlobalBuffer((__gm__ float *)xyz, 3 * N);
        idxGm.SetGlobalBuffer((__gm__ int32_t *)idx, NPOINTS);

        pipe.InitBuffer(xyzQueue, 1, 3 * N * sizeof(float));
        pipe.InitBuffer(distQueue, 1, N * sizeof(float));
        pipe.InitBuffer(idxQueue, 1, NPOINTS * sizeof(int32_t));
        pipe.InitBuffer(tmpQueue, 1, BLOCK_SIZE * sizeof(float));
        pipe.InitBuffer(blkQueue, 1, BLOCK_SIZE * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        // Load xyz from GM
        auto xyz = xyzQueue.AllocTensor<float>();
        AscendC::DataCopy(xyz, xyzGm, 3 * N);
        xyzQueue.EnQue(xyz);

        // Init distance
        auto dist = distQueue.AllocTensor<float>();
        AscendC::Duplicate(dist, 1e10f, N);
        distQueue.EnQue(dist);

        // Init idx to zeros (idx[0] = 0)
        auto idxLocal = idxQueue.AllocTensor<int32_t>();
        AscendC::Duplicate(idxLocal, (int32_t)0, NPOINTS);
        idxQueue.EnQue(idxLocal);

        int32_t old = 0;

        for (int32_t j = 1; j < NPOINTS; j++) {
            // Read centroid (sync via DeQue)
            xyz = xyzQueue.DeQue<float>();
            float x1 = xyz.GetValue(old);
            float y1 = xyz.GetValue(N + old);
            float z1 = xyz.GetValue(2 * N + old);
            xyzQueue.EnQue(xyz);

            // Distance computation in blocks
            for (int32_t b = 0; b < NUM_BLOCKS; b++) {
                int32_t base = b * BLOCK_SIZE;

                auto tmp = tmpQueue.AllocTensor<float>();
                auto blk = blkQueue.AllocTensor<float>();
                xyz = xyzQueue.DeQue<float>();

                // dx^2
                AscendC::Duplicate(tmp, x1, BLOCK_SIZE);
                AscendC::Sub(blk, xyz[base], tmp, BLOCK_SIZE);
                AscendC::Mul(blk, blk, blk, BLOCK_SIZE);

                // += dy^2
                AscendC::Duplicate(tmp, y1, BLOCK_SIZE);
                AscendC::Sub(tmp, xyz[N + base], tmp, BLOCK_SIZE);
                AscendC::Mul(tmp, tmp, tmp, BLOCK_SIZE);
                AscendC::Add(blk, blk, tmp, BLOCK_SIZE);

                // += dz^2
                AscendC::Duplicate(tmp, z1, BLOCK_SIZE);
                AscendC::Sub(tmp, xyz[2 * N + base], tmp, BLOCK_SIZE);
                AscendC::Mul(tmp, tmp, tmp, BLOCK_SIZE);
                AscendC::Add(blk, blk, tmp, BLOCK_SIZE);

                xyzQueue.EnQue(xyz);

                // min update (sync via DeQue)
                dist = distQueue.DeQue<float>();
                AscendC::Min(dist[base], dist[base], blk, BLOCK_SIZE);
                distQueue.EnQue(dist);

                // Release temp buffers
                tmpQueue.EnQue(tmp);
                tmpQueue.DeQue<float>();
                tmpQueue.FreeTensor(tmp);

                blkQueue.EnQue(blk);
                blkQueue.DeQue<float>();
                blkQueue.FreeTensor(blk);
            }

            // Argmax (sync via DeQue ensures all Min ops complete)
            dist = distQueue.DeQue<float>();
            float bestVal = -1.0f;
            int32_t bestIdx = 0;
            for (int32_t k = 0; k < N; k++) {
                float val = dist.GetValue(k);
                if (val > bestVal) {
                    bestVal = val;
                    bestIdx = k;
                }
            }
            distQueue.EnQue(dist);

            // Write idx[j]
            idxLocal = idxQueue.DeQue<int32_t>();
            idxLocal.SetValue(j, bestIdx);
            idxQueue.EnQue(idxLocal);

            old = bestIdx;
        }

        // Write back idx to GM
        idxLocal = idxQueue.DeQue<int32_t>();
        AscendC::DataCopy(idxGm, idxLocal, NPOINTS);
        idxQueue.FreeTensor(idxLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> xyzQueue, distQueue, tmpQueue, blkQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> idxQueue;
    AscendC::GlobalTensor<float> xyzGm;
    AscendC::GlobalTensor<int32_t> idxGm;
};

extern "C" __global__ __aicore__ void fps_custom(GM_ADDR xyz, GM_ADDR idx)
{
    KernelFps op;
    op.Init(xyz, idx);
    op.Process();
}

#ifndef ASCENDC_CPU_DEBUG
void fps_custom_do(uint32_t blockDim, void *stream, uint8_t *xyz, uint8_t *idx)
{
    fps_custom<<<blockDim, nullptr, stream>>>(xyz, idx);
}
#endif
