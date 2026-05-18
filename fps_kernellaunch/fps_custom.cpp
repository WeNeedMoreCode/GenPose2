/**
 * AscendC SIMD FPS (Furthest Point Sampling) kernel — optimized
 * Input:  xyz [3*N] float32 (transposed: x[0:N], y[N:2N], z[2N:3N])
 * Output: idx [NPOINTS] int32
 * Fixed params: N=1024, NPOINTS=512, blockDim=1
 *
 * Optimizations vs v1:
 *   - TBuf instead of TQue (no EnQue/DeQue overhead in inner loop)
 *   - Block-level WholeReduceMax for argmax (vector reduction, not scalar scan)
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

        pipe.InitBuffer(xyzBuf, 3 * N * sizeof(float));
        pipe.InitBuffer(distBuf, N * sizeof(float));
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
        AscendC::Duplicate(dist, 1e10f, N);
        AscendC::Duplicate(idxLocal, (int32_t)0, NPOINTS);
        pipe_barrier(PIPE_V);

        int32_t old = 0;

        for (int32_t j = 1; j < NPOINTS; j++) {
            float x1 = xyz.GetValue(old);
            float y1 = xyz.GetValue(N + old);
            float z1 = xyz.GetValue(2 * N + old);

            for (int32_t b = 0; b < NUM_BLOCKS; b++) {
                int32_t base = b * BLOCK_SIZE;

                AscendC::Duplicate(tmp, x1, BLOCK_SIZE);
                AscendC::Sub(blk, xyz[base], tmp, BLOCK_SIZE);
                AscendC::Mul(blk, blk, blk, BLOCK_SIZE);

                AscendC::Duplicate(tmp, y1, BLOCK_SIZE);
                AscendC::Sub(tmp, xyz[N + base], tmp, BLOCK_SIZE);
                AscendC::Mul(tmp, tmp, tmp, BLOCK_SIZE);
                AscendC::Add(blk, blk, tmp, BLOCK_SIZE);

                AscendC::Duplicate(tmp, z1, BLOCK_SIZE);
                AscendC::Sub(tmp, xyz[2 * N + base], tmp, BLOCK_SIZE);
                AscendC::Mul(tmp, tmp, tmp, BLOCK_SIZE);
                AscendC::Add(blk, blk, tmp, BLOCK_SIZE);

                AscendC::Min(dist[base], dist[base], blk, BLOCK_SIZE);
            }
            pipe_barrier(PIPE_V);

            // Block-level argmax: WholeReduceMax returns [value, index] per block
            float bestVal = -1.0f;
            int32_t bestIdx = 0;
            for (int32_t b = 0; b < NUM_BLOCKS; b++) {
                int32_t base = b * BLOCK_SIZE;
                AscendC::WholeReduceMax<float>(red, dist[base], 64, 1, 1, 1, 8);
                pipe_barrier(PIPE_V);
                float val = red.GetValue(0);
                if (val > bestVal) {
                    bestVal = val;
                    float idxFloat = red.GetValue(1);
                    bestIdx = base + *reinterpret_cast<uint32_t *>(&idxFloat);
                }
            }

            idxLocal.SetValue(j, bestIdx);
            old = bestIdx;
        }

        pipe_barrier(PIPE_V);
        AscendC::DataCopy(idxGm, idxLocal, NPOINTS);
        pipe_barrier(PIPE_V);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECIN> xyzBuf;
    AscendC::TBuf<AscendC::TPosition::VECIN> distBuf;
    AscendC::TBuf<AscendC::TPosition::VECOUT> idxBuf;
    AscendC::TBuf<AscendC::TPosition::VECIN> tmpBuf;
    AscendC::TBuf<AscendC::TPosition::VECIN> blkBuf;
    AscendC::TBuf<AscendC::TPosition::VECIN> redBuf;
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
