/**
 * AscendC multi-core Ball Query kernel (dynamic shape).
 *
 * Equivalent to CUDA ball_query_kernel_fast:
 *   For each (b, m), find up to nsample points in xyz[b, :, :] within radius of new_xyz[b, m, :].
 *   Output indices are in INDEX ORDER (not distance order).
 *   If fewer than nsample points found, fill remaining with first found index.
 *
 * No cross-core synchronization needed — each query is independent.
 * Multi-core strategy: distribute (b, m) queries across cores.
 */
#include "kernel_operator.h"
#include "ball_query_dynamic.h"

class KernelBallQueryDynamic {
public:
    __aicore__ inline void Init(GM_ADDR xyz, GM_ADDR new_xyz, GM_ADDR idx, GM_ADDR tiling)
    {
        // Read tiling data
        pipe.InitBuffer(tilingBuf, 8 * sizeof(int32_t));
        AscendC::GlobalTensor<int32_t> tilingGm;
        tilingGm.SetGlobalBuffer((__gm__ int32_t *)tiling, 8);
        auto tilingLocal = tilingBuf.Get<int32_t>();
        AscendC::DataCopy(tilingLocal, tilingGm, 8);
        pipe_barrier(PIPE_V);

        B = tilingLocal.GetValue(0);
        N = tilingLocal.GetValue(1);
        M = tilingLocal.GetValue(2);
        nsample = tilingLocal.GetValue(3);
        union { int32_t i; float f; } radiusCaster;
        radiusCaster.i = tilingLocal.GetValue(4);
        radius = radiusCaster.f;
        numCores = tilingLocal.GetValue(5);
        queriesPerCore = tilingLocal.GetValue(6);

        coreId = AscendC::GetBlockIdx();

        // Align counts to 8 for DataCopy
        xyzCount = N * 3;
        xyzCountAlign = (xyzCount + 7) / 8 * 8;
        nsampleAlign = (nsample + 7) / 8 * 8;

        // UB buffers
        pipe.InitBuffer(xyzRowBuf, xyzCountAlign * sizeof(float));
        pipe.InitBuffer(newXyzBuf, 16 * sizeof(float));
        pipe.InitBuffer(idxBuf, nsampleAlign * sizeof(int32_t));

        xyzGm.SetGlobalBuffer((__gm__ float *)xyz, (uint64_t)B * N * 3);
        newXyzGm.SetGlobalBuffer((__gm__ float *)new_xyz, (uint64_t)B * M * 3);
        idxGm.SetGlobalBuffer((__gm__ int32_t *)idx, (uint64_t)B * M * nsample);
    }

    __aicore__ inline void Process()
    {
        auto xyzRow = xyzRowBuf.Get<float>();
        auto newXyz = newXyzBuf.Get<float>();
        auto idxLocal = idxBuf.Get<int32_t>();

        int32_t totalQueries = B * M;
        int32_t myStart = coreId * queriesPerCore;
        int32_t myEnd = (myStart + queriesPerCore > totalQueries) ? totalQueries : myStart + queriesPerCore;

        float radius2 = radius * radius;

        for (int32_t q = myStart; q < myEnd; q++) {
            int32_t b = q / M;
            int32_t m = q % M;

            // 0. Clear idxLocal (matches PyTorch torch.zeros initialization)
            AscendC::Duplicate(idxLocal, (int32_t)0, nsampleAlign);
            pipe_barrier(PIPE_V);

            // 1. Read xyz[b, :, :] into UB
            uint64_t xyzOffset = (uint64_t)b * N * 3;
            AscendC::DataCopy(xyzRow, xyzGm[xyzOffset], xyzCountAlign);
            pipe_barrier(PIPE_V);

            // 2. Read new_xyz[b, m, :] into UB (aligned read + sub-offset)
            int32_t newRawIdx = b * M * 3 + m * 3;
            int32_t newAlignedIdx = (newRawIdx / 8) * 8;
            int32_t newSubOff = newRawIdx - newAlignedIdx;
            AscendC::DataCopy(newXyz, newXyzGm[newAlignedIdx], 16);
            pipe_barrier(PIPE_V);

            float nx = newXyz.GetValue(newSubOff + 0);
            float ny = newXyz.GetValue(newSubOff + 1);
            float nz = newXyz.GetValue(newSubOff + 2);

            // 3. Find points within radius
            int32_t cnt = 0;
            int32_t firstIdx = 0;
            for (int32_t k = 0; k < N; k++) {
                float x = xyzRow.GetValue(k * 3 + 0);
                float y = xyzRow.GetValue(k * 3 + 1);
                float z = xyzRow.GetValue(k * 3 + 2);
                float dx = nx - x;
                float dy = ny - y;
                float dz = nz - z;
                float d2 = dx * dx + dy * dy + dz * dz;

                if (d2 < radius2) {
                    if (cnt == 0) {
                        firstIdx = k;
                        for (int32_t l = 0; l < nsample; l++) {
                            idxLocal.SetValue(l, k);
                        }
                    }
                    if (cnt < nsample) {
                        idxLocal.SetValue(cnt, k);
                    }
                    cnt++;
                    if (cnt >= nsample) break;
                }
            }

            pipe_barrier(PIPE_V);

            // 4. Write idx[b, m, :] back to GM
            uint64_t idxOffset = (uint64_t)b * M * nsample + (uint64_t)m * nsample;
            AscendC::DataCopy(idxGm[idxOffset], idxLocal, nsampleAlign);
            pipe_barrier(PIPE_V);

            // Prevent AscendC compiler from overlapping this DMA with
            // the next iteration's Duplicate(idxLocal). Without this read
            // dependency, the compiler may pipeline the Duplicate ahead of
            // the DMA completion, corrupting the output.
            float _dummy = idxLocal.GetValue(0);
            (void)_dummy;
        }
    }

private:
    int32_t B, N, M, nsample, numCores, queriesPerCore;
    int32_t xyzCount, xyzCountAlign, nsampleAlign;
    float radius;
    int32_t coreId;

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECIN> tilingBuf;
    AscendC::TBuf<AscendC::TPosition::VECIN> xyzRowBuf;
    AscendC::TBuf<AscendC::TPosition::VECIN> newXyzBuf;
    AscendC::TBuf<AscendC::TPosition::VECIN> idxBuf;

    AscendC::GlobalTensor<float> xyzGm;
    AscendC::GlobalTensor<float> newXyzGm;
    AscendC::GlobalTensor<int32_t> idxGm;
};

extern "C" __global__ __aicore__ void ball_query_dynamic(
    GM_ADDR xyz, GM_ADDR new_xyz, GM_ADDR idx, GM_ADDR tiling)
{
    KernelBallQueryDynamic op;
    op.Init(xyz, new_xyz, idx, tiling);
    op.Process();
}
