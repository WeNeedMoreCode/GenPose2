/**
 * AscendC multi-core GroupPoints kernel (dynamic shape).
 *
 * Equivalent to CUDA group_points_kernel_fast:
 *   out[b, c, pt, s] = points[b, c, idx[b, pt, s]]
 *
 * No cross-core synchronization needed — each output element is independent.
 * Multi-core strategy: distribute (b, c) pairs across cores.
 */
#include "kernel_operator.h"
#include "group_points_dynamic.h"

class KernelGroupPointsDynamic {
public:
    __aicore__ inline void Init(GM_ADDR points, GM_ADDR idx, GM_ADDR out, GM_ADDR tiling)
    {
        // Read tiling data
        pipe.InitBuffer(tilingBuf, 8 * sizeof(int32_t));
        AscendC::GlobalTensor<int32_t> tilingGm;
        tilingGm.SetGlobalBuffer((__gm__ int32_t *)tiling, 8);
        auto tilingLocal = tilingBuf.Get<int32_t>();
        AscendC::DataCopy(tilingLocal, tilingGm, 8);
        pipe_barrier(PIPE_V);

        B = tilingLocal.GetValue(0);
        C = tilingLocal.GetValue(1);
        N = tilingLocal.GetValue(2);
        npoint = tilingLocal.GetValue(3);
        nsample = tilingLocal.GetValue(4);
        numCores = tilingLocal.GetValue(5);
        pairsPerCore = tilingLocal.GetValue(6);
        ptChunk = tilingLocal.GetValue(7);

        coreId = AscendC::GetBlockIdx();

        // Align nsample and ptChunk*nsample to 8 for DataCopy
        nsampleAlign = (nsample + 7) / 8 * 8;
        ptChunkNsampleAlign = (ptChunk * nsample + 7) / 8 * 8;

        // UB buffers
        // pointsRow: one row of points [N] = 4KB for N=1024
        pipe.InitBuffer(pointsRowBuf, N * sizeof(float));
        // idxBlock: idx for one pt_chunk [ptChunk * nsampleAlign]
        pipe.InitBuffer(idxBlockBuf, ptChunkNsampleAlign * sizeof(int32_t));
        // outBlock: output for one pt_chunk [ptChunk * nsampleAlign]
        pipe.InitBuffer(outBlockBuf, ptChunkNsampleAlign * sizeof(float));
        // tempBuf for DataCopy alignment padding
        pipe.InitBuffer(tmpBuf, 8 * sizeof(float));

        pointsGm.SetGlobalBuffer((__gm__ float *)points, (uint64_t)B * C * N);
        idxGm.SetGlobalBuffer((__gm__ int32_t *)idx, (uint64_t)B * npoint * nsample);
        outGm.SetGlobalBuffer((__gm__ float *)out, (uint64_t)B * C * npoint * nsample);
    }

    __aicore__ inline void Process()
    {
        auto pointsRow = pointsRowBuf.Get<float>();
        auto idxBlock = idxBlockBuf.Get<int32_t>();
        auto outBlock = outBlockBuf.Get<float>();

        int32_t totalPairs = B * C;
        int32_t myStart = coreId * pairsPerCore;
        int32_t myEnd = (myStart + pairsPerCore > totalPairs) ? totalPairs : myStart + pairsPerCore;

        for (int32_t p = myStart; p < myEnd; p++) {
            int32_t b = p / C;
            int32_t c = p % C;

            // 1. Read points[b, c, 0:N] into UB
            uint64_t pointsOffset = (uint64_t)b * C * N + (uint64_t)c * N;
            AscendC::DataCopy(pointsRow, pointsGm[pointsOffset], N);
            pipe_barrier(PIPE_V);

            // 2. Process npoint in chunks
            for (int32_t ptStart = 0; ptStart < npoint; ptStart += ptChunk) {
                int32_t curChunk = (ptStart + ptChunk > npoint) ? (npoint - ptStart) : ptChunk;
                int32_t curChunkAlign = (curChunk * nsample + 7) / 8 * 8;

                // 2a. Read idx[b, ptStart:ptStart+curChunk, 0:nsample]
                uint64_t idxOffset = (uint64_t)b * npoint * nsample + (uint64_t)ptStart * nsample;
                AscendC::DataCopy(idxBlock, idxGm[idxOffset], curChunkAlign);
                pipe_barrier(PIPE_V);

                // 2b. Gather: for each (pt, s), read idx value then read points[idx]
                for (int32_t pt = 0; pt < curChunk; pt++) {
                    for (int32_t s = 0; s < nsample; s++) {
                        int32_t idxVal = idxBlock.GetValue(pt * nsample + s);
                        float val = pointsRow.GetValue(idxVal);
                        outBlock.SetValue(pt * nsample + s, val);
                    }
                }
                pipe_barrier(PIPE_V);

                // 2c. Write out[b, c, ptStart:ptStart+curChunk, 0:nsample]
                uint64_t outOffset = (uint64_t)b * C * npoint * nsample
                                   + (uint64_t)c * npoint * nsample
                                   + (uint64_t)ptStart * nsample;
                AscendC::DataCopy(outGm[outOffset], outBlock, curChunkAlign);
                pipe_barrier(PIPE_V);
            }
        }
    }

private:
    int32_t B, C, N, npoint, nsample, numCores, pairsPerCore, ptChunk;
    int32_t nsampleAlign, ptChunkNsampleAlign;
    int32_t coreId;

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECIN> tilingBuf;
    AscendC::TBuf<AscendC::TPosition::VECIN> pointsRowBuf;
    AscendC::TBuf<AscendC::TPosition::VECIN> idxBlockBuf;
    AscendC::TBuf<AscendC::TPosition::VECIN> outBlockBuf;
    AscendC::TBuf<AscendC::TPosition::VECIN> tmpBuf;

    AscendC::GlobalTensor<float> pointsGm;
    AscendC::GlobalTensor<int32_t> idxGm;
    AscendC::GlobalTensor<float> outGm;
};

extern "C" __global__ __aicore__ void group_points_dynamic(
    GM_ADDR points, GM_ADDR idx, GM_ADDR out, GM_ADDR tiling)
{
    KernelGroupPointsDynamic op;
    op.Init(points, idx, out, tiling);
    op.Process();
}
