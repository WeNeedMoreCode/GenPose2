/**
 * Ball Query debug kernel: output per-query diagnostics for the FIRST query (b=0, m=0).
 *
 * Checkpoints:
 *   CP0: new_xyz values (nx, ny, nz)
 *   CP1: first 8 points' squared distances + pass/fail
 *   CP2: cnt (total points found), firstIdx
 *   CP3: first 8 idx values written
 *
 * Output: 64 floats to debug GM.
 */
#include "kernel_operator.h"
#include "ball_query_dynamic.h"

class KernelBallQueryDebug {
public:
    __aicore__ inline void Init(GM_ADDR xyz, GM_ADDR new_xyz, GM_ADDR idx, GM_ADDR tiling, GM_ADDR debug)
    {
        // Read tiling
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
        union { int32_t i; float f; } c;
        c.i = tilingLocal.GetValue(4);
        radius = c.f;
        numCores = tilingLocal.GetValue(5);
        queriesPerCore = tilingLocal.GetValue(6);
        coreId = AscendC::GetBlockIdx();

        xyzCount = N * 3;
        xyzCountAlign = (xyzCount + 7) / 8 * 8;
        nsampleAlign = (nsample + 7) / 8 * 8;

        pipe.InitBuffer(xyzRowBuf, xyzCountAlign * sizeof(float));
        pipe.InitBuffer(newXyzBuf, 16 * sizeof(float));
        pipe.InitBuffer(idxBuf, nsampleAlign * sizeof(int32_t));
        pipe.InitBuffer(debugBuf, 128 * sizeof(float));

        xyzGm.SetGlobalBuffer((__gm__ float *)xyz, (uint64_t)B * N * 3);
        newXyzGm.SetGlobalBuffer((__gm__ float *)new_xyz, (uint64_t)B * M * 3);
        idxGm.SetGlobalBuffer((__gm__ int32_t *)idx, (uint64_t)B * M * nsample);
        debugGm.SetGlobalBuffer((__gm__ float *)debug, 128);
    }

    __aicore__ inline void Process()
    {
        auto xyzRow = xyzRowBuf.Get<float>();
        auto newXyz = newXyzBuf.Get<float>();
        auto idxLocal = idxBuf.Get<int32_t>();
        auto dbg = debugBuf.Get<float>();

        float radius2 = radius * radius;
        int32_t totalQueries = B * M;
        int32_t myStart = coreId * queriesPerCore;
        int32_t myEnd = (myStart + queriesPerCore > totalQueries) ? totalQueries : myStart + queriesPerCore;

        bool isDebugQuery = false;
        bool isDebugQuery1 = false;
        float dbgNx = 0, dbgNy = 0, dbgNz = 0;
        float dbgDists[8] = {0};
        int32_t dbgD2Pass[8] = {0};
        int32_t dbgCnt = 0;
        int32_t dbgFirstIdx = -1;
        int32_t dbgIdxVals[8] = {0};
        float dbg1Nx = 0, dbg1Ny = 0, dbg1Nz = 0;
        float dbg1Dists[8] = {0};
        int32_t dbg1D2Pass[8] = {0};
        int32_t dbg1Cnt = 0;
        int32_t dbg1IdxVals[8] = {0};

        for (int32_t q = myStart; q < myEnd; q++) {
            int32_t b = q / M;
            int32_t m = q % M;

            AscendC::Duplicate(idxLocal, (int32_t)0, nsampleAlign);
            pipe_barrier(PIPE_V);

            uint64_t xyzOffset = (uint64_t)b * N * 3;
            AscendC::DataCopy(xyzRow, xyzGm[xyzOffset], xyzCountAlign);
            pipe_barrier(PIPE_V);

            int32_t newRawIdx = b * M * 3 + m * 3;
            int32_t newAlignedIdx = (newRawIdx / 8) * 8;
            int32_t newSubOff = newRawIdx - newAlignedIdx;
            AscendC::DataCopy(newXyz, newXyzGm[newAlignedIdx], 16);
            pipe_barrier(PIPE_V);

            float nx = newXyz.GetValue(newSubOff + 0);
            float ny = newXyz.GetValue(newSubOff + 1);
            float nz = newXyz.GetValue(newSubOff + 2);

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

            uint64_t idxOffset = (uint64_t)b * M * nsample + (uint64_t)m * nsample;
            AscendC::DataCopy(idxGm[idxOffset], idxLocal, nsampleAlign);
            pipe_barrier(PIPE_V);

            if (b == 0 && m == 0) {
                isDebugQuery = true;
                dbgNx = nx; dbgNy = ny; dbgNz = nz;
                dbgCnt = cnt;
                dbgFirstIdx = firstIdx;
                for (int32_t i = 0; i < 8 && i < N; i++) {
                    float x = xyzRow.GetValue(i * 3 + 0);
                    float y = xyzRow.GetValue(i * 3 + 1);
                    float z = xyzRow.GetValue(i * 3 + 2);
                    float dx = nx - x, dy = ny - y, dz = nz - z;
                    dbgDists[i] = dx * dx + dy * dy + dz * dz;
                    dbgD2Pass[i] = (dbgDists[i] < radius2) ? 1 : 0;
                }
                for (int32_t i = 0; i < 8; i++) {
                    dbgIdxVals[i] = idxLocal.GetValue(i);
                }
            }
            if (b == 0 && m == 1) {
                isDebugQuery1 = true;
                dbg1Nx = nx; dbg1Ny = ny; dbg1Nz = nz;
                dbg1Cnt = cnt;
                for (int32_t i = 0; i < 8 && i < N; i++) {
                    float x = xyzRow.GetValue(i * 3 + 0);
                    float y = xyzRow.GetValue(i * 3 + 1);
                    float z = xyzRow.GetValue(i * 3 + 2);
                    float dx = nx - x, dy = ny - y, dz = nz - z;
                    dbg1Dists[i] = dx * dx + dy * dy + dz * dz;
                    dbg1D2Pass[i] = (dbg1Dists[i] < radius2) ? 1 : 0;
                }
                for (int32_t i = 0; i < 8; i++) {
                    dbg1IdxVals[i] = idxLocal.GetValue(i);
                }
            }
        }

        if (isDebugQuery || isDebugQuery1) {
            AscendC::Duplicate(dbg, (float)-999, 128);
            pipe_barrier(PIPE_V);

            if (isDebugQuery) {
                dbg.SetValue(0, dbgNx);
                dbg.SetValue(1, dbgNy);
                dbg.SetValue(2, dbgNz);
                dbg.SetValue(3, radius);
                for (int32_t i = 0; i < 8; i++) {
                    dbg.SetValue(4 + i, dbgDists[i]);
                    dbg.SetValue(12 + i, (float)dbgD2Pass[i]);
                }
                dbg.SetValue(20, (float)dbgCnt);
                dbg.SetValue(21, (float)dbgFirstIdx);
                for (int32_t i = 0; i < 8; i++) {
                    dbg.SetValue(22 + i, (float)dbgIdxVals[i]);
                }
                dbg.SetValue(30, (float)N);
                dbg.SetValue(31, (float)nsample);
            }

            if (isDebugQuery1) {
                dbg.SetValue(32, dbg1Nx);
                dbg.SetValue(33, dbg1Ny);
                dbg.SetValue(34, dbg1Nz);
                dbg.SetValue(35, radius);
                for (int32_t i = 0; i < 8; i++) {
                    dbg.SetValue(36 + i, dbg1Dists[i]);
                    dbg.SetValue(44 + i, (float)dbg1D2Pass[i]);
                }
                dbg.SetValue(52, (float)dbg1Cnt);
                for (int32_t i = 0; i < 8; i++) {
                    dbg.SetValue(54 + i, (float)dbg1IdxVals[i]);
                }
                for (int32_t i = 0; i < 8 && i < N; i++) {
                    dbg.SetValue(64 + i, xyzRow.GetValue(i * 3 + 0));
                    dbg.SetValue(72 + i, xyzRow.GetValue(i * 3 + 1));
                    dbg.SetValue(80 + i, xyzRow.GetValue(i * 3 + 2));
                }
            }

            pipe_barrier(PIPE_V);
            AscendC::DataCopy(debugGm, dbg, 128);
            pipe_barrier(PIPE_V);
        }
    }

private:
    int32_t B, N, M, nsample, numCores, queriesPerCore, coreId;
    int32_t xyzCount, xyzCountAlign, nsampleAlign;
    float radius;

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECIN> tilingBuf;
    AscendC::TBuf<AscendC::TPosition::VECIN> xyzRowBuf;
    AscendC::TBuf<AscendC::TPosition::VECIN> newXyzBuf;
    AscendC::TBuf<AscendC::TPosition::VECIN> idxBuf;
    AscendC::TBuf<AscendC::TPosition::VECIN> debugBuf;

    AscendC::GlobalTensor<float> xyzGm;
    AscendC::GlobalTensor<float> newXyzGm;
    AscendC::GlobalTensor<int32_t> idxGm;
    AscendC::GlobalTensor<float> debugGm;
};

extern "C" __global__ __aicore__ void ball_query_debug(
    GM_ADDR xyz, GM_ADDR new_xyz, GM_ADDR idx, GM_ADDR tiling, GM_ADDR debug)
{
    KernelBallQueryDebug op;
    op.Init(xyz, new_xyz, idx, tiling, debug);
    op.Process();
}
