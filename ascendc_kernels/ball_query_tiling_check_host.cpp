/**
 * Host wrapper for tiling check: allocate debug buffer with sentinel -1.0,
 * launch kernel, synchronize, readback and print per-core tiling values.
 */
#include "acl/acl.h"
#include "aclrtlaunch_ball_query_tiling_check.h"
#include "ball_query_dynamic.h"
#include <cstdint>
#include <cstdio>

extern "C" int ball_query_tiling_check_run(
    int32_t B, int32_t N, int32_t M, int32_t nsample, float radius,
    int32_t num_cores, void *stream_ptr)
{
    aclrtStream stream = (aclrtStream)stream_ptr;
    if (stream == nullptr) return -2;

    int32_t totalQueries = B * M;
    int32_t queriesPerCore = (totalQueries + num_cores - 1) / num_cores;

    // Tiling data
    BallQueryTilingData tiling;
    tiling.B = B;
    tiling.N = N;
    tiling.M = M;
    tiling.nsample = nsample;
    tiling.radius = radius;
    tiling.numCores = num_cores;
    tiling.queriesPerCore = queriesPerCore;

    // Allocate and upload tiling
    void *tiling_buf = nullptr;
    aclError ret = aclrtMalloc(&tiling_buf, 8 * sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        fprintf(stderr, "[bq_tiling] aclrtMalloc tiling failed: %d\n", (int)ret);
        return -7;
    }
    ret = aclrtMemcpy(tiling_buf, sizeof(BallQueryTilingData), &tiling, sizeof(BallQueryTilingData), ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        fprintf(stderr, "[bq_tiling] aclrtMemcpy H2D tiling failed: %d\n", (int)ret);
        aclrtFree(tiling_buf);
        return -5;
    }

    // Allocate debug buffer, pre-fill with sentinel -1.0
    void *debug_buf = nullptr;
    ret = aclrtMalloc(&debug_buf, 8 * 8 * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        fprintf(stderr, "[bq_tiling] aclrtMalloc debug failed: %d\n", (int)ret);
        aclrtFree(tiling_buf);
        return -7;
    }
    float sentinel[64];
    for (int i = 0; i < 64; i++) sentinel[i] = -1.0f;
    ret = aclrtMemcpy(debug_buf, 64 * sizeof(float), sentinel, 64 * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        fprintf(stderr, "[bq_tiling] aclrtMemcpy H2D sentinel failed: %d\n", (int)ret);
        aclrtFree(tiling_buf);
        aclrtFree(debug_buf);
        return -5;
    }

    // Launch kernel
    fprintf(stderr, "[bq_tiling] Launching kernel: B=%d N=%d M=%d nsample=%d radius=%.3f numCores=%d\n",
            B, N, M, nsample, radius, num_cores);
    aclrtlaunch_ball_query_tiling_check(num_cores, stream, tiling_buf, debug_buf);

    // Synchronize
    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
        fprintf(stderr, "[bq_tiling] synchronize failed: %d\n", (int)ret);
        aclrtFree(tiling_buf);
        aclrtFree(debug_buf);
        return -6;
    }

    // Readback debug buffer
    float debug_host[64];
    ret = aclrtMemcpy(debug_host, 64 * sizeof(float), debug_buf, 64 * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        fprintf(stderr, "[bq_tiling] readback failed: %d\n", (int)ret);
        aclrtFree(tiling_buf);
        aclrtFree(debug_buf);
        return -5;
    }

    // Print per-core results
    fprintf(stderr, "[bq_tiling] === Per-core readback (expect N=%d) ===\n", N);
    int mismatch_count = 0;
    for (int c = 0; c < num_cores; c++) {
        float *d = &debug_host[c * 8];
        fprintf(stderr, "[bq_tiling] Core %d: B=%.0f N=%.0f M=%.0f nsample=%.0f radius=%.3f numCores=%.0f qpc=%.0f coreId=%.0f",
                c, d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7]);
        if (d[1] == -1.0f) {
            fprintf(stderr, "  [SENTINEL! kernel did not execute]\n");
            mismatch_count++;
        } else if ((int)d[1] != N) {
            fprintf(stderr, "  [MISMATCH! expect N=%d]\n", N);
            mismatch_count++;
        } else {
            fprintf(stderr, "  [OK]\n");
        }
    }
    fprintf(stderr, "[bq_tiling] Result: %d/%d cores correct\n\n", num_cores - mismatch_count, num_cores);

    aclrtFree(tiling_buf);
    aclrtFree(debug_buf);
    return mismatch_count > 0 ? -10 : 0;
}
