/**
 * Host wrapper for ball_query_debug kernel.
 * Launches kernel, synchronizes, readbacks debug buffer (128 floats).
 */
#include "acl/acl.h"
#include "aclrtlaunch_ball_query_debug.h"
#include "ball_query_dynamic.h"
#include <cstdint>
#include <cstdio>

static void *g_tiling_buf = nullptr;
static void *g_debug_buf = nullptr;

static bool ensure_buf(void **buf, int32_t size) {
    if (*buf == nullptr) {
        aclError ret = aclrtMalloc(buf, size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            fprintf(stderr, "[bq_debug] aclrtMalloc failed: ret=%d, size=%d\n", (int)ret, size);
            return false;
        }
    }
    return true;
}

extern "C" int ball_query_debug_run(
    void *xyz_ptr, void *new_xyz_ptr, void *idx_ptr,
    int32_t B, int32_t N, int32_t M, int32_t nsample, float radius,
    int32_t num_cores, void *stream_ptr)
{
    if (B <= 0 || N <= 0 || M <= 0 || nsample <= 0 || radius <= 0.0f || num_cores <= 0) return -1;

    aclrtStream stream = (aclrtStream)stream_ptr;
    if (stream == nullptr) return -2;

    int32_t totalQueries = B * M;
    int32_t queriesPerCore = (totalQueries + num_cores - 1) / num_cores;

    if (!ensure_buf(&g_tiling_buf, 8 * sizeof(int32_t))) return -7;
    if (!ensure_buf(&g_debug_buf, 128 * sizeof(float))) return -7;

    float sentinel[128];
    for (int i = 0; i < 128; i++) sentinel[i] = -999.0f;
    aclError ret = aclrtMemcpy(g_debug_buf, 128 * sizeof(float), sentinel, 128 * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        fprintf(stderr, "[bq_debug] sentinel H2D failed: %d\n", (int)ret);
        return -5;
    }

    BallQueryTilingData tiling;
    tiling.B = B;
    tiling.N = N;
    tiling.M = M;
    tiling.nsample = nsample;
    tiling.radius = radius;
    tiling.numCores = num_cores;
    tiling.queriesPerCore = queriesPerCore;

    ret = aclrtMemcpy(g_tiling_buf, sizeof(BallQueryTilingData),
                      &tiling, sizeof(BallQueryTilingData),
                      ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) return -5;

    fprintf(stderr, "[bq_debug] Launch: B=%d N=%d M=%d nsample=%d radius=%.4f cores=%d\n",
            B, N, M, nsample, radius, num_cores);

    aclrtlaunch_ball_query_debug(num_cores, stream,
        xyz_ptr, new_xyz_ptr, idx_ptr, g_tiling_buf, g_debug_buf);

    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
        fprintf(stderr, "[bq_debug] synchronize failed: %d\n", (int)ret);
        return -6;
    }

    float dbg[128];
    ret = aclrtMemcpy(dbg, 128 * sizeof(float), g_debug_buf, 128 * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        fprintf(stderr, "[bq_debug] readback failed: %d\n", (int)ret);
        return -5;
    }

    // Query (0,0) diagnostics
    if (dbg[0] != -999.0f) {
        fprintf(stderr, "\n[bq_debug] === Query (b=0, m=0) ===\n");
        fprintf(stderr, "[bq_debug] CP0: new_xyz = (%.8f, %.8f, %.8f)\n", dbg[0], dbg[1], dbg[2]);
        fprintf(stderr, "[bq_debug]      radius = %.8f, radius2 = %.10f\n", dbg[3], dbg[3]*dbg[3]);
        fprintf(stderr, "[bq_debug] CP1: First 8 squared distances:\n");
        for (int i = 0; i < 8; i++)
            fprintf(stderr, "[bq_debug]   k=%d: d2=%.12f  pass=%d\n", i, dbg[4+i], (int)dbg[12+i]);
        fprintf(stderr, "[bq_debug] CP2: cnt=%d, firstIdx=%d\n", (int)dbg[20], (int)dbg[21]);
        fprintf(stderr, "[bq_debug] CP3: idx[0..7] =");
        for (int i = 0; i < 8; i++) fprintf(stderr, " %d", (int)dbg[22+i]);
        fprintf(stderr, "\n[bq_debug] Meta: N=%d, nsample=%d\n", (int)dbg[30], (int)dbg[31]);
    }

    // Query (0,1) diagnostics
    if (dbg[32] != -999.0f) {
        fprintf(stderr, "\n[bq_debug] === Query (b=0, m=1) ===\n");
        fprintf(stderr, "[bq_debug] CP0: new_xyz = (%.8f, %.8f, %.8f)\n", dbg[32], dbg[33], dbg[34]);
        fprintf(stderr, "[bq_debug]      radius = %.8f, radius2 = %.10f\n", dbg[35], dbg[35]*dbg[35]);
        fprintf(stderr, "[bq_debug] CP1: First 8 squared distances:\n");
        for (int i = 0; i < 8; i++)
            fprintf(stderr, "[bq_debug]   k=%d: d2=%.12f  pass=%d\n", i, dbg[36+i], (int)dbg[44+i]);
        fprintf(stderr, "[bq_debug] CP2: cnt=%d\n", (int)dbg[52]);
        fprintf(stderr, "[bq_debug] CP3: idx[0..7] =");
        for (int i = 0; i < 8; i++) fprintf(stderr, " %d", (int)dbg[54+i]);
        fprintf(stderr, "\n");
        fprintf(stderr, "[bq_debug] xyz[0..7] (from this query's xyzRow):\n");
        for (int i = 0; i < 8; i++)
            fprintf(stderr, "[bq_debug]   k=%d: (%.8f, %.8f, %.8f)\n",
                    i, dbg[64+i], dbg[72+i], dbg[80+i]);
    } else {
        fprintf(stderr, "\n[bq_debug] Query (0,1) NOT captured (sentinel remains)\n");
    }

    fprintf(stderr, "[bq_debug] === End diagnostics ===\n\n");
    return 0;
}
