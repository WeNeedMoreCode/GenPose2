/**
 * Host wrapper for ball_query_debug kernel.
 * Launches kernel, synchronizes, readbacks debug buffer (64 floats).
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
    if (!ensure_buf(&g_debug_buf, 64 * sizeof(float))) return -7;

    // Pre-fill debug with sentinel
    float sentinel[64];
    for (int i = 0; i < 64; i++) sentinel[i] = -999.0f;
    aclError ret = aclrtMemcpy(g_debug_buf, 64 * sizeof(float), sentinel, 64 * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
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

    // Readback debug buffer
    float dbg[64];
    ret = aclrtMemcpy(dbg, 64 * sizeof(float), g_debug_buf, 64 * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        fprintf(stderr, "[bq_debug] readback failed: %d\n", (int)ret);
        return -5;
    }

    // Print diagnostics
    fprintf(stderr, "\n[bq_debug] === AscendC kernel diagnostics for query (b=0, m=0) ===\n");

    // Check sentinel
    bool has_sentinel = true;
    for (int i = 0; i < 4; i++) {
        if (dbg[i] != -999.0f) { has_sentinel = false; break; }
    }
    if (has_sentinel) {
        fprintf(stderr, "[bq_debug] SENTINEL detected — kernel did not write diagnostics!\n");
        return -10;
    }

    // CP0: new_xyz
    fprintf(stderr, "[bq_debug] CP0: new_xyz = (%.8f, %.8f, %.8f)\n", dbg[0], dbg[1], dbg[2]);
    fprintf(stderr, "[bq_debug]      radius = %.8f, radius2 = %.10f\n", dbg[3], dbg[3]*dbg[3]);

    // CP1: distances
    fprintf(stderr, "[bq_debug] CP1: First 8 squared distances:\n");
    for (int i = 0; i < 8; i++) {
        fprintf(stderr, "[bq_debug]   k=%d: d2=%.12f  pass=%d\n",
                i, dbg[4+i], (int)dbg[12+i]);
    }

    // CP2: cnt and firstIdx
    fprintf(stderr, "[bq_debug] CP2: cnt=%d, firstIdx=%d\n", (int)dbg[20], (int)dbg[21]);

    // CP3: idx values
    fprintf(stderr, "[bq_debug] CP3: idx[0..7] =");
    for (int i = 0; i < 8; i++) {
        fprintf(stderr, " %d", (int)dbg[22+i]);
    }
    fprintf(stderr, "\n");

    // Metadata
    fprintf(stderr, "[bq_debug] Meta: N=%d, nsample=%d\n", (int)dbg[30], (int)dbg[31]);

    // First 8 xyz coords
    fprintf(stderr, "[bq_debug] xyz[0..7]:\n");
    for (int i = 0; i < 8; i++) {
        fprintf(stderr, "[bq_debug]   k=%d: (%.8f, %.8f, %.8f)\n",
                i, dbg[32+i], dbg[40+i], dbg[48+i]);
    }

    fprintf(stderr, "[bq_debug] === End diagnostics ===\n\n");
    return 0;
}
