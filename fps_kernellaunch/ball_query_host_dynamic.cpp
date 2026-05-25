/**
 * Host wrapper for dynamic-shape multi-core Ball Query kernel.
 *
 * idx[b, m, :] = indices of points in xyz[b, :, :] within radius of new_xyz[b, m, :]
 *
 * No sync buffer needed — no cross-core synchronization.
 */
#include "acl/acl.h"
#include "aclrtlaunch_ball_query_dynamic.h"
#include "ball_query_dynamic.h"
#include <cstdint>
#include <cstdio>

static void *g_tiling_buf = nullptr;

static bool ensure_buf(void **buf, int32_t size) {
    if (*buf == nullptr) {
        aclError ret = aclrtMalloc(buf, size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            fprintf(stderr, "[bq_dynamic] aclrtMalloc failed: ret=%d, size=%d\n", (int)ret, size);
            return false;
        }
    }
    return true;
}

extern "C" int ball_query_run_dynamic(
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

    BallQueryTilingData tiling;
    tiling.B = B;
    tiling.N = N;
    tiling.M = M;
    tiling.nsample = nsample;
    tiling.radius = radius;
    tiling.numCores = num_cores;
    tiling.queriesPerCore = queriesPerCore;

    aclError ret = aclrtMemcpy(g_tiling_buf, sizeof(BallQueryTilingData),
                                &tiling, sizeof(BallQueryTilingData),
                                ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) return -5;

    aclrtlaunch_ball_query_dynamic(num_cores, stream,
        xyz_ptr, new_xyz_ptr, idx_ptr, g_tiling_buf);

    // No synchronize here — let torch.npu stream manage synchronization
    return 0;
}
