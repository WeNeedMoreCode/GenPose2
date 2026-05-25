/**
 * Host wrapper for dynamic-shape multi-core GroupPoints kernel.
 *
 * out[b, c, pt, s] = points[b, c, idx[b, pt, s]]
 *
 * No sync buffer needed — no cross-core synchronization.
 */
#include "acl/acl.h"
#include "aclrtlaunch_group_points_dynamic.h"
#include "group_points_dynamic.h"
#include <cstdint>
#include <cstdio>

constexpr int32_t PT_CHUNK = 64;

static void *g_tiling_buf = nullptr;

static bool ensure_buf(void **buf, int32_t size) {
    if (*buf == nullptr) {
        aclError ret = aclrtMalloc(buf, size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            fprintf(stderr, "[gp_dynamic] aclrtMalloc failed: ret=%d, size=%d\n", (int)ret, size);
            return false;
        }
    }
    return true;
}

extern "C" int group_points_run_dynamic(
    void *points_ptr, void *idx_ptr, void *out_ptr,
    int32_t B, int32_t C, int32_t N, int32_t npoint, int32_t nsample,
    int32_t num_cores, void *stream_ptr)
{
    if (B <= 0 || C <= 0 || N <= 0 || npoint <= 0 || nsample <= 0 || num_cores <= 0) return -1;

    aclrtStream stream = (aclrtStream)stream_ptr;
    if (stream == nullptr) return -2;

    int32_t totalPairs = B * C;
    int32_t pairsPerCore = (totalPairs + num_cores - 1) / num_cores;

    if (!ensure_buf(&g_tiling_buf, 8 * sizeof(int32_t))) return -7;

    GroupPointsTilingData tiling;
    tiling.B = B;
    tiling.C = C;
    tiling.N = N;
    tiling.npoint = npoint;
    tiling.nsample = nsample;
    tiling.numCores = num_cores;
    tiling.pairsPerCore = pairsPerCore;
    tiling.ptChunk = PT_CHUNK;

    aclError ret = aclrtMemcpy(g_tiling_buf, sizeof(GroupPointsTilingData),
                                &tiling, sizeof(GroupPointsTilingData),
                                ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) return -5;

    aclrtlaunch_group_points_dynamic(num_cores, stream,
        points_ptr, idx_ptr, out_ptr, g_tiling_buf);

    // No synchronize here — let torch.npu stream manage synchronization
    return 0;
}
