/**
 * Host wrapper for dynamic-shape multi-core FPS kernel.
 * Accepts N and npoints as runtime parameters.
 * stream_ptr: pass NULL to auto-create, or pass torch.npu.current_stream() pointer.
 */
#include "acl/acl.h"
#include "aclrtlaunch_fps_custom_dynamic.h"
#include "fps_custom_dynamic.h"
#include <cstdint>
#include <cstdio>

constexpr int32_t BLOCK_SIZE = 64;
constexpr int32_t MAX_N = 1024;
constexpr int32_t MAX_CORES = 8;

static aclrtStream g_stream = nullptr;
static void *g_results_buf = nullptr;
static void *g_scratch_buf = nullptr;
static void *g_sync_buf = nullptr;
static void *g_tiling_buf = nullptr;

static bool ensure_buf(void **buf, int32_t size) {
    if (*buf == nullptr) {
        aclError ret = aclrtMalloc(buf, size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            fprintf(stderr, "[fps_dynamic] aclrtMalloc failed: ret=%d, size=%d\n", (int)ret, size);
            return false;
        }
    }
    return true;
}

extern "C" int fps_run_dynamic(void *xyz_ptr, void *idx_ptr,
                                int32_t total_n, int32_t npoints, int32_t num_cores,
                                void *stream_ptr)
{
    fprintf(stderr, "[fps_dynamic] called: N=%d, npoints=%d, cores=%d, stream_ptr=%p\n",
            total_n, npoints, num_cores, stream_ptr);

    if (total_n <= 0 || npoints <= 0 || num_cores <= 0) {
        fprintf(stderr, "[fps_dynamic] invalid params: N=%d, npoints=%d, cores=%d\n",
                total_n, npoints, num_cores);
        return -1;
    }

    aclrtStream stream = (aclrtStream)stream_ptr;
    if (stream == nullptr) {
        if (g_stream == nullptr) {
            aclError ret = aclrtCreateStream(&g_stream);
            if (ret != ACL_SUCCESS) {
                fprintf(stderr, "[fps_dynamic] aclrtCreateStream failed: %d\n", (int)ret);
                return -2;
            }
            fprintf(stderr, "[fps_dynamic] created own stream=%p\n", g_stream);
        }
        stream = g_stream;
    } else {
        fprintf(stderr, "[fps_dynamic] using provided stream=%p\n", stream);
    }

    // Cap num_cores so chunk >= BLOCK_SIZE (hardware vector length)
    int32_t max_cores = total_n / BLOCK_SIZE;
    if (max_cores < 1) max_cores = 1;
    if (num_cores > max_cores) {
        num_cores = max_cores;
    }

    int32_t chunk = total_n / num_cores;
    int32_t blocks_per_core = chunk / BLOCK_SIZE;

    if (chunk * num_cores != total_n) {
        fprintf(stderr, "[fps_dynamic] totalN=%d not divisible by numCores=%d\n",
                total_n, num_cores);
        return -3;
    }
    if (blocks_per_core * BLOCK_SIZE != chunk) {
        fprintf(stderr, "[fps_dynamic] chunk=%d not divisible by BLOCK_SIZE=%d\n",
                chunk, BLOCK_SIZE);
        return -4;
    }

    // Allocate persistent buffers at max size
    if (!ensure_buf(&g_results_buf, MAX_CORES * (MAX_N / MAX_CORES) * sizeof(float))) return -7;
    if (!ensure_buf(&g_scratch_buf, MAX_CORES * 64 * sizeof(float))) return -7;
    if (!ensure_buf(&g_sync_buf, MAX_CORES * 8 * sizeof(int32_t))) return -7;
    if (!ensure_buf(&g_tiling_buf, 8 * sizeof(int32_t))) return -7;

    fprintf(stderr, "[fps_dynamic] buffers: xyz=%p idx=%p results=%p scratch=%p sync=%p tiling=%p\n",
            xyz_ptr, idx_ptr, g_results_buf, g_scratch_buf, g_sync_buf, g_tiling_buf);

    // Prepare tiling data
    FpsTilingData tiling;
    tiling.totalN = total_n;
    tiling.npoints = npoints;
    tiling.numCores = num_cores;
    tiling.chunk = chunk;
    tiling.blocksPerCore = blocks_per_core;

    fprintf(stderr, "[fps_dynamic] tiling: N=%d np=%d cores=%d chunk=%d bpc=%d\n",
            tiling.totalN, tiling.npoints, tiling.numCores, tiling.chunk, tiling.blocksPerCore);

    aclError ret = aclrtMemcpy(g_tiling_buf, sizeof(FpsTilingData),
                                &tiling, sizeof(FpsTilingData),
                                ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        fprintf(stderr, "[fps_dynamic] tiling H2D failed: %d\n", (int)ret);
        return -5;
    }

    // Clear sync buffer before launch (stale SyncAll flags may cause deadlock)
    {
        int32_t zeros[MAX_CORES * 8] = {0};
        aclrtMemcpy(g_sync_buf, sizeof(zeros), zeros, sizeof(zeros), ACL_MEMCPY_HOST_TO_DEVICE);
    }

    fprintf(stderr, "[fps_dynamic] launching kernel on stream=%p...\n", stream);
    aclrtlaunch_fps_custom_dynamic(num_cores, stream,
        xyz_ptr, idx_ptr, g_results_buf, g_scratch_buf, g_sync_buf, g_tiling_buf);

    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
        fprintf(stderr, "[fps_dynamic] synchronize failed: %d\n", (int)ret);
        return -6;
    }

    fprintf(stderr, "[fps_dynamic] done\n");
    return 0;
}
