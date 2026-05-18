/**
 * Host-side C wrapper for multi-core FPS kernel.
 * Allocates GM buffers for cross-core results and sync, reuses stream.
 */
#include "acl/acl.h"
#include <cstdint>

constexpr uint32_t NUM_CORES = 8;

extern void fps_custom_multicore_do(uint32_t blockDim, void *stream, uint8_t *xyz, uint8_t *idx, uint8_t *results, uint8_t *syncBuf);

static aclrtStream g_mc_stream = nullptr;
static void *g_results_buf = nullptr;
static void *g_sync_buf = nullptr;

extern "C" int fps_run_multicore(void *xyz_ptr, void *idx_ptr)
{
    if (g_mc_stream == nullptr) {
        aclrtCreateStream(&g_mc_stream);
    }
    if (g_results_buf == nullptr) {
        aclrtMalloc(&g_results_buf, NUM_CORES * 2 * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    }
    if (g_sync_buf == nullptr) {
        aclrtMalloc(&g_sync_buf, 32, ACL_MEM_MALLOC_HUGE_FIRST);
    }
    fps_custom_multicore_do(NUM_CORES, g_mc_stream,
        (uint8_t *)xyz_ptr, (uint8_t *)idx_ptr,
        (uint8_t *)g_results_buf, (uint8_t *)g_sync_buf);
    aclrtSynchronizeStream(g_mc_stream);
    return 0;
}
