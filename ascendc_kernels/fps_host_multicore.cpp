/**
 * Host-side C wrapper for multi-core FPS kernel (8 AI Cores).
 * Allocates GM results buffer for cross-core argmax, reuses stream.
 */
#include "acl/acl.h"
#include <cstdint>

constexpr uint32_t NUM_CORES = 8;

extern void fps_custom_multicore_do(uint32_t blockDim, void *stream, uint8_t *xyz, uint8_t *idx, uint8_t *results);

static aclrtStream g_mc_stream = nullptr;
static void *g_results_buf = nullptr;

extern "C" int fps_run_multicore(void *xyz_ptr, void *idx_ptr)
{
    if (g_mc_stream == nullptr) {
        aclrtCreateStream(&g_mc_stream);
    }
    if (g_results_buf == nullptr) {
        aclrtMalloc(&g_results_buf, NUM_CORES * 2 * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    }
    fps_custom_multicore_do(NUM_CORES, g_mc_stream,
        (uint8_t *)xyz_ptr, (uint8_t *)idx_ptr,
        (uint8_t *)g_results_buf);
    aclrtSynchronizeStream(g_mc_stream);
    return 0;
}
