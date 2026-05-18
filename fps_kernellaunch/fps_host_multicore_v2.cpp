/**
 * Host-side C wrapper for multi-core FPS v2 kernel.
 * Allocates 256-byte GM results buffer (NUM_CORES * 8 floats, DataCopy aligned).
 */
#include "acl/acl.h"
#include <cstdint>

constexpr uint32_t NUM_CORES = 8;
constexpr uint32_t RES_STRIDE = 8;

extern void fps_custom_multicore_v2_do(uint32_t blockDim, void *stream, uint8_t *xyz, uint8_t *idx, uint8_t *results);

static aclrtStream g_v2_stream = nullptr;
static void *g_v2_results_buf = nullptr;

extern "C" int fps_run_multicore_v2(void *xyz_ptr, void *idx_ptr)
{
    if (g_v2_stream == nullptr) {
        aclrtCreateStream(&g_v2_stream);
    }
    if (g_v2_results_buf == nullptr) {
        aclrtMalloc(&g_v2_results_buf, NUM_CORES * RES_STRIDE * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    }
    fps_custom_multicore_v2_do(NUM_CORES, g_v2_stream,
        (uint8_t *)xyz_ptr, (uint8_t *)idx_ptr,
        (uint8_t *)g_v2_results_buf);
    aclrtSynchronizeStream(g_v2_stream);
    return 0;
}
