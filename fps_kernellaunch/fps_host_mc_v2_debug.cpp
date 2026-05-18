/**
 * Host-side C wrapper for v2 debug multi-core FPS kernel.
 * Allocates GM results buffer (256B) + debug buffer (384B), copies debug back to host.
 */
#include "acl/acl.h"
#include <cstdint>

constexpr uint32_t NUM_CORES = 8;
constexpr uint32_t RES_STRIDE = 8;
constexpr uint32_t DEBUG_ITERS = 3;
constexpr uint32_t DEBUG_SIZE = NUM_CORES * 4 * DEBUG_ITERS;

extern void fps_custom_mc_v2_dbg_do(uint32_t blockDim, void *stream,
    uint8_t *xyz, uint8_t *idx, uint8_t *results, uint8_t *debug);

static aclrtStream g_v2dbg_stream = nullptr;
static void *g_v2dbg_results_buf = nullptr;
static void *g_v2dbg_debug_buf = nullptr;

extern "C" int fps_run_mc_v2_dbg(void *xyz_ptr, void *idx_ptr,
    float *debug_host, int debug_host_size)
{
    if (g_v2dbg_stream == nullptr) {
        aclrtCreateStream(&g_v2dbg_stream);
    }
    if (g_v2dbg_results_buf == nullptr) {
        aclrtMalloc(&g_v2dbg_results_buf, NUM_CORES * RES_STRIDE * sizeof(float),
            ACL_MEM_MALLOC_HUGE_FIRST);
    }
    if (g_v2dbg_debug_buf == nullptr) {
        aclrtMalloc(&g_v2dbg_debug_buf, DEBUG_SIZE * sizeof(float),
            ACL_MEM_MALLOC_HUGE_FIRST);
    }

    fps_custom_mc_v2_dbg_do(NUM_CORES, g_v2dbg_stream,
        (uint8_t *)xyz_ptr, (uint8_t *)idx_ptr,
        (uint8_t *)g_v2dbg_results_buf, (uint8_t *)g_v2dbg_debug_buf);
    aclrtSynchronizeStream(g_v2dbg_stream);

    if (debug_host != nullptr && debug_host_size >= (int)DEBUG_SIZE) {
        aclrtMemcpy(debug_host, DEBUG_SIZE * sizeof(float),
            g_v2dbg_debug_buf, DEBUG_SIZE * sizeof(float),
            ACL_MEMCPY_DEVICE_TO_HOST);
    }

    return 0;
}
