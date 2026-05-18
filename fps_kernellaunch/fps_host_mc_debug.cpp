/**
 * Host-side C wrapper for debug multi-core FPS kernel.
 * Allocates GM results buffer + debug buffer, copies debug data back to host.
 */
#include "acl/acl.h"
#include <cstdint>
#include <cstring>

constexpr uint32_t NUM_CORES = 8;
constexpr uint32_t DEBUG_ITERS = 3;
constexpr uint32_t DEBUG_SIZE = NUM_CORES * 4 * DEBUG_ITERS;  // 96 floats

extern void fps_custom_mc_debug_do(uint32_t blockDim, void *stream,
    uint8_t *xyz, uint8_t *idx, uint8_t *results, uint8_t *debug);

static aclrtStream g_dbg_stream = nullptr;
static void *g_dbg_results_buf = nullptr;
static void *g_dbg_debug_buf = nullptr;

extern "C" int fps_run_mc_debug(void *xyz_ptr, void *idx_ptr,
    float *debug_host, int debug_host_size)
{
    if (g_dbg_stream == nullptr) {
        aclrtCreateStream(&g_dbg_stream);
    }
    if (g_dbg_results_buf == nullptr) {
        aclrtMalloc(&g_dbg_results_buf, NUM_CORES * 2 * sizeof(float),
            ACL_MEM_MALLOC_HUGE_FIRST);
    }
    if (g_dbg_debug_buf == nullptr) {
        aclrtMalloc(&g_dbg_debug_buf, DEBUG_SIZE * sizeof(float),
            ACL_MEM_MALLOC_HUGE_FIRST);
    }

    fps_custom_mc_debug_do(NUM_CORES, g_dbg_stream,
        (uint8_t *)xyz_ptr, (uint8_t *)idx_ptr,
        (uint8_t *)g_dbg_results_buf, (uint8_t *)g_dbg_debug_buf);
    aclrtSynchronizeStream(g_dbg_stream);

    // Copy debug buffer back to host
    if (debug_host != nullptr && debug_host_size >= (int)DEBUG_SIZE) {
        aclrtMemcpy(debug_host, DEBUG_SIZE * sizeof(float),
            g_dbg_debug_buf, DEBUG_SIZE * sizeof(float),
            ACL_MEMCPY_DEVICE_TO_HOST);
    }

    return 0;
}
