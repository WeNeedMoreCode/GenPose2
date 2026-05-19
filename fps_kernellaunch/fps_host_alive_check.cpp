/**
 * Host wrapper for alive check kernel.
 */
#include "acl/acl.h"
#include "aclrtlaunch_fps_custom_alive_check.h"
#include <cstdint>

constexpr uint32_t NUM_CORES = 8;
constexpr uint32_t CHUNK = 1024 / NUM_CORES;
constexpr uint32_t PER_CORE = 8;
constexpr uint32_t ALIVE_DEBUG_SIZE = NUM_CORES * PER_CORE;
constexpr uint32_t SCRATCH_PER_CORE = 64;

static aclrtStream g_alive_stream = nullptr;
static void *g_alive_debug_buf = nullptr;
static void *g_alive_results_buf = nullptr;
static void *g_alive_scratch_buf = nullptr;

extern "C" int fps_run_alive_check(float *debug_host, int debug_host_size)
{
    if (g_alive_stream == nullptr) {
        aclrtCreateStream(&g_alive_stream);
    }
    if (g_alive_debug_buf == nullptr) {
        aclrtMalloc(&g_alive_debug_buf, ALIVE_DEBUG_SIZE * sizeof(float),
            ACL_MEM_MALLOC_HUGE_FIRST);
    }
    if (g_alive_results_buf == nullptr) {
        aclrtMalloc(&g_alive_results_buf, NUM_CORES * CHUNK * sizeof(float),
            ACL_MEM_MALLOC_HUGE_FIRST);
    }
    if (g_alive_scratch_buf == nullptr) {
        aclrtMalloc(&g_alive_scratch_buf, NUM_CORES * SCRATCH_PER_CORE * sizeof(float),
            ACL_MEM_MALLOC_HUGE_FIRST);
    }

    aclrtlaunch_fps_custom_alive_check(NUM_CORES, g_alive_stream,
        nullptr, nullptr, g_alive_results_buf, g_alive_debug_buf, g_alive_scratch_buf);
    aclrtSynchronizeStream(g_alive_stream);

    if (debug_host != nullptr && debug_host_size >= (int)ALIVE_DEBUG_SIZE) {
        aclrtMemcpy(debug_host, ALIVE_DEBUG_SIZE * sizeof(float),
            g_alive_debug_buf, ALIVE_DEBUG_SIZE * sizeof(float),
            ACL_MEMCPY_DEVICE_TO_HOST);
    }

    return 0;
}
