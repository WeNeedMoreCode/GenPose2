/**
 * Host wrapper for minimal multi-core SetValue test.
 */
#include "acl/acl.h"
#include "aclrtlaunch_fps_custom_minimal_test.h"
#include <cstdint>

constexpr uint32_t NUM_CORES = 8;
constexpr uint32_t FIELDS_PER_CORE = 4;
constexpr uint32_t DEBUG_SIZE = NUM_CORES * FIELDS_PER_CORE;

static aclrtStream g_minimal_stream = nullptr;
static void *g_minimal_debug_buf = nullptr;

extern "C" int fps_run_minimal_test(float *debug_host, int debug_host_size)
{
    if (g_minimal_stream == nullptr) {
        aclrtCreateStream(&g_minimal_stream);
    }
    if (g_minimal_debug_buf == nullptr) {
        aclrtMalloc(&g_minimal_debug_buf, DEBUG_SIZE * sizeof(float),
            ACL_MEM_MALLOC_HUGE_FIRST);
    }

    aclrtMemset(g_minimal_debug_buf, DEBUG_SIZE * sizeof(float),
        0, DEBUG_SIZE * sizeof(float));

    aclrtlaunch_fps_custom_minimal_test(NUM_CORES, g_minimal_stream,
        g_minimal_debug_buf);
    aclrtSynchronizeStream(g_minimal_stream);

    if (debug_host != nullptr && debug_host_size >= (int)DEBUG_SIZE) {
        aclrtMemcpy(debug_host, DEBUG_SIZE * sizeof(float),
            g_minimal_debug_buf, DEBUG_SIZE * sizeof(float),
            ACL_MEMCPY_DEVICE_TO_HOST);
    }

    return 0;
}
