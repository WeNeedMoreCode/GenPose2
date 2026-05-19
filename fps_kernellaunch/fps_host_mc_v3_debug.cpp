/**
 * Host-side C wrapper for v3 debug multi-core FPS kernel.
 * All scalar reads routed through GM (scratch buffer).
 * Also exports fps_run_minimal_test for basic multi-core Duplicate+DataCopy test.
 */
#include "acl/acl.h"
#include "aclrtlaunch_fps_custom_mc_v3_dbg.h"
#include "aclrtlaunch_fps_custom_minimal_test.h"
#include <cstdint>

constexpr uint32_t NUM_CORES = 8;
constexpr uint32_t CHUNK = 1024 / NUM_CORES;
constexpr uint32_t DEBUG_ITERS = 3;
constexpr uint32_t DIAG_FIELDS = 8;
constexpr uint32_t DEBUG_SIZE = NUM_CORES * DIAG_FIELDS * DEBUG_ITERS;
constexpr uint32_t SCRATCH_PER_CORE = 64;

// Minimal test constants
constexpr uint32_t MINI_PER_CORE = 8;
constexpr uint32_t MINI_DEBUG_SIZE = NUM_CORES * MINI_PER_CORE;

static aclrtStream g_v3dbg_stream = nullptr;
static void *g_v3dbg_results_buf = nullptr;
static void *g_v3dbg_debug_buf = nullptr;
static void *g_v3dbg_scratch_buf = nullptr;

// --- Minimal multi-core test (Duplicate + DataCopy) ---

static aclrtStream g_mini_stream = nullptr;
static void *g_mini_debug_buf = nullptr;
static void *g_mini_results_buf = nullptr;
static void *g_mini_scratch_buf = nullptr;

extern "C" int fps_run_minimal_test(float *debug_host, int debug_host_size)
{
    if (g_mini_stream == nullptr) {
        aclrtCreateStream(&g_mini_stream);
    }
    if (g_mini_debug_buf == nullptr) {
        aclrtMalloc(&g_mini_debug_buf, MINI_DEBUG_SIZE * sizeof(float),
            ACL_MEM_MALLOC_HUGE_FIRST);
    }
    if (g_mini_results_buf == nullptr) {
        aclrtMalloc(&g_mini_results_buf, NUM_CORES * CHUNK * sizeof(float),
            ACL_MEM_MALLOC_HUGE_FIRST);
    }
    if (g_mini_scratch_buf == nullptr) {
        aclrtMalloc(&g_mini_scratch_buf, NUM_CORES * SCRATCH_PER_CORE * sizeof(float),
            ACL_MEM_MALLOC_HUGE_FIRST);
    }

    // Pass 5 params (same as v3 debug), only debug is used by kernel
    aclrtlaunch_fps_custom_minimal_test(NUM_CORES, g_mini_stream,
        nullptr, nullptr, g_mini_results_buf, g_mini_debug_buf, g_mini_scratch_buf);
    aclrtSynchronizeStream(g_mini_stream);

    if (debug_host != nullptr && debug_host_size >= (int)MINI_DEBUG_SIZE) {
        aclrtMemcpy(debug_host, MINI_DEBUG_SIZE * sizeof(float),
            g_mini_debug_buf, MINI_DEBUG_SIZE * sizeof(float),
            ACL_MEMCPY_DEVICE_TO_HOST);
    }

    return 0;
}

// --- v3 debug FPS kernel ---

extern "C" int fps_run_mc_v3_dbg(void *xyz_ptr, void *idx_ptr,
    float *debug_host, int debug_host_size)
{
    if (g_v3dbg_stream == nullptr) {
        aclrtCreateStream(&g_v3dbg_stream);
    }
    if (g_v3dbg_results_buf == nullptr) {
        aclrtMalloc(&g_v3dbg_results_buf, NUM_CORES * CHUNK * sizeof(float),
            ACL_MEM_MALLOC_HUGE_FIRST);
    }
    if (g_v3dbg_debug_buf == nullptr) {
        aclrtMalloc(&g_v3dbg_debug_buf, DEBUG_SIZE * sizeof(float),
            ACL_MEM_MALLOC_HUGE_FIRST);
    }
    if (g_v3dbg_scratch_buf == nullptr) {
        aclrtMalloc(&g_v3dbg_scratch_buf, NUM_CORES * SCRATCH_PER_CORE * sizeof(float),
            ACL_MEM_MALLOC_HUGE_FIRST);
    }

    aclrtlaunch_fps_custom_mc_v3_dbg(NUM_CORES, g_v3dbg_stream,
        xyz_ptr, idx_ptr, g_v3dbg_results_buf, g_v3dbg_debug_buf, g_v3dbg_scratch_buf);
    aclrtSynchronizeStream(g_v3dbg_stream);

    if (debug_host != nullptr && debug_host_size >= (int)DEBUG_SIZE) {
        aclrtMemcpy(debug_host, DEBUG_SIZE * sizeof(float),
            g_v3dbg_debug_buf, DEBUG_SIZE * sizeof(float),
            ACL_MEMCPY_DEVICE_TO_HOST);
    }

    return 0;
}
