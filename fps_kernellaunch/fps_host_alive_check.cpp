/**
 * Host wrapper for alive check kernel — with detailed diagnostics.
 */
#include "acl/acl.h"
#include "aclrtlaunch_fps_custom_alive_check.h"
#include <cstdint>
#include <cstdio>

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
    fprintf(stderr, "[alive_check] Step 1: create stream\n");
    if (g_alive_stream == nullptr) {
        aclError ret = aclrtCreateStream(&g_alive_stream);
        fprintf(stderr, "[alive_check] aclrtCreateStream ret = %d\n", (int)ret);
        if (ret != ACL_SUCCESS) return -1;
    }

    fprintf(stderr, "[alive_check] Step 2: malloc debug buf (%u floats)\n", ALIVE_DEBUG_SIZE);
    if (g_alive_debug_buf == nullptr) {
        aclError ret = aclrtMalloc(&g_alive_debug_buf, ALIVE_DEBUG_SIZE * sizeof(float),
            ACL_MEM_MALLOC_HUGE_FIRST);
        fprintf(stderr, "[alive_check] aclrtMalloc(debug) ret = %d, ptr = %p\n", (int)ret, g_alive_debug_buf);
        if (ret != ACL_SUCCESS) return -2;
    }

    fprintf(stderr, "[alive_check] Step 3: malloc results buf\n");
    if (g_alive_results_buf == nullptr) {
        aclError ret = aclrtMalloc(&g_alive_results_buf, NUM_CORES * CHUNK * sizeof(float),
            ACL_MEM_MALLOC_HUGE_FIRST);
        fprintf(stderr, "[alive_check] aclrtMalloc(results) ret = %d, ptr = %p\n", (int)ret, g_alive_results_buf);
        if (ret != ACL_SUCCESS) return -3;
    }

    fprintf(stderr, "[alive_check] Step 4: malloc scratch buf\n");
    if (g_alive_scratch_buf == nullptr) {
        aclError ret = aclrtMalloc(&g_alive_scratch_buf, NUM_CORES * SCRATCH_PER_CORE * sizeof(float),
            ACL_MEM_MALLOC_HUGE_FIRST);
        fprintf(stderr, "[alive_check] aclrtMalloc(scratch) ret = %d, ptr = %p\n", (int)ret, g_alive_scratch_buf);
        if (ret != ACL_SUCCESS) return -4;
    }

    // Clear debug buffer to 0xFF so we can distinguish "kernel didn't run" from "kernel wrote 0"
    fprintf(stderr, "[alive_check] Step 5: memset debug buf to 0xFF\n");
    aclError memset_ret = aclrtMemset(g_alive_debug_buf, ALIVE_DEBUG_SIZE * sizeof(float),
        0xFF, ALIVE_DEBUG_SIZE * sizeof(float));
    fprintf(stderr, "[alive_check] aclrtMemset ret = %d\n", (int)memset_ret);

    fprintf(stderr, "[alive_check] Step 6: launch kernel (blockDim=%u)\n", NUM_CORES);
    uint32_t launch_ret = aclrtlaunch_fps_custom_alive_check(NUM_CORES, g_alive_stream,
        nullptr,              // xyz (unused)
        nullptr,              // idx (unused)
        g_alive_results_buf, // results
        g_alive_debug_buf,   // debug
        g_alive_scratch_buf  // scratch
    );
    fprintf(stderr, "[alive_check] aclrtlaunch ret = %u\n", launch_ret);

    fprintf(stderr, "[alive_check] Step 7: sync stream\n");
    aclError sync_ret = aclrtSynchronizeStream(g_alive_stream);
    fprintf(stderr, "[alive_check] aclrtSynchronizeStream ret = %d\n", (int)sync_ret);

    if (debug_host != nullptr && debug_host_size >= (int)ALIVE_DEBUG_SIZE) {
        fprintf(stderr, "[alive_check] Step 8: memcpy D2H (%u floats)\n", ALIVE_DEBUG_SIZE);
        aclError memcpy_ret = aclrtMemcpy(debug_host, ALIVE_DEBUG_SIZE * sizeof(float),
            g_alive_debug_buf, ALIVE_DEBUG_SIZE * sizeof(float),
            ACL_MEMCPY_DEVICE_TO_HOST);
        fprintf(stderr, "[alive_check] aclrtMemcpy ret = %d\n", (int)memcpy_ret);

        // Print first 16 raw values
        fprintf(stderr, "[alive_check] Raw device data (first 16): ");
        for (int i = 0; i < 16 && i < debug_host_size; i++) {
            fprintf(stderr, "%.1f ", debug_host[i]);
        }
        fprintf(stderr, "\n");
    }

    fprintf(stderr, "[alive_check] Done. launch_ret=%u sync_ret=%d\n", launch_ret, (int)sync_ret);
    return (launch_ret != 0) ? -(1000 + (int)launch_ret) : 0;
}
