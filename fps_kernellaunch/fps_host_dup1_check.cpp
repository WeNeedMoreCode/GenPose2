/**
 * Host wrapper for Duplicate count=1 vs count=8 check.
 */
#include "acl/acl.h"
#include "aclrtlaunch_fps_custom_dup1_check.h"
#include <cstdint>
#include <cstdio>

constexpr uint32_t NUM_CORES = 8;
constexpr uint32_t DEBUG_SIZE = NUM_CORES * 16;  // 8 floats per sub-test × 2

static aclrtStream g_stream = nullptr;
static void *g_debug_buf = nullptr;

extern "C" int fps_run_dup1_check(float *debug_host, int debug_host_size)
{
    fprintf(stderr, "[dup1] Step 1: create stream\n");
    if (g_stream == nullptr) {
        aclError ret = aclrtCreateStream(&g_stream);
        fprintf(stderr, "[dup1] aclrtCreateStream ret = %d\n", (int)ret);
        if (ret != ACL_SUCCESS) return -1;
    }

    fprintf(stderr, "[dup1] Step 2: malloc debug buf (%u floats)\n", DEBUG_SIZE);
    if (g_debug_buf == nullptr) {
        aclError ret = aclrtMalloc(&g_debug_buf, DEBUG_SIZE * sizeof(float),
            ACL_MEM_MALLOC_HUGE_FIRST);
        fprintf(stderr, "[dup1] aclrtMalloc ret = %d\n", (int)ret);
        if (ret != ACL_SUCCESS) return -2;
    }

    fprintf(stderr, "[dup1] Step 3: memset debug to 0xFF\n");
    aclrtMemset(g_debug_buf, DEBUG_SIZE * sizeof(float), 0xFF, DEBUG_SIZE * sizeof(float));

    fprintf(stderr, "[dup1] Step 4: launch kernel (blockDim=%u)\n", NUM_CORES);
    uint32_t launch_ret = aclrtlaunch_fps_custom_dup1_check(NUM_CORES, g_stream,
        nullptr, nullptr, nullptr, g_debug_buf, nullptr);
    fprintf(stderr, "[dup1] aclrtlaunch ret = %u\n", launch_ret);

    fprintf(stderr, "[dup1] Step 5: sync stream\n");
    aclError sync_ret = aclrtSynchronizeStream(g_stream);
    fprintf(stderr, "[dup1] aclrtSynchronizeStream ret = %d\n", (int)sync_ret);

    if (debug_host != nullptr && debug_host_size >= (int)DEBUG_SIZE) {
        fprintf(stderr, "[dup1] Step 6: memcpy D2H\n");
        aclrtMemcpy(debug_host, DEBUG_SIZE * sizeof(float),
            g_debug_buf, DEBUG_SIZE * sizeof(float),
            ACL_MEMCPY_DEVICE_TO_HOST);
    }

    fprintf(stderr, "[dup1] Done. launch=%u sync=%d\n", launch_ret, (int)sync_ret);
    return (launch_ret != 0) ? -(1000 + (int)launch_ret) : 0;
}
