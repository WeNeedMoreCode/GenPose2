/**
 * Host wrapper for GetValue on GM check.
 * Writes known values to input_buf (H2D), launches kernel, reads debug output.
 */
#include "acl/acl.h"
#include "aclrtlaunch_fps_custom_getval_check.h"
#include <cstdint>
#include <cstdio>

constexpr uint32_t NUM_CORES = 8;
constexpr uint32_t PER_CORE_INPUT = 8;
constexpr uint32_t INPUT_SIZE = NUM_CORES * PER_CORE_INPUT;  // 64 floats
constexpr uint32_t DEBUG_SIZE = NUM_CORES * 24;              // 192 floats (3 sub-tests × 8)

static aclrtStream g_stream = nullptr;
static void *g_input_buf = nullptr;
static void *g_debug_buf = nullptr;
static void *g_scratch_buf = nullptr;

extern "C" int fps_run_getval_check(float *debug_host, int debug_host_size)
{
    fprintf(stderr, "[getval] Step 1: create stream\n");
    if (g_stream == nullptr) {
        aclError ret = aclrtCreateStream(&g_stream);
        fprintf(stderr, "[getval] aclrtCreateStream ret = %d\n", (int)ret);
        if (ret != ACL_SUCCESS) return -1;
    }

    fprintf(stderr, "[getval] Step 2: malloc input buf (%u floats)\n", INPUT_SIZE);
    if (g_input_buf == nullptr) {
        aclError ret = aclrtMalloc(&g_input_buf, INPUT_SIZE * sizeof(float),
            ACL_MEM_MALLOC_HUGE_FIRST);
        fprintf(stderr, "[getval] aclrtMalloc(input) ret = %d\n", (int)ret);
        if (ret != ACL_SUCCESS) return -2;
    }

    fprintf(stderr, "[getval] Step 3: malloc debug buf\n");
    if (g_debug_buf == nullptr) {
        aclError ret = aclrtMalloc(&g_debug_buf, DEBUG_SIZE * sizeof(float),
            ACL_MEM_MALLOC_HUGE_FIRST);
        fprintf(stderr, "[getval] aclrtMalloc(debug) ret = %d\n", (int)ret);
        if (ret != ACL_SUCCESS) return -3;
    }

    fprintf(stderr, "[getval] Step 4: malloc scratch buf\n");
    if (g_scratch_buf == nullptr) {
        aclError ret = aclrtMalloc(&g_scratch_buf, INPUT_SIZE * sizeof(float),
            ACL_MEM_MALLOC_HUGE_FIRST);
        fprintf(stderr, "[getval] aclrtMalloc(scratch) ret = %d\n", (int)ret);
        if (ret != ACL_SUCCESS) return -4;
    }

    // Write known values to input: core 0 → 10.0, core 1 → 20.0, ..., core 7 → 80.0
    fprintf(stderr, "[getval] Step 5: write known values to input GM\n");
    float input_data[INPUT_SIZE];
    for (uint32_t c = 0; c < NUM_CORES; c++) {
        float val = (float)(c + 1) * 10.0f;
        for (uint32_t i = 0; i < PER_CORE_INPUT; i++) {
            input_data[c * PER_CORE_INPUT + i] = val;
        }
    }
    aclrtMemcpy(g_input_buf, INPUT_SIZE * sizeof(float),
        input_data, INPUT_SIZE * sizeof(float),
        ACL_MEMCPY_HOST_TO_DEVICE);
    fprintf(stderr, "[getval] H2D done\n");

    fprintf(stderr, "[getval] Step 6: memset debug to 0xFF\n");
    aclrtMemset(g_debug_buf, DEBUG_SIZE * sizeof(float), 0xFF, DEBUG_SIZE * sizeof(float));

    fprintf(stderr, "[getval] Step 7: launch kernel (blockDim=%u)\n", NUM_CORES);
    uint32_t launch_ret = aclrtlaunch_fps_custom_getval_check(NUM_CORES, g_stream,
        g_input_buf, nullptr, nullptr, g_debug_buf, g_scratch_buf);
    fprintf(stderr, "[getval] aclrtlaunch ret = %u\n", launch_ret);

    fprintf(stderr, "[getval] Step 8: sync stream\n");
    aclError sync_ret = aclrtSynchronizeStream(g_stream);
    fprintf(stderr, "[getval] aclrtSynchronizeStream ret = %d\n", (int)sync_ret);

    if (debug_host != nullptr && debug_host_size >= (int)DEBUG_SIZE) {
        fprintf(stderr, "[getval] Step 9: memcpy D2H\n");
        aclrtMemcpy(debug_host, DEBUG_SIZE * sizeof(float),
            g_debug_buf, DEBUG_SIZE * sizeof(float),
            ACL_MEMCPY_DEVICE_TO_HOST);
    }

    fprintf(stderr, "[getval] Done. launch=%u sync=%d\n", launch_ret, (int)sync_ret);
    return (launch_ret != 0) ? -(1000 + (int)launch_ret) : 0;
}
