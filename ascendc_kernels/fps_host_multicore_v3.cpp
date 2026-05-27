/**
 * Host-side C wrapper for multi-core FPS v3 kernel.
 * GM results buffer: NUM_CORES * CHUNK floats (each core writes 128 floats).
 */
#include "acl/acl.h"
#include <cstdint>

constexpr uint32_t NUM_CORES = 8;
constexpr uint32_t CHUNK = 1024 / NUM_CORES;  // 128

extern void fps_custom_multicore_v3_do(uint32_t blockDim, void *stream, uint8_t *xyz, uint8_t *idx, uint8_t *results);

static aclrtStream g_v3_stream = nullptr;
static void *g_v3_results_buf = nullptr;

extern "C" int fps_run_multicore_v3(void *xyz_ptr, void *idx_ptr)
{
    if (g_v3_stream == nullptr) {
        aclrtCreateStream(&g_v3_stream);
    }
    if (g_v3_results_buf == nullptr) {
        aclrtMalloc(&g_v3_results_buf, NUM_CORES * CHUNK * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    }
    fps_custom_multicore_v3_do(NUM_CORES, g_v3_stream,
        (uint8_t *)xyz_ptr, (uint8_t *)idx_ptr,
        (uint8_t *)g_v3_results_buf);
    aclrtSynchronizeStream(g_v3_stream);
    return 0;
}
