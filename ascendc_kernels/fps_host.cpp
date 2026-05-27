/**
 * Host-side C wrapper for FPS kernel, callable from Python via ctypes.
 * Assumes ACL is already initialized (by torch_npu) and device is set.
 * Stream is reused across calls to avoid create/destroy overhead.
 */
#include "acl/acl.h"
#include <cstdint>

extern void fps_custom_do(uint32_t blockDim, void *stream, uint8_t *xyz, uint8_t *idx);

static aclrtStream g_stream = nullptr;

extern "C" int fps_run_device(void *xyz_ptr, void *idx_ptr)
{
    if (g_stream == nullptr) {
        aclrtCreateStream(&g_stream);
    }
    fps_custom_do(1, g_stream, (uint8_t *)xyz_ptr, (uint8_t *)idx_ptr);
    aclrtSynchronizeStream(g_stream);
    return 0;
}
