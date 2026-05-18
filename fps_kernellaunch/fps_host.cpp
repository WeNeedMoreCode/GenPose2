/**
 * Host-side C wrapper for FPS kernel, callable from Python via ctypes.
 * Assumes ACL is already initialized (by torch_npu) and device is set.
 */
#include "acl/acl.h"
#include <cstdint>

extern void fps_custom_do(uint32_t blockDim, void *stream, uint8_t *xyz, uint8_t *idx);

extern "C" int fps_run_device(void *xyz_ptr, void *idx_ptr)
{
    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);
    fps_custom_do(1, stream, (uint8_t *)xyz_ptr, (uint8_t *)idx_ptr);
    aclrtSynchronizeStream(stream);
    aclrtDestroyStream(stream);
    return 0;
}
