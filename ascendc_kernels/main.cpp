/**
 * Host-side FPS test: generates random data, runs kernel, verifies against CPU reference
 */
#include "data_utils.h"
#ifndef ASCENDC_CPU_DEBUG
#include "acl/acl.h"
extern void fps_custom_do(uint32_t blockDim, void *stream, uint8_t *xyz, uint8_t *idx);
#else
#include "tikicpulib.h"
extern "C" __global__ __aicore__ void fps_custom(GM_ADDR xyz, GM_ADDR idx);
#endif

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

constexpr int32_t N = 1024;
constexpr int32_t NPOINTS = 512;

// CPU reference FPS
void cpu_fps(const float *xyz, int32_t *idx, int n, int npoints)
{
    // xyz layout: [3, n] -> x[0:n], y[n:2n], z[2n:3n]
    std::vector<float> distance(n, 1e10f);
    idx[0] = 0;
    int32_t old = 0;

    for (int j = 1; j < npoints; j++) {
        float x1 = xyz[old];
        float y1 = xyz[n + old];
        float z1 = xyz[2 * n + old];

        for (int k = 0; k < n; k++) {
            float dx = xyz[k] - x1;
            float dy = xyz[n + k] - y1;
            float dz = xyz[2 * n + k] - z1;
            float d = dx * dx + dy * dy + dz * dz;
            distance[k] = std::min(distance[k], d);
        }

        float bestVal = -1.0f;
        int32_t bestIdx = 0;
        for (int k = 0; k < n; k++) {
            if (distance[k] > bestVal) {
                bestVal = distance[k];
                bestIdx = k;
            }
        }
        idx[j] = bestIdx;
        old = bestIdx;
    }
}

int32_t main(int32_t argc, char *argv[])
{
    size_t xyzBytes = 3 * N * sizeof(float);
    size_t idxBytes = NPOINTS * sizeof(int32_t);

    // Generate random xyz [3, N] in transposed layout
    std::vector<float> xyzHost(3 * N);
    srand(42);
    for (int i = 0; i < 3 * N; i++) {
        xyzHost[i] = (float)(rand() % 1000) / 100.0f;
    }

    // CPU reference
    std::vector<int32_t> expectedIdx(NPOINTS);
    cpu_fps(xyzHost.data(), expectedIdx.data(), N, NPOINTS);

    std::vector<int32_t> resultIdx(NPOINTS, -1);

#ifdef ASCENDC_CPU_DEBUG
    uint8_t *xyz = (uint8_t *)AscendC::GmAlloc(xyzBytes);
    uint8_t *idx = (uint8_t *)AscendC::GmAlloc(idxBytes);
    memcpy(xyz, xyzHost.data(), xyzBytes);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(fps_custom, 1, xyz, idx);

    memcpy(resultIdx.data(), idx, idxBytes);
    AscendC::GmFree((void *)xyz);
    AscendC::GmFree((void *)idx);
#else
    CHECK_ACL(aclInit(nullptr));
    int32_t deviceId = 0;
    CHECK_ACL(aclrtSetDevice(deviceId));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    uint8_t *xyzDevice = nullptr;
    uint8_t *idxDevice = nullptr;
    CHECK_ACL(aclrtMalloc((void **)&xyzDevice, xyzBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&idxDevice, idxBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemcpy(xyzDevice, xyzBytes, xyzHost.data(), xyzBytes, ACL_MEMCPY_HOST_TO_DEVICE));

    fps_custom_do(1, stream, xyzDevice, idxDevice);
    CHECK_ACL(aclrtSynchronizeStream(stream));

    CHECK_ACL(aclrtMemcpy(resultIdx.data(), idxBytes, idxDevice, idxBytes, ACL_MEMCPY_DEVICE_TO_HOST));

    CHECK_ACL(aclrtFree(xyzDevice));
    CHECK_ACL(aclrtFree(idxDevice));
    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());
#endif

    // Verify
    bool pass = true;
    for (int i = 0; i < NPOINTS; i++) {
        if (resultIdx[i] != expectedIdx[i]) {
            printf("FAIL at [%d]: got %d, expected %d\n", i, resultIdx[i], expectedIdx[i]);
            pass = false;
            if (i > 10) {
                printf("... (stopped after 10 mismatches)\n");
                break;
            }
        }
    }
    printf("%s\n", pass ? "PASS" : "FAIL");

    return pass ? 0 : 1;
}
