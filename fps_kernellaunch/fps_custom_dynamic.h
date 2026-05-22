/**
 * Dynamic-shape FPS kernel.
 * All shape parameters read from TilingData (passed via GM), not constexpr.
 *
 * TilingData layout (passed as last GM argument):
 *   int32_t totalN       — total input points (e.g. 1024, 512, 256)
 *   int32_t npoints       — output sample count (e.g. 512, 256, 128)
 *   int32_t numCores      — number of AI cores (8)
 *   int32_t chunk         — totalN / numCores
 *   int32_t blocksPerCore — chunk / BLOCK_SIZE
 *
 * UB buffers allocated for MAX_N=1024 to fit within 128KB budget.
 */
#pragma once

struct FpsTilingData {
    int32_t totalN;
    int32_t npoints;
    int32_t numCores;
    int32_t chunk;
    int32_t blocksPerCore;
};
