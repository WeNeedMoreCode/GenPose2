/**
 * Minimal multi-core test: each core writes 4 known constants to GM.
 * No TBuf, no TPipe, no vector ops — only SetValue (scalar GM write).
 * Purpose: verify aclrtlaunch dispatches all 8 cores and SetValue works.
 */
#include "kernel_operator.h"

constexpr int32_t NUM_CORES = 8;
constexpr int32_t FIELDS_PER_CORE = 4;
constexpr int32_t DEBUG_SIZE = NUM_CORES * FIELDS_PER_CORE;

class KernelMinimalTest {
public:
    __aicore__ inline void Init(GM_ADDR debug)
    {
        debugGm.SetGlobalBuffer((__gm__ float *)debug, DEBUG_SIZE);
    }

    __aicore__ inline void Process()
    {
        int32_t coreId = AscendC::GetBlockIdx();
        int32_t off = coreId * FIELDS_PER_CORE;
        debugGm.SetValue(off + 0, 1e10f);
        debugGm.SetValue(off + 1, 3.14f);
        debugGm.SetValue(off + 2, -1.0f);
        debugGm.SetValue(off + 3, 42.0f);
    }

private:
    AscendC::GlobalTensor<float> debugGm;
};

extern "C" __global__ __aicore__ void fps_custom_minimal_test(GM_ADDR debug)
{
    KernelMinimalTest op;
    op.Init(debug);
    op.Process();
}
