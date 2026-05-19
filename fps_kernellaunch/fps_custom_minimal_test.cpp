/**
 * Minimal multi-core test: Duplicate + DataCopy (standard AscendC pattern).
 * Each core fills a UB buffer with its own value, then DataCopy to GM.
 * Core 0 → 100.0, Core 1 → 200.0, ..., Core 7 → 800.0
 */
#include "kernel_operator.h"

constexpr int32_t NUM_CORES = 8;
constexpr int32_t PER_CORE = 8;

class KernelMinimalTest {
public:
    __aicore__ inline void Init(GM_ADDR debug)
    {
        debugGm.SetGlobalBuffer((__gm__ float *)debug, NUM_CORES * PER_CORE);
        pipe.InitBuffer(buf, PER_CORE * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        int32_t coreId = AscendC::GetBlockIdx();
        auto local = buf.Get<float>();

        float coreVal = (coreId + 1) * 100.0f;
        AscendC::Duplicate(local, coreVal, PER_CORE);
        pipe_barrier(PIPE_V);

        AscendC::DataCopy(debugGm[coreId * PER_CORE], local, PER_CORE);
        pipe_barrier(PIPE_V);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECIN> buf;
    AscendC::GlobalTensor<float> debugGm;
};

extern "C" __global__ __aicore__ void fps_custom_minimal_test(GM_ADDR debug)
{
    KernelMinimalTest op;
    op.Init(debug);
    op.Process();
}
