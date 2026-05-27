/**
 * Minimal multi-core alive check: Duplicate + DataCopy per core.
 * 5-parameter signature matches v3 debug kernel for binary compatibility.
 */
#include "kernel_operator.h"

constexpr int32_t N = 1024;
constexpr int32_t NUM_CORES = 8;
constexpr int32_t CHUNK = N / NUM_CORES;
constexpr int32_t BLOCK_SIZE = 64;
constexpr int32_t PER_CORE = 8;

class KernelAliveCheck {
public:
    __aicore__ inline void Init(GM_ADDR xyz, GM_ADDR idx, GM_ADDR results, GM_ADDR debug, GM_ADDR scratch)
    {
        debugGm.SetGlobalBuffer((__gm__ float *)debug, NUM_CORES * PER_CORE);
        pipe.InitBuffer(buf1, CHUNK * sizeof(float));
        pipe.InitBuffer(buf2, BLOCK_SIZE * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        int32_t coreId = AscendC::GetBlockIdx();
        auto local = buf1.Get<float>();

        float coreVal = (coreId + 1) * 100.0f;
        AscendC::Duplicate(local, coreVal, PER_CORE);
        pipe_barrier(PIPE_V);

        AscendC::DataCopy(debugGm[coreId * PER_CORE], local, PER_CORE);
        pipe_barrier(PIPE_V);

        AscendC::SyncAll();
    }

private:
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECIN> buf1;
    AscendC::TBuf<AscendC::TPosition::VECIN> buf2;
    AscendC::GlobalTensor<float> debugGm;
};

extern "C" __global__ __aicore__ void fps_custom_alive_check(
    GM_ADDR xyz, GM_ADDR idx, GM_ADDR results, GM_ADDR debug, GM_ADDR scratch)
{
    KernelAliveCheck op;
    op.Init(xyz, idx, results, debug, scratch);
    op.Process();
}
