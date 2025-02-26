#include "flash_fwd_mla_kernel.h"

// template void run_mha_fwd_splitkv_mla_sm80<cutlass::half_t, 576>(Flash_fwd_mla_params &params, cudaStream_t stream);
template void run_mha_fwd_splitkv_mla<cutlass::half_t, 576>(Flash_fwd_mla_params &params, cudaStream_t stream);

// 声明 SM80 版本
template void run_mha_fwd_splitkv_mla_sm80<cutlass::half_t, 576>(Flash_fwd_mla_params &params, cudaStream_t stream) {
    // 暂时使用 SM90 实现
    run_mha_fwd_splitkv_mla<cutlass::half_t, 576>(params, stream);
}