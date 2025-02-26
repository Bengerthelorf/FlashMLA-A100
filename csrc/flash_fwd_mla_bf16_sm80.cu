#include "flash_fwd_mla_kernel.h"

// template void run_mha_fwd_splitkv_mla_sm80<cutlass::bfloat16_t, 576>(Flash_fwd_mla_params &params, cudaStream_t stream);
template void run_mha_fwd_splitkv_mla<cutlass::bfloat16_t, 576>(Flash_fwd_mla_params &params, cudaStream_t stream);

// 声明 SM90 版本
template void run_mha_fwd_splitkv_mla_sm80<cutlass::bfloat16_t, 576>(Flash_fwd_mla_params &params, cudaStream_t stream) {
    // 暂时使用 SM80 实现
    run_mha_fwd_splitkv_mla<cutlass::bfloat16_t, 576>(params, stream);
}