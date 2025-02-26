#include "flash_fwd_mla_kernel.h"

// 实例化SM90版本
// template void run_mha_fwd_splitkv_mla<cutlass::bfloat16_t, 576>(Flash_fwd_mla_params &params, cudaStream_t stream);

// 实例化SM80版本 - 不要再定义函数体，只需要实例化
template void run_mha_fwd_splitkv_mla_sm80<cutlass::bfloat16_t, 576>(Flash_fwd_mla_params &params, cudaStream_t stream);