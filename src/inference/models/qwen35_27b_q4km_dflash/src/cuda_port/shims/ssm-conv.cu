// Shim TU: pulls in vendored ssm-conv.cu and forces explicit template
// instantiations for the (apply_silu, split_d_inner=128, d_conv) combos we
// need at runtime. Upstream's ssm-conv.cu already has `static` stripped
// from `ssm_conv_f32` and `ssm_conv_long_token_f32` (see the CTOX-
// MODIFICATION comment there), but the instantiating caller lives in
// lucebox/dflash's `kernels/sm_86/ssm_conv.cu` which isn't in our vendor
// tree. This file plays that role for the cuda_port path.
//
// The vendored file's compile command still happens through our build.rs
// (we compile the shim, which `#include`s the vendored .cu — so the
// upstream source stays byte-identical on disk).

#include "../../../vendor/ggml-cuda/ssm-conv.cu"

// 8 short-token instantiations: ssm_conv_f32<apply_silu, 128, d_conv>
template __global__ void ssm_conv_f32<false, 128, 3>(const float *, const float *, int, int, int, int, float *, int, int, int, const int64_t);
template __global__ void ssm_conv_f32<false, 128, 4>(const float *, const float *, int, int, int, int, float *, int, int, int, const int64_t);
template __global__ void ssm_conv_f32<false, 128, 5>(const float *, const float *, int, int, int, int, float *, int, int, int, const int64_t);
template __global__ void ssm_conv_f32<false, 128, 9>(const float *, const float *, int, int, int, int, float *, int, int, int, const int64_t);
template __global__ void ssm_conv_f32<true,  128, 3>(const float *, const float *, int, int, int, int, float *, int, int, int, const int64_t);
template __global__ void ssm_conv_f32<true,  128, 4>(const float *, const float *, int, int, int, int, float *, int, int, int, const int64_t);
template __global__ void ssm_conv_f32<true,  128, 5>(const float *, const float *, int, int, int, int, float *, int, int, int, const int64_t);
template __global__ void ssm_conv_f32<true,  128, 9>(const float *, const float *, int, int, int, int, float *, int, int, int, const int64_t);

// 8 long-token instantiations: ssm_conv_long_token_f32<apply_silu, 128, d_conv, 32>
template __global__ void ssm_conv_long_token_f32<false, 128, 3, 32>(const float *, const float *, int, int, int, int, float *, int, int, int, const int64_t);
template __global__ void ssm_conv_long_token_f32<false, 128, 4, 32>(const float *, const float *, int, int, int, int, float *, int, int, int, const int64_t);
template __global__ void ssm_conv_long_token_f32<false, 128, 5, 32>(const float *, const float *, int, int, int, int, float *, int, int, int, const int64_t);
template __global__ void ssm_conv_long_token_f32<false, 128, 9, 32>(const float *, const float *, int, int, int, int, float *, int, int, int, const int64_t);
template __global__ void ssm_conv_long_token_f32<true,  128, 3, 32>(const float *, const float *, int, int, int, int, float *, int, int, int, const int64_t);
template __global__ void ssm_conv_long_token_f32<true,  128, 4, 32>(const float *, const float *, int, int, int, int, float *, int, int, int, const int64_t);
template __global__ void ssm_conv_long_token_f32<true,  128, 5, 32>(const float *, const float *, int, int, int, int, float *, int, int, int, const int64_t);
template __global__ void ssm_conv_long_token_f32<true,  128, 9, 32>(const float *, const float *, int, int, int, int, float *, int, int, int, const int64_t);
