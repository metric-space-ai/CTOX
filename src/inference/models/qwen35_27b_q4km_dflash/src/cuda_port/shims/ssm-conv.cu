// Shim TU: pulls in vendored ssm-conv.cu and forces explicit template
// instantiations for the (apply_silu, split_d_inner=128, d_conv) combos we
// need at runtime. Upstream's ssm-conv.cu already has `static` stripped
// from `ssm_conv_f32` and `ssm_conv_long_token_f32` (see the CTOX-
// MODIFICATION comment there), but bare `template __global__ void ...`
// declarations aren't enough to force PTX emission — nvcc only emits an
// entry when a __global__ kernel is actually referenced from host-side
// launch syntax or via a function-pointer take.
//
// The trick: host stubs that `<<<1,1,0,s>>>` each instantiation. The
// stubs are `__host__`-only, never called at runtime, but their presence
// forces nvcc to emit the corresponding device entries. We hide them
// under a name that won't collide with anything upstream.

#include "../../../vendor/ggml-cuda/ssm-conv.cu"

// clang-format off
// extern "C" + no static so nvcc's host-side DCE doesn't drop the
// function body before the `<<<>>>` launches trigger PTX emission.
extern "C" __host__ __attribute__((used)) __attribute__((visibility("default")))
void ctox_force_emit_ssm_conv(cudaStream_t s,
                                       const float *x, const float *w, float *y,
                                       int a, int b, int c, int d,
                                       int e, int f, int g, int64_t n) {
    // Short-token, apply_silu ∈ {0,1} × d_conv ∈ {3,4,5,9}
    ssm_conv_f32<false, 128, 3><<<1,1,0,s>>>(x, w, a, b, c, d, y, e, f, g, n);
    ssm_conv_f32<false, 128, 4><<<1,1,0,s>>>(x, w, a, b, c, d, y, e, f, g, n);
    ssm_conv_f32<false, 128, 5><<<1,1,0,s>>>(x, w, a, b, c, d, y, e, f, g, n);
    ssm_conv_f32<false, 128, 9><<<1,1,0,s>>>(x, w, a, b, c, d, y, e, f, g, n);
    ssm_conv_f32<true,  128, 3><<<1,1,0,s>>>(x, w, a, b, c, d, y, e, f, g, n);
    ssm_conv_f32<true,  128, 4><<<1,1,0,s>>>(x, w, a, b, c, d, y, e, f, g, n);
    ssm_conv_f32<true,  128, 5><<<1,1,0,s>>>(x, w, a, b, c, d, y, e, f, g, n);
    ssm_conv_f32<true,  128, 9><<<1,1,0,s>>>(x, w, a, b, c, d, y, e, f, g, n);
    // Long-token, apply_silu ∈ {0,1} × d_conv ∈ {3,4,5,9} × split_n_t=32
    ssm_conv_long_token_f32<false, 128, 3, 32><<<1,1,0,s>>>(x, w, a, b, c, d, y, e, f, g, n);
    ssm_conv_long_token_f32<false, 128, 4, 32><<<1,1,0,s>>>(x, w, a, b, c, d, y, e, f, g, n);
    ssm_conv_long_token_f32<false, 128, 5, 32><<<1,1,0,s>>>(x, w, a, b, c, d, y, e, f, g, n);
    ssm_conv_long_token_f32<false, 128, 9, 32><<<1,1,0,s>>>(x, w, a, b, c, d, y, e, f, g, n);
    ssm_conv_long_token_f32<true,  128, 3, 32><<<1,1,0,s>>>(x, w, a, b, c, d, y, e, f, g, n);
    ssm_conv_long_token_f32<true,  128, 4, 32><<<1,1,0,s>>>(x, w, a, b, c, d, y, e, f, g, n);
    ssm_conv_long_token_f32<true,  128, 5, 32><<<1,1,0,s>>>(x, w, a, b, c, d, y, e, f, g, n);
    ssm_conv_long_token_f32<true,  128, 9, 32><<<1,1,0,s>>>(x, w, a, b, c, d, y, e, f, g, n);
}
// clang-format on
