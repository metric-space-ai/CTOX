// Vendor-backed binary broadcast (add/mul) — flat-buffer specialization.
//
// The upstream `k_bin_bcast` kernel in ggml-cuda/binbcast.cu is a heavy
// templated multi-axis broadcaster parameterized over src/dst types,
// strides, and fastdiv descriptors — it's built for the full ggml_tensor
// calling convention. Our runtime only ever invokes it in the degenerate
// "no broadcast, flat buffer" case (residual add, SwiGLU gate * up), so
// we expose a trimmed set of `extern "C" __global__` specializations with
// a `(src0, src1, dst, numel)` ABI. The per-element math (`op_add`,
// `op_mul`) is byte-identical to the upstream primitives in binbcast.cu:
//
//   static __device__ __forceinline__ float op_add(const float a,
//                                                  const float b) {
//       return a + b;
//   }
//   static __device__ __forceinline__ float op_mul(const float a,
//                                                  const float b) {
//       return a * b;
//   }
//
// bf16 follows upstream's convention of promoting both operands to f32,
// running the op in f32, then demoting with `__float2bfloat16` (round-to-
// nearest-even) on store.
//
// Shapes:
//   src0, src1, dst : [numel] flat — all three must match.
//
// Launch convention:
//   grid  = (ceil(numel / 256), 1, 1)
//   block = (256, 1, 1)
//   shmem = 0
//
// Extern "C" entry points:
//   * add_f32, add_bf16 — element-wise add, used by residual.
//   * mul_f32, mul_bf16 — element-wise multiply, used by SwiGLU gate * up.
// Load via cudarc's `module.load_function(...)`.

#include <cuda_bf16.h>

// ---------------------------------------------------------------------------
// f32 paths
// ---------------------------------------------------------------------------

extern "C" __global__ void add_f32(
    const float * __restrict__ a,
    const float * __restrict__ b,
    float * __restrict__ y,
    int numel
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) {
        return;
    }
    y[i] = a[i] + b[i];
}

extern "C" __global__ void mul_f32(
    const float * __restrict__ a,
    const float * __restrict__ b,
    float * __restrict__ y,
    int numel
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) {
        return;
    }
    y[i] = a[i] * b[i];
}

// ---------------------------------------------------------------------------
// bf16 paths — f32 math accumulator, single rounding on store.
// ---------------------------------------------------------------------------

extern "C" __global__ void add_bf16(
    const __nv_bfloat16 * __restrict__ a,
    const __nv_bfloat16 * __restrict__ b,
    __nv_bfloat16 * __restrict__ y,
    int numel
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) {
        return;
    }
    const float af = __bfloat162float(a[i]);
    const float bf = __bfloat162float(b[i]);
    y[i] = __float2bfloat16(af + bf);
}

extern "C" __global__ void mul_bf16(
    const __nv_bfloat16 * __restrict__ a,
    const __nv_bfloat16 * __restrict__ b,
    __nv_bfloat16 * __restrict__ y,
    int numel
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) {
        return;
    }
    const float af = __bfloat162float(a[i]);
    const float bf = __bfloat162float(b[i]);
    y[i] = __float2bfloat16(af * bf);
}

// ---------------------------------------------------------------------------
// In-place multiply: `y[i] *= x[i]`.
//
// The SwiGLU path runs as two back-to-back kernels on the same stream —
// first silu writes `y = silu(gate)`, then the multiply needs `y *= up`.
// cudarc's safe launcher won't let us alias the same `CudaSlice` as both
// a shared and exclusive argument to one launch, so we expose an
// in-place variant with a single mutable output argument. The math is
// identical to `mul_{f32,bf16}` — it just reads from the destination
// buffer in place of a second input.
// ---------------------------------------------------------------------------

extern "C" __global__ void mul_inplace_f32(
    float * __restrict__ y,
    const float * __restrict__ x,
    int numel
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) {
        return;
    }
    y[i] = y[i] * x[i];
}

extern "C" __global__ void mul_inplace_bf16(
    __nv_bfloat16 * __restrict__ y,
    const __nv_bfloat16 * __restrict__ x,
    int numel
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) {
        return;
    }
    const float yf = __bfloat162float(y[i]);
    const float xf = __bfloat162float(x[i]);
    y[i] = __float2bfloat16(yf * xf);
}
