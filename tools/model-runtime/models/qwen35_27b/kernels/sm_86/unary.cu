// Vendor-backed unary activations.
//
// Pulls in the upstream ggml-cuda unary header for its canonical SiLU
// device primitive (`ggml_cuda_op_silu_single`). The upstream
// `unary_op_kernel<op_silu, T>` template is the reference implementation;
// because our runtime uses a flat-buffer (ptr, ptr, numel) ABI instead of
// ggml_tensor descriptors, we expose a pair of thin `extern "C"
// __global__` entry points that call the upstream __device__ primitive
// directly. This keeps the fp32 SiLU math byte-identical to upstream
// while letting cudarc load the kernel by its unmangled C name.
//
// Shapes:
//   x, y : [numel] flat — any shape works as long as x and y match.
//
// Launch convention:
//   grid  = (ceil(numel / 256), 1, 1)
//   block = (256, 1, 1)
//   shmem = 0
//
// Extern "C" entry points:
//   * silu_f32   — float32 in/out
//   * silu_bf16  — __nv_bfloat16 in/out, math done in f32
// Load via cudarc's `module.load_function("silu_f32" | "silu_bf16")`.

#include <cuda_bf16.h>

// Upstream SiLU device primitive:
//   __device__ __forceinline__ float ggml_cuda_op_silu_single(float x) {
//       return x / (1.0f + expf(-x));
//   }
#include "../../vendor/ggml-cuda/unary.cuh"

// ---------------------------------------------------------------------------
// f32 path
// ---------------------------------------------------------------------------

extern "C" __global__ void silu_f32(
    const float * __restrict__ x,
    float * __restrict__ y,
    int numel
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) {
        return;
    }
    y[i] = ggml_cuda_op_silu_single(x[i]);
}

// ---------------------------------------------------------------------------
// bf16 path — promote to f32 for the sigmoid, demote on store.
// ---------------------------------------------------------------------------

extern "C" __global__ void silu_bf16(
    const __nv_bfloat16 * __restrict__ x,
    __nv_bfloat16 * __restrict__ y,
    int numel
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) {
        return;
    }
    const float xf = __bfloat162float(x[i]);
    y[i] = __float2bfloat16(ggml_cuda_op_silu_single(xf));
}
