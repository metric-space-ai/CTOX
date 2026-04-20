/**
 * @file moe_reduce.cu
 * @brief Fused MoE-output reduction: dtype-cast + topk-weight broadcast-mul +
 *        topk-axis sum, in a single CUDA kernel.
 *
 * After the 3 gather matmuls of the MoE MLP (gate, up, down), candle's
 * `forward_fast` combines the top-k outputs via:
 *
 *     ys.to_dtype(F32).broadcast_mul(&topk_weights).sum(axis=-2).to_dtype(T)
 *
 * That's four CUDA kernels per MoE layer (cast, bcast_mul, reduce, cast_back)
 * — on a 40-layer MoE this is 160 launches/token before the rest of the
 * forward even runs. Fusing them into one kernel cuts that to 40. Each token
 * is independent, each output element is a small reduction over `topk`, so
 * the pattern is embarrassingly parallel and fits one block per (token, tile)
 * comfortably.
 *
 * Shapes at call time (decode, batch=1, max_seq_len=2048):
 *   ys            : [num_tokens, topk, hidden_dim]        (T = bf16 or f16)
 *   topk_weights  : [num_tokens, topk]                    (f32)
 *   out           : [num_tokens, hidden_dim]              (T = input dtype)
 *
 *   num_tokens   ≥ 1
 *   topk         ≤ 16  (Qwen3.6 = 8)
 *   hidden_dim   = multiple of 2 (2048 for Qwen3.6)
 *
 * One block per (token, hidden-tile); each thread handles ELEMS_PER_THREAD
 * contiguous hidden-dim elements. Per element, sum across topk accumulated
 * in f32, then cast back to T once at the end.
 */

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace ctox_moe_reduce {

template <typename T>
__device__ __forceinline__ float to_float(T x);
template <>
__device__ __forceinline__ float to_float<__nv_bfloat16>(__nv_bfloat16 x) {
    return __bfloat162float(x);
}
template <>
__device__ __forceinline__ float to_float<__half>(__half x) {
    return __half2float(x);
}

template <typename T>
__device__ __forceinline__ T from_float(float x);
template <>
__device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float x) {
    return __float2bfloat16(x);
}
template <>
__device__ __forceinline__ __half from_float<__half>(float x) {
    return __float2half(x);
}

// ELEMS_PER_THREAD = 4: 4 contiguous hidden-dim scalars per thread. For
// hidden_dim=2048, BLOCK_SIZE=256 ⇒ 2 tiles per token ⇒ 2 blocks covering
// hidden_dim per token. Keeps the kernel launch cheap while giving ILP.
template <typename T, int ELEMS_PER_THREAD = 4>
__global__ void moe_reduce_kernel(
    const T *__restrict__ ys,               // [num_tokens, topk, hidden_dim]
    const float *__restrict__ topk_weights, // [num_tokens, topk]
    T *__restrict__ out,                    // [num_tokens, hidden_dim]
    const int num_tokens, const int topk, const int hidden_dim) {
    const int tok = blockIdx.y;
    const int tile_start = blockIdx.x * blockDim.x * ELEMS_PER_THREAD;
    const int thread_offset = threadIdx.x * ELEMS_PER_THREAD;
    const int base = tile_start + thread_offset;

    if (tok >= num_tokens || base >= hidden_dim) return;

    // Accumulate in f32. `topk` is small (≤ 16); fully unroll-friendly.
    float acc[ELEMS_PER_THREAD];
#pragma unroll
    for (int e = 0; e < ELEMS_PER_THREAD; ++e) acc[e] = 0.0f;

    for (int k = 0; k < topk; ++k) {
        const float w = topk_weights[tok * topk + k];
        const T *row = ys + (tok * topk + k) * hidden_dim;
#pragma unroll
        for (int e = 0; e < ELEMS_PER_THREAD; ++e) {
            const int h = base + e;
            if (h < hidden_dim) {
                acc[e] += w * to_float<T>(row[h]);
            }
        }
    }

    T *out_row = out + tok * hidden_dim;
#pragma unroll
    for (int e = 0; e < ELEMS_PER_THREAD; ++e) {
        const int h = base + e;
        if (h < hidden_dim) {
            out_row[h] = from_float<T>(acc[e]);
        }
    }
}

} // namespace ctox_moe_reduce

extern "C" void moe_weighted_sum_bf16(
    const void *ys, const void *topk_weights, void *out,
    int32_t num_tokens, int32_t topk, int32_t hidden_dim, int64_t stream) {
    constexpr int BLOCK_SIZE = 256;
    constexpr int ELEMS_PER_THREAD = 4;
    const int tiles = (hidden_dim + BLOCK_SIZE * ELEMS_PER_THREAD - 1) /
                      (BLOCK_SIZE * ELEMS_PER_THREAD);
    dim3 grid(tiles, num_tokens);
    dim3 block(BLOCK_SIZE);
    ctox_moe_reduce::moe_reduce_kernel<__nv_bfloat16, ELEMS_PER_THREAD>
        <<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
            reinterpret_cast<const __nv_bfloat16 *>(ys),
            reinterpret_cast<const float *>(topk_weights),
            reinterpret_cast<__nv_bfloat16 *>(out),
            num_tokens, topk, hidden_dim);
}

extern "C" void moe_weighted_sum_f16(
    const void *ys, const void *topk_weights, void *out,
    int32_t num_tokens, int32_t topk, int32_t hidden_dim, int64_t stream) {
    constexpr int BLOCK_SIZE = 256;
    constexpr int ELEMS_PER_THREAD = 4;
    const int tiles = (hidden_dim + BLOCK_SIZE * ELEMS_PER_THREAD - 1) /
                      (BLOCK_SIZE * ELEMS_PER_THREAD);
    dim3 grid(tiles, num_tokens);
    dim3 block(BLOCK_SIZE);
    ctox_moe_reduce::moe_reduce_kernel<__half, ELEMS_PER_THREAD>
        <<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
            reinterpret_cast<const __half *>(ys),
            reinterpret_cast<const float *>(topk_weights),
            reinterpret_cast<__half *>(out),
            num_tokens, topk, hidden_dim);
}
