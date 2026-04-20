/**
 * @file moe_router.cu
 * @brief Fused MoE router: softmax + top-k + optional normalize,
 *        in a single CUDA kernel.
 *
 * Replaces the candle chain per MoE layer
 *     softmax(logits) → arg_sort_last_dim → narrow(0, topk)
 *     → contiguous → gather(weights, topk_ids) → broadcast_div(sum)
 * which is 6 kernel launches; this kernel emits 1. Over a 40-layer MoE
 * decode step that's 200 fewer launches per token.
 *
 * Shapes:
 *   router_logits : [num_tokens, num_experts]           (T = bf16 or f16)
 *   topk_ids      : [num_tokens, topk]                  (u32)
 *   topk_weights  : [num_tokens, topk]                  (f32, optionally
 *                                                        normalized to sum=1)
 *
 * Constraints:
 *   num_experts ≤ MAX_EXPERTS_PER_BLOCK (256) so a single block/thread-per-
 *   expert covers all of them. Typical MoE models: 64-256 experts.
 *   topk ≤ MAX_TOPK (16).
 *
 * Each block handles one token. Inside:
 *   1. Softmax across num_experts (block-wide reduce max + reduce sum).
 *   2. Iterative argmax: topk rounds, each round finds max over the
 *      (possibly masked) softmax vector and records index/value.
 *   3. Normalize the topk_weights by their sum if requested.
 */

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cfloat>

namespace ctox_moe_router {

constexpr int MAX_EXPERTS_PER_BLOCK = 256;
constexpr int MAX_TOPK = 16;
constexpr int WARP_SIZE = 32;

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

__device__ __forceinline__ float warp_reduce_max(float v) {
#pragma unroll
    for (int o = WARP_SIZE / 2; o > 0; o >>= 1) {
        float other = __shfl_xor_sync(0xffffffff, v, o, WARP_SIZE);
        v = v > other ? v : other;
    }
    return v;
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
    for (int o = WARP_SIZE / 2; o > 0; o >>= 1)
        v += __shfl_xor_sync(0xffffffff, v, o, WARP_SIZE);
    return v;
}

// BLOCK_SIZE = num_experts, one thread per expert. num_experts must be a
// multiple of WARP_SIZE and ≤ MAX_EXPERTS_PER_BLOCK.
template <typename T, int BLOCK_SIZE>
__global__ void router_kernel(
    const T *__restrict__ logits,        // [num_tokens, num_experts]
    uint32_t *__restrict__ topk_ids,     // [num_tokens, topk]
    float *__restrict__ topk_weights,    // [num_tokens, topk]
    const int num_tokens, const int num_experts, const int topk,
    const bool norm_topk_prob) {
    const int tok = blockIdx.x;
    if (tok >= num_tokens) return;

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;
    const int num_warps = BLOCK_SIZE / WARP_SIZE;

    __shared__ float s_softmax[MAX_EXPERTS_PER_BLOCK];
    __shared__ float s_warp_reduce[32]; // plenty for num_warps ≤ 32
    __shared__ int s_argmax_idx;
    __shared__ float s_argmax_val;

    // 1. Load logits, convert to f32 in shared.
    float logit = -FLT_MAX;
    if (tid < num_experts) {
        logit = to_float<T>(logits[tok * num_experts + tid]);
    }

    // 2. Block-wide max reduce.
    float max_val = warp_reduce_max(logit);
    if (lane == 0) s_warp_reduce[warp_id] = max_val;
    __syncthreads();
    if (warp_id == 0) {
        float v = (lane < num_warps) ? s_warp_reduce[lane] : -FLT_MAX;
        v = warp_reduce_max(v);
        if (lane == 0) s_warp_reduce[0] = v;
    }
    __syncthreads();
    float block_max = s_warp_reduce[0];

    // 3. exp(logit - max), block-wide sum.
    float e = 0.0f;
    if (tid < num_experts) {
        e = __expf(logit - block_max);
        s_softmax[tid] = e;
    }
    float sum = warp_reduce_sum(e);
    if (lane == 0) s_warp_reduce[warp_id] = sum;
    __syncthreads();
    if (warp_id == 0) {
        float v = (lane < num_warps) ? s_warp_reduce[lane] : 0.0f;
        v = warp_reduce_sum(v);
        if (lane == 0) s_warp_reduce[0] = v;
    }
    __syncthreads();
    float block_sum = s_warp_reduce[0];

    // 4. Normalize softmax in shared.
    if (tid < num_experts) {
        s_softmax[tid] = e / block_sum;
    }
    __syncthreads();

    // 5. Iterative argmax for top-k. Each round: block-wide argmax, record
    //    index and value, zero out slot. topk is small (≤16) so 16 passes
    //    max. This is simpler and usually faster than a bitonic sort for
    //    (topk ≪ num_experts).
    float topk_accum = 0.0f;
    for (int k = 0; k < topk; ++k) {
        float val = (tid < num_experts) ? s_softmax[tid] : -FLT_MAX;

        // Warp argmax: find max value + lane's original `tid`.
        int idx = tid;
        for (int o = WARP_SIZE / 2; o > 0; o >>= 1) {
            float other_val = __shfl_xor_sync(0xffffffff, val, o, WARP_SIZE);
            int other_idx = __shfl_xor_sync(0xffffffff, idx, o, WARP_SIZE);
            if (other_val > val || (other_val == val && other_idx < idx)) {
                val = other_val;
                idx = other_idx;
            }
        }
        // warp 0 lane 0 now holds warp0's max. Write per-warp to shared.
        __shared__ float s_w_val[32];
        __shared__ int s_w_idx[32];
        if (lane == 0) {
            s_w_val[warp_id] = val;
            s_w_idx[warp_id] = idx;
        }
        __syncthreads();
        if (warp_id == 0) {
            float v = (lane < num_warps) ? s_w_val[lane] : -FLT_MAX;
            int i = (lane < num_warps) ? s_w_idx[lane] : 0;
            for (int o = WARP_SIZE / 2; o > 0; o >>= 1) {
                float ov = __shfl_xor_sync(0xffffffff, v, o, WARP_SIZE);
                int oi = __shfl_xor_sync(0xffffffff, i, o, WARP_SIZE);
                if (ov > v || (ov == v && oi < i)) { v = ov; i = oi; }
            }
            if (lane == 0) {
                s_argmax_val = v;
                s_argmax_idx = i;
            }
        }
        __syncthreads();

        int max_idx = s_argmax_idx;
        float max_softmax = s_argmax_val;
        if (tid == 0) {
            topk_ids[tok * topk + k] = (uint32_t)max_idx;
            topk_weights[tok * topk + k] = max_softmax;
        }
        // Mask this slot for next round.
        if (tid == max_idx) s_softmax[tid] = -FLT_MAX;
        __syncthreads();

        topk_accum += max_softmax;
    }

    // 6. Optional normalization so topk_weights sum to 1.
    if (norm_topk_prob && tid < topk) {
        float w = topk_weights[tok * topk + tid];
        topk_weights[tok * topk + tid] = w / topk_accum;
    }
}

} // namespace ctox_moe_router

extern "C" void moe_router_bf16(
    const void *logits, void *topk_ids, void *topk_weights,
    int32_t num_tokens, int32_t num_experts, int32_t topk,
    int32_t norm_topk_prob, int64_t stream) {
    // Pick a block size equal to num_experts, rounded up to a multiple of 32.
    // For common MoE configs (64, 128, 256 experts) this is one of 64/128/256.
    // Unsupported sizes fall through to the host caller's slow path.
    auto launch = [&](auto block_size) {
        constexpr int BS = decltype(block_size)::value;
        dim3 grid(num_tokens);
        dim3 block(BS);
        ctox_moe_router::router_kernel<__nv_bfloat16, BS>
            <<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
                reinterpret_cast<const __nv_bfloat16 *>(logits),
                reinterpret_cast<uint32_t *>(topk_ids),
                reinterpret_cast<float *>(topk_weights),
                num_tokens, num_experts, topk, norm_topk_prob != 0);
    };
    if (num_experts == 64)  launch(std::integral_constant<int, 64>{});
    else if (num_experts == 128) launch(std::integral_constant<int, 128>{});
    else if (num_experts == 256) launch(std::integral_constant<int, 256>{});
    // else: unsupported; caller will have fallen back before invoking.
}

extern "C" void moe_router_f16(
    const void *logits, void *topk_ids, void *topk_weights,
    int32_t num_tokens, int32_t num_experts, int32_t topk,
    int32_t norm_topk_prob, int64_t stream) {
    auto launch = [&](auto block_size) {
        constexpr int BS = decltype(block_size)::value;
        dim3 grid(num_tokens);
        dim3 block(BS);
        ctox_moe_router::router_kernel<__half, BS>
            <<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
                reinterpret_cast<const __half *>(logits),
                reinterpret_cast<uint32_t *>(topk_ids),
                reinterpret_cast<float *>(topk_weights),
                num_tokens, num_experts, topk, norm_topk_prob != 0);
    };
    if (num_experts == 64)  launch(std::integral_constant<int, 64>{});
    else if (num_experts == 128) launch(std::integral_constant<int, 128>{});
    else if (num_experts == 256) launch(std::integral_constant<int, 256>{});
}
