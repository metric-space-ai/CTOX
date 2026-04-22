// Strided head gather / scatter for the full-attention layer.
//
// The attention loop used to walk every token with a separate
// `memcpy_dtod` per token (24 Q-heads × ~1024 tokens × 4 calls per
// head = ~100k launches per FA layer just for the per-head staging).
// These two kernels collapse that into a single kernel launch per
// head call: one thread per destination element, reading from the
// strided source position and writing to the contiguous destination
// (or vice-versa for scatter).
//
// Entry points:
//   * head_gather_bf16   — [n_tokens, n_heads, head_dim] → [n_tokens, head_dim]
//                          for packed-per-token activations.
//   * head_scatter_bf16  — [n_tokens, head_dim] → [n_tokens, n_heads, head_dim]
//                          mirror of gather.
//   * head_gather_slab_bf16 — same as gather_bf16 but the outer dim is
//                             `kv_len`, layout [max_ctx, n_kv_heads, head_dim]
//                             and we only read the first `kv_len` rows.
//                             (`max_ctx` is not needed as an argument because
//                             the source row stride is still `n_heads *
//                             head_dim` — leading unused rows inside the slab
//                             are simply not touched.)
//
// Launch convention (all three):
//   total = n_rows * head_dim
//   grid  = (ceil(total / 256), 1, 1)
//   block = (256, 1, 1)
//   shmem = 0

#include <cuda_bf16.h>

extern "C" __global__ void head_gather_bf16(
    const __nv_bfloat16 * __restrict__ src,  // [n_tokens, n_heads, head_dim]
    __nv_bfloat16 * __restrict__ dst,        // [n_tokens, head_dim]
    int n_tokens,
    int n_heads,
    int head_dim,
    int head
) {
    const int total = n_tokens * head_dim;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int t = idx / head_dim;
    const int d = idx - t * head_dim;
    dst[idx] = src[(t * n_heads + head) * head_dim + d];
}

extern "C" __global__ void head_scatter_bf16(
    const __nv_bfloat16 * __restrict__ src,  // [n_tokens, head_dim]
    __nv_bfloat16 * __restrict__ dst,        // [n_tokens, n_heads, head_dim]
    int n_tokens,
    int n_heads,
    int head_dim,
    int head
) {
    const int total = n_tokens * head_dim;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int t = idx / head_dim;
    const int d = idx - t * head_dim;
    dst[(t * n_heads + head) * head_dim + d] = src[idx];
}

extern "C" __global__ void head_gather_slab_bf16(
    const __nv_bfloat16 * __restrict__ src,  // [*, n_kv_heads, head_dim]
    __nv_bfloat16 * __restrict__ dst,        // [kv_len, head_dim]
    int kv_len,
    int n_kv_heads,
    int head_dim,
    int head
) {
    const int total = kv_len * head_dim;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int t = idx / head_dim;
    const int d = idx - t * head_dim;
    dst[idx] = src[(t * n_kv_heads + head) * head_dim + d];
}
