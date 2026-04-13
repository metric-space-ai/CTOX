#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <stdio.h>

#include "attention/dtype_bfloat16.cuh"
#include "attention/dtype_float16.cuh"
#include "attention/dtype_float32.cuh"
#include "cuda_compat.h"

#ifdef USE_ROCM
#include "quantization/fp8/amd/quant_utils.cuh"
#else
#include "quantization/fp8/nvidia/quant_utils.cuh"
#endif

#include <algorithm>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err));                                        \
      exit(err);                                                               \
    }                                                                          \
  } while (0)

namespace vllm {

constexpr int TURBO3_BLOCK_SIZE = 32;
constexpr int TURBO3_GROUP_SIZE = 128;
constexpr int TURBO3_BLOCK_BYTES = 14;
constexpr int TURBO3_GROUP_BYTES = 56;

__device__ __constant__ float kTurbo3Centroids[8] = {
    -0.190685f, -0.117832f, -0.065717f, -0.021460f,
     0.021460f,  0.065717f,  0.117832f,  0.190685f,
};

__device__ __constant__ float kTurboWhtSigns1[128] = {
    -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
     1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f,
    -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f,
     1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f,
    -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f,
     1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f,
    -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f,
     1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f,
};

__device__ __constant__ float kTurboWhtSigns2[128] = {
     1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f,
     1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f,
     1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f,
     1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f,
     1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f,
    -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f,
     1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f,
    -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f,
};

template <typename out_t>
__device__ __forceinline__ void store_from_float(out_t* dst, float value) {
  from_float(*dst, value);
}

template <>
__device__ __forceinline__ void store_from_float<float>(float* dst, float value) {
  *dst = value;
}

__device__ __forceinline__ void fwht_128_inplace(float* values) {
  for (int h = 1; h < TURBO3_GROUP_SIZE; h *= 2) {
    const int step = h * 2;
    for (int i = 0; i < TURBO3_GROUP_SIZE; i += step) {
      for (int j = i; j < i + h; ++j) {
        const float a = values[j];
        const float b = values[j + h];
        values[j] = a + b;
        values[j + h] = a - b;
      }
    }
  }
  constexpr float scale = 1.0f / 11.313708498984761f; // 1/sqrt(128)
  for (int i = 0; i < TURBO3_GROUP_SIZE; ++i) {
    values[i] *= scale;
  }
}

__device__ __forceinline__ float turbo3_half_to_float(const uint8_t lo,
                                                      const uint8_t hi) {
  const uint16_t bits = static_cast<uint16_t>(lo) |
                        (static_cast<uint16_t>(hi) << 8);
  return half_to_float(bits);
}

__device__ __forceinline__ void turbo3_decode_group(const uint8_t* packed_group,
                                                    float* out_group) {
  for (int block = 0; block < 4; ++block) {
    const uint8_t* block_ptr = packed_group + block * TURBO3_BLOCK_BYTES;
    const float norm = turbo3_half_to_float(block_ptr[0], block_ptr[1]);
    const uint8_t* low = block_ptr + 2;
    const uint8_t* high = block_ptr + 10;
    for (int j = 0; j < TURBO3_BLOCK_SIZE; ++j) {
      const uint8_t low2 = (low[j / 4] >> ((j % 4) * 2)) & 0x3;
      const uint8_t hi1 = (high[j / 8] >> (j % 8)) & 0x1;
      const uint8_t idx = low2 | (hi1 << 2);
      out_group[block * TURBO3_BLOCK_SIZE + j] =
          kTurbo3Centroids[idx] * norm;
    }
  }
}

template <typename out_t>
__global__ void gather_kv_cache_turbo3_kernel(
    const uint8_t *__restrict__ key_cache,
    const uint8_t *__restrict__ value_cache,
    out_t *__restrict__ k_out,
    out_t *__restrict__ v_out,
    const int32_t *__restrict__ block_table,
    const int32_t *__restrict__ cu_seq_lens,
    const int32_t num_tokens,
    const int32_t num_seqs,
    const int32_t block_size,
    const int32_t block_table_stride,
    const int32_t num_kv_heads,
    const int32_t head_size,
    const int32_t payload_bytes) {
  const int32_t token_id = blockIdx.x;
  const int32_t work_id = threadIdx.x;
  if (token_id >= num_tokens) {
    return;
  }

  const int32_t groups_per_head = head_size / TURBO3_GROUP_SIZE;
  const int32_t work_items = num_kv_heads * groups_per_head;
  if (work_id >= work_items) {
    return;
  }

  int32_t lo = 0, hi = num_seqs;
  while (lo < hi) {
    int32_t mid = (lo + hi + 1) / 2;
    if (cu_seq_lens[mid] <= token_id) {
      lo = mid;
    } else {
      hi = mid - 1;
    }
  }
  const int32_t batch_id = lo;
  const int32_t batch_offset = token_id - cu_seq_lens[batch_id];
  const int32_t block_table_id = batch_offset / block_size;
  const int32_t slot = batch_offset % block_size;
  const int32_t block_id =
      block_table[batch_id * block_table_stride + block_table_id];

  const int32_t head_idx = work_id / groups_per_head;
  const int32_t group_idx = work_id % groups_per_head;
  const int32_t group_byte_offset = group_idx * TURBO3_GROUP_BYTES;

  const int64_t cache_head_base =
      (static_cast<int64_t>(block_id) * num_kv_heads + head_idx) *
      payload_bytes * block_size;
  const int64_t out_base =
      (static_cast<int64_t>(token_id) * num_kv_heads + head_idx) * head_size +
      group_idx * TURBO3_GROUP_SIZE;

  uint8_t k_packed[TURBO3_GROUP_BYTES];
  uint8_t v_packed[TURBO3_GROUP_BYTES];
  #pragma unroll
  for (int i = 0; i < TURBO3_GROUP_BYTES; ++i) {
    const int64_t byte_index =
        static_cast<int64_t>(group_byte_offset + i) * block_size + slot;
    k_packed[i] = key_cache[cache_head_base + byte_index];
    v_packed[i] = value_cache[cache_head_base + byte_index];
  }

  float k_group[TURBO3_GROUP_SIZE];
  float v_group[TURBO3_GROUP_SIZE];
  turbo3_decode_group(k_packed, k_group);
  turbo3_decode_group(v_packed, v_group);

  #pragma unroll
  for (int i = 0; i < TURBO3_GROUP_SIZE; ++i) {
    store_from_float(&k_out[out_base + i], k_group[i]);
    store_from_float(&v_out[out_base + i], v_group[i]);
  }
}

/// Gather K and V from paged KV cache into contiguous output tensors.
///
/// One CUDA block per output token, 256 threads cooperatively copy
/// kv_heads * head_size elements for both K and V.
///
/// Uses binary search on cu_seq_lens to find batch_id, avoiding a
/// separate token_to_seq tensor.
///
/// K cache layout: [num_blocks, kv_heads, head_size/x, block_size, x]
/// V cache layout: [num_blocks, kv_heads, head_size, block_size]
/// K/V output:     [num_tokens, kv_heads, head_size]
template <typename cache_t, typename out_t, Fp8KVCacheDataType kv_dt>
__global__ void gather_kv_cache_kernel(
    const cache_t *__restrict__ key_cache,   // [num_blocks, kv_heads,
                                             //  head_size/x, block_size, x]
    const cache_t *__restrict__ value_cache, // [num_blocks, kv_heads,
                                             //  head_size, block_size]
    out_t *__restrict__ k_out,         // [num_tokens, kv_heads, head_size]
    out_t *__restrict__ v_out,         // [num_tokens, kv_heads, head_size]
    const float *__restrict__ k_scale, // scalar or nullptr
    const float *__restrict__ v_scale, // scalar or nullptr
    const int32_t *__restrict__ block_table, // [batch, max_blocks]
    const int32_t *__restrict__ cu_seq_lens, // [batch + 1]
    const int32_t num_tokens, const int32_t num_seqs, const int32_t block_size,
    const int32_t block_table_stride, const int32_t num_kv_heads,
    const int32_t head_size, const int32_t x) {
  const int32_t token_id = blockIdx.x;
  if (token_id >= num_tokens) {
    return;
  }

  // Binary search cu_seq_lens to find batch_id.
  // cu_seq_lens is [batch+1] with cumulative token counts.
  // We want the largest i such that cu_seq_lens[i] <= token_id.
  int32_t lo = 0, hi = num_seqs;
  while (lo < hi) {
    int32_t mid = (lo + hi + 1) / 2;
    if (cu_seq_lens[mid] <= token_id) {
      lo = mid;
    } else {
      hi = mid - 1;
    }
  }
  const int32_t batch_id = lo;

  const int32_t batch_offset = token_id - cu_seq_lens[batch_id];
  const int32_t block_table_id = batch_offset / block_size;
  const int32_t slot = batch_offset % block_size;
  const int32_t block_id =
      block_table[batch_id * block_table_stride + block_table_id];

  const int32_t n = num_kv_heads * head_size;
  const int64_t out_base =
      static_cast<int64_t>(token_id) * num_kv_heads * head_size;

  // Precompute strides
  const int64_t k_block_stride =
      static_cast<int64_t>(num_kv_heads) * (head_size / x) * block_size * x;
  const int64_t k_head_stride =
      static_cast<int64_t>(head_size / x) * block_size * x;
  const int64_t v_block_stride =
      static_cast<int64_t>(num_kv_heads) * head_size * block_size;
  const int64_t v_head_stride = static_cast<int64_t>(head_size) * block_size;

  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const int head_idx = i / head_size;
    const int d = i % head_size;

    // K: [block_id, head_idx, d/x, slot, d%x]
    const int x_idx = d / x;
    const int x_offset = d % x;
    const int64_t k_src_idx = static_cast<int64_t>(block_id) * k_block_stride +
                              head_idx * k_head_stride +
                              x_idx * block_size * x + slot * x + x_offset;

    // V: [block_id, head_idx, d, slot]
    const int64_t v_src_idx = static_cast<int64_t>(block_id) * v_block_stride +
                              head_idx * v_head_stride + d * block_size + slot;

    if constexpr (kv_dt == Fp8KVCacheDataType::kAuto) {
      k_out[out_base + i] = key_cache[k_src_idx];
      v_out[out_base + i] = value_cache[v_src_idx];
    } else {
      k_out[out_base + i] = fp8::scaled_convert<out_t, cache_t, kv_dt>(
          key_cache[k_src_idx], *k_scale);
      v_out[out_base + i] = fp8::scaled_convert<out_t, cache_t, kv_dt>(
          value_cache[v_src_idx], *v_scale);
    }
  }
}

} // namespace vllm

#define CALL_GATHER_KV_CACHE(OUT_T, CACHE_T, KV_DTYPE)                         \
  vllm::gather_kv_cache_kernel<CACHE_T, OUT_T, KV_DTYPE>                       \
      <<<grid, block, 0, stream>>>(                                            \
          reinterpret_cast<CACHE_T *>(key_cache),                              \
          reinterpret_cast<CACHE_T *>(value_cache),                            \
          reinterpret_cast<OUT_T *>(k_out), reinterpret_cast<OUT_T *>(v_out),  \
          reinterpret_cast<const float *>(k_scale),                            \
          reinterpret_cast<const float *>(v_scale), block_table, cu_seq_lens,  \
          num_tokens, num_seqs, block_size, block_table_stride, num_kv_heads,  \
          head_size, x);

extern "C" void gather_kv_cache(
    void *key_cache,   // [num_blocks, kv_heads, head_size/x, block_size, x]
    void *value_cache, // [num_blocks, kv_heads, head_size, block_size]
    void *k_out,       // [num_tokens, kv_heads, head_size]
    void *v_out,       // [num_tokens, kv_heads, head_size]
    void *k_scale,     // scalar or nullptr
    void *v_scale,     // scalar or nullptr
    const int32_t *block_table, // [batch, max_blocks]
    const int32_t *cu_seq_lens, // [batch + 1]
    int32_t num_tokens, int32_t num_seqs, int32_t block_size,
    int32_t block_table_stride, int32_t num_kv_heads, int32_t head_size,
    int32_t x, cudaStream_t stream,
    uint32_t out_dtype,  // 0 => f16; 1 => bf16; 2 => f32
    uint32_t cache_dtype // 0 => f16; 1 => bf16; 2 => f32; 3 => fp8_e4m3; 4 => turboquant3
) {
  if (num_tokens <= 0) {
    return;
  }
  dim3 grid(num_tokens);

  if (cache_dtype == 4) {
    const int32_t groups_per_head = head_size / vllm::TURBO3_GROUP_SIZE;
    dim3 block(std::min(num_kv_heads * groups_per_head, 512));
    const int32_t payload_bytes =
        (head_size / vllm::TURBO3_BLOCK_SIZE) * vllm::TURBO3_BLOCK_BYTES;
    if (out_dtype == 0) {
      vllm::gather_kv_cache_turbo3_kernel<uint16_t>
          <<<grid, block, 0, stream>>>(
              reinterpret_cast<uint8_t *>(key_cache),
              reinterpret_cast<uint8_t *>(value_cache),
              reinterpret_cast<uint16_t *>(k_out),
              reinterpret_cast<uint16_t *>(v_out),
              block_table, cu_seq_lens, num_tokens, num_seqs, block_size,
              block_table_stride, num_kv_heads, head_size, payload_bytes);
    } else if (out_dtype == 1) {
      vllm::gather_kv_cache_turbo3_kernel<__nv_bfloat16>
          <<<grid, block, 0, stream>>>(
              reinterpret_cast<uint8_t *>(key_cache),
              reinterpret_cast<uint8_t *>(value_cache),
              reinterpret_cast<__nv_bfloat16 *>(k_out),
              reinterpret_cast<__nv_bfloat16 *>(v_out),
              block_table, cu_seq_lens, num_tokens, num_seqs, block_size,
              block_table_stride, num_kv_heads, head_size, payload_bytes);
    } else if (out_dtype == 2) {
      vllm::gather_kv_cache_turbo3_kernel<float><<<grid, block, 0, stream>>>(
          reinterpret_cast<uint8_t *>(key_cache),
          reinterpret_cast<uint8_t *>(value_cache),
          reinterpret_cast<float *>(k_out), reinterpret_cast<float *>(v_out),
          block_table, cu_seq_lens, num_tokens, num_seqs, block_size,
          block_table_stride, num_kv_heads, head_size, payload_bytes);
    }
  } else if (cache_dtype == 3) {
    dim3 block(std::min(num_kv_heads * head_size, 512));
    // FP8 E4M3 cache -> dequantize to out_dtype
    if (out_dtype == 0) {
      CALL_GATHER_KV_CACHE(uint16_t, uint8_t,
                           vllm::Fp8KVCacheDataType::kFp8E4M3);
    } else if (out_dtype == 1) {
      CALL_GATHER_KV_CACHE(__nv_bfloat16, uint8_t,
                           vllm::Fp8KVCacheDataType::kFp8E4M3);
    } else if (out_dtype == 2) {
      CALL_GATHER_KV_CACHE(float, uint8_t, vllm::Fp8KVCacheDataType::kFp8E4M3);
    }
  } else {
    dim3 block(std::min(num_kv_heads * head_size, 512));
    // Non-FP8 cache: cache_t == out_t
    if (out_dtype == 0) {
      CALL_GATHER_KV_CACHE(uint16_t, uint16_t, vllm::Fp8KVCacheDataType::kAuto);
    } else if (out_dtype == 1) {
      CALL_GATHER_KV_CACHE(__nv_bfloat16, __nv_bfloat16,
                           vllm::Fp8KVCacheDataType::kAuto);
    } else if (out_dtype == 2) {
      CALL_GATHER_KV_CACHE(float, float, vllm::Fp8KVCacheDataType::kAuto);
    }
  }
  CUDA_CHECK(cudaGetLastError());
}
