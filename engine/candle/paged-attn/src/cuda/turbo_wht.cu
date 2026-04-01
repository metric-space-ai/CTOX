#include <stdint.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "cuda_compat.h"

namespace {

__device__ __constant__ float d_turbo_wht_signs1[128] = {
    -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f,
    1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f,
    1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f,
    1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f,
    1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f,
    -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f,
    1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f,
    1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f,
    1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f,
    1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f,
    1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f,
};

__device__ __constant__ float d_turbo_wht_signs2[128] = {
    1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f,
    -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f,
    1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f,
    -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f,
    -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f,
    -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f,
    -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f,
    -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f,
    -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f,
    1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f,
    -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f,
};

template <typename T>
__device__ __forceinline__ float load_val(const T* ptr, const int idx);

template <>
__device__ __forceinline__ float load_val<float>(const float* ptr, const int idx) {
  return ptr[idx];
}

template <>
__device__ __forceinline__ float load_val<half>(const half* ptr, const int idx) {
  return __half2float(ptr[idx]);
}

template <>
__device__ __forceinline__ float load_val<__nv_bfloat16>(const __nv_bfloat16* ptr, const int idx) {
  return __bfloat162float(ptr[idx]);
}

template <typename T>
__device__ __forceinline__ void store_val(T* ptr, const int idx, const float value);

template <>
__device__ __forceinline__ void store_val<float>(float* ptr, const int idx, const float value) {
  ptr[idx] = value;
}

template <>
__device__ __forceinline__ void store_val<half>(half* ptr, const int idx, const float value) {
  ptr[idx] = __float2half(value);
}

template <>
__device__ __forceinline__ void store_val<__nv_bfloat16>(__nv_bfloat16* ptr, const int idx, const float value) {
  ptr[idx] = __float2bfloat16(value);
}

template <typename T, bool FORWARD>
__global__ void k_turbo_wht_rotate(const T* __restrict__ src, T* __restrict__ dst,
                                   const int64_t n_elements) {
  const int64_t offset = static_cast<int64_t>(blockIdx.x) * 128;
  if (offset >= n_elements) {
    return;
  }

  __shared__ float buf[128];
  const int tid = threadIdx.x;
  if (tid < 128) {
    const float* s_pre = FORWARD ? d_turbo_wht_signs1 : d_turbo_wht_signs2;
    buf[tid] = load_val(src, offset + tid) * s_pre[tid];
  }
  __syncthreads();

  for (int h = 1; h < 128; h *= 2) {
    if (tid < 64) {
      const int j = (tid / h) * (2 * h) + (tid % h);
      const float a = buf[j];
      const float b = buf[j + h];
      buf[j] = a + b;
      buf[j + h] = a - b;
    }
    __syncthreads();
  }

  constexpr float inv_sqrt_128 = 0.08838834764831845f;
  if (tid < 128) {
    const float* s_post = FORWARD ? d_turbo_wht_signs2 : d_turbo_wht_signs1;
    store_val(dst, offset + tid, buf[tid] * inv_sqrt_128 * s_post[tid]);
  }
}

template <typename T>
void launch_rotate(const T* src, T* dst, const int64_t n_elements, cudaStream_t stream,
                   const bool forward) {
  const int blocks = static_cast<int>((n_elements + 127) / 128);
  if (forward) {
    k_turbo_wht_rotate<T, true><<<blocks, 128, 0, stream>>>(src, dst, n_elements);
  } else {
    k_turbo_wht_rotate<T, false><<<blocks, 128, 0, stream>>>(src, dst, n_elements);
  }
}

}  // namespace

extern "C" void turbo_rotate_f32(const void* src, void* dst, const long n_elements,
                                 cudaStream_t stream, const bool forward) {
  launch_rotate(reinterpret_cast<const float*>(src), reinterpret_cast<float*>(dst),
                static_cast<int64_t>(n_elements), stream, forward);
}

extern "C" void turbo_rotate_f16(const void* src, void* dst, const long n_elements,
                                 cudaStream_t stream, const bool forward) {
  launch_rotate(reinterpret_cast<const half*>(src), reinterpret_cast<half*>(dst),
                static_cast<int64_t>(n_elements), stream, forward);
}

extern "C" void turbo_rotate_bf16(const void* src, void* dst, const long n_elements,
                                  cudaStream_t stream, const bool forward) {
  launch_rotate(reinterpret_cast<const __nv_bfloat16*>(src),
                reinterpret_cast<__nv_bfloat16*>(dst),
                static_cast<int64_t>(n_elements), stream, forward);
}
