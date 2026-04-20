//! Rust wrapper for the fused MoE router CUDA kernel (`moe_router.cu`).
//! Replaces candle's 6-op router chain
//!   `softmax -> arg_sort -> narrow -> contiguous -> gather -> broadcast_div`
//! with a single kernel launch that produces `topk_ids` + (optionally
//! normalized) `topk_weights`.

use candle_core::{DType, Device, Result, Tensor};

/// Returns `(topk_ids, topk_weights)`.
///   - `router_logits`: `(num_tokens, num_experts)`, BF16 or F16, on CUDA
///   - `topk_ids`:      `(num_tokens, top_k)` u32
///   - `topk_weights`:  `(num_tokens, top_k)` f32
#[cfg(feature = "cuda")]
pub fn moe_router_cuda(
    router_logits: &Tensor,
    top_k: usize,
    norm_topk_prob: bool,
) -> Result<(Tensor, Tensor)> {
    use candle_core as candle;
    use candle_core::cuda_backend::cudarc::driver::DevicePtr;
    use core::ffi::c_void;

    let (num_tokens, num_experts) = router_logits.dims2()?;
    if !matches!(num_experts, 64 | 128 | 256) {
        candle::bail!(
            "moe_router_cuda: unsupported num_experts={num_experts} \
             (kernel is compile-time specialized for 64/128/256)"
        );
    }
    if top_k > 16 {
        candle::bail!("moe_router_cuda: top_k={top_k} > 16 (kernel limit)");
    }

    let dev = match router_logits.device() {
        Device::Cuda(d) => d.clone(),
        other => candle::bail!("moe_router_cuda: expected CUDA tensor, got {other:?}"),
    };

    let logits = router_logits.contiguous()?;
    let (ls_s, ls_l) = logits.storage_and_layout();
    let ls_offset = ls_l.start_offset();
    let stream_id = dev.cuda_stream().cu_stream() as i64;

    let ids_buf = unsafe { dev.alloc::<u32>(num_tokens * top_k) }?;
    let w_buf = unsafe { dev.alloc::<f32>(num_tokens * top_k) }?;

    match logits.dtype() {
        DType::BF16 => {
            let ls_s = match &*ls_s {
                candle::Storage::Cuda(c) => c.as_cuda_slice::<half::bf16>()?,
                _ => candle::bail!("router_logits storage must be cuda"),
            };
            unsafe {
                crate::cuda::ffi::moe_router_bf16(
                    ls_s.slice(ls_offset..).device_ptr(ls_s.stream()).0 as *const c_void,
                    ids_buf.device_ptr(ids_buf.stream()).0 as *mut c_void,
                    w_buf.device_ptr(w_buf.stream()).0 as *mut c_void,
                    num_tokens as i32,
                    num_experts as i32,
                    top_k as i32,
                    if norm_topk_prob { 1 } else { 0 },
                    stream_id,
                );
            }
        }
        DType::F16 => {
            let ls_s = match &*ls_s {
                candle::Storage::Cuda(c) => c.as_cuda_slice::<half::f16>()?,
                _ => candle::bail!("router_logits storage must be cuda"),
            };
            unsafe {
                crate::cuda::ffi::moe_router_f16(
                    ls_s.slice(ls_offset..).device_ptr(ls_s.stream()).0 as *const c_void,
                    ids_buf.device_ptr(ids_buf.stream()).0 as *mut c_void,
                    w_buf.device_ptr(w_buf.stream()).0 as *mut c_void,
                    num_tokens as i32,
                    num_experts as i32,
                    top_k as i32,
                    if norm_topk_prob { 1 } else { 0 },
                    stream_id,
                );
            }
        }
        other => candle::bail!("moe_router_cuda: unsupported dtype {other:?}"),
    }

    let ids_storage = candle::CudaStorage::wrap_cuda_slice(ids_buf, dev.clone());
    let w_storage = candle::CudaStorage::wrap_cuda_slice(w_buf, dev);
    let topk_ids = Tensor::from((
        candle::Storage::Cuda(ids_storage),
        (num_tokens, top_k),
    ));
    let topk_weights = Tensor::from((candle::Storage::Cuda(w_storage), (num_tokens, top_k)));
    Ok((topk_ids, topk_weights))
}

#[cfg(not(feature = "cuda"))]
#[allow(unused)]
pub fn moe_router_cuda(
    _router_logits: &Tensor,
    _top_k: usize,
    _norm_topk_prob: bool,
) -> Result<(Tensor, Tensor)> {
    candle_core::bail!("moe_router_cuda requires the cuda feature")
}
