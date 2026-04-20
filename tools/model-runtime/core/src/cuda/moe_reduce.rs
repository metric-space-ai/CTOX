//! Rust wrapper for the fused MoE-output weighted-sum CUDA kernel
//! (`moe_reduce.cu`). Replaces the candle-op chain
//! `ys.to_dtype(F32).broadcast_mul(topk_w).sum(Minus2).to_dtype(T)` —
//! four kernel launches per MoE layer — with a single launch that
//! accumulates the top-k broadcast-multiply in f32 inside the kernel
//! and writes the cast-back output directly.
//!
//! Call signature expects:
//!   ys: (num_tokens, topk, hidden_dim), dtype BF16 or F16, CUDA
//!   topk_weights: (num_tokens, topk), dtype F32, CUDA, contiguous
//!   returns: (num_tokens, hidden_dim), same dtype as `ys`

use candle_core::{DType, Device, Result, Tensor};

#[cfg(feature = "cuda")]
pub fn moe_weighted_sum_cuda(ys: &Tensor, topk_weights: &Tensor) -> Result<Tensor> {
    use candle_core as candle;
    use candle_core::cuda_backend::cudarc::driver::DevicePtr;
    use core::ffi::c_void;

    let (num_tokens, topk, hidden_dim) = ys.dims3()?;
    let (tw_tokens, tw_topk) = topk_weights.dims2()?;
    if tw_tokens != num_tokens || tw_topk != topk {
        candle::bail!(
            "moe_weighted_sum_cuda: topk_weights shape {:?} mismatches ys {:?}",
            topk_weights.shape().dims(),
            ys.shape().dims(),
        );
    }

    let dev = match ys.device() {
        Device::Cuda(d) => d.clone(),
        other => candle::bail!("moe_weighted_sum_cuda: expected CUDA tensor, got {other:?}"),
    };

    let ys = ys.contiguous()?;
    let topk_weights = topk_weights.to_dtype(DType::F32)?.contiguous()?;

    let (ys_s, ys_l) = ys.storage_and_layout();
    let ys_offset = ys_l.start_offset();
    let (tw_s, tw_l) = topk_weights.storage_and_layout();
    let tw_offset = tw_l.start_offset();

    let stream_id = dev.cuda_stream().cu_stream() as i64;

    match ys.dtype() {
        DType::BF16 => {
            let ys_s = match &*ys_s {
                candle::Storage::Cuda(c) => c.as_cuda_slice::<half::bf16>()?,
                _ => candle::bail!("ys storage must be cuda"),
            };
            let tw_s = match &*tw_s {
                candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
                _ => candle::bail!("topk_weights storage must be cuda"),
            };
            let out_buf = unsafe { dev.alloc::<half::bf16>(num_tokens * hidden_dim) }?;
            unsafe {
                crate::cuda::ffi::moe_weighted_sum_bf16(
                    ys_s.slice(ys_offset..).device_ptr(ys_s.stream()).0 as *const c_void,
                    tw_s.slice(tw_offset..).device_ptr(tw_s.stream()).0 as *const c_void,
                    out_buf.device_ptr(out_buf.stream()).0 as *mut c_void,
                    num_tokens as i32,
                    topk as i32,
                    hidden_dim as i32,
                    stream_id,
                );
            }
            let storage = candle::CudaStorage::wrap_cuda_slice(out_buf, dev);
            Ok(Tensor::from((
                candle::Storage::Cuda(storage),
                (num_tokens, hidden_dim),
            )))
        }
        DType::F16 => {
            let ys_s = match &*ys_s {
                candle::Storage::Cuda(c) => c.as_cuda_slice::<half::f16>()?,
                _ => candle::bail!("ys storage must be cuda"),
            };
            let tw_s = match &*tw_s {
                candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
                _ => candle::bail!("topk_weights storage must be cuda"),
            };
            let out_buf = unsafe { dev.alloc::<half::f16>(num_tokens * hidden_dim) }?;
            unsafe {
                crate::cuda::ffi::moe_weighted_sum_f16(
                    ys_s.slice(ys_offset..).device_ptr(ys_s.stream()).0 as *const c_void,
                    tw_s.slice(tw_offset..).device_ptr(tw_s.stream()).0 as *const c_void,
                    out_buf.device_ptr(out_buf.stream()).0 as *mut c_void,
                    num_tokens as i32,
                    topk as i32,
                    hidden_dim as i32,
                    stream_id,
                );
            }
            let storage = candle::CudaStorage::wrap_cuda_slice(out_buf, dev);
            Ok(Tensor::from((
                candle::Storage::Cuda(storage),
                (num_tokens, hidden_dim),
            )))
        }
        other => candle::bail!(
            "moe_weighted_sum_cuda: unsupported dtype {other:?} (want BF16 or F16)"
        ),
    }
}

#[cfg(not(feature = "cuda"))]
#[allow(unused)]
pub fn moe_weighted_sum_cuda(_ys: &Tensor, _topk_weights: &Tensor) -> Result<Tensor> {
    candle_core::bail!("moe_weighted_sum_cuda requires the cuda feature")
}
