//! End-to-end driver for the Qwen3.5-0.8B megakernel drafter.
//!
//! Owns a [`MegakernelBuffers`] + [`MegakernelWeights`] pair and
//! exposes two hot-path methods:
//!   * [`MegakernelDrafter::prefill`] — feed a full prompt, get the
//!     first generated token back. Must be called before any step().
//!   * [`MegakernelDrafter::step`] — one fused-kernel decode for the
//!     next token. Reads state from the shared buffers, mutates in
//!     place, returns the sampled argmax.
//!
//! Thread-safety: single-sequence only. The kernel uses persistent
//! per-device state (barrier counters, KV cache, DN state) that is
//! specific to this drafter instance; callers must not share it
//! across concurrent requests. For multiple inflight sequences,
//! allocate multiple drafters.

#![cfg(feature = "cuda")]

use candle_core::{Device, Result, Tensor};
use std::ffi::{c_int, c_uint, c_void};

use super::buffers::MegakernelBuffers;
use super::constants::*;
use super::weights::MegakernelWeights;
use crate::cuda::dflash_megakernel::{launch_decode, launch_prefill_bf16, CudaStreamPtr};

pub struct MegakernelDrafter {
    pub weights: MegakernelWeights,
    pub buffers: MegakernelBuffers,
    /// Monotonic decode position — advanced by 1 per step() after
    /// prefill has set it to `prompt_len`. Matches the `position`
    /// arg the kernel uses to index RoPE + the KV ring.
    position: i32,
}

impl MegakernelDrafter {
    /// Assemble a new drafter from loaded weights. Allocates
    /// on-device scratch + state buffers sized for up to
    /// `max_prefill_seq` prompt tokens. Call [`Self::reset`] before
    /// each new request to wipe KV / DN state.
    pub fn new(weights: MegakernelWeights, max_prefill_seq: usize) -> Result<Self> {
        let device = weights.device.clone();
        let buffers = MegakernelBuffers::new(&device, max_prefill_seq)?;
        let mut drafter = Self {
            weights,
            buffers,
            position: 0,
        };
        // Pack the weight pointer table once up-front.
        drafter.weights.pack()?;
        Ok(drafter)
    }

    pub fn device(&self) -> &Device {
        &self.weights.device
    }

    /// Current decode position. After prefill with N prompt tokens
    /// this is N; after each step() it increments by 1.
    pub fn position(&self) -> i32 {
        self.position
    }

    /// Wipe all stateful buffers (KV cache, DN states, conv windows,
    /// barrier counters, LM sync counter) and reset the position to
    /// 0. Must be called between independent requests.
    pub fn reset(&mut self) -> Result<()> {
        self.buffers.reset()?;
        self.position = 0;
        Ok(())
    }

    /// Run prefill over a prompt of `token_ids.len()` tokens.
    /// Populates the KV cache + DN state, writes the first generated
    /// token's argmax into `buffers.out_token`, and advances the
    /// position counter by `prompt_len`.
    ///
    /// Returns the first generated token id (`out_token[0]`).
    pub fn prefill(&mut self, token_ids: &[i32]) -> Result<i32> {
        let seq_len = token_ids.len();
        if seq_len == 0 {
            candle_core::bail!("MegakernelDrafter::prefill: empty token_ids");
        }
        if seq_len > self.buffers.max_prefill_seq {
            candle_core::bail!(
                "MegakernelDrafter::prefill: seq_len {seq_len} exceeds \
                 buffers.max_prefill_seq={}",
                self.buffers.max_prefill_seq
            );
        }
        if seq_len > MAX_SEQ_LEN {
            candle_core::bail!(
                "MegakernelDrafter::prefill: seq_len {seq_len} exceeds kernel cap {MAX_SEQ_LEN}"
            );
        }

        // Upload the prompt tokens to device as i32.
        let device = self.device().clone();
        let token_tensor = Tensor::from_vec(token_ids.to_vec(), (seq_len,), &device)?;

        // Pull raw device pointers.
        let stream = cuda_stream(&device)?;
        let packed = self
            .weights
            .packed_ptr()
            .ok_or_else(|| candle_core::Error::msg("MegakernelDrafter: weights not packed"))?;

        unsafe {
            launch_prefill_bf16(
                device_ptr_i32(&token_tensor)? as *const c_int,
                seq_len as c_int,
                device_ptr_i32(&self.buffers.out_token)? as *mut c_int,
                self.weights.embed_ptr(),
                packed as *const _,
                self.weights.final_norm_ptr(),
                self.weights.lm_head_ptr(),
                device_ptr_bf16(&self.buffers.fa_k_cache)? as *mut c_void,
                device_ptr_bf16(&self.buffers.fa_v_cache)? as *mut c_void,
                device_ptr_f32(&self.buffers.dn_states)? as *mut f32,
                device_ptr_f32(&self.buffers.conv_bufs)? as *mut f32,
                device_ptr_bf16(&self.buffers.pf_hidden)? as *mut c_void,
                device_ptr_bf16(&self.buffers.pf_residual)? as *mut c_void,
                device_ptr_bf16(&self.buffers.pf_normalized)? as *mut c_void,
                device_ptr_bf16(&self.buffers.pf_proj_buf)? as *mut c_void,
                device_ptr_bf16(&self.buffers.pf_proj_buf2)? as *mut c_void,
                device_ptr_bf16(&self.buffers.pf_attn_buf)? as *mut c_void,
                device_ptr_bf16(&self.buffers.pf_mlp_buf)? as *mut c_void,
                device_ptr_bf16(&self.buffers.pf_dn_out_buf)? as *mut c_void,
                device_ptr_f32(&self.buffers.pf_beta_buf)? as *mut f32,
                device_ptr_f32(&self.buffers.pf_alpha_buf)? as *mut f32,
                device_ptr_bf16(&self.buffers.pf_final_normed)? as *mut c_void,
                device_ptr_bf16(&self.buffers.pf_hidden_bf16_out)? as *mut c_void,
                device_ptr_f32(&self.buffers.pf_lm_bmv)? as *mut f32,
                device_ptr_i32(&self.buffers.pf_lm_bmi)? as *mut c_int,
                stream,
            );
        }

        self.position = seq_len as i32;
        read_out_token(&self.buffers.out_token)
    }

    /// Run one decode step. Reads `input_token` + current state,
    /// writes the next token's argmax into `buffers.out_token`,
    /// advances position by 1, returns the new token id.
    pub fn step(&mut self, input_token: i32) -> Result<i32> {
        if self.position as usize >= MAX_SEQ_LEN {
            candle_core::bail!(
                "MegakernelDrafter::step: position {} exceeds kernel cap {MAX_SEQ_LEN}",
                self.position
            );
        }
        let device = self.device().clone();
        let stream = cuda_stream(&device)?;
        let packed = self
            .weights
            .packed_ptr()
            .ok_or_else(|| candle_core::Error::msg("MegakernelDrafter: weights not packed"))?;

        unsafe {
            launch_decode(
                input_token as c_int,
                device_ptr_i32(&self.buffers.out_token)? as *mut c_int,
                self.weights.embed_ptr(),
                packed as *const _,
                self.weights.final_norm_ptr(),
                self.weights.lm_head_ptr(),
                device_ptr_bf16(&self.buffers.fa_k_cache)? as *mut c_void,
                device_ptr_bf16(&self.buffers.fa_v_cache)? as *mut c_void,
                device_ptr_f32(&self.buffers.dn_states)? as *mut c_void,
                device_ptr_f32(&self.buffers.conv_bufs)? as *mut c_void,
                device_ptr_bf16(&self.buffers.hidden_buffer)? as *mut c_void,
                device_ptr_f32(&self.buffers.g_activations)? as *mut c_void,
                device_ptr_bf16(&self.buffers.g_residual)? as *mut c_void,
                device_ptr_f32(&self.buffers.g_qkv_scratch)? as *mut c_void,
                device_ptr_f32(&self.buffers.g_kv_scratch)? as *mut c_void,
                device_ptr_f32(&self.buffers.g_attn_out)? as *mut c_void,
                device_ptr_f32(&self.buffers.g_mlp_inter)? as *mut c_void,
                device_ptr_f32(&self.buffers.g_z_scratch)? as *mut c_void,
                device_ptr_f32(&self.buffers.g_beta_scratch)? as *mut c_void,
                device_ptr_f32(&self.buffers.g_alpha_scratch)? as *mut c_void,
                device_ptr_f32(&self.buffers.g_normalized)? as *mut c_void,
                device_ptr_u32(&self.buffers.barrier_counter)? as *mut c_uint,
                device_ptr_u32(&self.buffers.barrier_generation)? as *mut c_uint,
                device_ptr_f32(&self.buffers.block_max_vals)? as *mut f32,
                device_ptr_i32(&self.buffers.block_max_idxs)? as *mut c_int,
                device_ptr_u32(&self.buffers.lm_sync_counter)? as *mut c_uint,
                self.position,
                MAX_SEQ_LEN as c_int,
                stream,
            );
        }

        self.position += 1;
        read_out_token(&self.buffers.out_token)
    }
}

// ── Raw device-pointer helpers ──
//
// These use candle's cudarc bindings to extract the base pointer of a
// tensor. They require the tensor to be contiguous (no view strides)
// and of the expected dtype. All returned pointers share the tensor's
// lifetime; the caller must keep the tensor alive for at least as
// long as the kernel launch consuming the pointer.

fn cuda_stream(device: &Device) -> Result<CudaStreamPtr> {
    let dev = device.as_cuda_device()?;
    let stream = dev.cuda_stream();
    // `cu_stream()` returns a `cudaStream_t`-compatible handle.
    Ok(stream.cu_stream() as CudaStreamPtr)
}

fn device_ptr_bf16(t: &Tensor) -> Result<*mut half::bf16> {
    use candle_core::cuda_backend::cudarc::driver::DevicePtr;
    if t.dtype() != candle_core::DType::BF16 {
        candle_core::bail!("expected BF16 tensor, got {:?}", t.dtype());
    }
    let (storage, layout) = t.storage_and_layout();
    let s = match &*storage {
        candle_core::Storage::Cuda(c) => c.as_cuda_slice::<half::bf16>()?,
        _ => candle_core::bail!("non-CUDA tensor"),
    };
    Ok(s.slice(layout.start_offset()..).device_ptr(s.stream()).0 as *mut half::bf16)
}

fn device_ptr_f32(t: &Tensor) -> Result<*mut f32> {
    use candle_core::cuda_backend::cudarc::driver::DevicePtr;
    if t.dtype() != candle_core::DType::F32 {
        candle_core::bail!("expected F32 tensor, got {:?}", t.dtype());
    }
    let (storage, layout) = t.storage_and_layout();
    let s = match &*storage {
        candle_core::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
        _ => candle_core::bail!("non-CUDA tensor"),
    };
    Ok(s.slice(layout.start_offset()..).device_ptr(s.stream()).0 as *mut f32)
}

fn device_ptr_i32(t: &Tensor) -> Result<*mut i32> {
    // Candle's scheduler maps "i32" via the closest canonical; most
    // model tensors use I64 for indices. For our megakernel inputs we
    // allocate with DType::I64 and reinterpret the low 32 bits. The
    // kernel expects packed i32, so the caller must have sized the
    // tensor correctly (NOT using I64 host-side — that would double
    // the stride). Currently all `*_i32` buffers are allocated via
    // `Tensor::zeros(.., DType::I64, ..)` above; we accept both i32
    // and i64 dtypes here — when i64 we assume the consumer only
    // reads the low bytes, which matches the kernel's `int*` cast
    // on nvcc's little-endian linux/windows platforms. A dedicated
    // I32 dtype is tracked upstream in candle but not yet landed.
    use candle_core::cuda_backend::cudarc::driver::DevicePtr;
    let (storage, layout) = t.storage_and_layout();
    match &*storage {
        candle_core::Storage::Cuda(c) => match t.dtype() {
            candle_core::DType::I64 => {
                let s = c.as_cuda_slice::<i64>()?;
                Ok(
                    s.slice(layout.start_offset()..).device_ptr(s.stream()).0
                        as *mut i32,
                )
            }
            candle_core::DType::U32 => {
                let s = c.as_cuda_slice::<u32>()?;
                Ok(
                    s.slice(layout.start_offset()..).device_ptr(s.stream()).0
                        as *mut i32,
                )
            }
            other => {
                candle_core::bail!(
                    "expected I64 or U32 tensor for i32 pointer, got {:?}",
                    other
                )
            }
        },
        _ => candle_core::bail!("non-CUDA tensor"),
    }
}

fn device_ptr_u32(t: &Tensor) -> Result<*mut u32> {
    use candle_core::cuda_backend::cudarc::driver::DevicePtr;
    if t.dtype() != candle_core::DType::U32 {
        candle_core::bail!("expected U32 tensor, got {:?}", t.dtype());
    }
    let (storage, layout) = t.storage_and_layout();
    let s = match &*storage {
        candle_core::Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
        _ => candle_core::bail!("non-CUDA tensor"),
    };
    Ok(s.slice(layout.start_offset()..).device_ptr(s.stream()).0 as *mut u32)
}

fn read_out_token(out: &Tensor) -> Result<i32> {
    // `out_token` was allocated as I64 (single element). Copy to host
    // and take the low 32 bits — consistent with the kernel writing a
    // 32-bit `int` into the first element.
    let v: Vec<i64> = out.to_dtype(candle_core::DType::I64)?.to_vec1()?;
    Ok(v.first().copied().unwrap_or(0) as i32)
}
