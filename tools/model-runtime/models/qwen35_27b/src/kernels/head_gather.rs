//! Strided head gather / scatter for the full-attention per-head loop.
//!
//! The full-attention forward needs to pull out one Q/K/V head at a
//! time from a packed `[n_tokens, n_heads, head_dim]` (or slab)
//! tensor. The original port did this with `memcpy_dtod` per token —
//! ~100k launches per FA layer just to stage head stripes, on top of
//! the actual matmul work. These kernels replace that loop with one
//! launch per stage: a single-thread-per-element strided copy that
//! reads from the source row `(t, head, :)` and writes to the dense
//! destination row `(t, :)` (or vice-versa for scatter).
//!
//! Three entry points:
//!
//!   * [`launch_head_gather_bf16`] — gather head `h` from a packed
//!     `[n_tokens, n_heads, head_dim]` activation (Q stream from the
//!     projection output).
//!   * [`launch_head_scatter_bf16`] — mirror of the above; writes a
//!     contiguous `[n_tokens, head_dim]` head result back into the
//!     packed `[n_tokens, n_heads, head_dim]` destination.
//!   * [`launch_head_gather_slab_bf16`] — same shape pattern as
//!     [`launch_head_gather_bf16`] but the outer dimension is
//!     `kv_len` (only the first `kv_len` rows of a KV slab are
//!     touched; the slab's max-context padding is simply skipped).
//!
//! All three keep the `OnceLock`-cached `CudaFunction` convention
//! used by the other kernel wrappers — see `rmsnorm.rs` for the
//! template.

use std::sync::{Arc, OnceLock};

use anyhow::{anyhow, Result};
use cudarc::driver::{CudaFunction, CudaSlice, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;
use half::bf16;

use ctox_cuda_primitives::device::DeviceContext;
use ctox_cuda_primitives::tensor::CudaTensor;

use super::HEAD_GATHER_PTX;

const BLOCK_DIM: u32 = 256;

static HEAD_GATHER_BF16_FN: OnceLock<CudaFunction> = OnceLock::new();
static HEAD_SCATTER_BF16_FN: OnceLock<CudaFunction> = OnceLock::new();
static HEAD_GATHER_SLAB_BF16_FN: OnceLock<CudaFunction> = OnceLock::new();

fn load_fn(
    device: &Arc<DeviceContext>,
    cache: &OnceLock<CudaFunction>,
    entry: &str,
) -> Result<CudaFunction> {
    if let Some(f) = cache.get() {
        return Ok(f.clone());
    }
    let ptx_src = std::str::from_utf8(HEAD_GATHER_PTX)
        .map_err(|e| anyhow!("head_gather.ptx not UTF-8: {}", e))?
        .to_string();
    let module = device
        .raw()
        .load_module(Ptx::from_src(ptx_src))
        .map_err(|e| anyhow!("load_module head_gather.ptx: {:?}", e))?;
    let f = module
        .load_function(entry)
        .map_err(|e| anyhow!("load_function {}: {:?}", entry, e))?;
    let _ = cache.set(f.clone());
    Ok(f)
}

fn launch_cfg(total: usize) -> LaunchConfig {
    let grid = ((total as u32) + BLOCK_DIM - 1) / BLOCK_DIM;
    LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (BLOCK_DIM, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// Gather head `head` from `src` `[n_tokens, n_heads, head_dim]`
/// into `dst` `[n_tokens, head_dim]`.
///
/// Replaces the previous per-token `memcpy_dtod` loop — one kernel
/// launch per call, regardless of `n_tokens`.
pub fn launch_head_gather_bf16(
    device: &Arc<DeviceContext>,
    src: &CudaTensor<bf16>,
    dst: &mut CudaTensor<bf16>,
    n_tokens: usize,
    n_heads: usize,
    head_dim: usize,
    head: usize,
) -> Result<()> {
    validate_packed_shapes(src, dst, n_tokens, n_heads, head_dim, head)?;
    if n_tokens == 0 || head_dim == 0 {
        return Ok(());
    }

    let f = load_fn(device, &HEAD_GATHER_BF16_FN, "head_gather_bf16")?;
    let stream = device.raw().default_stream();
    let n_tokens_i32 = n_tokens as i32;
    let n_heads_i32 = n_heads as i32;
    let head_dim_i32 = head_dim as i32;
    let head_i32 = head as i32;

    let mut launcher = stream.launch_builder(&f);
    launcher
        .arg(src.buf())
        .arg(dst.buf_mut())
        .arg(&n_tokens_i32)
        .arg(&n_heads_i32)
        .arg(&head_dim_i32)
        .arg(&head_i32);

    unsafe { launcher.launch(launch_cfg(n_tokens * head_dim)) }.map_err(|e| {
        anyhow!(
            "head_gather_bf16 launch (n_tokens={} n_heads={} head_dim={} head={}): {:?}",
            n_tokens,
            n_heads,
            head_dim,
            head,
            e
        )
    })?;
    Ok(())
}

/// Scatter `src` `[n_tokens, head_dim]` into slot `head` of
/// `dst` `[n_tokens, n_heads, head_dim]`.
///
/// Mirror of [`launch_head_gather_bf16`]; same launch cost.
pub fn launch_head_scatter_bf16(
    device: &Arc<DeviceContext>,
    src: &CudaTensor<bf16>,
    dst: &mut CudaTensor<bf16>,
    n_tokens: usize,
    n_heads: usize,
    head_dim: usize,
    head: usize,
) -> Result<()> {
    // Dst is stored flat as `[n_tokens, n_heads * head_dim]` — accept
    // either that 2D shape or the explicit 3D layout.
    let expected_dst_numel = n_tokens * n_heads * head_dim;
    if dst.numel() != expected_dst_numel {
        return Err(anyhow!(
            "head_scatter_bf16: dst numel {} != n_tokens*n_heads*head_dim={}",
            dst.numel(),
            expected_dst_numel
        ));
    }
    if src.shape() != [n_tokens, head_dim] {
        return Err(anyhow!(
            "head_scatter_bf16: src.shape {:?} != [n_tokens={}, head_dim={}]",
            src.shape(),
            n_tokens,
            head_dim
        ));
    }
    if head >= n_heads {
        return Err(anyhow!(
            "head_scatter_bf16: head {} >= n_heads {}",
            head,
            n_heads
        ));
    }
    if n_tokens == 0 || head_dim == 0 {
        return Ok(());
    }

    let f = load_fn(device, &HEAD_SCATTER_BF16_FN, "head_scatter_bf16")?;
    let stream = device.raw().default_stream();
    let n_tokens_i32 = n_tokens as i32;
    let n_heads_i32 = n_heads as i32;
    let head_dim_i32 = head_dim as i32;
    let head_i32 = head as i32;

    let mut launcher = stream.launch_builder(&f);
    launcher
        .arg(src.buf())
        .arg(dst.buf_mut())
        .arg(&n_tokens_i32)
        .arg(&n_heads_i32)
        .arg(&head_dim_i32)
        .arg(&head_i32);

    unsafe { launcher.launch(launch_cfg(n_tokens * head_dim)) }.map_err(|e| {
        anyhow!(
            "head_scatter_bf16 launch (n_tokens={} n_heads={} head_dim={} head={}): {:?}",
            n_tokens,
            n_heads,
            head_dim,
            head,
            e
        )
    })?;
    Ok(())
}

/// Gather head `head` from a KV slab.
///
/// Slab is stored as `[max_ctx, n_kv_heads, head_dim]` on device;
/// only the first `kv_len` rows hold valid data. The kernel reads
/// exactly those rows — the rest of the slab is untouched. Passing
/// a `&CudaSlice<bf16>` lets the caller hand us the slab directly
/// from `KvCache::k_slab()` without synthesizing a `CudaTensor`
/// wrapper.
pub fn launch_head_gather_slab_bf16(
    device: &Arc<DeviceContext>,
    slab: &CudaSlice<bf16>,
    dst: &mut CudaTensor<bf16>,
    kv_len: usize,
    n_kv_heads: usize,
    head_dim: usize,
    head: usize,
) -> Result<()> {
    if head >= n_kv_heads {
        return Err(anyhow!(
            "head_gather_slab_bf16: head {} >= n_kv_heads {}",
            head,
            n_kv_heads
        ));
    }
    if dst.shape() != [kv_len, head_dim] {
        return Err(anyhow!(
            "head_gather_slab_bf16: dst.shape {:?} != [kv_len={}, head_dim={}]",
            dst.shape(),
            kv_len,
            head_dim
        ));
    }
    // Slab must contain at least enough rows to reach the requested
    // `head` stripe at row `kv_len - 1`.
    let min_elems = (kv_len.saturating_sub(1)) * n_kv_heads * head_dim
        + (head + 1) * head_dim;
    if slab.len() < min_elems {
        return Err(anyhow!(
            "head_gather_slab_bf16: slab has {} elems; need at least {} for kv_len={} n_kv_heads={} head_dim={} head={}",
            slab.len(),
            min_elems,
            kv_len,
            n_kv_heads,
            head_dim,
            head
        ));
    }
    if kv_len == 0 || head_dim == 0 {
        return Ok(());
    }

    let f = load_fn(
        device,
        &HEAD_GATHER_SLAB_BF16_FN,
        "head_gather_slab_bf16",
    )?;
    let stream = device.raw().default_stream();
    let kv_len_i32 = kv_len as i32;
    let n_kv_heads_i32 = n_kv_heads as i32;
    let head_dim_i32 = head_dim as i32;
    let head_i32 = head as i32;

    let mut launcher = stream.launch_builder(&f);
    launcher
        .arg(slab)
        .arg(dst.buf_mut())
        .arg(&kv_len_i32)
        .arg(&n_kv_heads_i32)
        .arg(&head_dim_i32)
        .arg(&head_i32);

    unsafe { launcher.launch(launch_cfg(kv_len * head_dim)) }.map_err(|e| {
        anyhow!(
            "head_gather_slab_bf16 launch (kv_len={} n_kv_heads={} head_dim={} head={}): {:?}",
            kv_len,
            n_kv_heads,
            head_dim,
            head,
            e
        )
    })?;
    Ok(())
}

fn validate_packed_shapes(
    src: &CudaTensor<bf16>,
    dst: &CudaTensor<bf16>,
    n_tokens: usize,
    n_heads: usize,
    head_dim: usize,
    head: usize,
) -> Result<()> {
    // Packed-source shapes appear both as the explicit 3D
    // `[n_tokens, n_heads, head_dim]` and as a flattened
    // `[n_tokens, n_heads * head_dim]` — accept both. The kernel
    // only cares about numel and the supplied n_heads / head_dim.
    let expected_numel = n_tokens * n_heads * head_dim;
    if src.numel() != expected_numel {
        return Err(anyhow!(
            "head_gather_bf16: src numel {} != n_tokens*n_heads*head_dim={}",
            src.numel(),
            expected_numel
        ));
    }
    if dst.shape() != [n_tokens, head_dim] {
        return Err(anyhow!(
            "head_gather_bf16: dst.shape {:?} != [n_tokens={}, head_dim={}]",
            dst.shape(),
            n_tokens,
            head_dim
        ));
    }
    if head >= n_heads {
        return Err(anyhow!(
            "head_gather_bf16: head {} >= n_heads {}",
            head,
            n_heads
        ));
    }
    Ok(())
}
