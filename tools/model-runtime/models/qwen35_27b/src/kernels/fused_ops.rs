//! Device-side replacements for the three host round-trips that used
//! to sit inside the full-attention forward: `host_scale_add`,
//! `sigmoid_host_bf16`, and `elementwise_mul_host_bf16`.
//!
//! Those host round-trips were the last remaining per-layer CPU
//! syncs, and each one blocks CUDA graph capture outright (a graph
//! can only record work that stays on the device). Replacing them
//! with small element-wise kernels removes that barrier so A3
//! (decode-step graph capture) becomes implementable.
//!
//! ## Fusion choices
//!
//! `host_scale_add` was already one FMA per element, so it maps
//! cleanly to a single `scale_add_f32` kernel.
//!
//! The attention gate was two passes on the host — compute
//! `sigmoid(gate)` once, then multiply it into the attention output.
//! Those are fused into a single `sigmoid_mul_bf16` kernel here:
//!
//!   `attn_out[i] ← attn_out[i] * sigmoid(gate[i])`
//!
//! No intermediate gate-sigmoid buffer is materialized. The
//! standalone `sigmoid_bf16` entry is kept for callers that need
//! the sigmoid without the mul (GDN in particular will want it
//! once its hardcoded gate is replaced).

use std::sync::{Arc, OnceLock};

use anyhow::{anyhow, Result};
use cudarc::driver::{CudaFunction, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;
use half::bf16;

use ctox_cuda_primitives::device::DeviceContext;
use ctox_cuda_primitives::tensor::CudaTensor;

use super::FUSED_OPS_PTX;

const BLOCK_DIM: u32 = 256;

static SCALE_ADD_F32_FN: OnceLock<CudaFunction> = OnceLock::new();
static SCALE_ADD_WITH_BIAS_F32_FN: OnceLock<CudaFunction> = OnceLock::new();
static SIGMOID_MUL_BF16_FN: OnceLock<CudaFunction> = OnceLock::new();
static SIGMOID_BF16_FN: OnceLock<CudaFunction> = OnceLock::new();
static TRANSPOSE_2D_BF16_FN: OnceLock<CudaFunction> = OnceLock::new();
static ROW_SLICE_F32_FN: OnceLock<CudaFunction> = OnceLock::new();
static FILL_CONST_F32_FN: OnceLock<CudaFunction> = OnceLock::new();
static GDN_GATE_V_MEAN_STANDIN_F32_FN: OnceLock<CudaFunction> = OnceLock::new();

fn load_fn(
    device: &Arc<DeviceContext>,
    cache: &OnceLock<CudaFunction>,
    entry: &str,
) -> Result<CudaFunction> {
    if let Some(f) = cache.get() {
        return Ok(f.clone());
    }
    let ptx_src = std::str::from_utf8(FUSED_OPS_PTX)
        .map_err(|e| anyhow!("fused_ops.ptx not UTF-8: {}", e))?
        .to_string();
    let module = device
        .raw()
        .load_module(Ptx::from_src(ptx_src))
        .map_err(|e| anyhow!("load_module fused_ops.ptx: {:?}", e))?;
    let f = module
        .load_function(entry)
        .map_err(|e| anyhow!("load_function {}: {:?}", entry, e))?;
    let _ = cache.set(f.clone());
    Ok(f)
}

fn launch_cfg(numel: usize) -> LaunchConfig {
    let grid = ((numel as u32) + BLOCK_DIM - 1) / BLOCK_DIM;
    LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (BLOCK_DIM, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// `y[i] = x[i] * scale + y[i]` element-wise, f32.
///
/// Direct replacement for the old `host_scale_add`: caller passed
/// scaled attention scores + pre-built mask; this kernel folds the
/// scale multiplier into the mask-add without going through host
/// memory. Shapes must match, numel taken from `y.shape()`.
pub fn launch_scale_add_f32(
    device: &Arc<DeviceContext>,
    x: &CudaTensor<f32>,
    y: &mut CudaTensor<f32>,
    scale: f32,
) -> Result<()> {
    if x.shape() != y.shape() {
        return Err(anyhow!(
            "scale_add_f32: x.shape {:?} != y.shape {:?}",
            x.shape(),
            y.shape()
        ));
    }
    let numel = x.numel();
    if numel == 0 {
        return Ok(());
    }

    let f = load_fn(device, &SCALE_ADD_F32_FN, "scale_add_f32")?;
    let stream = device.raw().default_stream();
    let numel_i32 = numel as i32;
    let mut launcher = stream.launch_builder(&f);
    launcher
        .arg(x.buf())
        .arg(y.buf_mut())
        .arg(&scale)
        .arg(&numel_i32);

    unsafe { launcher.launch(launch_cfg(numel)) }
        .map_err(|e| anyhow!("scale_add_f32 launch (numel={}): {:?}", numel, e))?;
    Ok(())
}

/// `y[i] = x[i] * scale + bias[i]` element-wise, f32.
///
/// Three-buffer variant of [`launch_scale_add_f32`] — reads the
/// bias from a separate shared buffer instead of overwriting the
/// destination's previous contents. Lets the caller pre-upload the
/// causal mask once and reuse it across all `n_q_heads` attention
/// heads without an extra D2D memcpy per head.
pub fn launch_scale_add_with_bias_f32(
    device: &Arc<DeviceContext>,
    x: &CudaTensor<f32>,
    bias: &CudaTensor<f32>,
    y: &mut CudaTensor<f32>,
    scale: f32,
) -> Result<()> {
    if x.shape() != y.shape() || bias.shape() != y.shape() {
        return Err(anyhow!(
            "scale_add_with_bias_f32: shape mismatch x={:?} bias={:?} y={:?}",
            x.shape(),
            bias.shape(),
            y.shape()
        ));
    }
    let numel = y.numel();
    if numel == 0 {
        return Ok(());
    }

    let f = load_fn(
        device,
        &SCALE_ADD_WITH_BIAS_F32_FN,
        "scale_add_with_bias_f32",
    )?;
    let stream = device.raw().default_stream();
    let numel_i32 = numel as i32;
    let mut launcher = stream.launch_builder(&f);
    launcher
        .arg(x.buf())
        .arg(bias.buf())
        .arg(y.buf_mut())
        .arg(&scale)
        .arg(&numel_i32);

    unsafe { launcher.launch(launch_cfg(numel)) }.map_err(|e| {
        anyhow!(
            "scale_add_with_bias_f32 launch (numel={}): {:?}",
            numel,
            e
        )
    })?;
    Ok(())
}

/// `y[i] = y[i] * sigmoid(x[i])` element-wise, bf16.
///
/// Fuses the attention-gate sigmoid and the subsequent elementwise
/// multiply that used to sit on the host. `x` is the raw gate
/// projection; `y` holds the attention output before gating and is
/// overwritten in place with the gated result.
pub fn launch_sigmoid_mul_bf16(
    device: &Arc<DeviceContext>,
    x: &CudaTensor<bf16>,
    y: &mut CudaTensor<bf16>,
) -> Result<()> {
    if x.shape() != y.shape() {
        return Err(anyhow!(
            "sigmoid_mul_bf16: x.shape {:?} != y.shape {:?}",
            x.shape(),
            y.shape()
        ));
    }
    let numel = x.numel();
    if numel == 0 {
        return Ok(());
    }

    let f = load_fn(device, &SIGMOID_MUL_BF16_FN, "sigmoid_mul_bf16")?;
    let stream = device.raw().default_stream();
    let numel_i32 = numel as i32;
    let mut launcher = stream.launch_builder(&f);
    launcher.arg(x.buf()).arg(y.buf_mut()).arg(&numel_i32);

    unsafe { launcher.launch(launch_cfg(numel)) }
        .map_err(|e| anyhow!("sigmoid_mul_bf16 launch (numel={}): {:?}", numel, e))?;
    Ok(())
}

/// `y[i] = sigmoid(x[i])` element-wise, bf16.
///
/// Standalone sigmoid without the subsequent mul. Not currently used
/// by full-attention (that path is fused via
/// [`launch_sigmoid_mul_bf16`]); wired up for the GDN path which
/// still hardcodes its gate and will want a real sigmoid call once
/// that's replaced.
pub fn launch_sigmoid_bf16(
    device: &Arc<DeviceContext>,
    x: &CudaTensor<bf16>,
    y: &mut CudaTensor<bf16>,
) -> Result<()> {
    if x.shape() != y.shape() {
        return Err(anyhow!(
            "sigmoid_bf16: x.shape {:?} != y.shape {:?}",
            x.shape(),
            y.shape()
        ));
    }
    let numel = x.numel();
    if numel == 0 {
        return Ok(());
    }

    let f = load_fn(device, &SIGMOID_BF16_FN, "sigmoid_bf16")?;
    let stream = device.raw().default_stream();
    let numel_i32 = numel as i32;
    let mut launcher = stream.launch_builder(&f);
    launcher.arg(x.buf()).arg(y.buf_mut()).arg(&numel_i32);

    unsafe { launcher.launch(launch_cfg(numel)) }
        .map_err(|e| anyhow!("sigmoid_bf16 launch (numel={}): {:?}", numel, e))?;
    Ok(())
}

/// 2-D transpose: `src[rows, cols]` → `dst[cols, rows]`, bf16.
///
/// Called per attention head to build `K^T` for the score matmul.
/// Replaces the previous host-side transpose (download → transpose
/// on CPU → re-upload), which contributed 384 host round-trips per
/// forward on the 27B (16 FA layers × 24 Q-heads).
///
/// `dst` must already be allocated with shape `[cols, rows]`. The
/// kernel is a one-thread-per-element naive transpose — no
/// shared-memory staging because the call-site shapes (head
/// dim × kv_len) are small; a block-swap kernel would win for
/// larger transposes but costs more than it saves here.
pub fn launch_transpose_2d_bf16(
    device: &Arc<DeviceContext>,
    src: &CudaTensor<bf16>,
    dst: &mut CudaTensor<bf16>,
    rows: usize,
    cols: usize,
) -> Result<()> {
    if src.shape() != [rows, cols] {
        return Err(anyhow!(
            "transpose_2d_bf16: src.shape {:?} != [rows={}, cols={}]",
            src.shape(),
            rows,
            cols
        ));
    }
    if dst.shape() != [cols, rows] {
        return Err(anyhow!(
            "transpose_2d_bf16: dst.shape {:?} != [cols={}, rows={}]",
            dst.shape(),
            cols,
            rows
        ));
    }
    let numel = rows * cols;
    if numel == 0 {
        return Ok(());
    }

    let f = load_fn(device, &TRANSPOSE_2D_BF16_FN, "transpose_2d_bf16")?;
    let stream = device.raw().default_stream();
    let rows_i32 = rows as i32;
    let cols_i32 = cols as i32;
    let mut launcher = stream.launch_builder(&f);
    launcher
        .arg(src.buf())
        .arg(dst.buf_mut())
        .arg(&rows_i32)
        .arg(&cols_i32);

    unsafe { launcher.launch(launch_cfg(numel)) }.map_err(|e| {
        anyhow!(
            "transpose_2d_bf16 launch (rows={} cols={}): {:?}",
            rows,
            cols,
            e
        )
    })?;
    Ok(())
}

/// Strided row slice: `dst[t, :] = src[t, src_offset .. src_offset + slice_width]`
/// for each row `t`, f32.
///
/// Replaces a family of host round-trips inside GDN's forward:
/// `qkv_f32.to_host()` + three per-stream `Vec::copy_from_slice` +
/// three `from_host` uploads used to split the fused QKV projection
/// into its q / k / v streams.
///
/// Caller passes the source matrix's logical 2D shape
/// `[n_rows, src_cols]`, the column offset into each row, and the
/// slice width. `dst` must already be allocated as
/// `[n_rows, slice_width]`.
pub fn launch_row_slice_f32(
    device: &Arc<DeviceContext>,
    src: &CudaTensor<f32>,
    dst: &mut CudaTensor<f32>,
    n_rows: usize,
    src_cols: usize,
    src_offset: usize,
    slice_width: usize,
) -> Result<()> {
    if src.numel() < n_rows * src_cols {
        return Err(anyhow!(
            "row_slice_f32: src numel {} < n_rows*src_cols = {}*{} = {}",
            src.numel(),
            n_rows,
            src_cols,
            n_rows * src_cols
        ));
    }
    if dst.numel() < n_rows * slice_width {
        return Err(anyhow!(
            "row_slice_f32: dst numel {} < n_rows*slice_width = {}*{} = {}",
            dst.numel(),
            n_rows,
            slice_width,
            n_rows * slice_width
        ));
    }
    if src_offset + slice_width > src_cols {
        return Err(anyhow!(
            "row_slice_f32: src_offset {} + slice_width {} > src_cols {}",
            src_offset,
            slice_width,
            src_cols
        ));
    }
    let total = n_rows * slice_width;
    if total == 0 {
        return Ok(());
    }

    let f = load_fn(device, &ROW_SLICE_F32_FN, "row_slice_f32")?;
    let stream = device.raw().default_stream();
    let n_rows_i32 = n_rows as i32;
    let src_cols_i32 = src_cols as i32;
    let src_offset_i32 = src_offset as i32;
    let slice_width_i32 = slice_width as i32;
    let mut launcher = stream.launch_builder(&f);
    launcher
        .arg(src.buf())
        .arg(dst.buf_mut())
        .arg(&n_rows_i32)
        .arg(&src_cols_i32)
        .arg(&src_offset_i32)
        .arg(&slice_width_i32);

    unsafe { launcher.launch(launch_cfg(total)) }.map_err(|e| {
        anyhow!(
            "row_slice_f32 launch (rows={} src_cols={} off={} w={}): {:?}",
            n_rows,
            src_cols,
            src_offset,
            slice_width,
            e
        )
    })?;
    Ok(())
}

/// Fill every element of `y` with a constant scalar, f32.
///
/// Replaces the `from_host(vec![value; numel])` pattern used
/// throughout GDN (beta stand-in, zero-init buffers). No host
/// allocation, no upload, one kernel launch.
pub fn launch_fill_const_f32(
    device: &Arc<DeviceContext>,
    y: &mut CudaTensor<f32>,
    value: f32,
) -> Result<()> {
    let numel = y.numel();
    if numel == 0 {
        return Ok(());
    }

    let f = load_fn(device, &FILL_CONST_F32_FN, "fill_const_f32")?;
    let stream = device.raw().default_stream();
    let numel_i32 = numel as i32;
    let mut launcher = stream.launch_builder(&f);
    launcher.arg(y.buf_mut()).arg(&value).arg(&numel_i32);

    unsafe { launcher.launch(launch_cfg(numel)) }
        .map_err(|e| anyhow!("fill_const_f32 launch (numel={}): {:?}", numel, e))?;
    Ok(())
}

/// GDN gate stand-in (V-mean version): for each `(t, hi)` pair
/// writes `g[hi + t*H_v] = clamp(-|mean(v[t, hi, :])| - 1.0, -5.0, -1.0)`.
///
/// Matches the pre-real-weights host computation exactly. This is
/// the computation the previous `g_host = vec![...]; for t { for hi
/// { ... } }` + `from_host` did — now runs entirely on device, one
/// thread per `(t, hi)` pair, single launch per GDN forward.
///
/// `v` is laid out `[n_tokens, H_v, S_v]` row-major (the GDN kernel
/// ultimately reshapes it to `[S_v, H_v, n_tokens, n_seqs]`; this
/// kernel reads the same underlying linear buffer).
pub fn launch_gdn_gate_v_mean_standin_f32(
    device: &Arc<DeviceContext>,
    v: &CudaTensor<f32>,
    g: &mut CudaTensor<f32>,
    n_tokens: usize,
    h_v: usize,
    s_v: usize,
) -> Result<()> {
    let v_needed = n_tokens * h_v * s_v;
    if v.numel() < v_needed {
        return Err(anyhow!(
            "gdn_gate_v_mean_standin_f32: v numel {} < n_tokens*h_v*s_v = {}",
            v.numel(),
            v_needed
        ));
    }
    let g_needed = n_tokens * h_v;
    if g.numel() < g_needed {
        return Err(anyhow!(
            "gdn_gate_v_mean_standin_f32: g numel {} < n_tokens*h_v = {}",
            g.numel(),
            g_needed
        ));
    }
    if g_needed == 0 {
        return Ok(());
    }

    let f = load_fn(
        device,
        &GDN_GATE_V_MEAN_STANDIN_F32_FN,
        "gdn_gate_v_mean_standin_f32",
    )?;
    let stream = device.raw().default_stream();
    let n_tokens_i32 = n_tokens as i32;
    let h_v_i32 = h_v as i32;
    let s_v_i32 = s_v as i32;
    let mut launcher = stream.launch_builder(&f);
    launcher
        .arg(v.buf())
        .arg(g.buf_mut())
        .arg(&n_tokens_i32)
        .arg(&h_v_i32)
        .arg(&s_v_i32);

    unsafe { launcher.launch(launch_cfg(g_needed)) }.map_err(|e| {
        anyhow!(
            "gdn_gate_v_mean_standin_f32 launch (n_tokens={} h_v={} s_v={}): {:?}",
            n_tokens,
            h_v,
            s_v,
            e
        )
    })?;
    Ok(())
}
