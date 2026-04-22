//! Rust wrapper around the vendored llama.cpp `rms_norm_f32` template
//! from `ggml-cuda/norm.cu`. We specialize `<block_size=1024,
//! do_multiply=true, do_add=false>` — fused RMSNorm + per-channel weight
//! multiply, which is what Qwen3.5's norm layers use.
//!
//! The template has 23 parameters (see `norm.cu:74`) split into three
//! bands: core (x, dst, ncols, strides, eps), `mul` band (weight ptr,
//! strides, 4×uint3 fastdiv packings), and `add` band (same layout).
//! With `do_add=false` the add band's pointer/strides/packings are
//! ignored by the kernel but the arg slots still have to exist for the
//! launch to match the compiled signature. For the ignored `add`
//! pointer we carry a dummy 1-element `CudaSlice<f32>` (`NULL_F32_SLICE`
//! below) since cudarc's `PushKernelArg::arg` won't accept a raw
//! `*const f32`.
//!
//! Load chain:
//!   build.rs compiles `kernels/sm_86/norm.cu` → `norm.ptx` → embeds it
//!   as `NORM_PTX` in the auto-generated `ptx_registry.rs`. We load the
//!   Itanium-mangled symbol
//!   `_Z12rms_norm_f32ILi1024ELb1ELb0EE…` for the `<1024,true,false>`
//!   instantiation.

use std::sync::{Arc, OnceLock};

use anyhow::{anyhow, Result};
use cudarc::driver::{CudaFunction, CudaSlice, DeviceRepr, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;

use ctox_cuda_primitives::device::DeviceContext;
use ctox_cuda_primitives::tensor::CudaTensor;

// PTX blob comes from the parent module's auto-generated registry.
use super::NORM_PTX;

/// Itanium-mangled symbol for `rms_norm_f32<1024, true, false>` from
/// `vendor/ggml-cuda/norm.cu`. Verified against `norm.ptx` entry list.
const RMS_NORM_F32_1024_MUL_SYM: &str =
    "_Z12rms_norm_f32ILi1024ELb1ELb0EEvPKfPfilllfS1_lll5uint3S3_S3_S3_S1_lllS3_S3_S3_S3_";

/// ggml-cuda's `uint3` packed fastdiv descriptor: `(mp, L, divisor)`.
/// Laid out to match CUDA's `uint3` (three 32-bit uints, 12-byte
/// alignment implied by PTX `.param .align 4 .b8 …[12]`).
#[repr(C)]
#[derive(Clone, Copy)]
struct Uint3 {
    x: u32,
    y: u32,
    z: u32,
}
// Safe: plain POD matching the CUDA `uint3` ABI (3×u32, 4-byte aligned,
// no padding, identical on host and device).
unsafe impl DeviceRepr for Uint3 {}

/// Port of ggml-cuda's `init_fastdiv_values(uint64_t d)` from
/// `common.cuh:865`. Returns `(mp, L, d)` such that
/// `n / d == (mulhi(n, mp) + n) >> L`.
fn init_fastdiv(d: u32) -> Uint3 {
    assert!(d != 0, "init_fastdiv: divisor must be > 0");
    // L = ceil(log2(d)).
    let mut l: u32 = 0;
    while l < 32 && (1u32 << l) < d {
        l += 1;
    }
    // mp = floor(2^32 * (2^L - d) / d) + 1, exactly as upstream.
    let mp = (((1u64 << 32) * ((1u64 << l) - d as u64)) / d as u64 + 1) as u32;
    Uint3 { x: mp, y: l, z: d }
}

/// One-shot cache for the loaded kernel function. Same rationale as
/// the rest of the kernel wrappers: module loading is expensive and
/// this is hot-path.
static FN_CACHE: OnceLock<CudaFunction> = OnceLock::new();

fn rmsnorm_f32_fn(device: &Arc<DeviceContext>) -> Result<CudaFunction> {
    if let Some(f) = FN_CACHE.get() {
        return Ok(f.clone());
    }
    let ptx_src = std::str::from_utf8(NORM_PTX)
        .map_err(|e| anyhow!("norm.ptx not UTF-8: {}", e))?
        .to_string();
    let module = device
        .raw()
        .load_module(Ptx::from_src(ptx_src))
        .map_err(|e| anyhow!("load_module norm.ptx: {:?}", e))?;
    let f = module
        .load_function(RMS_NORM_F32_1024_MUL_SYM)
        .map_err(|e| anyhow!("load_function rms_norm_f32<1024,true,false>: {:?}", e))?;
    let _ = FN_CACHE.set(f.clone());
    Ok(f)
}

/// Cached zero-backed dummy slice used as the `add` pointer when
/// `do_add=false`. Allocated once per process on first call; the kernel
/// never dereferences it (the `if constexpr (do_add)` branch is
/// statically false for our instantiation).
static NULL_F32_SLICE: OnceLock<CudaSlice<f32>> = OnceLock::new();

fn null_f32_slice(device: &Arc<DeviceContext>) -> Result<&'static CudaSlice<f32>> {
    if let Some(s) = NULL_F32_SLICE.get() {
        return Ok(s);
    }
    let s = device
        .raw()
        .default_stream()
        .alloc_zeros::<f32>(1)
        .map_err(|e| anyhow!("alloc null_f32_slice: {:?}", e))?;
    // Race-tolerant: whoever wins stores the permanent slice; the
    // losing allocation is dropped (4 B of GPU memory). `get()` after
    // the set (or the race-winner's set) always returns the same slice.
    let _ = NULL_F32_SLICE.set(s);
    Ok(NULL_F32_SLICE.get().expect("just initialized"))
}

/// `y ← (x / sqrt(mean(x²) + eps)) * weight`
///
/// Shapes:
///   * `x`:      `[n_tokens, hidden_dim]` f32 row-major
///   * `weight`: `[hidden_dim]`            f32 (broadcast across rows)
///   * `y`:      `[n_tokens, hidden_dim]` f32 (pre-allocated output)
///
/// Does not synchronize the stream. Caller syncs at phase boundary.
pub fn launch_rmsnorm_f32(
    device: &Arc<DeviceContext>,
    x: &CudaTensor<f32>,
    weight: &CudaTensor<f32>,
    y: &mut CudaTensor<f32>,
    eps: f32,
) -> Result<()> {
    // Shape validation.
    if x.shape().len() != 2 {
        return Err(anyhow!(
            "rmsnorm: x must be 2D [n_tokens, hidden_dim], got {:?}",
            x.shape()
        ));
    }
    if weight.shape().len() != 1 {
        return Err(anyhow!(
            "rmsnorm: weight must be 1D [hidden_dim], got {:?}",
            weight.shape()
        ));
    }
    if y.shape() != x.shape() {
        return Err(anyhow!(
            "rmsnorm: y.shape {:?} != x.shape {:?}",
            y.shape(),
            x.shape()
        ));
    }
    let n_tokens = x.shape()[0];
    let hidden_dim = x.shape()[1];
    if weight.shape()[0] != hidden_dim {
        return Err(anyhow!(
            "rmsnorm: weight dim {} != x hidden_dim {}",
            weight.shape()[0],
            hidden_dim
        ));
    }
    if n_tokens == 0 || hidden_dim == 0 {
        // Nothing to do. Avoid launching a 0-block grid.
        return Ok(());
    }

    // Launch config matches upstream `rms_norm_mul_f32_cuda` on the
    // `ncols >= 1024` branch: block_dim.x = 1024, one block per row,
    // shmem = 32 × sizeof(float) for the warp-partials scratch.
    let cfg = LaunchConfig {
        grid_dim: (n_tokens as u32, 1, 1),
        block_dim: (1024, 1, 1),
        shared_mem_bytes: 32 * 4,
    };

    let f = rmsnorm_f32_fn(device)?;
    let stream = device.raw().default_stream();

    // Core band.
    let ncols: i32 = hidden_dim as i32;
    let stride_row: i64 = hidden_dim as i64;
    let stride_channel: i64 = 0;
    let stride_sample: i64 = 0;

    // Multiply band — weight is [hidden_dim] and broadcasts across
    // rows/channels/samples, so row/channel/sample strides are 0 and
    // the row/channel/sample "sizes" (for the fastmodulo reindex) are
    // all 1. The kernel always evaluates the `mul_ncols` fastmodulo
    // because `do_multiply=true`, so that one needs real values.
    let mul_stride_row: i64 = 0;
    let mul_stride_channel: i64 = 0;
    let mul_stride_sample: i64 = 0;
    let mul_ncols_packed = init_fastdiv(hidden_dim as u32);
    let mul_nrows_packed = init_fastdiv(1);
    let mul_nchannels_packed = init_fastdiv(1);
    let mul_nsamples_packed = init_fastdiv(1);

    // Add band — `do_add=false`, so the kernel never reads the add
    // pointer or touches the add fastmodulo packings. We still have to
    // fill the arg slots. cudarc's `PushKernelArg::arg` requires a
    // `DeviceRepr` value, so we pass a cached dummy `CudaSlice<f32>`
    // (the kernel ignores the pointer under `if constexpr (do_add)`).
    let null_add = null_f32_slice(device)?;
    let zero_stride: i64 = 0;
    let zero_packed = init_fastdiv(1);

    let mut launcher = stream.launch_builder(&f);
    launcher
        .arg(x.buf())
        .arg(y.buf_mut())
        .arg(&ncols)
        .arg(&stride_row)
        .arg(&stride_channel)
        .arg(&stride_sample)
        .arg(&eps)
        .arg(weight.buf())
        .arg(&mul_stride_row)
        .arg(&mul_stride_channel)
        .arg(&mul_stride_sample)
        .arg(&mul_ncols_packed)
        .arg(&mul_nrows_packed)
        .arg(&mul_nchannels_packed)
        .arg(&mul_nsamples_packed)
        .arg(null_add)
        .arg(&zero_stride)
        .arg(&zero_stride)
        .arg(&zero_stride)
        .arg(&zero_packed)
        .arg(&zero_packed)
        .arg(&zero_packed)
        .arg(&zero_packed);

    unsafe { launcher.launch(cfg) }.map_err(|e| {
        anyhow!(
            "rms_norm_f32<1024,true,false> launch (n_tokens={} hidden={}): {:?}",
            n_tokens,
            hidden_dim,
            e
        )
    })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// CPU reference — used by the on-host integration test.
    fn rmsnorm_cpu(x: &[f32], weight: &[f32], y: &mut [f32], eps: f32) {
        let n_tokens = x.len() / weight.len();
        let hidden_dim = weight.len();
        for t in 0..n_tokens {
            let row = &x[t * hidden_dim..(t + 1) * hidden_dim];
            let mean_sq: f32 = row.iter().map(|v| v * v).sum::<f32>() / hidden_dim as f32;
            let scale = 1.0 / (mean_sq + eps).sqrt();
            let y_row = &mut y[t * hidden_dim..(t + 1) * hidden_dim];
            for i in 0..hidden_dim {
                y_row[i] = row[i] * scale * weight[i];
            }
        }
    }

    /// Device-backed end-to-end. Ignored by default — run with:
    ///   cargo test -p ctox-qwen35-27b --features cuda --release -- \
    ///       --ignored --nocapture rmsnorm_vs_cpu_golden
    #[test]
    #[ignore]
    fn rmsnorm_vs_cpu_golden() {
        // Use a shape representative of Qwen3.5-27B (hidden=5120) so the
        // test exercises the warp fan-in path with >1 warp per block.
        let n_tokens = 8usize;
        let hidden_dim = 5120usize;
        let eps = 1e-6f32;

        // Deterministic pseudo-random via simple LCG so the test is
        // host-independent.
        let mut seed: u32 = 0x9E3779B9;
        let mut rand_f = || -> f32 {
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            // Map to roughly [-1, 1].
            ((seed >> 16) as f32 / 32768.0) - 1.0
        };
        let x_host: Vec<f32> = (0..n_tokens * hidden_dim).map(|_| rand_f()).collect();
        let w_host: Vec<f32> = (0..hidden_dim).map(|_| rand_f().abs() + 0.1).collect();

        // CPU golden.
        let mut y_cpu = vec![0.0f32; n_tokens * hidden_dim];
        rmsnorm_cpu(&x_host, &w_host, &mut y_cpu, eps);

        // Device run.
        let dev = Arc::new(DeviceContext::new(0).expect("cuda init"));
        let x = CudaTensor::<f32>::from_host(
            dev.clone(),
            vec![n_tokens, hidden_dim],
            &x_host,
        )
        .expect("upload x");
        let w = CudaTensor::<f32>::from_host(dev.clone(), vec![hidden_dim], &w_host)
            .expect("upload w");
        let mut y = CudaTensor::<f32>::zeros(dev.clone(), vec![n_tokens, hidden_dim])
            .expect("alloc y");

        launch_rmsnorm_f32(&dev, &x, &w, &mut y, eps).expect("launch");
        dev.synchronize().expect("sync");

        let y_gpu = y.to_host().expect("download y");

        // Compare. RMSNorm sums-of-squares over 5120 f32s in different
        // orders between CPU sequential and GPU warp-reduction; we
        // expect relative drift on the order of 5120 × machine_eps ≈ 3e-4.
        let mut max_abs = 0.0f32;
        let mut max_rel = 0.0f32;
        for (a, b) in y_cpu.iter().zip(y_gpu.iter()) {
            let d = (a - b).abs();
            if d > max_abs {
                max_abs = d;
            }
            let scale = a.abs().max(b.abs()).max(1e-6);
            let rel = d / scale;
            if rel > max_rel {
                max_rel = rel;
            }
        }
        eprintln!("rmsnorm diff: max_abs={:.6e} max_rel={:.6e}", max_abs, max_rel);
        // Tight tolerance — f32 RMSNorm with 5120 elements should match
        // within a few machine_eps.
        assert!(
            max_rel < 1e-3,
            "GPU result diverges from CPU golden: max_rel={}",
            max_rel
        );
    }
}
