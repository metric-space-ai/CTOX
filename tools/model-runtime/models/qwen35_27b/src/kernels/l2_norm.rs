//! Rust wrapper around the vendored llama.cpp `l2_norm_f32` template
//! from `ggml-cuda/norm.cu`. We specialize `<block_size=1024>` — fused
//! L2 row-normalize, one block per row, warp-reduce fan-in.
//!
//! Upstream's kernel is f32-only. The prior in-house bf16 kernel took
//! bf16 in and out with f32 math; the corresponding `launch_l2_norm_bf16`
//! wrapper is preserved as an error stub so the crate's public surface
//! doesn't break compile — all call sites go through `launch_l2_norm_f32`
//! now (GDN does a cast bf16→f32 before this kernel anyway).
//!
//! Load chain:
//!   build.rs compiles `kernels/sm_86/norm.cu` → `norm.ptx` → embeds it
//!   as `NORM_PTX` in the auto-generated `ptx_registry.rs`. We load the
//!   Itanium-mangled symbol `_Z11l2_norm_f32ILi1024EE…` for the `<1024>`
//!   instantiation.

use std::sync::{Arc, OnceLock};

use anyhow::{anyhow, Result};
use cudarc::driver::{CudaFunction, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;
use half::bf16;

use ctox_cuda_primitives::device::DeviceContext;
use ctox_cuda_primitives::tensor::CudaTensor;

// PTX blob comes from the parent module's auto-generated registry.
use super::NORM_PTX;

/// Itanium-mangled symbol for `l2_norm_f32<1024>` from
/// `vendor/ggml-cuda/norm.cu`. Verified against `norm.ptx` entry list.
const L2_NORM_F32_1024_SYM: &str = "_Z11l2_norm_f32ILi1024EEvPKfPfilllf";

/// One-shot cache for the loaded kernel function. Module loading is
/// expensive; hot-path wants zero setup cost after first call.
static FN_CACHE: OnceLock<CudaFunction> = OnceLock::new();

fn l2_norm_f32_fn(device: &Arc<DeviceContext>) -> Result<CudaFunction> {
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
        .load_function(L2_NORM_F32_1024_SYM)
        .map_err(|e| anyhow!("load_function l2_norm_f32<1024>: {:?}", e))?;
    let _ = FN_CACHE.set(f.clone());
    Ok(f)
}

/// `y[i, :] ← x[i, :] / sqrt(max(sum(x[i, :]^2), eps^2))`, f32 in/out.
/// (Matches PyTorch's `F.normalize` semantics, per upstream norm.cu.)
///
/// Shapes:
///   * `x`: `[n_rows, n_cols]` f32 row-major
///   * `y`: `[n_rows, n_cols]` f32 (pre-allocated output, same shape)
///
/// Does not synchronize the stream. Caller syncs at phase boundary.
pub fn launch_l2_norm_f32(
    device: &Arc<DeviceContext>,
    x: &CudaTensor<f32>,
    y: &mut CudaTensor<f32>,
    eps: f32,
) -> Result<()> {
    if x.shape().len() != 2 {
        return Err(anyhow!(
            "l2_norm: x must be 2D [n_rows, n_cols], got {:?}",
            x.shape()
        ));
    }
    if y.shape() != x.shape() {
        return Err(anyhow!(
            "l2_norm: y.shape {:?} != x.shape {:?}",
            y.shape(),
            x.shape()
        ));
    }
    let n_rows = x.shape()[0];
    let n_cols = x.shape()[1];
    if n_rows == 0 || n_cols == 0 {
        // Nothing to do. Avoid launching a 0-block grid.
        return Ok(());
    }

    // Launch config matches upstream `l2_norm_f32_cuda` on the
    // `ncols >= 1024` branch: block_dim.x = 1024, one block per row,
    // shmem = 32 × sizeof(float) for the warp-partials scratch.
    let cfg = LaunchConfig {
        grid_dim: (n_rows as u32, 1, 1),
        block_dim: (1024, 1, 1),
        shared_mem_bytes: 32 * 4,
    };

    let f = l2_norm_f32_fn(device)?;
    let stream = device.raw().default_stream();

    let n_cols_i32: i32 = n_cols as i32;
    let stride_row: i64 = n_cols as i64;
    let stride_channel: i64 = 0;
    let stride_sample: i64 = 0;

    let mut launcher = stream.launch_builder(&f);
    launcher
        .arg(x.buf())
        .arg(y.buf_mut())
        .arg(&n_cols_i32)
        .arg(&stride_row)
        .arg(&stride_channel)
        .arg(&stride_sample)
        .arg(&eps);

    unsafe { launcher.launch(cfg) }.map_err(|e| {
        anyhow!(
            "l2_norm_f32<1024> launch (n_rows={} n_cols={}): {:?}",
            n_rows,
            n_cols,
            e
        )
    })?;
    Ok(())
}

/// Legacy bf16 wrapper — preserved so the public re-export in
/// `kernels/mod.rs` keeps resolving. Upstream's kernel is f32-only;
/// callers must cast bf16 → f32 → bf16 around `launch_l2_norm_f32`. The
/// only in-tree caller today (GDN's Q/K path) already operates in f32
/// by the time it hits this kernel, so this stub is placeholder and
/// should never fire in production.
pub fn launch_l2_norm_bf16(
    _device: &Arc<DeviceContext>,
    _x: &CudaTensor<bf16>,
    _y: &mut CudaTensor<bf16>,
    _eps: f32,
) -> Result<()> {
    Err(anyhow!(
        "launch_l2_norm_bf16: vendored upstream kernel is f32 only; \
         caller must cast bf16 → f32 → bf16 around launch_l2_norm_f32"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// CPU reference matching upstream's `dst[col] = scale * x[col]`
    /// with `scale = rsqrt(max(sum_sq, eps^2))`.
    fn l2_norm_cpu_f32(x: &[f32], y: &mut [f32], n_rows: usize, n_cols: usize, eps: f32) {
        for r in 0..n_rows {
            let row = &x[r * n_cols..(r + 1) * n_cols];
            let sum_sq: f32 = row.iter().map(|v| v * v).sum::<f32>();
            let scale = 1.0 / sum_sq.max(eps * eps).sqrt();
            let y_row = &mut y[r * n_cols..(r + 1) * n_cols];
            for i in 0..n_cols {
                y_row[i] = row[i] * scale;
            }
        }
    }

    /// Deterministic pseudo-random via simple LCG so the test is host-
    /// independent.
    fn lcg_iter(seed: &mut u32) -> f32 {
        *seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
        ((*seed >> 16) as f32 / 32768.0) - 1.0
    }

    /// Device-backed end-to-end. Ignored by default — run with:
    ///   cargo test -p ctox-qwen35-27b --features cuda --release -- \
    ///       --ignored --nocapture l2_norm_vs_cpu_golden
    ///
    /// Shape `[n_rows=16, n_cols=1024]` exercises the `block_size=1024`
    /// warp fan-in path (32 warps × warp-reduce → 32-element shmem
    /// scratch → final warp-reduce). Smaller widths route to the
    /// `WARP_SIZE` instantiation which we don't wrap.
    #[test]
    #[ignore]
    fn l2_norm_vs_cpu_golden() {
        let n_rows = 16usize;
        let n_cols = 1024usize;
        let numel = n_rows * n_cols;
        let eps = 1e-6f32;

        let mut seed: u32 = 0x9E3779B9;
        let x_host: Vec<f32> = (0..numel).map(|_| lcg_iter(&mut seed)).collect();

        let mut y_cpu = vec![0.0f32; numel];
        l2_norm_cpu_f32(&x_host, &mut y_cpu, n_rows, n_cols, eps);

        // Device run.
        let dev = Arc::new(DeviceContext::new(0).expect("cuda init"));
        let x = CudaTensor::<f32>::from_host(dev.clone(), vec![n_rows, n_cols], &x_host)
            .expect("upload x");
        let mut y = CudaTensor::<f32>::zeros(dev.clone(), vec![n_rows, n_cols])
            .expect("alloc y");

        launch_l2_norm_f32(&dev, &x, &mut y, eps).expect("launch");
        dev.synchronize().expect("sync");

        let y_gpu = y.to_host().expect("download y");

        // f32 math, sums of 1024 squares — expect ~1024 × machine_eps
        // relative drift from reordering, well under 1e-4.
        let mut max_abs = 0.0f32;
        let mut max_rel = 0.0f32;
        for (a, b) in y_cpu.iter().zip(y_gpu.iter()) {
            let d = (a - b).abs();
            if d > max_abs {
                max_abs = d;
            }
            let scale_v = a.abs().max(b.abs()).max(1e-6);
            let rel = d / scale_v;
            if rel > max_rel {
                max_rel = rel;
            }
        }
        eprintln!(
            "l2_norm (vendored) diff: max_abs={:.6e} max_rel={:.6e}",
            max_abs, max_rel
        );
        assert!(
            max_rel < 1e-3,
            "GPU result diverges from CPU golden: max_rel={}",
            max_rel
        );
    }
}
