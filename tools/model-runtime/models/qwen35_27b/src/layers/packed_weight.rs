//! On-device packed weight carrier.
//!
//! A `[K, N]` weight matrix at GGUF-load time can arrive in several
//! dtypes:
//!
//!   * `bf16` — unpacked (what the first-port layers used everywhere).
//!   * `Q4_K_M`, `Q5_K`, `Q6_K`, `Q8_0` — llama.cpp-style block-quant
//!     bytes (packed on device as `CudaTensor<i8>`).
//!
//! Per-layer forward() code shouldn't have to know which of these it's
//! looking at. `PackedWeight` wraps the carrier + metadata and
//! dispatches to the right mat-vec / mat-mul kernel in a single
//! `matmul_f32` entry point. The residual-stream casts (bf16 → f32
//! at pre-projection, f32 → bf16 after) now flank every projection;
//! they were already present for rmsnorm, so the refactor is
//! essentially "leave the f32 staging tensor in place through the
//! projection rather than round-tripping through bf16".
//!
//! # API contract
//!
//! `matmul_f32(device, x, y)` computes
//!
//! ```text
//!     y[m, n]  ←  x[m, k] · A[k, n]
//! ```
//!
//! where `A` is the weight this `PackedWeight` represents. `x` and `y`
//! are both `CudaTensor<f32>`. Shapes are checked against
//! `self.dims()` (the stored `(k, n)`). `m` is inferred from
//! `x.shape()[0]` (with `y.shape()[0] == m` and `y.shape()[1] == n`).
//!
//! ## Dispatch table
//!
//! | Variant | Path                                         |
//! |---------|----------------------------------------------|
//! | `Bf16`  | cast x → bf16; `launch_matmul_bf16_f32` GEMM |
//! | `Q4K`   | loop m rows; per-row `launch_mmvq_q4k_f32`   |
//! | `Q5K`   | loop m rows; per-row `launch_mmvq_q5k_f32`   |
//! | `Q6K`   | loop m rows; per-row `launch_mmvq_q6k_f32`   |
//! | `Q8_0`  | loop m rows; per-row `launch_mmvq_q8_0_f32`  |
//! | `IQ4XS` | loop m rows; per-row `launch_mmvq_iq4_xs_f32`|
//! | `Zero`  | memset y to zeros; no kernel launch          |
//!
//! The per-row mmvq loop is strictly correctness-first; a batched
//! mmvq entry point is a Phase-5 optimization (same cost-class as
//! the existing `gather_head_from_packed` loops in the FA layer).

use std::sync::Arc;

use anyhow::{anyhow, Result};
use half::bf16;

use ctox_cuda_primitives::device::DeviceContext;
use ctox_cuda_primitives::tensor::CudaTensor;

use crate::kernels::{
    launch_cast_f32_to_bf16, launch_matmul_bf16_f32, launch_mmvq_iq4_xs_f32, launch_mmvq_q4k_f32,
    launch_mmvq_q5k_f32, launch_mmvq_q6k_f32, launch_mmvq_q8_0_f32,
};

/// A weight tensor as it lives on device — carrier type depends on
/// the GGUF dtype the loader saw at read time. Each variant stores the
/// logical `[K, N]` shape alongside the carrier so forward() code can
/// dispatch on the variant without having to know the packed byte
/// count per block.
///
/// `matmul_f32` below routes the call to the right kernel; callers
/// don't match on the variant directly.
pub enum PackedWeight {
    /// `[K, N]` bf16 dense. Used today for RMSNorm-adjacent linear
    /// ops and for smoke tests that synthesize random weights.
    Bf16 {
        t: CudaTensor<bf16>,
        k: usize,
        n: usize,
    },
    /// `[K, N]` Q4_K_M packed bytes (n_elements / 256 × 144 bytes).
    Q4K {
        t: CudaTensor<i8>,
        k: usize,
        n: usize,
    },
    /// `[K, N]` Q5_K packed bytes (n_elements / 256 × 176 bytes).
    Q5K {
        t: CudaTensor<i8>,
        k: usize,
        n: usize,
    },
    /// `[K, N]` Q6_K packed bytes (n_elements / 256 × 210 bytes).
    Q6K {
        t: CudaTensor<i8>,
        k: usize,
        n: usize,
    },
    /// `[K, N]` Q8_0 packed bytes (n_elements / 32 × 34 bytes).
    Q8_0 {
        t: CudaTensor<i8>,
        k: usize,
        n: usize,
    },
    /// `[K, N]` IQ4_XS packed bytes (n_elements / 256 × 136 bytes).
    /// The shipping 27B GGUF ships `ffn_gate.weight` and `ffn_up.weight`
    /// in this format; dispatch goes through `launch_mmvq_iq4_xs_f32`.
    IQ4XS {
        t: CudaTensor<i8>,
        k: usize,
        n: usize,
    },
    /// Zero placeholder — the weight wasn't loaded (missing from GGUF
    /// or unsupported dtype). Forward path still runs; output is all
    /// zeros for this projection. Matches the pre-refactor behavior
    /// of `load_bf16_placeholder` returning a zeroed bf16 tensor.
    Zero { k: usize, n: usize },
}

impl PackedWeight {
    /// `(K, N)` logical shape of the weight.
    pub fn dims(&self) -> (usize, usize) {
        match self {
            PackedWeight::Bf16 { k, n, .. } => (*k, *n),
            PackedWeight::Q4K { k, n, .. } => (*k, *n),
            PackedWeight::Q5K { k, n, .. } => (*k, *n),
            PackedWeight::Q6K { k, n, .. } => (*k, *n),
            PackedWeight::Q8_0 { k, n, .. } => (*k, *n),
            PackedWeight::IQ4XS { k, n, .. } => (*k, *n),
            PackedWeight::Zero { k, n } => (*k, *n),
        }
    }

    /// `y[m, n]  ←  x[m, k] · A[k, n]`, f32 in/out, dispatching on the
    /// carrier variant. `m` is taken from `x.shape()[0]`.
    ///
    /// Validates:
    ///   * `x.shape() == [m, k]`
    ///   * `y.shape() == [m, n]`
    ///
    /// Returns an error on any mismatch.
    pub fn matmul_f32(
        &self,
        device: &Arc<DeviceContext>,
        x: &CudaTensor<f32>,
        y: &mut CudaTensor<f32>,
    ) -> Result<()> {
        let (k, n) = self.dims();

        if x.shape().len() != 2 {
            return Err(anyhow!(
                "PackedWeight::matmul_f32: x must be 2D [m, k], got {:?}",
                x.shape()
            ));
        }
        let m = x.shape()[0];
        if x.shape()[1] != k {
            return Err(anyhow!(
                "PackedWeight::matmul_f32: x.shape()[1]={} != k={}",
                x.shape()[1],
                k
            ));
        }
        if y.shape() != [m, n] {
            return Err(anyhow!(
                "PackedWeight::matmul_f32: y.shape {:?} != [{}, {}]",
                y.shape(),
                m,
                n
            ));
        }

        match self {
            PackedWeight::Bf16 { t, .. } => matmul_bf16_batched(device, t, x, y, m, k, n),
            PackedWeight::Q4K { t, .. } => {
                matmul_q_rows(device, x, y, m, k, n, |dev, xr, yr| {
                    launch_mmvq_q4k_f32(dev, t, k, n, xr, yr)
                })
            }
            PackedWeight::Q5K { t, .. } => {
                matmul_q_rows(device, x, y, m, k, n, |dev, xr, yr| {
                    launch_mmvq_q5k_f32(dev, t, k, n, xr, yr)
                })
            }
            PackedWeight::Q6K { t, .. } => {
                matmul_q_rows(device, x, y, m, k, n, |dev, xr, yr| {
                    launch_mmvq_q6k_f32(dev, t, k, n, xr, yr)
                })
            }
            PackedWeight::Q8_0 { t, .. } => {
                matmul_q_rows(device, x, y, m, k, n, |dev, xr, yr| {
                    launch_mmvq_q8_0_f32(dev, t, k, n, xr, yr)
                })
            }
            PackedWeight::IQ4XS { t, .. } => {
                matmul_q_rows(device, x, y, m, k, n, |dev, xr, yr| {
                    launch_mmvq_iq4_xs_f32(dev, t, k, n, xr, yr)
                })
            }
            PackedWeight::Zero { .. } => zero_fill_f32(y, m * n),
        }
    }
}

/// `y[m, n] ← x[m, k] · A[k, n]` for the Bf16 variant.
///
/// cuBLAS wants bf16 inputs; we stage `x` into a bf16 scratch buffer
/// and call `launch_matmul_bf16_f32` which writes the f32 accumulator
/// directly. One bf16 scratch allocation per call — the caller's `y`
/// is already f32.
fn matmul_bf16_batched(
    device: &Arc<DeviceContext>,
    a_bf16: &CudaTensor<bf16>,
    x: &CudaTensor<f32>,
    y: &mut CudaTensor<f32>,
    m: usize,
    k: usize,
    n: usize,
) -> Result<()> {
    if a_bf16.shape() != [k, n] {
        return Err(anyhow!(
            "PackedWeight::Bf16: a.shape {:?} != [{}, {}]",
            a_bf16.shape(),
            k,
            n
        ));
    }
    let mut x_bf16 = CudaTensor::<bf16>::zeros(device.clone(), vec![m, k])?;
    // Need a bf16 view of x — do a device-side cast from f32 to bf16
    // via the existing kernel. Keeping the `f32` output on `y` lets
    // the caller keep downstream ops (softmax inputs, etc.) in f32
    // when they need the precision.
    launch_cast_f32_to_bf16(device, x, &mut x_bf16)?;
    launch_matmul_bf16_f32(device, &x_bf16, a_bf16, y, m, k, n)?;
    Ok(())
}

/// Loop per-row dispatch for Q4K/Q5K/Q6K/Q8_0 variants.
///
/// We copy row `t` of `x` into a size-`k` scratch tensor, call the
/// mmvq launcher (writes a size-`n` row), then copy that row back into
/// `y[t, :]`. Two D2D memcpys per row; cost class matches the existing
/// `gather_head_from_packed` loops in the FA layer. A batched mmvq
/// entry point would collapse both memcpys into a single launch.
fn matmul_q_rows<F>(
    device: &Arc<DeviceContext>,
    x: &CudaTensor<f32>,
    y: &mut CudaTensor<f32>,
    m: usize,
    k: usize,
    n: usize,
    mut launch: F,
) -> Result<()>
where
    F: FnMut(&Arc<DeviceContext>, &CudaTensor<f32>, &mut CudaTensor<f32>) -> Result<()>,
{
    let mut x_row = CudaTensor::<f32>::zeros(device.clone(), vec![k])?;
    let mut y_row = CudaTensor::<f32>::zeros(device.clone(), vec![n])?;
    let stream = device.raw().default_stream();
    for t in 0..m {
        let x_src_start = t * k;
        let x_src_end = x_src_start + k;
        let x_view = x.buf().slice(x_src_start..x_src_end);
        stream
            .memcpy_dtod(&x_view, x_row.buf_mut())
            .map_err(|e| anyhow!("PackedWeight::matmul_q_rows: x row {} memcpy: {:?}", t, e))?;

        launch(device, &x_row, &mut y_row)?;

        let y_dst_start = t * n;
        let y_dst_end = y_dst_start + n;
        let y_src_view = y_row.buf().slice(..n);
        let mut y_dst_view = y.buf_mut().slice_mut(y_dst_start..y_dst_end);
        stream
            .memcpy_dtod(&y_src_view, &mut y_dst_view)
            .map_err(|e| anyhow!("PackedWeight::matmul_q_rows: y row {} memcpy: {:?}", t, e))?;
    }
    Ok(())
}

/// Overwrite `y`'s first `numel` elements with zeros. Used by the
/// `Zero` variant. Implemented via a host zero-vector upload so we
/// don't need a dedicated memset kernel; `numel = m * n` is tiny
/// compared to any real projection and this runs once per `Zero`
/// forward, not inside the layer loop.
fn zero_fill_f32(y: &mut CudaTensor<f32>, numel: usize) -> Result<()> {
    let stream = y.device().raw().default_stream();
    let zeros = vec![0.0f32; numel];
    stream
        .memcpy_htod(&zeros, y.buf_mut())
        .map_err(|e| anyhow!("PackedWeight::Zero zero_fill htod ({}): {:?}", numel, e))?;
    Ok(())
}

