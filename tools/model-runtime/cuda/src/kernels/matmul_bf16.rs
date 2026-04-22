//! Dense bf16 × bf16 matmul with f32 accumulation. Two entry points:
//!
//!   * `launch_matmul_bf16_bf16` — C is bf16 (feed-forward / output
//!     projections whose downstream consumer wants bf16 activations).
//!   * `launch_matmul_bf16_f32`  — C is f32 (attention scores, where
//!     the subsequent softmax needs extra precision).
//!
//! Row-major throughout. Math is C[M,N] = A[M,K] · B[K,N]; caller
//! reshapes/permutes if it actually wants `A · B^T`. This kernel is the
//! full-precision complement to `mmvq_q4k` (Q4_K_M-quantized weights)
//! — use it for any matmul whose weights ship as plain bf16.
//!
//! Implementation: 32×32 output tile, one thread per output element,
//! TILE_K=32 loaded into shared with +1 padding to avoid bank
//! conflicts. No tensor-core MMA in this first port; plain fp32 FMAs
//! with bf16 fetches. See `kernels/matmul_bf16.cu` for the device-side
//! details.
//!
//! TODO: tensor-core MMA upgrade (wmma / mma.sync) once correctness is
//!       locked in — should deliver >4× throughput on sm_86.
//! TODO: shapes not divisible by 32. Current host validation rejects
//!       them; we'll add a padded-tile path or a separate small-tile
//!       kernel when a call site needs it.

use std::sync::{Arc, OnceLock};

use anyhow::{anyhow, Result};
use cudarc::driver::{CudaFunction, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;
use half::bf16;

use crate::device::DeviceContext;
use crate::tensor::CudaTensor;

// PTX blob emitted by build.rs for kernels/matmul_bf16.cu.
use super::MATMUL_BF16_PTX;

/// Tile dimensions mirror the kernel's `TILE_M / TILE_N / TILE_K` —
/// keep these in sync with `kernels/matmul_bf16.cu`.
const TILE_M: usize = 32;
const TILE_N: usize = 32;
const TILE_K: usize = 32;

static MATMUL_BF16_BF16_OUT_FN: OnceLock<CudaFunction> = OnceLock::new();
static MATMUL_BF16_F32_OUT_FN: OnceLock<CudaFunction> = OnceLock::new();

/// Load and cache one of the two entry-point symbols from the shared
/// PTX module. Pattern mirrors `mmq_q4k::load_mmq_q4k_fn`.
fn load_fn(
    device: &Arc<DeviceContext>,
    cache: &'static OnceLock<CudaFunction>,
    sym: &'static str,
) -> Result<CudaFunction> {
    if let Some(f) = cache.get() {
        return Ok(f.clone());
    }
    let ptx_src = std::str::from_utf8(MATMUL_BF16_PTX)
        .map_err(|e| anyhow!("matmul_bf16.ptx not UTF-8: {}", e))?
        .to_string();
    let module = device
        .raw()
        .load_module(Ptx::from_src(ptx_src))
        .map_err(|e| anyhow!("load_module matmul_bf16.ptx: {:?}", e))?;
    let f = module
        .load_function(sym)
        .map_err(|e| anyhow!("load_function {}: {:?}", sym, e))?;
    let _ = cache.set(f.clone());
    Ok(f)
}

/// Shared validation. Both entry points require:
///   * A is [M, K], B is [K, N], C is [M, N]
///   * M, K, N all nonzero and all divisible by 32 (tile alignment)
fn validate_shapes<CT>(
    a: &CudaTensor<bf16>,
    b: &CudaTensor<bf16>,
    c: &CudaTensor<CT>,
    m: usize,
    k: usize,
    n: usize,
) -> Result<()>
where
    CT: crate::tensor::TensorElem,
{
    if m == 0 || k == 0 || n == 0 {
        return Err(anyhow!(
            "matmul_bf16: m, k, n must all be nonzero (m={}, k={}, n={})",
            m,
            k,
            n
        ));
    }
    if !m.is_multiple_of(TILE_M) {
        return Err(anyhow!(
            "matmul_bf16: m must be a multiple of {} (got m={})",
            TILE_M,
            m
        ));
    }
    if !k.is_multiple_of(TILE_K) {
        return Err(anyhow!(
            "matmul_bf16: k must be a multiple of {} (got k={})",
            TILE_K,
            k
        ));
    }
    if !n.is_multiple_of(TILE_N) {
        return Err(anyhow!(
            "matmul_bf16: n must be a multiple of {} (got n={})",
            TILE_N,
            n
        ));
    }
    if a.numel() != m * k {
        return Err(anyhow!(
            "matmul_bf16: a.numel()={} != m*k={} (m={}, k={})",
            a.numel(),
            m * k,
            m,
            k
        ));
    }
    if b.numel() != k * n {
        return Err(anyhow!(
            "matmul_bf16: b.numel()={} != k*n={} (k={}, n={})",
            b.numel(),
            k * n,
            k,
            n
        ));
    }
    if c.numel() != m * n {
        return Err(anyhow!(
            "matmul_bf16: c.numel()={} != m*n={} (m={}, n={})",
            c.numel(),
            m * n,
            m,
            n
        ));
    }
    Ok(())
}

/// Launch configuration shared by both entry points — 32×32 output tile
/// → (N/32, M/32) grid of 1024-thread blocks.
fn launch_config(m: usize, n: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: ((n / TILE_N) as u32, (m / TILE_M) as u32, 1),
        block_dim: (TILE_N as u32, TILE_M as u32, 1),
        // Shared memory sized in the kernel via __shared__ arrays; no
        // dynamic shmem needed here.
        shared_mem_bytes: 0,
    }
}

/// `C[M,N] ← A[M,K] · B[K,N]` with bf16 in/out and f32 accumulation.
pub fn launch_matmul_bf16_bf16(
    device: &Arc<DeviceContext>,
    a: &CudaTensor<bf16>,
    b: &CudaTensor<bf16>,
    c: &mut CudaTensor<bf16>,
    m: usize,
    k: usize,
    n: usize,
) -> Result<()> {
    validate_shapes(a, b, c, m, k, n)?;

    let cfg = launch_config(m, n);
    let f = load_fn(device, &MATMUL_BF16_BF16_OUT_FN, "matmul_bf16_bf16_out")?;
    let stream = device.raw().default_stream();
    let m_i32 = m as i32;
    let k_i32 = k as i32;
    let n_i32 = n as i32;
    let mut launcher = stream.launch_builder(&f);
    launcher
        .arg(a.buf())
        .arg(b.buf())
        .arg(c.buf_mut())
        .arg(&m_i32)
        .arg(&k_i32)
        .arg(&n_i32);

    unsafe { launcher.launch(cfg) }
        .map_err(|e| anyhow!("matmul_bf16_bf16_out launch (m={} k={} n={}): {:?}", m, k, n, e))?;
    Ok(())
}

/// Same math as `launch_matmul_bf16_bf16` but writes the f32 accumulator
/// directly. Used for attention scores where the subsequent softmax
/// needs >bf16 precision.
pub fn launch_matmul_bf16_f32(
    device: &Arc<DeviceContext>,
    a: &CudaTensor<bf16>,
    b: &CudaTensor<bf16>,
    c: &mut CudaTensor<f32>,
    m: usize,
    k: usize,
    n: usize,
) -> Result<()> {
    validate_shapes(a, b, c, m, k, n)?;

    let cfg = launch_config(m, n);
    let f = load_fn(device, &MATMUL_BF16_F32_OUT_FN, "matmul_bf16_f32_out")?;
    let stream = device.raw().default_stream();
    let m_i32 = m as i32;
    let k_i32 = k as i32;
    let n_i32 = n as i32;
    let mut launcher = stream.launch_builder(&f);
    launcher
        .arg(a.buf())
        .arg(b.buf())
        .arg(c.buf_mut())
        .arg(&m_i32)
        .arg(&k_i32)
        .arg(&n_i32);

    unsafe { launcher.launch(cfg) }
        .map_err(|e| anyhow!("matmul_bf16_f32_out launch (m={} k={} n={}): {:?}", m, k, n, e))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// CPU golden matmul in f32 — used to compare against both GPU
    /// variants. Inputs come in as already-rounded-to-bf16 f32 values
    /// so the comparison isolates kernel math error from input
    /// representation error.
    fn matmul_cpu_f32(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for kk in 0..k {
                    acc += a[i * k + kk] * b[kk * n + j];
                }
                c[i * n + j] = acc;
            }
        }
    }

    /// Deterministic pseudo-random via simple LCG — host-independent
    /// so the test reproduces across architectures.
    fn lcg_iter(seed: &mut u32) -> f32 {
        *seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
        // Map to roughly [-1, 1].
        ((*seed >> 16) as f32 / 32768.0) - 1.0
    }

    /// Device-backed end-to-end. Ignored by default — run with:
    ///   cargo test -p ctox-engine-cuda --features cuda --release -- \
    ///       --ignored --nocapture matmul_bf16_vs_cpu_golden
    ///
    /// Shape (M=32, K=5120, N=5120) matches a single Qwen3.5-27B full-
    /// attention projection at decode time (one-token batch padded to
    /// the tile's M=32 row count).
    #[test]
    #[ignore]
    fn matmul_bf16_vs_cpu_golden() {
        let m = 32usize;
        let k = 5120usize;
        let n = 5120usize;

        // Generate f32 values, round to bf16, then round-trip back to
        // f32. The round-tripped values are the "true" inputs for both
        // the CPU golden and the GPU kernel — anything else would make
        // us measure bf16 storage error rather than kernel fidelity.
        let mut seed: u32 = 0x9E3779B9;
        let a_bf16: Vec<bf16> = (0..m * k)
            .map(|_| bf16::from_f32(lcg_iter(&mut seed) * 0.25))
            .collect();
        let b_bf16: Vec<bf16> = (0..k * n)
            .map(|_| bf16::from_f32(lcg_iter(&mut seed) * 0.25))
            .collect();
        let a_f32: Vec<f32> = a_bf16.iter().map(|v| v.to_f32()).collect();
        let b_f32: Vec<f32> = b_bf16.iter().map(|v| v.to_f32()).collect();

        // CPU golden.
        let mut c_cpu = vec![0.0f32; m * n];
        matmul_cpu_f32(&a_f32, &b_f32, &mut c_cpu, m, k, n);

        let dev = Arc::new(DeviceContext::new(0).expect("cuda init"));
        let a_gpu =
            CudaTensor::<bf16>::from_host(dev.clone(), vec![m, k], &a_bf16).expect("upload a");
        let b_gpu =
            CudaTensor::<bf16>::from_host(dev.clone(), vec![k, n], &b_bf16).expect("upload b");

        // -------- bf16 output path --------
        {
            let mut c_gpu =
                CudaTensor::<bf16>::zeros(dev.clone(), vec![m, n]).expect("alloc c bf16");
            launch_matmul_bf16_bf16(&dev, &a_gpu, &b_gpu, &mut c_gpu, m, k, n)
                .expect("launch bf16_out");
            dev.synchronize().expect("sync bf16_out");
            let c_host_bf16 = c_gpu.to_host().expect("download c bf16");
            let c_host: Vec<f32> = c_host_bf16.iter().map(|v| v.to_f32()).collect();

            let mut max_abs = 0.0f32;
            let mut max_rel = 0.0f32;
            for (a, b) in c_cpu.iter().zip(c_host.iter()) {
                let d = (a - b).abs();
                if d > max_abs {
                    max_abs = d;
                }
                // The output magnitudes are ~sqrt(k) * sigma^2 ≈ 4.5
                // with sigma=0.25 inputs, so anchor the relative scale
                // against that rather than zero.
                let scale = a.abs().max(b.abs()).max(1e-3);
                let rel = d / scale;
                if rel > max_rel {
                    max_rel = rel;
                }
            }
            eprintln!(
                "matmul_bf16 bf16_out diff: max_abs={:.6e} max_rel={:.6e}",
                max_abs, max_rel
            );
            // Output is bf16: 7-bit mantissa → ~2^-7 quantization on the
            // final store dominates. Task spec: < 5e-3.
            assert!(
                max_rel < 5e-3,
                "bf16_out GPU diverges from CPU golden: max_rel={}",
                max_rel
            );
        }

        // -------- f32 output path --------
        {
            let mut c_gpu =
                CudaTensor::<f32>::zeros(dev.clone(), vec![m, n]).expect("alloc c f32");
            launch_matmul_bf16_f32(&dev, &a_gpu, &b_gpu, &mut c_gpu, m, k, n)
                .expect("launch f32_out");
            dev.synchronize().expect("sync f32_out");
            let c_host = c_gpu.to_host().expect("download c f32");

            let mut max_abs = 0.0f32;
            let mut max_rel = 0.0f32;
            for (a, b) in c_cpu.iter().zip(c_host.iter()) {
                let d = (a - b).abs();
                if d > max_abs {
                    max_abs = d;
                }
                let scale = a.abs().max(b.abs()).max(1e-3);
                let rel = d / scale;
                if rel > max_rel {
                    max_rel = rel;
                }
            }
            eprintln!(
                "matmul_bf16 f32_out  diff: max_abs={:.6e} max_rel={:.6e}",
                max_abs, max_rel
            );
            // Output stays in f32 — only the reduction-order drift over
            // k=5120 elements should show up. Task spec: < 1e-4.
            assert!(
                max_rel < 1e-4,
                "f32_out GPU diverges from CPU golden: max_rel={}",
                max_rel
            );
        }
    }
}
