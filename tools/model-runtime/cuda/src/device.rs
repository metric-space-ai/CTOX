//! CUDA device + stream context. One per process (single device for
//! now — multi-GPU comes with NCCL wiring later). All allocations,
//! kernel launches, and memcpys go through a `DeviceContext`.
//!
//! Kept minimal intentionally: this is the ONE type that touches the
//! global CUDA context. Every other type in the crate borrows from
//! it by `Arc<DeviceContext>`.

use std::sync::Arc;

use anyhow::{Context, Result};
use cudarc::driver::CudaContext;

/// Handle on a single CUDA device + default stream.
///
/// The default stream is NOT the CUDA-legacy default — we explicitly
/// create a per-context stream so mixing with other CUDA libraries
/// (e.g. the FFI-loaded dflash reference during the transition
/// period) doesn't serialize everything through the legacy default.
///
/// Clone is cheap: internally `Arc<CudaContext>`.
#[derive(Clone)]
pub struct DeviceContext {
    ctx: Arc<CudaContext>,
    ordinal: usize,
}

impl DeviceContext {
    /// Initialize CUDA on device `ordinal` (0-indexed). Must be called
    /// once per process before any tensor allocation. Subsequent calls
    /// on the same ordinal return a fresh handle sharing the same
    /// underlying context (cudarc internally refcounts).
    pub fn new(ordinal: usize) -> Result<Self> {
        let ctx = CudaContext::new(ordinal)
            .with_context(|| format!("init CUDA context on device {}", ordinal))?;
        Ok(Self { ctx, ordinal })
    }

    /// The device ordinal this context drives.
    pub fn ordinal(&self) -> usize {
        self.ordinal
    }

    /// Underlying cudarc context — escape hatch for kernels that need
    /// the raw driver handle. Do not use for long-lived storage;
    /// prefer `Arc<DeviceContext>`.
    pub fn raw(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    /// Block the host until every op queued so far has completed.
    /// Use sparingly — this serializes the pipeline. Mostly for
    /// bench wall-time measurement and for draining before tear-down.
    pub fn synchronize(&self) -> Result<()> {
        self.ctx
            .default_stream()
            .synchronize()
            .context("synchronize default stream")?;
        Ok(())
    }
}

impl std::fmt::Debug for DeviceContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceContext")
            .field("ordinal", &self.ordinal)
            .finish()
    }
}
