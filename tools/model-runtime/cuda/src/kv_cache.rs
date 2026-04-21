//! KV cache — ring buffer of per-layer K/V slabs.
//!
//! Stored as raw device allocations (one `CudaSlice<bf16>` per slab)
//! with explicit position tracking. No candle `Tensor::cat` growth,
//! no copy-on-append — new tokens are written into slot `[n_filled]`
//! and `n_filled` advances.
//!
//! This is a skeleton: the append / read kernel paths land with the
//! first model port (Qwen3.5 attention). The struct here defines
//! the memory layout contract.

use std::sync::Arc;

use anyhow::Result;
use cudarc::driver::CudaSlice;
use half::bf16;

use crate::device::DeviceContext;

/// Per-layer K/V slabs.
///
/// Layout (per layer, K or V separately):
///   `[max_ctx × n_kv_heads × head_dim]` bf16, row-major along max_ctx.
///
/// Writes to token position `p` land at offset
///   `p × (n_kv_heads × head_dim) × sizeof(bf16)`.
///
/// `n_filled` tracks how many prefix positions hold valid content.
/// Attention kernels consume `buf[0..n_filled × slot_bytes]`.
///
/// Rolling / eviction (for >max_ctx sequences) is NOT handled here —
/// the caller (model forward code) is responsible for deciding when
/// to wrap and telling the attention kernel about the boundary. The
/// cache just owns storage.
pub struct KvCache {
    k_slabs: Vec<CudaSlice<bf16>>,
    v_slabs: Vec<CudaSlice<bf16>>,
    n_layers: usize,
    max_ctx: usize,
    n_kv_heads: usize,
    head_dim: usize,
    n_filled: usize,
    device: Arc<DeviceContext>,
}

impl KvCache {
    /// Allocate a fresh KV cache.
    pub fn new(
        device: Arc<DeviceContext>,
        n_layers: usize,
        max_ctx: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> Result<Self> {
        let slot_elems = n_kv_heads * head_dim;
        let slab_elems = max_ctx * slot_elems;
        let stream = device.raw().default_stream();

        let mut k_slabs = Vec::with_capacity(n_layers);
        let mut v_slabs = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            k_slabs.push(stream.alloc_zeros::<bf16>(slab_elems)?);
            v_slabs.push(stream.alloc_zeros::<bf16>(slab_elems)?);
        }
        Ok(Self {
            k_slabs,
            v_slabs,
            n_layers,
            max_ctx,
            n_kv_heads,
            head_dim,
            n_filled: 0,
            device,
        })
    }

    /// Number of layers this cache serves.
    pub fn n_layers(&self) -> usize {
        self.n_layers
    }

    /// Maximum context window in tokens.
    pub fn max_ctx(&self) -> usize {
        self.max_ctx
    }

    /// Number of K/V heads per layer.
    pub fn n_kv_heads(&self) -> usize {
        self.n_kv_heads
    }

    /// Dim of each head.
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// How many prefix positions hold valid content.
    pub fn n_filled(&self) -> usize {
        self.n_filled
    }

    /// Slot size in elements (`n_kv_heads × head_dim`).
    pub fn slot_elems(&self) -> usize {
        self.n_kv_heads * self.head_dim
    }

    /// Get the raw K slab for a layer. Kernels consume this plus
    /// `n_filled` to know how much is valid.
    pub fn k_slab(&self, layer: usize) -> &CudaSlice<bf16> {
        &self.k_slabs[layer]
    }

    /// Get the mutable K slab for a layer — for the attention write.
    pub fn k_slab_mut(&mut self, layer: usize) -> &mut CudaSlice<bf16> {
        &mut self.k_slabs[layer]
    }

    /// Mirror of `k_slab` / `k_slab_mut` for V.
    pub fn v_slab(&self, layer: usize) -> &CudaSlice<bf16> {
        &self.v_slabs[layer]
    }

    pub fn v_slab_mut(&mut self, layer: usize) -> &mut CudaSlice<bf16> {
        &mut self.v_slabs[layer]
    }

    /// Advance the fill counter by `n` tokens. Caller is responsible
    /// for having written those slots across all layers before
    /// calling this.
    pub fn advance(&mut self, n: usize) {
        debug_assert!(self.n_filled + n <= self.max_ctx, "KvCache overflow");
        self.n_filled += n;
    }

    /// Reset fill to zero. Does NOT zero the underlying buffers —
    /// attention kernels read only up to `n_filled`, so stale content
    /// in slots [n_filled..max_ctx] is never seen.
    pub fn reset(&mut self) {
        self.n_filled = 0;
    }

    pub fn device(&self) -> &Arc<DeviceContext> {
        &self.device
    }
}

impl std::fmt::Debug for KvCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KvCache")
            .field("n_layers", &self.n_layers)
            .field("max_ctx", &self.max_ctx)
            .field("n_kv_heads", &self.n_kv_heads)
            .field("head_dim", &self.head_dim)
            .field("n_filled", &self.n_filled)
            .finish()
    }
}
