//! DFlash block-diffusion draft model for Qwen3.5-27B speculative decoding.
//!
//! # What this is
//!
//! A 5-layer Qwen-shaped transformer whose only job is to propose 16
//! tokens at a time for the full 27B target to verify in a single
//! batched forward. The draft is **not** a standalone language model:
//! it consumes a concatenation of the target's last five hidden
//! states (`target_hidden_cat [5*5120, ctx_len]`) plus a fresh noise
//! embedding sequence (`noise_embed [5120, 16]`), fuses them via a
//! per-call `target_feat = rms_norm(fc @ target_hidden_cat)` (where
//! `fc [5120, 25600]` is the feature un-packer), then runs non-causal
//! attention with K/V drawn from **both** the fused target features
//! and the in-step noise sequence. The output is projected through
//! the target's own `lm_head` — shared, not owned here — to emit
//! 16 vocab logits in a single step.
//!
//! Porting source: `dflash-ref/dflash/src/qwen3_dflash_graph.cpp`
//! `build_draft_graph(...)`. Safetensors layout mirrors the
//! reference's `DraftWeights` (58 tensors total, all bf16).
//!
//! # Why this enables speculative decoding
//!
//! Target Qwen3.5-27B at Q4_K_M is ~16 GB of weights. At A6000's
//! 768 GB/s memory bandwidth, a naive single-token decode is hard
//! capped at ~48 tok/s (weights-only, before KV). Our measured
//! bare-metal single-token decode is 52 tok/s at 1024 context —
//! essentially at the memory wall. The FFI reference sustains
//! 100 tok/s because each target forward verifies ~16 draft tokens
//! at once, amortizing the 16 GB weight scan across ~14 committed
//! tokens. That is the entire speed gap.
//!
//! This module implements **the draft side only**. The chain/DDTree
//! verify loop that binds draft + target lives in
//! `target::SpeculativeDecoder` (next commit); the FA layer's
//! existing batched forward path already accepts `n_tokens > 1`
//! which is what verify needs.
//!
//! # Status
//!
//! This is the first landing of the draft path. The current commit
//! ships:
//!   * `DraftConfig` constants matching `DFLASH27B_*`
//!   * `DraftWeights` owned bf16 device tensors
//!   * `DraftWeights::load_safetensors` — a zero-copy-from-mmap loader
//!     for the 58-tensor `.safetensors` the reference ships
//!   * `DraftModel` wrapping the weights + a scratch allocator for
//!     the forward
//!
//! A minimal `DraftModel::forward` stub returns `not-implemented` for
//! now; the per-layer op sequencing lands alongside the verify loop
//! so every commit is end-to-end runnable.

#![cfg(feature = "cuda")]

use std::path::Path;
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use half::bf16;
use memmap2::Mmap;
use safetensors::SafeTensors;

use ctox_cuda_primitives::device::DeviceContext;
use ctox_cuda_primitives::tensor::CudaTensor;

/// All shape constants of the shipping draft match the target's
/// full-attention-layer dimensions. They are **fixed** for the
/// Qwen3.5-27B draft — the reference hard-codes them under
/// `DFLASH27B_DRAFT_*` and `DFLASH27B_TARGET_*` constants, so we
/// mirror that rather than deriving from a JSON config (the
/// shipped `.safetensors` has no JSON sidecar).
#[derive(Debug, Clone, Copy)]
pub struct DraftConfig {
    /// Hidden dimension. Matches the target's hidden so the
    /// draft's output can feed the target's `lm_head` directly.
    pub hidden: usize,
    /// Q projection dim = `n_head * head_dim`.
    pub q_dim: usize,
    /// K/V projection dim = `n_kv_heads * head_dim`.
    pub kv_dim: usize,
    /// SwiGLU intermediate dim.
    pub intermediate: usize,
    /// Number of attention heads.
    pub n_head: usize,
    /// Number of KV heads (GQA — `n_head % n_kv_heads == 0`).
    pub n_kv_heads: usize,
    /// Per-head dimension.
    pub head_dim: usize,
    /// Number of decoder layers.
    pub n_layers: usize,
    /// Block-diffusion proposal length — the draft predicts this many
    /// tokens per forward. Matches `DFLASH27B_DRAFT_BLOCK_SIZE`.
    pub block_size: usize,
    /// Number of target hidden layers that get concatenated into
    /// `target_hidden_cat` as the draft's cross-attention feature
    /// source. Matches `DFLASH27B_DRAFT_HIDDEN_LAYERS`.
    pub target_hidden_layers: usize,
    /// RMSNorm epsilon (shared with target).
    pub rms_eps: f32,
    /// RoPE base (theta). The reference uses `10_000_000.0` for
    /// Qwen3.5; RoPE mode is NEOX-style.
    pub rope_base: f32,
}

impl DraftConfig {
    /// Canonical config for the shipping Qwen3.5-27B draft.
    pub const fn qwen35_27b() -> Self {
        Self {
            hidden: 5120,
            q_dim: 4096,
            kv_dim: 1024,
            intermediate: 17408,
            n_head: 32,
            n_kv_heads: 8,
            head_dim: 128,
            n_layers: 5,
            block_size: 16,
            target_hidden_layers: 5,
            rms_eps: 1e-6,
            rope_base: 10_000_000.0,
        }
    }

    /// GQA group factor (Q heads per KV head). Used by the flash-attn
    /// launcher.
    #[inline]
    pub fn gqa_group(&self) -> usize {
        self.n_head / self.n_kv_heads
    }
}

/// Owned per-layer weight tensors. All tensors are bf16 on device;
/// the draft is not quantized (the target is, but the draft is small
/// enough — 3.3 GB at bf16 — that Q4_K would add decoding cost without
/// much VRAM savings).
pub struct DraftLayer {
    /// Pre-attention RMSNorm gain. Shape `[hidden]`.
    pub attn_norm: CudaTensor<bf16>,
    /// Q projection. Shape `[hidden, q_dim]` (row-major; the matmul
    /// expects input `[n_tokens, hidden]` × `[hidden, q_dim]`).
    pub w_q: CudaTensor<bf16>,
    /// K projection. Shape `[hidden, kv_dim]`.
    pub w_k: CudaTensor<bf16>,
    /// V projection. Shape `[hidden, kv_dim]`.
    pub w_v: CudaTensor<bf16>,
    /// O projection. Shape `[q_dim, hidden]`.
    pub w_o: CudaTensor<bf16>,
    /// Per-head Q RMSNorm gain. Shape `[head_dim]`.
    pub q_norm: CudaTensor<bf16>,
    /// Per-head K RMSNorm gain. Shape `[head_dim]`.
    pub k_norm: CudaTensor<bf16>,
    /// Pre-FFN RMSNorm gain. Shape `[hidden]`.
    pub ffn_norm: CudaTensor<bf16>,
    /// Gate projection (SwiGLU). Shape `[hidden, intermediate]`.
    pub w_gate: CudaTensor<bf16>,
    /// Up projection. Shape `[hidden, intermediate]`.
    pub w_up: CudaTensor<bf16>,
    /// Down projection. Shape `[intermediate, hidden]`.
    pub w_down: CudaTensor<bf16>,
}

/// Owned top-level draft weights (layers + shared feature fusion).
pub struct DraftWeights {
    pub config: DraftConfig,
    pub layers: Vec<DraftLayer>,
    /// Feature un-packer. Shape `[5*hidden, hidden]` in row-major (GGML
    /// layout `[hidden_target_cat, hidden]`). Applied as
    /// `target_feat = fc @ target_hidden_cat` → `[hidden, ctx_len]`,
    /// then rms-normalized by `hidden_norm`.
    pub fc: CudaTensor<bf16>,
    /// RMS gain applied to `target_feat` after `fc`. Shape `[hidden]`.
    pub hidden_norm: CudaTensor<bf16>,
    /// Final RMS gain applied to the output hidden state before
    /// handing it to `lm_head`. Shape `[hidden]`.
    pub out_norm: CudaTensor<bf16>,
}

impl DraftWeights {
    /// Load the 58 tensors of the reference draft `.safetensors` into
    /// device memory. `path` points at the single-file model bundle
    /// (`dflash-ref/dflash/models/draft/model.safetensors`).
    ///
    /// The loader is strict: every expected tensor must be present
    /// with an exactly-matching shape. Any mismatch surfaces with a
    /// message identifying the tensor so a mis-downloaded / truncated
    /// bundle fails fast.
    pub fn load_safetensors(
        device: Arc<DeviceContext>,
        path: impl AsRef<Path>,
        config: DraftConfig,
    ) -> Result<Self> {
        let path = path.as_ref();
        let file = std::fs::File::open(path)
            .with_context(|| format!("draft: open {}", path.display()))?;
        // SAFETY: memmap2 is safe for read-only files that are not
        // concurrently truncated. The model file is user-supplied and
        // not modified during inference.
        let mmap = unsafe { Mmap::map(&file) }
            .with_context(|| format!("draft: mmap {}", path.display()))?;
        let tensors = SafeTensors::deserialize(&mmap)
            .with_context(|| format!("draft: parse safetensors {}", path.display()))?;

        // Helper: fetch a named tensor, validate shape, upload as bf16
        // to the device. Accepts `[a, b]` or `[a]` shapes. The
        // safetensors layout is row-major with the FIRST dim as the
        // "rows" axis — matches our CudaTensor convention.
        let load_bf16 = |name: &str, expected: &[usize]| -> Result<CudaTensor<bf16>> {
            let view = tensors
                .tensor(name)
                .with_context(|| format!("draft: missing tensor {name}"))?;
            if view.dtype() != safetensors::Dtype::BF16 {
                return Err(anyhow!(
                    "draft: tensor {name} is {:?}, expected BF16",
                    view.dtype()
                ));
            }
            if view.shape() != expected {
                return Err(anyhow!(
                    "draft: tensor {name} shape {:?} != expected {:?}",
                    view.shape(),
                    expected
                ));
            }
            // view.data() is a &[u8] aligned to the mmap page. Reinterpret
            // as &[bf16] via bytemuck — bf16 is repr(transparent) over u16.
            let raw: &[u8] = view.data();
            let numel: usize = expected.iter().product();
            let bytes_needed = numel * std::mem::size_of::<bf16>();
            if raw.len() != bytes_needed {
                return Err(anyhow!(
                    "draft: tensor {name} bytes {} != numel*2 = {}",
                    raw.len(),
                    bytes_needed
                ));
            }
            let half_slice: &[bf16] = bytemuck::cast_slice(raw);
            CudaTensor::<bf16>::from_host(device.clone(), expected.to_vec(), half_slice)
                .with_context(|| format!("draft: upload {name}"))
        };

        // Top-level shared tensors.
        let fc = load_bf16(
            "fc.weight",
            &[config.hidden, config.target_hidden_layers * config.hidden],
        )?;
        let hidden_norm = load_bf16("hidden_norm.weight", &[config.hidden])?;
        let out_norm = load_bf16("norm.weight", &[config.hidden])?;

        // Per-layer tensors.
        let mut layers: Vec<DraftLayer> = Vec::with_capacity(config.n_layers);
        for il in 0..config.n_layers {
            let p = |suffix: &str| format!("layers.{il}.{suffix}");

            // Attention weights. `q_proj.weight` shape in the bundle is
            // `[q_dim, hidden]` (HF convention: out, in). We keep the
            // loaded shape as-is; per-matmul code transposes on
            // dispatch where needed.
            let w_q = load_bf16(&p("self_attn.q_proj.weight"), &[config.q_dim, config.hidden])?;
            let w_k = load_bf16(
                &p("self_attn.k_proj.weight"),
                &[config.kv_dim, config.hidden],
            )?;
            let w_v = load_bf16(
                &p("self_attn.v_proj.weight"),
                &[config.kv_dim, config.hidden],
            )?;
            let w_o = load_bf16(
                &p("self_attn.o_proj.weight"),
                &[config.hidden, config.q_dim],
            )?;
            let q_norm = load_bf16(&p("self_attn.q_norm.weight"), &[config.head_dim])?;
            let k_norm = load_bf16(&p("self_attn.k_norm.weight"), &[config.head_dim])?;

            // Layer norms.
            let attn_norm = load_bf16(&p("input_layernorm.weight"), &[config.hidden])?;
            let ffn_norm = load_bf16(&p("post_attention_layernorm.weight"), &[config.hidden])?;

            // SwiGLU MLP.
            let w_gate = load_bf16(
                &p("mlp.gate_proj.weight"),
                &[config.intermediate, config.hidden],
            )?;
            let w_up = load_bf16(
                &p("mlp.up_proj.weight"),
                &[config.intermediate, config.hidden],
            )?;
            let w_down = load_bf16(
                &p("mlp.down_proj.weight"),
                &[config.hidden, config.intermediate],
            )?;

            layers.push(DraftLayer {
                attn_norm,
                w_q,
                w_k,
                w_v,
                w_o,
                q_norm,
                k_norm,
                ffn_norm,
                w_gate,
                w_up,
                w_down,
            });
        }

        tracing::info!(
            layers = config.n_layers,
            hidden = config.hidden,
            q_heads = config.n_head,
            kv_heads = config.n_kv_heads,
            block_size = config.block_size,
            "draft model loaded from {}",
            path.display()
        );

        Ok(DraftWeights {
            config,
            layers,
            fc,
            hidden_norm,
            out_norm,
        })
    }
}

/// Draft model handle — owns the weights and will eventually hold
/// per-forward scratch buffers (sized for `block_size × hidden` Q
/// and `ctx_len × hidden` cross-features).
pub struct DraftModel {
    pub weights: DraftWeights,
    pub device: Arc<DeviceContext>,
}

impl DraftModel {
    /// Load the draft bundle from a `.safetensors` path.
    pub fn load_from_safetensors(
        device: Arc<DeviceContext>,
        path: impl AsRef<Path>,
        config: DraftConfig,
    ) -> Result<Self> {
        let weights = DraftWeights::load_safetensors(device.clone(), path, config)?;
        Ok(Self { weights, device })
    }

    /// Config accessor.
    #[inline]
    pub fn config(&self) -> &DraftConfig {
        &self.weights.config
    }

    /// Draft forward — propose `block_size` tokens given the target's
    /// recent hidden-state history.
    ///
    /// # Inputs
    /// * `noise_embed` — `[block_size, hidden]` bf16. Seed sequence;
    ///   the reference embeds `[last_commit_tok, MASK, MASK, ..., MASK]`
    ///   through the target's `tok_embd` to produce this.
    /// * `target_hidden_cat` — `[ctx_len, target_hidden_layers * hidden]`
    ///   bf16. The last `target_hidden_layers` hidden states of the
    ///   target, concatenated along the feature axis per position.
    /// * `positions_q` — `[block_size]` i32.
    /// * `positions_k` — `[ctx_len + block_size]` i32.
    /// * `lm_head` — the target's LM head `[hidden, vocab]` bf16 (or
    ///   quantized — handed through the `PackedWeight` dispatch).
    ///
    /// # Output
    /// Logits of shape `[block_size, vocab]` on device.
    ///
    /// The per-layer op sequence is identical to
    /// `dflash-ref::build_draft_graph`. This scaffold lands the type
    /// signature; the body follows in the chain-verify commit (keeps
    /// each commit end-to-end runnable and testable).
    pub fn forward(
        &self,
        _noise_embed: &CudaTensor<bf16>,
        _target_hidden_cat: &CudaTensor<bf16>,
        _positions_q: &CudaTensor<i32>,
        _positions_k: &CudaTensor<i32>,
        _lm_head: &crate::layers::packed_weight::PackedWeight,
    ) -> Result<CudaTensor<f32>> {
        Err(anyhow!(
            "DraftModel::forward: body not yet wired — lands with the chain-verify commit"
        ))
    }
}

#[cfg(test)]
mod tests {
    //! Integration tests — only meaningful with the real draft bundle
    //! on disk. Gated behind an env var so `cargo test` stays green on
    //! machines without the fixture.

    use super::*;

    const DRAFT_PATH_ENV: &str = "CTOX_QWEN35_DRAFT_SAFETENSORS";

    fn draft_path() -> Option<std::path::PathBuf> {
        std::env::var_os(DRAFT_PATH_ENV).map(std::path::PathBuf::from)
    }

    #[test]
    #[ignore]
    fn draft_loads_clean() {
        let Some(path) = draft_path() else {
            eprintln!(
                "skipping: set {DRAFT_PATH_ENV} to the reference draft model.safetensors to run this test"
            );
            return;
        };

        let dev =
            Arc::new(DeviceContext::new(0).expect("cuda init for draft_loads_clean"));
        let cfg = DraftConfig::qwen35_27b();
        let model = DraftModel::load_from_safetensors(dev, &path, cfg)
            .expect("load draft safetensors");

        assert_eq!(model.weights.layers.len(), cfg.n_layers);
        assert_eq!(model.weights.fc.shape(), &[cfg.hidden, cfg.target_hidden_layers * cfg.hidden]);
        assert_eq!(model.weights.hidden_norm.shape(), &[cfg.hidden]);
        assert_eq!(model.weights.out_norm.shape(), &[cfg.hidden]);
        for (il, layer) in model.weights.layers.iter().enumerate() {
            assert_eq!(layer.attn_norm.shape(), &[cfg.hidden], "layer {il}");
            assert_eq!(layer.w_q.shape(), &[cfg.q_dim, cfg.hidden], "layer {il}");
            assert_eq!(layer.w_k.shape(), &[cfg.kv_dim, cfg.hidden], "layer {il}");
            assert_eq!(layer.w_v.shape(), &[cfg.kv_dim, cfg.hidden], "layer {il}");
            assert_eq!(layer.w_o.shape(), &[cfg.hidden, cfg.q_dim], "layer {il}");
            assert_eq!(layer.q_norm.shape(), &[cfg.head_dim], "layer {il}");
            assert_eq!(layer.k_norm.shape(), &[cfg.head_dim], "layer {il}");
            assert_eq!(layer.ffn_norm.shape(), &[cfg.hidden], "layer {il}");
            assert_eq!(layer.w_gate.shape(), &[cfg.intermediate, cfg.hidden], "layer {il}");
            assert_eq!(layer.w_up.shape(), &[cfg.intermediate, cfg.hidden], "layer {il}");
            assert_eq!(layer.w_down.shape(), &[cfg.hidden, cfg.intermediate], "layer {il}");
        }

        eprintln!(
            "draft: loaded {} layers, hidden={}, block_size={}",
            cfg.n_layers, cfg.hidden, cfg.block_size
        );
    }
}
