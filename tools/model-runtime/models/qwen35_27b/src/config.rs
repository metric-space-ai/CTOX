//! Qwen3.5 architectural constants.
//!
//! Two construction paths:
//!   * [`Qwen35Config::QWEN35_27B`] / [`Qwen35Config::qwen35_27b`] —
//!     hardcoded constant matching the shipping 27B build. Used by
//!     smoke tests that run on synthetic weights without opening a
//!     GGUF file.
//!   * [`Qwen35Config::from_metadata`] — factory that pulls the actual
//!     dimensions out of a GGUF's `qwen35.*` metadata section (see
//!     [`crate::gguf_loader::parse_qwen35_metadata`]). Prefer this path when
//!     loading real weights so the config stays in lockstep with the
//!     file — we hit a real mismatch once already, where the const said
//!     `n_head=40 / n_head_kv=8 / head_dim=128` but the shipping 27B
//!     GGUF reports `n_head=24 / n_head_kv=4 / head_dim=256`.
//!
//! Qwen3.5 hybrid's GDN layers use a **different** head-count mapping
//! than its FullAttention layers:
//!
//!   * FA: `n_q_heads = 24`, `n_kv_heads = 4`, `head_dim = 256`
//!   * GDN: `num_v_heads = 48`, `num_k_heads = 16`,
//!          `head_k_dim = head_v_dim = gdn_ssm_dim = 128`
//!
//! The GDN layer's fused `attn_qkv.weight` is
//! `[hidden, num_k_heads * head_k_dim * 2 + num_v_heads * head_v_dim]
//!  = [5120, 10240]` (Q||K||V layout). `ssm_out.weight` is
//! `[num_v_heads * head_v_dim, hidden] = [6144, 5120]`.

use crate::gguf_loader::Qwen35Metadata;

/// Static layer-shape parameters for Qwen3.5 models.
#[derive(Debug, Clone, Copy)]
pub struct Qwen35Config {
    /// Residual-stream width.
    pub hidden_dim: usize,
    /// Number of query heads per FullAttention layer.
    pub n_q_heads: usize,
    /// Number of key/value heads per FullAttention layer. GQA ratio =
    /// `n_q_heads / n_kv_heads`.
    pub n_kv_heads: usize,
    /// Size of each attention head along the feature axis (FA layers).
    pub head_dim: usize,
    /// SSM recurrent-state width (`S_v` in the GDN kernel). Equals the
    /// per-head Q/K/V feature width in the GDN layout — both
    /// `head_k_dim` and `head_v_dim` collapse to this value on 27B.
    pub gdn_ssm_dim: usize,
    /// Number of GDN value heads (`SSM_DT_RANK` / `num_v_heads` in the
    /// dflash reference). This is the kernel's `H` template parameter.
    /// Shipping 27B GGUF: 48 (derived from
    /// `qwen35.ssm.inner_size / qwen35.ssm.state_size = 6144 / 128`,
    /// also stored directly as `qwen35.ssm.time_step_rank`).
    pub gdn_num_v_heads: usize,
    /// Number of GDN key heads (`SSM_N_GROUP` / `num_k_heads` in the
    /// reference). GQA broadcast factor from K heads up to V heads is
    /// `num_v_heads / num_k_heads = 48 / 16 = 3` on 27B. Shipping 27B
    /// GGUF: 16 (stored as `qwen35.ssm.group_count`).
    pub gdn_num_k_heads: usize,
    /// Inner (SwiGLU-intermediate) width of the FFN block. Each
    /// Qwen3.5 layer's FFN maps `hidden_dim → intermediate_dim` via
    /// two projections (gate, up), applies `silu(gate) * up`, then
    /// projects back `intermediate_dim → hidden_dim` via `down`.
    /// Shipping 27B GGUF: 17408 (via `qwen35.feed_forward_length`).
    pub intermediate_dim: usize,
    /// RoPE base. Shipping 27B GGUF: 10_000_000.
    pub rope_theta: f32,
    /// RMSNorm epsilon.
    pub rms_eps: f32,
    /// Upper bound on sequence length / KV cache ring size.
    pub max_position_embeddings: usize,
}

impl Qwen35Config {
    /// Qwen3.5 — 27B hybrid.
    ///
    /// Values match the shipping `Qwen3.5-27B-Q4_K_M.gguf` inspected
    /// via `parse_qwen35_metadata`. Cross-checked against dflash's
    /// `gguf_target_loader.cpp` (n_embd=5120, n_head=24, n_head_kv=4,
    /// kl=vl=256, n_ff=17408, full_attention_interval=4) and
    /// `qwen35_target_graph.cpp` (SSM_DT_RANK=num_v_heads=48,
    /// SSM_N_GROUP=num_k_heads=16, SSM_D_STATE=head_k_dim=head_v_dim=128).
    pub const QWEN35_27B: Self = Self {
        hidden_dim: 5120,
        n_q_heads: 24,
        n_kv_heads: 4,
        head_dim: 256,
        gdn_ssm_dim: 128,
        gdn_num_v_heads: 48,
        gdn_num_k_heads: 16,
        intermediate_dim: 17_408,
        rope_theta: 10_000_000.0,
        rms_eps: 1e-6,
        max_position_embeddings: 131_072,
    };

    /// Convenience factory. Returns the same config as
    /// [`Self::QWEN35_27B`].
    pub fn qwen35_27b() -> Self {
        Self::QWEN35_27B
    }

    /// Build a config from GGUF metadata. Prefer this path over
    /// [`Self::QWEN35_27B`] whenever a GGUF is available — it locks
    /// the layer dimensions to the file rather than to a bake-time
    /// constant that can drift.
    ///
    /// `gdn_ssm_dim` comes from `qwen35.ssm.state_size` on the
    /// reference target (128 on the shipping 27B). The caller passes
    /// it explicitly here because this file doesn't model the GDN
    /// side of the metadata. `gdn_num_v_heads` and `gdn_num_k_heads`
    /// are taken from the parsed metadata (sourced from
    /// `qwen35.ssm.time_step_rank` and `qwen35.ssm.group_count`
    /// respectively) with 48/16 fallbacks for older files.
    pub fn from_metadata(m: &Qwen35Metadata, gdn_ssm_dim: usize) -> Self {
        Self {
            hidden_dim: m.embedding_length,
            n_q_heads: m.head_count,
            n_kv_heads: m.head_count_kv,
            // GGUF distinguishes `key_length` / `value_length`; the
            // shipping 27B has them equal (kl=vl=256) and dflash
            // collapses them to one `HEAD_DIM`. We follow dflash.
            head_dim: m.key_length,
            gdn_ssm_dim,
            gdn_num_v_heads: m.ssm_time_step_rank,
            gdn_num_k_heads: m.ssm_group_count,
            intermediate_dim: m.feed_forward_length,
            rope_theta: m.rope_theta,
            rms_eps: m.rms_eps,
            max_position_embeddings: m.context_length,
        }
    }

    /// `n_q_heads * head_dim` — Q projection output width.
    pub const fn q_dim(&self) -> usize {
        self.n_q_heads * self.head_dim
    }

    /// `n_kv_heads * head_dim` — K/V projection output width.
    pub const fn kv_dim(&self) -> usize {
        self.n_kv_heads * self.head_dim
    }

    /// Number of Q heads per KV head (GQA group size, FullAttention).
    pub const fn gqa_group(&self) -> usize {
        self.n_q_heads / self.n_kv_heads
    }

    /// Fused `attn_qkv.weight` output width for GDN layers:
    ///   `2 * num_k_heads * head_k_dim + num_v_heads * head_v_dim`
    ///
    /// With `head_k_dim == head_v_dim == gdn_ssm_dim`, this is
    /// `gdn_ssm_dim * (2 * num_k_heads + num_v_heads)`. On 27B:
    /// `128 * (32 + 48) = 128 * 80 = 10240`, matching the Q5_K
    /// `[hidden, 10240]` tensor in the shipping GGUF.
    pub const fn gdn_qkv_proj_dim(&self) -> usize {
        2 * self.gdn_num_k_heads * self.gdn_ssm_dim + self.gdn_num_v_heads * self.gdn_ssm_dim
    }

    /// `ssm_out.weight` K dimension: `num_v_heads * head_v_dim`. On
    /// 27B: `48 * 128 = 6144`.
    pub const fn gdn_inner_dim(&self) -> usize {
        self.gdn_num_v_heads * self.gdn_ssm_dim
    }
}
