//! The `Model` and `GenerativeModel` traits — the two abstractions
//! over per-model backends.
//!
//! Each model crate (e.g. `ctox-qwen35-27b`) provides a type that
//! implements at least one of these traits. CTOX's serving loop
//! drives inference only through them — never touching model-specific
//! types like `Qwen35Target`, `CudaTensor`, or any CUDA primitive
//! directly.
//!
//! Design stance: both traits are deliberately small. Streaming,
//! batch composition, and KV-pool orchestration live in
//! [`crate::serving`] (landing when the first two model crates are
//! in place and we can see what's genuinely common vs
//! model-specific).
//!
//! ## Why two traits?
//!
//! * [`Model`] is the canonical **forward-pass** trait. A single call
//!   returns logits for one step of one sequence. This is what the
//!   bare-metal native port exposes (`Qwen35Target::forward`), and
//!   what the streaming sampler on top of it needs.
//!
//! * [`GenerativeModel`] is the **sequence-level** trait. A single
//!   call takes a prompt + budget and returns a decoded token sequence
//!   plus wall-clock stats. This matches backends that do sampling
//!   and (optionally) speculative decoding internally — the DFlash
//!   FFI reference library is the canonical example. Exposing logits
//!   per step is not always possible (the FFI drives the whole
//!   speculative-decode loop in C++ and only hands us the final token
//!   stream), so we model it at the level it actually provides.
//!
//! A backend that implements only `GenerativeModel` is still fully
//! usable from the serving layer — we just lose fine-grained control
//! over sampling until the native port replaces it.

use anyhow::Result;

/// Input to a forward pass — token ids plus any auxiliary state the
/// serving layer tracks per sequence.
#[derive(Debug, Clone)]
pub struct ModelInput<'a> {
    /// Flat sequence of input tokens for the current step. For
    /// chat-style serving this is `[last_committed_token]` after
    /// prefill completes; for prefill it's the whole prompt.
    pub tokens: &'a [i32],

    /// Absolute position of each token in its sequence — used to
    /// drive RoPE / MRoPE. For simple text models this is
    /// `[past_kv_len, past_kv_len + 1, ...]`. Multi-axis variants
    /// (Qwen3.5 MRoPE) require axis 0..2 to hold the text position;
    /// model-specific code fans this out internally.
    pub positions: &'a [i32],

    /// Running KV-cache fill count before this call.
    pub past_kv_len: usize,
}

/// Output of a forward pass — next-token distribution and internal
/// bookkeeping.
#[derive(Debug)]
pub struct ModelOutput {
    /// Log-probabilities (or raw logits, model-specific) over the
    /// vocabulary for the LAST position of the input.
    pub logits: Vec<f32>,

    /// Number of KV-cache slots consumed by this call. Sequence
    /// state layer adds this to `past_kv_len` to advance.
    pub advanced_kv: usize,
}

/// Forward-pass trait for native-kernel backends.
///
/// Intentionally not `async_trait` today — kernel launches are
/// synchronous from the CPU's perspective. The async-ness happens
/// one level up in the serving layer (stream composition across
/// concurrent sequences).
pub trait Model: Send + Sync {
    /// Human-readable identifier — e.g. `"qwen35-27b-q4km"`.
    fn id(&self) -> &'static str;

    /// Vocabulary size — needed by samplers.
    fn vocab_size(&self) -> usize;

    /// Tokenize a string → ids.
    fn encode(&self, text: &str) -> Result<Vec<i32>>;

    /// Detokenize ids → string.
    fn decode(&self, ids: &[i32]) -> Result<String>;

    /// Run one forward pass for a single sequence.
    fn forward(&mut self, input: ModelInput<'_>) -> Result<ModelOutput>;
}

/// Per-call generation statistics — everything a caller needs to
/// report tok/s or measure speculative-decode effectiveness.
///
/// Fields that don't apply to a given backend should be zero:
/// non-speculative backends report `n_draft_steps = 0`, etc.
#[derive(Debug, Clone, Copy, Default)]
pub struct GenerateStats {
    /// Number of tokens actually emitted beyond the prompt.
    pub n_generated: usize,
    /// Number of speculative draft steps run. `0` for non-speculative.
    pub n_draft_steps: usize,
    /// Total tokens accepted across all draft steps. `0` for
    /// non-speculative.
    pub n_accepted: usize,
    /// Total tokens proposed across all draft steps (for an
    /// acceptance-rate computation). `0` for non-speculative.
    pub n_proposed: usize,
    /// Wall-clock seconds spent inside `generate`.
    pub wall_s: f64,
    /// Decode throughput reported by the backend, in tokens per
    /// second. This is usually stricter than `n_generated / wall_s`
    /// because it excludes prefill; backends that don't separate
    /// the two should set it to the same ratio.
    pub decode_tok_s: f64,
    /// Last token id emitted — handy for chat continuations where
    /// the caller wants to check the stop condition without
    /// re-scanning the full output.
    pub last_tok: i32,
}

/// Sequence-level trait for backends that do sampling (and optionally
/// speculative decoding) internally.
///
/// `&mut self` because every backend in this category holds internal
/// state — KV cache, draft snapshots, CUDA context — that mutates
/// across calls even when the caller thinks of generation as
/// functional. The trait makes no claim about whether state is
/// reset between calls; individual backends document that. (E.g.
/// [`crate::backends::dflash::DflashBackend`] resets state per call
/// because the C API does.)
pub trait GenerativeModel: Send {
    /// Human-readable identifier — e.g. `"qwen35-27b-dflash"`.
    fn id(&self) -> &str;

    /// Vocabulary size — needed by samplers / assertion checks.
    fn vocab_size(&self) -> usize;

    /// Tokenize a string → ids.
    fn encode(&self, text: &str) -> Result<Vec<i32>>;

    /// Detokenize ids → string.
    fn decode(&self, ids: &[i32]) -> Result<String>;

    /// Generate up to `n_new` tokens continuing from `prompt_ids`.
    ///
    /// Returns the full token sequence the backend produced (prompt
    /// + continuation; the backend may trim at EOS before reaching
    ///   `n_new`) plus timing stats.
    fn generate(
        &mut self,
        prompt_ids: &[i32],
        n_new: usize,
    ) -> Result<(Vec<i32>, GenerateStats)>;
}
