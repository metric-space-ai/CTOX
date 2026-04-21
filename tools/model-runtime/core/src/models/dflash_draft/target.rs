//! Target-side abstraction for DFlash speculative decoding.
//!
//! The target in a DFlash pipeline is always a hybrid Qwen3.5 (dense
//! attention every 4 layers, Gated DeltaNet in between, 64 layers for
//! the 27B variant). But from the stepper's point of view (commit
//! that follows this one), the target only needs to expose three
//! capabilities:
//!
//!   1. run a forward of `input_ids` with feature capture at the
//!      configured layer indices,
//!   2. expose the token embedding, for the draft to use,
//!   3. expose the lm_head, for the draft to use.
//!
//! Nothing else — the KV cache, paged-attention metadata, device
//! mapping, all stay encapsulated inside the target. That makes the
//! stepper trivially mockable (swap in a hand-built target that
//! returns canned logits + capture tensors for unit tests) and lets
//! the concrete scheduler wiring happen in its own commit without
//! this one growing an extra 300 lines of plumbing.

use candle_core::{Result, Tensor};
use candle_nn::Embedding;

use super::capture::FeatureCapture;
use crate::kv_cache::RecurrentStateSnapshot;

/// Trait the DFlash stepper consumes. Implementations must be
/// `Send + Sync` for integration into the async pipeline loop; our
/// concrete Qwen3.5 implementation satisfies this via its existing
/// `Arc<tokio::sync::Mutex<dyn Pipeline>>` wrapper.
pub trait DFlashTargetForward: Send + Sync {
    /// Run one target forward pass with the given `input_ids`, asking
    /// the text model to snapshot its post-layer hidden states at
    /// `capture.layer_ids`.
    ///
    /// Returns logits of shape `[batch=1, seq, vocab]`. The stepper
    /// consumes only the last `block_size` of those to compare with
    /// the draft's candidates.
    ///
    /// `past_kv_len` is the number of tokens the target has already
    /// processed (i.e. the position of the first element of
    /// `input_ids`). Implementations use this to set RoPE offsets and
    /// paged-attention slot mapping correctly.
    ///
    /// `capture` is reset by the implementation at the start of the
    /// call (same semantics as
    /// [`crate::vision_models::qwen3_5::Qwen3_5TextModel::forward_embeds_with_capture`]).
    fn forward_with_capture(
        &self,
        input_ids: &Tensor,
        past_kv_len: usize,
        capture: &mut FeatureCapture,
    ) -> Result<Tensor>;

    /// Like [`Self::forward_with_capture`], but with an explicit
    /// attention mask instead of an implicit causal one. Used by
    /// DDTree tree verify where `input_ids` is a DFS-flattened tree
    /// of speculated tokens and the mask encodes ancestor-only
    /// visibility per tree node.
    ///
    /// `attention_mask` must have shape `[1, 1, seq_len, past_kv_len +
    /// seq_len]` and the target's dtype, with `0.0` on allowed
    /// positions and `-inf` elsewhere. See
    /// [`crate::models::dflash_draft::build_tree_mask`] for the mask
    /// builder that matches this contract.
    ///
    /// Default implementation falls back to
    /// [`Self::forward_with_capture`] so chain-mode targets (tests,
    /// mock targets) stay functional without having to reimplement
    /// the masked path. Concrete targets that need DDTree must
    /// override this.
    fn forward_with_capture_masked(
        &self,
        input_ids: &Tensor,
        past_kv_len: usize,
        _attention_mask: &Tensor,
        capture: &mut FeatureCapture,
    ) -> Result<Tensor> {
        // Fallback: ignore the mask. Fine for chain-only callers; the
        // real Qwen35 target overrides this.
        self.forward_with_capture(input_ids, past_kv_len, capture)
    }

    /// Return a reference to the target's token embedding layer.
    /// Shared with the draft for input embedding (the draft has no
    /// embedding of its own).
    fn embed_tokens(&self) -> &Embedding;

    /// Project `hidden` through the target's `lm_head`.
    ///
    /// Modelled as a method instead of an accessor because the target's
    /// live lm_head is typically an `Arc<dyn QuantMethod>` (Q4_K_M after
    /// ISQ), not a plain `Linear` — there's no uniform way to hand out
    /// a borrow that works for both. Input shape `[..., hidden_size]`,
    /// output `[..., vocab_size]`.
    fn apply_lm_head(&self, hidden: &Tensor) -> Result<Tensor>;

    /// Snapshot the target's hybrid recurrent state for rollback.
    ///
    /// The stepper calls this *before* the verify forward so that, once
    /// the accept loop picks the committed-tokens prefix, it can undo
    /// the Gated-DeltaNet state advance the verify forward caused and
    /// replay only the accepted tokens through a second "commit"
    /// forward. Without this rollback, rejected-draft KV/recurrent
    /// state pollutes the cache and the next step conditions on
    /// garbage — producing incoherent output in practice.
    ///
    /// See the reference graph's comment for the same invariant:
    /// > "Restore SSM state. Replay the accepted tokens through
    /// >  target (batched cleanly advanced only by what was committed)."
    fn snapshot_recurrent_state(&self) -> Result<Vec<RecurrentStateSnapshot>>;

    /// Restore a previously-taken recurrent-state snapshot.
    fn restore_recurrent_state(&self, snapshots: &[RecurrentStateSnapshot]) -> Result<()>;

    /// Truncate every full-attention layer's KV cache to `len`. Paired
    /// with [`Self::restore_recurrent_state`] in the stepper's verify
    /// rollback.
    fn truncate_attention_to(&self, len: usize) -> Result<()>;
}
