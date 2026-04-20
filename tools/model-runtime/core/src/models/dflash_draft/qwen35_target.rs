//! Concrete `DFlashTargetForward` implementation for the hybrid
//! Qwen3.5 text model (Qwen3.5-27B Gated-DeltaNet target, 64 layers,
//! same arch the reference DFlash implementation targets).
//!
//! Two of the three trait methods are thin accessors — [`embed_tokens`]
//! and [`apply_lm_head`] forward straight to
//! [`crate::vision_models::qwen3_5::Qwen3_5TextModel::embed_tokens_layer`]
//! and
//! [`crate::vision_models::qwen3_5::Qwen3_5TextModel::apply_lm_head`].
//!
//! The third, `forward_with_capture`, delegates to
//! [`crate::vision_models::qwen3_5::Qwen3_5TextModel::forward_with_dflash_capture`]
//! — a sibling helper on the text model that assembles MRoPE
//! position_ids, causal attention mask, and `FlashParams` for an
//! arbitrary-length `input_ids` slice at a given `past_kv_len`.
//!
//! Keeping the metadata assembly on the text model (rather than in
//! this trait impl) means the logic lives next to the regular
//! inference path it mirrors — if the engine's input processor grows
//! a new field or the rotary embedding changes, the DFlash helper
//! sits right there and is easy to keep in sync.

use std::sync::Arc;

use candle_core::{Result, Tensor};
use candle_nn::Embedding;

use super::capture::FeatureCapture;
use super::target::DFlashTargetForward;
use crate::vision_models::qwen3_5::Qwen3_5TextModel;

/// Owns a shared handle to the Qwen3.5 text model. Created alongside
/// the target pipeline so target and stepper see exactly the same
/// weights and KV cache.
///
/// `Arc<Qwen3_5TextModel>` keeps the struct cheap to clone when the
/// async pipeline needs the stepper on one thread and the scheduler
/// on another.
pub struct Qwen35DFlashTarget {
    text: Arc<Qwen3_5TextModel>,
}

impl Qwen35DFlashTarget {
    pub fn new(text: Arc<Qwen3_5TextModel>) -> Self {
        Self { text }
    }

    pub fn text(&self) -> &Qwen3_5TextModel {
        &self.text
    }
}

impl DFlashTargetForward for Qwen35DFlashTarget {
    fn forward_with_capture(
        &self,
        input_ids: &Tensor,
        past_kv_len: usize,
        capture: &mut FeatureCapture,
    ) -> Result<Tensor> {
        self.text
            .forward_with_dflash_capture(input_ids, past_kv_len, Some(capture))
    }

    fn embed_tokens(&self) -> &Embedding {
        self.text.embed_tokens_layer()
    }

    fn apply_lm_head(&self, hidden: &Tensor) -> Result<Tensor> {
        self.text.apply_lm_head(hidden)
    }
}
