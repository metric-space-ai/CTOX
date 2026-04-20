//! Config deserialisation for `z-lab/Qwen3.5-27B-DFlash` and future
//! DFlash checkpoints. Matches the HuggingFace `config.json` exactly;
//! unknown fields are tolerated so newer checkpoints load without a
//! code change.
//!
//! Reference config (Qwen3.5-27B-DFlash):
//! ```json
//! {
//!   "architectures": ["DFlashDraftModel"],
//!   "block_size": 16,
//!   "dflash_config": {
//!     "mask_token_id": 248070,
//!     "target_layer_ids": [1, 16, 31, 46, 61]
//!   },
//!   "head_dim": 128,
//!   "hidden_size": 5120,
//!   "intermediate_size": 17408,
//!   "num_attention_heads": 32,
//!   "num_hidden_layers": 5,
//!   "num_key_value_heads": 8,
//!   "num_target_layers": 64,
//!   "rms_norm_eps": 1e-06,
//!   "rope_theta": 10000000,
//!   "vocab_size": 248320,
//!   ...
//! }
//! ```

use serde::Deserialize;

/// Nested `dflash_config` sub-object — DFlash-specific parameters that
/// aren't standard Qwen3 config.
#[derive(Deserialize, Debug, Clone)]
pub struct DFlashSubConfig {
    /// Vocabulary id that the draft interprets as the block-diffusion
    /// mask. Positions 1..block_size of `input_ids` are filled with
    /// this id during each inference step; the draft denoises them in
    /// one forward pass. For Qwen3.5-27B-DFlash this is 248070.
    pub mask_token_id: u32,

    /// Indices of the target model's transformer layers whose post-
    /// attention hidden states are captured and fed into the draft as
    /// cross-attention KV. For Qwen3.5-27B the canonical set is
    /// `[1, 16, 31, 46, 61]` — early, quarter, half, three-quarter, and
    /// pre-final representations of the target.
    ///
    /// The length of this vector is the `target_stack_count` that
    /// multiplies `hidden_size` in the draft's `fc` projection
    /// (`[target_stack_count * hidden_size -> hidden_size]`).
    pub target_layer_ids: Vec<usize>,
}

/// Full DFlash draft configuration. Field names match the HF
/// `config.json`; non-DFlash fields (tie_word_embeddings, attention
/// flags, rope_scaling, etc.) are intentionally absent here — the
/// draft has a fixed architecture and those have no effect on the
/// forward.
#[derive(Deserialize, Debug, Clone)]
pub struct DFlashDraftConfig {
    /// Number of masked positions produced per forward pass (a.k.a.
    /// "block" in the paper). For Qwen3.5-27B-DFlash this is 16; the
    /// draft always emits exactly 16 candidate tokens per call.
    pub block_size: usize,

    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,

    /// Number of layers in the *target* model, recorded here for
    /// sanity-checking the feature capture hook in the pipeline.
    pub num_target_layers: usize,

    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub vocab_size: usize,

    #[serde(rename = "dflash_config")]
    pub dflash: DFlashSubConfig,
}

impl DFlashDraftConfig {
    /// Size of the `fc` projection's input side. Equals
    /// `target_layer_ids.len() * hidden_size`.
    pub fn fused_target_feature_dim(&self) -> usize {
        self.dflash.target_layer_ids.len() * self.hidden_size
    }

    pub fn q_dim(&self) -> usize {
        self.num_attention_heads * self.head_dim
    }

    pub fn kv_dim(&self) -> usize {
        self.num_key_value_heads * self.head_dim
    }

    /// Basic consistency check — catches checkpoint/config mismatch
    /// early rather than at a matmul shape failure deep inside forward.
    pub fn validate(&self) -> Result<(), String> {
        if self.num_hidden_layers == 0 {
            return Err(format!(
                "DFlashDraftConfig: num_hidden_layers={} must be > 0",
                self.num_hidden_layers
            ));
        }
        if self.head_dim * self.num_attention_heads == 0 {
            return Err(format!(
                "DFlashDraftConfig: zero head_dim ({}) or num_attention_heads ({})",
                self.head_dim, self.num_attention_heads
            ));
        }
        if self.dflash.target_layer_ids.is_empty() {
            return Err("DFlashDraftConfig: dflash_config.target_layer_ids is empty".into());
        }
        if self.block_size == 0 {
            return Err("DFlashDraftConfig: block_size must be > 0".into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_qwen35_27b_dflash_config() {
        // Subset of the real config.json; confirms the serde mapping
        // accepts the canonical checkpoint.
        let json = r#"{
            "architectures": ["DFlashDraftModel"],
            "block_size": 16,
            "dflash_config": {
                "mask_token_id": 248070,
                "target_layer_ids": [1, 16, 31, 46, 61]
            },
            "head_dim": 128,
            "hidden_size": 5120,
            "intermediate_size": 17408,
            "max_position_embeddings": 262144,
            "num_attention_heads": 32,
            "num_hidden_layers": 5,
            "num_key_value_heads": 8,
            "num_target_layers": 64,
            "rms_norm_eps": 1e-06,
            "rope_theta": 10000000,
            "vocab_size": 248320
        }"#;
        let cfg: DFlashDraftConfig = serde_json::from_str(json).unwrap();
        cfg.validate().unwrap();
        assert_eq!(cfg.block_size, 16);
        assert_eq!(cfg.dflash.mask_token_id, 248070);
        assert_eq!(cfg.dflash.target_layer_ids, vec![1, 16, 31, 46, 61]);
        assert_eq!(cfg.fused_target_feature_dim(), 5 * 5120);
        assert_eq!(cfg.q_dim(), 32 * 128);
        assert_eq!(cfg.kv_dim(), 8 * 128);
    }

    #[test]
    fn validate_rejects_empty_target_layers() {
        let mut cfg = DFlashDraftConfig {
            block_size: 16,
            hidden_size: 5120,
            intermediate_size: 17408,
            num_attention_heads: 32,
            num_hidden_layers: 5,
            num_key_value_heads: 8,
            head_dim: 128,
            num_target_layers: 64,
            max_position_embeddings: 262144,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000_000.0,
            vocab_size: 248320,
            dflash: DFlashSubConfig {
                mask_token_id: 248070,
                target_layer_ids: vec![],
            },
        };
        assert!(cfg.validate().is_err());
        cfg.dflash.target_layer_ids = vec![1];
        assert!(cfg.validate().is_ok());
    }
}
