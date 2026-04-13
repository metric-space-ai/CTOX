#![allow(dead_code)]

use serde::Deserialize;
use std::collections::HashMap;

fn default_qwen3_tts_speaker_mel_dim() -> usize {
    128
}

fn default_qwen3_tts_speaker_enc_dim() -> usize {
    1024
}

fn default_qwen3_tts_speaker_enc_channels() -> Vec<usize> {
    vec![512, 512, 512, 512, 1536]
}

fn default_qwen3_tts_speaker_enc_kernel_sizes() -> Vec<usize> {
    vec![5, 3, 3, 3, 1]
}

fn default_qwen3_tts_speaker_enc_dilations() -> Vec<usize> {
    vec![1, 2, 3, 4, 1]
}

fn default_qwen3_tts_speaker_enc_attention_channels() -> usize {
    128
}

fn default_qwen3_tts_speaker_enc_res2net_scale() -> usize {
    8
}

fn default_qwen3_tts_speaker_enc_se_channels() -> usize {
    128
}

fn default_qwen3_tts_speaker_sample_rate() -> usize {
    24_000
}

fn default_qwen3_tts_hidden_act() -> String {
    "silu".to_string()
}

fn default_qwen3_tts_initializer_range() -> f64 {
    0.02
}

fn default_qwen3_tts_use_cache() -> bool {
    true
}

fn default_qwen3_tts_attention_bias() -> bool {
    false
}

fn default_qwen3_tts_use_sliding_window() -> bool {
    false
}

fn default_qwen3_tts_attention_dropout() -> f64 {
    0.0
}

#[derive(Debug, Clone, Deserialize)]
pub struct Qwen3TtsRopeScalingConfig {
    pub rope_type: Option<String>,
    #[serde(rename = "type")]
    pub legacy_type: Option<String>,
    pub factor: Option<f64>,
    pub original_max_position_embeddings: Option<usize>,
    pub attention_factor: Option<f64>,
    pub beta_fast: Option<f64>,
    pub beta_slow: Option<f64>,
    pub short_factor: Option<Vec<f64>>,
    pub long_factor: Option<Vec<f64>>,
    pub low_freq_factor: Option<f64>,
    pub high_freq_factor: Option<f64>,
    pub mrope_section: Option<Vec<usize>>,
    pub interleaved: Option<bool>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Qwen3TtsSpeakerEncoderConfig {
    #[serde(default = "default_qwen3_tts_speaker_mel_dim")]
    pub mel_dim: usize,
    #[serde(default = "default_qwen3_tts_speaker_enc_dim")]
    pub enc_dim: usize,
    #[serde(default = "default_qwen3_tts_speaker_enc_channels")]
    pub enc_channels: Vec<usize>,
    #[serde(default = "default_qwen3_tts_speaker_enc_kernel_sizes")]
    pub enc_kernel_sizes: Vec<usize>,
    #[serde(default = "default_qwen3_tts_speaker_enc_dilations")]
    pub enc_dilations: Vec<usize>,
    #[serde(default = "default_qwen3_tts_speaker_enc_attention_channels")]
    pub enc_attention_channels: usize,
    #[serde(default = "default_qwen3_tts_speaker_enc_res2net_scale")]
    pub enc_res2net_scale: usize,
    #[serde(default = "default_qwen3_tts_speaker_enc_se_channels")]
    pub enc_se_channels: usize,
    #[serde(default = "default_qwen3_tts_speaker_sample_rate")]
    pub sample_rate: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Qwen3TtsTalkerCodePredictorConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub max_position_embeddings: usize,
    pub head_dim: usize,
    #[serde(default = "default_qwen3_tts_hidden_act")]
    pub hidden_act: String,
    #[serde(default = "default_qwen3_tts_initializer_range")]
    pub initializer_range: f64,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    #[serde(default)]
    pub rope_scaling: Option<Qwen3TtsRopeScalingConfig>,
    #[serde(default = "default_qwen3_tts_attention_bias")]
    pub attention_bias: bool,
    #[serde(default = "default_qwen3_tts_use_cache")]
    pub use_cache: bool,
    #[serde(default = "default_qwen3_tts_use_sliding_window")]
    pub use_sliding_window: bool,
    #[serde(default)]
    pub sliding_window: Option<usize>,
    #[serde(default = "default_qwen3_tts_attention_dropout")]
    pub attention_dropout: f64,
    pub num_code_groups: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Qwen3TtsTalkerConfig {
    pub codec_bos_id: u32,
    pub codec_eos_token_id: u32,
    pub codec_think_id: u32,
    pub codec_nothink_id: u32,
    pub codec_pad_id: u32,
    pub codec_think_bos_id: u32,
    pub codec_think_eos_id: u32,
    pub codec_language_id: HashMap<String, u32>,
    pub spk_id: HashMap<String, Vec<u32>>,
    pub spk_is_dialect: HashMap<String, serde_json::Value>,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub max_position_embeddings: usize,
    pub head_dim: usize,
    #[serde(default = "default_qwen3_tts_hidden_act")]
    pub hidden_act: String,
    #[serde(default = "default_qwen3_tts_initializer_range")]
    pub initializer_range: f64,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    #[serde(default)]
    pub rope_scaling: Option<Qwen3TtsRopeScalingConfig>,
    #[serde(default = "default_qwen3_tts_attention_bias")]
    pub attention_bias: bool,
    #[serde(default = "default_qwen3_tts_use_cache")]
    pub use_cache: bool,
    #[serde(default = "default_qwen3_tts_use_sliding_window")]
    pub use_sliding_window: bool,
    #[serde(default)]
    pub sliding_window: Option<usize>,
    #[serde(default = "default_qwen3_tts_attention_dropout")]
    pub attention_dropout: f64,
    pub text_hidden_size: usize,
    pub text_vocab_size: usize,
    pub vocab_size: usize,
    pub num_code_groups: usize,
    pub position_id_per_seconds: usize,
    pub code_predictor_config: Qwen3TtsTalkerCodePredictorConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Qwen3TtsConfig {
    #[serde(default)]
    pub im_start_token_id: Option<u32>,
    #[serde(default)]
    pub im_end_token_id: Option<u32>,
    pub model_type: String,
    pub tokenizer_type: String,
    pub tts_model_size: String,
    pub tts_model_type: String,
    pub tts_bos_token_id: u32,
    pub tts_eos_token_id: u32,
    pub tts_pad_token_id: u32,
    pub speaker_encoder_config: Qwen3TtsSpeakerEncoderConfig,
    pub talker_config: Qwen3TtsTalkerConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Qwen3TtsTokenizerDecoderConfig {
    pub attention_bias: bool,
    pub attention_dropout: f64,
    pub latent_dim: usize,
    pub codebook_dim: usize,
    pub codebook_size: usize,
    pub decoder_dim: usize,
    pub hidden_act: String,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub layer_scale_initial_scale: f64,
    pub max_position_embeddings: usize,
    pub head_dim: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub num_quantizers: usize,
    pub num_semantic_quantizers: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub semantic_codebook_size: usize,
    pub sliding_window: usize,
    pub upsample_rates: Vec<usize>,
    pub upsampling_ratios: Vec<usize>,
    pub vector_quantization_hidden_dimension: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Qwen3TtsTokenizerEncoderConfig {
    #[serde(rename = "_frame_rate")]
    pub frame_rate: f64,
    pub attention_bias: bool,
    pub attention_dropout: f64,
    pub audio_channels: usize,
    pub codebook_dim: usize,
    pub codebook_size: usize,
    pub compress: usize,
    pub dilation_growth_rate: usize,
    pub dtype: String,
    pub head_dim: usize,
    pub hidden_act: String,
    pub hidden_size: usize,
    pub initializer_range: f64,
    pub intermediate_size: usize,
    pub kernel_size: usize,
    pub last_kernel_size: usize,
    pub layer_scale_initial_scale: f64,
    pub max_position_embeddings: usize,
    pub norm_eps: f64,
    pub normalize: bool,
    pub num_attention_heads: usize,
    pub num_filters: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub num_quantizers: usize,
    pub num_residual_layers: usize,
    pub num_semantic_quantizers: usize,
    pub pad_mode: String,
    pub residual_kernel_size: usize,
    pub rope_theta: f64,
    pub sampling_rate: usize,
    pub sliding_window: usize,
    pub trim_right_ratio: f64,
    pub upsample_groups: usize,
    pub upsampling_ratios: Vec<usize>,
    pub use_cache: bool,
    pub use_causal_conv: bool,
    pub use_conv_shortcut: bool,
    pub use_streaming: bool,
    pub vector_quantization_hidden_dimension: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Qwen3TtsTokenizerConfig {
    pub model_type: String,
    pub input_sample_rate: usize,
    pub output_sample_rate: usize,
    pub decode_upsample_rate: usize,
    pub encode_downsample_rate: usize,
    pub decoder_config: Qwen3TtsTokenizerDecoderConfig,
    pub encoder_config: Qwen3TtsTokenizerEncoderConfig,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn speaker_config_uses_python_defaults_when_hf_config_omits_them() {
        let cfg: Qwen3TtsSpeakerEncoderConfig =
            serde_json::from_str(r#"{"enc_dim":1024,"sample_rate":24000}"#).unwrap();
        assert_eq!(cfg.mel_dim, 128);
        assert_eq!(cfg.enc_channels, vec![512, 512, 512, 512, 1536]);
        assert_eq!(cfg.enc_kernel_sizes, vec![5, 3, 3, 3, 1]);
        assert_eq!(cfg.enc_dilations, vec![1, 2, 3, 4, 1]);
        assert_eq!(cfg.enc_attention_channels, 128);
        assert_eq!(cfg.enc_res2net_scale, 8);
        assert_eq!(cfg.enc_se_channels, 128);
    }

    #[test]
    fn tokenizer_config_parses_reference_shape() {
        let cfg: Qwen3TtsTokenizerConfig = serde_json::from_str(
            r#"{
                "model_type":"qwen3_tts_tokenizer_12hz",
                "input_sample_rate":24000,
                "output_sample_rate":24000,
                "decode_upsample_rate":1920,
                "encode_downsample_rate":1920,
                "decoder_config":{
                    "attention_bias":false,
                    "attention_dropout":0.0,
                    "latent_dim":1024,
                    "codebook_dim":512,
                    "codebook_size":2048,
                    "decoder_dim":1536,
                    "hidden_act":"silu",
                    "hidden_size":512,
                    "intermediate_size":1024,
                    "layer_scale_initial_scale":0.01,
                    "max_position_embeddings":8000,
                    "head_dim":64,
                    "num_attention_heads":16,
                    "num_hidden_layers":8,
                    "num_key_value_heads":16,
                    "num_quantizers":16,
                    "num_semantic_quantizers":1,
                    "rms_norm_eps":1e-5,
                    "rope_theta":10000.0,
                    "semantic_codebook_size":4096,
                    "sliding_window":72,
                    "upsample_rates":[8,5,4,3],
                    "upsampling_ratios":[2,2],
                    "vector_quantization_hidden_dimension":512
                },
                "encoder_config":{
                    "_frame_rate":12.5,
                    "attention_bias":false,
                    "attention_dropout":0.0,
                    "audio_channels":1,
                    "codebook_dim":256,
                    "codebook_size":2048,
                    "compress":2,
                    "dilation_growth_rate":2,
                    "dtype":"float32",
                    "head_dim":64,
                    "hidden_act":"gelu",
                    "hidden_size":512,
                    "initializer_range":0.02,
                    "intermediate_size":2048,
                    "kernel_size":7,
                    "last_kernel_size":3,
                    "layer_scale_initial_scale":0.01,
                    "max_position_embeddings":8000,
                    "norm_eps":1e-5,
                    "normalize":false,
                    "num_attention_heads":8,
                    "num_filters":64,
                    "num_hidden_layers":8,
                    "num_key_value_heads":8,
                    "num_quantizers":32,
                    "num_residual_layers":1,
                    "num_semantic_quantizers":1,
                    "pad_mode":"constant",
                    "residual_kernel_size":3,
                    "rope_theta":10000.0,
                    "sampling_rate":24000,
                    "sliding_window":250,
                    "trim_right_ratio":1.0,
                    "upsample_groups":512,
                    "upsampling_ratios":[8,6,5,4],
                    "use_cache":false,
                    "use_causal_conv":true,
                    "use_conv_shortcut":false,
                    "use_streaming":false,
                    "vector_quantization_hidden_dimension":256
                }
            }"#,
        )
        .unwrap();
        assert_eq!(cfg.model_type, "qwen3_tts_tokenizer_12hz");
        assert_eq!(cfg.decode_upsample_rate, 1920);
        assert_eq!(cfg.decoder_config.num_quantizers, 16);
        assert_eq!(cfg.encoder_config.sampling_rate, 24_000);
    }
}
