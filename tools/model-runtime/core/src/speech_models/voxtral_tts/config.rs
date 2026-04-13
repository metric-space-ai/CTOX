use serde::Deserialize;
use std::collections::BTreeMap;

#[allow(dead_code)]
#[derive(Debug, Clone, Deserialize)]
pub struct VoxtralTtsAudioEncodingArgs {
    pub codebook_pattern: Option<String>,
    pub interleave_audio_tokens_per_segment: Option<usize>,
    pub interleave_text_tokens_per_segment: Option<usize>,
    pub single_trailing_segment: Option<bool>,
    pub num_codebooks: usize,
    pub sampling_rate: usize,
    pub frame_rate: f64,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Deserialize)]
pub struct VoxtralTtsAcousticTransformerArgs {
    pub input_dim: usize,
    pub dim: usize,
    pub n_layers: usize,
    pub head_dim: usize,
    pub hidden_dim: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub use_biases: bool,
    pub rope_theta: f64,
    pub sigma: f64,
    pub sigma_max: Option<f64>,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Deserialize)]
pub struct VoxtralTtsAudioModelArgs {
    pub semantic_codebook_size: usize,
    pub acoustic_codebook_size: usize,
    pub n_acoustic_codebook: usize,
    pub audio_encoding_args: VoxtralTtsAudioEncodingArgs,
    pub audio_token_id: u32,
    pub begin_audio_token_id: u32,
    pub input_embedding_concat_type: Option<String>,
    pub acoustic_transformer_args: VoxtralTtsAcousticTransformerArgs,
    pub p_uncond: Option<f64>,
    pub text_feature_bugged: Option<bool>,
    pub condition_dropped_token_id: Option<u32>,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Deserialize)]
pub struct VoxtralTtsAudioTokenizerArgs {
    pub channels: usize,
    pub sampling_rate: usize,
    pub pretransform_patch_size: usize,
    pub patch_proj_kernel_size: usize,
    pub semantic_codebook_size: usize,
    pub semantic_dim: usize,
    pub acoustic_codebook_size: usize,
    pub acoustic_dim: usize,
    pub conv_weight_norm: bool,
    pub causal: bool,
    pub attn_sliding_window_size: usize,
    pub half_attn_window_upon_downsampling: bool,
    pub dim: usize,
    pub hidden_dim: usize,
    pub head_dim: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub qk_norm_eps: f64,
    pub qk_norm: bool,
    pub use_biases: bool,
    pub norm_eps: f64,
    pub layer_scale: bool,
    pub layer_scale_init: Option<f64>,
    #[serde(default)]
    pub encoder_transformer_lengths_str: Option<String>,
    #[serde(default)]
    pub encoder_convs_kernels_str: Option<String>,
    #[serde(default)]
    pub encoder_convs_strides_str: Option<String>,
    pub decoder_transformer_lengths_str: String,
    pub decoder_convs_kernels_str: String,
    pub decoder_convs_strides_str: String,
    #[serde(default)]
    pub voice: BTreeMap<String, usize>,
}

#[allow(dead_code)]
impl VoxtralTtsAudioTokenizerArgs {
    pub fn encoder_transformer_lengths(&self) -> Vec<usize> {
        self.encoder_transformer_lengths_str
            .as_deref()
            .unwrap_or_default()
            .split(',')
            .filter_map(|x| x.trim().parse::<usize>().ok())
            .collect()
    }

    pub fn encoder_convs_kernels(&self) -> Vec<usize> {
        self.encoder_convs_kernels_str
            .as_deref()
            .unwrap_or_default()
            .split(',')
            .filter_map(|x| x.trim().parse::<usize>().ok())
            .collect()
    }

    pub fn encoder_convs_strides(&self) -> Vec<usize> {
        self.encoder_convs_strides_str
            .as_deref()
            .unwrap_or_default()
            .split(',')
            .filter_map(|x| x.trim().parse::<usize>().ok())
            .collect()
    }

    pub fn decoder_transformer_lengths(&self) -> Vec<usize> {
        self.decoder_transformer_lengths_str
            .split(',')
            .filter_map(|x| x.trim().parse::<usize>().ok())
            .collect()
    }

    pub fn decoder_convs_kernels(&self) -> Vec<usize> {
        self.decoder_convs_kernels_str
            .split(',')
            .filter_map(|x| x.trim().parse::<usize>().ok())
            .collect()
    }

    pub fn decoder_convs_strides(&self) -> Vec<usize> {
        self.decoder_convs_strides_str
            .split(',')
            .filter_map(|x| x.trim().parse::<usize>().ok())
            .collect()
    }

    pub fn downsample_factor(&self) -> usize {
        let encoder_strides = self.encoder_convs_strides();
        let strides = if encoder_strides.is_empty() {
            self.decoder_convs_strides()
        } else {
            encoder_strides
        };
        self.pretransform_patch_size * strides.into_iter().product::<usize>()
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone, Deserialize)]
pub struct VoxtralTtsMultimodalConfig {
    pub audio_model_args: VoxtralTtsAudioModelArgs,
    pub audio_tokenizer_args: VoxtralTtsAudioTokenizerArgs,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Deserialize)]
pub struct VoxtralTtsConfig {
    pub dim: usize,
    pub n_layers: usize,
    pub head_dim: usize,
    pub hidden_dim: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub vocab_size: usize,
    pub rope_theta: f64,
    pub norm_eps: f64,
    pub tied_embeddings: bool,
    pub max_seq_len: usize,
    pub max_position_embeddings: Option<usize>,
    pub model_type: String,
    pub multimodal: VoxtralTtsMultimodalConfig,
}

impl VoxtralTtsConfig {
    pub fn language_hidden_size(&self) -> usize {
        self.dim
    }

    pub fn voice_names(&self) -> Vec<&str> {
        self.multimodal
            .audio_tokenizer_args
            .voice
            .keys()
            .map(String::as_str)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::VoxtralTtsConfig;

    #[test]
    fn parses_reference_shape() {
        let cfg: VoxtralTtsConfig = serde_json::from_str(
            r#"{
                "dim":3072,
                "n_layers":26,
                "head_dim":128,
                "hidden_dim":9216,
                "n_heads":32,
                "n_kv_heads":8,
                "vocab_size":131072,
                "rope_theta":1000000.0,
                "norm_eps":1e-5,
                "tied_embeddings":true,
                "max_seq_len":65536,
                "model_type":"voxtral_tts",
                "multimodal":{
                    "audio_model_args":{
                        "semantic_codebook_size":8192,
                        "acoustic_codebook_size":21,
                        "n_acoustic_codebook":36,
                        "audio_encoding_args":{
                            "num_codebooks":37,
                            "sampling_rate":24000,
                            "frame_rate":12.5
                        },
                        "audio_token_id":24,
                        "begin_audio_token_id":25,
                        "acoustic_transformer_args":{
                            "input_dim":3072,
                            "dim":3072,
                            "n_layers":3,
                            "head_dim":128,
                            "hidden_dim":9216,
                            "n_heads":32,
                            "n_kv_heads":8,
                            "use_biases":false,
                            "rope_theta":10000.0,
                            "sigma":1e-5
                        }
                    },
                    "audio_tokenizer_args":{
                        "channels":1,
                        "sampling_rate":24000,
                        "pretransform_patch_size":240,
                        "patch_proj_kernel_size":7,
                        "semantic_codebook_size":8192,
                        "semantic_dim":256,
                        "acoustic_codebook_size":21,
                        "acoustic_dim":36,
                        "conv_weight_norm":true,
                        "causal":true,
                        "attn_sliding_window_size":16,
                        "half_attn_window_upon_downsampling":true,
                        "dim":1024,
                        "hidden_dim":4096,
                        "head_dim":128,
                        "n_heads":8,
                        "n_kv_heads":8,
                        "qk_norm_eps":1e-6,
                        "qk_norm":true,
                        "use_biases":false,
                        "norm_eps":1e-2,
                        "layer_scale":true,
                        "encoder_transformer_lengths_str":"2,2,2,2",
                        "encoder_convs_kernels_str":"4,4,4,3",
                        "encoder_convs_strides_str":"2,2,2,1",
                        "decoder_transformer_lengths_str":"2,2,2,2",
                        "decoder_convs_kernels_str":"3,4,4,4",
                        "decoder_convs_strides_str":"1,2,2,2",
                        "voice":{"casual_female":0,"casual_male":1}
                    }
                }
            }"#,
        )
        .unwrap();
        assert_eq!(cfg.model_type, "voxtral_tts");
        assert_eq!(
            cfg.multimodal
                .audio_model_args
                .audio_encoding_args
                .num_codebooks,
            37
        );
        assert_eq!(
            cfg.multimodal.audio_tokenizer_args.voice["casual_female"],
            0
        );
        assert_eq!(
            cfg.multimodal.audio_tokenizer_args.downsample_factor(),
            1920
        );
        assert_eq!(
            cfg.multimodal.audio_tokenizer_args.encoder_convs_strides(),
            vec![2, 2, 2, 1]
        );
    }

    #[test]
    fn parses_decoder_only_audio_tokenizer_shape() {
        let cfg: VoxtralTtsConfig = serde_json::from_str(
            r#"{
                "dim":3072,
                "n_layers":26,
                "head_dim":128,
                "hidden_dim":9216,
                "n_heads":32,
                "n_kv_heads":8,
                "vocab_size":131072,
                "rope_theta":1000000.0,
                "norm_eps":1e-5,
                "tied_embeddings":true,
                "max_seq_len":65536,
                "model_type":"voxtral_tts",
                "multimodal":{
                    "audio_model_args":{
                        "semantic_codebook_size":8192,
                        "acoustic_codebook_size":21,
                        "n_acoustic_codebook":36,
                        "audio_encoding_args":{
                            "num_codebooks":37,
                            "sampling_rate":24000,
                            "frame_rate":12.5
                        },
                        "audio_token_id":24,
                        "begin_audio_token_id":25,
                        "acoustic_transformer_args":{
                            "input_dim":3072,
                            "dim":3072,
                            "n_layers":3,
                            "head_dim":128,
                            "hidden_dim":9216,
                            "n_heads":32,
                            "n_kv_heads":8,
                            "use_biases":false,
                            "rope_theta":10000.0,
                            "sigma":1e-5
                        }
                    },
                    "audio_tokenizer_args":{
                        "channels":1,
                        "sampling_rate":24000,
                        "pretransform_patch_size":240,
                        "patch_proj_kernel_size":7,
                        "semantic_codebook_size":8192,
                        "semantic_dim":256,
                        "acoustic_codebook_size":21,
                        "acoustic_dim":36,
                        "conv_weight_norm":true,
                        "causal":true,
                        "attn_sliding_window_size":16,
                        "half_attn_window_upon_downsampling":true,
                        "dim":1024,
                        "hidden_dim":4096,
                        "head_dim":128,
                        "n_heads":8,
                        "n_kv_heads":8,
                        "qk_norm_eps":1e-6,
                        "qk_norm":true,
                        "use_biases":false,
                        "norm_eps":0.01,
                        "layer_scale":true,
                        "layer_scale_init":0.01,
                        "decoder_transformer_lengths_str":"2,2,2,2",
                        "decoder_convs_kernels_str":"3,4,4,4",
                        "decoder_convs_strides_str":"1,2,2,2"
                    }
                }
            }"#,
        )
        .unwrap();
        assert!(cfg
            .multimodal
            .audio_tokenizer_args
            .encoder_convs_strides()
            .is_empty());
        assert_eq!(
            cfg.multimodal.audio_tokenizer_args.downsample_factor(),
            1920
        );
    }
}
