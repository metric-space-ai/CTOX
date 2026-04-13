mod bs1770;
mod dia;
mod qwen3_tts;
pub mod utils;
mod voxtral_tts;

use std::{str::FromStr, sync::Arc};

pub use dia::{DiaConfig, DiaPipeline};
#[allow(unused_imports)]
pub use qwen3_tts::{
    prepare_request as prepare_qwen3_tts_request, Qwen3TtsAudioProcessor, Qwen3TtsConfig,
    Qwen3TtsPreparedRequest, Qwen3TtsRopeScalingConfig, Qwen3TtsSpeakerEncoder,
    Qwen3TtsSpeakerEncoderConfig, Qwen3TtsTalker, Qwen3TtsTalkerCodePredictorConfig,
    Qwen3TtsTalkerConfig, Qwen3TtsTaskType, Qwen3TtsTokenizerConfig, Qwen3TtsTokenizerDecoder,
    Qwen3TtsTokenizerDecoderConfig, Qwen3TtsTokenizerEncoder, Qwen3TtsTokenizerEncoderConfig,
};
use serde::{Deserialize, Serialize};
#[allow(unused_imports)]
pub use voxtral_tts::{
    build_voxtral_tts_input_ids, load_voice_embedding_pt, prepare_decode_chunks,
    prepare_request as prepare_voxtral_tts_request, AudioSpecialToken, TorchStorageKind,
    TorchTensorArchive, VoxtralTtsAcousticTransformer, VoxtralTtsAudioEncodingArgs,
    VoxtralTtsAudioModelArgs, VoxtralTtsAudioTokenizer, VoxtralTtsAudioTokenizerArgs,
    VoxtralTtsConfig, VoxtralTtsLanguageModel, VoxtralTtsMultimodalConfig,
    VoxtralTtsPreparedRequest,
};

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq)]
pub enum SpeechLoaderType {
    #[serde(rename = "dia")]
    Dia,
    #[serde(rename = "qwen3_tts")]
    Qwen3Tts,
    #[serde(rename = "voxtral_tts")]
    VoxtralTts,
}

impl FromStr for SpeechLoaderType {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "dia" => Ok(Self::Dia),
            "qwen3_tts" | "qwen3-tts" | "qwen3tts" => Ok(Self::Qwen3Tts),
            "voxtral_tts" | "voxtral-tts" | "voxtraltts" => Ok(Self::VoxtralTts),
            a => Err(format!(
                "Unknown architecture `{a}`. Possible architectures: `dia`, `qwen3_tts`, `voxtral_tts`."
            )),
        }
    }
}

impl SpeechLoaderType {
    /// Auto-detect speech loader type from a config.json string.
    /// Extend this when adding new speech pipelines.
    pub fn auto_detect_from_config(config: &str) -> Option<Self> {
        if let Ok(cfg) = serde_json::from_str::<Qwen3TtsConfig>(config) {
            if cfg.model_type == "qwen3_tts" && cfg.tokenizer_type == "qwen3_tts_tokenizer_12hz" {
                return Some(Self::Qwen3Tts);
            }
        }
        if let Ok(cfg) = serde_json::from_str::<VoxtralTtsConfig>(config) {
            if cfg.model_type == "voxtral_tts" {
                return Some(Self::VoxtralTts);
            }
        }
        if serde_json::from_str::<DiaConfig>(config).is_ok() {
            return Some(Self::Dia);
        }
        None
    }
}

#[derive(Clone, Copy, Debug)]
pub enum SpeechGenerationConfig {
    Dia {
        max_tokens: Option<usize>,
        cfg_scale: f32,
        temperature: f32,
        top_p: f32,
        top_k: Option<usize>,
    },
    Qwen3Tts {
        max_new_tokens: Option<usize>,
        temperature: f32,
        top_p: f32,
        top_k: Option<usize>,
        repetition_penalty: f32,
        subtalker_do_sample: bool,
        subtalker_temperature: f32,
        subtalker_top_p: f32,
        subtalker_top_k: Option<usize>,
    },
    VoxtralTts {
        max_new_tokens: Option<usize>,
        temperature: f32,
        top_p: f32,
        top_k: Option<usize>,
    },
}

#[derive(Clone, Debug, Deserialize)]
struct Qwen3TtsGenerationConfigFile {
    max_new_tokens: Option<usize>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<usize>,
    repetition_penalty: Option<f32>,
    subtalker_dosample: Option<bool>,
    subtalker_temperature: Option<f32>,
    subtalker_top_p: Option<f32>,
    subtalker_top_k: Option<usize>,
}

impl SpeechGenerationConfig {
    pub fn default(ty: SpeechLoaderType) -> Self {
        match ty {
            SpeechLoaderType::Dia => Self::Dia {
                max_tokens: None,
                cfg_scale: 3.,
                temperature: 1.3,
                top_p: 0.95,
                top_k: Some(35),
            },
            SpeechLoaderType::Qwen3Tts => Self::Qwen3Tts {
                max_new_tokens: Some(2048),
                temperature: 0.9,
                top_p: 1.0,
                top_k: Some(50),
                repetition_penalty: 1.05,
                subtalker_do_sample: true,
                subtalker_temperature: 0.9,
                subtalker_top_p: 1.0,
                subtalker_top_k: Some(50),
            },
            SpeechLoaderType::VoxtralTts => Self::VoxtralTts {
                max_new_tokens: Some(2500),
                temperature: 0.8,
                top_p: 1.0,
                top_k: None,
            },
        }
    }

    pub fn from_qwen3_tts_generation_config(raw: &str) -> Option<Self> {
        let parsed = serde_json::from_str::<Qwen3TtsGenerationConfigFile>(raw).ok()?;
        Some(Self::Qwen3Tts {
            max_new_tokens: parsed.max_new_tokens.or(Some(2048)),
            temperature: parsed.temperature.unwrap_or(0.9),
            top_p: parsed.top_p.unwrap_or(1.0),
            top_k: parsed.top_k.or(Some(50)),
            repetition_penalty: parsed.repetition_penalty.unwrap_or(1.05),
            subtalker_do_sample: parsed.subtalker_dosample.unwrap_or(true),
            subtalker_temperature: parsed.subtalker_temperature.unwrap_or(0.9),
            subtalker_top_p: parsed.subtalker_top_p.unwrap_or(1.0),
            subtalker_top_k: parsed.subtalker_top_k.or(Some(50)),
        })
    }
}

#[derive(Clone, Debug)]
pub struct SpeechGenerationOutput {
    pub pcm: Arc<Vec<f32>>,
    pub rate: usize,
    pub channels: usize,
}
