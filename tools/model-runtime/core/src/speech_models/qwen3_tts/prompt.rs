use anyhow::Result;
use engine_audio::AudioInput;
use tokenizers::Tokenizer;

use crate::request::SpeechGenerationRequest;

use super::Qwen3TtsConfig;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Qwen3TtsTaskType {
    Base,
    CustomVoice,
    VoiceDesign,
}

#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct Qwen3TtsPreparedRequest {
    pub task_type: Qwen3TtsTaskType,
    pub input_ids: Vec<u32>,
    pub instruct_ids: Option<Vec<u32>>,
    pub ref_ids: Option<Vec<u32>>,
    pub ref_codes: Option<Vec<Vec<u32>>>,
    pub language: String,
    pub speaker: Option<String>,
    pub ref_audio_input: Option<AudioInput>,
    pub ref_text: Option<String>,
    pub icl_mode: bool,
    pub x_vector_only_mode: bool,
    pub requires_ref_codes: bool,
}

fn infer_task_type(
    cfg: &Qwen3TtsConfig,
    request: &SpeechGenerationRequest,
) -> Result<Qwen3TtsTaskType> {
    let normalized = request
        .task_type
        .as_deref()
        .map(|s| s.trim().to_ascii_lowercase())
        .unwrap_or_else(|| cfg.tts_model_type.trim().to_ascii_lowercase());
    match normalized.as_str() {
        "base" | "voiceclone" | "voice_clone" => Ok(Qwen3TtsTaskType::Base),
        "customvoice" | "custom_voice" => Ok(Qwen3TtsTaskType::CustomVoice),
        "voicedesign" | "voice_design" => Ok(Qwen3TtsTaskType::VoiceDesign),
        other => anyhow::bail!("Unsupported Qwen3-TTS task type `{other}`."),
    }
}

fn build_assistant_text(text: &str) -> String {
    format!("<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n")
}

fn build_ref_text(text: &str) -> String {
    format!("<|im_start|>assistant\n{text}<|im_end|>\n")
}

fn build_instruct_text(text: &str) -> String {
    format!("<|im_start|>user\n{text}<|im_end|>\n")
}

fn encode_text(tokenizer: &Tokenizer, text: &str) -> Result<Vec<u32>> {
    Ok(tokenizer
        .encode_fast(text, false)
        .map_err(anyhow::Error::msg)?
        .get_ids()
        .to_vec())
}

pub fn prepare_request(
    cfg: &Qwen3TtsConfig,
    tokenizer: &Tokenizer,
    request: &SpeechGenerationRequest,
) -> Result<Qwen3TtsPreparedRequest> {
    let input = request.input.trim();
    if input.is_empty() {
        anyhow::bail!("Qwen3-TTS input must not be empty.");
    }

    let task_type = infer_task_type(cfg, request)?;
    let language = request
        .language
        .clone()
        .unwrap_or_else(|| "Auto".to_string());
    let speaker = request.speaker.clone();
    let ref_text = request.ref_text.clone();
    let x_vector_only_mode = request.x_vector_only_mode.unwrap_or(false);
    let icl_mode = request.icl_mode.unwrap_or(!x_vector_only_mode);

    let input_ids = encode_text(tokenizer, &build_assistant_text(input))?;
    let instruct_ids = request
        .instructions
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|instructions| encode_text(tokenizer, &build_instruct_text(instructions)))
        .transpose()?;
    let ref_ids = ref_text
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|reference| encode_text(tokenizer, &build_ref_text(reference)))
        .transpose()?;
    let ref_codes = request.ref_code.clone();

    match task_type {
        Qwen3TtsTaskType::CustomVoice => {
            if speaker.is_none() {
                anyhow::bail!("Qwen3-TTS CustomVoice requests require `speaker`.");
            }
        }
        Qwen3TtsTaskType::VoiceDesign => {
            if instruct_ids.is_none() {
                anyhow::bail!("Qwen3-TTS VoiceDesign requests require `instructions`.");
            }
        }
        Qwen3TtsTaskType::Base => {
            if request.ref_audio_input.is_none() {
                anyhow::bail!("Qwen3-TTS Base requests require resolved `ref_audio` input.");
            }
            if !x_vector_only_mode && ref_ids.is_none() {
                anyhow::bail!(
                    "Qwen3-TTS Base requests require `ref_text` unless `x_vector_only_mode=true`."
                );
            }
        }
    }

    let requires_ref_codes =
        matches!(task_type, Qwen3TtsTaskType::Base) && !x_vector_only_mode && icl_mode;

    Ok(Qwen3TtsPreparedRequest {
        task_type,
        input_ids,
        instruct_ids,
        ref_ids,
        ref_codes,
        language,
        speaker,
        ref_audio_input: request.ref_audio_input.clone(),
        ref_text,
        icl_mode,
        x_vector_only_mode,
        requires_ref_codes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::request::SpeechGenerationRequest;
    use ahash::AHashMap;

    fn tokenizer() -> Tokenizer {
        let vocab = AHashMap::from([
            ("[UNK]".to_string(), 0u32),
            ("<|im_start|>".to_string(), 1u32),
            ("<|im_end|>".to_string(), 2u32),
            ("assistant".to_string(), 3u32),
            ("user".to_string(), 4u32),
            ("Hello".to_string(), 5u32),
            ("Ref".to_string(), 6u32),
            ("Design".to_string(), 7u32),
        ]);
        let model = tokenizers::models::wordlevel::WordLevel::builder()
            .vocab(vocab)
            .unk_token("[UNK]".to_string())
            .build()
            .unwrap();
        let mut tokenizer = Tokenizer::new(model);
        tokenizer.add_special_tokens(
            &[
                "[UNK]",
                "<|im_start|>",
                "<|im_end|>",
                "assistant",
                "user",
                "Hello",
                "Ref",
                "Design",
            ]
            .into_iter()
            .map(|s| tokenizers::tokenizer::AddedToken::from(s.to_string(), true))
            .collect::<Vec<_>>(),
        );
        tokenizer
    }

    fn cfg(tts_model_type: &str) -> Qwen3TtsConfig {
        serde_json::from_str(&format!(
            r#"{{
                "model_type":"qwen3_tts",
                "tokenizer_type":"qwen3_tts_tokenizer_12hz",
                "tts_model_size":"0b6",
                "tts_model_type":"{tts_model_type}",
                "tts_bos_token_id":151672,
                "tts_eos_token_id":151673,
                "tts_pad_token_id":151671,
                "speaker_encoder_config":{{"enc_dim":1024,"sample_rate":24000}},
                "talker_config":{{
                    "codec_bos_id":1,
                    "codec_eos_token_id":2,
                    "codec_think_id":3,
                    "codec_nothink_id":4,
                    "codec_pad_id":5,
                    "codec_think_bos_id":6,
                    "codec_think_eos_id":7,
                    "codec_language_id":{{"english":8}},
                    "spk_id":{{"vivian":[9,10]}},
                    "spk_is_dialect":{{"vivian":false}},
                    "hidden_size":1024,
                    "intermediate_size":3072,
                    "num_hidden_layers":2,
                    "num_attention_heads":8,
                    "num_key_value_heads":8,
                    "max_position_embeddings":128,
                    "head_dim":128,
                    "rms_norm_eps":1e-5,
                    "rope_theta":10000.0,
                    "text_hidden_size":2048,
                    "text_vocab_size":151936,
                    "vocab_size":3072,
                    "num_code_groups":16,
                    "position_id_per_seconds":13,
                    "code_predictor_config":{{
                        "vocab_size":2048,
                        "hidden_size":1024,
                        "intermediate_size":3072,
                        "num_hidden_layers":2,
                        "num_attention_heads":8,
                        "num_key_value_heads":8,
                        "max_position_embeddings":128,
                        "head_dim":128,
                        "rms_norm_eps":1e-5,
                        "rope_theta":10000.0,
                        "num_code_groups":16
                    }}
                }}
            }}"#
        ))
        .unwrap()
    }

    #[test]
    fn prepares_custom_voice_request() {
        let request = SpeechGenerationRequest {
            input: "Hello".to_string(),
            speaker: Some("Vivian".to_string()),
            language: Some("English".to_string()),
            instructions: None,
            task_type: Some("CustomVoice".to_string()),
            ref_audio: None,
            ref_audio_input: None,
            ref_text: None,
            ref_code: None,
            icl_mode: None,
            x_vector_only_mode: None,
            max_new_tokens: None,
            temperature: None,
            top_p: None,
            top_k: None,
            repetition_penalty: None,
        };
        let prepared = prepare_request(&cfg("custom_voice"), &tokenizer(), &request).unwrap();
        assert_eq!(prepared.task_type, Qwen3TtsTaskType::CustomVoice);
        assert_eq!(prepared.language, "English");
        assert_eq!(prepared.speaker.as_deref(), Some("Vivian"));
    }

    #[test]
    fn base_requires_ref_audio_and_ref_text_when_not_xvec_only() {
        let request = SpeechGenerationRequest {
            input: "Hello".to_string(),
            speaker: None,
            language: None,
            instructions: None,
            task_type: Some("Base".to_string()),
            ref_audio: Some("ignored".to_string()),
            ref_audio_input: None,
            ref_text: None,
            ref_code: None,
            icl_mode: None,
            x_vector_only_mode: Some(false),
            max_new_tokens: None,
            temperature: None,
            top_p: None,
            top_k: None,
            repetition_penalty: None,
        };
        let error = prepare_request(&cfg("base"), &tokenizer(), &request)
            .unwrap_err()
            .to_string();
        assert!(error.contains("resolved `ref_audio`"));
    }

    #[test]
    fn base_with_ref_audio_and_ref_text_requires_ref_codes() {
        let request = SpeechGenerationRequest {
            input: "Hello".to_string(),
            speaker: None,
            language: Some("English".to_string()),
            instructions: None,
            task_type: Some("Base".to_string()),
            ref_audio: Some("ignored".to_string()),
            ref_audio_input: Some(AudioInput {
                samples: vec![0.0; 24_000],
                sample_rate: 24_000,
                channels: 1,
            }),
            ref_text: Some("Ref".to_string()),
            ref_code: None,
            icl_mode: Some(true),
            x_vector_only_mode: Some(false),
            max_new_tokens: None,
            temperature: None,
            top_p: None,
            top_k: None,
            repetition_penalty: None,
        };
        let prepared = prepare_request(&cfg("base"), &tokenizer(), &request).unwrap();
        assert_eq!(prepared.task_type, Qwen3TtsTaskType::Base);
        assert_eq!(prepared.ref_text.as_deref(), Some("Ref"));
        assert!(prepared.ref_ids.is_some());
        assert!(prepared.requires_ref_codes);
        assert!(prepared.icl_mode);
        assert!(!prepared.x_vector_only_mode);
    }

    #[test]
    fn base_xvector_only_does_not_require_ref_codes() {
        let request = SpeechGenerationRequest {
            input: "Hello".to_string(),
            speaker: None,
            language: Some("English".to_string()),
            instructions: None,
            task_type: Some("Base".to_string()),
            ref_audio: Some("ignored".to_string()),
            ref_audio_input: Some(AudioInput {
                samples: vec![0.0; 24_000],
                sample_rate: 24_000,
                channels: 1,
            }),
            ref_text: None,
            ref_code: None,
            icl_mode: None,
            x_vector_only_mode: Some(true),
            max_new_tokens: None,
            temperature: None,
            top_p: None,
            top_k: None,
            repetition_penalty: None,
        };
        let prepared = prepare_request(&cfg("base"), &tokenizer(), &request).unwrap();
        assert_eq!(prepared.task_type, Qwen3TtsTaskType::Base);
        assert!(!prepared.requires_ref_codes);
        assert!(prepared.x_vector_only_mode);
    }
}
