use anyhow::{anyhow, bail, Result};
use tokenizers::Tokenizer;

use crate::request::SpeechGenerationRequest;
use crate::utils::tokenizer::TekkenTextEncoder;

use super::VoxtralTtsConfig;

#[derive(Debug, Clone)]
pub struct VoxtralTtsPreparedRequest {
    pub text_token_ids: Vec<u32>,
    pub voice: String,
    pub instructions: Option<String>,
}

const BOS_TOKEN: &str = "<s>";
const AUDIO_TOKEN: &str = "[AUDIO]";
const BEGIN_AUDIO_TOKEN: &str = "[BEGIN_AUDIO]";
const TEXT_TO_AUDIO_TOKEN: &str = "[NEXT_AUDIO_TEXT]";
const AUDIO_TO_TEXT_TOKEN: &str = "[REPEAT_AUDIO_TEXT]";

fn get_special_token_id(tokenizer: &Tokenizer, token: &str) -> Result<u32> {
    tokenizer
        .token_to_id(token)
        .ok_or_else(|| anyhow!("Voxtral-TTS tokenizer is missing required special token `{token}`"))
}

fn resolve_voice_name(cfg: &VoxtralTtsConfig, requested: Option<&str>) -> Result<String> {
    let available = cfg.voice_names();
    if available.is_empty() {
        bail!("Voxtral-TTS config does not declare any preset voices.");
    }

    let requested = requested
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or("casual_female");
    if let Some(matched) = available
        .iter()
        .find(|candidate| candidate.eq_ignore_ascii_case(requested))
    {
        return Ok((*matched).to_string());
    }

    bail!(
        "unknown Voxtral-TTS voice `{requested}`; available preset voices: {}",
        available.join(", ")
    );
}

fn looks_like_german(text: &str) -> bool {
    let lower = text.to_lowercase();
    if lower.contains('ä') || lower.contains('ö') || lower.contains('ü') || lower.contains('ß')
    {
        return true;
    }
    let markers = [
        "hallo",
        "dieser",
        "deutscher",
        "muss",
        "bleiben",
        " und ",
        " ist ",
        " nicht ",
        " auf ",
        " aus ",
    ];
    markers
        .into_iter()
        .filter(|marker| lower.contains(marker))
        .count()
        >= 2
}

pub fn prepare_request(
    cfg: &VoxtralTtsConfig,
    tokenizer: &Tokenizer,
    text_encoder: Option<&TekkenTextEncoder>,
    request: &SpeechGenerationRequest,
) -> Result<VoxtralTtsPreparedRequest> {
    let prompt = request.input.trim().to_string();
    if prompt.is_empty() {
        bail!("Voxtral-TTS input must not be empty.");
    }
    let inferred_voice = if request.speaker.is_none()
        && looks_like_german(&prompt)
        && cfg.voice_names().iter().any(|voice| *voice == "de_female")
    {
        Some("de_female")
    } else {
        None
    };
    let voice = resolve_voice_name(cfg, request.speaker.as_deref().or(inferred_voice))?;
    let instructions = request
        .instructions
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned);
    let text_token_ids = if let Some(text_encoder) = text_encoder {
        text_encoder.encode_ordinary(prompt.as_str())?
    } else {
        tokenizer
            .encode(prompt.as_str(), false)
            .map_err(|err| anyhow!("failed to tokenize Voxtral-TTS prompt: {err}"))?
            .get_ids()
            .to_vec()
    };

    Ok(VoxtralTtsPreparedRequest {
        text_token_ids,
        voice,
        instructions,
    })
}

pub fn build_input_ids(
    tokenizer: &Tokenizer,
    prepared: &VoxtralTtsPreparedRequest,
    voice_embedding_frames: usize,
) -> Result<Vec<u32>> {
    if voice_embedding_frames == 0 {
        bail!("Voxtral-TTS voice embeddings must contain at least one frame.");
    }
    let bos = get_special_token_id(tokenizer, BOS_TOKEN)?;
    let audio = get_special_token_id(tokenizer, AUDIO_TOKEN)?;
    let begin_audio = get_special_token_id(tokenizer, BEGIN_AUDIO_TOKEN)?;
    let text_to_audio = get_special_token_id(tokenizer, TEXT_TO_AUDIO_TOKEN)?;
    let audio_to_text = get_special_token_id(tokenizer, AUDIO_TO_TEXT_TOKEN)?;

    let mut input_ids =
        Vec::with_capacity(5 + voice_embedding_frames + prepared.text_token_ids.len());
    input_ids.push(bos);
    input_ids.push(begin_audio);
    input_ids.extend(std::iter::repeat_n(audio, voice_embedding_frames));
    input_ids.push(text_to_audio);
    input_ids.extend(prepared.text_token_ids.iter().copied());
    input_ids.push(audio_to_text);
    input_ids.push(begin_audio);
    Ok(input_ids)
}

#[cfg(test)]
mod tests {
    use tokenizers::models::wordlevel::WordLevel;
    use tokenizers::pre_tokenizers::whitespace::Whitespace;

    use super::*;

    fn cfg() -> VoxtralTtsConfig {
        serde_json::from_str(
            r#"{
                "dim":3072,
                "n_layers":26,
                "head_dim":128,
                "hidden_dim":9216,
                "n_heads":32,
                "n_kv_heads":8,
                "vocab_size":8,
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
        .unwrap()
    }

    fn cfg_with_german_voice() -> VoxtralTtsConfig {
        let mut cfg = cfg();
        cfg.multimodal
            .audio_tokenizer_args
            .voice
            .insert("de_female".to_string(), 2);
        cfg
    }

    fn tokenizer() -> Tokenizer {
        let model = WordLevel::builder()
            .vocab(
                [
                    ("[UNK]".to_string(), 0),
                    ("<s>".to_string(), 1),
                    ("[AUDIO]".to_string(), 2),
                    ("[BEGIN_AUDIO]".to_string(), 3),
                    ("[NEXT_AUDIO_TEXT]".to_string(), 4),
                    ("[REPEAT_AUDIO_TEXT]".to_string(), 5),
                    ("hello".to_string(), 6),
                    ("world".to_string(), 7),
                ]
                .into_iter()
                .collect(),
            )
            .unk_token("[UNK]".to_string())
            .build()
            .unwrap();
        let mut tokenizer = Tokenizer::new(model);
        tokenizer.with_pre_tokenizer(Some(Whitespace));
        tokenizer.add_special_tokens(
            &[
                BOS_TOKEN,
                AUDIO_TOKEN,
                BEGIN_AUDIO_TOKEN,
                TEXT_TO_AUDIO_TOKEN,
                AUDIO_TO_TEXT_TOKEN,
            ]
            .into_iter()
            .map(|tok| tokenizers::AddedToken::from(tok.to_string(), true))
            .collect::<Vec<_>>(),
        );
        tokenizer
    }

    #[test]
    fn defaults_to_casual_female() {
        let prepared = prepare_request(
            &cfg(),
            &tokenizer(),
            None,
            &SpeechGenerationRequest {
                input: "hello world".to_string(),
                speaker: None,
                language: None,
                instructions: None,
                task_type: None,
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
            },
        )
        .unwrap();
        assert_eq!(prepared.voice, "casual_female");
        assert_eq!(prepared.text_token_ids, vec![6, 7]);
    }

    #[test]
    fn builds_reference_voice_prompt_shape() {
        let prepared = prepare_request(
            &cfg(),
            &tokenizer(),
            None,
            &SpeechGenerationRequest {
                input: "hello world".to_string(),
                speaker: Some("casual_male".to_string()),
                language: None,
                instructions: None,
                task_type: None,
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
            },
        )
        .unwrap();
        let ids = build_input_ids(&tokenizer(), &prepared, 3).unwrap();
        assert_eq!(ids, vec![1, 3, 2, 2, 2, 4, 6, 7, 5, 3]);
    }

    #[test]
    fn defaults_to_german_voice_for_german_text() {
        let prepared = prepare_request(
            &cfg_with_german_voice(),
            &tokenizer(),
            None,
            &SpeechGenerationRequest {
                input: "Hallo aus dem nativen Candle Port. Dieser Satz muss bleiben.".to_string(),
                speaker: None,
                language: None,
                instructions: None,
                task_type: None,
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
            },
        )
        .unwrap();
        assert_eq!(prepared.voice, "de_female");
    }

    #[test]
    fn rejects_unknown_voice() {
        let err = prepare_request(
            &cfg(),
            &tokenizer(),
            None,
            &SpeechGenerationRequest {
                input: "hello".to_string(),
                speaker: Some("robot".to_string()),
                language: None,
                instructions: None,
                task_type: None,
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
            },
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("unknown Voxtral-TTS voice"));
    }
}
