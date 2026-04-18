//! ## Speech generation functionality and route handler.

use std::{error::Error, sync::Arc};

use anyhow::{Context, Result};
use axum::{
    body::Bytes,
    extract::{Json, State},
    http::{self, HeaderMap, HeaderValue, StatusCode},
    response::IntoResponse,
};
use base64::{prelude::BASE64_STANDARD, Engine};
use engine_core::{
    speech_utils::{self, Sample},
    Constraint, MistralRs, NormalRequest, Request, RequestMessage, Response, SamplingParams,
    SpeechGenerationRequest as CoreSpeechGenerationRequest,
};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc::{Receiver, Sender};

use crate::{
    handler_core::{
        base_process_non_streaming_response, create_response_channel, send_request,
        ErrorToResponse, JsonError,
    },
    openai::{AudioResponseFormat, SpeechGenerationRequest},
    types::SharedMistralRsState,
    util::{parse_audio_url, sanitize_error_message, validate_model_name},
};

fn trimmed_optional(value: Option<String>) -> Option<String> {
    value
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SpeechBackend {
    Qwen3Tts,
    VoxtralTts,
}

fn infer_speech_backend(model: &str) -> SpeechBackend {
    let lowered = model.trim().to_ascii_lowercase();
    if lowered.contains("voxtral") {
        SpeechBackend::VoxtralTts
    } else {
        SpeechBackend::Qwen3Tts
    }
}

fn resolve_effective_model_name(requested_model: &str, state: &MistralRs) -> Result<String> {
    if requested_model != "default" {
        return Ok(requested_model.to_string());
    }
    state
        .get_default_model_id()
        .map_err(anyhow::Error::msg)?
        .ok_or_else(|| anyhow::anyhow!("No default speech model is configured."))
}

fn infer_qwen3_tts_task_type(
    model: &str,
    task_type: Option<&str>,
    speaker: Option<&str>,
    instructions: Option<&str>,
    ref_audio: Option<&str>,
) -> &'static str {
    if let Some(task_type) = task_type {
        let normalized = task_type.trim().to_ascii_lowercase();
        match normalized.as_str() {
            "customvoice" | "custom_voice" => return "CustomVoice",
            "voicedesign" | "voice_design" => return "VoiceDesign",
            "base" | "voiceclone" | "voice_clone" => return "Base",
            _ => {}
        }
    }

    let lowered = model.trim().to_ascii_lowercase();
    if lowered.contains("customvoice") {
        return "CustomVoice";
    }
    if lowered.contains("voicedesign") {
        return "VoiceDesign";
    }
    if speaker.is_some() {
        return "CustomVoice";
    }
    if instructions.is_some() && ref_audio.is_none() {
        return "VoiceDesign";
    }
    "Base"
}

fn normalize_qwen3_tts_request(
    oairequest: SpeechGenerationRequest,
) -> Result<SpeechGenerationRequest> {
    let input = oairequest.input.trim().to_string();
    if input.is_empty() {
        anyhow::bail!("Speech generation input must not be empty.");
    }

    let speaker = trimmed_optional(oairequest.speaker);
    let language = trimmed_optional(oairequest.language).or(Some("Auto".to_string()));
    let instructions = trimmed_optional(oairequest.instructions);
    let ref_audio = trimmed_optional(oairequest.ref_audio);
    let ref_text = trimmed_optional(oairequest.ref_text);
    let x_vector_only_mode = oairequest.x_vector_only_mode.unwrap_or(false);
    let task_type = infer_qwen3_tts_task_type(
        &oairequest.model,
        oairequest.task_type.as_deref(),
        speaker.as_deref(),
        instructions.as_deref(),
        ref_audio.as_deref(),
    );

    match task_type {
        "CustomVoice" => {
            if speaker.is_none() {
                anyhow::bail!("Qwen3-TTS CustomVoice requests require `speaker`.");
            }
        }
        "VoiceDesign" => {
            if instructions.is_none() {
                anyhow::bail!("Qwen3-TTS VoiceDesign requests require `instructions`.");
            }
        }
        "Base" => {
            if ref_audio.is_none() {
                anyhow::bail!("Qwen3-TTS Base voice-clone requests require `ref_audio`.");
            }
            if !x_vector_only_mode && ref_text.is_none() {
                anyhow::bail!(
                    "Qwen3-TTS Base voice-clone requests require `ref_text` unless `x_vector_only_mode=true`."
                );
            }
        }
        _ => unreachable!(),
    }

    Ok(SpeechGenerationRequest {
        model: oairequest.model,
        input,
        response_format: oairequest.response_format,
        speaker,
        language,
        instructions,
        task_type: Some(task_type.to_string()),
        ref_audio,
        ref_text,
        ref_code: oairequest.ref_code,
        icl_mode: oairequest.icl_mode,
        x_vector_only_mode: Some(x_vector_only_mode),
        max_new_tokens: oairequest.max_new_tokens,
        temperature: oairequest.temperature,
        top_p: oairequest.top_p,
        top_k: oairequest.top_k,
        repetition_penalty: oairequest.repetition_penalty,
    })
}

fn normalize_voxtral_tts_request(
    oairequest: SpeechGenerationRequest,
) -> Result<SpeechGenerationRequest> {
    let input = oairequest.input.trim().to_string();
    if input.is_empty() {
        anyhow::bail!("Speech generation input must not be empty.");
    }

    if oairequest.ref_audio.is_some()
        || oairequest.ref_text.is_some()
        || oairequest.ref_code.is_some()
        || oairequest.icl_mode.is_some()
        || oairequest.x_vector_only_mode.is_some()
    {
        anyhow::bail!(
            "Voxtral-TTS currently supports preset-voice generation only; reference-audio and clone-specific fields are not supported."
        );
    }
    if oairequest.task_type.is_some() {
        anyhow::bail!("Voxtral-TTS does not accept `task_type`.");
    }

    Ok(SpeechGenerationRequest {
        model: oairequest.model,
        input,
        response_format: oairequest.response_format,
        speaker: trimmed_optional(oairequest.speaker),
        language: trimmed_optional(oairequest.language),
        instructions: trimmed_optional(oairequest.instructions),
        task_type: None,
        ref_audio: None,
        ref_text: None,
        ref_code: None,
        icl_mode: None,
        x_vector_only_mode: None,
        max_new_tokens: oairequest.max_new_tokens,
        temperature: oairequest.temperature,
        top_p: oairequest.top_p,
        top_k: oairequest.top_k,
        repetition_penalty: None,
    })
}

fn normalize_speech_request(
    oairequest: SpeechGenerationRequest,
    backend: SpeechBackend,
) -> Result<SpeechGenerationRequest> {
    match backend {
        SpeechBackend::Qwen3Tts => normalize_qwen3_tts_request(oairequest),
        SpeechBackend::VoxtralTts => normalize_voxtral_tts_request(oairequest),
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalSpeechGenerationResponse {
    pub response_format: AudioResponseFormat,
    pub content_type: String,
    pub sample_rate_hz: usize,
    pub channels: usize,
    pub audio_base64: String,
}

/// Represents different types of speech generation responses.
pub enum SpeechGenerationResponder {
    InternalError(Box<dyn Error>),
    ValidationError(Box<dyn Error>),
    RawResponse(axum::response::Response),
}

impl IntoResponse for SpeechGenerationResponder {
    /// Converts the speech generation responder into an HTTP response.
    fn into_response(self) -> axum::response::Response {
        match self {
            SpeechGenerationResponder::InternalError(e) => {
                JsonError::new(sanitize_error_message(e.as_ref()))
                    .to_response(http::StatusCode::INTERNAL_SERVER_ERROR)
            }
            SpeechGenerationResponder::ValidationError(e) => {
                JsonError::new(sanitize_error_message(e.as_ref()))
                    .to_response(http::StatusCode::UNPROCESSABLE_ENTITY)
            }
            SpeechGenerationResponder::RawResponse(resp) => resp,
        }
    }
}

/// Parses and validates a speech generation request.
///
/// This function transforms a speech generation request into the
/// request format used by engine.rs.
pub async fn parse_request(
    oairequest: SpeechGenerationRequest,
    state: Arc<MistralRs>,
    tx: Sender<Response>,
) -> Result<(Request, AudioResponseFormat)> {
    let effective_model = resolve_effective_model_name(&oairequest.model, &state)?;
    let backend = infer_speech_backend(&effective_model);
    let oairequest = normalize_speech_request(oairequest, backend)?;
    let repr = serde_json::to_string(&oairequest).expect("Serialization of request failed.");
    MistralRs::maybe_log_request(state.clone(), repr);

    // Validate that the requested model matches the loaded model
    validate_model_name(&oairequest.model, state.clone())?;

    let ref_audio_input = match &oairequest.ref_audio {
        Some(ref_audio) => Some(parse_audio_url(ref_audio).await?),
        None => None,
    };

    let request = Request::Normal(Box::new(NormalRequest {
        id: state.next_request_id(),
        messages: RequestMessage::SpeechGeneration {
            request: CoreSpeechGenerationRequest {
                input: oairequest.input,
                speaker: oairequest.speaker,
                language: oairequest.language,
                instructions: oairequest.instructions,
                task_type: oairequest.task_type,
                ref_audio: oairequest.ref_audio,
                ref_audio_input,
                ref_text: oairequest.ref_text,
                ref_code: oairequest.ref_code,
                icl_mode: oairequest.icl_mode,
                x_vector_only_mode: oairequest.x_vector_only_mode,
                max_new_tokens: oairequest.max_new_tokens,
                temperature: oairequest.temperature.map(|v| v as f32),
                top_p: oairequest.top_p.map(|v| v as f32),
                top_k: oairequest.top_k,
                repetition_penalty: oairequest.repetition_penalty,
            },
        },
        sampling_params: SamplingParams::deterministic(),
        response: tx,
        return_logprobs: false,
        is_streaming: false,
        suffix: None,
        constraint: Constraint::None,
        tool_choice: None,
        tools: None,
        logits_processors: None,
        return_raw_logits: false,
        web_search_options: None,
        model_id: if oairequest.model == "default" {
            None
        } else {
            Some(oairequest.model.clone())
        },
        truncate_sequence: false,
    }));

    Ok((request, oairequest.response_format))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::openai::AudioResponseFormat;

    fn base_request() -> SpeechGenerationRequest {
        SpeechGenerationRequest {
            model: "Qwen/Qwen3-TTS-12Hz-0.6B-Base".to_string(),
            input: "Hello world".to_string(),
            response_format: AudioResponseFormat::Wav,
            speaker: None,
            language: None,
            instructions: None,
            task_type: None,
            ref_audio: Some("https://example.com/ref.wav".to_string()),
            ref_text: Some("Hello".to_string()),
            ref_code: None,
            icl_mode: None,
            x_vector_only_mode: Some(false),
            max_new_tokens: Some(512),
            temperature: None,
            top_p: None,
            top_k: None,
            repetition_penalty: None,
        }
    }

    #[test]
    fn normalizes_qwen_base_defaults() {
        let normalized = normalize_speech_request(base_request(), SpeechBackend::Qwen3Tts).unwrap();
        assert_eq!(normalized.task_type.as_deref(), Some("Base"));
        assert_eq!(normalized.language.as_deref(), Some("Auto"));
    }

    #[test]
    fn custom_voice_requires_speaker() {
        let mut request = base_request();
        request.model = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice".to_string();
        request.ref_audio = None;
        request.ref_text = None;
        request.task_type = Some("CustomVoice".to_string());
        let error = normalize_speech_request(request, SpeechBackend::Qwen3Tts)
            .unwrap_err()
            .to_string();
        assert!(error.contains("require `speaker`"));
    }

    #[test]
    fn voice_design_requires_instructions() {
        let mut request = base_request();
        request.model = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign".to_string();
        request.ref_audio = None;
        request.ref_text = None;
        request.task_type = Some("VoiceDesign".to_string());
        let error = normalize_speech_request(request, SpeechBackend::Qwen3Tts)
            .unwrap_err()
            .to_string();
        assert!(error.contains("require `instructions`"));
    }

    #[test]
    fn voxtral_preserves_absent_speaker_for_core_voice_selection() {
        let normalized = normalize_speech_request(
            SpeechGenerationRequest {
                model: "engineai/Voxtral-4B-TTS-2603".to_string(),
                input: "Hello world".to_string(),
                response_format: AudioResponseFormat::Wav,
                speaker: None,
                language: None,
                instructions: Some("calm, slightly warm".to_string()),
                task_type: None,
                ref_audio: None,
                ref_text: None,
                ref_code: None,
                icl_mode: None,
                x_vector_only_mode: None,
                max_new_tokens: Some(512),
                temperature: Some(0.7),
                top_p: Some(0.95),
                top_k: Some(32),
                repetition_penalty: Some(1.1),
            },
            SpeechBackend::VoxtralTts,
        )
        .unwrap();
        assert_eq!(normalized.speaker, None);
        assert_eq!(
            normalized.instructions.as_deref(),
            Some("calm, slightly warm")
        );
        assert_eq!(normalized.repetition_penalty, None);
        assert_eq!(normalized.task_type, None);
    }

    #[test]
    fn voxtral_rejects_clone_inputs() {
        let err = normalize_speech_request(
            SpeechGenerationRequest {
                model: "engineai/Voxtral-4B-TTS-2603".to_string(),
                input: "Hello world".to_string(),
                response_format: AudioResponseFormat::Wav,
                speaker: Some("casual_male".to_string()),
                language: None,
                instructions: None,
                task_type: None,
                ref_audio: Some("https://example.com/ref.wav".to_string()),
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
            SpeechBackend::VoxtralTts,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("preset-voice generation only"));
    }
}

/// Speech generation endpoint handler.
#[utoipa::path(
    post,
    tag = "Mistral.rs",
    path = "/v1/audio/speech",
    request_body = SpeechGenerationRequest,
    responses((status = 200, description = "Speech generation"))
)]
pub async fn speech_generation(
    State(state): State<Arc<MistralRs>>,
    Json(oairequest): Json<SpeechGenerationRequest>,
) -> SpeechGenerationResponder {
    let (tx, mut rx) = create_response_channel(None);

    let (request, response_format) = match parse_request(oairequest, state.clone(), tx).await {
        Ok(x) => x,
        Err(e) => return handle_error(state, e.into()),
    };

    // Validate response format here
    if !matches!(
        response_format,
        AudioResponseFormat::Wav | AudioResponseFormat::Pcm
    ) {
        return SpeechGenerationResponder::ValidationError(Box::new(JsonError::new(
            "Only support wav/pcm response format.".to_string(),
        )));
    }

    if let Err(e) = send_request(&state, request).await {
        return handle_error(state, e.into());
    }

    process_non_streaming_response(&mut rx, state, response_format).await
}

pub async fn create_local_speech(
    state: SharedMistralRsState,
    oairequest: SpeechGenerationRequest,
) -> Result<LocalSpeechGenerationResponse> {
    let (tx, mut rx) = create_response_channel(None);

    let (request, response_format) = parse_request(oairequest, state.clone(), tx).await?;
    if !matches!(
        response_format,
        AudioResponseFormat::Wav | AudioResponseFormat::Pcm
    ) {
        anyhow::bail!("Only support wav/pcm response format.");
    }

    send_request(&state, request)
        .await
        .map_err(|err| anyhow::anyhow!(sanitize_error_message(err.as_ref())))?;

    match rx.recv().await {
        Some(Response::InternalError(err)) => {
            Err(anyhow::anyhow!(sanitize_error_message(err.as_ref())))
        }
        Some(Response::ValidationError(err)) => {
            Err(anyhow::anyhow!(sanitize_error_message(err.as_ref())))
        }
        Some(Response::CompletionModelError(message, _)) => Err(anyhow::anyhow!(message)),
        Some(Response::Speech {
            pcm,
            rate,
            channels,
        }) => encode_local_speech_response(&pcm, rate, channels, response_format),
        Some(other) => Err(anyhow::anyhow!(
            "unexpected speech generation response: {}",
            local_response_type_name(&other)
        )),
        None => Err(anyhow::anyhow!(
            "no response received from speech generation model"
        )),
    }
}

/// Helper function to handle speech generation errors and logging them.
pub fn handle_error(
    state: SharedMistralRsState,
    e: Box<dyn std::error::Error + Send + Sync + 'static>,
) -> SpeechGenerationResponder {
    let sanitized_msg = sanitize_error_message(&*e);
    let e = anyhow::Error::msg(sanitized_msg);
    MistralRs::maybe_log_error(state, &*e);
    SpeechGenerationResponder::InternalError(e.into())
}

fn encode_local_speech_response(
    pcm: &[f32],
    rate: usize,
    channels: usize,
    response_format: AudioResponseFormat,
) -> Result<LocalSpeechGenerationResponse> {
    let pcm_endianness = "s16le";
    let content_type = response_format.audio_content_type(rate, channels, pcm_endianness);
    let encoded = match response_format {
        AudioResponseFormat::Pcm => {
            let samples: &[f32] = pcm;
            let mut buf = Vec::with_capacity(samples.len() * std::mem::size_of::<i64>());
            for &sample in samples {
                buf.extend_from_slice(&sample.to_i16().to_le_bytes());
            }
            buf
        }
        AudioResponseFormat::Wav => {
            let mut buf = Vec::new();
            speech_utils::write_pcm_as_wav(&mut buf, pcm, rate as u32, channels as u16)
                .context("failed to encode local WAV response")?;
            buf
        }
        _ => anyhow::bail!("Only support wav/pcm response format."),
    };

    Ok(LocalSpeechGenerationResponse {
        response_format,
        content_type,
        sample_rate_hz: rate,
        channels,
        audio_base64: BASE64_STANDARD.encode(encoded),
    })
}

/// Process non-streaming speech generation responses.
pub async fn process_non_streaming_response(
    rx: &mut Receiver<Response>,
    state: SharedMistralRsState,
    response_format: AudioResponseFormat,
) -> SpeechGenerationResponder {
    base_process_non_streaming_response(
        rx,
        state,
        |state, response| match_responses(state, response, response_format),
        handle_error,
    )
    .await
}

/// Matches and processes different types of model responses into appropriate speech generation responses.
pub fn match_responses(
    state: SharedMistralRsState,
    response: Response,
    response_format: AudioResponseFormat,
) -> SpeechGenerationResponder {
    match response {
        Response::InternalError(e) => {
            MistralRs::maybe_log_error(state, &*e);
            SpeechGenerationResponder::InternalError(e)
        }
        Response::ValidationError(e) => SpeechGenerationResponder::ValidationError(e),
        Response::ImageGeneration(_) => unreachable!(),
        Response::CompletionModelError(m, _) => {
            let e = anyhow::Error::msg(m.to_string());
            MistralRs::maybe_log_error(state, &*e);
            SpeechGenerationResponder::InternalError(e.into())
        }
        Response::CompletionDone(_) => unreachable!(),
        Response::CompletionChunk(_) => unreachable!(),
        Response::Chunk(_) => unreachable!(),
        Response::Done(_) => unreachable!(),
        Response::ModelError(_, _) => unreachable!(),
        Response::Speech {
            pcm,
            rate,
            channels,
        } => {
            let local_response =
                match encode_local_speech_response(&pcm, rate, channels, response_format) {
                    Ok(response) => response,
                    Err(err) => return SpeechGenerationResponder::InternalError(err.into()),
                };
            let mut headers = HeaderMap::new();
            headers.insert(
                http::header::CONTENT_TYPE,
                HeaderValue::from_str(&local_response.content_type).unwrap(),
            );
            let bytes = match BASE64_STANDARD.decode(local_response.audio_base64) {
                Ok(encoded) => Bytes::from(encoded),
                Err(err) => {
                    return SpeechGenerationResponder::InternalError(
                        anyhow::Error::msg(format!(
                            "failed to decode local speech response payload: {err}"
                        ))
                        .into(),
                    )
                }
            };

            SpeechGenerationResponder::RawResponse((StatusCode::OK, headers, bytes).into_response())
        }
        Response::Raw { .. } => unreachable!(),
        Response::Embeddings { .. } => unreachable!(),
    }
}

fn local_response_type_name(response: &Response) -> &'static str {
    match response {
        Response::InternalError(_) => "internal_error",
        Response::ValidationError(_) => "validation_error",
        Response::ModelError(_, _) => "model_error",
        Response::Done(_) => "done",
        Response::Chunk(_) => "chunk",
        Response::CompletionModelError(_, _) => "completion_model_error",
        Response::CompletionDone(_) => "completion_done",
        Response::CompletionChunk(_) => "completion_chunk",
        Response::ImageGeneration(_) => "image_generation",
        Response::Speech { .. } => "speech",
        Response::Raw { .. } => "raw",
        Response::Embeddings { .. } => "embeddings",
    }
}
