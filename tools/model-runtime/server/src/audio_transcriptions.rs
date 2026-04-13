//! OpenAI-compatible audio transcription endpoint.

use anyhow::Result;
use axum::{
    extract::{Multipart, State},
    http::{self, HeaderMap, HeaderValue, StatusCode},
    response::IntoResponse,
};
use either::Either;
use engine_core::{
    AudioInput, Constraint, MistralRs, NormalRequest, Request, RequestMessage, Response,
    SamplingParams,
};
use indexmap::IndexMap;
use serde_json::Value;
use std::{error::Error, sync::Arc};

use crate::{
    handler_core::{create_response_channel, send_request, ErrorToResponse, JsonError},
    openai::AudioTranscriptionResponse,
    types::SharedMistralRsState,
    util::sanitize_error_message,
};

#[derive(Debug, Default)]
struct ParsedTranscriptionRequest {
    file_bytes: Vec<u8>,
    language: Option<String>,
    prompt: Option<String>,
    response_format: Option<String>,
}

pub enum AudioTranscriptionResponder {
    InternalError(Box<dyn Error>),
    ValidationError(Box<dyn Error>),
    RawResponse(axum::response::Response),
}

impl IntoResponse for AudioTranscriptionResponder {
    fn into_response(self) -> axum::response::Response {
        match self {
            Self::InternalError(e) => JsonError::new(sanitize_error_message(e.as_ref()))
                .to_response(StatusCode::INTERNAL_SERVER_ERROR),
            Self::ValidationError(e) => JsonError::new(sanitize_error_message(e.as_ref()))
                .to_response(StatusCode::UNPROCESSABLE_ENTITY),
            Self::RawResponse(resp) => resp,
        }
    }
}

fn build_transcription_prompt(language: Option<&str>, prompt: Option<&str>) -> String {
    let mut text = match prompt {
        Some(prompt) if !prompt.trim().is_empty() => prompt.trim().to_string(),
        _ => String::new(),
    };
    if let Some(language) = language.filter(|value| !value.trim().is_empty()) {
        if !text.is_empty() {
            text.push(' ');
        }
        text.push_str("The spoken language is likely ");
        text.push_str(language.trim());
        text.push('.');
    }
    text
}

fn transcription_content(
    prompt: &str,
    state: &SharedMistralRsState,
) -> Vec<IndexMap<String, Value>> {
    let prefixed_prompt = match state.get_model_category(None) {
        Ok(engine_core::ModelCategory::Vision { prefixer }) => {
            prefixer.prefix_audio(vec![0], prompt)
        }
        _ => prompt.to_string(),
    };

    let mut parts = Vec::new();

    let mut audio_part = IndexMap::new();
    audio_part.insert("type".to_string(), Value::String("audio".to_string()));
    parts.push(audio_part);

    let mut text_part = IndexMap::new();
    text_part.insert("type".to_string(), Value::String("text".to_string()));
    text_part.insert("text".to_string(), Value::String(prefixed_prompt));
    parts.push(text_part);

    parts
}

async fn parse_transcription_request(
    mut multipart: Multipart,
) -> Result<ParsedTranscriptionRequest> {
    let mut parsed = ParsedTranscriptionRequest::default();
    while let Some(field) = multipart.next_field().await? {
        let Some(name) = field.name().map(ToOwned::to_owned) else {
            continue;
        };
        match name.as_str() {
            "file" => {
                parsed.file_bytes = field.bytes().await?.to_vec();
            }
            "language" => {
                parsed.language = Some(field.text().await?);
            }
            "prompt" => {
                parsed.prompt = Some(field.text().await?);
            }
            "response_format" => {
                parsed.response_format = Some(field.text().await?);
            }
            _ => {
                let _ = field.bytes().await;
            }
        }
    }

    if parsed.file_bytes.is_empty() {
        anyhow::bail!("missing `file` field in multipart transcription request");
    }

    Ok(parsed)
}

fn handle_error(
    state: SharedMistralRsState,
    e: Box<dyn std::error::Error + Send + Sync + 'static>,
) -> AudioTranscriptionResponder {
    let sanitized_msg = sanitize_error_message(&*e);
    let e = anyhow::Error::msg(sanitized_msg);
    MistralRs::maybe_log_error(state, &*e);
    AudioTranscriptionResponder::InternalError(e.into())
}

fn response_format_or_default(value: Option<String>) -> String {
    value
        .unwrap_or_else(|| "json".to_string())
        .to_ascii_lowercase()
}

fn transcription_success_response(
    response_format: &str,
    text: String,
) -> AudioTranscriptionResponder {
    match response_format {
        "json" | "verbose_json" => {
            let payload = AudioTranscriptionResponse { text };
            AudioTranscriptionResponder::RawResponse((
                StatusCode::OK,
                [(http::header::CONTENT_TYPE, HeaderValue::from_static("application/json"))],
                serde_json::to_vec(&payload).unwrap_or_default(),
            )
                .into_response())
        }
        "text" => {
            let mut headers = HeaderMap::new();
            headers.insert(
                http::header::CONTENT_TYPE,
                HeaderValue::from_static("text/plain; charset=utf-8"),
            );
            AudioTranscriptionResponder::RawResponse((StatusCode::OK, headers, text).into_response())
        }
        other => AudioTranscriptionResponder::ValidationError(Box::new(JsonError::new(format!(
            "Unsupported transcription response format `{other}`. Supported values: json, verbose_json, text."
        )))),
    }
}

#[utoipa::path(
    post,
    tag = "Mistral.rs",
    path = "/v1/audio/transcriptions",
    responses((status = 200, description = "Audio transcription", body = AudioTranscriptionResponse))
)]
pub async fn audio_transcriptions(
    State(state): State<Arc<MistralRs>>,
    multipart: Multipart,
) -> AudioTranscriptionResponder {
    let parsed = match parse_transcription_request(multipart).await {
        Ok(parsed) => parsed,
        Err(err) => return AudioTranscriptionResponder::ValidationError(err.into()),
    };

    let audio = match AudioInput::from_bytes(&parsed.file_bytes) {
        Ok(audio) => audio,
        Err(err) => return AudioTranscriptionResponder::ValidationError(err.into()),
    };

    let prompt = build_transcription_prompt(parsed.language.as_deref(), parsed.prompt.as_deref());
    let mut message = IndexMap::new();
    message.insert("role".to_string(), Either::Left("user".to_string()));
    message.insert(
        "content".to_string(),
        Either::Right(transcription_content(&prompt, &state)),
    );

    let (tx, mut rx) = create_response_channel(None);
    let request = Request::Normal(Box::new(NormalRequest {
        id: state.next_request_id(),
        messages: RequestMessage::VisionChat {
            images: Vec::new(),
            audios: vec![audio],
            messages: vec![message],
            enable_thinking: Some(false),
            reasoning_effort: None,
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
        model_id: None,
        truncate_sequence: false,
    }));

    if let Err(err) = send_request(&state, request).await {
        return handle_error(state, err.into());
    }

    match rx.recv().await {
        Some(Response::Done(done)) => {
            let text = done
                .choices
                .first()
                .and_then(|choice| choice.message.content.clone())
                .unwrap_or_default();
            transcription_success_response(
                &response_format_or_default(parsed.response_format),
                text,
            )
        }
        Some(Response::ModelError(message, partial)) => {
            let text = partial
                .choices
                .first()
                .and_then(|choice| choice.message.content.clone())
                .unwrap_or_default();
            if text.is_empty() {
                AudioTranscriptionResponder::InternalError(anyhow::Error::msg(message).into())
            } else {
                transcription_success_response(
                    &response_format_or_default(parsed.response_format),
                    text,
                )
            }
        }
        Some(Response::ValidationError(err)) => AudioTranscriptionResponder::ValidationError(err),
        Some(Response::InternalError(err)) => AudioTranscriptionResponder::InternalError(err),
        Some(other) => AudioTranscriptionResponder::InternalError(
            anyhow::Error::msg(format!(
                "unexpected transcription response: {:?}",
                other_type_name(&other)
            ))
            .into(),
        ),
        None => AudioTranscriptionResponder::InternalError(
            anyhow::Error::msg("no response received from transcription model").into(),
        ),
    }
}

fn other_type_name(response: &Response) -> &'static str {
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
