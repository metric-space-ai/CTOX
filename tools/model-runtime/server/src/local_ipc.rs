use std::io::ErrorKind;
use std::path::Path;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Context;
use anyhow::Result;
use futures::future::poll_fn;
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, BufWriter};
use tokio::net::{UnixListener, UnixStream};
use tracing::warn;
use uuid::Uuid;

use crate::audio_transcriptions::{
    LocalAudioTranscriptionRequest, LocalAudioTranscriptionResponse, create_local_transcription,
};
use crate::embeddings::{LocalEmbeddingsRequest, LocalEmbeddingsResponse, create_local_embeddings};
use crate::responses::{
    OpenResponsesCreateRequest, OpenResponsesStreamEvent, create_local_openresponses_streamer,
};
use crate::responses_types::enums::ResponseStatus;
use crate::responses_types::resource::{ResponseError, ResponseResource};
use crate::speech_generation::{LocalSpeechGenerationResponse, create_local_speech};
use crate::types::SharedMistralRsState;

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum LocalIpcRequest {
    ResponsesCreate(OpenResponsesCreateRequest),
    EmbeddingsCreate(LocalEmbeddingsRequest),
    TranscriptionCreate(LocalAudioTranscriptionRequest),
    SpeechCreate(crate::openai::SpeechGenerationRequest),
    RuntimeHealth,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum LocalIpcResponse {
    Embeddings(LocalEmbeddingsResponse),
    Transcription(LocalAudioTranscriptionResponse),
    Speech(LocalSpeechGenerationResponse),
    RuntimeHealth(LocalRuntimeHealthResponse),
    Error { code: String, message: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalRuntimeHealthResponse {
    pub healthy: bool,
    pub default_model: Option<String>,
    pub loaded_models: Vec<String>,
}

pub async fn serve_local_openresponses_socket(
    state: SharedMistralRsState,
    transport_endpoint: PathBuf,
) -> Result<()> {
    if let Some(parent) = transport_endpoint.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create socket dir {}", parent.display()))?;
    }

    remove_stale_transport_endpoint(&transport_endpoint)?;
    let listener = UnixListener::bind(&transport_endpoint).with_context(|| {
        format!(
            "failed to bind local responses socket {}",
            transport_endpoint.display()
        )
    })?;
    let _cleanup = SocketCleanup(transport_endpoint.clone());

    loop {
        let (stream, _) = listener.accept().await.with_context(|| {
            format!(
                "failed to accept local responses socket connection on {}",
                transport_endpoint.display()
            )
        })?;
        let state = state.clone();
        tokio::spawn(async move {
            if let Err(err) = handle_connection(state, stream).await {
                warn!("local responses socket connection failed: {err:#}");
            }
        });
    }
}

async fn handle_connection(state: SharedMistralRsState, stream: UnixStream) -> Result<()> {
    let (reader, writer) = stream.into_split();
    let mut reader = BufReader::new(reader);
    let mut writer = BufWriter::new(writer);
    let mut line = String::new();

    let read = reader
        .read_line(&mut line)
        .await
        .context("failed to read local responses socket request")?;
    if read == 0 {
        return Ok(());
    }

    match serde_json::from_str::<LocalIpcRequest>(line.trim()) {
        Ok(LocalIpcRequest::ResponsesCreate(request)) => {
            return handle_responses_request(state, &mut writer, request).await;
        }
        Ok(LocalIpcRequest::EmbeddingsCreate(request)) => {
            return handle_embeddings_request(state, &mut writer, request).await;
        }
        Ok(LocalIpcRequest::TranscriptionCreate(request)) => {
            return handle_transcription_request(state, &mut writer, request).await;
        }
        Ok(LocalIpcRequest::SpeechCreate(request)) => {
            return handle_speech_request(state, &mut writer, request).await;
        }
        Ok(LocalIpcRequest::RuntimeHealth) => {
            return handle_runtime_health_request(state, &mut writer).await;
        }
        Err(err) => {
            let response = LocalIpcResponse::Error {
                code: "invalid_request".to_string(),
                message: err.to_string(),
            };
            write_json_line(&mut writer, &response).await?;
            return Ok(());
        }
    };
}

async fn handle_responses_request(
    state: SharedMistralRsState,
    writer: &mut BufWriter<tokio::net::unix::OwnedWriteHalf>,
    request: OpenResponsesCreateRequest,
) -> Result<()> {
    let failed_model = request.model.clone();
    let streamer = match create_local_openresponses_streamer(state, request).await {
        Ok(streamer) => streamer,
        Err(err) => {
            write_failed_event(writer, &failed_model, "invalid_prompt", err.to_string()).await?;
            return Ok(());
        }
    };

    let mut streamer = Box::pin(streamer);
    while let Some(event) = poll_fn(|cx| streamer.as_mut().poll_next_openresponses_event(cx)).await
    {
        write_json_line(writer, &event).await?;
    }

    writer
        .flush()
        .await
        .context("failed to flush local responses socket writer")
}

async fn handle_embeddings_request(
    state: SharedMistralRsState,
    writer: &mut BufWriter<tokio::net::unix::OwnedWriteHalf>,
    request: LocalEmbeddingsRequest,
) -> Result<()> {
    let response = match create_local_embeddings(state, request).await {
        Ok(response) => LocalIpcResponse::Embeddings(response),
        Err(err) => LocalIpcResponse::Error {
            code: "embedding_failed".to_string(),
            message: err.to_string(),
        },
    };
    write_json_line(writer, &response).await
}

async fn handle_transcription_request(
    state: SharedMistralRsState,
    writer: &mut BufWriter<tokio::net::unix::OwnedWriteHalf>,
    request: LocalAudioTranscriptionRequest,
) -> Result<()> {
    let response = match create_local_transcription(state, request).await {
        Ok(response) => LocalIpcResponse::Transcription(response),
        Err(err) => LocalIpcResponse::Error {
            code: "transcription_failed".to_string(),
            message: err.to_string(),
        },
    };
    write_json_line(writer, &response).await
}

async fn handle_speech_request(
    state: SharedMistralRsState,
    writer: &mut BufWriter<tokio::net::unix::OwnedWriteHalf>,
    request: crate::openai::SpeechGenerationRequest,
) -> Result<()> {
    let response = match create_local_speech(state, request).await {
        Ok(response) => LocalIpcResponse::Speech(response),
        Err(err) => LocalIpcResponse::Error {
            code: "speech_generation_failed".to_string(),
            message: err.to_string(),
        },
    };
    write_json_line(writer, &response).await
}

async fn handle_runtime_health_request(
    state: SharedMistralRsState,
    writer: &mut BufWriter<tokio::net::unix::OwnedWriteHalf>,
) -> Result<()> {
    let response = LocalIpcResponse::RuntimeHealth(LocalRuntimeHealthResponse {
        healthy: true,
        default_model: state.get_default_model_id().unwrap_or(None),
        loaded_models: state.list_models().unwrap_or_default(),
    });
    write_json_line(writer, &response).await
}

async fn write_failed_event(
    writer: &mut BufWriter<tokio::net::unix::OwnedWriteHalf>,
    model: &str,
    code: &str,
    message: String,
) -> Result<()> {
    let mut response = ResponseResource::new(
        format!("resp_{}", Uuid::new_v4()),
        model.to_string(),
        now_unix_seconds(),
    );
    response.status = ResponseStatus::Failed;
    response.error = Some(ResponseError::new(code.to_string(), message));

    let event = OpenResponsesStreamEvent::ResponseFailed {
        sequence_number: 0,
        response,
    };
    write_json_line(writer, &event).await
}

async fn write_json_line<T: Serialize>(
    writer: &mut BufWriter<tokio::net::unix::OwnedWriteHalf>,
    event: &T,
) -> Result<()> {
    let serialized =
        serde_json::to_string(event).context("failed to encode local responses socket event")?;
    writer
        .write_all(serialized.as_bytes())
        .await
        .context("failed to write local responses socket event")?;
    writer
        .write_all(b"\n")
        .await
        .context("failed to terminate local responses socket event")?;
    writer
        .flush()
        .await
        .context("failed to flush local responses socket event")
}

fn now_unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn remove_stale_transport_endpoint(transport_endpoint: &Path) -> Result<()> {
    match std::fs::remove_file(transport_endpoint) {
        Ok(()) => Ok(()),
        Err(err) if err.kind() == ErrorKind::NotFound => Ok(()),
        Err(err) => Err(err).with_context(|| {
            format!(
                "failed to remove stale local responses socket {}",
                transport_endpoint.display()
            )
        }),
    }
}

struct SocketCleanup(PathBuf);

impl Drop for SocketCleanup {
    fn drop(&mut self) {
        #[cfg(unix)]
        if socket_listener_accepts(&self.0) {
            return;
        }
        let _ = std::fs::remove_file(&self.0);
    }
}

#[cfg(unix)]
fn socket_listener_accepts(path: &Path) -> bool {
    if !path.exists() {
        return false;
    }
    std::os::unix::net::UnixStream::connect(path).is_ok()
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;

    #[test]
    fn socket_cleanup_keeps_path_when_another_listener_is_active() {
        let root = std::env::temp_dir().join(format!("ctox-local-ipc-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root).unwrap();
        let transport_endpoint = root.join("responses.sock");
        let _listener = std::os::unix::net::UnixListener::bind(&transport_endpoint).unwrap();

        let cleanup = SocketCleanup(transport_endpoint.clone());
        drop(cleanup);

        assert!(transport_endpoint.exists());
        std::fs::remove_file(&transport_endpoint).unwrap();
        std::fs::remove_dir_all(&root).unwrap();
    }
}
