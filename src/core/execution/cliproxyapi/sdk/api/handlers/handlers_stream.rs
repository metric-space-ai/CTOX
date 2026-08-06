// ref: sdk/api/handlers/handlers_stream.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use tokio::sync::mpsc;

use crate::sdk::cliproxy::executor::StreamChunk;

#[derive(Debug)]
pub struct StreamBootstrap {
    pub first_chunk: Option<StreamChunk>,
    pub committed: bool,
}

pub async fn bootstrap_stream(
    cancellation: &super::HandlerCancellation,
    chunks: &mut mpsc::Receiver<StreamChunk>,
) -> Result<StreamBootstrap, crate::sdk::cliproxy::executor::ExecutionError> {
    if cancellation.is_cancelled() {
        return Ok(StreamBootstrap {
            first_chunk: None,
            committed: false,
        });
    }
    match chunks.recv().await {
        Some(chunk) if chunk.error.is_some() => Err(chunk.error.expect("checked above")),
        Some(chunk) => Ok(StreamBootstrap {
            first_chunk: Some(chunk),
            committed: true,
        }),
        None => Ok(StreamBootstrap {
            first_chunk: None,
            committed: true,
        }),
    }
}

pub fn validate_sse_data_json(chunk: &[u8]) -> Result<(), serde_json::Error> {
    for line in chunk.split(|byte| *byte == b'\n') {
        let line = line.strip_suffix(b"\r").unwrap_or(line);
        let Some(data) = line.strip_prefix(b"data:") else {
            continue;
        };
        let data = trim_ascii(data);
        if data.is_empty() || data == b"[DONE]" {
            continue;
        }
        serde_json::from_slice::<serde_json::Value>(data)?;
    }
    Ok(())
}

fn trim_ascii(mut bytes: &[u8]) -> &[u8] {
    while bytes.first().is_some_and(u8::is_ascii_whitespace) {
        bytes = &bytes[1..];
    }
    while bytes.last().is_some_and(u8::is_ascii_whitespace) {
        bytes = &bytes[..bytes.len() - 1];
    }
    bytes
}

#[cfg(test)]
#[path = "handlers_stream_bootstrap_test.rs"]
mod handlers_stream_bootstrap_test;
