// Origin: CTOX
// Port-Status: adapted_to_ctox
// Port-Note: CTOX process-frame transport replacing in-process plugin calls
// License: AGPL-3.0-only

use std::collections::HashMap;
use std::fmt;

use serde_json::value::RawValue;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

use crate::sdk::pluginabi::{Envelope, Error as PluginError};

use super::rpc_schema::{
    decode_process_frame, encode_process_frame, ProcessCodecError, ProcessMessage, MAX_FRAME_BYTES,
};

pub const MAX_INFLIGHT_REQUESTS: usize = 256;

pub async fn read_process_message<R>(
    reader: &mut R,
) -> Result<Option<ProcessMessage>, ProcessTransportError>
where
    R: AsyncRead + Unpin,
{
    let mut prefix = [0_u8; 4];
    let first = reader
        .read(&mut prefix[..1])
        .await
        .map_err(|_| ProcessTransportError::Io)?;
    if first == 0 {
        return Ok(None);
    }
    reader
        .read_exact(&mut prefix[1..])
        .await
        .map_err(|_| ProcessTransportError::TruncatedFrame)?;
    let length = u32::from_be_bytes(prefix) as usize;
    if length > MAX_FRAME_BYTES {
        return Err(ProcessTransportError::FrameTooLarge);
    }
    let mut frame = Vec::with_capacity(4 + length);
    frame.extend_from_slice(&prefix);
    frame.resize(4 + length, 0);
    reader
        .read_exact(&mut frame[4..])
        .await
        .map_err(|_| ProcessTransportError::TruncatedFrame)?;
    decode_process_frame(&frame)
        .map(Some)
        .map_err(ProcessTransportError::Codec)
}

pub async fn write_process_message<W>(
    writer: &mut W,
    message: &ProcessMessage,
) -> Result<(), ProcessTransportError>
where
    W: AsyncWrite + Unpin,
{
    let frame = encode_process_frame(message).map_err(ProcessTransportError::Codec)?;
    writer
        .write_all(&frame)
        .await
        .map_err(|_| ProcessTransportError::Io)?;
    writer.flush().await.map_err(|_| ProcessTransportError::Io)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProcessTransportError {
    Io,
    TruncatedFrame,
    FrameTooLarge,
    Codec(ProcessCodecError),
}

impl fmt::Display for ProcessTransportError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io => formatter.write_str("plugin process transport failed"),
            Self::TruncatedFrame => formatter.write_str("plugin process frame is truncated"),
            Self::FrameTooLarge => {
                formatter.write_str("plugin process frame exceeds the configured limit")
            }
            Self::Codec(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for ProcessTransportError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestMode {
    Unary,
    Stream,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct InflightRequest {
    mode: RequestMode,
    deadline_unix_ms: Option<u64>,
    next_sequence: u64,
}

#[derive(Debug)]
pub struct InflightRequests {
    limit: usize,
    requests: HashMap<String, InflightRequest>,
}

impl Default for InflightRequests {
    fn default() -> Self {
        Self::new()
    }
}

impl InflightRequests {
    pub fn new() -> Self {
        Self {
            limit: MAX_INFLIGHT_REQUESTS,
            requests: HashMap::new(),
        }
    }

    pub fn with_limit(limit: usize) -> Result<Self, ProcessSessionError> {
        if limit == 0 || limit > MAX_INFLIGHT_REQUESTS {
            return Err(ProcessSessionError::InvalidInflightLimit);
        }
        Ok(Self {
            limit,
            requests: HashMap::new(),
        })
    }

    pub fn len(&self) -> usize {
        self.requests.len()
    }

    pub fn is_empty(&self) -> bool {
        self.requests.is_empty()
    }

    pub fn begin(
        &mut self,
        request_id: String,
        mode: RequestMode,
        deadline_unix_ms: Option<u64>,
        now_unix_ms: u64,
    ) -> Result<(), ProcessSessionError> {
        if request_id.is_empty() || request_id.len() > super::rpc_schema::MAX_REQUEST_ID_BYTES {
            return Err(ProcessSessionError::InvalidRequestId);
        }
        if deadline_unix_ms.is_some_and(|deadline| deadline <= now_unix_ms) {
            return Err(ProcessSessionError::DeadlineExpired);
        }
        if self.requests.contains_key(&request_id) {
            return Err(ProcessSessionError::DuplicateRequest);
        }
        if self.requests.len() >= self.limit {
            return Err(ProcessSessionError::InflightLimit);
        }
        self.requests.insert(
            request_id,
            InflightRequest {
                mode,
                deadline_unix_ms,
                next_sequence: 0,
            },
        );
        Ok(())
    }

    pub fn observe(
        &mut self,
        message: ProcessMessage,
        now_unix_ms: u64,
    ) -> Result<ProcessEvent, ProcessSessionError> {
        message
            .validate()
            .map_err(|_| ProcessSessionError::InvalidMessage)?;
        let request_id = message_request_id(&message).to_owned();
        if matches!(message, ProcessMessage::Cancel { .. }) {
            return Ok(ProcessEvent::Cancelled {
                was_active: self.requests.remove(&request_id).is_some(),
                request_id,
            });
        }
        let Some(request) = self.requests.get(&request_id) else {
            return Err(ProcessSessionError::UnknownRequest);
        };
        if request
            .deadline_unix_ms
            .is_some_and(|deadline| deadline <= now_unix_ms)
        {
            self.requests.remove(&request_id);
            return Err(ProcessSessionError::DeadlineExpired);
        }

        match message {
            ProcessMessage::Response { envelope, .. } => {
                if request.mode != RequestMode::Unary {
                    return Err(ProcessSessionError::UnexpectedMessage);
                }
                self.requests.remove(&request_id);
                Ok(ProcessEvent::UnaryResponse {
                    request_id,
                    envelope,
                })
            }
            ProcessMessage::StreamChunk {
                sequence, payload, ..
            } => {
                if request.mode != RequestMode::Stream {
                    return Err(ProcessSessionError::UnexpectedMessage);
                }
                if sequence != request.next_sequence {
                    return Err(ProcessSessionError::InvalidStreamSequence);
                }
                self.requests
                    .get_mut(&request_id)
                    .expect("request checked above")
                    .next_sequence += 1;
                Ok(ProcessEvent::StreamChunk {
                    request_id,
                    sequence,
                    payload,
                })
            }
            ProcessMessage::StreamEnd {
                next_sequence,
                error,
                ..
            } => {
                if request.mode != RequestMode::Stream {
                    return Err(ProcessSessionError::UnexpectedMessage);
                }
                if next_sequence != request.next_sequence {
                    return Err(ProcessSessionError::InvalidStreamSequence);
                }
                self.requests.remove(&request_id);
                Ok(ProcessEvent::StreamEnd {
                    request_id,
                    next_sequence,
                    error,
                })
            }
            ProcessMessage::Request { .. } => Err(ProcessSessionError::UnexpectedMessage),
            ProcessMessage::Cancel { .. } => unreachable!("cancel handled above"),
        }
    }

    pub fn expire(&mut self, now_unix_ms: u64) -> Vec<String> {
        let mut expired = Vec::new();
        self.requests.retain(|request_id, request| {
            let keep = request
                .deadline_unix_ms
                .is_none_or(|deadline| deadline > now_unix_ms);
            if !keep {
                expired.push(request_id.clone());
            }
            keep
        });
        expired.sort();
        expired
    }

    pub fn abort_all(&mut self) -> Vec<String> {
        let mut aborted: Vec<_> = self
            .requests
            .drain()
            .map(|(request_id, _)| request_id)
            .collect();
        aborted.sort();
        aborted
    }
}

pub enum ProcessEvent {
    UnaryResponse {
        request_id: String,
        envelope: Envelope,
    },
    StreamChunk {
        request_id: String,
        sequence: u64,
        payload: Box<RawValue>,
    },
    StreamEnd {
        request_id: String,
        next_sequence: u64,
        error: Option<PluginError>,
    },
    Cancelled {
        request_id: String,
        was_active: bool,
    },
}

impl fmt::Debug for ProcessEvent {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut debug = formatter.debug_struct("ProcessEvent");
        match self {
            Self::UnaryResponse {
                request_id,
                envelope,
            } => debug
                .field("kind", &"unary_response")
                .field("request_id", request_id)
                .field("ok", &envelope.ok)
                .field("has_result", &envelope.result.is_some())
                .field("has_error", &envelope.error.is_some()),
            Self::StreamChunk {
                request_id,
                sequence,
                payload,
            } => debug
                .field("kind", &"stream_chunk")
                .field("request_id", request_id)
                .field("sequence", sequence)
                .field("payload", &"[REDACTED]")
                .field("payload_bytes", &payload.get().len()),
            Self::StreamEnd {
                request_id,
                next_sequence,
                error,
            } => debug
                .field("kind", &"stream_end")
                .field("request_id", request_id)
                .field("next_sequence", next_sequence)
                .field("has_error", &error.is_some()),
            Self::Cancelled {
                request_id,
                was_active,
            } => debug
                .field("kind", &"cancelled")
                .field("request_id", request_id)
                .field("was_active", was_active),
        };
        debug.finish()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProcessSessionError {
    InvalidInflightLimit,
    InvalidRequestId,
    InvalidMessage,
    DuplicateRequest,
    InflightLimit,
    UnknownRequest,
    DeadlineExpired,
    UnexpectedMessage,
    InvalidStreamSequence,
}

impl fmt::Display for ProcessSessionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::InvalidInflightLimit => "plugin inflight limit is invalid",
            Self::InvalidRequestId => "plugin request identifier is invalid",
            Self::InvalidMessage => "plugin process message is invalid",
            Self::DuplicateRequest => "plugin request identifier is already active",
            Self::InflightLimit => "plugin inflight request limit reached",
            Self::UnknownRequest => "plugin message references an inactive request",
            Self::DeadlineExpired => "plugin request deadline expired",
            Self::UnexpectedMessage => "plugin message does not match the request mode",
            Self::InvalidStreamSequence => "plugin stream sequence is invalid",
        };
        formatter.write_str(message)
    }
}

impl std::error::Error for ProcessSessionError {}

fn message_request_id(message: &ProcessMessage) -> &str {
    match message {
        ProcessMessage::Request { request_id, .. }
        | ProcessMessage::Response { request_id, .. }
        | ProcessMessage::Cancel { request_id, .. }
        | ProcessMessage::StreamChunk { request_id, .. }
        | ProcessMessage::StreamEnd { request_id, .. } => request_id,
    }
}
