// Origin: CTOX
// Port-Status: adapted_to_ctox
// Port-Note: Windows LocalTransport boundary for isolated plugin processes
// License: AGPL-3.0-only

use std::ffi::OsStr;
use std::fmt;
use std::path::Path;
use std::time::Duration;

use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use subtle::ConstantTimeEq;
use tokio::net::windows::named_pipe::{NamedPipeServer, ServerOptions};
use tokio::time::timeout;

use crate::sdk::pluginabi::{Envelope, SCHEMA_VERSION};

use super::process_transport::{read_process_message, write_process_message};
use super::rpc_schema::{
    decode_upstream_json, encode_upstream_json, ProcessMessage, PROCESS_PROTOCOL_VERSION,
};

const HANDSHAKE_METHOD: &str = "ctox.handshake";
const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(5);

pub struct WindowsPluginEndpoint {
    server: Option<NamedPipeServer>,
    pipe_name: String,
}

impl WindowsPluginEndpoint {
    pub fn bind(runtime_root: &Path, instance_id: &str) -> Result<Self, WindowsTransportError> {
        validate_instance_id(instance_id)?;
        if !runtime_root.is_absolute() || !runtime_root.is_dir() {
            return Err(WindowsTransportError::UnsafePath);
        }
        let pipe_name = pipe_name(runtime_root, instance_id);
        let server = ServerOptions::new()
            .first_pipe_instance(true)
            .reject_remote_clients(true)
            .create(&pipe_name)
            .map_err(|_| WindowsTransportError::Bind)?;
        Ok(Self {
            server: Some(server),
            pipe_name,
        })
    }

    pub fn endpoint_argument(&self) -> &OsStr {
        OsStr::new(&self.pipe_name)
    }

    pub async fn accept_verified(
        &mut self,
        expected_plugin_id: &str,
        one_shot_token: &[u8],
    ) -> Result<WindowsPluginConnection, WindowsTransportError> {
        validate_instance_id(expected_plugin_id)?;
        if one_shot_token.len() < 32 {
            return Err(WindowsTransportError::Handshake);
        }
        let mut server = self.server.take().ok_or(WindowsTransportError::Accept)?;
        timeout(HANDSHAKE_TIMEOUT, server.connect())
            .await
            .map_err(|_| WindowsTransportError::Timeout)?
            .map_err(|_| WindowsTransportError::Accept)?;
        let mut nonce = [0_u8; 24];
        getrandom::fill(&mut nonce).map_err(|_| WindowsTransportError::Randomness)?;
        let request_id = format!("handshake-{}", URL_SAFE_NO_PAD.encode(nonce));
        let request = ProcessMessage::Request {
            protocol_version: PROCESS_PROTOCOL_VERSION,
            request_id: request_id.clone(),
            method: HANDSHAKE_METHOD.into(),
            deadline_unix_ms: None,
            payload: encode_upstream_json(&HandshakeRequest {
                schema_version: SCHEMA_VERSION,
                nonce: URL_SAFE_NO_PAD.encode(nonce),
            })
            .map_err(|_| WindowsTransportError::Handshake)?,
        };
        timeout(
            HANDSHAKE_TIMEOUT,
            write_process_message(&mut server, &request),
        )
        .await
        .map_err(|_| WindowsTransportError::Timeout)?
        .map_err(|_| WindowsTransportError::Handshake)?;
        let response = timeout(HANDSHAKE_TIMEOUT, read_process_message(&mut server))
            .await
            .map_err(|_| WindowsTransportError::Timeout)?
            .map_err(|_| WindowsTransportError::Handshake)?
            .ok_or(WindowsTransportError::Handshake)?;
        let ProcessMessage::Response {
            request_id: response_id,
            envelope,
            ..
        } = response
        else {
            return Err(WindowsTransportError::Handshake);
        };
        if response_id != request_id || !envelope.ok || envelope.error.is_some() {
            return Err(WindowsTransportError::Handshake);
        }
        let result = envelope.result.ok_or(WindowsTransportError::Handshake)?;
        let response: HandshakeResponse =
            decode_upstream_json(&result).map_err(|_| WindowsTransportError::Handshake)?;
        let expected_proof = handshake_proof(
            one_shot_token,
            &nonce,
            expected_plugin_id,
            response.schema_version,
        );
        if response.schema_version > SCHEMA_VERSION
            || response.plugin_id != expected_plugin_id
            || response
                .proof
                .as_bytes()
                .ct_eq(expected_proof.as_bytes())
                .unwrap_u8()
                != 1
        {
            return Err(WindowsTransportError::Handshake);
        }
        Ok(WindowsPluginConnection {
            stream: server,
            plugin_id: response.plugin_id,
        })
    }
}

pub struct WindowsPluginConnection {
    stream: NamedPipeServer,
    plugin_id: String,
}

impl WindowsPluginConnection {
    pub fn plugin_id(&self) -> &str {
        &self.plugin_id
    }

    pub fn stream_mut(&mut self) -> &mut NamedPipeServer {
        &mut self.stream
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct HandshakeRequest {
    pub schema_version: u32,
    pub nonce: String,
}

impl Default for HandshakeRequest {
    fn default() -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            nonce: String::new(),
        }
    }
}

#[derive(Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct HandshakeResponse {
    pub schema_version: u32,
    pub plugin_id: String,
    pub proof: String,
}

pub fn handshake_response_message(
    request_id: String,
    plugin_id: String,
    schema_version: u32,
    proof: String,
) -> Result<ProcessMessage, WindowsTransportError> {
    validate_instance_id(&plugin_id)?;
    let result = encode_upstream_json(&HandshakeResponse {
        schema_version,
        plugin_id,
        proof,
    })
    .map_err(|_| WindowsTransportError::Handshake)?;
    Ok(ProcessMessage::Response {
        protocol_version: PROCESS_PROTOCOL_VERSION,
        request_id,
        envelope: Envelope::success(Some(result)),
    })
}

pub fn handshake_proof(
    one_shot_token: &[u8],
    nonce: &[u8],
    plugin_id: &str,
    schema_version: u32,
) -> String {
    let mut digest = Sha256::new();
    digest.update(b"ctox-cliproxyapi-plugin-handshake-v1\0");
    digest.update((one_shot_token.len() as u64).to_be_bytes());
    digest.update(one_shot_token);
    digest.update((nonce.len() as u64).to_be_bytes());
    digest.update(nonce);
    digest.update((plugin_id.len() as u64).to_be_bytes());
    digest.update(plugin_id.as_bytes());
    digest.update(schema_version.to_be_bytes());
    URL_SAFE_NO_PAD.encode(digest.finalize())
}

impl fmt::Debug for HandshakeResponse {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("HandshakeResponse")
            .field("schema_version", &self.schema_version)
            .field("plugin_id", &self.plugin_id)
            .field("proof", &"[REDACTED]")
            .finish()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowsTransportError {
    InvalidInstanceId,
    UnsafePath,
    Bind,
    Accept,
    Randomness,
    Timeout,
    Handshake,
}

impl fmt::Display for WindowsTransportError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::InvalidInstanceId => "plugin instance identifier is invalid",
            Self::UnsafePath => "plugin named-pipe root is unsafe",
            Self::Bind => "plugin named pipe could not be created",
            Self::Accept => "plugin named pipe could not accept a peer",
            Self::Randomness => "plugin handshake randomness is unavailable",
            Self::Timeout => "plugin handshake timed out",
            Self::Handshake => "plugin handshake failed",
        };
        formatter.write_str(message)
    }
}

impl std::error::Error for WindowsTransportError {}

fn validate_instance_id(instance_id: &str) -> Result<(), WindowsTransportError> {
    if instance_id.is_empty()
        || instance_id.len() > 64
        || !instance_id.bytes().all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-' || byte == b'_'
        })
    {
        return Err(WindowsTransportError::InvalidInstanceId);
    }
    Ok(())
}

fn pipe_name(runtime_root: &Path, instance_id: &str) -> String {
    let mut digest = Sha256::new();
    digest.update(b"ctox-cliproxyapi-plugin-pipe-v1\0");
    digest.update(runtime_root.as_os_str().to_string_lossy().as_bytes());
    digest.update(b"\0");
    digest.update(instance_id.as_bytes());
    let digest = digest.finalize();
    let suffix = digest[..12]
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    format!(r"\\.\pipe\ctox-cliproxyapi-{suffix}")
}
