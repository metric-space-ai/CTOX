// Origin: CTOX
// Port-Status: adapted_to_ctox
// Port-Note: Unix LocalTransport boundary for isolated plugin processes
// License: AGPL-3.0-only

use std::fmt;
use std::fs;
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::{FileTypeExt, MetadataExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::time::Duration;

use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use subtle::ConstantTimeEq;
use tokio::net::{UnixListener, UnixStream};
use tokio::time::timeout;

use crate::sdk::pluginabi::{Envelope, SCHEMA_VERSION};

use super::process_transport::{read_process_message, write_process_message};
use super::rpc_schema::{
    decode_upstream_json, encode_upstream_json, ProcessMessage, PROCESS_PROTOCOL_VERSION,
};

const SOCKET_NAMESPACE: &str = ".cpa";
const SOCKET_FILE: &str = "s";
const MAX_SOCKET_PATH_BYTES: usize = 100;
const HANDSHAKE_METHOD: &str = "ctox.handshake";
const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(5);

pub struct UnixPluginEndpoint {
    listener: UnixListener,
    socket_path: PathBuf,
    instance_dir: PathBuf,
    socket_identity: (u64, u64),
}

impl UnixPluginEndpoint {
    pub fn bind(runtime_root: &Path, instance_id: &str) -> Result<Self, UnixTransportError> {
        validate_instance_id(instance_id)?;
        let namespace = runtime_root.join(SOCKET_NAMESPACE);
        ensure_private_directory(&namespace)?;
        let instance_dir = namespace.join(instance_id);
        ensure_private_directory(&instance_dir)?;
        let socket_path = instance_dir.join(SOCKET_FILE);
        if socket_path.as_os_str().as_bytes().len() > MAX_SOCKET_PATH_BYTES {
            return Err(UnixTransportError::SocketPathTooLong);
        }
        remove_stale_socket(&socket_path)?;

        let listener = UnixListener::bind(&socket_path).map_err(|_| UnixTransportError::Bind)?;
        fs::set_permissions(&socket_path, fs::Permissions::from_mode(0o600))
            .map_err(|_| UnixTransportError::Permissions)?;
        let metadata = fs::symlink_metadata(&socket_path).map_err(|_| UnixTransportError::Bind)?;
        Ok(Self {
            listener,
            socket_path,
            instance_dir,
            socket_identity: (metadata.dev(), metadata.ino()),
        })
    }

    pub fn socket_path(&self) -> &Path {
        &self.socket_path
    }

    pub fn endpoint_argument(&self) -> &std::ffi::OsStr {
        self.socket_path.as_os_str()
    }

    /// Accepts a same-user Unix-socket peer and verifies protocol/schema/plugin
    /// claims. PID binding remains the future child supervisor's responsibility.
    pub async fn accept_verified(
        &self,
        expected_plugin_id: &str,
        one_shot_token: &[u8],
    ) -> Result<UnixPluginConnection, UnixTransportError> {
        validate_instance_id(expected_plugin_id)?;
        if one_shot_token.len() < 32 {
            return Err(UnixTransportError::Handshake);
        }
        let (mut stream, _) = timeout(HANDSHAKE_TIMEOUT, self.listener.accept())
            .await
            .map_err(|_| UnixTransportError::Timeout)?
            .map_err(|_| UnixTransportError::Accept)?;
        let mut nonce = [0_u8; 24];
        getrandom::fill(&mut nonce).map_err(|_| UnixTransportError::Randomness)?;
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
            .map_err(|_| UnixTransportError::Handshake)?,
        };
        timeout(
            HANDSHAKE_TIMEOUT,
            write_process_message(&mut stream, &request),
        )
        .await
        .map_err(|_| UnixTransportError::Timeout)?
        .map_err(|_| UnixTransportError::Handshake)?;
        let response = timeout(HANDSHAKE_TIMEOUT, read_process_message(&mut stream))
            .await
            .map_err(|_| UnixTransportError::Timeout)?
            .map_err(|_| UnixTransportError::Handshake)?
            .ok_or(UnixTransportError::Handshake)?;
        let ProcessMessage::Response {
            request_id: response_id,
            envelope,
            ..
        } = response
        else {
            return Err(UnixTransportError::Handshake);
        };
        if response_id != request_id || !envelope.ok || envelope.error.is_some() {
            return Err(UnixTransportError::Handshake);
        }
        let result = envelope.result.ok_or(UnixTransportError::Handshake)?;
        let response: HandshakeResponse =
            decode_upstream_json(&result).map_err(|_| UnixTransportError::Handshake)?;
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
            return Err(UnixTransportError::Handshake);
        }
        Ok(UnixPluginConnection {
            stream,
            plugin_id: response.plugin_id,
        })
    }
}

impl Drop for UnixPluginEndpoint {
    fn drop(&mut self) {
        if fs::symlink_metadata(&self.socket_path).is_ok_and(|metadata| {
            metadata.file_type().is_socket()
                && (metadata.dev(), metadata.ino()) == self.socket_identity
        }) {
            let _ = fs::remove_file(&self.socket_path);
        }
        let _ = fs::remove_dir(&self.instance_dir);
    }
}

pub struct UnixPluginConnection {
    stream: UnixStream,
    plugin_id: String,
}

impl UnixPluginConnection {
    pub fn plugin_id(&self) -> &str {
        &self.plugin_id
    }

    pub fn into_stream(self) -> UnixStream {
        self.stream
    }

    pub fn stream_mut(&mut self) -> &mut UnixStream {
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
) -> Result<ProcessMessage, UnixTransportError> {
    validate_instance_id(&plugin_id)?;
    let result = encode_upstream_json(&HandshakeResponse {
        schema_version,
        plugin_id,
        proof,
    })
    .map_err(|_| UnixTransportError::Handshake)?;
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
pub enum UnixTransportError {
    InvalidInstanceId,
    UnsafePath,
    SocketPathTooLong,
    CreateDirectory,
    Permissions,
    Bind,
    Accept,
    Randomness,
    Timeout,
    Handshake,
}

impl fmt::Display for UnixTransportError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::InvalidInstanceId => "plugin instance identifier is invalid",
            Self::UnsafePath => "plugin socket path is unsafe",
            Self::SocketPathTooLong => "plugin Unix socket path is too long",
            Self::CreateDirectory => "plugin socket directory could not be created",
            Self::Permissions => "plugin socket permissions could not be applied",
            Self::Bind => "plugin Unix socket could not be bound",
            Self::Accept => "plugin Unix socket could not accept a peer",
            Self::Randomness => "plugin handshake randomness is unavailable",
            Self::Timeout => "plugin handshake timed out",
            Self::Handshake => "plugin handshake failed",
        };
        formatter.write_str(message)
    }
}

impl std::error::Error for UnixTransportError {}

fn validate_instance_id(instance_id: &str) -> Result<(), UnixTransportError> {
    if instance_id.is_empty()
        || instance_id.len() > 64
        || !instance_id.bytes().all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-' || byte == b'_'
        })
    {
        return Err(UnixTransportError::InvalidInstanceId);
    }
    Ok(())
}

fn ensure_private_directory(path: &Path) -> Result<(), UnixTransportError> {
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_symlink() || !metadata.is_dir() => {
            return Err(UnixTransportError::UnsafePath);
        }
        Ok(_) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            fs::create_dir(path).map_err(|_| UnixTransportError::CreateDirectory)?;
        }
        Err(_) => return Err(UnixTransportError::CreateDirectory),
    }
    fs::set_permissions(path, fs::Permissions::from_mode(0o700))
        .map_err(|_| UnixTransportError::Permissions)
}

fn remove_stale_socket(path: &Path) -> Result<(), UnixTransportError> {
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_socket() => {
            fs::remove_file(path).map_err(|_| UnixTransportError::UnsafePath)
        }
        Ok(_) => Err(UnixTransportError::UnsafePath),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(_) => Err(UnixTransportError::UnsafePath),
    }
}
