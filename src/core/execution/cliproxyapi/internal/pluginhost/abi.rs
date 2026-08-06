// ref: internal/pluginhost/abi.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// Port-Note: replaces the in-process ABI with CTOX process isolation
// License: MIT (upstream); modifications AGPL-3.0-only

//! Process-isolated replacement for upstream's in-process plugin ABI.

use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;

use serde_json::value::RawValue;
use tokio::sync::mpsc;

use crate::sdk::pluginabi::{Envelope, ABI_VERSION};

pub const PLUGIN_HOST_ABI_VERSION: u32 = ABI_VERSION;

pub type PluginFuture<'a, T> =
    Pin<Box<dyn Future<Output = Result<T, PluginClientError>> + Send + 'a>>;

#[derive(Clone, Debug)]
pub struct PluginCall {
    pub method: String,
    pub payload: Box<RawValue>,
    pub deadline_unix_ms: Option<u64>,
}

#[derive(Debug)]
pub struct PluginStream {
    pub chunks: mpsc::Receiver<Result<Box<RawValue>, PluginClientError>>,
}

pub trait PluginClient: Send + Sync {
    fn call<'a>(&'a self, call: PluginCall) -> PluginFuture<'a, Envelope>;
    fn call_stream<'a>(&'a self, call: PluginCall) -> PluginFuture<'a, PluginStream>;
    fn shutdown<'a>(&'a self) -> PluginFuture<'a, ()>;
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PluginArtifact {
    pub plugin_id: String,
    pub executable: PathBuf,
}

pub trait PluginLoader: Send + Sync {
    fn open<'a>(&'a self, artifact: &'a PluginArtifact) -> PluginFuture<'a, Arc<dyn PluginClient>>;
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PluginClientError {
    Closed,
    Cancelled,
    DeadlineExceeded,
    InvalidRequest,
    InvalidResponse,
    UnsupportedCapability,
    Transport(String),
    Plugin {
        code: String,
        message: String,
        retryable: bool,
        http_status: i32,
    },
}

impl std::fmt::Display for PluginClientError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Closed => formatter.write_str("plugin client is closed"),
            Self::Cancelled => formatter.write_str("plugin call was cancelled"),
            Self::DeadlineExceeded => formatter.write_str("plugin call deadline exceeded"),
            Self::InvalidRequest => formatter.write_str("plugin request is invalid"),
            Self::InvalidResponse => formatter.write_str("plugin response is invalid"),
            Self::UnsupportedCapability => formatter.write_str("plugin capability is unsupported"),
            Self::Transport(message) => write!(formatter, "plugin transport failed: {message}"),
            Self::Plugin { code, message, .. } => write!(formatter, "plugin {code}: {message}"),
        }
    }
}

impl std::error::Error for PluginClientError {}
