// ref: internal/tui/client.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use serde_json::Value;
use std::fmt;
use std::io;
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Method {
    Get,
    Post,
    Put,
    Patch,
    Delete,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ManagementRequest {
    pub method: Method,
    pub path: String,
    pub authorization: Option<String>,
    pub body: Vec<u8>,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ManagementResponse {
    pub status: u16,
    pub body: Vec<u8>,
}
pub trait ManagementTransport: Send + Sync {
    fn request(&self, request: ManagementRequest) -> io::Result<ManagementResponse>;
}

pub struct Client {
    endpoint: String,
    secret: Mutex<String>,
    transport: Arc<dyn ManagementTransport>,
}
impl fmt::Debug for Client {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Client")
            .field("endpoint", &self.endpoint)
            .field(
                "has_secret",
                &!self
                    .secret
                    .lock()
                    .unwrap_or_else(|p| p.into_inner())
                    .is_empty(),
            )
            .finish()
    }
}
impl Client {
    pub fn new(
        endpoint: impl Into<String>,
        secret: impl Into<String>,
        transport: Arc<dyn ManagementTransport>,
    ) -> io::Result<Self> {
        let endpoint = endpoint.into();
        let parsed = url::Url::parse(&endpoint)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidInput, error))?;
        if !matches!(parsed.scheme(), "http" | "https") {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "management endpoint must be http(s)",
            ));
        }
        Ok(Self {
            endpoint: endpoint.trim_end_matches('/').to_owned(),
            secret: Mutex::new(secret.into()),
            transport,
        })
    }
    pub fn set_secret(&self, secret: impl Into<String>) {
        *self.secret.lock().unwrap_or_else(|p| p.into_inner()) = secret.into();
    }
    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }
    pub fn request(&self, method: Method, path: &str, body: Vec<u8>) -> io::Result<Vec<u8>> {
        if !path.starts_with('/') || path.contains("..") {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "management path must be absolute and traversal-free",
            ));
        }
        let secret = self
            .secret
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clone();
        let response = self.transport.request(ManagementRequest {
            method,
            path: path.to_owned(),
            authorization: (!secret.is_empty()).then(|| format!("Bearer {secret}")),
            body,
        })?;
        if !(200..300).contains(&response.status) {
            return Err(io::Error::other(format!(
                "management request failed with status {}",
                response.status
            )));
        }
        Ok(response.body)
    }
    pub fn get_json(&self, path: &str) -> io::Result<Value> {
        serde_json::from_slice(&self.request(Method::Get, path, Vec::new())?)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))
    }
    pub fn get_config(&self) -> io::Result<Value> {
        self.get_json("/v0/management/config")
    }
    pub fn get_auth_files(&self) -> io::Result<Vec<Value>> {
        as_array(self.get_json("/v0/management/auth-files")?, "files")
    }
    pub fn delete_auth_file(&self, name: &str) -> io::Result<()> {
        self.request(
            Method::Delete,
            &format!("/v0/management/auth-files/{}", encode_segment(name)),
            Vec::new(),
        )
        .map(|_| ())
    }
    pub fn patch_auth_file(&self, name: &str, fields: &Value) -> io::Result<()> {
        self.request(
            Method::Patch,
            &format!("/v0/management/auth-files/{}", encode_segment(name)),
            serde_json::to_vec(fields).unwrap_or_default(),
        )
        .map(|_| ())
    }
    pub fn get_api_keys(&self) -> io::Result<Vec<String>> {
        as_array(self.get_json("/v0/management/api-keys")?, "api-keys").map(|values| {
            values
                .into_iter()
                .filter_map(|value| value.as_str().map(str::to_owned))
                .collect()
        })
    }
    pub fn replace_api_keys(&self, keys: &[String]) -> io::Result<()> {
        self.request(
            Method::Put,
            "/v0/management/api-keys",
            serde_json::to_vec(&serde_json::json!({"api-keys": keys})).unwrap_or_default(),
        )
        .map(|_| ())
    }
    pub fn get_logs(&self, after: i64, limit: usize) -> io::Result<(Vec<String>, i64)> {
        let value = self.get_json(&format!("/v0/management/logs?after={after}&limit={limit}"))?;
        let cursor = value.get("cursor").and_then(Value::as_i64).unwrap_or(after);
        let lines = value
            .get("lines")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .map(str::to_owned)
            .collect();
        Ok((lines, cursor))
    }
    pub fn get_auth_status(&self, state: &str) -> io::Result<Value> {
        self.get_json(&format!(
            "/v0/management/oauth/status?state={}",
            encode_segment(state)
        ))
    }
    pub fn cancel_auth_session(&self, state: &str) -> io::Result<()> {
        self.request(
            Method::Delete,
            &format!("/v0/management/oauth/session/{}", encode_segment(state)),
            Vec::new(),
        )
        .map(|_| ())
    }
}
fn encode_segment(value: &str) -> String {
    percent_encoding::utf8_percent_encode(value, percent_encoding::NON_ALPHANUMERIC).to_string()
}
fn as_array(value: Value, key: &str) -> io::Result<Vec<Value>> {
    value
        .get(key)
        .or(Some(&value))
        .and_then(Value::as_array)
        .cloned()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, format!("missing {key} array")))
}
