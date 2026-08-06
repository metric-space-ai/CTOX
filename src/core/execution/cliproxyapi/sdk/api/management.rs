// ref: sdk/api/management.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

//! Public, instance-owned management embedding surface.
//!
//! Upstream forwards Gin handlers and process-global OAuth session helpers.
//! CTOX keeps the same six endpoint capabilities but injects the handler,
//! clock, persistence, and route registry.  Constructing this facade never
//! binds a listener, reads environment variables, or creates global state.

use std::collections::BTreeMap;
use std::fmt;
use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};

use serde::Serialize;

use crate::sdk::cliproxy::auth::{PostAuthContext, RequestInfo};

const DEFAULT_OAUTH_SESSION_TTL: Duration = Duration::from_secs(30 * 60);
const DEFAULT_COMPLETED_SESSION_TTL: Duration = Duration::from_secs(60);
const MAX_OAUTH_STATE_LENGTH: usize = 128;

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ManagementConfig {
    pub auth_dir: PathBuf,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ManagementEndpoint {
    RequestAnthropicToken,
    RequestCodexToken,
    RequestAntigravityToken,
    RequestKimiToken,
    GetAuthStatus,
    PostOAuthCallback,
}

impl ManagementEndpoint {
    #[must_use]
    pub const fn method(self) -> &'static str {
        match self {
            Self::PostOAuthCallback => "POST",
            _ => "GET",
        }
    }

    #[must_use]
    pub const fn path(self) -> &'static str {
        match self {
            Self::RequestAnthropicToken => "/v0/management/anthropic-auth-url",
            Self::RequestCodexToken => "/v0/management/codex-auth-url",
            Self::RequestAntigravityToken => "/v0/management/antigravity-auth-url",
            Self::RequestKimiToken => "/v0/management/kimi-auth-url",
            Self::GetAuthStatus => "/v0/management/get-auth-status",
            Self::PostOAuthCallback => "/v0/management/oauth-callback",
        }
    }
}

pub const MANAGEMENT_TOKEN_ENDPOINTS: [ManagementEndpoint; 6] = [
    ManagementEndpoint::RequestAnthropicToken,
    ManagementEndpoint::RequestCodexToken,
    ManagementEndpoint::RequestAntigravityToken,
    ManagementEndpoint::RequestKimiToken,
    ManagementEndpoint::GetAuthStatus,
    ManagementEndpoint::PostOAuthCallback,
];

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ManagementRequest {
    pub query: BTreeMap<String, Vec<String>>,
    pub headers: BTreeMap<String, Vec<String>>,
    pub body: Vec<u8>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ManagementResponse {
    pub status: u16,
    pub headers: BTreeMap<String, String>,
    pub body: Vec<u8>,
}

impl ManagementResponse {
    #[must_use]
    pub fn error(status: u16, message: &str) -> Self {
        Self {
            status,
            headers: BTreeMap::from([("content-type".to_owned(), "application/json".to_owned())]),
            body: serde_json::to_vec(&serde_json::json!({"error": message}))
                .unwrap_or_else(|_| b"{}".to_vec()),
        }
    }
}

pub trait ManagementEndpointHandler: Send + Sync {
    fn handle(
        &self,
        endpoint: ManagementEndpoint,
        request: &ManagementRequest,
    ) -> ManagementResponse;
}

pub trait ManagementRouteRegistry {
    type Error;

    fn register(
        &mut self,
        method: &'static str,
        path: &'static str,
        endpoint: ManagementEndpoint,
    ) -> Result<(), Self::Error>;
}

#[derive(Clone)]
pub struct Handler {
    config: ManagementConfig,
    config_file_path: Option<PathBuf>,
    endpoint_handler: Arc<dyn ManagementEndpointHandler>,
}

impl fmt::Debug for Handler {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Handler")
            .field("config", &self.config)
            .field("config_file_path", &self.config_file_path)
            .field("has_endpoint_handler", &true)
            .finish()
    }
}

impl Handler {
    #[must_use]
    pub fn new(
        config: ManagementConfig,
        config_file_path: Option<PathBuf>,
        endpoint_handler: Arc<dyn ManagementEndpointHandler>,
    ) -> Self {
        Self {
            config,
            config_file_path,
            endpoint_handler,
        }
    }

    #[must_use]
    pub fn config(&self) -> &ManagementConfig {
        &self.config
    }

    #[must_use]
    pub fn config_file_path(&self) -> Option<&Path> {
        self.config_file_path.as_deref()
    }

    pub fn dispatch(
        &self,
        endpoint: ManagementEndpoint,
        request: &ManagementRequest,
    ) -> ManagementResponse {
        self.endpoint_handler.handle(endpoint, request)
    }

    pub fn register_token_routes<R>(&self, registry: &mut R) -> Result<(), R::Error>
    where
        R: ManagementRouteRegistry,
    {
        for endpoint in MANAGEMENT_TOKEN_ENDPOINTS {
            registry.register(endpoint.method(), endpoint.path(), endpoint)?;
        }
        Ok(())
    }
}

#[must_use]
pub fn new_handler(
    config: ManagementConfig,
    config_file_path: impl Into<PathBuf>,
    endpoint_handler: Arc<dyn ManagementEndpointHandler>,
) -> Handler {
    Handler::new(config, Some(config_file_path.into()), endpoint_handler)
}

#[must_use]
pub fn new_handler_without_config_file_path(
    config: ManagementConfig,
    endpoint_handler: Arc<dyn ManagementEndpointHandler>,
) -> Handler {
    Handler::new(config, None, endpoint_handler)
}

pub trait ManagementTokenRequester: Send + Sync {
    fn request_anthropic_token(&self, request: &ManagementRequest) -> ManagementResponse;
    fn request_codex_token(&self, request: &ManagementRequest) -> ManagementResponse;
    fn request_antigravity_token(&self, request: &ManagementRequest) -> ManagementResponse;
    fn request_kimi_token(&self, request: &ManagementRequest) -> ManagementResponse;
    fn get_auth_status(&self, request: &ManagementRequest) -> ManagementResponse;
    fn post_oauth_callback(&self, request: &ManagementRequest) -> ManagementResponse;
}

#[derive(Clone, Debug)]
pub struct LimitedManagementTokenRequester {
    handler: Handler,
}

impl ManagementTokenRequester for LimitedManagementTokenRequester {
    fn request_anthropic_token(&self, request: &ManagementRequest) -> ManagementResponse {
        self.handler
            .dispatch(ManagementEndpoint::RequestAnthropicToken, request)
    }

    fn request_codex_token(&self, request: &ManagementRequest) -> ManagementResponse {
        self.handler
            .dispatch(ManagementEndpoint::RequestCodexToken, request)
    }

    fn request_antigravity_token(&self, request: &ManagementRequest) -> ManagementResponse {
        self.handler
            .dispatch(ManagementEndpoint::RequestAntigravityToken, request)
    }

    fn request_kimi_token(&self, request: &ManagementRequest) -> ManagementResponse {
        self.handler
            .dispatch(ManagementEndpoint::RequestKimiToken, request)
    }

    fn get_auth_status(&self, request: &ManagementRequest) -> ManagementResponse {
        self.handler
            .dispatch(ManagementEndpoint::GetAuthStatus, request)
    }

    fn post_oauth_callback(&self, request: &ManagementRequest) -> ManagementResponse {
        self.handler
            .dispatch(ManagementEndpoint::PostOAuthCallback, request)
    }
}

#[must_use]
pub fn new_management_token_requester(handler: Handler) -> LimitedManagementTokenRequester {
    LimitedManagementTokenRequester { handler }
}

pub trait ManagementPersistence: Send + Sync {
    fn write_config(&self, path: &Path, data: &[u8]) -> io::Result<()>;
    fn write_oauth_callback(&self, path: &Path, data: &[u8]) -> io::Result<()>;
}

#[derive(Clone, Copy, Debug, Default)]
pub struct FileManagementPersistence;

impl ManagementPersistence for FileManagementPersistence {
    fn write_config(&self, path: &Path, data: &[u8]) -> io::Result<()> {
        let mut file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;
        file.write_all(data)?;
        file.sync_all()?;
        file.flush()
    }

    fn write_oauth_callback(&self, path: &Path, data: &[u8]) -> io::Result<()> {
        let parent = path.parent().ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "callback path has no parent")
        })?;
        fs::create_dir_all(parent)?;
        let mut options = OpenOptions::new();
        options.write(true).create(true).truncate(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let mut file = options.open(path)?;
        file.write_all(data)?;
        file.sync_all()?;
        file.flush()
    }
}

pub fn write_config(
    persistence: &dyn ManagementPersistence,
    path: &Path,
    data: &[u8],
) -> io::Result<()> {
    let normalized = normalize_comment_indentation(data);
    persistence.write_config(path, &normalized)
}

fn normalize_comment_indentation(data: &[u8]) -> Vec<u8> {
    let mut output = Vec::with_capacity(data.len());
    for (index, line) in data.split(|byte| *byte == b'\n').enumerate() {
        if index != 0 {
            output.push(b'\n');
        }
        let first_non_whitespace = line
            .iter()
            .position(|byte| !matches!(byte, b' ' | b'\t'))
            .unwrap_or(line.len());
        if line.get(first_non_whitespace) == Some(&b'#') {
            output.extend_from_slice(&line[first_non_whitespace..]);
        } else {
            output.extend_from_slice(line);
        }
    }
    output
}

pub trait OAuthClock: Send + Sync {
    fn now(&self) -> SystemTime;
}

#[derive(Clone, Copy, Debug, Default)]
pub struct SystemOAuthClock;

impl OAuthClock for SystemOAuthClock {
    fn now(&self) -> SystemTime {
        SystemTime::now()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct OAuthSession {
    provider: String,
    status: String,
    completed: bool,
    expires_at: SystemTime,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OAuthSessionView {
    pub provider: String,
    pub status: String,
}

pub struct OAuthSessionStore {
    clock: Arc<dyn OAuthClock>,
    ttl: Duration,
    completed_ttl: Duration,
    sessions: Mutex<BTreeMap<String, OAuthSession>>,
}

impl fmt::Debug for OAuthSessionStore {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OAuthSessionStore")
            .field("ttl", &self.ttl)
            .field("completed_ttl", &self.completed_ttl)
            .finish_non_exhaustive()
    }
}

impl OAuthSessionStore {
    #[must_use]
    pub fn new(clock: Arc<dyn OAuthClock>, ttl: Duration) -> Self {
        let ttl = if ttl.is_zero() {
            DEFAULT_OAUTH_SESSION_TTL
        } else {
            ttl
        };
        Self {
            clock,
            ttl,
            completed_ttl: ttl.min(DEFAULT_COMPLETED_SESSION_TTL),
            sessions: Mutex::new(BTreeMap::new()),
        }
    }

    pub fn register(&self, state: &str, provider: &str) {
        let state = state.trim();
        let provider = provider.trim().to_ascii_lowercase();
        if state.is_empty() || provider.is_empty() {
            return;
        }
        let now = self.clock.now();
        let mut sessions = self
            .sessions
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        purge_expired(&mut sessions, now);
        sessions.insert(
            state.to_owned(),
            OAuthSession {
                provider,
                status: String::new(),
                completed: false,
                expires_at: now + self.ttl,
            },
        );
    }

    pub fn set_error(&self, state: &str, message: &str) {
        let state = state.trim();
        if state.is_empty() {
            return;
        }
        let now = self.clock.now();
        let mut sessions = self
            .sessions
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        purge_expired(&mut sessions, now);
        if let Some(session) = sessions.get_mut(state).filter(|session| !session.completed) {
            let message = message.trim();
            session.status = if message.is_empty() {
                "Authentication failed".to_owned()
            } else {
                message.to_owned()
            };
            session.expires_at = now + self.ttl;
        }
    }

    pub fn complete(&self, state: &str) {
        let state = state.trim();
        if state.is_empty() {
            return;
        }
        let now = self.clock.now();
        let mut sessions = self
            .sessions
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        purge_expired(&mut sessions, now);
        if let Some(session) = sessions.get_mut(state).filter(|session| !session.completed) {
            session.status.clear();
            session.completed = true;
            session.expires_at = now + self.completed_ttl;
        }
    }

    pub fn complete_by_provider(&self, provider: &str) -> usize {
        let provider = provider.trim();
        if provider.is_empty() {
            return 0;
        }
        let now = self.clock.now();
        let mut sessions = self
            .sessions
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        purge_expired(&mut sessions, now);
        let mut count = 0;
        for session in sessions.values_mut() {
            if !session.completed && session.provider.eq_ignore_ascii_case(provider) {
                session.status.clear();
                session.completed = true;
                session.expires_at = now + self.completed_ttl;
                count += 1;
            }
        }
        count
    }

    #[must_use]
    pub fn get(&self, state: &str) -> Option<OAuthSessionView> {
        let now = self.clock.now();
        let mut sessions = self
            .sessions
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        purge_expired(&mut sessions, now);
        sessions
            .get(state.trim())
            .filter(|session| !session.completed)
            .map(|session| OAuthSessionView {
                provider: session.provider.clone(),
                status: session.status.clone(),
            })
    }

    #[must_use]
    pub fn is_pending(&self, state: &str, provider: &str) -> bool {
        let now = self.clock.now();
        let mut sessions = self
            .sessions
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        purge_expired(&mut sessions, now);
        sessions.get(state.trim()).is_some_and(|session| {
            !session.completed
                && session.status.is_empty()
                && (provider.trim().is_empty()
                    || session.provider.eq_ignore_ascii_case(provider.trim()))
        })
    }
}

fn purge_expired(sessions: &mut BTreeMap<String, OAuthSession>, now: SystemTime) {
    sessions.retain(|_, session| now <= session.expires_at);
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OAuthManagementError {
    InvalidState,
    UnsupportedProvider,
    SessionNotPending,
    EmptyAuthDirectory,
    Persistence,
}

impl fmt::Display for OAuthManagementError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::InvalidState => "invalid oauth state",
            Self::UnsupportedProvider => "unsupported oauth provider",
            Self::SessionNotPending => "oauth session is not pending",
            Self::EmptyAuthDirectory => "auth dir is empty",
            Self::Persistence => "oauth callback persistence failed",
        })
    }
}

impl std::error::Error for OAuthManagementError {}

pub fn validate_oauth_state(state: &str) -> Result<(), OAuthManagementError> {
    let state = state.trim();
    if state.is_empty()
        || state.len() > MAX_OAUTH_STATE_LENGTH
        || state.contains('/')
        || state.contains('\\')
        || state.contains("..")
        || !state.chars().all(|character| {
            character.is_ascii_alphanumeric() || matches!(character, '-' | '_' | '.')
        })
    {
        return Err(OAuthManagementError::InvalidState);
    }
    Ok(())
}

pub fn normalize_oauth_provider(provider: &str) -> Result<&'static str, OAuthManagementError> {
    match provider.trim().to_ascii_lowercase().as_str() {
        "anthropic" | "claude" => Ok("anthropic"),
        "codex" | "openai" => Ok("codex"),
        "antigravity" | "anti-gravity" => Ok("antigravity"),
        "xai" | "x-ai" | "x.ai" | "grok" => Ok("xai"),
        _ => Err(OAuthManagementError::UnsupportedProvider),
    }
}

fn normalize_callback_provider(provider: &str) -> Result<String, OAuthManagementError> {
    if let Ok(provider) = normalize_oauth_provider(provider) {
        return Ok(provider.to_owned());
    }
    let provider = provider.trim().to_ascii_lowercase();
    if provider.is_empty()
        || !provider.chars().all(|character| {
            character.is_ascii_lowercase() || character.is_ascii_digit() || character == '-'
        })
    {
        return Err(OAuthManagementError::UnsupportedProvider);
    }
    Ok(provider)
}

#[derive(Serialize)]
struct OAuthCallbackPayload<'a> {
    code: &'a str,
    state: &'a str,
    error: &'a str,
}

pub fn write_oauth_callback_file(
    persistence: &dyn ManagementPersistence,
    auth_dir: &Path,
    provider: &str,
    state: &str,
    code: &str,
    error_message: &str,
) -> Result<PathBuf, OAuthManagementError> {
    if auth_dir.as_os_str().is_empty() {
        return Err(OAuthManagementError::EmptyAuthDirectory);
    }
    let provider = normalize_callback_provider(provider)?;
    validate_oauth_state(state)?;
    let state = state.trim();
    let path = auth_dir.join(format!(".oauth-{provider}-{state}.oauth"));
    let payload = serde_json::to_vec(&OAuthCallbackPayload {
        code: code.trim(),
        state,
        error: error_message.trim(),
    })
    .map_err(|_| OAuthManagementError::Persistence)?;
    persistence
        .write_oauth_callback(&path, &payload)
        .map_err(|_| OAuthManagementError::Persistence)?;
    Ok(path)
}

pub fn write_oauth_callback_file_for_pending_session(
    persistence: &dyn ManagementPersistence,
    sessions: &OAuthSessionStore,
    auth_dir: &Path,
    provider: &str,
    state: &str,
    code: &str,
    error_message: &str,
) -> Result<PathBuf, OAuthManagementError> {
    let provider = normalize_callback_provider(provider)?;
    if !sessions.is_pending(state, &provider) {
        return Err(OAuthManagementError::SessionNotPending);
    }
    write_oauth_callback_file(persistence, auth_dir, &provider, state, code, error_message)
}

#[must_use]
pub fn populate_auth_context(
    context: &PostAuthContext,
    request: &ManagementRequest,
) -> PostAuthContext {
    context.with_request_info(RequestInfo {
        query: request.query.clone(),
        headers: request.headers.clone(),
    })
}
