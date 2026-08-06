// ref: internal/api/handlers/management/oauth_sessions.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::BTreeMap;
use std::fmt;
use std::sync::{Arc, Mutex};

const DEFAULT_SESSION_TTL_MS: i64 = 30 * 60 * 1_000;
const COMPLETED_SESSION_TTL_MS: i64 = 60 * 1_000;
const MAX_OAUTH_STATE_LENGTH: usize = 128;

pub trait ManagementOAuthClock: Send + Sync {
    fn now_ms(&self) -> i64;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ManagementOAuthSessionSource {
    Builtin,
    Plugin,
}

#[derive(Clone, PartialEq)]
pub struct ManagementOAuthSession {
    pub provider: String,
    pub status: String,
    pub source: ManagementOAuthSessionSource,
    pub metadata: BTreeMap<String, serde_json::Value>,
    pub completed: bool,
    pub created_at_ms: i64,
    pub expires_at_ms: i64,
}

impl fmt::Debug for ManagementOAuthSession {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ManagementOAuthSession")
            .field("provider", &self.provider)
            .field("status", &self.status)
            .field("source", &self.source)
            .field("metadata", &"[REDACTED]")
            .field("completed", &self.completed)
            .field("created_at_ms", &self.created_at_ms)
            .field("expires_at_ms", &self.expires_at_ms)
            .finish()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ManagementOAuthSessionError {
    InvalidState,
    UnsupportedProvider,
    SessionExists,
    SessionNotPending,
    StateUnavailable,
}

pub struct ManagementOAuthSessions {
    clock: Arc<dyn ManagementOAuthClock>,
    ttl_ms: i64,
    completed_ttl_ms: i64,
    sessions: Mutex<BTreeMap<String, ManagementOAuthSession>>,
}

impl ManagementOAuthSessions {
    #[must_use]
    pub fn new(clock: Arc<dyn ManagementOAuthClock>) -> Self {
        Self::with_ttl(clock, DEFAULT_SESSION_TTL_MS)
    }

    #[must_use]
    pub fn with_ttl(clock: Arc<dyn ManagementOAuthClock>, ttl_ms: i64) -> Self {
        let ttl_ms = ttl_ms.max(1);
        Self {
            clock,
            ttl_ms,
            completed_ttl_ms: ttl_ms.min(COMPLETED_SESSION_TTL_MS),
            sessions: Mutex::new(BTreeMap::new()),
        }
    }

    pub fn register_builtin(
        &self,
        state: &str,
        provider: &str,
    ) -> Result<(), ManagementOAuthSessionError> {
        let provider = normalize_oauth_provider(provider)?;
        self.register(
            state,
            &provider,
            ManagementOAuthSessionSource::Builtin,
            BTreeMap::new(),
        )
    }

    pub fn register_plugin(
        &self,
        state: &str,
        provider: &str,
        metadata: BTreeMap<String, serde_json::Value>,
    ) -> Result<(), ManagementOAuthSessionError> {
        let provider = normalize_plugin_oauth_provider(provider)?;
        self.register(
            state,
            &provider,
            ManagementOAuthSessionSource::Plugin,
            metadata,
        )
    }

    fn register(
        &self,
        state: &str,
        provider: &str,
        source: ManagementOAuthSessionSource,
        metadata: BTreeMap<String, serde_json::Value>,
    ) -> Result<(), ManagementOAuthSessionError> {
        let state = validate_oauth_state(state)?;
        let now = self.clock.now_ms();
        let mut sessions = self.lock()?;
        purge_expired(&mut sessions, now);
        if sessions.contains_key(state) {
            return Err(ManagementOAuthSessionError::SessionExists);
        }
        sessions.insert(
            state.to_owned(),
            ManagementOAuthSession {
                provider: provider.to_owned(),
                status: String::new(),
                source,
                metadata,
                completed: false,
                created_at_ms: now,
                expires_at_ms: now.saturating_add(self.ttl_ms),
            },
        );
        Ok(())
    }

    pub fn set_error(
        &self,
        state: &str,
        message: &str,
    ) -> Result<bool, ManagementOAuthSessionError> {
        let state = validate_oauth_state(state)?;
        let now = self.clock.now_ms();
        let mut sessions = self.lock()?;
        purge_expired(&mut sessions, now);
        let Some(session) = sessions.get_mut(state) else {
            return Ok(false);
        };
        if session.completed {
            return Ok(false);
        }
        session.status = match message.trim() {
            "" => "Authentication failed".to_owned(),
            message => message.to_owned(),
        };
        session.expires_at_ms = now.saturating_add(self.ttl_ms);
        Ok(true)
    }

    pub fn complete(&self, state: &str) -> Result<bool, ManagementOAuthSessionError> {
        let state = validate_oauth_state(state)?;
        let now = self.clock.now_ms();
        let mut sessions = self.lock()?;
        purge_expired(&mut sessions, now);
        let Some(session) = sessions.get_mut(state) else {
            return Ok(false);
        };
        if session.completed {
            return Ok(false);
        }
        complete_session(session, now, self.completed_ttl_ms);
        Ok(true)
    }

    pub fn complete_provider(
        &self,
        provider: &str,
        source: ManagementOAuthSessionSource,
    ) -> Result<usize, ManagementOAuthSessionError> {
        let provider = match source {
            ManagementOAuthSessionSource::Builtin => normalize_oauth_provider(provider)?,
            ManagementOAuthSessionSource::Plugin => normalize_plugin_oauth_provider(provider)?,
        };
        let now = self.clock.now_ms();
        let mut sessions = self.lock()?;
        purge_expired(&mut sessions, now);
        let mut completed = 0;
        for session in sessions.values_mut() {
            if !session.completed && session.source == source && session.provider == provider {
                complete_session(session, now, self.completed_ttl_ms);
                completed += 1;
            }
        }
        Ok(completed)
    }

    pub fn details(
        &self,
        state: &str,
    ) -> Result<Option<ManagementOAuthSession>, ManagementOAuthSessionError> {
        let state = validate_oauth_state(state)?;
        let now = self.clock.now_ms();
        let mut sessions = self.lock()?;
        purge_expired(&mut sessions, now);
        Ok(sessions.get(state).cloned())
    }

    pub fn visible_status(
        &self,
        state: &str,
    ) -> Result<Option<(String, String)>, ManagementOAuthSessionError> {
        Ok(self
            .details(state)?
            .filter(|session| !session.completed)
            .map(|session| (session.provider, session.status)))
    }

    pub fn is_pending(
        &self,
        state: &str,
        provider: &str,
    ) -> Result<bool, ManagementOAuthSessionError> {
        let provider = normalize_callback_provider(provider)?;
        Ok(self.details(state)?.is_some_and(|session| {
            !session.completed && session.status.is_empty() && session.provider == provider
        }))
    }

    pub fn guard_pending_for_save(
        &self,
        state: &str,
        provider: &str,
    ) -> Result<(), ManagementOAuthSessionError> {
        if self.is_pending(state, provider)? {
            Ok(())
        } else {
            Err(ManagementOAuthSessionError::SessionNotPending)
        }
    }

    pub fn cancel(&self, state: &str) -> Result<bool, ManagementOAuthSessionError> {
        let state = validate_oauth_state(state)?;
        let now = self.clock.now_ms();
        let mut sessions = self.lock()?;
        purge_expired(&mut sessions, now);
        let pending = sessions
            .get(state)
            .is_some_and(|session| !session.completed && session.status.is_empty());
        if pending {
            sessions.remove(state);
        }
        Ok(pending)
    }

    fn lock(
        &self,
    ) -> Result<
        std::sync::MutexGuard<'_, BTreeMap<String, ManagementOAuthSession>>,
        ManagementOAuthSessionError,
    > {
        self.sessions
            .lock()
            .map_err(|_| ManagementOAuthSessionError::StateUnavailable)
    }
}

impl fmt::Debug for ManagementOAuthSessions {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ManagementOAuthSessions")
            .field("ttl_ms", &self.ttl_ms)
            .field("completed_ttl_ms", &self.completed_ttl_ms)
            .finish_non_exhaustive()
    }
}

fn complete_session(session: &mut ManagementOAuthSession, now: i64, completed_ttl_ms: i64) {
    session.status.clear();
    session.metadata.clear();
    session.completed = true;
    session.expires_at_ms = now.saturating_add(completed_ttl_ms);
}

fn purge_expired(sessions: &mut BTreeMap<String, ManagementOAuthSession>, now: i64) {
    sessions.retain(|_, session| session.expires_at_ms >= now);
}

pub fn validate_oauth_state(state: &str) -> Result<&str, ManagementOAuthSessionError> {
    let state = state.trim();
    if state.is_empty()
        || state.len() > MAX_OAUTH_STATE_LENGTH
        || state.contains("..")
        || !state
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
    {
        return Err(ManagementOAuthSessionError::InvalidState);
    }
    Ok(state)
}

pub fn normalize_oauth_provider(provider: &str) -> Result<String, ManagementOAuthSessionError> {
    match provider.trim().to_ascii_lowercase().as_str() {
        "anthropic" | "claude" => Ok("anthropic".to_owned()),
        "codex" | "openai" => Ok("codex".to_owned()),
        "antigravity" | "anti-gravity" => Ok("antigravity".to_owned()),
        "xai" | "x-ai" | "x.ai" | "grok" => Ok("xai".to_owned()),
        _ => Err(ManagementOAuthSessionError::UnsupportedProvider),
    }
}

pub fn normalize_plugin_oauth_provider(
    provider: &str,
) -> Result<String, ManagementOAuthSessionError> {
    let provider = provider.trim().to_ascii_lowercase();
    if provider.is_empty()
        || !provider
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
    {
        return Err(ManagementOAuthSessionError::UnsupportedProvider);
    }
    Ok(provider)
}

fn normalize_callback_provider(provider: &str) -> Result<String, ManagementOAuthSessionError> {
    normalize_oauth_provider(provider).or_else(|_| normalize_plugin_oauth_provider(provider))
}
