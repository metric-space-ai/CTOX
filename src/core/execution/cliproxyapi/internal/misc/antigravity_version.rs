// ref: internal/misc/antigravity_version.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::error::Error;
use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, SystemTime};

pub const ANTIGRAVITY_FALLBACK_VERSION: &str = "2.2.1";
pub const ANTIGRAVITY_HUB_PLATFORM: &str = "darwin/arm64";
pub const ANTIGRAVITY_NODE_API_CLIENT_UA: &str = "google-api-nodejs-client/10.3.0";
pub const ANTIGRAVITY_GOOG_API_CLIENT_UA: &str = "gl-node/22.21.1";
pub const ANTIGRAVITY_VERSION_CACHE_TTL: Duration = Duration::from_secs(6 * 60 * 60);
pub const ANTIGRAVITY_FETCH_TIMEOUT: Duration = Duration::from_secs(10);
pub const ANTIGRAVITY_HUB_LATEST_MANIFEST_URL: &str = "https://antigravity-hub-auto-updater-974169037036.us-central1.run.app/manifest/latest-arm64-mac.yml";
const MAX_MANIFEST_BYTES: usize = 4096;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AntigravityManifestRequest {
    pub url: String,
    pub user_agent: &'static str,
    pub cache_control: &'static str,
    pub timeout: Duration,
    pub max_response_bytes: usize,
}

pub type AntigravityManifestFuture<'a> =
    Pin<Box<dyn Future<Output = Result<Vec<u8>, AntigravityVersionError>> + Send + 'a>>;

pub trait AntigravityManifestTransport: Send + Sync {
    fn fetch<'a>(&'a self, request: AntigravityManifestRequest) -> AntigravityManifestFuture<'a>;
}

pub trait AntigravityVersionClock: Send + Sync {
    fn now(&self) -> SystemTime;
}

#[derive(Debug, Default)]
pub struct SystemAntigravityVersionClock;

impl AntigravityVersionClock for SystemAntigravityVersionClock {
    fn now(&self) -> SystemTime {
        SystemTime::now()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AntigravityVersionError {
    Transport(String),
    ManifestTooLarge,
    InvalidManifest,
    InvalidVersion(String),
}

impl fmt::Display for AntigravityVersionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Transport(message) => write!(formatter, "manifest transport failed: {message}"),
            Self::ManifestTooLarge => formatter.write_str("manifest exceeds 4096 bytes"),
            Self::InvalidManifest => formatter.write_str("manifest has no version field"),
            Self::InvalidVersion(version) => {
                write!(formatter, "manifest contains invalid version {version:?}")
            }
        }
    }
}

impl Error for AntigravityVersionError {}

#[derive(Debug)]
struct VersionState {
    version: String,
    expires_at: SystemTime,
}

/// Instance-owned replacement for upstream's mutable package cache and
/// `sync.Once` goroutine. The CTOX supervisor owns the updater and transport;
/// request paths only read this bounded cache.
pub struct AntigravityVersionCache {
    manifest_url: String,
    transport: Arc<dyn AntigravityManifestTransport>,
    clock: Arc<dyn AntigravityVersionClock>,
    state: Mutex<VersionState>,
}

impl AntigravityVersionCache {
    pub fn new(
        manifest_url: impl Into<String>,
        transport: Arc<dyn AntigravityManifestTransport>,
        clock: Arc<dyn AntigravityVersionClock>,
    ) -> Result<Self, AntigravityVersionError> {
        let manifest_url = manifest_url.into();
        if manifest_url.trim().is_empty() {
            return Err(AntigravityVersionError::Transport(
                "manifest URL is empty".into(),
            ));
        }
        Ok(Self {
            manifest_url,
            transport,
            clock,
            state: Mutex::new(VersionState {
                version: ANTIGRAVITY_FALLBACK_VERSION.into(),
                expires_at: SystemTime::UNIX_EPOCH,
            }),
        })
    }

    pub async fn refresh(&self) -> Result<String, AntigravityVersionError> {
        let result = self
            .transport
            .fetch(AntigravityManifestRequest {
                url: self.manifest_url.clone(),
                user_agent: "electron-builder",
                cache_control: "no-cache",
                timeout: ANTIGRAVITY_FETCH_TIMEOUT,
                max_response_bytes: MAX_MANIFEST_BYTES,
            })
            .await
            .and_then(|body| parse_manifest_version(&body));
        let now = self.clock.now();
        let mut state = lock_recover(&self.state);
        match result {
            Ok(version) => {
                state.version.clone_from(&version);
                state.expires_at = now + ANTIGRAVITY_VERSION_CACHE_TTL;
                Ok(version)
            }
            Err(error) => {
                if state.version.is_empty() || now >= state.expires_at {
                    state.version = ANTIGRAVITY_FALLBACK_VERSION.into();
                    state.expires_at = now + ANTIGRAVITY_VERSION_CACHE_TTL;
                }
                Err(error)
            }
        }
    }

    pub fn latest_version(&self) -> String {
        let state = lock_recover(&self.state);
        if !state.version.is_empty() && self.clock.now() < state.expires_at {
            state.version.clone()
        } else {
            ANTIGRAVITY_FALLBACK_VERSION.into()
        }
    }

    pub fn user_agent(&self) -> String {
        format!(
            "antigravity/hub/{} {ANTIGRAVITY_HUB_PLATFORM}",
            self.latest_version()
        )
    }

    pub fn request_user_agent(&self, configured: &str) -> String {
        self.base_user_agent(configured)
    }

    pub fn load_code_assist_user_agent(&self, configured: &str) -> String {
        self.request_user_agent(configured)
    }

    pub fn onboard_user_user_agent(&self, configured: &str) -> String {
        let configured = configured.trim();
        if configured.is_empty() {
            return format!("{} {ANTIGRAVITY_NODE_API_CLIENT_UA}", self.user_agent());
        }
        let lower = configured.to_ascii_lowercase();
        if !is_antigravity_family(&lower) {
            return configured.into();
        }
        if lower.contains("google-api-nodejs-client/") {
            return configured.into();
        }
        format!(
            "{} {ANTIGRAVITY_NODE_API_CLIENT_UA}",
            self.base_user_agent(configured)
        )
    }

    pub fn version_from_user_agent(&self, user_agent: &str) -> String {
        let base = self.base_user_agent(user_agent);
        let lower = base.to_ascii_lowercase();
        let rest = if lower.starts_with("antigravity/hub/") {
            &base["antigravity/hub/".len()..]
        } else if lower.starts_with("antigravity/") {
            &base["antigravity/".len()..]
        } else {
            return self.latest_version();
        };
        rest.split_whitespace()
            .next()
            .filter(|value| !value.is_empty())
            .map_or_else(|| self.latest_version(), str::to_owned)
    }

    pub async fn run_updater(self: Arc<Self>, mut shutdown: tokio::sync::watch::Receiver<bool>) {
        let _ = self.refresh().await;
        let mut interval = tokio::time::interval(ANTIGRAVITY_VERSION_CACHE_TTL / 2);
        interval.tick().await;
        loop {
            tokio::select! {
                _ = interval.tick() => { let _ = self.refresh().await; }
                changed = shutdown.changed() => {
                    if changed.is_err() || *shutdown.borrow() { return; }
                }
            }
        }
    }

    fn base_user_agent(&self, configured: &str) -> String {
        let configured = configured.trim();
        if configured.is_empty() {
            return self.user_agent();
        }
        let lower = configured.to_ascii_lowercase();
        if is_antigravity_family(&lower) {
            if let Some(index) = lower.find(" google-api-nodejs-client/") {
                let trimmed = configured[..index].trim();
                if !trimmed.is_empty() {
                    return trimmed.into();
                }
            }
        }
        configured.into()
    }
}

impl fmt::Debug for AntigravityVersionCache {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AntigravityVersionCache")
            .field("manifest_url", &"[REDACTED]")
            .field("cached_version", &lock_recover(&self.state).version)
            .finish_non_exhaustive()
    }
}

fn is_antigravity_family(lower: &str) -> bool {
    lower.starts_with("antigravity/hub/") || lower.starts_with("antigravity/")
}

fn parse_manifest_version(body: &[u8]) -> Result<String, AntigravityVersionError> {
    if body.len() > MAX_MANIFEST_BYTES {
        return Err(AntigravityVersionError::ManifestTooLarge);
    }
    let body = std::str::from_utf8(body).map_err(|_| AntigravityVersionError::InvalidManifest)?;
    let version = body.lines().find_map(|line| {
        let (key, value) = line.split_once(':')?;
        key.trim().eq_ignore_ascii_case("version").then(|| {
            value
                .trim()
                .trim_matches(|character| matches!(character, '\'' | '"'))
                .to_owned()
        })
    });
    let version = version.ok_or(AntigravityVersionError::InvalidManifest)?;
    if !is_valid_semver(&version) {
        return Err(AntigravityVersionError::InvalidVersion(version));
    }
    Ok(version)
}

fn is_valid_semver(version: &str) -> bool {
    let mut parts = version.split('.');
    (0..3).all(|_| {
        parts
            .next()
            .is_some_and(|part| !part.is_empty() && part.bytes().all(|byte| byte.is_ascii_digit()))
    }) && parts.next().is_none()
}

fn lock_recover<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|error| error.into_inner())
}
