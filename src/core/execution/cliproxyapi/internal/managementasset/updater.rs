// ref: internal/managementasset/updater.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

//! Instance-scoped management-panel asset updater.
//!
//! Upstream keeps configuration, throttling, scheduling and credentials in
//! process globals.  CTOX injects all four through this object instead.  The
//! wire format, release selection, digest handling, fallback behavior and
//! atomic replacement semantics remain equivalent to the pinned Go source.

use crate::internal::httpfetch::{get_bytes, Headers, HttpDoer, HttpFetchError};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fmt;
use std::fs::{self, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;
use tokio::time::Instant;
use url::Url;

pub const DEFAULT_MANAGEMENT_RELEASE_URL: &str =
    "https://api.github.com/repos/router-for-me/Cli-Proxy-API-Management-Center/releases/latest";
pub const DEFAULT_MANAGEMENT_FALLBACK_URL: &str = "https://cpamc.router-for.me/";
pub const MANAGEMENT_FILE_NAME: &str = "management.html";
pub const HTTP_USER_AGENT: &str = "CLIProxyAPI-management-updater";
pub const MANAGEMENT_SYNC_MIN_INTERVAL: Duration = Duration::from_secs(30);
pub const UPDATE_CHECK_INTERVAL: Duration = Duration::from_secs(3 * 60 * 60);
pub const MAX_ASSET_DOWNLOAD_SIZE: usize = 50 << 20;

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ManagementConfig {
    pub home_enabled: bool,
    pub disable_control_panel: bool,
    pub disable_auto_update_panel: bool,
    pub proxy_url: String,
    pub panel_github_repository: String,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct AssetPaths {
    /// Typed replacement for upstream's `MANAGEMENT_STATIC_PATH` environment
    /// variable. It may name either the directory or `management.html`.
    pub static_override: Option<PathBuf>,
    /// Typed replacement for upstream's process-global writable directory.
    pub writable_path: Option<PathBuf>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct FetchAuth {
    /// Repository URL associated with this credential.
    pub git_url: String,
    /// Secret is only sent to `api.github.com`, and only when `git_url` is a
    /// GitHub HTTPS URL. The caller owns storage and redaction.
    pub github_token: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HttpPolicy {
    allowed_hosts: BTreeSet<String>,
}

impl Default for HttpPolicy {
    fn default() -> Self {
        Self::new([
            "api.github.com",
            "github.com",
            "objects.githubusercontent.com",
            "github-releases.githubusercontent.com",
            "release-assets.githubusercontent.com",
            "cpamc.router-for.me",
        ])
    }
}

impl HttpPolicy {
    pub fn new(hosts: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self {
            allowed_hosts: hosts
                .into_iter()
                .map(Into::into)
                .map(|host: String| host.trim().to_ascii_lowercase())
                .filter(|host| !host.is_empty())
                .collect(),
        }
    }

    fn validate(&self, value: &str) -> Result<Url, UpdateError> {
        let parsed = Url::parse(value).map_err(|_| UpdateError::UnsafeUrl)?;
        if parsed.scheme() != "https"
            || !parsed.username().is_empty()
            || parsed.password().is_some()
            || parsed.port().is_some()
            || parsed
                .host_str()
                .is_none_or(|host| !self.allowed_hosts.contains(&host.to_ascii_lowercase()))
        {
            return Err(UpdateError::UnsafeUrl);
        }
        Ok(parsed)
    }
}

#[derive(Debug)]
pub enum UpdateError {
    InvalidStaticDirectory,
    UnsafePath,
    UnsafeUrl,
    Fetch(HttpFetchError),
    Decode(serde_json::Error),
    AssetMissing,
    DigestMismatch { expected: String, actual: String },
    Io(io::Error),
}

impl fmt::Display for UpdateError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidStaticDirectory => formatter.write_str("empty static directory"),
            Self::UnsafePath => formatter.write_str("unsafe management asset path"),
            Self::UnsafeUrl => formatter.write_str("management asset URL rejected by policy"),
            Self::Fetch(error) => write!(formatter, "fetch management asset: {error}"),
            Self::Decode(error) => write!(formatter, "decode release response: {error}"),
            Self::AssetMissing => write!(
                formatter,
                "management asset {MANAGEMENT_FILE_NAME} not found in latest release"
            ),
            Self::DigestMismatch { expected, actual } => write!(
                formatter,
                "management asset digest mismatch: expected {expected} got {actual}"
            ),
            Self::Io(error) => write!(formatter, "management asset I/O: {error}"),
        }
    }
}

impl std::error::Error for UpdateError {}

impl From<HttpFetchError> for UpdateError {
    fn from(value: HttpFetchError) -> Self {
        Self::Fetch(value)
    }
}

impl From<io::Error> for UpdateError {
    fn from(value: io::Error) -> Self {
        Self::Io(value)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SyncOutcome {
    Throttled { asset_exists: bool },
    AlreadyCurrent,
    Updated { sha256: String },
    UpdatedFromFallback { sha256: String },
    RetainedExisting { reason: String },
    Missing { reason: String },
}

impl SyncOutcome {
    #[must_use]
    pub fn asset_exists(&self) -> bool {
        !matches!(
            self,
            Self::Throttled {
                asset_exists: false
            } | Self::Missing { .. }
        )
    }
}

#[derive(Debug, Deserialize)]
struct ReleaseAsset {
    name: String,
    browser_download_url: String,
    #[serde(default)]
    digest: String,
}

#[derive(Debug, Deserialize)]
struct ReleaseResponse {
    assets: Vec<ReleaseAsset>,
}

pub trait ManagementConfigSource: Send + Sync {
    fn current(&self) -> Option<ManagementConfig>;
}

/// Instance-owned selector for the transport matching the current typed proxy
/// configuration. Naming the boundary keeps the updater API readable while
/// preserving the injected, per-attempt selection contract.
pub type ManagementClientSelector<C> = Arc<dyn Fn(&str) -> Arc<C> + Send + Sync>;

/// One updater instance corresponds to one CTOX runtime instance.
pub struct ManagementAssetUpdater<C> {
    client: Arc<C>,
    client_selector: Option<ManagementClientSelector<C>>,
    auth: FetchAuth,
    policy: HttpPolicy,
    fallback_url: Option<String>,
    sync_min_interval: Duration,
    update_check_interval: Duration,
    last_update_check: Mutex<Option<Instant>>,
    sync_gate: Mutex<()>,
}

impl<C: HttpDoer> ManagementAssetUpdater<C> {
    #[must_use]
    pub fn new(client: Arc<C>, auth: FetchAuth) -> Self {
        Self {
            client,
            client_selector: None,
            auth,
            policy: HttpPolicy::default(),
            fallback_url: Some(DEFAULT_MANAGEMENT_FALLBACK_URL.to_owned()),
            sync_min_interval: MANAGEMENT_SYNC_MIN_INTERVAL,
            update_check_interval: UPDATE_CHECK_INTERVAL,
            last_update_check: Mutex::new(None),
            sync_gate: Mutex::new(()),
        }
    }

    #[must_use]
    pub fn with_policy(mut self, policy: HttpPolicy) -> Self {
        self.policy = policy;
        self
    }

    /// Injects the typed proxy/client boundary used on every attempt. This is
    /// how a hot-reloaded `proxy_url` selects a new client without environment
    /// mutation or an updater-owned transport implementation.
    #[must_use]
    pub fn with_client_selector(mut self, selector: ManagementClientSelector<C>) -> Self {
        self.client_selector = Some(selector);
        self
    }

    #[must_use]
    pub fn with_fallback_url(mut self, fallback_url: Option<String>) -> Self {
        self.fallback_url = fallback_url;
        self
    }

    #[must_use]
    pub fn with_intervals(mut self, sync_min: Duration, update_check: Duration) -> Self {
        self.sync_min_interval = sync_min;
        self.update_check_interval = update_check;
        self
    }

    /// Coalesces concurrent attempts, applies instance-local throttling and
    /// returns whether the asset exists after the attempt.
    pub async fn ensure_latest(
        &self,
        static_dir: &Path,
        proxy_url: &str,
        panel_repository: &str,
    ) -> SyncOutcome {
        if static_dir.as_os_str().is_empty() {
            return SyncOutcome::Missing {
                reason: UpdateError::InvalidStaticDirectory.to_string(),
            };
        }
        let _sync = self.sync_gate.lock().await;
        let local_path = static_dir.join(MANAGEMENT_FILE_NAME);
        let exists = regular_asset_exists(&local_path);

        let mut last = self.last_update_check.lock().await;
        let now = Instant::now();
        if last.is_some_and(|previous| now.duration_since(previous) < self.sync_min_interval) {
            return SyncOutcome::Throttled {
                asset_exists: exists,
            };
        }
        *last = Some(now);
        drop(last);

        let client = self.client_for_proxy(proxy_url);
        match self
            .sync_once(client.as_ref(), static_dir, &local_path, panel_repository)
            .await
        {
            Ok(outcome) => outcome,
            Err(error) if exists => SyncOutcome::RetainedExisting {
                reason: error.to_string(),
            },
            Err(primary) => match self.fetch_fallback(client.as_ref(), &local_path).await {
                Ok(outcome) => outcome,
                Err(fallback) => SyncOutcome::Missing {
                    reason: format!("{primary}; fallback: {fallback}"),
                },
            },
        }
    }

    async fn sync_once(
        &self,
        client: &C,
        static_dir: &Path,
        local_path: &Path,
        panel_repository: &str,
    ) -> Result<SyncOutcome, UpdateError> {
        prepare_static_directory(static_dir, local_path)?;
        let release_url = resolve_release_url(panel_repository);
        self.policy.validate(&release_url)?;
        let local_hash = file_sha256(local_path).ok();
        let (asset, remote_hash) = self.fetch_latest_asset(client, &release_url).await?;

        if remote_hash
            .as_ref()
            .zip(local_hash.as_ref())
            .is_some_and(|(remote, local)| remote.eq_ignore_ascii_case(local))
        {
            return Ok(SyncOutcome::AlreadyCurrent);
        }

        let (data, downloaded_hash) = self
            .download_asset(client, &asset.browser_download_url)
            .await?;
        if remote_hash
            .as_ref()
            .is_some_and(|expected| !expected.eq_ignore_ascii_case(&downloaded_hash))
        {
            return Err(UpdateError::DigestMismatch {
                expected: remote_hash.unwrap_or_default(),
                actual: downloaded_hash,
            });
        }
        atomic_write_file(local_path, &data)?;
        Ok(SyncOutcome::Updated {
            sha256: downloaded_hash,
        })
    }

    async fn fetch_latest_asset(
        &self,
        client: &C,
        release_url: &str,
    ) -> Result<(ReleaseAsset, Option<String>), UpdateError> {
        let parsed = self.policy.validate(release_url)?;
        let mut headers = Headers::from([
            (
                "Accept".to_owned(),
                "application/vnd.github+json".to_owned(),
            ),
            ("User-Agent".to_owned(), HTTP_USER_AGENT.to_owned()),
        ]);
        if parsed.host_str() == Some("api.github.com") && self.auth.authorizes_github() {
            if let Some(token) = self
                .auth
                .github_token
                .as_deref()
                .map(str::trim)
                .filter(|token| !token.is_empty())
            {
                headers.insert("Authorization".to_owned(), format!("Bearer {token}"));
            }
        }
        let data = get_bytes(client, release_url, &headers, 0).await?;
        let release: ReleaseResponse =
            serde_json::from_slice(&data).map_err(UpdateError::Decode)?;
        let asset = release
            .assets
            .into_iter()
            .find(|asset| asset.name.eq_ignore_ascii_case(MANAGEMENT_FILE_NAME))
            .ok_or(UpdateError::AssetMissing)?;
        let digest = parse_digest(&asset.digest);
        Ok((asset, (!digest.is_empty()).then_some(digest)))
    }

    async fn download_asset(
        &self,
        client: &C,
        download_url: &str,
    ) -> Result<(Vec<u8>, String), UpdateError> {
        self.policy.validate(download_url)?;
        let headers = Headers::from([("User-Agent".to_owned(), HTTP_USER_AGENT.to_owned())]);
        let data = get_bytes(client, download_url, &headers, MAX_ASSET_DOWNLOAD_SIZE).await?;
        let hash = sha256_hex(&data);
        Ok((data, hash))
    }

    async fn fetch_fallback(
        &self,
        client: &C,
        local_path: &Path,
    ) -> Result<SyncOutcome, UpdateError> {
        let fallback = self
            .fallback_url
            .as_deref()
            .ok_or(UpdateError::AssetMissing)?;
        let (data, hash) = self.download_asset(client, fallback).await?;
        atomic_write_file(local_path, &data)?;
        Ok(SyncOutcome::UpdatedFromFallback { sha256: hash })
    }

    /// Runs the upstream immediate check plus periodic checks. Configuration is
    /// loaded afresh for every iteration, preserving hot reload without a
    /// process-global pointer. Dropping or resolving `shutdown` stops the loop.
    pub async fn run_auto_updater<S, F>(
        &self,
        config_path: &Path,
        paths: &AssetPaths,
        source: &S,
        shutdown: F,
    ) where
        S: ManagementConfigSource,
        F: std::future::Future<Output = ()>,
    {
        if config_path.as_os_str().is_empty() {
            return;
        }
        tokio::pin!(shutdown);
        let mut ticker = tokio::time::interval(self.update_check_interval);
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        loop {
            tokio::select! {
                () = &mut shutdown => return,
                _ = ticker.tick() => {
                    let Some(config) = source.current() else { continue };
                    if auto_update_skip_reason(Some(&config)).1 { continue; }
                    let Some(directory) = static_dir(config_path, paths) else { continue; };
                    let _ = self.ensure_latest(
                        &directory,
                        &config.proxy_url,
                        &config.panel_github_repository,
                    ).await;
                }
            }
        }
    }

    fn client_for_proxy(&self, proxy_url: &str) -> Arc<C> {
        self.client_selector.as_ref().map_or_else(
            || Arc::clone(&self.client),
            |selector| selector(proxy_url.trim()),
        )
    }
}

impl FetchAuth {
    fn authorizes_github(&self) -> bool {
        Url::parse(self.git_url.trim()).is_ok_and(|url| {
            url.scheme() == "https"
                && url
                    .host_str()
                    .is_some_and(|host| host.eq_ignore_ascii_case("github.com"))
        })
    }
}

#[must_use]
pub fn auto_update_skip_reason(config: Option<&ManagementConfig>) -> (&'static str, bool) {
    let Some(config) = config else {
        return ("config not yet available", true);
    };
    if config.home_enabled {
        return ("cluster mode enabled", true);
    }
    if config.disable_control_panel {
        return ("control panel disabled", true);
    }
    if config.disable_auto_update_panel {
        return ("disable-auto-update-panel is enabled", true);
    }
    ("", false)
}

#[must_use]
pub fn static_dir(config_file_path: &Path, paths: &AssetPaths) -> Option<PathBuf> {
    if let Some(override_path) = paths.static_override.as_deref() {
        return Some(if is_management_file(override_path) {
            override_path.parent()?.to_path_buf()
        } else {
            override_path.to_path_buf()
        });
    }
    if let Some(writable) = paths.writable_path.as_deref() {
        return Some(writable.join("static"));
    }
    if config_file_path.as_os_str().is_empty() {
        return None;
    }
    let base = if config_file_path.is_dir() {
        config_file_path
    } else {
        config_file_path.parent()?
    };
    Some(base.join("static"))
}

#[must_use]
pub fn file_path(config_file_path: &Path, paths: &AssetPaths) -> Option<PathBuf> {
    if let Some(override_path) = paths.static_override.as_deref() {
        return Some(if is_management_file(override_path) {
            override_path.to_path_buf()
        } else {
            override_path.join(MANAGEMENT_FILE_NAME)
        });
    }
    static_dir(config_file_path, paths).map(|path| path.join(MANAGEMENT_FILE_NAME))
}

#[must_use]
pub fn resolve_release_url(repository: &str) -> String {
    let repository = repository.trim();
    if repository.is_empty() {
        return DEFAULT_MANAGEMENT_RELEASE_URL.to_owned();
    }
    let Ok(mut parsed) = Url::parse(repository) else {
        return DEFAULT_MANAGEMENT_RELEASE_URL.to_owned();
    };
    if parsed.scheme() != "https" || parsed.port().is_some() || !parsed.username().is_empty() {
        return DEFAULT_MANAGEMENT_RELEASE_URL.to_owned();
    }
    let Some(host) = parsed.host_str().map(str::to_ascii_lowercase) else {
        return DEFAULT_MANAGEMENT_RELEASE_URL.to_owned();
    };
    if host == "api.github.com" {
        let path = parsed.path().trim_end_matches('/');
        let path = if path.to_ascii_lowercase().ends_with("/releases/latest") {
            path.to_owned()
        } else {
            format!("{path}/releases/latest")
        };
        parsed.set_path(&path);
        parsed.set_query(None);
        parsed.set_fragment(None);
        return parsed.to_string();
    }
    if host == "github.com" {
        let mut parts = parsed.path_segments().into_iter().flatten();
        if let (Some(owner), Some(repository)) = (parts.next(), parts.next()) {
            if !owner.is_empty() && !repository.is_empty() {
                return format!(
                    "https://api.github.com/repos/{owner}/{}/releases/latest",
                    repository.trim_end_matches(".git")
                );
            }
        }
    }
    DEFAULT_MANAGEMENT_RELEASE_URL.to_owned()
}

#[must_use]
pub fn parse_digest(digest: &str) -> String {
    digest
        .trim()
        .split_once(':')
        .map_or(digest.trim(), |(_, value)| value.trim())
        .to_ascii_lowercase()
}

fn is_management_file(path: &Path) -> bool {
    path.file_name().is_some_and(|name| {
        name.to_string_lossy()
            .eq_ignore_ascii_case(MANAGEMENT_FILE_NAME)
    })
}

fn prepare_static_directory(static_dir: &Path, local_path: &Path) -> Result<(), UpdateError> {
    if static_dir.as_os_str().is_empty() || local_path.parent() != Some(static_dir) {
        return Err(UpdateError::UnsafePath);
    }
    if fs::symlink_metadata(static_dir).is_ok_and(|metadata| metadata.file_type().is_symlink()) {
        return Err(UpdateError::UnsafePath);
    }
    fs::create_dir_all(static_dir)?;
    if fs::symlink_metadata(local_path).is_ok_and(|metadata| metadata.file_type().is_symlink()) {
        return Err(UpdateError::UnsafePath);
    }
    Ok(())
}

fn regular_asset_exists(path: &Path) -> bool {
    fs::symlink_metadata(path).is_ok_and(|metadata| metadata.is_file())
}

fn file_sha256(path: &Path) -> io::Result<String> {
    let metadata = fs::symlink_metadata(path)?;
    if !metadata.is_file() || metadata.file_type().is_symlink() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "not a regular file",
        ));
    }
    let mut file = fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 16 * 1024];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn sha256_hex(data: &[u8]) -> String {
    format!("{:x}", Sha256::digest(data))
}

fn atomic_write_file(path: &Path, data: &[u8]) -> Result<(), UpdateError> {
    let parent = path.parent().ok_or(UpdateError::UnsafePath)?;
    if fs::symlink_metadata(path).is_ok_and(|metadata| metadata.file_type().is_symlink()) {
        return Err(UpdateError::UnsafePath);
    }
    let mut random = [0_u8; 12];
    getrandom::fill(&mut random).map_err(|error| {
        UpdateError::Io(io::Error::other(format!("temporary name entropy: {error}")))
    })?;
    let temporary = parent.join(format!("management-{}.html", hex_lower(&random)));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)?;
    let result = (|| -> io::Result<()> {
        file.write_all(data)?;
        file.sync_all()?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            file.set_permissions(fs::Permissions::from_mode(0o644))?;
        }
        drop(file);
        fs::rename(&temporary, path)?;
        if let Ok(directory) = fs::File::open(parent) {
            let _ = directory.sync_all();
        }
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result.map_err(UpdateError::Io)
}

fn hex_lower(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut value = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        value.push(char::from(HEX[usize::from(byte >> 4)]));
        value.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    value
}
