// ref: internal/util/util.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::fmt;
use std::future::Future;
use std::path::{Component, Path, PathBuf};
use std::pin::Pin;

/// Gemini/Vertex function-name normalization from the pinned upstream.
///
/// Every non-ASCII scalar is replaced before truncation, so the resulting
/// string is ASCII and Go's 64-byte limit is also a valid UTF-8 boundary.
#[must_use]
pub fn sanitize_function_name(name: &str) -> String {
    if name.is_empty() {
        return String::new();
    }
    let mut sanitized = name
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || "_.:-".contains(character) {
                character
            } else {
                '_'
            }
        })
        .collect::<String>();
    if !sanitized
        .as_bytes()
        .first()
        .is_some_and(|first| first.is_ascii_alphabetic() || *first == b'_')
    {
        if sanitized.len() >= 64 {
            sanitized.truncate(63);
        }
        sanitized.insert(0, '_');
    }
    sanitized.truncate(sanitized.len().min(64));
    sanitized
}

/// The log levels understood at the gateway/host boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HostLogLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
    Off,
}

/// A pure decision returned to the CTOX host. The port deliberately does not
/// mutate a process-global logger.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogLevelDecision {
    Keep {
        level: HostLogLevel,
    },
    Change {
        from: HostLogLevel,
        to: HostLogLevel,
        debug: bool,
    },
}

/// Typed host-owned utility settings. This replaces the upstream environment
/// lookup and keeps both values in the runtime configuration boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UtilityHostConfig {
    debug: bool,
    writable_path: Option<PathBuf>,
}

impl UtilityHostConfig {
    #[must_use]
    pub fn new(debug: bool, writable_path: Option<&str>) -> Self {
        let writable_path = writable_path
            .map(str::trim)
            .filter(|path| !path.is_empty())
            .map(Path::new)
            .map(clean_path);
        Self {
            debug,
            writable_path,
        }
    }

    #[must_use]
    pub fn debug(&self) -> bool {
        self.debug
    }

    #[must_use]
    pub fn writable_path(&self) -> Option<&Path> {
        self.writable_path.as_deref()
    }
}

/// Computes the log-level transition which the host may apply.
#[must_use]
pub fn log_level_decision(config: &UtilityHostConfig, current: HostLogLevel) -> LogLevelDecision {
    let desired = if config.debug() {
        HostLogLevel::Debug
    } else {
        HostLogLevel::Info
    };
    if current == desired {
        LogLevelDecision::Keep { level: current }
    } else {
        LogLevelDecision::Change {
            from: current,
            to: desired,
            debug: config.debug(),
        }
    }
}

/// Error returned when a tilde-prefixed auth directory cannot be expanded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolveAuthDirError {
    HomeDirectoryUnavailable,
}

impl fmt::Display for ResolveAuthDirError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("resolve auth dir: home directory unavailable")
    }
}

impl std::error::Error for ResolveAuthDirError {}

/// Resolves the configured auth directory without consulting ambient process
/// state. Both the home directory and the upstream default are supplied by the
/// host explicitly.
pub fn resolve_auth_dir(
    auth_dir: &str,
    home_dir: Option<&Path>,
    default_auth_dir: &str,
) -> Result<PathBuf, ResolveAuthDirError> {
    let auth_dir = if auth_dir.is_empty() {
        default_auth_dir
    } else {
        auth_dir
    };
    let Some(remainder) = auth_dir.strip_prefix('~') else {
        return Ok(clean_path(Path::new(auth_dir)));
    };
    let home = home_dir.ok_or(ResolveAuthDirError::HomeDirectoryUnavailable)?;
    let remainder = remainder.trim_start_matches(['/', '\\']);
    if remainder.is_empty() {
        return Ok(clean_path(home));
    }
    let normalized = remainder.replace('\\', "/");
    Ok(clean_path(&home.join(normalized)))
}

/// Redacted failure categories which an injected auth store may expose. No
/// path, token, record, or backend error string crosses this boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuthStoreFailureKind {
    Unavailable,
    PermissionDenied,
    InvalidData,
    Other,
}

impl fmt::Display for AuthStoreFailureKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Unavailable => "unavailable",
            Self::PermissionDenied => "permission denied",
            Self::InvalidData => "invalid data",
            Self::Other => "other",
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AuthStoreListError {
    kind: AuthStoreFailureKind,
}

impl AuthStoreListError {
    #[must_use]
    pub const fn new(kind: AuthStoreFailureKind) -> Self {
        Self { kind }
    }

    #[must_use]
    pub const fn kind(self) -> AuthStoreFailureKind {
        self.kind
    }
}

impl fmt::Display for AuthStoreListError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "auth record store list failed ({})", self.kind)
    }
}

impl std::error::Error for AuthStoreListError {}

pub type AuthStoreListFuture<'a, Record> =
    Pin<Box<dyn Future<Output = Result<Vec<Record>, AuthStoreListError>> + Send + 'a>>;

/// Minimal async store contract injected by CTOX. The utility owns neither a
/// filesystem implementation nor persistence authority.
pub trait AuthRecordStore {
    type Record: Send;

    fn list(&self) -> AuthStoreListFuture<'_, Self::Record>;
}

/// Counts auth records through an injected store. Missing stores retain the Go
/// nil-store result of zero; store failures remain typed and safely redacted so
/// the host chooses whether and where to report them.
pub async fn count_auth_files<Store>(store: Option<&Store>) -> Result<usize, AuthStoreListError>
where
    Store: AuthRecordStore + ?Sized,
{
    let Some(store) = store else {
        return Ok(0);
    };
    store.list().await.map(|records| records.len())
}

/// Returns the normalized writable directory selected by typed runtime
/// configuration. No environment lookup or lowercase fallback is performed.
#[must_use]
pub fn writable_path(config: &UtilityHostConfig) -> Option<PathBuf> {
    config.writable_path().map(Path::to_path_buf)
}

fn clean_path(path: &Path) -> PathBuf {
    let mut cleaned = PathBuf::new();
    let mut has_root = false;
    for component in path.components() {
        match component {
            Component::Prefix(prefix) => cleaned.push(prefix.as_os_str()),
            Component::RootDir => {
                cleaned.push(component.as_os_str());
                has_root = true;
            }
            Component::CurDir => {}
            Component::ParentDir => {
                let last_is_normal = cleaned
                    .components()
                    .next_back()
                    .is_some_and(|last| matches!(last, Component::Normal(_)));
                if last_is_normal {
                    cleaned.pop();
                } else if !has_root {
                    cleaned.push("..");
                }
            }
            Component::Normal(part) => cleaned.push(part),
        }
    }
    if cleaned.as_os_str().is_empty() {
        PathBuf::from(".")
    } else {
        cleaned
    }
}
