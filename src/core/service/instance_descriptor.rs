// Origin: CTOX
// License: AGPL-3.0-only

//! Local-instance descriptor for desktop discovery.
//!
//! CTOX Desktop discovers a locally running daemon by reading
//! `<state-root>/instance.json`. The state root is the directory
//! [`crate::paths::runtime_dir`] resolves to — `CTOX_STATE_ROOT` when the
//! installer set it (`~/.local/state/ctox` by default), otherwise
//! `<root>/runtime`.
//!
//! The consumer validates the document against a strict schema and rejects
//! unknown properties, so the serialized shape here is a contract:
//!
//! ```json
//! {"version":1,"instanceId":"biz_…","displayName":"…","status":"running",
//!  "lastSeenAt":1755000000000,"healthUrl":"http://127.0.0.1:8788/health"}
//! ```
//!
//! Discovery downgrades a self-declared `running` descriptor whose
//! `lastSeenAt` is older than 120 s, so the daemon refreshes the timestamp
//! every [`HEARTBEAT_INTERVAL`].
//!
//! Nothing here may block or fail the daemon: every write is best-effort and
//! logs instead of propagating.

use serde::Serialize;
use std::io::Write;
use std::path::Path;
use std::path::PathBuf;
use std::sync::OnceLock;
use std::thread;
use std::time::Duration;
use std::time::SystemTime;
use std::time::UNIX_EPOCH;

/// File name inside the state root. Fixed by the desktop consumer contract.
pub const DESCRIPTOR_FILE_NAME: &str = "instance.json";

/// Refresh period for `lastSeenAt`. Must stay well below the consumer's 120 s
/// staleness window.
pub const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(45);

/// Schema version understood by the desktop discovery reader.
const DESCRIPTOR_VERSION: u32 = 1;

/// Consumer limit: the descriptor may not exceed 64 KiB.
const MAX_DESCRIPTOR_BYTES: usize = 64 * 1024;

const MAX_INSTANCE_ID_CHARS: usize = 128;
const MAX_DISPLAY_NAME_CHARS: usize = 256;
const MAX_HEALTH_URL_CHARS: usize = 2048;

/// Used when the Business OS instance id is unavailable or sanitizes away.
/// A stable constant is better than a random id: discovery keys on it.
const FALLBACK_INSTANCE_ID: &str = "ctox-local";

/// CTOX has no per-instance display name today, so the constant is the honest
/// answer. Replace it here if an instance label ever becomes first-class.
const DEFAULT_DISPLAY_NAME: &str = "CTOX Local Instance";

/// Business OS MCP surface. Its `GET /health` is credential-free and bound to
/// loopback; the descriptor only points at it, it does not create it.
const BUSINESS_OS_MCP_AUTOSTART_KEY: &str = "CTOX_BUSINESS_OS_MCP_AUTOSTART";
const BUSINESS_OS_MCP_ADDR_KEY: &str = "CTOX_BUSINESS_OS_MCP_ADDR";
const BUSINESS_OS_MCP_DEFAULT_ADDR: &str = "127.0.0.1:8788";

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub enum DescriptorStatus {
    Running,
    Stopped,
}

impl DescriptorStatus {
    fn as_str(&self) -> &'static str {
        match self {
            DescriptorStatus::Running => "running",
            DescriptorStatus::Stopped => "stopped",
        }
    }
}

/// The exact document written to disk. Field order is the serialized order.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct InstanceDescriptor {
    pub version: u32,
    #[serde(rename = "instanceId")]
    pub instance_id: String,
    #[serde(rename = "displayName")]
    pub display_name: String,
    pub status: String,
    #[serde(rename = "lastSeenAt")]
    pub last_seen_at: i64,
    #[serde(rename = "healthUrl", skip_serializing_if = "Option::is_none")]
    pub health_url: Option<String>,
}

/// The parts of the descriptor that do not change while the daemon runs.
/// Resolved once at startup so the heartbeat and the exit hook never touch
/// SQLite or the config store again.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InstanceIdentity {
    pub instance_id: String,
    pub display_name: String,
    pub health_url: Option<String>,
}

impl InstanceIdentity {
    pub fn descriptor(&self, status: DescriptorStatus, last_seen_at: i64) -> InstanceDescriptor {
        InstanceDescriptor {
            version: DESCRIPTOR_VERSION,
            instance_id: self.instance_id.clone(),
            display_name: self.display_name.clone(),
            status: status.as_str().to_string(),
            last_seen_at,
            health_url: self.health_url.clone(),
        }
    }
}

/// Everything the exit hook needs, captured at startup.
struct PublishedInstance {
    path: PathBuf,
    identity: InstanceIdentity,
}

static PUBLISHED: OnceLock<PublishedInstance> = OnceLock::new();

/// `<state-root>/instance.json`.
pub fn descriptor_path(root: &Path) -> PathBuf {
    crate::paths::runtime_dir(root).join(DESCRIPTOR_FILE_NAME)
}

pub fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| i64::try_from(elapsed.as_millis()).unwrap_or(i64::MAX))
        .unwrap_or(0)
}

/// Publishes a `running` descriptor, registers the clean-shutdown rewrite and
/// starts the heartbeat. Safe to call more than once; only the first call
/// takes effect.
pub fn start(root: &Path) {
    let published = PublishedInstance {
        path: descriptor_path(root),
        identity: resolve_identity(root),
    };
    publish(
        &published.path,
        &published.identity,
        DescriptorStatus::Running,
    );

    if PUBLISHED.set(published).is_err() {
        // Already started; the running heartbeat owns the descriptor.
        return;
    }
    register_exit_hook();

    let _ = thread::Builder::new()
        .name("ctox-instance-descriptor".to_string())
        .spawn(move || loop {
            thread::sleep(HEARTBEAT_INTERVAL);
            heartbeat_tick();
        });
}

/// One heartbeat: rewrite the published descriptor with a fresh `lastSeenAt`.
fn heartbeat_tick() {
    if let Some(published) = PUBLISHED.get() {
        publish(
            &published.path,
            &published.identity,
            DescriptorStatus::Running,
        );
    }
}

/// Best-effort `stopped` rewrite. Called from the process exit hook so both
/// service stop paths (`ServiceIpcRequest::Stop` and `POST
/// /ctox/service/stop`, which both end in `std::process::exit`) are covered
/// without the service loop having to remember it. A SIGKILL or SIGTERM
/// leaves the last `running` descriptor behind — that is what the consumer's
/// 120 s staleness downgrade is for.
pub fn mark_stopped() {
    if let Some(published) = PUBLISHED.get() {
        publish(
            &published.path,
            &published.identity,
            DescriptorStatus::Stopped,
        );
    }
}

#[cfg(unix)]
fn register_exit_hook() {
    extern "C" fn on_exit() {
        // A panic must never cross the C boundary.
        let _ = std::panic::catch_unwind(mark_stopped);
    }
    // SAFETY: `atexit` only stores the function pointer; `on_exit` performs
    // plain filesystem work and cannot unwind.
    unsafe {
        libc::atexit(on_exit);
    }
}

#[cfg(not(unix))]
fn register_exit_hook() {}

/// Writes the descriptor, logging instead of propagating any failure.
pub fn publish(path: &Path, identity: &InstanceIdentity, status: DescriptorStatus) -> bool {
    let descriptor = identity.descriptor(status, now_ms());
    match write_descriptor(path, &descriptor) {
        Ok(()) => true,
        Err(failure) => {
            eprintln!(
                "ctox service: instance descriptor write failed for {}: {failure}",
                path.display()
            );
            false
        }
    }
}

/// Atomic, owner-only write: temp file in the same directory, then rename.
pub fn write_descriptor(path: &Path, descriptor: &InstanceDescriptor) -> std::io::Result<()> {
    let payload = serialize(descriptor)?;
    if payload.len() > MAX_DESCRIPTOR_BYTES {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "instance descriptor is {} bytes, limit {MAX_DESCRIPTOR_BYTES}",
                payload.len()
            ),
        ));
    }
    let parent = path.parent().ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("instance descriptor path {} has no parent", path.display()),
        )
    })?;
    std::fs::create_dir_all(parent)?;

    let temp_path = parent.join(format!("{DESCRIPTOR_FILE_NAME}.{}.tmp", std::process::id()));
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let write_result = options.open(&temp_path).and_then(|mut file| {
        file.write_all(payload.as_bytes())
            .and_then(|()| file.sync_all())
    });
    if let Err(failure) = write_result {
        let _ = std::fs::remove_file(&temp_path);
        return Err(failure);
    }
    if let Err(failure) = std::fs::rename(&temp_path, path) {
        let _ = std::fs::remove_file(&temp_path);
        return Err(failure);
    }
    Ok(())
}

/// The exact bytes written. Compact JSON, no trailing newline.
pub fn serialize(descriptor: &InstanceDescriptor) -> std::io::Result<String> {
    serde_json::to_string(descriptor)
        .map_err(|failure| std::io::Error::new(std::io::ErrorKind::InvalidData, failure))
}

/// Reuses the daemon's existing stable Business OS instance identity so the
/// descriptor names the same instance the rest of CTOX does.
pub fn resolve_identity(root: &Path) -> InstanceIdentity {
    let instance_id = match crate::business_os::store::stable_instance_id(root) {
        Ok(value) => sanitize_instance_id(&value),
        Err(failure) => {
            eprintln!("ctox service: instance descriptor id unavailable: {failure:#}");
            FALLBACK_INSTANCE_ID.to_string()
        }
    };
    InstanceIdentity {
        instance_id,
        display_name: sanitize_display_name(DEFAULT_DISPLAY_NAME),
        health_url: resolve_health_url(root),
    }
}

/// `^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$`.
pub fn sanitize_instance_id(raw: &str) -> String {
    let mut sanitized = String::new();
    for character in raw.trim().chars() {
        if sanitized.is_empty() {
            if character.is_ascii_alphanumeric() {
                sanitized.push(character);
            }
            continue;
        }
        if character.is_ascii_alphanumeric() || matches!(character, '.' | '_' | ':' | '-') {
            sanitized.push(character);
        }
        if sanitized.len() >= MAX_INSTANCE_ID_CHARS {
            break;
        }
    }
    if sanitized.is_empty() {
        return FALLBACK_INSTANCE_ID.to_string();
    }
    sanitized
}

/// At most 256 characters, no control characters.
pub fn sanitize_display_name(raw: &str) -> String {
    let sanitized = raw
        .chars()
        .filter(|character| !character.is_control())
        .take(MAX_DISPLAY_NAME_CHARS)
        .collect::<String>();
    let trimmed = sanitized.trim();
    if trimmed.is_empty() {
        return DEFAULT_DISPLAY_NAME.to_string();
    }
    trimmed.to_string()
}

/// The Business OS MCP surface already serves a credential-free `GET /health`
/// on loopback. No new endpoint is introduced: when that surface is disabled
/// or bound off loopback, the field is omitted.
pub fn resolve_health_url(root: &Path) -> Option<String> {
    if !config_bool(root, BUSINESS_OS_MCP_AUTOSTART_KEY, true) {
        return None;
    }
    let addr = crate::inference::runtime_env::env_or_config(root, BUSINESS_OS_MCP_ADDR_KEY)
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| BUSINESS_OS_MCP_DEFAULT_ADDR.to_string());
    health_url_for_addr(addr.trim())
}

/// Only loopback addresses become a `healthUrl`.
pub fn health_url_for_addr(addr: &str) -> Option<String> {
    let host = addr.rsplit_once(':').map(|(host, _)| host).unwrap_or(addr);
    let host = host.trim_matches(|character| matches!(character, '[' | ']'));
    if !matches!(host, "127.0.0.1" | "localhost" | "::1") {
        return None;
    }
    let url = format!("http://{addr}/health");
    if url.chars().count() > MAX_HEALTH_URL_CHARS {
        return None;
    }
    Some(url)
}

fn config_bool(root: &Path, key: &str, default: bool) -> bool {
    match crate::inference::runtime_env::env_or_config(root, key)
        .map(|value| value.trim().to_ascii_lowercase())
        .as_deref()
    {
        Some("1" | "true" | "yes" | "on") => true,
        Some("0" | "false" | "no" | "off") => false,
        _ => default,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity() -> InstanceIdentity {
        InstanceIdentity {
            instance_id: "biz_2f1c8e64-9a0d-4f2b-9c4d-6f0a1b2c3d4e".to_string(),
            display_name: "CTOX Local Instance".to_string(),
            health_url: Some("http://127.0.0.1:8788/health".to_string()),
        }
    }

    #[test]
    fn descriptor_serializes_to_the_desktop_contract() {
        let descriptor = identity().descriptor(DescriptorStatus::Running, 1_755_000_000_000);
        assert_eq!(
            serialize(&descriptor).expect("serialize"),
            "{\"version\":1,\
             \"instanceId\":\"biz_2f1c8e64-9a0d-4f2b-9c4d-6f0a1b2c3d4e\",\
             \"displayName\":\"CTOX Local Instance\",\
             \"status\":\"running\",\
             \"lastSeenAt\":1755000000000,\
             \"healthUrl\":\"http://127.0.0.1:8788/health\"}"
        );
    }

    #[test]
    fn optional_health_url_is_omitted_entirely() {
        let mut identity = identity();
        identity.health_url = None;
        let descriptor = identity.descriptor(DescriptorStatus::Stopped, 42);
        assert_eq!(
            serialize(&descriptor).expect("serialize"),
            "{\"version\":1,\
             \"instanceId\":\"biz_2f1c8e64-9a0d-4f2b-9c4d-6f0a1b2c3d4e\",\
             \"displayName\":\"CTOX Local Instance\",\
             \"status\":\"stopped\",\
             \"lastSeenAt\":42}"
        );
    }

    #[test]
    fn write_is_atomic_owner_only_and_leaves_no_temp_file() {
        let temp = tempfile::tempdir().expect("tempdir");
        let path = temp.path().join("state").join(DESCRIPTOR_FILE_NAME);
        let descriptor = identity().descriptor(DescriptorStatus::Running, 7);
        write_descriptor(&path, &descriptor).expect("write");

        assert_eq!(
            std::fs::read_to_string(&path).expect("read"),
            serialize(&descriptor).expect("serialize")
        );
        let leftovers = std::fs::read_dir(path.parent().expect("parent"))
            .expect("read_dir")
            .flatten()
            .filter(|entry| entry.file_name() != std::ffi::OsStr::new(DESCRIPTOR_FILE_NAME))
            .count();
        assert_eq!(leftovers, 0, "temp file survived the atomic write");

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(&path)
                .expect("metadata")
                .permissions()
                .mode();
            assert_eq!(mode & 0o777, 0o600, "descriptor must be owner-only");
        }
    }

    #[test]
    fn heartbeat_refreshes_last_seen_at_in_place() {
        let temp = tempfile::tempdir().expect("tempdir");
        let path = temp.path().join(DESCRIPTOR_FILE_NAME);
        let identity = identity();

        let first = identity.descriptor(DescriptorStatus::Running, 1_000);
        write_descriptor(&path, &first).expect("first write");
        let second = identity.descriptor(DescriptorStatus::Running, 61_000);
        write_descriptor(&path, &second).expect("heartbeat write");

        let written = std::fs::read_to_string(&path).expect("read");
        assert!(written.contains("\"lastSeenAt\":61000"), "{written}");
        assert!(written.contains("\"status\":\"running\""), "{written}");
        assert!(now_ms() > 1_700_000_000_000, "now_ms must be epoch millis");
    }

    #[test]
    fn clean_shutdown_rewrites_the_descriptor_as_stopped() {
        let temp = tempfile::tempdir().expect("tempdir");
        let path = temp.path().join(DESCRIPTOR_FILE_NAME);
        let identity = identity();

        assert!(publish(&path, &identity, DescriptorStatus::Running));
        assert!(std::fs::read_to_string(&path)
            .expect("read")
            .contains("\"status\":\"running\""));

        assert!(publish(&path, &identity, DescriptorStatus::Stopped));
        let written = std::fs::read_to_string(&path).expect("read");
        assert!(written.contains("\"status\":\"stopped\""), "{written}");
        assert!(!written.contains("\"status\":\"running\""), "{written}");
    }

    #[test]
    fn instance_id_is_sanitized_to_the_consumer_pattern() {
        assert_eq!(
            sanitize_instance_id("biz_2f1c8e64-9a0d-4f2b"),
            "biz_2f1c8e64-9a0d-4f2b"
        );
        // Leading non-alphanumerics are dropped, interior illegal bytes too.
        assert_eq!(sanitize_instance_id("__biz/os id\n#7"), "bizosid7");
        assert_eq!(sanitize_instance_id("--/#biz os/7"), "bizos7");
        assert_eq!(sanitize_instance_id("///"), FALLBACK_INSTANCE_ID);
        assert_eq!(sanitize_instance_id(""), FALLBACK_INSTANCE_ID);
        assert_eq!(
            sanitize_instance_id(&"a".repeat(400)).chars().count(),
            MAX_INSTANCE_ID_CHARS
        );

        let sanitized = sanitize_instance_id("--/#biz os/7");
        assert!(sanitized
            .chars()
            .all(|character| character.is_ascii_alphanumeric()
                || matches!(character, '.' | '_' | ':' | '-')));
        assert!(sanitized
            .chars()
            .next()
            .expect("non-empty")
            .is_ascii_alphanumeric());
    }

    #[test]
    fn display_name_drops_control_characters_and_caps_length() {
        assert_eq!(sanitize_display_name("CTOX\u{7}Local\n"), "CTOXLocal");
        assert_eq!(sanitize_display_name("   "), DEFAULT_DISPLAY_NAME);
        assert_eq!(
            sanitize_display_name(&"x".repeat(400)).chars().count(),
            MAX_DISPLAY_NAME_CHARS
        );
    }

    #[test]
    fn health_url_only_points_at_an_existing_loopback_endpoint() {
        assert_eq!(
            health_url_for_addr("127.0.0.1:8788"),
            Some("http://127.0.0.1:8788/health".to_string())
        );
        assert_eq!(
            health_url_for_addr("localhost:8788"),
            Some("http://localhost:8788/health".to_string())
        );
        assert_eq!(health_url_for_addr("0.0.0.0:8788"), None);
        assert_eq!(health_url_for_addr("192.168.1.10:8788"), None);
    }

    #[test]
    fn write_failure_is_logged_and_never_propagates() {
        let temp = tempfile::tempdir().expect("tempdir");
        let blocker = temp.path().join("blocker");
        std::fs::write(&blocker, b"not a directory").expect("write blocker");
        // The parent is a regular file, so create_dir_all cannot succeed.
        let path = blocker.join("nested").join(DESCRIPTOR_FILE_NAME);

        assert!(
            write_descriptor(&path, &identity().descriptor(DescriptorStatus::Running, 1)).is_err()
        );
        assert!(!publish(&path, &identity(), DescriptorStatus::Running));
        assert!(!path.exists());
    }
}
