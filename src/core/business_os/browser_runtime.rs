// Origin: CTOX
// License: AGPL-3.0-only

//! Native session registry for persistent remote-browser runtimes.
//!
//! Each Business OS `browser_sessions` row that is `active` is backed here by a
//! long-lived Chromium/Patchright process (see
//! [`crate::web_stack::spawn_persistent_browser`]). The native RxDB peer drives
//! these sessions: lifecycle commands (`browser.session.start`,
//! `browser.navigate`, ...) and the periodic input/frame maintenance loop both
//! route through this manager. The manager owns no RxDB state; it only owns the
//! live processes and serializes access to each one.

use std::collections::HashMap;
use std::path::Path;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use anyhow::Context;
use anyhow::Result;
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::web_stack::spawn_persistent_browser;
use crate::web_stack::PersistentBrowserHandle;
use crate::web_stack::PersistentBrowserSpawn;

/// One live browser process plus its viewport, guarded so only one request runs
/// against the process at a time. The handle does blocking stdin/stdout IO, so
/// every access happens inside `spawn_blocking`.
pub struct LiveBrowserSession {
    handle: Mutex<PersistentBrowserHandle>,
    profile_key: String,
    pub viewport_w: u64,
    pub viewport_h: u64,
    pub downloads_dir: PathBuf,
    pub owner_user_id: String,
    pub root: PathBuf,
    last_input_seq: AtomicU64,
    clipboard: Mutex<Option<(String, Instant)>>,
}

impl LiveBrowserSession {
    /// The automation script this session is actually running, plus its path.
    ///
    /// The pinned `ctox-web-stack` keeps `runner_path` private and exposes no
    /// accessor for it, so the path is recovered from the live process instead:
    /// the generated runner is `<root>/.ctox-browser-live-<pid>-<ts>.mjs`, and
    /// node was started with it as its script argument. Matching the candidates
    /// in `root` against that command line stays session-exact even when one
    /// CTOX process owns several sessions, which a name glob alone would not.
    ///
    /// Read-only: the runner already loaded the file, so editing it here would
    /// race with a live process and change nothing about the running session.
    pub fn runner_script(&self) -> Result<(String, String)> {
        let pid = self
            .handle
            .lock()
            .map_err(|_| anyhow::anyhow!("browser runtime handle is poisoned"))?
            .process_id();
        let path = self.runner_script_path(pid)?;
        let script = std::fs::read_to_string(&path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        Ok((script, path.display().to_string()))
    }

    /// Locate the runner file that `pid` was started with, inside this
    /// session's reference directory.
    fn runner_script_path(&self, pid: u32) -> Result<PathBuf> {
        select_runner_script(
            &runner_script_dirs(&self.root),
            pid,
            &runner_command_line(pid),
        )
    }

    pub fn record_input_seq(&self, seq: u64) {
        self.last_input_seq.fetch_max(seq, Ordering::Relaxed);
    }

    pub fn last_input_seq(&self) -> u64 {
        self.last_input_seq.load(Ordering::Relaxed)
    }

    pub fn set_clipboard(&self, value: String) {
        if let Ok(mut clipboard) = self.clipboard.lock() {
            *clipboard = Some((value, Instant::now()));
        }
    }

    pub fn clipboard(&self) -> Option<String> {
        let mut clipboard = self.clipboard.lock().ok()?;
        let Some((value, created_at)) = clipboard.as_ref() else {
            return None;
        };
        if created_at.elapsed() > Duration::from_secs(60) {
            *clipboard = None;
            return None;
        }
        Some(value.clone())
    }

    pub fn clear_clipboard(&self) {
        if let Ok(mut clipboard) = self.clipboard.lock() {
            *clipboard = None;
        }
    }
}

/// Process registry keyed by `session_id`.
pub struct BrowserRuntimeManager {
    sessions: Mutex<HashMap<String, Arc<LiveBrowserSession>>>,
    crash_history: Mutex<HashMap<String, Vec<Instant>>>,
    spawn_lock: tokio::sync::Mutex<()>,
}

static MANAGER: OnceLock<BrowserRuntimeManager> = OnceLock::new();

/// Global, lazily created manager shared by the command consumer and the
/// maintenance loop.
pub fn browser_runtime_manager() -> &'static BrowserRuntimeManager {
    MANAGER.get_or_init(BrowserRuntimeManager::new)
}

fn browser_profile_key(profile_owner: &str, session_id: &str, private_profile: bool) -> String {
    if private_profile
        || session_id.starts_with("browser_session_web_stack_auth_")
        || session_id.starts_with("browser_session_web_stack_public_")
    {
        format!("{profile_owner}:{session_id}")
    } else {
        profile_owner.to_string()
    }
}

#[derive(Debug, Clone)]
pub struct BrowserSessionAutomationRequest {
    pub session_id: String,
    pub dir: Option<PathBuf>,
    pub timeout_ms: Option<u64>,
    pub source: String,
}

impl BrowserRuntimeManager {
    fn new() -> Self {
        Self {
            sessions: Mutex::new(HashMap::new()),
            crash_history: Mutex::new(HashMap::new()),
            spawn_lock: tokio::sync::Mutex::new(()),
        }
    }

    fn record_crash(&self, session_id: &str) {
        if let Ok(mut history) = self.crash_history.lock() {
            history
                .entry(session_id.to_string())
                .or_default()
                .push(Instant::now());
        }
    }

    fn clear_crash_history(&self, session_id: &str) {
        if let Ok(mut history) = self.crash_history.lock() {
            history.remove(session_id);
        }
    }

    /// Remove runner handles whose process has already exited. A dead handle is
    /// not a browser session and must not consume either runtime budget.
    fn reap_exited_sessions(&self) -> Vec<String> {
        let exited = {
            let Ok(mut sessions) = self.sessions.lock() else {
                return Vec::new();
            };
            let exited = sessions
                .iter()
                .filter_map(|(session_id, session)| {
                    let running = match session.handle.lock() {
                        Ok(mut handle) => match handle.is_running() {
                            Ok(running) => running,
                            Err(err) => {
                                eprintln!(
                                    "[business-os] browser session liveness check failed \
                                     session_id={session_id}: {err:#}"
                                );
                                false
                            }
                        },
                        Err(_) => false,
                    };
                    (!running).then_some(session_id.clone())
                })
                .collect::<Vec<_>>();
            for session_id in &exited {
                sessions.remove(session_id);
            }
            exited
        };
        for session_id in &exited {
            self.record_crash(session_id);
            eprintln!(
                "[business-os] reaped browser session without a live process \
                 session_id={session_id}"
            );
        }
        exited
    }

    /// Return an existing process-backed session, if any.
    pub fn get(&self, session_id: &str) -> Option<Arc<LiveBrowserSession>> {
        self.reap_exited_sessions();
        self.sessions.lock().ok()?.get(session_id).cloned()
    }

    /// True when a live process currently backs `session_id`.
    pub fn has_session(&self, session_id: &str) -> bool {
        self.get(session_id).is_some()
    }

    /// All currently process-backed session ids.
    pub fn active_session_ids(&self) -> Vec<String> {
        self.reap_exited_sessions();
        self.sessions
            .lock()
            .map(|map| map.keys().cloned().collect())
            .unwrap_or_default()
    }

    /// Return the existing session or spawn a fresh persistent runtime for it.
    pub async fn ensure_session(
        &self,
        root: PathBuf,
        dir: Option<PathBuf>,
        session_id: &str,
        viewport_w: u64,
        viewport_h: u64,
        profile_owner: &str,
        private_profile: bool,
        user_requested_start: bool,
    ) -> Result<Arc<LiveBrowserSession>> {
        if let Some(session) = self.get(session_id) {
            return Ok(session);
        }
        // Chromium permits only one live process per persistent profile. Session
        // starts can race (or replace a disconnected logical session), so make
        // profile handoff atomic and stop the previous process before launching
        // the same tenant/user profile again.
        let _spawn_guard = self.spawn_lock.lock().await;
        if let Some(session) = self.get(session_id) {
            return Ok(session);
        }
        let profile_key = browser_profile_key(profile_owner, session_id, private_profile);
        let conflicting_session_ids = self
            .sessions
            .lock()
            .map(|sessions| {
                sessions
                    .iter()
                    .filter_map(|(id, session)| {
                        (session.profile_key == profile_key).then_some(id.clone())
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        for conflicting_session_id in conflicting_session_ids {
            self.stop(&conflicting_session_id).await;
        }
        if let Ok(mut history) = self.crash_history.lock() {
            let crashes = history.entry(session_id.to_string()).or_default();
            crashes.retain(|at| at.elapsed() < Duration::from_secs(5 * 60));
            if crashes.len() >= 3 {
                if user_requested_start {
                    eprintln!(
                        "[business-os] browser crash-loop guard overridden by explicit start \
                         session_id={session_id} recent_failures={} ",
                        crashes.len()
                    );
                    crashes.clear();
                } else {
                    let detail = format!(
                        "browser crash-loop protection paused automatic restart for session \
                         `{session_id}` after {} failures within 5 minutes; start the session \
                         again to request one fresh attempt",
                        crashes.len()
                    );
                    eprintln!(
                        "[business-os] browser session start blocked session_id={session_id} \
                         gate=crash_loop recent_failures={}: {detail}",
                        crashes.len()
                    );
                    anyhow::bail!(detail);
                }
            }
        }
        let max_sessions = crate::inference::runtime_env::get_runtime_env_value(
            &root,
            "CTOX_BROWSER_MAX_SESSIONS",
        )
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(16)
        .clamp(1, 128);
        let max_sessions_per_user = crate::inference::runtime_env::get_runtime_env_value(
            &root,
            "CTOX_BROWSER_MAX_SESSIONS_PER_USER",
        )
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(3)
        .clamp(1, 16);
        if let Ok(sessions) = self.sessions.lock() {
            let live_sessions = sessions.len();
            if live_sessions >= max_sessions {
                let detail = format!(
                    "browser session limit reached: {live_sessions}/{max_sessions} live \
                     sessions; stop another active session and try again"
                );
                eprintln!(
                    "[business-os] browser session start blocked session_id={session_id} \
                     gate=global_budget live_sessions={live_sessions} limit={max_sessions}: \
                     {detail}"
                );
                anyhow::bail!(detail);
            }
            let live_user_sessions = sessions
                .values()
                .filter(|session| session.owner_user_id == profile_owner)
                .count();
            if live_user_sessions >= max_sessions_per_user {
                let detail = format!(
                    "browser session limit for this user reached: \
                     {live_user_sessions}/{max_sessions_per_user} live sessions; stop one of \
                     your active sessions and try again"
                );
                eprintln!(
                    "[business-os] browser session start blocked session_id={session_id} \
                     gate=user_budget live_user_sessions={live_user_sessions} \
                     limit={max_sessions_per_user}: {detail}"
                );
                anyhow::bail!(detail);
            }
        }
        let digest = Sha256::digest(profile_key.as_bytes());
        let profile_kind = if private_profile {
            "private"
        } else {
            "profiles"
        };
        let profile_dir = Some(
            crate::paths::runtime_dir(&root)
                .join("browser")
                .join(profile_kind)
                .join(format!("{:x}", digest)),
        );
        let downloads_dir = crate::paths::runtime_dir(&root)
            .join("browser/downloads")
            .join(format!("{:x}", digest));
        let spawn = PersistentBrowserSpawn {
            dir,
            viewport_w,
            viewport_h,
            profile_dir,
            private_profile,
            egress_allow_hosts: ctox_web_stack::browser_egress_allow_hosts_from_config(&root),
            downloads_dir: Some(downloads_dir.clone()),
        };
        let session_root = root.clone();
        eprintln!(
            "[business-os] browser session start requested session_id={session_id} \
             private_profile={private_profile} viewport={}x{}",
            viewport_w, viewport_h
        );
        let handle =
            match tokio::task::spawn_blocking(move || spawn_persistent_browser(&root, &spawn))
                .await
                .context("browser runtime spawn worker panicked")
                .and_then(|result| result)
            {
                Ok(handle) => handle,
                Err(err) => {
                    self.record_crash(session_id);
                    eprintln!(
                        "[business-os] browser session start failed session_id={session_id}: \
                     {err:#}"
                    );
                    return Err(err);
                }
            };
        let runner_pid = handle.process_id();
        let session = Arc::new(LiveBrowserSession {
            handle: Mutex::new(handle),
            profile_key,
            viewport_w,
            viewport_h,
            downloads_dir,
            owner_user_id: profile_owner.to_string(),
            root: session_root,
            last_input_seq: AtomicU64::new(0),
            clipboard: Mutex::new(None),
        });
        if let Ok(mut map) = self.sessions.lock() {
            map.insert(session_id.to_string(), Arc::clone(&session));
        }
        self.clear_crash_history(session_id);
        eprintln!(
            "[business-os] browser session start succeeded session_id={session_id} \
             runner_pid={runner_pid}; persistent Chromium reported ready"
        );
        Ok(session)
    }

    /// Send one operation to a live session and await its JSON response.
    pub async fn request(
        &self,
        session: &Arc<LiveBrowserSession>,
        op: &str,
        params: Value,
    ) -> Result<Value> {
        let session = Arc::clone(session);
        let op = op.to_string();
        tokio::task::spawn_blocking(move || {
            let mut handle = session
                .handle
                .lock()
                .map_err(|_| anyhow::anyhow!("browser runtime handle poisoned"))?;
            handle.request(&op, params)
        })
        .await
        .context("browser runtime request worker panicked")?
    }

    /// Send one operation with a process-level deadline. This is used for
    /// operator-provided automation where page code or browser cleanup can
    /// otherwise leave the daemon blocked indefinitely.
    pub async fn request_with_timeout(
        &self,
        session: &Arc<LiveBrowserSession>,
        op: &str,
        params: Value,
        timeout: Duration,
    ) -> Result<Value> {
        let session = Arc::clone(session);
        let op = op.to_string();
        tokio::task::spawn_blocking(move || {
            let mut handle = session
                .handle
                .lock()
                .map_err(|_| anyhow::anyhow!("browser runtime handle poisoned"))?;
            handle.request_with_timeout(&op, params, timeout)
        })
        .await
        .context("browser runtime timed request worker panicked")?
    }

    /// Drop a session from the registry and shut its process down gracefully.
    pub async fn stop(&self, session_id: &str) {
        let removed = self
            .sessions
            .lock()
            .ok()
            .and_then(|mut map| map.remove(session_id));
        if let Some(session) = removed {
            let _ = tokio::task::spawn_blocking(move || {
                if let Ok(mut handle) = session.handle.lock() {
                    handle.shutdown();
                }
            })
            .await;
        }
        self.clear_crash_history(session_id);
    }

    /// Forget a session without a graceful close (used when the process is
    /// already dead). The `Drop` impl on the handle still kills any remnant.
    pub fn drop_session(&self, session_id: &str) {
        if let Ok(mut map) = self.sessions.lock() {
            map.remove(session_id);
        }
    }

    pub fn drop_session_after_crash(&self, session_id: &str) {
        self.drop_session(session_id);
        self.record_crash(session_id);
    }
}

/// Every directory a session's runner file can live in.
///
/// `root` is the CTOX root, not the reference directory -- the generated runner
/// sits under `runtime/browser/interactive-reference` there, while a packaged
/// install can keep it under the state root instead. The runtime decides
/// between them at startup (`browser_runtime_reference_dir` in the peer), so
/// searching only one of them finds nothing on the other kind of install.
fn runner_script_dirs(root: &Path) -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    if let Some(configured) = std::env::var_os("CTOX_WEB_BROWSER_REFERENCE_DIR") {
        dirs.push(PathBuf::from(configured));
    }
    dirs.push(root.join("runtime/browser/interactive-reference"));
    if let Some(home) = std::env::var_os("HOME") {
        dirs.push(PathBuf::from(home).join(".local/state/ctox/browser/interactive-reference"));
    }
    // Kept last so an install that really does keep the runner beside the root
    // still resolves, rather than regressing into "no candidates".
    dirs.push(root.to_path_buf());
    dirs
}

/// Pick the runner file that `pid` was started with out of `root`.
///
/// Several sessions of one CTOX process share a reference directory, so the
/// command line -- which carries the runner path as node's script argument --
/// is what makes the answer session-exact. Comparing file names rather than
/// splitting the command line keeps paths containing spaces working.
fn select_runner_script(dirs: &[PathBuf], pid: u32, command_line: &str) -> Result<PathBuf> {
    let mut only_candidate: Option<PathBuf> = None;
    let mut candidates = 0usize;
    for dir in dirs {
        // A missing directory is not an error: which of the candidates the
        // runtime actually used is decided at startup, so the others simply
        // do not exist on this install.
        let Ok(entries) = std::fs::read_dir(dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            if !name.starts_with(".ctox-browser-live-") || !name.ends_with(".mjs") {
                continue;
            }
            candidates += 1;
            if command_line.contains(&name) {
                return Ok(entry.path());
            }
            only_candidate = Some(entry.path());
        }
    }
    // A single candidate is unambiguous even when the command line could not be
    // read back, so a hardened `ps` still yields the right answer for the
    // common case of one live session.
    if candidates == 1 {
        if let Some(path) = only_candidate {
            return Ok(path);
        }
    }
    anyhow::bail!(
        "found no runner script for pid {pid} among {candidates} candidate(s) in {}",
        dirs.iter()
            .map(|dir| dir.display().to_string())
            .collect::<Vec<_>>()
            .join(", ")
    )
}

/// The command line of the persistent runner process.
///
/// Best effort: an empty string means "could not tell", which
/// [`select_runner_script`] treats as ambiguity rather than as a match.
fn runner_command_line(pid: u32) -> String {
    #[cfg(unix)]
    {
        std::process::Command::new("ps")
            .arg("-o")
            .arg("command=")
            .arg("-p")
            .arg(pid.to_string())
            .output()
            .map(|output| String::from_utf8_lossy(&output.stdout).into_owned())
            .unwrap_or_default()
    }
    #[cfg(not(unix))]
    {
        let _ = pid;
        String::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn web_stack_auth_sessions_use_source_scoped_profiles() {
        let owner = "ctox";
        let dnb = "browser_session_web_stack_auth_dnbhoovers-com";
        let xing = "browser_session_web_stack_auth_xing-com";

        assert_ne!(
            Sha256::digest(browser_profile_key(owner, dnb, false)),
            Sha256::digest(browser_profile_key(owner, xing, false))
        );
        assert_eq!(
            browser_profile_key(owner, "browser_session_regular", false),
            owner
        );
    }

    #[test]
    fn web_stack_public_sessions_use_source_scoped_profiles() {
        let owner = "ctox";
        let companyhouse = "browser_session_web_stack_public_companyhouse-de";
        let northdata = "browser_session_web_stack_public_northdata-de";

        assert_ne!(
            Sha256::digest(browser_profile_key(owner, companyhouse, false)),
            Sha256::digest(browser_profile_key(owner, northdata, false))
        );
        assert_ne!(
            browser_profile_key(owner, companyhouse, false),
            browser_profile_key(owner, "browser_session_regular", false)
        );
    }

    #[test]
    fn runner_script_is_picked_by_command_line_not_by_name_glob() {
        let dir = std::env::temp_dir().join(format!(
            "ctox-runner-select-{}-{}",
            std::process::id(),
            line!()
        ));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let mine = dir.join(".ctox-browser-live-42-1000.mjs");
        let other = dir.join(".ctox-browser-live-42-2000.mjs");
        std::fs::write(&mine, "// mine").expect("write");
        std::fs::write(&other, "// other").expect("write");

        // Two sessions of one CTOX process share the directory, so only the
        // command line tells them apart.
        let command_line = format!("node {}", mine.display());
        let picked =
            select_runner_script(std::slice::from_ref(&dir), 42, &command_line).expect("picked");
        assert_eq!(picked, mine);

        // Without a usable command line the answer stays ambiguous rather than
        // guessing one of the two.
        assert!(select_runner_script(std::slice::from_ref(&dir), 42, "").is_err());

        std::fs::remove_file(&other).expect("remove");
        // A single candidate is unambiguous even with no command line.
        assert_eq!(
            select_runner_script(std::slice::from_ref(&dir), 42, "").expect("sole candidate"),
            mine
        );

        std::fs::remove_file(&mine).expect("remove");
        assert!(select_runner_script(std::slice::from_ref(&dir), 42, "").is_err());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn runner_script_is_found_below_the_root_not_only_beside_it() {
        // The session root is the CTOX root; the runner lives in the reference
        // directory below it. Searching the root alone found nothing on a real
        // install -- the live view answered "0 candidate(s)" while the runner
        // was running the whole time.
        let root = std::env::temp_dir().join(format!(
            "ctox-runner-root-{}-{}",
            std::process::id(),
            line!()
        ));
        let reference = root.join("runtime/browser/interactive-reference");
        std::fs::create_dir_all(&reference).expect("temp dirs");
        let runner = reference.join(".ctox-browser-live-99-1000.mjs");
        std::fs::write(&runner, "// runner").expect("write");

        let dirs = runner_script_dirs(&root);
        assert!(
            dirs.contains(&reference),
            "reference directory must be searched, got {dirs:?}"
        );
        assert_eq!(
            select_runner_script(&dirs, 99, &format!("node {}", runner.display()))
                .expect("found below root"),
            runner
        );
        std::fs::remove_dir_all(&root).ok();
    }
}
