use crate::codex_text_encoding::bytes_to_string_smart;
use crate::contracts::Paths;
use crate::contracts::now_iso;
use crate::runtime_db::record_agent_event;
use anyhow::Context;
use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use codex_app_server_protocol::CommandExecOutputDeltaNotification;
use codex_app_server_protocol::CommandExecOutputStream;
use codex_app_server_protocol::CommandExecParams;
use codex_app_server_protocol::CommandExecResizeParams;
use codex_app_server_protocol::CommandExecResizeResponse;
use codex_app_server_protocol::CommandExecResponse;
use codex_app_server_protocol::CommandExecTerminalSize;
use codex_app_server_protocol::CommandExecTerminateParams;
use codex_app_server_protocol::CommandExecTerminateResponse;
use codex_app_server_protocol::CommandExecWriteParams;
use codex_app_server_protocol::CommandExecWriteResponse;
use codex_shell_command::command_safety::is_dangerous_command::command_might_be_dangerous;
use codex_shell_command::command_safety::is_safe_command::is_known_safe_command;
use codex_shell_command::parse_command::parse_command;
use codex_utils_pty::DEFAULT_OUTPUT_BYTES_CAP;
use codex_utils_pty::ProcessHandle;
use codex_utils_pty::SpawnedProcess;
use codex_utils_pty::TerminalSize;
use codex_utils_pty::spawn_pipe_process;
use codex_utils_pty::spawn_pipe_process_no_stdin;
use codex_utils_pty::spawn_pty_process;
use std::collections::HashMap;
use std::future::Future;
use std::sync::Arc;
use std::sync::LazyLock;
use std::sync::Mutex as StdMutex;
use std::sync::atomic::AtomicI64;
use std::sync::atomic::Ordering;
use tokio::sync::Mutex;
use tokio::sync::mpsc;
use tokio::sync::Notify;
use tokio::sync::oneshot;
use tokio::time::Duration;
use tokio::time::Instant;

static COMMAND_EXEC_MANAGER: LazyLock<CommandExecManager> =
    LazyLock::new(CommandExecManager::default);
const DEFAULT_EXEC_COMMAND_TIMEOUT_MS: u64 = 10_000;
const IO_DRAIN_TIMEOUT_MS: u64 = 2_000;
const EXEC_OUTPUT_DELTA_EVENT_BYTES: usize = 4096;

#[derive(Clone, Debug)]
pub struct SessionSnapshot {
    pub session_id: String,
    pub created_at: String,
    pub status: String,
    pub cwd: String,
    pub tty: bool,
    pub stream_stdin: bool,
    pub stream_stdout_stderr: bool,
    pub output_bytes_cap: Option<usize>,
    pub command: Vec<String>,
    pub exit_code: Option<i32>,
    pub stdout: String,
    pub stderr: String,
}

#[derive(Clone, Debug)]
pub struct OneShotExecResult {
    pub session_id: String,
    pub snapshot: SessionSnapshot,
}

#[derive(Clone)]
struct ManagedSession {
    created_at: String,
    cwd: String,
    tty: bool,
    stream_stdin: bool,
    stream_stdout_stderr: bool,
    output_bytes_cap: Option<usize>,
    command: Vec<String>,
    shared: Arc<SessionShared>,
    control_tx: Option<mpsc::Sender<CommandControlRequest>>,
}

struct SessionShared {
    status: StdMutex<String>,
    finished: StdMutex<bool>,
    exit_code: StdMutex<Option<i32>>,
    stdout: StdMutex<Vec<u8>>,
    stderr: StdMutex<Vec<u8>>,
    completion_notify: Arc<Notify>,
}

impl SessionShared {
    fn new() -> Self {
        Self {
            status: StdMutex::new("active".to_string()),
            finished: StdMutex::new(false),
            exit_code: StdMutex::new(None),
            stdout: StdMutex::new(Vec::new()),
            stderr: StdMutex::new(Vec::new()),
            completion_notify: Arc::new(Notify::new()),
        }
    }
}

#[derive(Default, Clone)]
struct CommandExecManager {
    sessions: Arc<Mutex<HashMap<String, ManagedSession>>>,
    next_id: Arc<AtomicI64>,
}

enum CommandControl {
    Write { delta: Vec<u8>, close_stdin: bool },
    Resize { size: TerminalSize },
    Terminate,
}

struct CommandControlRequest {
    control: CommandControl,
    response_tx: oneshot::Sender<anyhow::Result<()>>,
}

pub fn start_session(paths: &Paths, request: CommandExecParams) -> anyhow::Result<String> {
    run_async(COMMAND_EXEC_MANAGER.start(paths.clone(), request))
}

pub fn write_session(paths: &Paths, request: CommandExecWriteParams) -> anyhow::Result<String> {
    run_async(COMMAND_EXEC_MANAGER.write(paths.clone(), request))
}

pub fn resize_session(paths: &Paths, request: CommandExecResizeParams) -> anyhow::Result<String> {
    run_async(COMMAND_EXEC_MANAGER.resize(paths.clone(), request))
}

pub fn terminate_session(
    paths: &Paths,
    request: CommandExecTerminateParams,
) -> anyhow::Result<String> {
    run_async(COMMAND_EXEC_MANAGER.terminate(paths.clone(), request))
}

pub fn read_session(paths: &Paths, session_id: &str) -> anyhow::Result<String> {
    run_async(COMMAND_EXEC_MANAGER.read(paths.clone(), session_id.to_string()))
}

pub fn list_sessions(paths: &Paths) -> anyhow::Result<String> {
    run_async(COMMAND_EXEC_MANAGER.list(paths.clone()))
}

pub fn snapshot_sessions() -> anyhow::Result<Vec<SessionSnapshot>> {
    run_async(COMMAND_EXEC_MANAGER.snapshots())
}

pub fn snapshot_session(session_id: &str) -> anyhow::Result<Option<SessionSnapshot>> {
    run_async(COMMAND_EXEC_MANAGER.snapshot(session_id.to_string()))
}

pub fn run_one_shot_command(
    paths: &Paths,
    request: CommandExecParams,
) -> anyhow::Result<OneShotExecResult> {
    run_async(COMMAND_EXEC_MANAGER.run_one_shot(paths.clone(), request))
}

impl CommandExecManager {
    async fn start(&self, paths: Paths, request: CommandExecParams) -> anyhow::Result<String> {
        let CommandExecParams {
            command,
            process_id,
            tty,
            stream_stdin,
            stream_stdout_stderr,
            output_bytes_cap,
            disable_output_cap,
            disable_timeout,
            timeout_ms,
            cwd,
            env,
            size,
            sandbox_policy: _,
        } = request;

        if command.is_empty() {
            anyhow::bail!("exec-start requires a non-empty command");
        }
        if size.is_some() && !tty {
            anyhow::bail!("command/exec size requires tty: true");
        }
        if disable_output_cap && output_bytes_cap.is_some() {
            anyhow::bail!("command/exec cannot set both outputBytesCap and disableOutputCap");
        }
        if disable_timeout && timeout_ms.is_some() {
            anyhow::bail!("command/exec cannot set both timeoutMs and disableTimeout");
        }
        if process_id.is_none() && (tty || stream_stdin || stream_stdout_stderr) {
            anyhow::bail!(
                "command/exec tty or streaming requires a client-supplied processId"
            );
        }

        let timeout_ms = match timeout_ms {
            Some(value) => Some(
                u64::try_from(value)
                    .with_context(|| format!("command/exec timeoutMs must be non-negative, got {value}"))?,
            ),
            None => None,
        };
        let output_bytes_cap = if disable_output_cap {
            None
        } else {
            Some(output_bytes_cap.unwrap_or(DEFAULT_OUTPUT_BYTES_CAP))
        };

        let session_id = process_id
            .unwrap_or_else(|| format!("exec-{}", self.next_id.fetch_add(1, Ordering::Relaxed) + 1));
        let cwd = cwd.unwrap_or_else(|| paths.root.clone());
        let stream_stdin = tty || stream_stdin;
        let stream_stdout_stderr = tty || stream_stdout_stderr;
        let (program, args) = command
            .split_first()
            .context("exec-start requires a non-empty command")?;
        let parsed = parse_command(&command);
        let shared = Arc::new(SessionShared::new());
        let (control_tx, control_rx) = mpsc::channel(32);
        let env = codex_exec_env(env);

        let spawned = if tty {
            spawn_pty_process(
                program,
                args,
                cwd.as_path(),
                &env,
                &None,
                terminal_size_from_protocol(size.unwrap_or(CommandExecTerminalSize {
                    rows: 24,
                    cols: 80,
                }))?,
            )
            .await
        } else if stream_stdin {
            spawn_pipe_process(program, args, cwd.as_path(), &env, &None).await
        } else {
            spawn_pipe_process_no_stdin(program, args, cwd.as_path(), &env, &None).await
        }
        .with_context(|| format!("failed to spawn codex-backed exec session {session_id}"))?;

        {
            let mut sessions = self.sessions.lock().await;
            if sessions.contains_key(&session_id) {
                anyhow::bail!("exec session already exists: {session_id}");
            }
            sessions.insert(
                session_id.clone(),
                ManagedSession {
                    created_at: now_iso(),
                    cwd: cwd.display().to_string(),
                    tty,
                    stream_stdin,
                    stream_stdout_stderr,
                    output_bytes_cap,
                    command: command.clone(),
                    shared: shared.clone(),
                    control_tx: Some(control_tx),
                },
            );
        }

        let _ = record_agent_event(
            &paths,
            "exec/start",
            None,
            "",
            &format!("Started codex-backed exec session {session_id}."),
            &serde_json::to_string(&serde_json::json!({
                "sessionId": session_id,
                "command": command,
                "cwd": cwd,
                "tty": tty,
                "streamStdin": stream_stdin,
                "streamStdoutStderr": stream_stdout_stderr,
                "timeoutMs": timeout_ms,
                "outputBytesCap": output_bytes_cap,
                "knownSafeCommand": is_known_safe_command(&command),
                "mightBeDangerousCommand": command_might_be_dangerous(&command),
                "parsedCommand": parsed,
            }))
            .unwrap_or_else(|_| "{}".to_string()),
        );

        let session_id_for_task = session_id.clone();
        tokio::spawn(run_session(
            paths,
            session_id_for_task,
            shared,
            spawned,
            control_rx,
            timeout_ms,
            stream_stdin,
            stream_stdout_stderr,
            output_bytes_cap,
        ));

        Ok(format!("exec session started: {session_id}"))
    }

    async fn write(&self, paths: Paths, request: CommandExecWriteParams) -> anyhow::Result<String> {
        if request.delta_base64.is_none() && !request.close_stdin {
            anyhow::bail!("command/exec/write requires deltaBase64 or closeStdin");
        }
        let delta = match request.delta_base64 {
            Some(value) => STANDARD
                .decode(value)
                .context("invalid deltaBase64 for exec session write")?,
            None => Vec::new(),
        };
        let session_id = request.process_id.clone();
        self.send_control(
            paths,
            session_id.clone(),
            CommandControl::Write {
                delta,
                close_stdin: request.close_stdin,
            },
            "exec/write",
            "Wrote to codex-backed exec session.",
            &CommandExecWriteResponse {},
        )
        .await?;
        Ok(format!("exec session write sent: {session_id}"))
    }

    async fn resize(&self, paths: Paths, request: CommandExecResizeParams) -> anyhow::Result<String> {
        let session_id = request.process_id.clone();
        self.send_control(
            paths,
            session_id.clone(),
            CommandControl::Resize {
                size: terminal_size_from_protocol(request.size)?,
            },
            "exec/resize",
            "Resized codex-backed exec session.",
            &CommandExecResizeResponse {},
        )
        .await?;
        Ok(format!("exec session resized: {session_id}"))
    }

    async fn terminate(
        &self,
        paths: Paths,
        request: CommandExecTerminateParams,
    ) -> anyhow::Result<String> {
        let session_id = request.process_id.clone();
        self.send_control(
            paths,
            session_id.clone(),
            CommandControl::Terminate,
            "exec/terminate",
            "Terminate requested for codex-backed exec session.",
            &CommandExecTerminateResponse {},
        )
        .await?;
        Ok(format!("exec session terminate requested: {session_id}"))
    }

    async fn list(&self, _paths: Paths) -> anyhow::Result<String> {
        let snapshots = self.snapshots().await?;
        if snapshots.is_empty() {
            return Ok("No exec sessions.".to_string());
        }
        let mut lines = Vec::new();
        for snapshot in snapshots {
            let cap = snapshot
                .output_bytes_cap
                .map(|value| value.to_string())
                .unwrap_or_else(|| "unbounded".to_string());
            lines.push(format!(
                "{} :: status={} :: exit={} :: tty={} :: stdin={} :: stdoutStderrStream={} :: outputCap={} :: cwd={} :: {:?}",
                snapshot.session_id,
                snapshot.status,
                snapshot
                    .exit_code
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "none".to_string()),
                snapshot.tty,
                snapshot.stream_stdin,
                snapshot.stream_stdout_stderr,
                cap,
                snapshot.cwd,
                snapshot.command
            ));
        }
        Ok(lines.join("\n"))
    }

    async fn read(&self, _paths: Paths, session_id: String) -> anyhow::Result<String> {
        let snapshot = self
            .snapshot(session_id.clone())
            .await?
            .with_context(|| format!("no exec session found: {session_id}"))?;
        Ok(render_snapshot(&snapshot))
    }

    async fn snapshots(&self) -> anyhow::Result<Vec<SessionSnapshot>> {
        let sessions = self.sessions.lock().await;
        let mut snapshots = Vec::new();
        for (session_id, session) in sessions.iter() {
            snapshots.push(snapshot_from_managed_session(
                session_id.clone(),
                session,
            ));
        }
        snapshots.sort_by(|left, right| left.session_id.cmp(&right.session_id));
        Ok(snapshots)
    }

    async fn snapshot(&self, session_id: String) -> anyhow::Result<Option<SessionSnapshot>> {
        let sessions = self.sessions.lock().await;
        Ok(sessions
            .get(&session_id)
            .map(|session| snapshot_from_managed_session(session_id, session)))
    }

    async fn run_one_shot(
        &self,
        paths: Paths,
        mut request: CommandExecParams,
    ) -> anyhow::Result<OneShotExecResult> {
        if request.tty {
            anyhow::bail!("one-shot command/exec requests must not use tty");
        }
        if request.stream_stdin {
            anyhow::bail!("one-shot command/exec requests must not enable streamStdin");
        }

        let session_id = request.process_id.clone().unwrap_or_else(|| {
            format!(
                "oneshot-exec-{}",
                self.next_id.fetch_add(1, Ordering::Relaxed) + 1
            )
        });
        request.process_id = Some(session_id.clone());
        request.stream_stdout_stderr = false;

        self.start(paths, request).await?;
        self.wait_for_completion(&session_id).await?;
        let snapshot = {
            let sessions = self.sessions.lock().await;
            let session = sessions
                .get(&session_id)
                .with_context(|| format!("completed exec session disappeared: {session_id}"))?;
            snapshot_from_managed_session_with_limit(session_id.clone(), session, None)
        };
        self.remove_session(&session_id).await;

        Ok(OneShotExecResult {
            session_id,
            snapshot,
        })
    }

    async fn send_control<T: serde::Serialize>(
        &self,
        paths: Paths,
        session_id: String,
        control: CommandControl,
        event_method: &str,
        event_body: &str,
        response: &T,
    ) -> anyhow::Result<()> {
        let control_tx = {
            let sessions = self.sessions.lock().await;
            sessions
                .get(&session_id)
                .and_then(|session| session.control_tx.clone())
                .with_context(|| format!("no active exec session found: {session_id}"))?
        };
        let (response_tx, response_rx) = oneshot::channel();
        control_tx
            .send(CommandControlRequest {
                control,
                response_tx,
            })
            .await
            .with_context(|| format!("failed to send exec control to session {session_id}"))?;
        response_rx
            .await
            .with_context(|| format!("exec session {session_id} dropped control response"))??;
        let _ = record_agent_event(
            &paths,
            event_method,
            None,
            "",
            event_body,
            &serde_json::to_string(&serde_json::json!({
                "sessionId": session_id,
                "response": response,
            }))
            .unwrap_or_else(|_| "{}".to_string()),
        );
        Ok(())
    }

    async fn wait_for_completion(&self, session_id: &str) -> anyhow::Result<()> {
        loop {
            let finished = {
                let sessions = self.sessions.lock().await;
                let session = sessions
                    .get(session_id)
                    .with_context(|| format!("no active exec session found: {session_id}"))?;
                session
                    .shared
                    .finished
                    .lock()
                    .ok()
                    .map(|value| *value)
                    .unwrap_or(false)
            };
            if finished {
                return Ok(());
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
    }

    async fn remove_session(&self, session_id: &str) {
        let mut sessions = self.sessions.lock().await;
        sessions.remove(session_id);
    }
}

async fn run_session(
    paths: Paths,
    session_id: String,
    shared: Arc<SessionShared>,
    spawned: SpawnedProcess,
    mut control_rx: mpsc::Receiver<CommandControlRequest>,
    timeout_ms: Option<u64>,
    stream_stdin: bool,
    stream_stdout_stderr: bool,
    output_bytes_cap: Option<usize>,
) {
    let SpawnedProcess {
        session,
        mut stdout_rx,
        mut stderr_rx,
        exit_rx,
    } = spawned;
    let mut exit_rx = exit_rx;
    let mut timeout = Box::pin(async move {
        if let Some(timeout_ms) = timeout_ms {
            tokio::time::sleep(Duration::from_millis(timeout_ms)).await;
        } else {
            std::future::pending::<()>().await;
        }
    });
    let mut saw_exit = false;
    let mut timed_out = false;
    let mut stdout_open = true;
    let mut stderr_open = true;
    let mut stdout_observed = 0usize;
    let mut stderr_observed = 0usize;
    let mut stdout_cap_reached = false;
    let mut stderr_cap_reached = false;
    let mut drain_timeout: Option<Instant> = None;

    loop {
        if saw_exit && !stdout_open && !stderr_open {
            break;
        }
        let drain_deadline = drain_timeout;
        let mut drain_future = Box::pin(async {
            if let Some(deadline) = drain_deadline {
                tokio::time::sleep_until(deadline).await;
            } else {
                std::future::pending::<()>().await;
            }
        });
        tokio::select! {
            maybe_stdout = stdout_rx.recv(), if stdout_open => {
                match maybe_stdout {
                    Some(chunk) => {
                        let (capped_len, cap_reached) = compute_capped_len(
                            chunk.len(),
                            output_bytes_cap,
                            &mut stdout_observed,
                        );
                        let capped_chunk = &chunk[..capped_len];
                        if stream_stdout_stderr {
                            if !stdout_cap_reached {
                                emit_output_delta(
                                    &paths,
                                    &session_id,
                                    CommandExecOutputStream::Stdout,
                                    capped_chunk,
                                    cap_reached,
                                );
                            }
                        } else if !stdout_cap_reached {
                            append_limited(&shared.stdout, capped_chunk);
                        }
                        if cap_reached {
                            stdout_cap_reached = true;
                        }
                    }
                    None => stdout_open = false,
                }
            }
            maybe_stderr = stderr_rx.recv(), if stderr_open => {
                match maybe_stderr {
                    Some(chunk) => {
                        let (capped_len, cap_reached) = compute_capped_len(
                            chunk.len(),
                            output_bytes_cap,
                            &mut stderr_observed,
                        );
                        let capped_chunk = &chunk[..capped_len];
                        if stream_stdout_stderr {
                            if !stderr_cap_reached {
                                emit_output_delta(
                                    &paths,
                                    &session_id,
                                    CommandExecOutputStream::Stderr,
                                    capped_chunk,
                                    cap_reached,
                                );
                            }
                        } else if !stderr_cap_reached {
                            append_limited(&shared.stderr, capped_chunk);
                        }
                        if cap_reached {
                            stderr_cap_reached = true;
                        }
                    }
                    None => stderr_open = false,
                }
            }
            request = control_rx.recv() => {
                match request {
                    Some(CommandControlRequest { control, response_tx }) => {
                        let result = match control {
                            CommandControl::Write { delta, close_stdin } => handle_write(&session, stream_stdin, delta, close_stdin).await,
                            CommandControl::Resize { size } => session.resize(size).context("failed to resize exec session"),
                            CommandControl::Terminate => {
                                session.request_terminate();
                                Ok(())
                            }
                        };
                        let _ = response_tx.send(result);
                    }
                    None => {
                        session.request_terminate();
                    }
                }
            }
            _ = timeout.as_mut(), if !saw_exit && timeout_ms.is_some() && !timed_out => {
                session.request_terminate();
                tokio::time::sleep(Duration::from_millis(250)).await;
                if !session.has_exited() {
                    session.terminate();
                }
                timed_out = true;
                if let Ok(mut status) = shared.status.lock() {
                    *status = "timeout".to_string();
                }
                let _ = record_agent_event(
                    &paths,
                    "exec/timeout",
                    None,
                    "",
                    &format!("Codex-backed exec session {} timed out.", session_id),
                    &serde_json::to_string(&serde_json::json!({
                        "sessionId": session_id,
                        "timeoutMs": timeout_ms,
                    }))
                    .unwrap_or_else(|_| "{}".to_string()),
                );
            }
            result = &mut exit_rx, if !saw_exit => {
                let exit_code = if timed_out {
                    124
                } else {
                    result.unwrap_or(-1)
                };
                if let Ok(mut value) = shared.exit_code.lock() {
                    *value = Some(exit_code);
                }
                if let Ok(mut status) = shared.status.lock() {
                    *status = if timed_out {
                        "timeout".to_string()
                    } else if exit_code == 0 {
                        "completed".to_string()
                    } else {
                        "failed".to_string()
                    };
                }
                let response = CommandExecResponse {
                    exit_code,
                    stdout: bytes_to_string_smart(
                        &shared.stdout.lock().map(|value| value.clone()).unwrap_or_default(),
                    ),
                    stderr: bytes_to_string_smart(
                        &shared.stderr.lock().map(|value| value.clone()).unwrap_or_default(),
                    ),
                };
                let _ = record_agent_event(
                    &paths,
                    "exec/completed",
                    None,
                    "",
                    &format!("Codex-backed exec session {} completed.", session_id),
                    &serde_json::to_string(&serde_json::json!({
                        "sessionId": session_id,
                        "response": response,
                    }))
                    .unwrap_or_else(|_| "{}".to_string()),
                );
                saw_exit = true;
                drain_timeout = Some(Instant::now() + Duration::from_millis(IO_DRAIN_TIMEOUT_MS));
            }
            _ = &mut drain_future, if saw_exit => {
                break;
            }
        }
    }
    if let Ok(mut finished) = shared.finished.lock() {
        *finished = true;
    }
    shared.completion_notify.notify_waiters();
}

async fn handle_write(
    session: &ProcessHandle,
    stream_stdin: bool,
    delta: Vec<u8>,
    close_stdin: bool,
) -> anyhow::Result<()> {
    if !stream_stdin {
        anyhow::bail!("stdin streaming is not enabled for this exec session");
    }
    if !delta.is_empty() {
        session
            .writer_sender()
            .send(delta)
            .await
            .context("failed to write to exec session stdin")?;
    }
    if close_stdin {
        session.close_stdin();
    }
    Ok(())
}

fn snapshot_from_managed_session(
    session_id: String,
    session: &ManagedSession,
) -> SessionSnapshot {
    snapshot_from_managed_session_with_limit(session_id, session, Some(8_000))
}

fn snapshot_from_managed_session_with_limit(
    session_id: String,
    session: &ManagedSession,
    output_char_limit: Option<usize>,
) -> SessionSnapshot {
    let stdout = bytes_to_string_smart(
        &session
            .shared
            .stdout
            .lock()
            .map(|value| value.clone())
            .unwrap_or_default(),
    );
    let stderr = bytes_to_string_smart(
        &session
            .shared
            .stderr
            .lock()
            .map(|value| value.clone())
            .unwrap_or_default(),
    );
    SessionSnapshot {
        session_id,
        created_at: session.created_at.clone(),
        status: session
            .shared
            .status
            .lock()
            .map(|value| value.clone())
            .unwrap_or_else(|_| "poisoned".to_string()),
        cwd: session.cwd.clone(),
        tty: session.tty,
        stream_stdin: session.stream_stdin,
        stream_stdout_stderr: session.stream_stdout_stderr,
        output_bytes_cap: session.output_bytes_cap,
        command: session.command.clone(),
        exit_code: session.shared.exit_code.lock().ok().and_then(|value| *value),
        stdout: output_char_limit
            .map(|limit| trim_output(&stdout, limit))
            .unwrap_or(stdout),
        stderr: output_char_limit
            .map(|limit| trim_output(&stderr, limit))
            .unwrap_or(stderr),
    }
}

fn render_snapshot(snapshot: &SessionSnapshot) -> String {
    let response = CommandExecResponse {
        exit_code: snapshot.exit_code.unwrap_or(-1),
        stdout: snapshot.stdout.clone(),
        stderr: snapshot.stderr.clone(),
    };
    format!(
        "session={} :: status={} :: exit={} :: tty={} :: stdin={} :: stdoutStderrStream={} :: outputCap={} :: cwd={} :: {:?}\n--- command/exec response ---\n{}\n--- stdout ---\n{}\n--- stderr ---\n{}",
        snapshot.session_id,
        snapshot.status,
        snapshot
            .exit_code
            .map(|value| value.to_string())
            .unwrap_or_else(|| "none".to_string()),
        snapshot.tty,
        snapshot.stream_stdin,
        snapshot.stream_stdout_stderr,
        snapshot
            .output_bytes_cap
            .map(|value| value.to_string())
            .unwrap_or_else(|| "unbounded".to_string()),
        snapshot.cwd,
        snapshot.command,
        serde_json::to_string_pretty(&response).unwrap_or_else(|_| "{}".to_string()),
        snapshot.stdout,
        snapshot.stderr
    )
}

fn run_async<F, T>(future: F) -> anyhow::Result<T>
where
    F: Future<Output = anyhow::Result<T>> + Send + 'static,
    T: Send + 'static,
{
    if let Ok(handle) = tokio::runtime::Handle::try_current() {
        return tokio::task::block_in_place(move || handle.block_on(future));
    }

    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("failed to build temporary runtime for command-exec control")?
        .block_on(future)
}

fn append_limited(buffer: &StdMutex<Vec<u8>>, chunk: &[u8]) {
    if chunk.is_empty() {
        return;
    }
    if let Ok(mut buffer) = buffer.lock() {
        buffer.extend_from_slice(chunk);
    }
}

fn compute_capped_len(
    chunk_len: usize,
    output_bytes_cap: Option<usize>,
    observed_num_bytes: &mut usize,
) -> (usize, bool) {
    match output_bytes_cap {
        Some(output_bytes_cap) => {
            let capped_len = output_bytes_cap
                .saturating_sub(*observed_num_bytes)
                .min(chunk_len);
            *observed_num_bytes += capped_len;
            (capped_len, *observed_num_bytes == output_bytes_cap)
        }
        None => (chunk_len, false),
    }
}

fn emit_output_delta(
    paths: &Paths,
    session_id: &str,
    stream: CommandExecOutputStream,
    chunk: &[u8],
    cap_reached: bool,
) {
    let truncated = &chunk[..chunk.len().min(EXEC_OUTPUT_DELTA_EVENT_BYTES)];
    let notification = CommandExecOutputDeltaNotification {
        process_id: session_id.to_string(),
        stream,
        delta_base64: STANDARD.encode(truncated),
        cap_reached,
    };
    let _ = record_agent_event(
        paths,
        "exec/outputDelta",
        None,
        "",
        &format!("{} bytes on {:?} for exec session {}", chunk.len(), stream, session_id),
        &serde_json::to_string(&notification).unwrap_or_else(|_| "{}".to_string()),
    );
}

pub(crate) fn terminal_size_from_protocol(size: CommandExecTerminalSize) -> anyhow::Result<TerminalSize> {
    if size.rows == 0 || size.cols == 0 {
        anyhow::bail!("command/exec size rows and cols must be greater than 0");
    }
    Ok(TerminalSize {
        rows: size.rows,
        cols: size.cols,
    })
}

fn codex_exec_env(
    env_overrides: Option<HashMap<String, Option<String>>>,
) -> HashMap<String, String> {
    let mut env = HashMap::from([
        ("NO_COLOR".to_string(), "1".to_string()),
        ("TERM".to_string(), "xterm-256color".to_string()),
        ("LANG".to_string(), "C.UTF-8".to_string()),
        ("LC_CTYPE".to_string(), "C.UTF-8".to_string()),
        ("LC_ALL".to_string(), "C.UTF-8".to_string()),
        ("PAGER".to_string(), "cat".to_string()),
        ("GIT_PAGER".to_string(), "cat".to_string()),
        ("GH_PAGER".to_string(), "cat".to_string()),
        ("CODEX_CI".to_string(), "1".to_string()),
    ]);
    if let Some(env_overrides) = env_overrides {
        for (key, value) in env_overrides {
            match value {
                Some(value) => {
                    env.insert(key, value);
                }
                None => {
                    env.remove(&key);
                }
            }
        }
    }
    env
}

fn trim_output(value: &str, max_chars: usize) -> String {
    if value.chars().count() <= max_chars {
        value.to_string()
    } else {
        value.chars().take(max_chars).collect::<String>() + "..."
    }
}
