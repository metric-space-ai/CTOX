use crate::bootstrap::handle_attach_line_detailed;
use crate::contracts::load_bios;
use crate::contracts::Paths;
use crate::runtime_db::AgentEventRecord;
use crate::runtime_db::AgentThreadRecord;
use crate::runtime_db::AgentTurnRecord;
use crate::runtime_db::FocusStateRecord;
use crate::runtime_db::TaskRecord;
use crate::runtime_db::list_agent_events_since;
use crate::runtime_db::list_open_tasks;
use crate::runtime_db::list_recent_agent_events;
use crate::runtime_db::load_active_agent_turn;
use crate::runtime_db::load_agent_thread;
use crate::runtime_db::load_focus_state;
use crate::runtime_db::load_latest_completed_agent_turn;
use anyhow::Context;
use chrono::DateTime;
use chrono::Local;
use chrono::Utc;
use crossterm::cursor::MoveTo;
use crossterm::event;
use crossterm::event::Event as TerminalEvent;
use crossterm::event::KeyCode;
use crossterm::event::KeyEvent;
use crossterm::event::KeyModifiers;
use crossterm::execute;
use crossterm::queue;
use crossterm::style::Print;
use crossterm::terminal;
use crossterm::terminal::Clear;
use crossterm::terminal::ClearType;
use crossterm::terminal::EnterAlternateScreen;
use crossterm::terminal::LeaveAlternateScreen;
use serde::Deserialize;
use serde::Serialize;
use std::collections::VecDeque;
use std::env;
use std::fs;
use std::io;
use std::io::BufRead;
use std::io::BufReader;
use std::io::IsTerminal;
use std::io::Write;
use std::os::unix::net::UnixStream as StdUnixStream;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::thread;
use std::time::Duration;
use std::time::Instant;
use tokio::io::AsyncBufReadExt;
use tokio::io::AsyncWriteExt;
use tokio::io::BufReader as TokioBufReader;
use tokio::net::UnixListener;
use tokio::net::UnixStream;

const EVENT_HISTORY_LIMIT: usize = 120;
const PENDING_PREVIEW_LIMIT: usize = 3;
const LOCAL_NOTICE_LIMIT: usize = 80;

#[derive(Debug, Serialize, Deserialize)]
struct AttachRequest {
    line: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct AttachResponse {
    ok: bool,
    output: String,
    #[serde(default)]
    queued_task_id: Option<i64>,
    #[serde(default)]
    queued_task_title: Option<String>,
}

#[derive(Debug, Clone)]
struct AttachUiSnapshot {
    bios_url: Option<String>,
    thread: Option<AgentThreadRecord>,
    focus: Option<FocusStateRecord>,
    active_turn: Option<AgentTurnRecord>,
    last_turn: Option<AgentTurnRecord>,
    open_tasks: Vec<TaskRecord>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PendingInputStatus {
    Queued,
    Active,
    Completed,
}

#[derive(Debug, Clone)]
struct PendingInputItem {
    task_id: Option<i64>,
    task_title: String,
    message: String,
    status: PendingInputStatus,
    completed_at: Option<Instant>,
    seen_open: bool,
}

struct AttachTui {
    snapshot: AttachUiSnapshot,
    event_lines: Vec<String>,
    local_notices: VecDeque<String>,
    pending_inputs: Vec<PendingInputItem>,
    input: String,
    spinner_phase: usize,
}

enum TuiControl {
    Continue,
    Exit,
}

struct TerminalUiGuard;

impl TerminalUiGuard {
    fn enter(stdout: &mut io::Stdout) -> anyhow::Result<Self> {
        terminal::enable_raw_mode().context("failed to enable raw mode")?;
        execute!(stdout, EnterAlternateScreen).context("failed to enter alternate screen")?;
        Ok(Self)
    }
}

impl Drop for TerminalUiGuard {
    fn drop(&mut self) {
        let _ = terminal::disable_raw_mode();
        let mut stdout = io::stdout();
        let _ = execute!(stdout, LeaveAlternateScreen);
    }
}

pub fn spawn_attach_server(paths: Paths) {
    tokio::spawn(async move {
        if let Err(err) = run_attach_server(paths).await {
            eprintln!("attach server failed: {err}");
        }
    });
}

async fn run_attach_server(paths: Paths) -> anyhow::Result<()> {
    if paths.attach_socket_path.exists() {
        fs::remove_file(&paths.attach_socket_path).with_context(|| {
            format!(
                "failed to remove stale attach socket {}",
                paths.attach_socket_path.display()
            )
        })?;
    }

    let listener = UnixListener::bind(&paths.attach_socket_path).with_context(|| {
        format!(
            "failed to bind attach socket {}",
            paths.attach_socket_path.display()
        )
    })?;
    eprintln!(
        "CTO-Agent attach socket listening on {}",
        paths.attach_socket_path.display()
    );

    loop {
        let (stream, _) = listener.accept().await?;
        let connection_paths = paths.clone();
        tokio::spawn(async move {
            if let Err(err) = handle_connection(connection_paths, stream).await {
                eprintln!("attach connection failed: {err}");
            }
        });
    }
}

async fn handle_connection(paths: Paths, stream: UnixStream) -> anyhow::Result<()> {
    let (reader_half, mut writer_half) = stream.into_split();
    let mut lines = TokioBufReader::new(reader_half).lines();

    while let Some(line) = lines.next_line().await? {
        let request: AttachRequest =
            serde_json::from_str(&line).context("failed to parse attach request")?;
        let response = match handle_attach_line_detailed(&paths, &request.line) {
            Ok(outcome) => AttachResponse {
                ok: true,
                output: outcome.output,
                queued_task_id: outcome.queued_task_id,
                queued_task_title: outcome.queued_task_title,
            },
            Err(err) => AttachResponse {
                ok: false,
                output: err.to_string(),
                queued_task_id: None,
                queued_task_title: None,
            },
        };
        let encoded = serde_json::to_string(&response)?;
        writer_half.write_all(encoded.as_bytes()).await?;
        writer_half.write_all(b"\n").await?;
    }

    Ok(())
}

pub fn run_attach_cli(paths: &Paths, args: &[String]) -> anyhow::Result<()> {
    if !args.is_empty() {
        let output = send_attach_line(paths, &args.join(" "))?;
        println!("{output}");
        return Ok(());
    }

    if io::stdin().is_terminal() && io::stdout().is_terminal() {
        match run_attach_tui(paths) {
            Ok(()) => return Ok(()),
            Err(err) => {
                eprintln!("attach tui unavailable, falling back to line mode: {err}");
            }
        }
    }

    run_attach_line_cli(paths)
}

pub fn send_attach_line(paths: &Paths, line: &str) -> anyhow::Result<String> {
    send_attach_request(paths, line).map(|response| response.output)
}

fn send_attach_request(paths: &Paths, line: &str) -> anyhow::Result<AttachResponse> {
    let stream = StdUnixStream::connect(&paths.attach_socket_path);
    let mut stream = match stream {
        Ok(stream) => stream,
        Err(_) => {
            let outcome = handle_attach_line_detailed(paths, line)?;
            return Ok(AttachResponse {
                ok: true,
                output: outcome.output,
                queued_task_id: outcome.queued_task_id,
                queued_task_title: outcome.queued_task_title,
            });
        }
    };
    let request = AttachRequest {
        line: line.trim().to_string(),
    };
    let encoded = serde_json::to_string(&request)?;
    stream.write_all(encoded.as_bytes())?;
    stream.write_all(b"\n")?;
    stream.flush()?;

    let mut reader = BufReader::new(stream);
    let mut response_line = String::new();
    reader.read_line(&mut response_line)?;
    if response_line.trim().is_empty() {
        anyhow::bail!("attach socket returned empty response");
    }
    let response: AttachResponse =
        serde_json::from_str(response_line.trim()).context("failed to parse attach response")?;
    if response.ok {
        Ok(response)
    } else {
        anyhow::bail!(response.output)
    }
}

fn run_attach_tui(paths: &Paths) -> anyhow::Result<()> {
    let mut stdout = io::stdout();
    let _guard = TerminalUiGuard::enter(&mut stdout)?;
    let mut ui = AttachTui::new(paths);
    ui.render(&mut stdout)?;

    let mut last_refresh = Instant::now();
    loop {
        if event::poll(Duration::from_millis(125)).context("failed to poll terminal events")? {
            match event::read().context("failed to read terminal event")? {
                TerminalEvent::Key(key_event) => {
                    if matches!(ui.handle_key_event(paths, key_event), TuiControl::Exit) {
                        break;
                    }
                }
                TerminalEvent::Paste(text) => {
                    ui.input.push_str(&text);
                }
                TerminalEvent::Resize(_, _) => {}
                _ => {}
            }
        }

        if last_refresh.elapsed() >= Duration::from_millis(350) {
            ui.refresh(paths);
            last_refresh = Instant::now();
        }

        ui.spinner_phase = (ui.spinner_phase + 1) % 4;
        ui.render(&mut stdout)?;
    }

    Ok(())
}

fn run_attach_line_cli(paths: &Paths) -> anyhow::Result<()> {
    println!("{}", render_attach_banner(paths));
    let display_lock = Arc::new(Mutex::new(()));
    let recent_events = list_recent_agent_events(paths, 8)?;
    let mut last_seen_id = 0_i64;
    println!("Live-Ereignisse:");
    if !recent_events.is_empty() {
        let _guard = display_lock.lock().expect("display lock poisoned");
        for event in recent_events.iter().rev() {
            println!("{}", format_event_line(event));
            last_seen_id = last_seen_id.max(event.id);
        }
    } else {
        println!("  Noch keine Agentenereignisse vorhanden.");
    }
    let running = Arc::new(AtomicBool::new(true));
    let tail_running = running.clone();
    let tail_paths = paths.clone();
    let tail_lock = display_lock.clone();
    let tail_thread = thread::spawn(move || {
        let mut seen = last_seen_id;
        while tail_running.load(Ordering::Relaxed) {
            match list_agent_events_since(&tail_paths, seen, 24) {
                Ok(events) if !events.is_empty() => {
                    let _guard = tail_lock.lock().expect("display lock poisoned");
                    for event in events {
                        seen = seen.max(event.id);
                        println!("\r{}", format_event_line(&event));
                    }
                    print!("interrupt> ");
                    let _ = io::stdout().flush();
                }
                Ok(_) => {}
                Err(_) => {}
            }
            thread::sleep(Duration::from_millis(350));
        }
    });
    let stdin = io::stdin();
    let mut lock = stdin.lock();
    let mut line = String::new();
    loop {
        {
            let _guard = display_lock.lock().expect("display lock poisoned");
            print!("interrupt> ");
            io::stdout().flush()?;
        }
        line.clear();
        match lock.read_line(&mut line) {
            Ok(0) => {
                println!();
                break;
            }
            Ok(_) => {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }
                let output = send_attach_line(paths, trimmed)?;
                let _guard = display_lock.lock().expect("display lock poisoned");
                println!("{output}");
            }
            Err(err) => return Err(err.into()),
        }
    }
    running.store(false, Ordering::Relaxed);
    let _ = tail_thread.join();
    Ok(())
}

impl AttachTui {
    fn new(paths: &Paths) -> Self {
        let mut ui = Self {
            snapshot: collect_ui_snapshot(paths),
            event_lines: Vec::new(),
            local_notices: VecDeque::new(),
            pending_inputs: Vec::new(),
            input: String::new(),
            spinner_phase: 0,
        };
        ui.refresh(paths);
        ui
    }

    fn refresh(&mut self, paths: &Paths) {
        self.snapshot = collect_ui_snapshot(paths);
        self.event_lines = list_recent_agent_events(paths, EVENT_HISTORY_LIMIT)
            .unwrap_or_default()
            .into_iter()
            .rev()
            .map(|event| format_activity_line(&event))
            .collect();
        self.update_pending_inputs();
    }

    fn handle_key_event(&mut self, paths: &Paths, key_event: KeyEvent) -> TuiControl {
        match key_event {
            KeyEvent {
                code: KeyCode::Char('c'),
                modifiers,
                ..
            } if modifiers.contains(KeyModifiers::CONTROL) => return TuiControl::Exit,
            KeyEvent {
                code: KeyCode::Char('d'),
                modifiers,
                ..
            } if modifiers.contains(KeyModifiers::CONTROL) && self.input.is_empty() => {
                return TuiControl::Exit;
            }
            KeyEvent {
                code: KeyCode::Esc, ..
            } => {
                self.input.clear();
            }
            KeyEvent {
                code: KeyCode::Backspace,
                ..
            } => {
                self.input.pop();
            }
            KeyEvent {
                code: KeyCode::Enter,
                ..
            } => {
                if let Err(err) = self.submit_input(paths) {
                    self.push_local_notice("error", &err.to_string());
                }
            }
            KeyEvent {
                code: KeyCode::Char(character),
                modifiers,
                ..
            } if !modifiers.contains(KeyModifiers::CONTROL)
                && !modifiers.contains(KeyModifiers::ALT) =>
            {
                self.input.push(character);
            }
            _ => {}
        }

        TuiControl::Continue
    }

    fn submit_input(&mut self, paths: &Paths) -> anyhow::Result<()> {
        let trimmed = self.input.trim().to_string();
        if trimmed.is_empty() {
            return Ok(());
        }

        let response = send_attach_request(paths, &trimmed)?;
        if trimmed.starts_with('/') {
            self.push_local_notice("cmd", &trimmed);
            self.push_local_notice("ack", first_visible_line(&response.output));
        } else {
            self.push_local_notice("you", &trimmed);
            self.pending_inputs.push(PendingInputItem {
                task_id: response.queued_task_id,
                task_title: response
                    .queued_task_title
                    .clone()
                    .unwrap_or_else(|| trimmed.clone()),
                message: trimmed.clone(),
                status: PendingInputStatus::Queued,
                completed_at: None,
                seen_open: false,
            });
            if let Some(task_id) = response.queued_task_id {
                let task_title = response
                    .queued_task_title
                    .clone()
                    .unwrap_or_else(|| trimmed.clone());
                self.push_local_notice("queued", &format!("#{task_id} {task_title}"));
            } else {
                self.push_local_notice("ack", first_visible_line(&response.output));
            }
        }

        self.input.clear();
        self.refresh(paths);
        Ok(())
    }

    fn push_local_notice(&mut self, label: &str, text: &str) {
        let timestamp = Local::now().format("%H:%M").to_string();
        for line in text.lines().map(str::trim).filter(|line| !line.is_empty()) {
            self.local_notices.push_back(format!(
                "{timestamp}  {:<6}  {}",
                label,
                compact_text(line, 140)
            ));
        }
        while self.local_notices.len() > LOCAL_NOTICE_LIMIT {
            self.local_notices.pop_front();
        }
    }

    fn update_pending_inputs(&mut self) {
        let now = Instant::now();
        let active_task_id = self.snapshot.focus.as_ref().and_then(|focus| focus.active_task_id);
        let active_turn_task_id = self.snapshot.active_turn.as_ref().map(|turn| turn.task_id);
        let last_turn_task_id = self.snapshot.last_turn.as_ref().map(|turn| turn.task_id);

        for item in &mut self.pending_inputs {
            let open_task = item.task_id.and_then(|task_id| {
                self.snapshot
                    .open_tasks
                    .iter()
                    .find(|task| task.id == task_id)
            });

            if let Some(task) = open_task {
                item.seen_open = true;
                item.task_title = task.title.clone();
                item.status = if task.status == "active" || active_task_id == Some(task.id) {
                    PendingInputStatus::Active
                } else {
                    PendingInputStatus::Queued
                };
                item.completed_at = None;
                continue;
            }

            if item.task_id.is_some()
                && (active_task_id == item.task_id || active_turn_task_id == item.task_id)
            {
                item.status = PendingInputStatus::Active;
                item.completed_at = None;
                continue;
            }

            if item.task_id.is_some() && (last_turn_task_id == item.task_id || item.seen_open) {
                item.status = PendingInputStatus::Completed;
                if item.completed_at.is_none() {
                    item.completed_at = Some(now);
                }
            }
        }

        self.pending_inputs.retain(|item| {
            item.status != PendingInputStatus::Completed
                || item
                    .completed_at
                    .map(|completed_at| now.duration_since(completed_at) < Duration::from_secs(6))
                    .unwrap_or(true)
        });
    }

    fn render(&self, stdout: &mut io::Stdout) -> anyhow::Result<()> {
        let (width, height) = terminal::size().context("failed to read terminal size")?;
        let width_usize = usize::from(width);
        let height_usize = usize::from(height);

        if width_usize < 72 || height_usize < 20 {
            return self.render_compact(stdout, width_usize, height_usize);
        }

        let frame_width = width_usize;
        let minimum_activity_rows = 3usize;
        let fixed_rows = 1 + 2 + 1 + 4 + 1 + 4 + 1 + 1 + 1 + 2;
        let available_activity = height_usize
            .saturating_sub(fixed_rows)
            .max(minimum_activity_rows);

        let mut frame_lines = Vec::new();
        frame_lines.push(frame_top(
            "CTO-Agent Attach",
            &self.primary_status_line(),
            frame_width,
        ));
        frame_lines.push(frame_row(
            &format!(
                "BIOS {}",
                self.snapshot
                    .bios_url
                    .as_deref()
                    .unwrap_or("unavailable")
            ),
            frame_width,
        ));
        frame_lines.push(frame_row(&self.secondary_status_line(), frame_width));
        frame_lines.push(frame_separator("Now", frame_width));
        frame_lines.extend(self.now_rows(frame_width));
        frame_lines.push(frame_separator("Next", frame_width));
        frame_lines.extend(self.next_rows(frame_width));
        frame_lines.push(frame_separator("Activity", frame_width));
        frame_lines.extend(self.activity_rows(frame_width, available_activity));
        frame_lines.push(frame_bottom(frame_width));

        let input_hint = if self.snapshot.active_turn.is_some() {
            "Enter queues for next reprioritize · /status for details · Ctrl-C exit · Esc clears draft"
        } else {
            "Enter sends command or interrupt · /status for details · Ctrl-C exit · Esc clears draft"
        };
        let prompt = "interrupt> ";
        let (input_display, cursor_col) = input_display(&self.input, width_usize, prompt);

        let mut screen_lines = Vec::new();
        screen_lines.extend(frame_lines);
        screen_lines.push(input_hint.to_string());
        screen_lines.push(format!("{prompt}{input_display}"));

        let fitted_lines = screen_lines
            .into_iter()
            .take(height_usize)
            .map(|line| fit_line(&line, width_usize))
            .collect::<Vec<_>>();
        let cursor_row = fitted_lines.len().saturating_sub(1) as u16;
        let cursor_col = cursor_col.min(width_usize.saturating_sub(1)) as u16;
        render_screen_lines(stdout, &fitted_lines, cursor_col, cursor_row)
    }

    fn render_compact(
        &self,
        stdout: &mut io::Stdout,
        width: usize,
        height: usize,
    ) -> anyhow::Result<()> {
        let prompt = "interrupt> ";
        let (input_text, cursor_col) = input_display(&self.input, width, prompt);
        let mode = self
            .snapshot
            .thread
            .as_ref()
            .map(|thread| short_mode(&thread.current_mode))
            .unwrap_or_else(|| "UNKNOWN".to_string());
        let active_task = self
            .snapshot
            .focus
            .as_ref()
            .map(|focus| describe_active_task(&focus.active_task_id, &focus.active_task_title))
            .unwrap_or_else(|| "keine laufende Aufgabe".to_string());
        let status_line = match &self.snapshot.active_turn {
            Some(turn) => {
                let elapsed = elapsed_text(&turn.created_at).unwrap_or_else(|| "n/a".to_string());
                format!("Working {elapsed} | {} | {}", mode, compact_text(&active_task, 72))
            }
            None => format!("Idle | {} | {}", mode, compact_text(&active_task, 72)),
        };
        let pending_count = self.pending_inputs.len();
        let latest_event = self
            .local_notices
            .back()
            .cloned()
            .or_else(|| self.event_lines.last().cloned())
            .unwrap_or_else(|| "Noch keine Events.".to_string());

        let mut lines = vec![
            "=== CTO-Agent Attach Terminal ===".to_string(),
            status_line,
            format!("Queued Inputs: {pending_count}"),
            fit_line(&latest_event, width),
            "Enter queues input | Ctrl-C exit | /status commands".to_string(),
            format!("{prompt}{input_text}"),
        ];
        if lines.len() > height {
            let keep = height.saturating_sub(1);
            let mut compact = vec!["=== CTO-Agent Attach Terminal ===".to_string()];
            compact.extend(lines.into_iter().rev().take(keep).collect::<Vec<_>>().into_iter().rev());
            lines = compact;
        }

        let fitted = lines
            .into_iter()
            .take(height)
            .map(|line| fit_line(&line, width))
            .collect::<Vec<_>>();
        let cursor_row = fitted.len().saturating_sub(1) as u16;
        let cursor_col = cursor_col.min(width.saturating_sub(1)) as u16;
        render_screen_lines(stdout, &fitted, cursor_col, cursor_row)
    }

    fn primary_status_line(&self) -> String {
        let lifecycle = self
            .snapshot
            .thread
            .as_ref()
            .map(|thread| thread.lifecycle_status.as_str())
            .unwrap_or("unknown");
        let queue_depth = self
            .snapshot
            .thread
            .as_ref()
            .map(|thread| thread.queue_depth.to_string())
            .unwrap_or_else(|| "?".to_string());
        let mode = self
            .snapshot
            .thread
            .as_ref()
            .map(|thread| short_mode(&thread.current_mode))
            .unwrap_or_else(|| "UNKNOWN".to_string());
        let turn = self
            .snapshot
            .active_turn
            .as_ref()
            .map(|turn| format!("turn #{}", turn.id))
            .unwrap_or_else(|| "no active turn".to_string());
        format!(
            "{}  •  {}  •  queue {}  •  {}",
            lifecycle.to_uppercase(),
            mode,
            queue_depth,
            turn
        )
    }

    fn secondary_status_line(&self) -> String {
        if let Some(focus) = self.snapshot.focus.as_ref()
            && (focus.active_task_id.is_some() || !focus.active_task_title.trim().is_empty())
        {
            return format!(
                "Task {}",
                describe_active_task(&focus.active_task_id, &focus.active_task_title)
            );
        }
        if let Some(turn) = self.snapshot.active_turn.as_ref() {
            return format!(
                "Task #{} {}",
                turn.task_id,
                compact_text(&turn.task_title, 140)
            );
        }
        "Task keine laufende Aufgabe".to_string()
    }

    fn now_rows(&self, width: usize) -> Vec<String> {
        let mut rows = Vec::new();
        if let Some(turn) = &self.snapshot.active_turn {
            let elapsed = elapsed_text(&turn.created_at).unwrap_or_else(|| "n/a".to_string());
            let spinner = match self.spinner_phase % 4 {
                0 => '●',
                1 => '◐',
                2 => '○',
                _ => '◑',
            };
            rows.push(frame_row(
                &format!("#{} {}", turn.task_id, compact_text(&turn.task_title, 120)),
                width,
            ));
            rows.push(frame_row(
                &format!(
                    "{spinner} running  •  {}  •  {}",
                    elapsed,
                    short_mode(&turn.mode_at_start)
                ),
                width,
            ));
            rows.push(frame_row(
                &format!(
                    "Summary  {}",
                    compact_text(
                        turn.summary
                            .as_deref()
                            .filter(|summary| !summary.trim().is_empty())
                            .unwrap_or("Bounded step läuft gerade."),
                        140
                    )
                ),
                width,
            ));
            let last_done = self
                .snapshot
                .last_turn
                .as_ref()
                .map(|turn| {
                    format!(
                        "Last done  #{} {}",
                        turn.task_id,
                        compact_text(&turn.task_title, 100)
                    )
                })
                .unwrap_or_else(|| "Last done  none".to_string());
            rows.push(frame_row(&last_done, width));
        } else if let Some(turn) = &self.snapshot.last_turn {
            rows.push(frame_row("Idle. Kein bounded Turn läuft gerade.", width));
            rows.push(frame_row(
                &format!("#{} {}", turn.task_id, compact_text(&turn.task_title, 120)),
                width,
            ));
            rows.push(frame_row(
                &format!("Status  {}", turn.status),
                width,
            ));
            rows.push(frame_row(
                &format!(
                    "Summary  {}",
                    compact_text(
                        turn.summary
                            .as_deref()
                            .filter(|summary| !summary.trim().is_empty())
                            .unwrap_or("Keine Summary gespeichert."),
                        140
                    )
                ),
                width,
            ));
        } else {
            rows.push(frame_row("Noch keine bounded Turns vorhanden.", width));
            rows.push(frame_row("Sobald ein Task gestartet wird, erscheint hier der Live-Status.", width));
            rows.push(frame_row("Typing now creates an interrupt task for the next cycle.", width));
            rows.push(frame_row("Use /status, /turns or /events for deeper inspection.", width));
        }
        rows
    }

    fn next_rows(&self, width: usize) -> Vec<String> {
        let mut rows = Vec::new();
        if !self.pending_inputs.is_empty() {
            for item in self.pending_inputs.iter().take(PENDING_PREVIEW_LIMIT) {
                let icon = match item.status {
                    PendingInputStatus::Queued => "○",
                    PendingInputStatus::Active => "●",
                    PendingInputStatus::Completed => "✓",
                };
                let task_ref = item
                    .task_id
                    .map(|task_id| format!("#{} ", task_id))
                    .unwrap_or_default();
                let label = if item.task_title.trim().is_empty() {
                    item.message.as_str()
                } else {
                    item.task_title.as_str()
                };
                rows.push(frame_row(
                    &format!("{icon} {}{}", task_ref, compact_text(label, 120)),
                    width,
                ));
            }
            if self.pending_inputs.len() > PENDING_PREVIEW_LIMIT {
                rows.push(frame_row(
                    &format!(
                        "+ {} weitere vorgemerkte Eingabe(n)",
                        self.pending_inputs.len() - PENDING_PREVIEW_LIMIT
                    ),
                    width,
                ));
            }
            while rows.len() < 4 {
                rows.push(frame_row(" ", width));
            }
            return rows;
        }

        let mut next_tasks = self
            .snapshot
            .open_tasks
            .iter()
            .filter(|task| Some(task.id) != self.snapshot.focus.as_ref().and_then(|focus| focus.active_task_id))
            .take(PENDING_PREVIEW_LIMIT)
            .map(|task| format!("○ #{} {}", task.id, compact_text(&task.title, 118)))
            .collect::<Vec<_>>();

        if next_tasks.is_empty() {
            next_tasks.push("Keine vorgemerkten Owner-Interrupts. Neue Eingaben erscheinen hier.".to_string());
        }

        for line in next_tasks {
            rows.push(frame_row(&line, width));
        }
        while rows.len() < 4 {
            rows.push(frame_row(" ", width));
        }
        rows
    }

    fn activity_rows(&self, width: usize, height: usize) -> Vec<String> {
        let mut merged = self.event_lines.clone();
        merged.extend(self.local_notices.iter().cloned());
        if merged.is_empty() {
            return vec![frame_row("Noch keine Aktivitaet vorhanden.", width)];
        }

        let start = merged.len().saturating_sub(height);
        let mut rows = merged[start..]
            .iter()
            .map(|line| frame_row(line, width))
            .collect::<Vec<_>>();
        while rows.len() < height {
            rows.push(frame_row(" ", width));
        }
        rows
    }
}

fn collect_ui_snapshot(paths: &Paths) -> AttachUiSnapshot {
    let bios = load_bios(paths);
    AttachUiSnapshot {
        bios_url: bios_url(&bios.website_path),
        thread: load_agent_thread(paths).ok(),
        focus: load_focus_state(paths).ok(),
        active_turn: load_active_agent_turn(paths).ok().flatten(),
        last_turn: load_latest_completed_agent_turn(paths).ok().flatten(),
        open_tasks: list_open_tasks(paths, 8).unwrap_or_default(),
    }
}

fn render_attach_banner(paths: &Paths) -> String {
    let snapshot = collect_ui_snapshot(paths);
    let mut lines = vec!["=== CTO-Agent Attach Terminal ===".to_string()];
    if let Some(url) = snapshot.bios_url {
        lines.push(format!("Kleinhirn-BIOS: {url}"));
    }
    if let Some(thread) = snapshot.thread {
        lines.push(format!(
            "Loop: lifecycle={} | mode={} | queue_depth={}",
            thread.lifecycle_status,
            describe_mode(&thread.current_mode),
            thread.queue_depth
        ));
        if !thread.note.trim().is_empty() {
            lines.push(format!("Loop-Notiz: {}", compact_text(&thread.note, 220)));
        }
    }
    if let Some(focus) = snapshot.focus {
        lines.push(format!(
            "Aktive Aufgabe: {}",
            describe_active_task(&focus.active_task_id, &focus.active_task_title)
        ));
        if !focus.note.trim().is_empty() {
            lines.push(format!("Fokus: {}", compact_text(&focus.note, 220)));
        }
    }
    if let Some(turn) = snapshot.active_turn {
        lines.push(format!(
            "Laufender Turn: turn#{} | task#{} {} | start_mode={}",
            turn.id,
            turn.task_id,
            compact_text(&turn.task_title, 120),
            describe_mode(&turn.mode_at_start)
        ));
        lines.push(format!(
            "Turn-Zusammenfassung: {}",
            compact_text(
                turn.summary
                    .as_deref()
                    .filter(|summary| !summary.trim().is_empty())
                    .unwrap_or("noch offen, der aktuelle bounded Schritt laeuft gerade."),
                240
            )
        ));
    } else if let Some(turn) = snapshot.last_turn {
        lines.push(format!(
            "Letzter Turn: turn#{} | task#{} {} | status={}",
            turn.id,
            turn.task_id,
            compact_text(&turn.task_title, 120),
            turn.status
        ));
        lines.push(format!(
            "Letzte Zusammenfassung: {}",
            compact_text(
                turn.summary
                    .as_deref()
                    .filter(|summary| !summary.trim().is_empty())
                    .unwrap_or("keine Summary gespeichert"),
                240
            )
        ));
    } else {
        lines.push("Turn-Zusammenfassung: noch keine bounded Turns vorhanden.".to_string());
    }
    lines.push(
        "Interrupt-Semantik: Jede freie Eingabe wird als Interrupt aufgenommen. Der laufende bounded Schritt wird erst sauber beendet; danach schaut der Agent im naechsten Repriorisierungszyklus auf die neue Eingabe.".to_string(),
    );
    lines.push("Kurzbefehle: /help | /status | /turns | /events".to_string());
    lines.push("-----------------------------------".to_string());
    lines.join("\n")
}

fn bios_url(website_path: &str) -> Option<String> {
    let path = website_path.trim();
    if path.is_empty() {
        return None;
    }
    if path.starts_with("http://") || path.starts_with("https://") {
        return Some(path.to_string());
    }
    let port = env::var("CTO_AGENT_PORT")
        .ok()
        .and_then(|raw| raw.parse::<u16>().ok())
        .unwrap_or(8443);
    let normalized_path = if path.starts_with('/') {
        path.to_string()
    } else {
        format!("/{path}")
    };
    Some(format!("https://127.0.0.1:{port}{normalized_path}"))
}

fn describe_mode(mode: &str) -> String {
    match mode {
        "reprioritize" => "reprioritize (Meta-/Priorisierungsmodus)".to_string(),
        "execute_task" => "execute_task (Aufgabenmodus)".to_string(),
        "recovery" => "recovery (Stabilisierung nach Restart/Incident)".to_string(),
        "self_preservation" => "self_preservation (Loop-Schutz)".to_string(),
        other => other.to_string(),
    }
}

fn describe_active_task(active_task_id: &Option<i64>, active_task_title: &str) -> String {
    match active_task_id {
        Some(task_id) if !active_task_title.trim().is_empty() => {
            format!("#{} {}", task_id, compact_text(active_task_title, 140))
        }
        Some(task_id) => format!("#{}", task_id),
        None if active_task_title.trim().is_empty() => "keine laufende Aufgabe".to_string(),
        None => compact_text(active_task_title, 140),
    }
}

fn short_mode(mode: &str) -> String {
    match mode {
        "reprioritize" => "META".to_string(),
        "execute_task" => "EXEC".to_string(),
        "recovery" => "RECOVERY".to_string(),
        "self_preservation" => "GUARD".to_string(),
        other => other.to_uppercase(),
    }
}

fn first_visible_line(text: &str) -> &str {
    text.lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .unwrap_or(text.trim())
}

fn frame_top(title: &str, right: &str, width: usize) -> String {
    if width <= 2 {
        return fit_line(title, width);
    }
    let inner = width - 2;
    let left = format!("─ {} ", title);
    let right = if right.trim().is_empty() {
        String::new()
    } else {
        format!(" {} ", compact_text(right, inner / 2))
    };
    let left_len = left.chars().count();
    let right_len = right.chars().count();
    if left_len + right_len >= inner {
        let body = fit_line(&format!("{title} {}", compact_text(right.trim(), inner / 2)), inner);
        return format!("┌{}┐", pad_to_width(&body, inner));
    }
    let fill = "─".repeat(inner - left_len - right_len);
    format!("┌{}{}{}┐", left, fill, right)
}

fn frame_separator(title: &str, width: usize) -> String {
    if width <= 2 {
        return fit_line(title, width);
    }
    let inner = width - 2;
    let label = format!("─ {} ", title);
    let label_len = label.chars().count();
    if label_len >= inner {
        return format!("├{}┤", pad_to_width(&fit_line(title, inner), inner));
    }
    let fill = "─".repeat(inner - label_len);
    format!("├{}{}┤", label, fill)
}

fn frame_row(content: &str, width: usize) -> String {
    if width <= 4 {
        return fit_line(content, width);
    }
    let inner = width - 4;
    let fitted = fit_line(content, inner);
    format!("│ {} │", pad_to_width(&fitted, inner))
}

fn frame_bottom(width: usize) -> String {
    if width <= 2 {
        return String::new();
    }
    format!("└{}┘", "─".repeat(width - 2))
}

fn pad_to_width(text: &str, width: usize) -> String {
    let text_width = text.chars().count();
    if text_width >= width {
        return text.chars().take(width).collect();
    }
    format!("{text}{}", " ".repeat(width - text_width))
}

fn format_activity_line(event: &AgentEventRecord) -> String {
    let time = short_timestamp(&event.created_at);
    let method = match event.method.as_str() {
        "interrupt/queued" => "interrupt",
        "turn/started" => "turn start",
        "turn/completed" => "turn done",
        "turn/interrupt" => "interrupt",
        "turn/steer" => "interrupt",
        "task/queued" => "queued",
        "task/selected" => "task active",
        "task/completed" => "task done",
        "task/blocked" => "task blocked",
        "task/requeued" => "task requeued",
        "loop/recoveryActivated" => "recovery",
        "loop/incidentOpened" => "incident",
        "loop/incidentResolved" => "resolved",
        other => other,
    };
    let task_ref = event
        .active_task_id
        .map(|id| format!("#{} ", id))
        .unwrap_or_default();
    let detail = if event.body.trim().is_empty() {
        event.active_task_title.as_str()
    } else {
        event.body.as_str()
    };
    compact_text(
        &format!("{time}  {:<11} {}{}", method, task_ref, detail),
        140,
    )
}

fn short_timestamp(value: &str) -> String {
    DateTime::parse_from_rfc3339(value)
        .map(|dt| dt.with_timezone(&Local).format("%H:%M").to_string())
        .unwrap_or_else(|_| value.to_string())
}

fn compact_text(text: &str, limit: usize) -> String {
    let normalized = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if normalized.chars().count() <= limit {
        return normalized;
    }
    let truncated = normalized
        .chars()
        .take(limit.saturating_sub(3))
        .collect::<String>();
    format!("{truncated}...")
}

fn fit_line(text: &str, width: usize) -> String {
    if width == 0 {
        return String::new();
    }
    let text_len = text.chars().count();
    if text_len <= width {
        return text.to_string();
    }
    if width <= 3 {
        return text.chars().take(width).collect();
    }
    let prefix = text.chars().take(width - 3).collect::<String>();
    format!("{prefix}...")
}

fn input_display(input: &str, width: usize, prompt: &str) -> (String, usize) {
    let prompt_width = prompt.chars().count();
    let available = width.saturating_sub(prompt_width);
    if available == 0 {
        return (String::new(), prompt_width.min(width));
    }

    let input_width = input.chars().count();
    if input_width <= available {
        return (input.to_string(), prompt_width + input_width);
    }

    if available <= 3 {
        let tail = input
            .chars()
            .rev()
            .take(available)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect::<String>();
        return (tail, width);
    }

    let tail = input
        .chars()
        .rev()
        .take(available - 3)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect::<String>();
    (format!("...{tail}"), width)
}

fn render_screen_lines(
    stdout: &mut io::Stdout,
    lines: &[String],
    cursor_col: u16,
    cursor_row: u16,
) -> anyhow::Result<()> {
    queue!(stdout, MoveTo(0, 0), Clear(ClearType::All))?;
    for (row, line) in lines.iter().enumerate() {
        queue!(stdout, MoveTo(0, row as u16), Print(line))?;
    }
    queue!(stdout, MoveTo(cursor_col, cursor_row))?;
    stdout.flush()?;
    Ok(())
}

fn elapsed_text(created_at: &str) -> Option<String> {
    let parsed = DateTime::parse_from_rfc3339(created_at).ok()?;
    let elapsed = Utc::now().signed_duration_since(parsed.with_timezone(&Utc));
    if elapsed.num_seconds() < 0 {
        return None;
    }
    Some(format_elapsed(Duration::from_secs(elapsed.num_seconds() as u64)))
}

fn format_elapsed(duration: Duration) -> String {
    let elapsed_secs = duration.as_secs();
    if elapsed_secs < 60 {
        return format!("{elapsed_secs}s");
    }
    if elapsed_secs < 3600 {
        return format!("{}m {:02}s", elapsed_secs / 60, elapsed_secs % 60);
    }
    format!(
        "{}h {:02}m {:02}s",
        elapsed_secs / 3600,
        (elapsed_secs % 3600) / 60,
        elapsed_secs % 60
    )
}

fn format_event_line(event: &AgentEventRecord) -> String {
    let method = match event.method.as_str() {
        "interrupt/queued" => "INTERRUPT queued",
        "turn/started" => "TURN start",
        "turn/completed" => "TURN done",
        "turn/interrupt" => "TURN interrupt",
        "turn/steer" => "TURN steer",
        "task/queued" => "TASK queued",
        "task/selected" => "TASK selected",
        "task/completed" => "TASK done",
        "task/blocked" => "TASK blocked",
        "task/requeued" => "TASK requeued",
        "loop/recoveryActivated" => "LOOP recovery",
        "loop/incidentOpened" => "LOOP incident",
        "loop/incidentResolved" => "LOOP resolved",
        other => other,
    };
    let task_ref = event
        .active_task_id
        .map(|id| format!("#{} {}", id, compact_text(&event.active_task_title, 100)))
        .unwrap_or_else(|| event.active_task_title.clone());
    if task_ref.trim().is_empty() {
        format!(
            "[{}] {} :: {}",
            event.created_at,
            method,
            compact_text(&event.body, 200)
        )
    } else {
        format!(
            "[{}] {} :: {} :: {}",
            event.created_at,
            method,
            task_ref,
            compact_text(&event.body, 200)
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_turn(id: i64, task_id: i64, title: &str) -> AgentTurnRecord {
        AgentTurnRecord {
            id,
            created_at: "2026-03-18T15:08:00+00:00".to_string(),
            updated_at: "2026-03-18T15:08:00+00:00".to_string(),
            task_id,
            task_title: title.to_string(),
            trigger: "supervisor_tick".to_string(),
            mode_at_start: "execute_task".to_string(),
            mode_at_end: None,
            status: "in_progress".to_string(),
            summary: Some("Bounded step laeuft gerade.".to_string()),
            output: None,
            completed_at: None,
        }
    }

    fn sample_ui() -> AttachTui {
        let mut notices = VecDeque::new();
        notices.push_back(
            "16:09  you     Michael Welsch: BIOS-Link bitte sichtbarer machen".to_string(),
        );
        notices.push_back(
            "16:09  queued  #413 BIOS-Link im Attach-Screen anzeigen".to_string(),
        );
        notices.push_back(
            "16:09  interrupt  #412 Signal am laufenden Thread vermerkt.".to_string(),
        );
        AttachTui {
            snapshot: AttachUiSnapshot {
                bios_url: Some("https://127.0.0.1:8443/bios".to_string()),
                thread: Some(AgentThreadRecord {
                    thread_key: "main".to_string(),
                    created_at: "2026-03-18T15:00:00+00:00".to_string(),
                    updated_at: "2026-03-18T15:09:00+00:00".to_string(),
                    lifecycle_status: "running".to_string(),
                    current_mode: "execute_task".to_string(),
                    active_turn_id: Some(184),
                    active_task_id: Some(412),
                    queue_depth: 2,
                    note: String::new(),
                }),
                focus: Some(FocusStateRecord {
                    mode: "execute_task".to_string(),
                    active_task_id: Some(412),
                    active_task_title: "Watchdog-Regression im Browser-Installer pruefen"
                        .to_string(),
                    queue_depth: 2,
                    last_reprioritized_at: None,
                    last_task_completed_at: None,
                    note: String::new(),
                }),
                active_turn: Some(sample_turn(
                    184,
                    412,
                    "Watchdog-Regression im Browser-Installer pruefen",
                )),
                last_turn: Some(AgentTurnRecord {
                    id: 183,
                    created_at: "2026-03-18T15:06:00+00:00".to_string(),
                    updated_at: "2026-03-18T15:07:00+00:00".to_string(),
                    task_id: 409,
                    task_title: "Browser-Healthcheck absichern".to_string(),
                    trigger: "supervisor_tick".to_string(),
                    mode_at_start: "review".to_string(),
                    mode_at_end: Some("reprioritize".to_string()),
                    status: "completed".to_string(),
                    summary: Some("Browser-Healthcheck abgesichert.".to_string()),
                    output: None,
                    completed_at: Some("2026-03-18T15:07:00+00:00".to_string()),
                }),
                open_tasks: vec![
                    TaskRecord {
                        id: 413,
                        created_at: "2026-03-18T15:09:00+00:00".to_string(),
                        updated_at: "2026-03-18T15:09:00+00:00".to_string(),
                        parent_task_id: None,
                        worker_job_id: None,
                        source_interrupt_id: None,
                        source_channel: "terminal".to_string(),
                        speaker: "Michael Welsch".to_string(),
                        task_kind: "homepage_bridge".to_string(),
                        title: "BIOS-Link im Attach-Screen anzeigen".to_string(),
                        detail: String::new(),
                        trust_level: "owner".to_string(),
                        priority_score: 1000,
                        status: "queued".to_string(),
                        run_count: 0,
                        last_checkpoint_summary: None,
                        last_checkpoint_at: None,
                        last_output: None,
                    },
                    TaskRecord {
                        id: 414,
                        created_at: "2026-03-18T15:09:00+00:00".to_string(),
                        updated_at: "2026-03-18T15:09:00+00:00".to_string(),
                        parent_task_id: None,
                        worker_job_id: None,
                        source_interrupt_id: None,
                        source_channel: "terminal".to_string(),
                        speaker: "Michael Welsch".to_string(),
                        task_kind: "system".to_string(),
                        title: "Interrupt-Warteschlange im UI sichtbarer machen".to_string(),
                        detail: String::new(),
                        trust_level: "owner".to_string(),
                        priority_score: 990,
                        status: "queued".to_string(),
                        run_count: 0,
                        last_checkpoint_summary: None,
                        last_checkpoint_at: None,
                        last_output: None,
                    },
                ],
            },
            event_lines: vec![
                "16:08  task active  #412 Aufgabe wurde zur aktiven Arbeit gezogen.".to_string(),
                "16:08  turn start   #412 Turn 184 gestartet.".to_string(),
            ],
            local_notices: notices,
            pending_inputs: vec![
                PendingInputItem {
                    task_id: Some(413),
                    task_title: "BIOS-Link im Attach-Screen anzeigen".to_string(),
                    message: "Michael Welsch: BIOS-Link bitte sichtbarer machen".to_string(),
                    status: PendingInputStatus::Queued,
                    completed_at: None,
                    seen_open: true,
                },
                PendingInputItem {
                    task_id: Some(414),
                    task_title: "Interrupt-Warteschlange im UI sichtbarer machen".to_string(),
                    message: "Michael Welsch: zeig mir danach bitte die letzten drei Turns"
                        .to_string(),
                    status: PendingInputStatus::Queued,
                    completed_at: None,
                    seen_open: true,
                },
            ],
            input: "Michael Welsch: zeig mir danach bitte die letzten drei Turns".to_string(),
            spinner_phase: 0,
        }
    }

    #[test]
    fn attach_primary_status_matches_reference_headline() {
        let ui = sample_ui();
        let headline = ui.primary_status_line();
        assert!(headline.contains("RUNNING"));
        assert!(headline.contains("EXEC"));
        assert!(headline.contains("queue 2"));
        assert!(headline.contains("turn #184"));
    }

    #[test]
    fn attach_reference_sections_contain_expected_rows() {
        let ui = sample_ui();
        let now_rows = ui.now_rows(120).join("\n");
        let next_rows = ui.next_rows(120).join("\n");
        let activity_rows = ui.activity_rows(120, 8).join("\n");

        assert!(ui.secondary_status_line().contains("#412 Watchdog-Regression"));
        assert!(now_rows.contains("Summary  Bounded step laeuft gerade."));
        assert!(now_rows.contains("Last done  #409 Browser-Healthcheck absichern"));
        assert!(next_rows.contains("○ #413 BIOS-Link im Attach-Screen anzeigen"));
        assert!(next_rows.contains("○ #414 Interrupt-Warteschlange im UI sichtbarer machen"));
        assert!(activity_rows.contains("16:08  task active  #412 Aufgabe wurde zur aktiven Arbeit gezogen."));
        assert!(activity_rows.contains("16:08  turn start   #412 Turn 184 gestartet."));
        assert!(activity_rows.contains("16:09  you"));
        assert!(activity_rows.contains("16:09  queued"));
        assert!(activity_rows.contains("16:09  interrupt"));
    }

    #[test]
    fn interrupt_events_render_like_reference_label() {
        let event = AgentEventRecord {
            id: 1,
            created_at: "2026-03-18T15:09:00+00:00".to_string(),
            method: "turn/interrupt".to_string(),
            active_task_id: Some(412),
            active_task_title: "Watchdog-Regression".to_string(),
            body: "Signal am laufenden Thread vermerkt.".to_string(),
            payload_json: "{}".to_string(),
        };
        let line = format_activity_line(&event);
        assert!(line.contains("interrupt"));
        assert!(!line.contains("turn steer"));
    }
}
