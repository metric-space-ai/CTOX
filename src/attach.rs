use crate::bootstrap::handle_attach_line_detailed;
use crate::brain_runtime::apply_targeted_kleinhirn_upgrade;
use crate::brain_runtime::GrosshirnRuntimeSnapshot;
use crate::brain_runtime::KleinhirnRuntimeSnapshot;
use crate::brain_runtime::load_grosshirn_runtime_snapshot;
use crate::brain_runtime::load_kleinhirn_runtime_snapshot;
use crate::brain_runtime::load_runtime_env_map;
use crate::brain_runtime::save_runtime_env_map;
use crate::contracts::BootEntry;
use crate::contracts::control_plane_bind_host;
use crate::contracts::control_plane_public_base_url;
use crate::contracts::InstallationBootstrapState;
use crate::contracts::Organigram;
use crate::contracts::Paths;
use crate::contracts::load_bios;
use crate::contracts::load_boot_entries;
use crate::contracts::load_installation_bootstrap_state;
use crate::contracts::load_organigram;
use crate::contracts::normalize_runtime_model_choice;
use crate::contracts::now_iso;
use crate::contracts::save_installation_bootstrap_state;
use crate::contracts::save_organigram;
use crate::lifecycle::factory_reset_installation;
use crate::runtime_db::AgentEventRecord;
use crate::runtime_db::AgentThreadRecord;
use crate::runtime_db::AgentTurnRecord;
use crate::runtime_db::BrainRoutingState;
use crate::runtime_db::BrainUsageRollup;
use crate::runtime_db::FocusStateRecord;
use crate::runtime_db::TaskRecord;
use crate::runtime_db::load_brain_usage_rollup;
use crate::runtime_db::list_agent_events_since;
use crate::runtime_db::list_open_tasks;
use crate::runtime_db::list_recent_agent_events;
use crate::runtime_db::list_recent_completed_owner_tasks;
use crate::runtime_db::load_active_agent_turn;
use crate::runtime_db::load_agent_thread;
use crate::runtime_db::load_brain_routing_state;
use crate::runtime_db::load_focus_state;
use crate::runtime_db::load_latest_completed_agent_turn;
use anyhow::Context;
use chrono::DateTime;
use chrono::Local;
use chrono::Utc;
use crossterm::cursor::MoveTo;
use crossterm::event;
use crossterm::event::DisableBracketedPaste;
use crossterm::event::EnableBracketedPaste;
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
use std::collections::BTreeMap;
use std::collections::VecDeque;
use std::fs;
use std::io;
use std::io::BufRead;
use std::io::BufReader;
use std::io::IsTerminal;
use std::io::Write;
use std::os::unix::net::UnixStream as StdUnixStream;
use std::process::Command;
use std::process::Stdio;
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
const FACTORY_RESET_CONFIRM_WINDOW_SECS: u64 = 3;
const CHAT_HISTORY_LIMIT: usize = 14;
const CHAT_IDLE_SECS: u64 = 180;

const GPT_OSS_MODEL: &str = "openai/gpt-oss-20b";
const AUTHORIZED_MODEL_OPTIONS: &[&str] = &[
    GPT_OSS_MODEL,
    "openai/gpt-5.4-nano",
    "openai/gpt-5.4-mini",
    "openai/gpt-5.4",
];
const SIMPLE_MODEL_OPTIONS: &[&str] = AUTHORIZED_MODEL_OPTIONS;
const MEDIUM_MODEL_OPTIONS: &[&str] = AUTHORIZED_MODEL_OPTIONS;
const RED_MODEL_OPTIONS: &[&str] = AUTHORIZED_MODEL_OPTIONS;
const LOCAL_MODEL_OPTIONS: &[&str] = &[GPT_OSS_MODEL];
const GROSSHIRN_MODEL_OPTIONS: &[&str] = AUTHORIZED_MODEL_OPTIONS;

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
    brain_routing: BrainRoutingState,
    brain_usage: BrainUsageRollup,
    kleinhirn_runtime: Option<KleinhirnRuntimeSnapshot>,
    grosshirn_runtime: Option<GrosshirnRuntimeSnapshot>,
    focus: Option<FocusStateRecord>,
    active_turn: Option<AgentTurnRecord>,
    last_turn: Option<AgentTurnRecord>,
    open_tasks: Vec<TaskRecord>,
    completed_owner_tasks: Vec<TaskRecord>,
    boot_entries: Vec<BootEntry>,
    organigram: Organigram,
    installation_bootstrap: InstallationBootstrapState,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AttachPage {
    Tasks,
    Chat,
    Settings,
}

#[derive(Debug, Clone)]
struct SettingsItem {
    key: &'static str,
    label: &'static str,
    value: String,
    secret: bool,
    choices: Vec<&'static str>,
    help: &'static str,
}

struct SaveSettingsOutcome {
    runtime_restart_required: bool,
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
    created_at_label: String,
    status: PendingInputStatus,
    completed_at: Option<Instant>,
    seen_open: bool,
}

struct AttachTui {
    snapshot: AttachUiSnapshot,
    event_lines: Vec<String>,
    local_notices: VecDeque<String>,
    pending_inputs: Vec<PendingInputItem>,
    page: AttachPage,
    chat_input: String,
    settings_items: Vec<SettingsItem>,
    settings_selected: usize,
    settings_dirty: bool,
    spinner_phase: usize,
    factory_reset_armed_until: Option<Instant>,
}

enum TuiControl {
    Continue,
    Exit,
}

struct TerminalUiGuard;

impl TerminalUiGuard {
    fn enter(stdout: &mut io::Stdout) -> anyhow::Result<Self> {
        terminal::enable_raw_mode().context("failed to enable raw mode")?;
        execute!(stdout, EnterAlternateScreen, EnableBracketedPaste)
            .context("failed to enter alternate screen")?;
        Ok(Self)
    }
}

impl Drop for TerminalUiGuard {
    fn drop(&mut self) {
        let _ = terminal::disable_raw_mode();
        let mut stdout = io::stdout();
        let _ = execute!(stdout, DisableBracketedPaste, LeaveAlternateScreen);
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
                    match ui.page {
                        AttachPage::Tasks => {}
                        AttachPage::Chat => ui.chat_input.push_str(&text),
                        AttachPage::Settings => {
                            if let Some(item) = ui.settings_items.get_mut(ui.settings_selected) {
                                item.value.push_str(&text);
                                ui.settings_dirty = true;
                            }
                        }
                    }
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
        println!("  No agent events available yet.");
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
                    print!("Chat to CTO: ");
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
            print!("Chat to CTO: ");
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
            page: AttachPage::Tasks,
            chat_input: String::new(),
            settings_items: load_settings_items(paths),
            settings_selected: 0,
            settings_dirty: false,
            spinner_phase: 0,
            factory_reset_armed_until: None,
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
        if !self.settings_dirty {
            self.settings_items = load_settings_items(paths);
            self.settings_selected = self
                .settings_selected
                .min(self.settings_items.len().saturating_sub(1));
        }
        self.update_pending_inputs();
    }

    fn handle_key_event(&mut self, paths: &Paths, key_event: KeyEvent) -> TuiControl {
        if self
            .factory_reset_armed_until
            .map(|deadline| deadline <= Instant::now())
            .unwrap_or(false)
        {
            self.factory_reset_armed_until = None;
        }

        match key_event {
            KeyEvent {
                code: KeyCode::Char('p'),
                modifiers,
                ..
            } if modifiers.contains(KeyModifiers::CONTROL) => {
                if let Err(err) = self.handle_factory_reset_shortcut(paths) {
                    let mut refreshed = AttachTui::new(paths);
                    refreshed.push_local_notice("error", &err.to_string());
                    *self = refreshed;
                }
            }
            KeyEvent {
                code: KeyCode::Char('c'),
                modifiers,
                ..
            } if modifiers.contains(KeyModifiers::CONTROL) => return TuiControl::Exit,
            KeyEvent {
                code: KeyCode::Char('d'),
                modifiers,
                ..
            } if modifiers.contains(KeyModifiers::CONTROL) && self.active_editor_is_empty() => {
                return TuiControl::Exit;
            }
            KeyEvent {
                code: KeyCode::Tab, ..
            } => {
                self.page = match self.page {
                    AttachPage::Tasks => AttachPage::Chat,
                    AttachPage::Chat => AttachPage::Settings,
                    AttachPage::Settings => AttachPage::Tasks,
                };
            }
            _ => match self.page {
                AttachPage::Tasks => {}
                AttachPage::Chat => self.handle_chat_key_event(paths, key_event),
                AttachPage::Settings => self.handle_settings_key_event(paths, key_event),
            },
        }

        TuiControl::Continue
    }

    fn handle_chat_key_event(&mut self, paths: &Paths, key_event: KeyEvent) {
        match key_event {
            KeyEvent {
                code: KeyCode::Esc, ..
            } => {
                self.chat_input.clear();
            }
            KeyEvent {
                code: KeyCode::Backspace,
                ..
            } => {
                self.chat_input.pop();
            }
            KeyEvent {
                code: KeyCode::Enter,
                ..
            } => {
                if let Err(err) = self.submit_chat_input(paths) {
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
                self.chat_input.push(character);
            }
            _ => {}
        }
    }

    fn handle_settings_key_event(&mut self, paths: &Paths, key_event: KeyEvent) {
        if self.settings_items.is_empty() {
            return;
        }

        match key_event {
            KeyEvent {
                code: KeyCode::Up, ..
            } => {
                self.settings_selected = self.settings_selected.saturating_sub(1);
            }
            KeyEvent {
                code: KeyCode::Down,
                ..
            } => {
                self.settings_selected = (self.settings_selected + 1)
                    .min(self.settings_items.len().saturating_sub(1));
            }
            KeyEvent {
                code: KeyCode::Left,
                ..
            } => {
                self.cycle_selected_setting_choice(-1);
            }
            KeyEvent {
                code: KeyCode::Right,
                ..
            } => {
                self.cycle_selected_setting_choice(1);
            }
            KeyEvent {
                code: KeyCode::Esc, ..
            } => {
                if let Some(item) = self.settings_items.get_mut(self.settings_selected) {
                    item.value.clear();
                    self.settings_dirty = true;
                }
            }
            KeyEvent {
                code: KeyCode::Backspace,
                ..
            } => {
                if let Some(item) = self.settings_items.get_mut(self.settings_selected) {
                    item.value.pop();
                    self.settings_dirty = true;
                }
            }
            KeyEvent {
                code: KeyCode::Enter,
                ..
            } => {
                if let Err(err) = self.save_settings(paths) {
                    self.push_local_notice("error", &err.to_string());
                }
            }
            KeyEvent {
                code: KeyCode::Char(character),
                modifiers,
                ..
            } if modifiers.contains(KeyModifiers::CONTROL)
                && matches!(character, 's' | 'S') =>
            {
                if let Err(err) = self.save_settings(paths) {
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
                if let Some(item) = self.settings_items.get_mut(self.settings_selected) {
                    item.value.push(character);
                    self.settings_dirty = true;
                }
            }
            _ => {}
        }
    }

    fn active_editor_is_empty(&self) -> bool {
        match self.page {
            AttachPage::Tasks => true,
            AttachPage::Chat => self.chat_input.is_empty(),
            AttachPage::Settings => self
                .settings_items
                .get(self.settings_selected)
                .map(|item| item.value.is_empty())
                .unwrap_or(true),
        }
    }

    fn cycle_selected_setting_choice(&mut self, direction: isize) {
        let Some(item) = self.settings_items.get_mut(self.settings_selected) else {
            return;
        };
        if item.choices.is_empty() {
            return;
        }
        let choices = item.choices.as_slice();
        let current_index = choices
            .iter()
            .position(|choice| choice.eq_ignore_ascii_case(item.value.trim()))
            .unwrap_or(0);
        let next_index = if direction < 0 {
            current_index.checked_sub(1).unwrap_or(choices.len() - 1)
        } else {
            (current_index + 1) % choices.len()
        };
        item.value = choices[next_index].to_string();
        self.settings_dirty = true;
    }

    fn handle_factory_reset_shortcut(&mut self, paths: &Paths) -> anyhow::Result<()> {
        let now = Instant::now();
        let armed = self
            .factory_reset_armed_until
            .map(|deadline| deadline > now)
            .unwrap_or(false);
        if !armed {
            self.factory_reset_armed_until =
                Some(now + Duration::from_secs(FACTORY_RESET_CONFIRM_WINDOW_SECS));
            self.push_local_notice(
                "warn",
                "Factory reset armed. Press Ctrl-P again within 3s to wipe BIOS/root state, rebuild the runtime DB, and restart from a fresh installation.",
            );
            return Ok(());
        }

        self.factory_reset_armed_until = None;
        perform_attach_factory_reset(paths)?;
        let mut refreshed = AttachTui::new(paths);
        refreshed.push_local_notice(
            "reset",
            "Factory reset completed. The CTO-Agent was restarted from a fresh installation state.",
        );
        *self = refreshed;
        Ok(())
    }

    fn submit_chat_input(&mut self, paths: &Paths) -> anyhow::Result<()> {
        let trimmed = self.chat_input.trim().to_string();
        if trimmed.is_empty() {
            return Ok(());
        }

        let response = send_attach_request(paths, &trimmed)?;
        let visible_input = owner_visible_input_echo(&trimmed);
        self.push_local_notice("you", &visible_input);
        self.pending_inputs.push(PendingInputItem {
            task_id: response.queued_task_id,
            task_title: response
                .queued_task_title
                .clone()
                .unwrap_or_else(|| visible_input.clone()),
            message: visible_input.clone(),
            created_at_label: Local::now().format("%H:%M").to_string(),
            status: PendingInputStatus::Queued,
            completed_at: None,
            seen_open: false,
        });
        if let Some(task_id) = response.queued_task_id {
            let task_title = response
                .queued_task_title
                .clone()
                .unwrap_or_else(|| visible_input.clone());
            self.push_local_notice("queued", &format!("#{task_id} {task_title}"));
        } else {
            self.push_local_notice("ack", first_visible_line(&response.output));
        }

        self.chat_input.clear();
        self.refresh(paths);
        Ok(())
    }

    fn save_settings(&mut self, paths: &Paths) -> anyhow::Result<()> {
        let outcome = save_settings_items(paths, &self.settings_items)?;
        if outcome.runtime_restart_required {
            restart_runtime_after_network_settings_change(paths)?;
        }
        self.settings_dirty = false;
        self.snapshot = collect_ui_snapshot(paths);
        self.settings_items = load_settings_items(paths);
        self.push_local_notice(
            "saved",
            "Attach settings persisted to organigram, installation bootstrap, and runtime env.",
        );
        if outcome.runtime_restart_required {
            self.push_local_notice(
                "restart",
                "The runtime was restarted so BIOS network changes take effect immediately.",
            );
        }
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
        let active_task_id = self
            .snapshot
            .focus
            .as_ref()
            .and_then(|focus| focus.active_task_id);
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

        match self.page {
            AttachPage::Tasks => self.render_tasks_page(stdout, width_usize, height_usize),
            AttachPage::Chat => self.render_chat_page(stdout, width_usize, height_usize),
            AttachPage::Settings => self.render_settings_page(stdout, width_usize, height_usize),
        }
    }

    fn render_compact(
        &self,
        stdout: &mut io::Stdout,
        width: usize,
        height: usize,
    ) -> anyhow::Result<()> {
        let prompt = self.editor_prompt();
        let (input_text, cursor_col) = input_display(self.editor_text(), width, &prompt);
        let mode = self
            .snapshot
            .thread
            .as_ref()
            .map(|thread| short_mode(&thread.current_mode))
            .unwrap_or_else(|| "UNKNOWN".to_string());
        let brain_route = brain_route_label(&self.snapshot.brain_routing.route_mode);
        let brain_model = compact_text(&current_brain_model_label(&self.snapshot), 48);
        let usage = &self.snapshot.brain_usage;
        let active_task = self
            .snapshot
            .focus
            .as_ref()
            .map(|focus| describe_active_task(&focus.active_task_id, &focus.active_task_title))
            .unwrap_or_else(|| "no active task".to_string());
        let status_line = match &self.snapshot.active_turn {
            Some(turn) => {
                let elapsed = elapsed_text(&turn.created_at).unwrap_or_else(|| "n/a".to_string());
                format!("Working {elapsed} | {} | {}", mode, brain_route)
            }
            None => format!("Idle | {} | {}", mode, brain_route),
        };
        let pending_count = self.pending_inputs.len();
        let latest_done = self
            .snapshot
            .completed_owner_tasks
            .first()
            .map(format_completed_owner_task_line)
            .unwrap_or_else(|| "No completed owner tasks yet.".to_string());

        let mut lines = vec![
            "=== CTO-Agent Attach Terminal ===".to_string(),
            format!("Page: {}", self.page_label()),
            status_line,
            format!("Model: {brain_model}"),
            format!(
                "Tokens: in {} | out {} | total {}",
                usage.total_input_tokens, usage.total_output_tokens, usage.total_tokens
            ),
            format!("Task: {}", compact_text(&active_task, 72)),
            format!("Queued Chats: {pending_count}"),
            fit_line(&format!("Last done: {latest_done}"), width),
            "Tab page | Enter act/save | Ctrl-P reset | Ctrl-C exit".to_string(),
        ];
        if !prompt.is_empty() {
            lines.push(format!("{prompt}{input_text}"));
        }
        if lines.len() > height {
            let keep = height.saturating_sub(1);
            let mut compact = vec!["=== CTO-Agent Attach Terminal ===".to_string()];
            compact.extend(
                lines
                    .into_iter()
                    .rev()
                    .take(keep)
                    .collect::<Vec<_>>()
                    .into_iter()
                    .rev(),
            );
            lines = compact;
        }

        let fitted = lines
            .into_iter()
            .take(height)
            .map(|line| fit_line(&line, width))
            .collect::<Vec<_>>();
        let cursor_row = fitted.len().saturating_sub(1) as u16;
        let cursor_col = if prompt.is_empty() {
            0
        } else {
            cursor_col.min(width.saturating_sub(1))
        } as u16;
        render_screen_lines(stdout, &fitted, cursor_col, cursor_row)
    }

    fn render_tasks_page(
        &self,
        stdout: &mut io::Stdout,
        width: usize,
        height: usize,
    ) -> anyhow::Result<()> {
        let frame_width = width;
        let agent_rows = self.agent_rows(frame_width);
        let current_rows = self.now_rows(frame_width);
        let queue_rows = self.next_rows(frame_width);
        let fixed_rows = 2 + 1 + agent_rows.len() + 1 + current_rows.len() + 1 + queue_rows.len() + 1 + 1;
        let completed_rows = height.saturating_sub(fixed_rows).max(4);

        let mut frame_lines = vec![
            frame_top("CTO-Agent Tasks", &self.primary_status_line(), frame_width),
            frame_row(
                &format!(
                    "BIOS {}  |  Page {}",
                    self.snapshot.bios_url.as_deref().unwrap_or("unavailable"),
                    self.page_label()
                ),
                frame_width,
            ),
            frame_separator("Agent", frame_width),
        ];
        frame_lines.extend(agent_rows);
        frame_lines.push(frame_separator("Current", frame_width));
        frame_lines.extend(current_rows);
        frame_lines.push(frame_separator("Queue", frame_width));
        frame_lines.extend(queue_rows);
        frame_lines.push(frame_separator("Completed", frame_width));
        frame_lines.extend(self.completed_owner_task_rows(frame_width, completed_rows));
        frame_lines.push(frame_bottom(frame_width));

        let mut screen_lines = frame_lines;
        screen_lines.push("Tab Chat · Tab Settings · Ctrl-P reset · Ctrl-C exit".to_string());
        let fitted_lines = screen_lines
            .into_iter()
            .take(height)
            .map(|line| fit_line(&line, width))
            .collect::<Vec<_>>();
        render_screen_lines(stdout, &fitted_lines, 0, fitted_lines.len().saturating_sub(1) as u16)
    }

    fn render_chat_page(
        &self,
        stdout: &mut io::Stdout,
        width: usize,
        height: usize,
    ) -> anyhow::Result<()> {
        let frame_width = width;
        let prompt = self.editor_prompt();
        let (input_display, cursor_col) = input_display(self.editor_text(), width, &prompt);
        let input_hint = if self.snapshot.active_turn.is_some() {
            "Enter queues raw chat for the next safe boundary. Tab cycles to Tasks and Settings."
        } else {
            "Enter sends raw chat now. Tab cycles to Tasks and Settings."
        };
        let agent_rows = self.agent_rows(frame_width);
        let fixed_rows = 2 + 1 + agent_rows.len() + 1 + 1 + 1 + 1 + 1;
        let chat_rows = height.saturating_sub(fixed_rows).max(10);

        let mut frame_lines = vec![
            frame_top("CTO-Agent Chat", &self.primary_status_line(), frame_width),
            frame_row(
                &format!(
                    "BIOS {}  |  Page {}",
                    self.snapshot.bios_url.as_deref().unwrap_or("unavailable"),
                    self.page_label()
                ),
                frame_width,
            ),
            frame_separator("Agent", frame_width),
        ];
        frame_lines.extend(agent_rows);
        frame_lines.push(frame_separator("Conversation", frame_width));
        frame_lines.extend(self.chat_rows(frame_width, chat_rows));
        frame_lines.push(frame_separator("Compose", frame_width));
        frame_lines.push(frame_row(input_hint, frame_width));
        frame_lines.push(frame_row(&format!("{prompt}{input_display}"), frame_width));
        frame_lines.push(frame_bottom(frame_width));
        let fitted_lines = frame_lines
            .into_iter()
            .take(height)
            .map(|line| fit_line(&line, width))
            .collect::<Vec<_>>();
        let cursor_row = fitted_lines.len().saturating_sub(2) as u16;
        let cursor_col = cursor_col.min(width.saturating_sub(1)) as u16;
        render_screen_lines(stdout, &fitted_lines, cursor_col, cursor_row)
    }

    fn render_settings_page(
        &self,
        stdout: &mut io::Stdout,
        width: usize,
        height: usize,
    ) -> anyhow::Result<()> {
        let frame_width = width;
        let fixed_rows = 2 + 1 + 3 + 1 + 1 + 2;
        let list_rows = height.saturating_sub(fixed_rows).max(8);
        let prompt = self.editor_prompt();
        let (input_display, cursor_col) = input_display(self.editor_text(), width, &prompt);
        let dirty = if self.settings_dirty { "DIRTY" } else { "SAVED" };

        let mut frame_lines = vec![
            frame_top("CTO-Agent Settings", &self.primary_status_line(), frame_width),
            frame_row(
                &format!(
                    "Status {}  |  Use Up/Down to select, Left/Right to cycle the three compaction tiers, Enter or Ctrl-S to save, Tab cycles Tasks and Chat",
                    dirty
                ),
                frame_width,
            ),
            frame_separator("Fields", frame_width),
        ];
        frame_lines.extend(self.settings_rows(frame_width, list_rows));
        frame_lines.push(frame_separator("Selected", frame_width));
        frame_lines.push(frame_row(&self.selected_setting_help_line(), frame_width));
        frame_lines.push(frame_bottom(frame_width));

        let mut screen_lines = frame_lines;
        screen_lines.push("Esc clears selected value · secrets are masked in the list view".to_string());
        screen_lines.push(format!("{prompt}{input_display}"));

        let fitted_lines = screen_lines
            .into_iter()
            .take(height)
            .map(|line| fit_line(&line, width))
            .collect::<Vec<_>>();
        let cursor_row = fitted_lines.len().saturating_sub(1) as u16;
        let cursor_col = cursor_col.min(width.saturating_sub(1)) as u16;
        render_screen_lines(stdout, &fitted_lines, cursor_col, cursor_row)
    }

    fn page_label(&self) -> &'static str {
        match self.page {
            AttachPage::Tasks => "TASKS",
            AttachPage::Chat => "CHAT",
            AttachPage::Settings => "SETTINGS",
        }
    }

    fn editor_prompt(&self) -> String {
        match self.page {
            AttachPage::Tasks => String::new(),
            AttachPage::Chat => "Chat to CTO: ".to_string(),
            AttachPage::Settings => {
                let label = self
                    .settings_items
                    .get(self.settings_selected)
                    .map(|item| item.label)
                    .unwrap_or("Setting");
                format!("Edit {label}: ")
            }
        }
    }

    fn editor_text(&self) -> &str {
        match self.page {
            AttachPage::Tasks => "",
            AttachPage::Chat => &self.chat_input,
            AttachPage::Settings => self
                .settings_items
                .get(self.settings_selected)
                .map(|item| item.value.as_str())
                .unwrap_or(""),
        }
    }

    fn agent_rows(&self, width: usize) -> Vec<String> {
        let usage = &self.snapshot.brain_usage;
        let simple_slot = self.setting_value("simple_model").unwrap_or("?");
        let medium_slot = self.setting_value("medium_model").unwrap_or("?");
        let red_slot = self.setting_value("red_model").unwrap_or("?");
        let last_model = if usage.last_model_id.trim().is_empty() {
            "no model call recorded yet".to_string()
        } else {
            format!(
                "{} via {} at {}",
                usage.last_model_id,
                if usage.last_brain_tier.trim().is_empty() {
                    "unknown".to_string()
                } else {
                    usage.last_brain_tier.to_uppercase()
                },
                usage
                    .last_recorded_at
                    .as_deref()
                    .map(short_timestamp)
                    .unwrap_or_else(|| "n/a".to_string())
            )
        };
        vec![
            frame_row(
                &format!(
                    "Runtime active {}  |  Route {}  |  Chat session {}",
                    compact_text(&current_brain_model_label(&self.snapshot), 56),
                    brain_route_label(&self.snapshot.brain_routing.route_mode),
                    self.chat_session_label()
                ),
                width,
            ),
            frame_row(
                &format!(
                    "Tokens total  in {}  out {}  all {}",
                    usage.total_input_tokens, usage.total_output_tokens, usage.total_tokens
                ),
                width,
            ),
            frame_row(
                &format!(
                    "Compact slots  S {}  |  M {}  |  R {}",
                    compact_text(simple_slot, 18),
                    compact_text(medium_slot, 18),
                    compact_text(red_slot, 18)
                ),
                width,
            ),
            frame_row(
                &format!(
                    "Last usage  {}  |  in {} out {} all {}",
                    compact_text(&last_model, 72),
                    usage.last_input_tokens,
                    usage.last_output_tokens,
                    usage.last_total_tokens
                ),
                width,
            ),
        ]
    }

    fn setting_value(&self, key: &str) -> Option<&str> {
        self.settings_items
            .iter()
            .find(|item| item.key == key)
            .map(|item| item.value.as_str())
    }

    fn task_rows(&self, width: usize) -> Vec<String> {
        let focus = self.snapshot.focus.as_ref();
        let background_task = focus
            .map(|focus| describe_active_task(&focus.active_task_id, &focus.active_task_title))
            .unwrap_or_else(|| "no active task".to_string());
        let active_chat = self
            .pending_inputs
            .iter()
            .find(|item| item.status == PendingInputStatus::Active)
            .map(|item| {
                let task_ref = item
                    .task_id
                    .map(|task_id| format!("#{task_id} "))
                    .unwrap_or_default();
                format!("{task_ref}{}", compact_text(&item.task_title, 88))
            });
        let mut rows = vec![frame_row(
            &match active_chat {
                Some(active_chat) => format!("Chat     active {}", active_chat),
                None => "Chat     no active owner chat; new input will reprioritize at the next safe boundary".to_string(),
            },
            width,
        )];
        rows.push(frame_row(
            &format!("Loop     {}", compact_text(&background_task, 104)),
            width,
        ));
        rows.extend(self.chat_queue_rows(width, 2));
        rows.extend(self.completed_owner_task_rows(width, 2));
        rows
    }

    fn open_task_rows(&self, width: usize, limit: usize) -> Vec<String> {
        let active_id = self
            .snapshot
            .focus
            .as_ref()
            .and_then(|focus| focus.active_task_id);
        let open_tasks = self
            .snapshot
            .open_tasks
            .iter()
            .filter(|task| Some(task.id) != active_id)
            .collect::<Vec<_>>();
        if open_tasks.is_empty() {
            return vec![
                frame_row("Open     no queued tasks", width),
                frame_row("Open     new owner chat will appear here immediately", width),
            ];
        }

        let mut rows = open_tasks
            .iter()
            .take(limit)
            .enumerate()
            .map(|(index, task)| {
                let prefix = if index == 0 { "Open" } else { "Open+" };
                frame_row(
                    &format!("{prefix:<8} #{} {}", task.id, compact_text(&task.title, 96)),
                    width,
                )
            })
            .collect::<Vec<_>>();
        if open_tasks.len() > limit {
            rows.push(frame_row(
                &format!("Open+    {} more queued task(s)", open_tasks.len() - limit),
                width,
            ));
        }
        while rows.len() < limit {
            rows.push(frame_row(" ", width));
        }
        rows.truncate(limit);
        rows
    }

    fn chat_rows(&self, width: usize, height: usize) -> Vec<String> {
        let mut raw_lines = recent_chat_lines(&self.snapshot, CHAT_HISTORY_LIMIT);
        for line in self.pending_chat_overlay_lines(CHAT_HISTORY_LIMIT) {
            if !raw_lines.iter().any(|existing| existing == &line) {
                raw_lines.push(line);
            }
        }
        if raw_lines.len() > CHAT_HISTORY_LIMIT {
            raw_lines = raw_lines.split_off(raw_lines.len().saturating_sub(CHAT_HISTORY_LIMIT));
        }
        let mut lines = raw_lines
            .into_iter()
            .map(|line| frame_row(&line, width))
            .collect::<Vec<_>>();
        if lines.is_empty() {
            lines.push(frame_row(
                "No chat history yet. Start typing below to create the first chat interrupt.",
                width,
            ));
        }
        while lines.len() < height {
            lines.insert(0, frame_row(" ", width));
        }
        if lines.len() > height {
            lines = lines.split_off(lines.len().saturating_sub(height));
        }
        lines
    }

    fn settings_rows(&self, width: usize, height: usize) -> Vec<String> {
        let mut rows = Vec::new();
        if self.settings_items.is_empty() {
            rows.push(frame_row("No settings fields available.", width));
        } else {
            let max_start = self.settings_items.len().saturating_sub(height);
            let window_start = self.settings_selected.saturating_sub(height / 2).min(max_start);
            let window_end = (window_start + height).min(self.settings_items.len());
            for (index, item) in self.settings_items[window_start..window_end].iter().enumerate() {
                let absolute_index = window_start + index;
                let marker = if absolute_index == self.settings_selected {
                    ">"
                } else {
                    " "
                };
                let choices = if item.choices.is_empty() { "" } else { " *" };
                rows.push(frame_row(
                    &format!(
                        "{marker} {:<18} {}{}",
                        item.label,
                        compact_text(&display_setting_value(item), 88),
                        choices
                    ),
                    width,
                ));
            }
        }
        while rows.len() < height {
            rows.push(frame_row(" ", width));
        }
        rows.truncate(height);
        rows
    }

    fn pending_chat_overlay_lines(&self, limit: usize) -> Vec<String> {
        let mut lines = Vec::new();
        for item in self.pending_inputs.iter().rev().take(limit).rev() {
            let message = compact_text(&item.message, 92);
            if message.is_empty() {
                continue;
            }
            lines.push(format!(
                "{}  {:<9}> {}",
                item.created_at_label,
                "owner",
                message
            ));
        }
        lines
    }

    fn chat_queue_rows(&self, width: usize, limit: usize) -> Vec<String> {
        let mut rows = Vec::new();
        if !self.pending_inputs.is_empty() {
            for item in self.pending_inputs.iter().take(limit) {
                let icon = match item.status {
                    PendingInputStatus::Queued => "queued",
                    PendingInputStatus::Active => "active",
                    PendingInputStatus::Completed => "done",
                };
                let task_ref = item
                    .task_id
                    .map(|task_id| format!("#{task_id} "))
                    .unwrap_or_default();
                rows.push(frame_row(
                    &format!(
                        "Queue    {:<6} {}{}",
                        icon,
                        task_ref,
                        compact_text(&item.task_title, 84)
                    ),
                    width,
                ));
            }
        } else {
            let owner_open_tasks = self
                .snapshot
                .open_tasks
                .iter()
                .filter(|task| is_chat_related_task(task))
                .take(limit)
                .collect::<Vec<_>>();
            if owner_open_tasks.is_empty() {
                rows.push(frame_row(
                    "Queue    no pending owner chat tasks",
                    width,
                ));
            } else {
                for task in owner_open_tasks {
                    rows.push(frame_row(
                        &format!(
                            "Queue    #{} {}",
                            task.id,
                            compact_text(&task.title, 88)
                        ),
                        width,
                    ));
                }
            }
        }
        while rows.len() < limit {
            rows.push(frame_row(" ", width));
        }
        rows.truncate(limit);
        rows
    }

    fn selected_setting_help_line(&self) -> String {
        let Some(item) = self.settings_items.get(self.settings_selected) else {
            return "No field selected.".to_string();
        };
        let mut line = item.help.to_string();
        if matches!(item.key, "simple_model" | "medium_model" | "red_model")
            && item.choices.len() == 1
            && item.choices.first().copied() == Some(GPT_OSS_MODEL)
        {
            line.push_str(" External OpenAI models unlock after setting an OpenAI Key or Grosshirn Key.");
        }
        if !item.choices.is_empty() {
            line.push_str(" Options: ");
            line.push_str(&item.choices.join(" | "));
        }
        if let Some(notice) = self.local_notices.back() {
            line.push_str(" Last: ");
            line.push_str(notice);
        }
        compact_text(&line, 140)
    }

    fn chat_session_label(&self) -> &'static str {
        if !self.pending_inputs.is_empty() || has_recent_owner_chat(&self.snapshot) {
            "CHAT ACTIVE"
        } else {
            "NORMAL TASKS"
        }
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
            "{}  |  {}  |  {}  |  queue {}  |  {}",
            lifecycle.to_uppercase(),
            mode,
            brain_route_label(&self.snapshot.brain_routing.route_mode),
            queue_depth,
            turn
        )
    }

    fn brain_status_line(&self) -> String {
        format!(
            "Brain [{}]  |  Model {}",
            brain_route_label(&self.snapshot.brain_routing.route_mode),
            current_brain_model_label(&self.snapshot)
        )
    }

    fn now_rows(&self, width: usize) -> Vec<String> {
        let mut rows = Vec::new();
        if let Some(turn) = &self.snapshot.active_turn {
            let elapsed = elapsed_text(&turn.created_at).unwrap_or_else(|| "n/a".to_string());
            let spinner = match self.spinner_phase % 4 {
                0 => '|',
                1 => '/',
                2 => '-',
                _ => '\\',
            };
            rows.push(frame_row(
                &format!("#{} {}", turn.task_id, compact_text(&turn.task_title, 120)),
                width,
            ));
            rows.push(frame_row(
                &format!(
                    "{spinner} running  |  {}  |  {}",
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
                            .unwrap_or("Bounded step is still running."),
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
            rows.push(frame_row(
                "Idle. No bounded turn is running right now.",
                width,
            ));
            rows.push(frame_row(
                &format!("#{} {}", turn.task_id, compact_text(&turn.task_title, 120)),
                width,
            ));
            rows.push(frame_row(&format!("Status  {}", turn.status), width));
            rows.push(frame_row(
                &format!(
                    "Summary  {}",
                    compact_text(
                        turn.summary
                            .as_deref()
                            .filter(|summary| !summary.trim().is_empty())
                            .unwrap_or("No summary stored."),
                        140
                    )
                ),
                width,
            ));
        } else {
            rows.push(frame_row("No bounded turns exist yet.", width));
            rows.push(frame_row(
                "Live status will appear here as soon as a task starts.",
                width,
            ));
            rows.push(frame_row(
                "Typing now queues chat to the CTO for the next cycle.",
                width,
            ));
            rows.push(frame_row(
                "Use /status, /turns or /events for deeper inspection.",
                width,
            ));
        }
        rows
    }

    fn next_rows(&self, width: usize) -> Vec<String> {
        let mut rows = Vec::new();
        if !self.pending_inputs.is_empty() {
            for item in self.pending_inputs.iter().take(PENDING_PREVIEW_LIMIT) {
                let icon = match item.status {
                    PendingInputStatus::Queued => "o",
                    PendingInputStatus::Active => "*",
                    PendingInputStatus::Completed => "x",
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
                        "+ {} additional queued input(s)",
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
            .filter(|task| {
                Some(task.id)
                    != self
                        .snapshot
                        .focus
                        .as_ref()
                        .and_then(|focus| focus.active_task_id)
            })
            .take(PENDING_PREVIEW_LIMIT)
            .map(|task| format!("o #{} {}", task.id, compact_text(&task.title, 118)))
            .collect::<Vec<_>>();

        if next_tasks.is_empty() {
            next_tasks.push("No queued owner chats yet. New input will appear here.".to_string());
        }

        for line in next_tasks {
            rows.push(frame_row(&line, width));
        }
        while rows.len() < 4 {
            rows.push(frame_row(" ", width));
        }
        rows
    }

    fn completed_owner_task_rows(&self, width: usize, height: usize) -> Vec<String> {
        let mut rows = Vec::new();
        if self.snapshot.completed_owner_tasks.is_empty() {
            rows.push(frame_row("No completed owner tasks yet.", width));
        } else {
            for task in &self.snapshot.completed_owner_tasks {
                if rows.len() >= height {
                    break;
                }
                rows.push(frame_row(&format_completed_owner_task_line(task), width));
                if rows.len() >= height {
                    break;
                }
                rows.push(frame_row(&format_completed_owner_task_comment(task), width));
            }
        }
        while rows.len() < height {
            rows.push(frame_row(" ", width));
        }
        rows
    }
}

fn collect_ui_snapshot(paths: &Paths) -> AttachUiSnapshot {
    let bios = load_bios(paths);
    let mut boot_entries = load_boot_entries(paths);
    if boot_entries.len() > CHAT_HISTORY_LIMIT * 4 {
        boot_entries = boot_entries.split_off(boot_entries.len() - (CHAT_HISTORY_LIMIT * 4));
    }
    AttachUiSnapshot {
        bios_url: bios_url(&bios.website_path),
        thread: load_agent_thread(paths).ok(),
        brain_routing: load_brain_routing_state(paths).unwrap_or_default(),
        brain_usage: load_brain_usage_rollup(paths).unwrap_or_default(),
        kleinhirn_runtime: load_kleinhirn_runtime_snapshot(paths),
        grosshirn_runtime: load_grosshirn_runtime_snapshot(paths),
        focus: load_focus_state(paths).ok(),
        active_turn: load_active_agent_turn(paths).ok().flatten(),
        last_turn: load_latest_completed_agent_turn(paths).ok().flatten(),
        open_tasks: list_open_tasks(paths, 8).unwrap_or_default(),
        completed_owner_tasks: list_recent_completed_owner_tasks(paths, 8).unwrap_or_default(),
        boot_entries,
        organigram: load_organigram(paths),
        installation_bootstrap: load_installation_bootstrap_state(paths),
    }
}

fn render_attach_banner(paths: &Paths) -> String {
    let snapshot = collect_ui_snapshot(paths);
    let mut lines = vec!["=== CTO-Agent Attach Terminal ===".to_string()];
    if let Some(url) = snapshot.bios_url.as_deref() {
        lines.push(format!("Kleinhirn BIOS: {url}"));
    }
    if let Some(thread) = snapshot.thread.as_ref() {
        lines.push(format!(
            "Loop: lifecycle={} | mode={} | brain={} | queue_depth={}",
            thread.lifecycle_status,
            describe_mode(&thread.current_mode),
            brain_route_label(&snapshot.brain_routing.route_mode),
            thread.queue_depth
        ));
        if !thread.note.trim().is_empty() {
            lines.push(format!("Loop Note: {}", compact_text(&thread.note, 220)));
        }
    }
    lines.push(format!(
        "Brain: {} | Model: {}",
        brain_route_label(&snapshot.brain_routing.route_mode),
        current_brain_model_label(&snapshot)
    ));
    if let Some(focus) = snapshot.focus.as_ref() {
        lines.push(format!(
            "Active Task: {}",
            describe_active_task(&focus.active_task_id, &focus.active_task_title)
        ));
        if !focus.note.trim().is_empty() {
            lines.push(format!("Focus: {}", compact_text(&focus.note, 220)));
        }
    }
    if let Some(turn) = snapshot.active_turn.as_ref() {
        lines.push(format!(
            "Running Turn: turn#{} | task#{} {} | start_mode={}",
            turn.id,
            turn.task_id,
            compact_text(&turn.task_title, 120),
            describe_mode(&turn.mode_at_start)
        ));
        lines.push(format!(
            "Turn Summary: {}",
            compact_text(
                turn.summary
                    .as_deref()
                    .filter(|summary| !summary.trim().is_empty())
                    .unwrap_or("still open, the current bounded step is still running."),
                240
            )
        ));
    } else if let Some(turn) = snapshot.last_turn.as_ref() {
        lines.push(format!(
            "Last Turn: turn#{} | task#{} {} | status={}",
            turn.id,
            turn.task_id,
            compact_text(&turn.task_title, 120),
            turn.status
        ));
        lines.push(format!(
            "Last Summary: {}",
            compact_text(
                turn.summary
                    .as_deref()
                    .filter(|summary| !summary.trim().is_empty())
                    .unwrap_or("no summary stored"),
                240
            )
        ));
    } else {
        lines.push("Turn Summary: no bounded turns exist yet.".to_string());
    }
    lines.push(
        "Chat behavior: every free-form input is queued as chat to the CTO. The running bounded step is finished cleanly first; after that the agent sees the new input in the next reprioritization cycle.".to_string(),
    );
    lines.push("Shortcuts: /help | /status | /turns | /events | Ctrl-P reset".to_string());
    lines.push("-----------------------------------".to_string());
    lines.join("\n")
}

fn load_settings_items(paths: &Paths) -> Vec<SettingsItem> {
    let organigram = load_organigram(paths);
    let installation = load_installation_bootstrap_state(paths);
    let env_map = load_runtime_env_map(paths).unwrap_or_default();
    let external_model_access = has_external_model_access(&env_map);

    let owner_name = first_non_empty(&[
        organigram.owner.name.as_str(),
        installation.owner_name.as_str(),
    ]);
    let owner_email = first_non_empty(&[
        organigram.owner.email.as_str(),
        installation.owner_contact_email.as_str(),
    ]);
    let simple_model = clamp_model_choice(
        "simple_model",
        env_map
            .get("CTO_AGENT_COMPACT_SIMPLE_MODEL")
            .map(String::as_str),
        external_model_access,
    );
    let medium_model = clamp_model_choice(
        "medium_model",
        env_map
            .get("CTO_AGENT_COMPACT_MEDIUM_MODEL")
            .map(String::as_str),
        external_model_access,
    );
    let red_model = clamp_model_choice(
        "red_model",
        env_map.get("CTO_AGENT_COMPACT_RED_MODEL").map(String::as_str),
        external_model_access,
    );

    vec![
        SettingsItem {
            key: "owner_name",
            label: "Owner Name",
            value: owner_name,
            secret: false,
            choices: Vec::new(),
            help: "Human owner name used for chat labeling and trust-aware routing.",
        },
        SettingsItem {
            key: "owner_email",
            label: "Owner Email",
            value: owner_email,
            secret: false,
            choices: Vec::new(),
            help: "Primary owner email address.",
        },
        SettingsItem {
            key: "owner_contact",
            label: "Owner Reach",
            value: installation.owner_contact_info,
            secret: false,
            choices: Vec::new(),
            help: "How the owner can be reached beyond mail, for example phone, Signal, calendar or assistant notes.",
        },
        SettingsItem {
            key: "bind_host",
            label: "Bind Host",
            value: env_map
                .get("CTO_AGENT_BIND_HOST")
                .cloned()
                .filter(|value| !value.trim().is_empty())
                .unwrap_or_else(control_plane_bind_host),
            secret: false,
            choices: Vec::new(),
            help: "Socket bind host for the BIOS/control plane. Use 0.0.0.0 to expose it on the machine IP instead of localhost only.",
        },
        SettingsItem {
            key: "public_base_url",
            label: "Public BIOS URL",
            value: env_map
                .get("CTO_AGENT_PUBLIC_BASE_URL")
                .cloned()
                .filter(|value| !value.trim().is_empty())
                .unwrap_or_else(control_plane_public_base_url),
            secret: false,
            choices: Vec::new(),
            help: "Public base URL shown in the TUI and used for BIOS links, for example https://100.96.1.7:8443.",
        },
        SettingsItem {
            key: "tls_alt_names",
            label: "TLS Alt Names",
            value: env_map
                .get("CTO_AGENT_TLS_ALT_NAMES")
                .cloned()
                .unwrap_or_default(),
            secret: false,
            choices: Vec::new(),
            help: "Comma-separated hostnames or IPs to include in the self-signed BIOS certificate.",
        },
        SettingsItem {
            key: "mail_address",
            label: "Mail Address",
            value: env_map
                .get("CTO_EMAIL_ADDRESS")
                .cloned()
                .unwrap_or_default(),
            secret: false,
            choices: Vec::new(),
            help: "Mailbox address used by the CTO-Agent mail interrupt bridge.",
        },
        SettingsItem {
            key: "mail_password",
            label: "Mail Password",
            value: env_map
                .get("CTO_EMAIL_PASSWORD")
                .cloned()
                .unwrap_or_default(),
            secret: true,
            choices: Vec::new(),
            help: "Mailbox password or app password.",
        },
        SettingsItem {
            key: "mail_imap_host",
            label: "IMAP Host",
            value: env_map
                .get("CTO_EMAIL_IMAP_HOST")
                .cloned()
                .unwrap_or_default(),
            secret: false,
            choices: Vec::new(),
            help: "Incoming IMAP host.",
        },
        SettingsItem {
            key: "mail_imap_port",
            label: "IMAP Port",
            value: env_map
                .get("CTO_EMAIL_IMAP_PORT")
                .cloned()
                .unwrap_or_default(),
            secret: false,
            choices: Vec::new(),
            help: "Incoming IMAP port, for example 993.",
        },
        SettingsItem {
            key: "mail_smtp_host",
            label: "SMTP Host",
            value: env_map
                .get("CTO_EMAIL_SMTP_HOST")
                .cloned()
                .unwrap_or_default(),
            secret: false,
            choices: Vec::new(),
            help: "Outgoing SMTP host.",
        },
        SettingsItem {
            key: "mail_smtp_port",
            label: "SMTP Port",
            value: env_map
                .get("CTO_EMAIL_SMTP_PORT")
                .cloned()
                .unwrap_or_default(),
            secret: false,
            choices: Vec::new(),
            help: "Outgoing SMTP port, for example 465 or 587.",
        },
        SettingsItem {
            key: "openai_api_key",
            label: "OpenAI Key",
            value: env_map.get("OPENAI_API_KEY").cloned().unwrap_or_default(),
            secret: true,
            choices: Vec::new(),
            help: "Generic OpenAI-compatible API key fallback.",
        },
        SettingsItem {
            key: "grosshirn_api_key",
            label: "Grosshirn Key",
            value: env_map
                .get("CTO_AGENT_GROSSHIRN_API_KEY")
                .cloned()
                .unwrap_or_default(),
            secret: true,
            choices: Vec::new(),
            help: "Dedicated API key for external grosshirn routing.",
        },
        SettingsItem {
            key: "simple_model",
            label: "Simple Model",
            value: simple_model,
            secret: false,
            choices: model_choices_for_setting("simple_model", external_model_access),
            help: "Compaction tier 1. The compact cycle chooses this slot when progress remains strong.",
        },
        SettingsItem {
            key: "medium_model",
            label: "Medium Model",
            value: medium_model,
            secret: false,
            choices: model_choices_for_setting("medium_model", external_model_access),
            help: "Compaction tier 2. The compact cycle chooses this slot when progress is mixed or unstable.",
        },
        SettingsItem {
            key: "red_model",
            label: "Red Model",
            value: red_model,
            secret: false,
            choices: model_choices_for_setting("red_model", external_model_access),
            help: "Compaction tier 3. The compact cycle chooses this slot when progress is in the red zone.",
        },
    ]
}

fn save_settings_items(
    paths: &Paths,
    items: &[SettingsItem],
) -> anyhow::Result<SaveSettingsOutcome> {
    let mut organigram = load_organigram(paths);
    let mut installation = load_installation_bootstrap_state(paths);
    let mut env_map = load_runtime_env_map(paths).unwrap_or_default();
    let previous_bind_host = env_map.get("CTO_AGENT_BIND_HOST").cloned().unwrap_or_default();
    let previous_public_base_url = env_map
        .get("CTO_AGENT_PUBLIC_BASE_URL")
        .cloned()
        .unwrap_or_default();
    let previous_tls_alt_names = env_map
        .get("CTO_AGENT_TLS_ALT_NAMES")
        .cloned()
        .unwrap_or_default();
    let mut saw_mail_address = false;
    let current_local_model = env_map
        .get("CTO_AGENT_KLEINHIRN_RUNTIME_MODEL")
        .cloned()
        .or_else(|| env_map.get("CTO_AGENT_KLEINHIRN_MODEL").cloned())
        .map(|value| normalize_runtime_model_choice(&value))
        .unwrap_or_default();
    let mut requested_local_model: Option<String> = None;

    for item in items {
        let value = item.value.trim().to_string();
        match item.key {
            "owner_name" => {
                organigram.owner.name = value.clone();
                installation.owner_name = value;
            }
            "owner_email" => {
                organigram.owner.email = value.clone();
                installation.owner_contact_email = value;
            }
            "owner_contact" => installation.owner_contact_info = value,
            "bind_host" => upsert_env_value(&mut env_map, "CTO_AGENT_BIND_HOST", &value),
            "public_base_url" => {
                upsert_env_value(&mut env_map, "CTO_AGENT_PUBLIC_BASE_URL", &value)
            }
            "tls_alt_names" => upsert_env_value(&mut env_map, "CTO_AGENT_TLS_ALT_NAMES", &value),
            "mail_address" => {
                saw_mail_address = !value.is_empty();
                upsert_env_value(&mut env_map, "CTO_EMAIL_ADDRESS", &value);
            }
            "mail_password" => upsert_env_value(&mut env_map, "CTO_EMAIL_PASSWORD", &value),
            "mail_imap_host" => upsert_env_value(&mut env_map, "CTO_EMAIL_IMAP_HOST", &value),
            "mail_imap_port" => upsert_env_value(&mut env_map, "CTO_EMAIL_IMAP_PORT", &value),
            "mail_smtp_host" => upsert_env_value(&mut env_map, "CTO_EMAIL_SMTP_HOST", &value),
            "mail_smtp_port" => upsert_env_value(&mut env_map, "CTO_EMAIL_SMTP_PORT", &value),
            "openai_api_key" => upsert_env_value(&mut env_map, "OPENAI_API_KEY", &value),
            "grosshirn_api_key" => {
                upsert_env_value(&mut env_map, "CTO_AGENT_GROSSHIRN_API_KEY", &value)
            }
            "kleinhirn_model" => {
                requested_local_model = Some(value);
            }
            "grosshirn_model" => upsert_env_value(
                &mut env_map,
                "CTO_AGENT_GROSSHIRN_MODEL",
                &normalize_runtime_model_choice(&value),
            ),
            "simple_model" => upsert_env_value(
                &mut env_map,
                "CTO_AGENT_COMPACT_SIMPLE_MODEL",
                &normalize_runtime_model_choice(&value),
            ),
            "medium_model" => upsert_env_value(
                &mut env_map,
                "CTO_AGENT_COMPACT_MEDIUM_MODEL",
                &normalize_runtime_model_choice(&value),
            ),
            "red_model" => upsert_env_value(
                &mut env_map,
                "CTO_AGENT_COMPACT_RED_MODEL",
                &normalize_runtime_model_choice(&value),
            ),
            _ => {}
        }
    }

    let external_model_access = has_external_model_access(&env_map);
    requested_local_model = requested_local_model.map(|requested| {
        clamp_model_choice(
            "kleinhirn_model",
            Some(requested.as_str()),
            external_model_access,
        )
    });
    clamp_persisted_model_choice(
        &mut env_map,
        "CTO_AGENT_COMPACT_SIMPLE_MODEL",
        "simple_model",
        external_model_access,
    );
    clamp_persisted_model_choice(
        &mut env_map,
        "CTO_AGENT_COMPACT_MEDIUM_MODEL",
        "medium_model",
        external_model_access,
    );
    clamp_persisted_model_choice(
        &mut env_map,
        "CTO_AGENT_COMPACT_RED_MODEL",
        "red_model",
        external_model_access,
    );
    let effective_red_model = env_map
        .get("CTO_AGENT_COMPACT_RED_MODEL")
        .cloned()
        .unwrap_or_else(|| {
            default_model_choice_for_setting("red_model", external_model_access).to_string()
        });
    upsert_env_value(
        &mut env_map,
        "CTO_AGENT_GROSSHIRN_MODEL",
        &effective_red_model,
    );

    if saw_mail_address {
        installation.email_assignment_mode = "assigned_now".to_string();
    }
    if !installation.owner_name.trim().is_empty()
        || !installation.owner_contact_email.trim().is_empty()
        || !installation.owner_contact_info.trim().is_empty()
        || saw_mail_address
    {
        installation.status = "captured".to_string();
    }

    organigram.updated_at = now_iso();
    installation.updated_at = now_iso();
    save_organigram(paths, &organigram)?;
    save_installation_bootstrap_state(paths, &installation)?;
    save_runtime_env_map(paths, &env_map)?;

    let runtime_restart_required = env_map
        .get("CTO_AGENT_BIND_HOST")
        .cloned()
        .unwrap_or_default()
        != previous_bind_host
        || env_map
            .get("CTO_AGENT_PUBLIC_BASE_URL")
            .cloned()
            .unwrap_or_default()
            != previous_public_base_url
        || env_map
            .get("CTO_AGENT_TLS_ALT_NAMES")
            .cloned()
            .unwrap_or_default()
            != previous_tls_alt_names;

    if let Some(local_model) = requested_local_model
        .filter(|value| !value.trim().is_empty())
        .filter(|value| !value.eq_ignore_ascii_case(&current_local_model))
    {
        apply_targeted_kleinhirn_upgrade(paths, Some(&local_model))
            .with_context(|| format!("failed to switch active local kleinhirn to {local_model}"))?;
    }

    Ok(SaveSettingsOutcome {
        runtime_restart_required,
    })
}

fn first_non_empty(values: &[&str]) -> String {
    values
        .iter()
        .map(|value| value.trim())
        .find(|value| !value.is_empty())
        .unwrap_or("")
        .to_string()
}

fn has_external_model_access(env_map: &BTreeMap<String, String>) -> bool {
    ["OPENAI_API_KEY", "CTO_AGENT_GROSSHIRN_API_KEY"]
        .iter()
        .filter_map(|key| env_map.get(*key))
        .any(|value| !value.trim().is_empty())
}

fn model_choices_for_setting(setting_key: &str, external_model_access: bool) -> Vec<&'static str> {
    match setting_key {
        "kleinhirn_model" => LOCAL_MODEL_OPTIONS.to_vec(),
        "grosshirn_model" => {
            if external_model_access {
                GROSSHIRN_MODEL_OPTIONS.to_vec()
            } else {
                LOCAL_MODEL_OPTIONS.to_vec()
            }
        }
        "simple_model" => {
            if external_model_access {
                SIMPLE_MODEL_OPTIONS.to_vec()
            } else {
                LOCAL_MODEL_OPTIONS.to_vec()
            }
        }
        "medium_model" => {
            if external_model_access {
                MEDIUM_MODEL_OPTIONS.to_vec()
            } else {
                LOCAL_MODEL_OPTIONS.to_vec()
            }
        }
        "red_model" => {
            if external_model_access {
                RED_MODEL_OPTIONS.to_vec()
            } else {
                LOCAL_MODEL_OPTIONS.to_vec()
            }
        }
        _ => Vec::new(),
    }
}

fn default_model_choice_for_setting(
    setting_key: &str,
    external_model_access: bool,
) -> &'static str {
    if !external_model_access {
        return GPT_OSS_MODEL;
    }
    match setting_key {
        "kleinhirn_model" => GPT_OSS_MODEL,
        "grosshirn_model" => "openai/gpt-5.4",
        "simple_model" => GPT_OSS_MODEL,
        "medium_model" => "openai/gpt-5.4-mini",
        "red_model" => "openai/gpt-5.4",
        _ => GPT_OSS_MODEL,
    }
}

fn clamp_model_choice(
    setting_key: &str,
    raw_value: Option<&str>,
    external_model_access: bool,
) -> String {
    let choices = model_choices_for_setting(setting_key, external_model_access);
    let default_choice = default_model_choice_for_setting(setting_key, external_model_access);
    let Some(raw_value) = raw_value else {
        return default_choice.to_string();
    };
    let normalized = normalize_runtime_model_choice(raw_value);
    if choices
        .iter()
        .any(|choice| choice.eq_ignore_ascii_case(&normalized))
    {
        normalized
    } else {
        default_choice.to_string()
    }
}

fn clamp_persisted_model_choice(
    env_map: &mut BTreeMap<String, String>,
    env_key: &str,
    setting_key: &str,
    external_model_access: bool,
) {
    let clamped = clamp_model_choice(
        setting_key,
        env_map.get(env_key).map(String::as_str),
        external_model_access,
    );
    upsert_env_value(env_map, env_key, &clamped);
}

fn upsert_env_value(env_map: &mut BTreeMap<String, String>, key: &str, value: &str) {
    if value.trim().is_empty() {
        env_map.remove(key);
    } else {
        env_map.insert(key.to_string(), value.trim().to_string());
    }
}

fn display_setting_value(item: &SettingsItem) -> String {
    if item.value.trim().is_empty() {
        return "(empty)".to_string();
    }
    if !item.secret {
        return item.value.clone();
    }
    let visible_tail = item
        .value
        .chars()
        .rev()
        .take(4)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect::<String>();
    let mask_len = item.value.chars().count().saturating_sub(visible_tail.chars().count());
    format!("{}{}", "*".repeat(mask_len.max(4)), visible_tail)
}

fn recent_chat_lines(snapshot: &AttachUiSnapshot, limit: usize) -> Vec<String> {
    let mut lines = Vec::new();
    let mut last_signature: Option<(String, String)> = None;
    for entry in snapshot
        .boot_entries
        .iter()
        .filter(|entry| is_chat_entry(snapshot, entry))
    {
        let speaker_label = chat_speaker_label(snapshot, &entry.speaker).to_string();
        let message = compact_text(&entry.message, 92);
        let signature = (speaker_label.clone(), message.clone());
        if last_signature.as_ref() == Some(&signature) {
            continue;
        }
        lines.push(format!(
            "{}  {:<9}> {}",
            short_timestamp(&entry.timestamp),
            speaker_label,
            message
        ));
        last_signature = Some(signature);
    }
    if lines.len() > limit {
        lines = lines.split_off(lines.len().saturating_sub(limit));
    }
    lines
}

fn has_recent_owner_chat(snapshot: &AttachUiSnapshot) -> bool {
    snapshot
        .boot_entries
        .iter()
        .rev()
        .find(|entry| owner_matches_snapshot(snapshot, &entry.speaker))
        .and_then(|entry| DateTime::parse_from_rfc3339(&entry.timestamp).ok())
        .map(|dt| {
            let age = Utc::now().signed_duration_since(dt.with_timezone(&Utc));
            age.num_seconds() >= 0 && age.num_seconds() as u64 <= CHAT_IDLE_SECS
        })
        .unwrap_or(false)
}

fn is_chat_entry(snapshot: &AttachUiSnapshot, entry: &BootEntry) -> bool {
    if owner_matches_snapshot(snapshot, &entry.speaker) {
        return true;
    }
    is_cto_speaker(&entry.speaker) && is_owner_visible_chat_text(&entry.message)
}

fn chat_speaker_label(snapshot: &AttachUiSnapshot, speaker: &str) -> &'static str {
    if is_cto_speaker(speaker) {
        "cto-agent"
    } else if owner_matches_snapshot(snapshot, speaker) {
        "owner"
    } else {
        "chat"
    }
}

fn is_cto_speaker(speaker: &str) -> bool {
    speaker.trim().eq_ignore_ascii_case("cto-agent")
}

fn is_owner_visible_chat_text(message: &str) -> bool {
    let trimmed = message.trim();
    if trimmed.is_empty() {
        return false;
    }
    if trimmed.starts_with('{') || trimmed.starts_with('[') {
        return false;
    }
    let lowered = trimmed.to_ascii_lowercase();
    !lowered.contains("\"taskstatus\"")
        && !lowered.contains("\"nextmode\"")
        && !lowered.contains("\"checkpointsummary\"")
        && !lowered.contains("the kleinhirn endpoint responded with unusable output")
        && !lowered.contains("reclassify the task instead")
        && !lowered.contains("current agent mode: mode=")
}

fn owner_visible_input_echo(message: &str) -> String {
    message.trim().to_string()
}

fn owner_matches_snapshot(snapshot: &AttachUiSnapshot, speaker: &str) -> bool {
    let speaker = speaker.trim();
    if speaker.is_empty() || is_cto_speaker(speaker) || speaker.eq_ignore_ascii_case("installer") {
        return false;
    }

    let candidates = [
        snapshot.organigram.owner.name.as_str(),
        snapshot.organigram.owner.email.as_str(),
        snapshot.installation_bootstrap.owner_name.as_str(),
        snapshot.installation_bootstrap.owner_contact_email.as_str(),
    ];

    let speaker_lower = speaker.to_ascii_lowercase();
    let has_named_owner = candidates.iter().any(|candidate| !candidate.trim().is_empty());
    for candidate in candidates {
        let candidate = candidate.trim();
        if candidate.is_empty() {
            continue;
        }
        let candidate_lower = candidate.to_ascii_lowercase();
        if speaker_lower == candidate_lower
            || speaker_lower.contains(&candidate_lower)
            || candidate_lower.contains(&speaker_lower)
        {
            return true;
        }
    }

    !has_named_owner
}

fn is_chat_related_task(task: &TaskRecord) -> bool {
    task.source_channel == "attach_terminal"
        || task.task_kind == "owner_interrupt"
        || task.title.trim().eq_ignore_ascii_case("Chatten")
}

fn bios_url(website_path: &str) -> Option<String> {
    let path = website_path.trim();
    if path.is_empty() {
        return None;
    }
    if path.starts_with("http://") || path.starts_with("https://") {
        return Some(path.to_string());
    }
    let normalized_path = if path.starts_with('/') {
        path.to_string()
    } else {
        format!("/{path}")
    };
    Some(format!(
        "{}{}",
        control_plane_public_base_url(),
        normalized_path
    ))
}

fn perform_attach_factory_reset(paths: &Paths) -> anyhow::Result<()> {
    stop_running_runtime_for_factory_reset(paths)?;
    factory_reset_installation(paths)?;
    spawn_detached_runtime(paths)?;
    wait_for_attach_runtime(paths, Duration::from_secs(12))
}

fn restart_runtime_after_network_settings_change(paths: &Paths) -> anyhow::Result<()> {
    if let Some(pid) = runtime_pid_from_lock(paths).filter(|pid| *pid != std::process::id()) {
        if process_is_alive(pid) {
            send_signal(pid, "-TERM")
                .with_context(|| format!("failed to restart CTO-Agent process {pid}"))?;
            let _ = wait_for_process_exit(pid, Duration::from_secs(6));
        }
    }

    if wait_for_attach_runtime(paths, Duration::from_secs(12)).is_ok() {
        return Ok(());
    }

    spawn_detached_runtime(paths)?;
    wait_for_attach_runtime(paths, Duration::from_secs(12))
}

fn stop_running_runtime_for_factory_reset(paths: &Paths) -> anyhow::Result<()> {
    if let Some(pid) = runtime_pid_from_lock(paths).filter(|pid| *pid != std::process::id()) {
        if process_is_alive(pid) {
            send_signal(pid, "-TERM")
                .with_context(|| format!("failed to stop running CTO-Agent process {pid}"))?;
            if !wait_for_process_exit(pid, Duration::from_secs(5)) {
                send_signal(pid, "-KILL").with_context(|| {
                    format!("failed to force-stop stuck CTO-Agent process {pid}")
                })?;
                if !wait_for_process_exit(pid, Duration::from_secs(2)) {
                    anyhow::bail!("CTO-Agent process {pid} did not stop for factory reset");
                }
            }
        }
    }

    let _ = fs::remove_file(&paths.runtime_lock_path);
    let _ = fs::remove_file(&paths.attach_socket_path);
    Ok(())
}

fn spawn_detached_runtime(paths: &Paths) -> anyhow::Result<()> {
    let current_exe = std::env::current_exe().context("failed to resolve current executable")?;
    Command::new(current_exe)
        .current_dir(&paths.root)
        .env("CTO_AGENT_ROOT", &paths.root)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .context("failed to restart CTO-Agent after factory reset")?;
    Ok(())
}

fn wait_for_attach_runtime(paths: &Paths, timeout: Duration) -> anyhow::Result<()> {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if attach_socket_ready(paths) {
            return Ok(());
        }
        thread::sleep(Duration::from_millis(150));
    }
    anyhow::bail!("factory reset succeeded, but the CTO-Agent attach socket did not come back")
}

fn attach_socket_ready(paths: &Paths) -> bool {
    if !paths.attach_socket_path.exists() {
        return false;
    }
    StdUnixStream::connect(&paths.attach_socket_path).is_ok()
}

fn runtime_pid_from_lock(paths: &Paths) -> Option<u32> {
    let text = fs::read_to_string(&paths.runtime_lock_path).ok()?;
    serde_json::from_str::<serde_json::Value>(&text)
        .ok()
        .and_then(|value| value.get("pid").and_then(|item| item.as_u64()))
        .map(|value| value as u32)
}

fn process_is_alive(pid: u32) -> bool {
    Command::new("kill")
        .arg("-0")
        .arg(pid.to_string())
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

fn send_signal(pid: u32, signal: &str) -> anyhow::Result<()> {
    let status = Command::new("kill")
        .arg(signal)
        .arg(pid.to_string())
        .status()
        .with_context(|| format!("failed to invoke kill {} {}", signal, pid))?;
    if !status.success() {
        anyhow::bail!("kill {} {} exited with {}", signal, pid, status);
    }
    Ok(())
}

fn wait_for_process_exit(pid: u32, timeout: Duration) -> bool {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if !process_is_alive(pid) {
            return true;
        }
        thread::sleep(Duration::from_millis(150));
    }
    !process_is_alive(pid)
}

fn describe_mode(mode: &str) -> String {
    match mode {
        "reprioritize" => "reprioritize (meta/prioritization mode)".to_string(),
        "execute_task" => "execute_task (task mode)".to_string(),
        "recovery" => "recovery (stabilization after restart or incident)".to_string(),
        "self_preservation" => "self_preservation (loop protection)".to_string(),
        other => other.to_string(),
    }
}

fn describe_active_task(active_task_id: &Option<i64>, active_task_title: &str) -> String {
    match active_task_id {
        Some(task_id) if !active_task_title.trim().is_empty() => {
            format!("#{} {}", task_id, compact_text(active_task_title, 140))
        }
        Some(task_id) => format!("#{}", task_id),
        None if active_task_title.trim().is_empty() => "no active task".to_string(),
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

fn brain_route_label(route_mode: &str) -> &'static str {
    match route_mode {
        "grosshirn" => "GROSSHIRN",
        _ => "KLEINHIRN",
    }
}

fn current_brain_model_label(snapshot: &AttachUiSnapshot) -> String {
    if snapshot.brain_routing.route_mode == "grosshirn" {
        return snapshot
            .grosshirn_runtime
            .as_ref()
            .map(|runtime| {
                format_model_label(runtime.official_label.as_deref(), runtime.model.as_deref())
            })
            .unwrap_or_else(|| "externes Modell unbekannt".to_string());
    }

    snapshot
        .kleinhirn_runtime
        .as_ref()
        .map(|runtime| {
            format_model_label(
                runtime
                    .official_label
                    .as_deref()
                    .or(runtime.policy_model.as_deref()),
                runtime
                    .runtime_model
                    .as_deref()
                    .or(runtime.policy_model.as_deref()),
            )
        })
        .unwrap_or_else(|| "lokales Modell unbekannt".to_string())
}

fn format_model_label(label: Option<&str>, model_id: Option<&str>) -> String {
    let label = label.map(str::trim).filter(|value| !value.is_empty());
    let model_id = model_id.map(str::trim).filter(|value| !value.is_empty());
    match (label, model_id) {
        (Some(label), Some(model_id)) if !label.eq_ignore_ascii_case(model_id) => {
            format!("{label} ({model_id})")
        }
        (Some(label), _) => label.to_string(),
        (None, Some(model_id)) => model_id.to_string(),
        (None, None) => "unbekannt".to_string(),
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
    let left = format!("[{}]", title);
    let right = compact_text(right.trim(), inner / 2);
    let mut body = if right.is_empty() {
        left
    } else {
        format!("{left} {right}")
    };
    body = fit_line(&body, inner);
    let fill = "-".repeat(inner.saturating_sub(body.chars().count()));
    format!("+{}{}+", body, fill)
}

fn frame_separator(title: &str, width: usize) -> String {
    if width <= 2 {
        return fit_line(title, width);
    }
    let inner = width - 2;
    let label = format!("-- {} ", title);
    let label_len = label.chars().count();
    if label_len >= inner {
        return format!("+{}+", pad_to_width(&fit_line(title, inner), inner));
    }
    let fill = "-".repeat(inner - label_len);
    format!("+{}{}+", label, fill)
}

fn frame_row(content: &str, width: usize) -> String {
    if width <= 4 {
        return fit_line(content, width);
    }
    let inner = width - 4;
    let fitted = fit_line(content, inner);
    format!("| {} |", pad_to_width(&fitted, inner))
}

fn frame_bottom(width: usize) -> String {
    if width <= 2 {
        return String::new();
    }
    format!("+{}+", "-".repeat(width - 2))
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
        "interrupt/replied" => "reply",
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

fn format_completed_owner_task_line(task: &TaskRecord) -> String {
    compact_text(
        &format!(
            "{}  [done] #{} {}",
            short_timestamp(&task.updated_at),
            task.id,
            task.title
        ),
        140,
    )
}

fn format_completed_owner_task_comment(task: &TaskRecord) -> String {
    let comment = task
        .last_output
        .as_deref()
        .filter(|value| !value.trim().is_empty())
        .or_else(|| {
            task.last_checkpoint_summary
                .as_deref()
                .filter(|value| !value.trim().is_empty())
        })
        .and_then(|value| value.lines().map(str::trim).find(|line| !line.is_empty()))
        .unwrap_or("No completion note stored.");
    format!("   {}", compact_text(comment, 132))
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
    let display_input = normalize_input_for_display(input);
    let prompt_width = prompt.chars().count();
    let available = width.saturating_sub(prompt_width);
    if available == 0 {
        return (String::new(), prompt_width.min(width));
    }

    let input_width = display_input.chars().count();
    if input_width <= available {
        return (display_input, prompt_width + input_width);
    }

    if available <= 3 {
        let tail = display_input
            .chars()
            .rev()
            .take(available)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect::<String>();
        return (tail, width);
    }

    let tail = display_input
        .chars()
        .rev()
        .take(available - 3)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect::<String>();
    (format!("...{tail}"), width)
}

fn normalize_input_for_display(input: &str) -> String {
    input
        .replace("\r\n", "\n")
        .replace('\r', "\n")
        .replace('\n', " \\n ")
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
    Some(format_elapsed(Duration::from_secs(
        elapsed.num_seconds() as u64
    )))
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
        "interrupt/replied" => "INTERRUPT reply",
        "turn/started" => "TURN start",
        "turn/completed" => "TURN done",
        "turn/interrupt" => "TURN interrupt",
        "turn/steer" => "TURN steer",
        "task/queued" => "TASK queued",
        "task/selected" => "TASK selected",
        "task/completed" => "TASK done",
        "task/blocked" => "TASK blocked",
        "task/requeued" => "TASK requeued",
        "context/progress" => "CONTEXT progress",
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
    use std::ffi::OsString;
    use std::sync::Mutex;
    use std::sync::OnceLock;
    use std::time::SystemTime;
    use std::time::UNIX_EPOCH;

    fn env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    fn unique_test_root(label: &str) -> std::path::PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time should be after unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "cto_agent_attach_{label}_{}_{}",
            std::process::id(),
            nanos
        ))
    }

    struct EnvGuard(Option<OsString>);

    impl EnvGuard {
        fn set_cto_root(root: &std::path::Path) -> Self {
            let previous = std::env::var_os("CTO_AGENT_ROOT");
            unsafe {
                std::env::set_var("CTO_AGENT_ROOT", root);
            }
            Self(previous)
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            if let Some(previous) = self.0.take() {
                unsafe {
                    std::env::set_var("CTO_AGENT_ROOT", previous);
                }
            } else {
                unsafe {
                    std::env::remove_var("CTO_AGENT_ROOT");
                }
            }
        }
    }

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
            summary: Some("Bounded step is still running.".to_string()),
            output: None,
            completed_at: None,
        }
    }

    fn sample_ui() -> AttachTui {
        let mut notices = VecDeque::new();
        notices.push_back(
            "16:09  you     Michael Welsch: Please make the BIOS link more visible".to_string(),
        );
        notices
            .push_back("16:09  queued  #413 Show the BIOS link on the attach screen".to_string());
        notices
            .push_back("16:09  interrupt  #412 Signal recorded on the running thread.".to_string());
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
                brain_routing: BrainRoutingState {
                    route_mode: "kleinhirn".to_string(),
                    boosted_task_id: None,
                    boosted_task_title: String::new(),
                    boost_reason: String::new(),
                    boost_started_at: None,
                    boost_last_used_at: None,
                    boost_expires_at: None,
                    cooldown_until: None,
                    last_deactivation_reason: String::new(),
                },
                brain_usage: BrainUsageRollup {
                    total_input_tokens: 4200,
                    total_output_tokens: 1700,
                    total_tokens: 5900,
                    last_model_id: "openai/gpt-oss-20b".to_string(),
                    last_brain_tier: "kleinhirn".to_string(),
                    last_input_tokens: 512,
                    last_output_tokens: 181,
                    last_total_tokens: 693,
                    last_recorded_at: Some("2026-03-18T15:08:30+00:00".to_string()),
                },
                kleinhirn_runtime: Some(KleinhirnRuntimeSnapshot {
                    policy_model: Some("gpt-oss-20b".to_string()),
                    runtime_model: Some("openai/gpt-oss-20b".to_string()),
                    official_label: Some("GPT-OSS 20B".to_string()),
                    adapter: Some("mistralrs_gpt_oss_harmony_completion".to_string()),
                }),
                grosshirn_runtime: Some(GrosshirnRuntimeSnapshot {
                    model: Some("gpt-5.4".to_string()),
                    official_label: Some("GPT-5.4".to_string()),
                    adapter: Some("openai_responses".to_string()),
                }),
                focus: Some(FocusStateRecord {
                    mode: "execute_task".to_string(),
                    active_task_id: Some(412),
                    active_task_title: "Check the watchdog regression in the browser installer"
                        .to_string(),
                    queue_depth: 2,
                    last_reprioritized_at: None,
                    last_task_completed_at: None,
                    note: String::new(),
                }),
                active_turn: Some(sample_turn(
                    184,
                    412,
                    "Check the watchdog regression in the browser installer",
                )),
                last_turn: Some(AgentTurnRecord {
                    id: 183,
                    created_at: "2026-03-18T15:06:00+00:00".to_string(),
                    updated_at: "2026-03-18T15:07:00+00:00".to_string(),
                    task_id: 409,
                    task_title: "Secure the browser health check".to_string(),
                    trigger: "supervisor_tick".to_string(),
                    mode_at_start: "review".to_string(),
                    mode_at_end: Some("reprioritize".to_string()),
                    status: "completed".to_string(),
                    summary: Some("Browser health check secured.".to_string()),
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
                        title: "Show the BIOS link on the attach screen".to_string(),
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
                        title: "Make the interrupt queue more visible in the UI".to_string(),
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
                completed_owner_tasks: vec![
                    TaskRecord {
                        id: 409,
                        created_at: "2026-03-18T15:05:00+00:00".to_string(),
                        updated_at: "2026-03-18T15:07:00+00:00".to_string(),
                        parent_task_id: None,
                        worker_job_id: None,
                        source_interrupt_id: None,
                        source_channel: "bios".to_string(),
                        speaker: "Michael Welsch".to_string(),
                        task_kind: "owner_interrupt".to_string(),
                        title: "Secure the browser health check".to_string(),
                        detail: String::new(),
                        trust_level: "owner".to_string(),
                        priority_score: 1000,
                        status: "done".to_string(),
                        run_count: 1,
                        last_checkpoint_summary: Some("Browser health check secured.".to_string()),
                        last_checkpoint_at: Some("2026-03-18T15:07:00+00:00".to_string()),
                        last_output: Some("All checks green.".to_string()),
                    },
                    TaskRecord {
                        id: 408,
                        created_at: "2026-03-18T15:01:00+00:00".to_string(),
                        updated_at: "2026-03-18T15:04:00+00:00".to_string(),
                        parent_task_id: None,
                        worker_job_id: None,
                        source_interrupt_id: None,
                        source_channel: "terminal".to_string(),
                        speaker: "Michael Welsch".to_string(),
                        task_kind: "owner_interrupt".to_string(),
                        title: "Restart the browser watchdog".to_string(),
                        detail: String::new(),
                        trust_level: "owner".to_string(),
                        priority_score: 990,
                        status: "done".to_string(),
                        run_count: 1,
                        last_checkpoint_summary: Some("Watchdog restarted.".to_string()),
                        last_checkpoint_at: Some("2026-03-18T15:04:00+00:00".to_string()),
                        last_output: Some("watchdog restart ok".to_string()),
                    },
                ],
                boot_entries: vec![
                    BootEntry {
                        timestamp: "2026-03-18T15:09:00+00:00".to_string(),
                        speaker: "Michael Welsch".to_string(),
                        message: "Please make the BIOS link more visible".to_string(),
                    },
                    BootEntry {
                        timestamp: "2026-03-18T15:09:30+00:00".to_string(),
                        speaker: "cto-agent".to_string(),
                        message: "The request is queued as Chatten and will reprioritize at the next safe boundary.".to_string(),
                    },
                ],
                organigram: Organigram {
                    owner: crate::contracts::OwnerRef {
                        name: "Michael Welsch".to_string(),
                        email: "michael.welsch@metric-space.ai".to_string(),
                        role: "owner".to_string(),
                    },
                    reports_to: String::new(),
                    ceo: String::new(),
                    board: Vec::new(),
                    peer_cxos: Vec::new(),
                    subordinates: crate::contracts::Subordinates {
                        people: Vec::new(),
                        agents: Vec::new(),
                        vendors: Vec::new(),
                    },
                    updated_at: "2026-03-18T15:09:00+00:00".to_string(),
                },
                installation_bootstrap: InstallationBootstrapState {
                    version: 1,
                    status: "captured".to_string(),
                    owner_name: "Michael Welsch".to_string(),
                    owner_contact_email: "michael.welsch@metric-space.ai".to_string(),
                    owner_contact_info: "Signal first, then mail.".to_string(),
                    terminal_command: "cto".to_string(),
                    terminal_low_level_note: "Terminal fallback remains available.".to_string(),
                    dashboard_note: "Dashboard ready.".to_string(),
                    owner_may_drop_later_via_terminal_or_dashboard: true,
                    email_assignment_mode: "assigned_now".to_string(),
                    email_bootstrap_note: String::new(),
                    installer_free_text: String::new(),
                    updated_at: "2026-03-18T15:09:00+00:00".to_string(),
                },
            },
            event_lines: vec![
                "16:08  task active  #412 Task was pulled into active work.".to_string(),
                "16:08  turn start   #412 Turn 184 started.".to_string(),
            ],
            local_notices: notices,
            pending_inputs: vec![
                PendingInputItem {
                    task_id: Some(413),
                    task_title: "Show the BIOS link on the attach screen".to_string(),
                    message: "Michael Welsch: Please make the BIOS link more visible".to_string(),
                    created_at_label: "16:09".to_string(),
                    status: PendingInputStatus::Queued,
                    completed_at: None,
                    seen_open: true,
                },
                PendingInputItem {
                    task_id: Some(414),
                    task_title: "Make the interrupt queue more visible in the UI".to_string(),
                    message: "Michael Welsch: after that please show me the last three turns"
                        .to_string(),
                    created_at_label: "16:09".to_string(),
                    status: PendingInputStatus::Queued,
                    completed_at: None,
                    seen_open: true,
                },
            ],
            page: AttachPage::Chat,
            chat_input: "Michael Welsch: after that please show me the last three turns"
                .to_string(),
            settings_items: vec![
                SettingsItem {
                    key: "owner_name",
                    label: "Owner Name",
                    value: "Michael Welsch".to_string(),
                    secret: false,
                    choices: Vec::new(),
                    help: "Owner identity.",
                },
                SettingsItem {
                    key: "simple_model",
                    label: "Simple Model",
                    value: "openai/gpt-oss-20b".to_string(),
                    secret: false,
                    choices: SIMPLE_MODEL_OPTIONS.to_vec(),
                    help: "Simple slot.",
                },
            ],
            settings_selected: 0,
            settings_dirty: false,
            spinner_phase: 0,
            factory_reset_armed_until: None,
        }
    }

    #[test]
    fn attach_primary_status_matches_reference_headline() {
        let ui = sample_ui();
        let headline = ui.primary_status_line();
        assert!(headline.contains("RUNNING"));
        assert!(headline.contains("EXEC"));
        assert!(headline.contains("KLEINHIRN"));
        assert!(headline.contains("queue 2"));
        assert!(headline.contains("turn #184"));
    }

    #[test]
    fn input_display_flattens_multiline_paste_for_preview() {
        let (display, cursor_col) = input_display("owner:\nline two", 80, "Chat to CTO: ");
        assert_eq!(display, "owner: \\n line two");
        assert_eq!(
            cursor_col,
            "Chat to CTO: owner: \\n line two".chars().count()
        );
    }

    #[test]
    fn model_setting_choices_collapse_without_external_key() {
        let env_map = BTreeMap::new();
        assert!(!has_external_model_access(&env_map));
        assert_eq!(
            model_choices_for_setting("simple_model", false),
            vec![GPT_OSS_MODEL]
        );
        assert_eq!(
            model_choices_for_setting("grosshirn_model", false),
            vec![GPT_OSS_MODEL]
        );
        assert_eq!(
            clamp_model_choice("red_model", Some("openai/gpt-5.4"), false),
            GPT_OSS_MODEL
        );
    }

    #[test]
    fn model_setting_choices_unlock_with_external_key() {
        let mut env_map = BTreeMap::new();
        env_map.insert("OPENAI_API_KEY".to_string(), "sk-test".to_string());
        assert!(has_external_model_access(&env_map));
        assert_eq!(
            model_choices_for_setting("medium_model", true),
            MEDIUM_MODEL_OPTIONS.to_vec()
        );
        assert_eq!(
            default_model_choice_for_setting("medium_model", true),
            "openai/gpt-5.4-mini"
        );
        assert_eq!(
            clamp_model_choice("red_model", Some("openai/gpt-5.4"), true),
            "openai/gpt-5.4"
        );
    }

    #[test]
    fn load_settings_items_only_exposes_three_compaction_model_controls() -> anyhow::Result<()> {
        let _guard = env_lock().lock().expect("test env lock poisoned");
        let root = unique_test_root("settings_model_controls");
        std::fs::create_dir_all(root.join("runtime"))?;
        std::fs::write(
            root.join("runtime/kleinhirn.env"),
            "OPENAI_API_KEY='sk-test'\nCTO_AGENT_COMPACT_SIMPLE_MODEL='openai/gpt-oss-20b'\nCTO_AGENT_COMPACT_MEDIUM_MODEL='openai/gpt-5.4-mini'\nCTO_AGENT_COMPACT_RED_MODEL='openai/gpt-5.4'\nCTO_AGENT_GROSSHIRN_MODEL='openai/gpt-5.4'\nCTO_AGENT_KLEINHIRN_RUNTIME_MODEL='openai/gpt-oss-20b'\n",
        )?;
        let _env = EnvGuard::set_cto_root(&root);
        let paths = Paths::discover()?;

        let items = load_settings_items(&paths);
        let model_keys = items
            .iter()
            .filter(|item| item.key.ends_with("_model"))
            .map(|item| item.key)
            .collect::<Vec<_>>();

        assert_eq!(model_keys, vec!["simple_model", "medium_model", "red_model"]);

        std::fs::remove_dir_all(&root).ok();
        Ok(())
    }

    #[test]
    fn save_settings_items_persist_mail_and_bios_network_settings() -> anyhow::Result<()> {
        let _guard = env_lock().lock().expect("test env lock poisoned");
        let root = unique_test_root("settings_persist");
        std::fs::create_dir_all(root.join("runtime"))?;
        std::fs::write(
            root.join("runtime/kleinhirn.env"),
            "CTO_AGENT_KLEINHIRN_RUNTIME_MODEL='openai/gpt-oss-20b'\n",
        )?;
        let _env = EnvGuard::set_cto_root(&root);
        let paths = Paths::discover()?;
        let items = vec![
            SettingsItem {
                key: "bind_host",
                label: "Bind Host",
                value: "0.0.0.0".to_string(),
                secret: false,
                choices: Vec::new(),
                help: "",
            },
            SettingsItem {
                key: "public_base_url",
                label: "Public BIOS URL",
                value: "https://100.96.1.7:8443".to_string(),
                secret: false,
                choices: Vec::new(),
                help: "",
            },
            SettingsItem {
                key: "tls_alt_names",
                label: "TLS Alt Names",
                value: "100.96.1.7".to_string(),
                secret: false,
                choices: Vec::new(),
                help: "",
            },
            SettingsItem {
                key: "mail_address",
                label: "Mail Address",
                value: "cto1@metric-space.ai".to_string(),
                secret: false,
                choices: Vec::new(),
                help: "",
            },
            SettingsItem {
                key: "mail_password",
                label: "Mail Password",
                value: "supersecret".to_string(),
                secret: true,
                choices: Vec::new(),
                help: "",
            },
        ];

        let outcome = save_settings_items(&paths, &items)?;
        let env_map = load_runtime_env_map(&paths)?;

        assert!(outcome.runtime_restart_required);
        assert_eq!(
            env_map.get("CTO_AGENT_BIND_HOST").map(String::as_str),
            Some("0.0.0.0")
        );
        assert_eq!(
            env_map.get("CTO_AGENT_PUBLIC_BASE_URL").map(String::as_str),
            Some("https://100.96.1.7:8443")
        );
        assert_eq!(
            env_map.get("CTO_AGENT_TLS_ALT_NAMES").map(String::as_str),
            Some("100.96.1.7")
        );
        assert_eq!(
            env_map.get("CTO_EMAIL_ADDRESS").map(String::as_str),
            Some("cto1@metric-space.ai")
        );
        assert_eq!(
            env_map.get("CTO_EMAIL_PASSWORD").map(String::as_str),
            Some("supersecret")
        );

        std::fs::remove_dir_all(&root).ok();
        Ok(())
    }

    #[test]
    fn attach_reference_sections_contain_expected_rows() {
        let ui = sample_ui();
        let now_rows = ui.now_rows(120).join("\n");
        let next_rows = ui.next_rows(120).join("\n");
        let done_rows = ui.completed_owner_task_rows(120, 8).join("\n");

        assert!(ui.brain_status_line().contains("Brain [KLEINHIRN]"));
        assert!(ui.brain_status_line().contains("GPT-OSS 20B"));
        assert!(now_rows.contains("Summary  Bounded step is still running."));
        assert!(now_rows.contains("Last done  #409 Secure the browser health check"));
        assert!(next_rows.contains("o #413 Show the BIOS link on the attach screen"));
        assert!(next_rows.contains("o #414 Make the interrupt queue more visible in the UI"));
        assert!(done_rows.contains("16:07  [done] #409 Secure the browser health check"));
        assert!(done_rows.contains("All checks green."));
        assert!(done_rows.contains("16:04  [done] #408 Restart the browser watchdog"));
        assert!(done_rows.contains("watchdog restart ok"));
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

    #[test]
    fn interrupt_reply_events_render_with_reply_label() {
        let event = AgentEventRecord {
            id: 2,
            created_at: "2026-03-18T15:10:00+00:00".to_string(),
            method: "interrupt/replied".to_string(),
            active_task_id: Some(413),
            active_task_title: "Answer the owner question".to_string(),
            body: "The mail draft is ready.".to_string(),
            payload_json: "{}".to_string(),
        };
        assert!(format_activity_line(&event).contains("reply"));
        assert!(format_event_line(&event).contains("INTERRUPT reply"));
    }

    #[test]
    fn recent_chat_lines_hide_internal_cto_noise_and_dedup_adjacent_duplicates() {
        let mut ui = sample_ui();
        ui.snapshot.boot_entries = vec![
            BootEntry {
                timestamp: "2026-03-18T15:09:00+00:00".to_string(),
                speaker: "Michael Welsch".to_string(),
                message: "Hast du die Mail schon eingerichtet?".to_string(),
            },
            BootEntry {
                timestamp: "2026-03-18T15:09:30+00:00".to_string(),
                speaker: "cto-agent".to_string(),
                message: "Ich prüfe gerade den Mailpfad.".to_string(),
            },
            BootEntry {
                timestamp: "2026-03-18T15:09:31+00:00".to_string(),
                speaker: "cto-agent".to_string(),
                message: "Ich prüfe gerade den Mailpfad.".to_string(),
            },
            BootEntry {
                timestamp: "2026-03-18T15:09:40+00:00".to_string(),
                speaker: "cto-agent".to_string(),
                message:
                    "The kleinhirn endpoint responded with unusable output. Reclassify the task instead.".to_string(),
            },
            BootEntry {
                timestamp: "2026-03-18T15:09:50+00:00".to_string(),
                speaker: "cto-agent".to_string(),
                message:
                    "{\"taskStatus\":\"continue\",\"nextMode\":\"execute_task\"}".to_string(),
            },
        ];

        let lines = recent_chat_lines(&ui.snapshot, 12).join("\n");
        assert!(lines.contains("owner"));
        assert!(lines.contains("cto-agent"));
        assert!(!lines.contains("unusable output"));
        assert!(!lines.contains("\"taskStatus\""));
        assert_eq!(lines.matches("Ich prüfe gerade den Mailpfad.").count(), 1);
    }

    #[test]
    fn chat_rows_keep_pending_owner_input_visible_before_persisted_history_catches_up() {
        let mut ui = sample_ui();
        ui.snapshot.boot_entries.clear();

        let lines = ui.chat_rows(140, 8).join("\n");

        assert!(lines.contains("owner"));
        assert!(lines.contains("Please make the BIOS link more visible"));
        assert!(lines.contains("after that please show me the last three turns"));
    }

    #[test]
    fn owner_visible_input_echo_keeps_chat_input_raw() {
        let line = owner_visible_input_echo(
            "email: cto1@metric-space.ai password: supersecret imap host: imap.one.com",
        );

        assert_eq!(
            line,
            "email: cto1@metric-space.ai password: supersecret imap host: imap.one.com"
        );
    }

    #[test]
    fn tasks_page_has_no_active_editor_prompt() {
        let mut ui = sample_ui();
        ui.page = AttachPage::Tasks;

        assert_eq!(ui.page_label(), "TASKS");
        assert!(ui.editor_prompt().is_empty());
        assert!(ui.editor_text().is_empty());
        assert!(ui.active_editor_is_empty());
    }

    #[test]
    fn tab_cycles_tasks_chat_settings() -> anyhow::Result<()> {
        let _guard = env_lock().lock().expect("test env lock poisoned");
        let root = unique_test_root("tab_cycle_pages");
        std::fs::create_dir_all(&root)?;
        let _env = EnvGuard::set_cto_root(&root);
        let paths = Paths::discover()?;
        let mut ui = sample_ui();
        ui.page = AttachPage::Tasks;

        ui.handle_key_event(&paths, KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE));
        assert_eq!(ui.page, AttachPage::Chat);

        ui.handle_key_event(&paths, KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE));
        assert_eq!(ui.page, AttachPage::Settings);

        ui.handle_key_event(&paths, KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE));
        assert_eq!(ui.page, AttachPage::Tasks);

        std::fs::remove_dir_all(&root).ok();
        Ok(())
    }

    #[test]
    fn ctrl_p_arms_factory_reset_before_executing() -> anyhow::Result<()> {
        let _guard = env_lock().lock().expect("test env lock poisoned");
        let root = unique_test_root("ctrl_p_confirm");
        std::fs::create_dir_all(&root)?;
        let _env = EnvGuard::set_cto_root(&root);
        let paths = Paths::discover()?;
        let mut ui = sample_ui();

        let outcome = ui.handle_key_event(
            &paths,
            KeyEvent::new(KeyCode::Char('p'), KeyModifiers::CONTROL),
        );

        assert!(matches!(outcome, TuiControl::Continue));
        assert!(ui.factory_reset_armed_until.is_some());
        assert!(
            ui.local_notices
                .back()
                .expect("expected local notice")
                .contains("Factory reset armed")
        );

        std::fs::remove_dir_all(&root).ok();
        Ok(())
    }
}
