use anyhow::Context;
use anyhow::Result;
use crossterm::event;
use crossterm::event::Event as TerminalEvent;
use crossterm::event::KeyCode;
use crossterm::event::KeyEvent;
use crossterm::event::KeyModifiers;
use crossterm::execute;
use crossterm::terminal;
use crossterm::terminal::EnterAlternateScreen;
use crossterm::terminal::LeaveAlternateScreen;
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use std::collections::BTreeMap;
use std::io;
use std::path::Path;
use std::path::PathBuf;
use std::process::Command;
use std::sync::mpsc;
use std::sync::mpsc::Receiver;
use std::thread;
use std::time::Duration;
use std::time::Instant;

use crate::channels;
use crate::execution_baseline;
use crate::lcm;
use crate::responses_proxy::ProxyTelemetry;
use crate::runtime_config;

mod render;

const CHAT_CONVERSATION_ID: i64 = 1;
const DEFAULT_ACTIVE_MODEL: &str = "openai/gpt-oss-20b";
const DEFAULT_MAX_CONTEXT: usize = 131_072;
const DEFAULT_COMPACTION_THRESHOLD_PERCENT: usize = 75;
const CTOX_CHAT_SYSTEM_PROMPT: &str = include_str!("../prompts/ctox_chat_system_prompt.md");

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Page {
    Chat,
    Settings,
}

#[derive(Debug, Clone)]
struct SettingItem {
    key: &'static str,
    label: &'static str,
    value: String,
    secret: bool,
    choices: Vec<&'static str>,
    help: &'static str,
}

#[derive(Debug, Clone)]
struct HeaderState {
    model: String,
    max_context: usize,
    compact_at: usize,
    current_tokens: usize,
    tokens_per_second: Option<f64>,
    last_input_tokens: Option<u64>,
    last_output_tokens: Option<u64>,
    last_total_tokens: Option<u64>,
}

impl Default for HeaderState {
    fn default() -> Self {
        Self {
            model: DEFAULT_ACTIVE_MODEL.to_string(),
            max_context: DEFAULT_MAX_CONTEXT,
            compact_at: DEFAULT_MAX_CONTEXT * DEFAULT_COMPACTION_THRESHOLD_PERCENT / 100,
            current_tokens: 0,
            tokens_per_second: None,
            last_input_tokens: None,
            last_output_tokens: None,
            last_total_tokens: None,
        }
    }
}

#[derive(Debug)]
enum WorkerEvent {
    Completed { reply: String },
    Failed { message: String },
}

struct App {
    root: PathBuf,
    db_path: PathBuf,
    page: Page,
    chat_input: String,
    chat_messages: Vec<lcm::MessageRecord>,
    status_line: String,
    spinner_phase: usize,
    header: HeaderState,
    settings_items: Vec<SettingItem>,
    settings_selected: usize,
    settings_dirty: bool,
    worker_rx: Option<Receiver<WorkerEvent>>,
    request_in_flight: bool,
}

pub fn run_tui(root: &Path) -> Result<()> {
    let db_path = root.join("runtime/ctox_lcm.db");
    let _ = lcm::LcmEngine::open(&db_path, lcm::LcmConfig::default())?;

    let mut stdout = io::stdout();
    let _guard = TerminalGuard::enter(&mut stdout)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend).context("failed to initialize TUI terminal")?;
    let mut app = App::new(root.to_path_buf(), db_path);
    app.refresh()?;
    terminal
        .draw(|frame| render::draw(frame, &app))
        .context("failed to draw initial TUI frame")?;

    let mut last_refresh = Instant::now();
    loop {
        if event::poll(Duration::from_millis(125)).context("failed to poll terminal events")? {
            match event::read().context("failed to read terminal event")? {
                TerminalEvent::Key(key_event) => {
                    if app.handle_key_event(key_event)? {
                        break;
                    }
                }
                TerminalEvent::Resize(_, _) => {}
                TerminalEvent::Paste(text) => app.handle_paste(&text),
                _ => {}
            }
        }

        if last_refresh.elapsed() >= Duration::from_millis(350) {
            app.refresh()?;
            last_refresh = Instant::now();
        }

        app.poll_worker()?;
        app.spinner_phase = (app.spinner_phase + 1) % 4;
        terminal
            .draw(|frame| render::draw(frame, &app))
            .context("failed to draw TUI frame")?;
    }

    Ok(())
}

impl App {
    fn new(root: PathBuf, db_path: PathBuf) -> Self {
        Self {
            root: root.clone(),
            db_path,
            page: Page::Chat,
            chat_input: String::new(),
            chat_messages: Vec::new(),
            status_line: "Tab switches between Chat and Settings.".to_string(),
            spinner_phase: 0,
            header: HeaderState::default(),
            settings_items: load_settings_items(&root),
            settings_selected: 0,
            settings_dirty: false,
            worker_rx: None,
            request_in_flight: false,
        }
    }

    fn handle_paste(&mut self, text: &str) {
        match self.page {
            Page::Chat => self.chat_input.push_str(text),
            Page::Settings => {
                if let Some(item) = self.settings_items.get_mut(self.settings_selected) {
                    item.value.push_str(text);
                    self.settings_dirty = true;
                }
            }
        }
    }

    fn handle_key_event(&mut self, key_event: KeyEvent) -> Result<bool> {
        if key_event.modifiers.contains(KeyModifiers::CONTROL)
            && matches!(key_event.code, KeyCode::Char('c') | KeyCode::Char('q'))
        {
            return Ok(true);
        }
        if key_event.modifiers.contains(KeyModifiers::CONTROL)
            && matches!(key_event.code, KeyCode::Char('s'))
        {
            self.save_settings()?;
            return Ok(false);
        }

        match key_event.code {
            KeyCode::Tab | KeyCode::BackTab => {
                self.page = match self.page {
                    Page::Chat => Page::Settings,
                    Page::Settings => Page::Chat,
                };
                return Ok(false);
            }
            _ => {}
        }

        match self.page {
            Page::Chat => self.handle_chat_key(key_event)?,
            Page::Settings => self.handle_settings_key(key_event)?,
        }

        Ok(false)
    }

    fn handle_chat_key(&mut self, key_event: KeyEvent) -> Result<()> {
        match key_event.code {
            KeyCode::Enter => self.submit_chat_request()?,
            KeyCode::Backspace => {
                self.chat_input.pop();
            }
            KeyCode::Char(ch) if !key_event.modifiers.contains(KeyModifiers::CONTROL) => {
                self.chat_input.push(ch);
            }
            _ => {}
        }
        Ok(())
    }

    fn handle_settings_key(&mut self, key_event: KeyEvent) -> Result<()> {
        match key_event.code {
            KeyCode::Up => {
                self.settings_selected = self.settings_selected.saturating_sub(1);
            }
            KeyCode::Down => {
                self.settings_selected =
                    (self.settings_selected + 1).min(self.settings_items.len().saturating_sub(1));
            }
            KeyCode::Left => self.cycle_setting(false),
            KeyCode::Right => self.cycle_setting(true),
            KeyCode::Backspace => {
                if let Some(item) = self.settings_items.get_mut(self.settings_selected) {
                    item.value.pop();
                    self.settings_dirty = true;
                }
            }
            KeyCode::Enter => self.cycle_setting(true),
            KeyCode::Char(ch) if !key_event.modifiers.contains(KeyModifiers::CONTROL) => {
                if let Some(item) = self.settings_items.get_mut(self.settings_selected) {
                    if item.choices.is_empty() {
                        item.value.push(ch);
                        self.settings_dirty = true;
                    }
                }
            }
            _ => {}
        }
        Ok(())
    }

    fn cycle_setting(&mut self, forward: bool) {
        let Some(item) = self.settings_items.get_mut(self.settings_selected) else {
            return;
        };
        if item.choices.is_empty() {
            return;
        }
        let current_index = item
            .choices
            .iter()
            .position(|choice| choice.eq_ignore_ascii_case(item.value.trim()))
            .unwrap_or(0);
        let next_index = if forward {
            (current_index + 1) % item.choices.len()
        } else if current_index == 0 {
            item.choices.len() - 1
        } else {
            current_index - 1
        };
        item.value = item.choices[next_index].to_string();
        self.settings_dirty = true;
    }

    fn submit_chat_request(&mut self) -> Result<()> {
        let prompt = self.chat_input.trim().to_string();
        if prompt.is_empty() {
            self.status_line = "Chat input is empty.".to_string();
            return Ok(());
        }
        if self.request_in_flight {
            self.status_line = "A Codex request is already running.".to_string();
            return Ok(());
        }
        let settings = settings_map_from_items(&self.settings_items);
        let max_context = read_usize_setting(&settings, "CTOX_CHAT_MODEL_MAX_CONTEXT", DEFAULT_MAX_CONTEXT);
        let prompt_for_worker = prompt.clone();
        self.chat_input.clear();
        self.status_line = "Running Codex request...".to_string();
        self.request_in_flight = true;
        lcm::run_add_message(&self.db_path, CHAT_CONVERSATION_ID, "user", &prompt)
            .context("failed to persist user message into LCM")?;
        self.refresh_chat_messages()?;

        let root = self.root.clone();
        let db_path = self.db_path.clone();
        let (tx, rx) = mpsc::channel();
        self.worker_rx = Some(rx);
        thread::spawn(move || {
            let result = run_chat_turn(&root, &db_path, &settings, max_context, &prompt_for_worker);
            let event = match result {
                Ok(reply) => WorkerEvent::Completed { reply },
                Err(err) => WorkerEvent::Failed {
                    message: err.to_string(),
                },
            };
            let _ = tx.send(event);
        });
        Ok(())
    }

    fn poll_worker(&mut self) -> Result<()> {
        let Some(rx) = self.worker_rx.as_ref() else {
            return Ok(());
        };
        let Ok(event) = rx.try_recv() else {
            return Ok(());
        };
        self.worker_rx = None;
        self.request_in_flight = false;
        match event {
            WorkerEvent::Completed { reply } => {
                self.status_line = format!("Assistant replied with {} chars.", reply.chars().count());
            }
            WorkerEvent::Failed { message } => {
                self.status_line = format!("Codex request failed: {message}");
                let _ = lcm::run_add_message(
                    &self.db_path,
                    CHAT_CONVERSATION_ID,
                    "assistant",
                    &format!("Request failed: {message}"),
                );
            }
        }
        self.refresh()?;
        Ok(())
    }

    fn refresh(&mut self) -> Result<()> {
        self.refresh_chat_messages()?;
        self.refresh_header();
        Ok(())
    }

    fn refresh_chat_messages(&mut self) -> Result<()> {
        let snapshot = lcm::run_dump(&self.db_path, CHAT_CONVERSATION_ID)?;
        self.chat_messages = snapshot.messages;
        Ok(())
    }

    fn refresh_header(&mut self) {
        let settings = settings_map_from_items(&self.settings_items);
        let max_context = read_usize_setting(&settings, "CTOX_CHAT_MODEL_MAX_CONTEXT", DEFAULT_MAX_CONTEXT);
        let compact_percent = read_usize_setting(
            &settings,
            "CTOX_CHAT_COMPACTION_THRESHOLD_PERCENT",
            DEFAULT_COMPACTION_THRESHOLD_PERCENT,
        )
        .clamp(1, 99);
        let compact_at = max_context.saturating_mul(compact_percent) / 100;
        let current_tokens = current_context_tokens(&self.db_path, max_context).unwrap_or(0);
        let proxy_telemetry = load_proxy_telemetry(&self.root).ok().flatten();
        self.header = HeaderState {
            model: settings
                .get("CTOX_CHAT_MODEL")
                .cloned()
                .filter(|value| !value.trim().is_empty())
                .unwrap_or_else(|| DEFAULT_ACTIVE_MODEL.to_string()),
            max_context,
            compact_at,
            current_tokens,
            tokens_per_second: proxy_telemetry.as_ref().and_then(|value| value.last_tokens_per_second),
            last_input_tokens: proxy_telemetry.as_ref().and_then(|value| value.last_input_tokens),
            last_output_tokens: proxy_telemetry.as_ref().and_then(|value| value.last_output_tokens),
            last_total_tokens: proxy_telemetry.as_ref().and_then(|value| value.last_total_tokens),
        };
    }

    fn save_settings(&mut self) -> Result<()> {
        let env_map = settings_map_from_items(&self.settings_items);
        runtime_config::save_runtime_env_map(&self.root, &env_map)?;
        channels::sync_prompt_identity(&self.root, &env_map)?;
        self.settings_dirty = false;
        self.status_line = "Settings saved to runtime/ctox.env.".to_string();
        self.refresh_header();
        Ok(())
    }

    fn header_lines(&self, width: usize) -> Vec<String> {
        let model_line = format!(
            "Model {}  Max Context {}  Compact @ {}  Current {}  TPS {}",
            self.header.model,
            self.header.max_context,
            self.header.compact_at,
            self.header.current_tokens,
            self.header
                .tokens_per_second
                .map(|value| format!("{value:.1}"))
                .unwrap_or_else(|| "-".to_string())
        );
        let usage_line = format!(
            "Last usage in/out/total: {}/{}/{}",
            self.header
                .last_input_tokens
                .map(|value| value.to_string())
                .unwrap_or_else(|| "-".to_string()),
            self.header
                .last_output_tokens
                .map(|value| value.to_string())
                .unwrap_or_else(|| "-".to_string()),
            self.header
                .last_total_tokens
                .map(|value| value.to_string())
                .unwrap_or_else(|| "-".to_string())
        );
        let bar_width = width.saturating_sub(16).max(12);
        let marker_index = if self.header.max_context == 0 {
            0
        } else {
            ((self.header.compact_at.min(self.header.max_context) as f64 / self.header.max_context as f64)
                * bar_width as f64)
                .floor() as usize
        };
        let fill_index = if self.header.max_context == 0 {
            0
        } else {
            ((self.header.current_tokens.min(self.header.max_context) as f64
                / self.header.max_context as f64)
                * bar_width as f64)
                .floor() as usize
        };
        let mut bar = String::with_capacity(bar_width + 2);
        bar.push('[');
        for idx in 0..bar_width {
            if idx == marker_index.min(bar_width.saturating_sub(1)) {
                bar.push('|');
            } else if idx < fill_index {
                bar.push('=');
            } else {
                bar.push(' ');
            }
        }
        bar.push(']');
        vec![model_line, usage_line, format!("Context {bar}")]
    }
}

fn load_settings_items(root: &Path) -> Vec<SettingItem> {
    let env_map = runtime_config::load_runtime_env_map(root).unwrap_or_default();
    let active_model = env_map
        .get("CTOX_CHAT_MODEL")
        .cloned()
        .unwrap_or_else(|| DEFAULT_ACTIVE_MODEL.to_string());
    let max_context = env_map
        .get("CTOX_CHAT_MODEL_MAX_CONTEXT")
        .cloned()
        .unwrap_or_else(|| DEFAULT_MAX_CONTEXT.to_string());
    let compaction_threshold = env_map
        .get("CTOX_CHAT_COMPACTION_THRESHOLD_PERCENT")
        .cloned()
        .unwrap_or_else(|| DEFAULT_COMPACTION_THRESHOLD_PERCENT.to_string());
    vec![
        SettingItem {
            key: "CTOX_OWNER_NAME",
            label: "Owner Name",
            value: env_map.get("CTOX_OWNER_NAME").cloned().unwrap_or_default(),
            secret: false,
            choices: Vec::new(),
            help: "Display name of the personal owner of CTOX.",
        },
        SettingItem {
            key: "CTOX_OWNER_PREFERRED_CHANNEL",
            label: "Owner Channel",
            value: env_map
                .get("CTOX_OWNER_PREFERRED_CHANNEL")
                .cloned()
                .unwrap_or_default(),
            secret: false,
            choices: vec!["tui", "email", "jami"],
            help: "Preferred outbound owner contact channel when CTOX initiates contact.",
        },
        SettingItem {
            key: "CTOX_CHAT_MODEL",
            label: "Active Model",
            value: active_model,
            secret: false,
            choices: vec!["openai/gpt-oss-20b", "Qwen/Qwen3.5-27B"],
            help: "Verified local model families exposed by CTOX.",
        },
        SettingItem {
            key: "CTOX_CHAT_MODEL_MAX_CONTEXT",
            label: "Max Context",
            value: max_context,
            secret: false,
            choices: Vec::new(),
            help: "Header context window size in tokens.",
        },
        SettingItem {
            key: "CTOX_CHAT_COMPACTION_THRESHOLD_PERCENT",
            label: "Compact %",
            value: compaction_threshold,
            secret: false,
            choices: vec!["60", "70", "75", "80", "85"],
            help: "Header and LCM compaction trigger as a percentage of max context.",
        },
        SettingItem {
            key: "OPENAI_API_KEY",
            label: "OpenAI Key",
            value: env_map.get("OPENAI_API_KEY").cloned().unwrap_or_default(),
            secret: true,
            choices: Vec::new(),
            help: "Used when the active model is not routed through the local proxy.",
        },
        SettingItem {
            key: "CTO_AGENT_COMPACT_SIMPLE_MODEL",
            label: "Simple Model",
            value: env_map
                .get("CTO_AGENT_COMPACT_SIMPLE_MODEL")
                .cloned()
                .unwrap_or_else(|| DEFAULT_ACTIVE_MODEL.to_string()),
            secret: false,
            choices: vec!["openai/gpt-oss-20b", "Qwen/Qwen3.5-27B"],
            help: "Compact tier 1 local model slot.",
        },
        SettingItem {
            key: "CTO_AGENT_COMPACT_MEDIUM_MODEL",
            label: "Medium Model",
            value: env_map
                .get("CTO_AGENT_COMPACT_MEDIUM_MODEL")
                .cloned()
                .unwrap_or_else(|| "Qwen/Qwen3.5-27B".to_string()),
            secret: false,
            choices: vec!["openai/gpt-oss-20b", "Qwen/Qwen3.5-27B"],
            help: "Compact tier 2 local model slot.",
        },
        SettingItem {
            key: "CTO_AGENT_COMPACT_RED_MODEL",
            label: "Red Model",
            value: env_map
                .get("CTO_AGENT_COMPACT_RED_MODEL")
                .cloned()
                .unwrap_or_else(|| "Qwen/Qwen3.5-27B".to_string()),
            secret: false,
            choices: vec!["openai/gpt-oss-20b", "Qwen/Qwen3.5-27B"],
            help: "Compact tier 3 local model slot.",
        },
        SettingItem {
            key: "CTO_EMAIL_PROVIDER",
            label: "Mail Provider",
            value: env_map
                .get("CTO_EMAIL_PROVIDER")
                .cloned()
                .unwrap_or_else(|| "imap".to_string()),
            secret: false,
            choices: vec!["imap", "graph", "ews", "activesync"],
            help: "Email adapter mode. Outlook works through graph or ews.",
        },
        SettingItem {
            key: "CTO_EMAIL_ADDRESS",
            label: "Mail Address",
            value: env_map.get("CTO_EMAIL_ADDRESS").cloned().unwrap_or_default(),
            secret: false,
            choices: Vec::new(),
            help: "Primary mailbox address.",
        },
        SettingItem {
            key: "CTO_EMAIL_PASSWORD",
            label: "Mail Password",
            value: env_map.get("CTO_EMAIL_PASSWORD").cloned().unwrap_or_default(),
            secret: true,
            choices: Vec::new(),
            help: "IMAP, SMTP or basic-auth password.",
        },
        SettingItem {
            key: "CTO_EMAIL_IMAP_HOST",
            label: "IMAP Host",
            value: env_map.get("CTO_EMAIL_IMAP_HOST").cloned().unwrap_or_default(),
            secret: false,
            choices: Vec::new(),
            help: "Incoming IMAP host.",
        },
        SettingItem {
            key: "CTO_EMAIL_IMAP_PORT",
            label: "IMAP Port",
            value: env_map.get("CTO_EMAIL_IMAP_PORT").cloned().unwrap_or_else(|| "993".to_string()),
            secret: false,
            choices: Vec::new(),
            help: "Incoming IMAP port.",
        },
        SettingItem {
            key: "CTO_EMAIL_SMTP_HOST",
            label: "SMTP Host",
            value: env_map.get("CTO_EMAIL_SMTP_HOST").cloned().unwrap_or_default(),
            secret: false,
            choices: Vec::new(),
            help: "Outgoing SMTP host.",
        },
        SettingItem {
            key: "CTO_EMAIL_SMTP_PORT",
            label: "SMTP Port",
            value: env_map.get("CTO_EMAIL_SMTP_PORT").cloned().unwrap_or_else(|| "465".to_string()),
            secret: false,
            choices: Vec::new(),
            help: "Outgoing SMTP port.",
        },
        SettingItem {
            key: "CTO_EMAIL_GRAPH_ACCESS_TOKEN",
            label: "Graph Token",
            value: env_map
                .get("CTO_EMAIL_GRAPH_ACCESS_TOKEN")
                .cloned()
                .unwrap_or_default(),
            secret: true,
            choices: Vec::new(),
            help: "Microsoft Graph bearer token for Outlook or M365 mailboxes.",
        },
        SettingItem {
            key: "CTO_EMAIL_GRAPH_USER",
            label: "Graph User",
            value: env_map
                .get("CTO_EMAIL_GRAPH_USER")
                .cloned()
                .unwrap_or_else(|| "me".to_string()),
            secret: false,
            choices: Vec::new(),
            help: "Graph user target, for example me.",
        },
        SettingItem {
            key: "CTO_EMAIL_EWS_URL",
            label: "EWS URL",
            value: env_map.get("CTO_EMAIL_EWS_URL").cloned().unwrap_or_default(),
            secret: false,
            choices: Vec::new(),
            help: "Exchange Web Services endpoint for Outlook or Exchange.",
        },
        SettingItem {
            key: "CTO_EMAIL_EWS_AUTH_TYPE",
            label: "EWS Auth",
            value: env_map
                .get("CTO_EMAIL_EWS_AUTH_TYPE")
                .cloned()
                .unwrap_or_else(|| "basic".to_string()),
            secret: false,
            choices: vec!["basic", "bearer"],
            help: "EWS auth mode.",
        },
        SettingItem {
            key: "CTO_EMAIL_EWS_USERNAME",
            label: "EWS User",
            value: env_map
                .get("CTO_EMAIL_EWS_USERNAME")
                .cloned()
                .unwrap_or_default(),
            secret: false,
            choices: Vec::new(),
            help: "EWS username when auth mode is basic.",
        },
        SettingItem {
            key: "CTO_EMAIL_EWS_BEARER_TOKEN",
            label: "EWS Token",
            value: env_map
                .get("CTO_EMAIL_EWS_BEARER_TOKEN")
                .cloned()
                .unwrap_or_default(),
            secret: true,
            choices: Vec::new(),
            help: "EWS bearer token when auth mode is bearer.",
        },
        SettingItem {
            key: "CTO_JAMI_ACCOUNT_ID",
            label: "Jami Account",
            value: env_map.get("CTO_JAMI_ACCOUNT_ID").cloned().unwrap_or_default(),
            secret: false,
            choices: Vec::new(),
            help: "Jami account id used by the transport adapter.",
        },
        SettingItem {
            key: "CTO_JAMI_PROFILE_NAME",
            label: "Jami Profile",
            value: env_map.get("CTO_JAMI_PROFILE_NAME").cloned().unwrap_or_default(),
            secret: false,
            choices: Vec::new(),
            help: "Jami profile display name.",
        },
        SettingItem {
            key: "CTO_JAMI_INBOX_DIR",
            label: "Jami Inbox",
            value: env_map.get("CTO_JAMI_INBOX_DIR").cloned().unwrap_or_default(),
            secret: false,
            choices: Vec::new(),
            help: "Inbound Jami drop directory.",
        },
        SettingItem {
            key: "CTO_JAMI_OUTBOX_DIR",
            label: "Jami Outbox",
            value: env_map.get("CTO_JAMI_OUTBOX_DIR").cloned().unwrap_or_default(),
            secret: false,
            choices: Vec::new(),
            help: "Outbound Jami drop directory.",
        },
        SettingItem {
            key: "CTO_JAMI_ARCHIVE_DIR",
            label: "Jami Archive",
            value: env_map.get("CTO_JAMI_ARCHIVE_DIR").cloned().unwrap_or_default(),
            secret: false,
            choices: Vec::new(),
            help: "Archived Jami message directory.",
        },
        SettingItem {
            key: "CTO_JAMI_DBUS_ENV_FILE",
            label: "Jami DBus Env",
            value: env_map
                .get("CTO_JAMI_DBUS_ENV_FILE")
                .cloned()
                .unwrap_or_default(),
            secret: false,
            choices: Vec::new(),
            help: "DBus environment file used by the Jami adapter.",
        },
    ]
}

fn settings_map_from_items(items: &[SettingItem]) -> BTreeMap<String, String> {
    let mut env_map = BTreeMap::new();
    for item in items {
        let trimmed = item.value.trim();
        if trimmed.is_empty() {
            continue;
        }
        env_map.insert(item.key.to_string(), trimmed.to_string());
    }
    env_map
}

fn run_chat_turn(
    root: &Path,
    db_path: &Path,
    settings: &BTreeMap<String, String>,
    max_context: usize,
    prompt: &str,
) -> Result<String> {
    let engine = lcm::LcmEngine::open(db_path, lcm::LcmConfig::default())?;
    let _ = engine.continuity_init_documents(CHAT_CONVERSATION_ID)?;
    let decision = engine.evaluate_compaction(CHAT_CONVERSATION_ID, max_context as i64)?;
    if decision.should_compact {
        let _ = engine.compact(
            CHAT_CONVERSATION_ID,
            max_context as i64,
            &lcm::HeuristicSummarizer,
            false,
        )?;
    }
    refresh_continuity_documents(root, settings, &engine)?;
    let snapshot = engine.snapshot(CHAT_CONVERSATION_ID)?;
    let continuity = engine.continuity_show_all(CHAT_CONVERSATION_ID)?;
    let rendered_prompt = render_chat_prompt(&snapshot, &continuity, prompt);
    let reply = invoke_codex_exec(root, settings, &rendered_prompt)?;
    lcm::run_add_message(db_path, CHAT_CONVERSATION_ID, "assistant", &reply)?;
    let engine = lcm::LcmEngine::open(db_path, lcm::LcmConfig::default())?;
    refresh_continuity_documents(root, settings, &engine)?;
    Ok(reply)
}

fn render_chat_prompt(
    snapshot: &lcm::LcmSnapshot,
    continuity: &lcm::ContinuityShowAll,
    latest_user_prompt: &str,
) -> String {
    let mut lines = vec![
        "Reply to the latest user turn using the structured context below.".to_string(),
        String::new(),
        "Continuity documents:".to_string(),
        continuity_block("Narrative", &continuity.narrative.content),
        continuity_block("Anchors", &continuity.anchors.content),
        continuity_block("Focus", &continuity.focus.content),
        String::new(),
        "Conversation context:".to_string(),
    ];
    for item in &snapshot.context_items {
        match item.item_type {
            lcm::ContextItemType::Message => {
                if let Some(message_id) = item.message_id {
                    if let Some(message) = snapshot.messages.iter().find(|entry| entry.message_id == message_id) {
                        lines.push(format!("{}: {}", message.role, message.content));
                    }
                }
            }
            lcm::ContextItemType::Summary => {
                if let Some(summary_id) = item.summary_id.as_deref() {
                    if let Some(summary) = snapshot.summaries.iter().find(|entry| entry.summary_id == summary_id) {
                        lines.push(format!("summary: {}", summary.content));
                    }
                }
            }
        }
    }
    lines.push(String::new());
    lines.push(format!("Latest user turn: {latest_user_prompt}"));
    lines.join("\n")
}

fn continuity_block(label: &str, content: &str) -> String {
    format!("## {label}\n{}", content.trim_end())
}

fn refresh_continuity_documents(
    root: &Path,
    settings: &BTreeMap<String, String>,
    engine: &lcm::LcmEngine,
) -> Result<()> {
    for kind in [
        lcm::ContinuityKind::Narrative,
        lcm::ContinuityKind::Anchors,
        lcm::ContinuityKind::Focus,
    ] {
        let payload = engine.continuity_build_prompt(CHAT_CONVERSATION_ID, kind)?;
        let diff = invoke_codex_exec(root, settings, &payload.prompt)?;
        if !diff.trim().is_empty() {
            let _ = engine.continuity_apply_diff(CHAT_CONVERSATION_ID, kind, diff.trim())?;
        }
    }
    Ok(())
}

fn invoke_codex_exec(root: &Path, settings: &BTreeMap<String, String>, prompt: &str) -> Result<String> {
    let dependencies = execution_baseline::discover_vendored_dependency_paths(root);
    let model = settings
        .get("CTOX_CHAT_MODEL")
        .cloned()
        .unwrap_or_else(|| DEFAULT_ACTIVE_MODEL.to_string());
    channels::sync_prompt_identity(root, settings)?;
    let rendered_system_prompt = render_chat_system_prompt(root, settings)?;
    let base_instructions_override = toml_multiline_override("base_instructions", &rendered_system_prompt);
    let mut args = vec![
        "-m".to_string(),
        model.clone(),
        "--skip-git-repo-check".to_string(),
        "-c".to_string(),
        base_instructions_override,
        "-c".to_string(),
        "include_apply_patch_tool=true".to_string(),
        "-c".to_string(),
        "web_search=\"disabled\"".to_string(),
        prompt.to_string(),
    ];

    let use_local_provider = model.eq_ignore_ascii_case("openai/gpt-oss-20b");
    if use_local_provider {
        let proxy_host = settings
            .get("CTOX_PROXY_HOST")
            .cloned()
            .unwrap_or_else(|| "127.0.0.1".to_string());
        let proxy_port = settings
            .get("CTOX_PROXY_PORT")
            .cloned()
            .unwrap_or_else(|| "12434".to_string());
        let provider_config = format!(
            "model_providers.cto_local={{name=\"cto-local\",base_url=\"http://{proxy_host}:{proxy_port}/v1\",wire_api=\"responses\",requires_openai_auth=false}}"
        );
        args.splice(
            4..4,
            [
                "-c".to_string(),
                "model_provider=\"cto_local\"".to_string(),
                "-c".to_string(),
                provider_config,
            ],
        );
    }

    let mut command = if dependencies.codex_exec_binary.exists() {
        let mut cmd = Command::new(&dependencies.codex_exec_binary);
        cmd.current_dir(root);
        cmd
    } else {
        let mut cmd = Command::new("cargo");
        cmd.current_dir(&dependencies.codex_rs_root);
        cmd.args([
            "run",
            "--quiet",
            "--release",
            "-p",
            "codex-exec",
            "--bin",
            "codex-exec",
            "--",
        ]);
        cmd
    };
    command.args(&args);
    for (key, value) in settings {
        command.env(key, value);
    }
    command.env("CTOX_ROOT", root);
    command.env("CTOX_CONTEXT_DB", root.join("runtime/ctox_lcm.db"));
    let output = command
        .output()
        .with_context(|| "failed to launch codex-exec for TUI chat".to_string())?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
        let message = if !stderr.is_empty() { stderr } else { stdout };
        anyhow::bail!("{message}");
    }
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
    let response = if !stdout.is_empty() { stdout } else { stderr };
    if response.is_empty() {
        anyhow::bail!("codex-exec returned empty output");
    }
    Ok(response)
}

fn toml_multiline_override(key: &str, value: &str) -> String {
    let escaped = value.replace("\"\"\"", "\\\"\\\"\\\"");
    format!("{key} = \"\"\"\n{escaped}\n\"\"\"")
}

fn render_chat_system_prompt(root: &Path, settings: &BTreeMap<String, String>) -> Result<String> {
    let owner = channels::load_prompt_identity(root, settings)?;
    let channels_block = owner.channels.join("\n");
    let preferred_channel = owner
        .preferred_channel
        .unwrap_or_else(|| "not set".to_string());
    Ok(CTOX_CHAT_SYSTEM_PROMPT
        .replace("{{OWNER_NAME}}", &owner.owner_name)
        .replace("{{OWNER_CHANNELS}}", &channels_block)
        .replace("{{OWNER_PREFERRED_CHANNEL}}", &preferred_channel))
}

fn current_context_tokens(db_path: &Path, max_context: usize) -> Result<usize> {
    let engine = lcm::LcmEngine::open(db_path, lcm::LcmConfig::default())?;
    let decision = engine.evaluate_compaction(CHAT_CONVERSATION_ID, max_context as i64)?;
    Ok(decision.current_tokens.max(0) as usize)
}

fn load_proxy_telemetry(root: &Path) -> Result<Option<ProxyTelemetry>> {
    let host = runtime_config::env_or_config(root, "CTOX_PROXY_HOST").unwrap_or_else(|| "127.0.0.1".to_string());
    let port = runtime_config::env_or_config(root, "CTOX_PROXY_PORT").unwrap_or_else(|| "12434".to_string());
    let url = format!("http://{host}:{port}/ctox/telemetry");
    let response = match ureq::get(&url).call() {
        Ok(response) => response,
        Err(_) => return Ok(None),
    };
    let text = response
        .into_string()
        .context("failed to read proxy telemetry response")?;
    let parsed = serde_json::from_str(&text).context("failed to parse proxy telemetry json")?;
    Ok(Some(parsed))
}

fn read_usize_setting(settings: &BTreeMap<String, String>, key: &str, default: usize) -> usize {
    settings
        .get(key)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(default)
}

fn mask_secret(value: &str) -> String {
    if value.chars().count() <= 4 {
        return "*".repeat(value.chars().count());
    }
    let tail: String = value.chars().rev().take(4).collect::<Vec<_>>().into_iter().rev().collect();
    format!("{}{}", "*".repeat(value.chars().count().saturating_sub(4)), tail)
}

struct TerminalGuard;

impl TerminalGuard {
    fn enter(stdout: &mut io::Stdout) -> Result<Self> {
        terminal::enable_raw_mode().context("failed to enable raw mode")?;
        execute!(stdout, EnterAlternateScreen)?;
        Ok(Self)
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        let mut stdout = io::stdout();
        let _ = terminal::disable_raw_mode();
        let _ = execute!(stdout, LeaveAlternateScreen);
    }
}
