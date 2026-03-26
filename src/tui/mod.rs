use anyhow::Context;
use anyhow::Result;
use crossterm::cursor;
use crossterm::event;
use crossterm::event::Event as TerminalEvent;
use crossterm::event::KeyCode;
use crossterm::event::KeyEvent;
use crossterm::event::KeyModifiers;
use crossterm::execute;
use crossterm::terminal;
use crossterm::terminal::ClearType;
use crossterm::terminal::EnterAlternateScreen;
use crossterm::terminal::LeaveAlternateScreen;
use qrcode::types::Color as QrColor;
use qrcode::QrCode;
use ratatui::backend::CrosstermBackend;
use ratatui::Terminal;
use serde::Deserialize;
use serde::Serialize;
use std::collections::BTreeMap;
use std::collections::HashMap;
use std::collections::HashSet;
use std::collections::VecDeque;
use std::io;
use std::path::Path;
use std::path::PathBuf;
use std::process::Command;
use std::time::Duration;
use std::time::Instant;

use crate::backend_manager;
use crate::channels;
use crate::chat_runtime;
use crate::execution_baseline;
use crate::lcm;
use crate::responses_proxy::ProxyTelemetry;
use crate::runtime_config;
use crate::runtime_planner;
use crate::service;

mod render;

const DEFAULT_ACTIVE_MODEL: &str = "openai/gpt-oss-20b";
const DEFAULT_CHAT_SOURCE: &str = "local";
const DEFAULT_API_PROVIDER: &str = "openai";
const DEFAULT_CHAT_PRESET: &str = "Quality";
const DEFAULT_COMMUNICATION_PATH: &str = "tui";
const CHAT_PRESET_CHOICES: &[&str] = &["Quality", "Max Context", "Performance"];
const API_PROVIDER_CHOICES: &[&str] = &["openai"];
const COMMUNICATION_PATH_CHOICES: &[&str] = &["tui", "email", "jami"];
const EMAIL_PROVIDER_CHOICES: &[&str] = &["imap", "graph", "ews"];
const EMAIL_EWS_AUTH_CHOICES: &[&str] = &["basic", "oauth2"];

fn supported_local_chat_model_choices() -> Vec<&'static str> {
    execution_baseline::SUPPORTED_CHAT_MODELS
        .iter()
        .copied()
        .filter(|model| !execution_baseline::is_openai_api_chat_model(model))
        .collect()
}

fn has_openai_api_key(env_map: &BTreeMap<String, String>) -> bool {
    env_map
        .get("OPENAI_API_KEY")
        .map(|value| value.trim().starts_with("sk-"))
        .unwrap_or(false)
}

fn infer_chat_source(env_map: &BTreeMap<String, String>) -> String {
    env_map
        .get("CTOX_CHAT_SOURCE")
        .cloned()
        .or_else(|| {
            runtime_config::configured_chat_model_from_map(env_map)
                .filter(|value| execution_baseline::is_openai_api_chat_model(value))
                .map(|_| "api".to_string())
        })
        .unwrap_or_else(|| DEFAULT_CHAT_SOURCE.to_string())
}

fn infer_api_provider(env_map: &BTreeMap<String, String>) -> String {
    env_map
        .get("CTOX_API_PROVIDER")
        .cloned()
        .or_else(|| {
            runtime_config::configured_chat_model_from_map(env_map)
                .filter(|value| execution_baseline::is_openai_api_chat_model(value))
                .map(|_| "openai".to_string())
        })
        .unwrap_or_else(|| DEFAULT_API_PROVIDER.to_string())
}

fn supported_chat_model_choices(env_map: &BTreeMap<String, String>) -> Vec<&'static str> {
    if infer_chat_source(env_map).eq_ignore_ascii_case("api") {
        if infer_api_provider(env_map).eq_ignore_ascii_case("openai") && has_openai_api_key(env_map)
        {
            return execution_baseline::SUPPORTED_OPENAI_API_CHAT_MODELS.to_vec();
        }
        return Vec::new();
    }
    supported_local_chat_model_choices()
}

fn supported_embedding_model_choices() -> Vec<&'static str> {
    execution_baseline::SUPPORTED_EMBEDDING_MODELS.to_vec()
}

fn supported_stt_model_choices() -> Vec<&'static str> {
    execution_baseline::SUPPORTED_STT_MODELS.to_vec()
}

fn supported_tts_model_choices() -> Vec<&'static str> {
    execution_baseline::SUPPORTED_TTS_MODELS.to_vec()
}
const DEFAULT_MAX_CONTEXT: usize = 131_072;
const DEFAULT_COMPACTION_THRESHOLD_PERCENT: usize = 75;
const DEFAULT_COMPACTION_MIN_TOKENS: usize = 12_288;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Page {
    Chat,
    Skills,
    Settings,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SettingsView {
    General,
    Model,
}

#[derive(Debug, Clone)]
struct SettingItem {
    key: &'static str,
    label: &'static str,
    value: String,
    saved_value: String,
    secret: bool,
    choices: Vec<&'static str>,
    help: &'static str,
    kind: SettingKind,
}

#[derive(Debug, Clone)]
struct HeaderState {
    model: String,
    base_model: String,
    boost_model: Option<String>,
    boost_active: bool,
    boost_remaining_seconds: Option<u64>,
    boost_reason: Option<String>,
    max_context: usize,
    realized_context: usize,
    configured_context: usize,
    compact_at: usize,
    compact_percent: usize,
    compact_min_tokens: usize,
    current_tokens: usize,
    tokens_per_second: Option<f64>,
    avg_tokens_per_second: Option<f64>,
    last_input_tokens: Option<u64>,
    last_output_tokens: Option<u64>,
    last_total_tokens: Option<u64>,
    gpu_cards: Vec<GpuCardState>,
    estimate_mode: bool,
    chat_plan: Option<runtime_planner::ChatRuntimePlan>,
}

impl Default for HeaderState {
    fn default() -> Self {
        Self {
            model: DEFAULT_ACTIVE_MODEL.to_string(),
            base_model: DEFAULT_ACTIVE_MODEL.to_string(),
            boost_model: None,
            boost_active: false,
            boost_remaining_seconds: None,
            boost_reason: None,
            max_context: DEFAULT_MAX_CONTEXT,
            realized_context: DEFAULT_MAX_CONTEXT,
            configured_context: DEFAULT_MAX_CONTEXT,
            compact_at: DEFAULT_MAX_CONTEXT * DEFAULT_COMPACTION_THRESHOLD_PERCENT / 100,
            compact_percent: DEFAULT_COMPACTION_THRESHOLD_PERCENT,
            compact_min_tokens: DEFAULT_COMPACTION_MIN_TOKENS,
            current_tokens: 0,
            tokens_per_second: None,
            avg_tokens_per_second: None,
            last_input_tokens: None,
            last_output_tokens: None,
            last_total_tokens: None,
            gpu_cards: Vec::new(),
            estimate_mode: false,
            chat_plan: None,
        }
    }
}

#[derive(Debug, Clone, Default)]
struct GpuModelUsage {
    model: String,
    short_label: String,
    used_mb: u64,
}

#[derive(Debug, Clone, Default)]
struct GpuCardState {
    index: usize,
    name: String,
    used_mb: u64,
    total_mb: u64,
    utilization: u64,
    allocations: Vec<GpuModelUsage>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct ModelPerfStats {
    samples: u64,
    avg_tokens_per_second: f64,
    last_tokens_per_second: Option<f64>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct JamiResolvedEnvelope {
    #[serde(default)]
    ok: bool,
    #[serde(rename = "resolvedAccount")]
    resolved_account: Option<JamiResolvedAccount>,
    #[serde(default)]
    error: Option<String>,
    #[serde(rename = "dbusEnvFile", default)]
    dbus_env_file: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct JamiResolvedAccount {
    #[serde(rename = "accountId")]
    account_id: String,
    #[serde(rename = "accountType")]
    account_type: String,
    username: String,
    #[serde(rename = "shareUri")]
    share_uri: String,
    #[serde(rename = "displayName")]
    display_name: String,
    #[serde(default)]
    provisioned: bool,
}

#[derive(Debug, Clone, Default)]
struct JamiResolveOutcome {
    account: Option<JamiResolvedAccount>,
    error: Option<String>,
    dbus_env_file: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SettingKind {
    Env,
    ServiceToggle,
}

struct App {
    root: PathBuf,
    db_path: PathBuf,
    page: Page,
    chat_input: String,
    chat_messages: Vec<lcm::MessageRecord>,
    draft_queue: VecDeque<String>,
    activity_log: Vec<String>,
    communication_feed: Vec<channels::CommunicationFeedItem>,
    status_line: String,
    spinner_phase: usize,
    header: HeaderState,
    settings_items: Vec<SettingItem>,
    settings_selected: usize,
    settings_view: SettingsView,
    settings_menu_open: bool,
    settings_menu_index: usize,
    jami_qr_lines: Vec<String>,
    last_jami_qr_key: String,
    jami_runtime_account: Option<JamiResolvedAccount>,
    settings_dirty: bool,
    service_status: service::ServiceStatus,
    request_in_flight: bool,
    model_perf_stats: BTreeMap<String, ModelPerfStats>,
    last_recorded_response_at: Option<String>,
    gpu_cards: Vec<GpuCardState>,
    last_gpu_refresh_at: Option<Instant>,
    chat_preset_bundle: Option<runtime_planner::ChatPresetBundle>,
    skill_catalog: Vec<SkillCatalogEntry>,
    skills_selected: usize,
}

#[derive(Debug, Clone, Default)]
struct SkillCatalogEntry {
    name: String,
    source: String,
    skill_path: PathBuf,
    description: String,
    helper_tools: Vec<String>,
    resources: Vec<String>,
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
        let service_status =
            service::service_status_snapshot(&root).unwrap_or_else(|_| service::ServiceStatus {
                running: false,
                busy: false,
                pid: None,
                listen_addr: "127.0.0.1:12435".to_string(),
                autostart_enabled: false,
                manager: "process".to_string(),
                pending_count: 0,
                pending_previews: Vec::new(),
                active_source_label: None,
                recent_events: Vec::new(),
                last_error: None,
                last_completed_at: None,
                last_reply_chars: None,
            });
        Self {
            root: root.clone(),
            db_path,
            page: Page::Chat,
            chat_input: String::new(),
            chat_messages: Vec::new(),
            draft_queue: VecDeque::new(),
            activity_log: Vec::new(),
            communication_feed: Vec::new(),
            status_line: "Tab chat/skills/settings · Ctrl-C quit · Enter open/save".to_string(),
            spinner_phase: 0,
            header: HeaderState::default(),
            settings_items: load_settings_items(&root),
            settings_selected: 0,
            settings_view: SettingsView::Model,
            settings_menu_open: false,
            settings_menu_index: 0,
            jami_qr_lines: Vec::new(),
            last_jami_qr_key: String::new(),
            jami_runtime_account: None,
            settings_dirty: false,
            service_status,
            request_in_flight: false,
            model_perf_stats: load_model_perf_stats(&root),
            last_recorded_response_at: None,
            gpu_cards: Vec::new(),
            last_gpu_refresh_at: None,
            chat_preset_bundle: None,
            skill_catalog: load_skill_catalog(&root),
            skills_selected: 0,
        }
    }

    fn handle_paste(&mut self, text: &str) {
        match self.page {
            Page::Chat => self.chat_input.push_str(text),
            Page::Skills => {}
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
                self.settings_menu_open = false;
                self.page = match self.page {
                    Page::Chat => Page::Skills,
                    Page::Skills => Page::Settings,
                    Page::Settings => Page::Chat,
                };
                return Ok(false);
            }
            _ => {}
        }

        match self.page {
            Page::Chat => self.handle_chat_key(key_event)?,
            Page::Skills => self.handle_skills_key(key_event),
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
        if self.settings_menu_open {
            match key_event.code {
                KeyCode::Up => self.move_settings_menu(-1),
                KeyCode::Down => self.move_settings_menu(1),
                KeyCode::Enter => self.commit_settings_menu_choice()?,
                KeyCode::Esc | KeyCode::Left => self.settings_menu_open = false,
                _ => {}
            }
            return Ok(());
        }
        match key_event.code {
            KeyCode::Up => {
                self.move_settings_selection(-1);
            }
            KeyCode::Down => {
                self.move_settings_selection(1);
            }
            KeyCode::Char('[') => self.switch_settings_view(SettingsView::General),
            KeyCode::Char(']') => self.switch_settings_view(SettingsView::Model),
            KeyCode::Left => self.cycle_setting(false)?,
            KeyCode::Right => self.cycle_setting(true)?,
            KeyCode::Backspace => {
                if let Some(item) = self.current_setting_mut() {
                    if item.kind == SettingKind::Env {
                        item.value.pop();
                        self.settings_dirty = true;
                    }
                }
            }
            KeyCode::Enter => self.activate_selected_setting()?,
            KeyCode::Char(ch) if !key_event.modifiers.contains(KeyModifiers::CONTROL) => {
                if let Some(item) = self.current_setting_mut() {
                    if item.kind == SettingKind::Env && item.choices.is_empty() {
                        item.value.push(ch);
                        self.settings_dirty = true;
                    }
                }
            }
            _ => {}
        }
        Ok(())
    }

    fn handle_skills_key(&mut self, key_event: KeyEvent) {
        match key_event.code {
            KeyCode::Up => self.move_skills_selection(-1),
            KeyCode::Down => self.move_skills_selection(1),
            KeyCode::Char('r') | KeyCode::Char('R') => {
                self.skill_catalog = load_skill_catalog(&self.root);
                if self.skills_selected >= self.skill_catalog.len() {
                    self.skills_selected = self.skill_catalog.len().saturating_sub(1);
                }
                self.status_line = format!("Reloaded {} skill entries.", self.skill_catalog.len());
            }
            _ => {}
        }
    }

    fn cycle_setting(&mut self, forward: bool) -> Result<()> {
        let Some(item) = self.current_setting_mut() else {
            return Ok(());
        };
        if item.kind != SettingKind::Env || item.choices.is_empty() {
            return Ok(());
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
        Ok(())
    }

    fn activate_selected_setting(&mut self) -> Result<()> {
        match self.current_setting() {
            Some(item) if item.kind == SettingKind::Env && self.setting_is_dirty(item) => {
                self.save_settings()
            }
            Some(item) if item.kind == SettingKind::Env => {
                if item.choices.is_empty() {
                    Ok(())
                } else {
                    self.settings_menu_index = item
                        .choices
                        .iter()
                        .position(|choice| choice.eq_ignore_ascii_case(item.value.trim()))
                        .unwrap_or(0);
                    self.settings_menu_open = true;
                    Ok(())
                }
            }
            Some(item) if item.kind == SettingKind::ServiceToggle => self.toggle_service(),
            Some(_) => Ok(()),
            None => Ok(()),
        }
    }

    fn toggle_service(&mut self) -> Result<()> {
        self.status_line = if self.service_status.running {
            service::stop_background(&self.root)?
        } else {
            service::start_background(&self.root)?
        };
        self.push_local_activity(self.status_line.clone());
        self.refresh()?;
        Ok(())
    }

    fn submit_chat_request(&mut self) -> Result<()> {
        let prompt = self.chat_input.trim().to_string();
        if prompt.is_empty() {
            self.status_line = "Chat input is empty.".to_string();
            return Ok(());
        }
        if !self.service_status.running {
            self.status_line =
                "CTOX loop is not running. Start it in Settings or with `ctox start`.".to_string();
            return Ok(());
        }
        if self.request_in_flight {
            self.draft_queue.push_back(prompt.clone());
            self.chat_input.clear();
            self.status_line = format!(
                "Prompt queued locally. {} draft(s) waiting.",
                self.draft_queue.len()
            );
            self.push_local_activity(format!(
                "Queued local draft: {}",
                summarize_inline(&prompt, 72)
            ));
            return Ok(());
        }
        self.chat_input.clear();
        service::submit_chat_prompt(&self.root, &prompt)?;
        self.status_line = "CTOX loop accepted the request.".to_string();
        self.push_local_activity(format!(
            "Submitted prompt: {}",
            summarize_inline(&prompt, 72)
        ));
        self.request_in_flight = true;
        Ok(())
    }

    fn poll_worker(&mut self) -> Result<()> {
        self.refresh()
    }

    fn refresh_service_status(&mut self) {
        let previous = self.service_status.clone();
        self.service_status = service::service_status_snapshot(&self.root).unwrap_or_else(|_| {
            service::ServiceStatus {
                running: false,
                busy: false,
                pid: None,
                listen_addr: "127.0.0.1:12435".to_string(),
                autostart_enabled: false,
                manager: "process".to_string(),
                pending_count: 0,
                pending_previews: Vec::new(),
                active_source_label: None,
                recent_events: Vec::new(),
                last_error: None,
                last_completed_at: None,
                last_reply_chars: None,
            }
        });
        self.request_in_flight = self.service_status.running && self.service_status.busy;
        if previous.busy && !self.service_status.busy {
            self.status_line = match self.service_status.last_error.as_deref() {
                Some(err) => format!("CTOX loop failed: {err}"),
                None => format!(
                    "CTOX loop completed reply{}.",
                    self.service_status
                        .last_reply_chars
                        .map(|count| format!(" with {count} chars"))
                        .unwrap_or_default()
                ),
            };
            if self.chat_input.trim().is_empty() {
                if let Some(next) = self.draft_queue.pop_front() {
                    self.chat_input = next;
                    self.push_local_activity("Moved next queued draft into composer".to_string());
                    self.status_line = format!("{} Draft ready in composer.", self.status_line);
                }
            }
        } else if !previous.running && self.service_status.running {
            self.status_line = format!(
                "CTOX loop connected at {}.",
                self.service_status.listen_addr
            );
        } else if previous.running && !self.service_status.running {
            self.status_line = "CTOX loop stopped.".to_string();
        }
        self.sync_activity_log();
    }

    fn service_summary(&self) -> String {
        let persist = if self.service_status.autostart_enabled {
            "autostart on"
        } else {
            "autostart off"
        };
        if self.service_status.running {
            if self.service_status.busy {
                format!(
                    "running on {} (busy, {}, {})",
                    self.service_status.listen_addr, self.service_status.manager, persist
                )
            } else {
                format!(
                    "running on {} (idle, {}, {})",
                    self.service_status.listen_addr, self.service_status.manager, persist
                )
            }
        } else {
            format!(
                "stopped ({}, {}, {})",
                self.service_status.listen_addr, self.service_status.manager, persist
            )
        }
    }

    fn rendered_setting_value(&self, item: &SettingItem) -> String {
        match item.kind {
            SettingKind::Env => {
                let mut rendered = if item.secret && !item.value.trim().is_empty() {
                    mask_secret(&item.value)
                } else if item.value.trim().is_empty() {
                    "(empty)".to_string()
                } else {
                    item.value.clone()
                };
                if self.setting_is_dirty(item) {
                    rendered.push_str(" *");
                }
                rendered
            }
            SettingKind::ServiceToggle => self.service_summary(),
        }
    }

    fn selected_setting_help(&self) -> String {
        self.current_setting()
            .map(|item| match item.kind {
                SettingKind::Env => {
                    let mut lines = vec![item.help.to_string()];
                    if self.setting_is_dirty(item) {
                        lines.push("Pending change. Enter saves it.".to_string());
                    } else if !item.choices.is_empty() {
                        lines.push("Enter opens the choice menu.".to_string());
                    }
                    lines.join("\n")
                }
                SettingKind::ServiceToggle => {
                    let action = if self.service_status.running {
                        "stop"
                    } else {
                        "start"
                    };
                    format!(
                        "loop {}\n{}\nEnter to {}\nCtrl-C quits CTOX.",
                        if self.service_status.running {
                            "up"
                        } else {
                            "down"
                        },
                        self.service_summary(),
                        action
                    )
                }
            })
            .unwrap_or_else(|| "No setting selected.".to_string())
    }

    fn settings_env_map(&self) -> BTreeMap<String, String> {
        settings_map_from_items(&self.settings_items)
    }

    fn saved_settings_env_map(&self) -> BTreeMap<String, String> {
        settings_map_from_items(
            &self
                .settings_items
                .iter()
                .map(|item| SettingItem {
                    key: item.key,
                    label: item.label,
                    value: item.saved_value.clone(),
                    saved_value: item.saved_value.clone(),
                    secret: item.secret,
                    choices: item.choices.clone(),
                    help: item.help,
                    kind: item.kind,
                })
                .collect::<Vec<_>>(),
        )
    }

    fn visible_setting_indices(&self) -> Vec<usize> {
        self.settings_items
            .iter()
            .enumerate()
            .filter_map(|(idx, item)| {
                (self.setting_visible(item) && self.setting_in_view(item)).then_some(idx)
            })
            .collect()
    }

    fn setting_in_view(&self, item: &SettingItem) -> bool {
        let _ = item;
        true
    }

    fn setting_visible(&self, item: &SettingItem) -> bool {
        if item.kind == SettingKind::ServiceToggle {
            return true;
        }
        let api_provider = self
            .value_for_setting("CTOX_API_PROVIDER")
            .unwrap_or(DEFAULT_API_PROVIDER);
        let chat_source = self
            .value_for_setting("CTOX_CHAT_SOURCE")
            .unwrap_or(DEFAULT_CHAT_SOURCE);
        match item.key {
            "CTOX_CHAT_SOURCE"
            | "CTOX_CHAT_MODEL"
            | "CTOX_CHAT_MODEL_BOOST"
            | "CTOX_BOOST_DEFAULT_MINUTES"
            | "CTOX_EMBEDDING_MODEL"
            | "CTOX_STT_MODEL"
            | "CTOX_TTS_MODEL"
            | "CTOX_CHAT_COMPACTION_THRESHOLD_PERCENT"
            | "CTOX_OWNER_NAME"
            | "CTOX_OWNER_EMAIL_ADDRESS"
            | "CTOX_ALLOWED_EMAIL_DOMAIN"
            | "CTOX_EMAIL_ADMIN_POLICIES"
            | "CTOX_OWNER_PREFERRED_CHANNEL" => true,
            "CTOX_API_PROVIDER" => chat_source.eq_ignore_ascii_case("api"),
            "OPENAI_API_KEY" => {
                chat_source.eq_ignore_ascii_case("api")
                    && api_provider.eq_ignore_ascii_case("openai")
            }
            "CTOX_CHAT_LOCAL_PRESET" => chat_source.eq_ignore_ascii_case("local"),
            "CTO_EMAIL_ADDRESS" | "CTO_EMAIL_PASSWORD" | "CTO_EMAIL_PROVIDER" => self
                .value_for_setting("CTOX_OWNER_PREFERRED_CHANNEL")
                .unwrap_or(DEFAULT_COMMUNICATION_PATH)
                .eq_ignore_ascii_case("email"),
            "CTO_EMAIL_IMAP_HOST"
            | "CTO_EMAIL_IMAP_PORT"
            | "CTO_EMAIL_SMTP_HOST"
            | "CTO_EMAIL_SMTP_PORT" => {
                self.value_for_setting("CTOX_OWNER_PREFERRED_CHANNEL")
                    .unwrap_or(DEFAULT_COMMUNICATION_PATH)
                    .eq_ignore_ascii_case("email")
                    && self
                        .value_for_setting("CTO_EMAIL_PROVIDER")
                        .unwrap_or("imap")
                        .eq_ignore_ascii_case("imap")
            }
            "CTO_EMAIL_GRAPH_USER" => {
                self.value_for_setting("CTOX_OWNER_PREFERRED_CHANNEL")
                    .unwrap_or(DEFAULT_COMMUNICATION_PATH)
                    .eq_ignore_ascii_case("email")
                    && self
                        .value_for_setting("CTO_EMAIL_PROVIDER")
                        .unwrap_or("imap")
                        .eq_ignore_ascii_case("graph")
            }
            "CTO_EMAIL_EWS_URL" | "CTO_EMAIL_EWS_AUTH_TYPE" | "CTO_EMAIL_EWS_USERNAME" => {
                self.value_for_setting("CTOX_OWNER_PREFERRED_CHANNEL")
                    .unwrap_or(DEFAULT_COMMUNICATION_PATH)
                    .eq_ignore_ascii_case("email")
                    && self
                        .value_for_setting("CTO_EMAIL_PROVIDER")
                        .unwrap_or("imap")
                        .eq_ignore_ascii_case("ews")
            }
            "CTO_JAMI_ACCOUNT_ID" | "CTO_JAMI_PROFILE_NAME" => self
                .value_for_setting("CTOX_OWNER_PREFERRED_CHANNEL")
                .unwrap_or(DEFAULT_COMMUNICATION_PATH)
                .eq_ignore_ascii_case("jami"),
            _ => false,
        }
    }

    fn value_for_setting(&self, key: &str) -> Option<&str> {
        self.settings_items
            .iter()
            .find(|item| item.key == key)
            .map(|item| item.value.trim())
            .filter(|value| !value.is_empty())
    }

    fn current_setting(&self) -> Option<&SettingItem> {
        self.settings_items.get(self.settings_selected)
    }

    fn current_setting_mut(&mut self) -> Option<&mut SettingItem> {
        self.settings_items.get_mut(self.settings_selected)
    }

    fn setting_is_dirty(&self, item: &SettingItem) -> bool {
        item.value.trim() != item.saved_value.trim()
    }

    fn move_settings_selection(&mut self, delta: isize) {
        let visible = self.visible_setting_indices();
        if visible.is_empty() {
            return;
        }
        let current_pos = visible
            .iter()
            .position(|idx| *idx == self.settings_selected)
            .unwrap_or(0);
        let next_pos = if delta.is_negative() {
            current_pos.saturating_sub(delta.unsigned_abs())
        } else {
            (current_pos + delta as usize).min(visible.len().saturating_sub(1))
        };
        self.settings_selected = visible[next_pos];
    }

    fn switch_settings_view(&mut self, view: SettingsView) {
        if self.settings_view == view {
            return;
        }
        self.settings_view = view;
        if let Some(first) = self.visible_setting_indices().first().copied() {
            self.settings_selected = first;
        }
    }

    fn refresh_dynamic_setting_choices(&mut self) {
        let env_map = self.settings_env_map();
        for item in &mut self.settings_items {
            match item.key {
                "CTOX_CHAT_MODEL" => item.choices = supported_chat_model_choices(&env_map),
                "CTOX_API_PROVIDER" => item.choices = API_PROVIDER_CHOICES.to_vec(),
                "CTOX_CHAT_LOCAL_PRESET" => item.choices = runtime_planner::chat_preset_choices(),
                _ => {}
            }
        }
    }

    fn refresh(&mut self) -> Result<()> {
        self.refresh_dynamic_setting_choices();
        self.refresh_service_status();
        self.refresh_chat_messages()?;
        self.refresh_communication_feed();
        self.refresh_skill_catalog();
        self.refresh_gpu_cards();
        self.refresh_header();
        self.refresh_jami_qr();
        Ok(())
    }

    fn refresh_skill_catalog(&mut self) {
        let refreshed = load_skill_catalog(&self.root);
        if refreshed.len() != self.skill_catalog.len()
            || refreshed
                .iter()
                .zip(self.skill_catalog.iter())
                .any(|(left, right)| {
                    left.skill_path != right.skill_path
                        || left.description != right.description
                        || left.helper_tools != right.helper_tools
                        || left.resources != right.resources
                })
        {
            self.skill_catalog = refreshed;
            if self.skills_selected >= self.skill_catalog.len() {
                self.skills_selected = self.skill_catalog.len().saturating_sub(1);
            }
        }
    }

    fn move_skills_selection(&mut self, delta: isize) {
        if self.skill_catalog.is_empty() {
            self.skills_selected = 0;
            return;
        }
        let next = if delta.is_negative() {
            self.skills_selected.saturating_sub(delta.unsigned_abs())
        } else {
            (self.skills_selected + delta as usize).min(self.skill_catalog.len().saturating_sub(1))
        };
        self.skills_selected = next;
    }

    fn refresh_gpu_cards(&mut self) {
        let should_refresh = self
            .last_gpu_refresh_at
            .map(|at| at.elapsed() >= Duration::from_secs(2))
            .unwrap_or(true);
        if !should_refresh {
            return;
        }
        if let Ok(cards) = sample_gpu_cards() {
            self.gpu_cards = cards;
            self.last_gpu_refresh_at = Some(Instant::now());
        }
    }

    fn refresh_chat_messages(&mut self) -> Result<()> {
        let snapshot = lcm::run_dump(&self.db_path, chat_runtime::CHAT_CONVERSATION_ID)?;
        self.chat_messages = snapshot.messages;
        Ok(())
    }

    fn refresh_communication_feed(&mut self) {
        self.communication_feed =
            channels::load_recent_communication_feed(&self.root, 10).unwrap_or_default();
    }

    fn refresh_header(&mut self) {
        let mut saved_settings = self.saved_settings_env_map();
        let mut draft_settings = self.settings_env_map();
        normalize_runtime_model_settings(&mut saved_settings);
        normalize_runtime_model_settings(&mut draft_settings);
        let estimate_mode = self.has_draft_runtime_estimate();
        let settings = if estimate_mode {
            &draft_settings
        } else {
            &saved_settings
        };
        let proxy_telemetry = load_proxy_telemetry(&self.root).ok().flatten();
        let model = proxy_telemetry
            .as_ref()
            .and_then(|telemetry| telemetry.active_model.clone())
            .or_else(|| runtime_config::effective_chat_model_from_map(settings))
            .unwrap_or_else(|| DEFAULT_ACTIVE_MODEL.to_string());
        let loaded_model = proxy_telemetry
            .as_ref()
            .and_then(|telemetry| telemetry.active_model.clone())
            .or_else(|| runtime_config::effective_chat_model_from_map(&saved_settings))
            .unwrap_or_else(|| DEFAULT_ACTIVE_MODEL.to_string());
        let base_model = proxy_telemetry
            .as_ref()
            .and_then(|telemetry| telemetry.base_model.clone())
            .or_else(|| runtime_config::configured_chat_model_from_map(&saved_settings))
            .unwrap_or_else(|| loaded_model.clone());

        let selected_bundle = if infer_chat_source(settings).eq_ignore_ascii_case("local") {
            runtime_planner::preview_chat_preset_bundle(&self.root, settings)
                .ok()
                .flatten()
        } else {
            None
        };
        let saved_bundle = if infer_chat_source(&saved_settings).eq_ignore_ascii_case("local") {
            runtime_planner::preview_chat_preset_bundle(&self.root, &saved_settings)
                .ok()
                .flatten()
        } else {
            None
        };
        self.chat_preset_bundle = selected_bundle.clone();

        let realized_context_from_runtime = saved_settings
            .get("CTOX_VLLM_SERVE_REALIZED_MAX_SEQ_LEN")
            .or_else(|| saved_settings.get("CTOX_CHAT_MODEL_REALIZED_CONTEXT"))
            .and_then(|value| value.trim().parse::<usize>().ok());

        let planned_context = selected_bundle
            .as_ref()
            .map(|bundle| bundle.selected_plan.max_seq_len as usize);
        let saved_context = saved_bundle
            .as_ref()
            .map(|bundle| bundle.selected_plan.max_seq_len as usize);
        let configured_context = planned_context
            .or(saved_context)
            .or_else(|| {
                execution_baseline::runtime_config_for_model(&model)
                    .ok()
                    .and_then(|runtime| runtime.max_seq_len.map(|value| value as usize))
            })
            .unwrap_or(DEFAULT_MAX_CONTEXT)
            .min(DEFAULT_MAX_CONTEXT);
        let realized_context = if estimate_mode {
            planned_context.unwrap_or(configured_context)
        } else {
            realized_context_from_runtime
                .or(saved_context)
                .unwrap_or(configured_context)
        }
        .min(DEFAULT_MAX_CONTEXT);
        let gpu_cards =
            if let Some(plan) = selected_bundle.as_ref().map(|bundle| &bundle.selected_plan) {
                gpu_cards_from_plan(plan, settings)
            } else {
                let expected_live_models = configured_runtime_models(&saved_settings);
                if estimate_mode {
                    estimate_gpu_cards(
                        &self.gpu_cards,
                        &loaded_model,
                        &model,
                        settings,
                        realized_context,
                    )
                } else {
                    filter_gpu_cards_to_models(&self.gpu_cards, &expected_live_models)
                }
            };
        let effective_context = configured_context
            .min(realized_context)
            .min(DEFAULT_MAX_CONTEXT);
        let planned_compact_percent = selected_bundle
            .as_ref()
            .map(|bundle| bundle.selected_plan.compaction_threshold_percent as usize);
        let saved_compact_percent = saved_bundle
            .as_ref()
            .map(|bundle| bundle.selected_plan.compaction_threshold_percent as usize);
        let compact_percent = if estimate_mode {
            planned_compact_percent
                .or(saved_compact_percent)
                .unwrap_or_else(|| {
                    read_usize_setting(
                        settings,
                        "CTOX_CHAT_COMPACTION_THRESHOLD_PERCENT",
                        DEFAULT_COMPACTION_THRESHOLD_PERCENT,
                    )
                })
        } else {
            saved_compact_percent
                .or(planned_compact_percent)
                .unwrap_or_else(|| {
                    read_usize_setting(
                        &saved_settings,
                        "CTOX_CHAT_COMPACTION_THRESHOLD_PERCENT",
                        DEFAULT_COMPACTION_THRESHOLD_PERCENT,
                    )
                })
        }
        .clamp(1, 99);
        let planned_compact_min = selected_bundle
            .as_ref()
            .map(|bundle| bundle.selected_plan.compaction_min_tokens as usize);
        let saved_compact_min = saved_bundle
            .as_ref()
            .map(|bundle| bundle.selected_plan.compaction_min_tokens as usize);
        let compact_min_tokens = if estimate_mode {
            planned_compact_min
                .or(saved_compact_min)
                .unwrap_or_else(|| {
                    read_usize_setting(
                        settings,
                        "CTOX_CHAT_COMPACTION_MIN_TOKENS",
                        DEFAULT_COMPACTION_MIN_TOKENS,
                    )
                })
        } else {
            saved_compact_min
                .or(planned_compact_min)
                .unwrap_or_else(|| {
                    read_usize_setting(
                        &saved_settings,
                        "CTOX_CHAT_COMPACTION_MIN_TOKENS",
                        DEFAULT_COMPACTION_MIN_TOKENS,
                    )
                })
        };
        let compact_at =
            compute_compaction_threshold(effective_context, compact_percent, compact_min_tokens);
        let current_tokens = current_context_tokens(&self.db_path, effective_context).unwrap_or(0);
        self.record_proxy_model_sample(proxy_telemetry.as_ref());
        let avg_tokens_per_second = if estimate_mode {
            selected_bundle
                .as_ref()
                .map(|bundle| bundle.selected_plan.expected_tok_s)
                .or_else(|| estimated_tokens_per_second(&model, &self.model_perf_stats))
        } else {
            self.model_perf_stats
                .get(model.trim())
                .map(|stats| stats.avg_tokens_per_second)
                .or_else(|| {
                    saved_bundle
                        .as_ref()
                        .map(|bundle| bundle.selected_plan.expected_tok_s)
                })
        };
        self.header = HeaderState {
            model,
            base_model,
            boost_model: proxy_telemetry
                .as_ref()
                .and_then(|value| value.boost_model.clone()),
            boost_active: proxy_telemetry
                .as_ref()
                .map(|value| value.boost_active)
                .unwrap_or(false),
            boost_remaining_seconds: proxy_telemetry
                .as_ref()
                .and_then(|value| value.boost_remaining_seconds),
            boost_reason: proxy_telemetry
                .as_ref()
                .and_then(|value| value.boost_reason.clone()),
            max_context: effective_context,
            realized_context,
            configured_context,
            compact_at,
            compact_percent,
            compact_min_tokens,
            current_tokens,
            tokens_per_second: proxy_telemetry
                .as_ref()
                .and_then(|value| value.last_tokens_per_second),
            avg_tokens_per_second,
            last_input_tokens: proxy_telemetry
                .as_ref()
                .and_then(|value| value.last_input_tokens),
            last_output_tokens: proxy_telemetry
                .as_ref()
                .and_then(|value| value.last_output_tokens),
            last_total_tokens: proxy_telemetry
                .as_ref()
                .and_then(|value| value.last_total_tokens),
            gpu_cards,
            estimate_mode,
            chat_plan: selected_bundle.map(|bundle| bundle.selected_plan),
        };
    }

    fn has_draft_runtime_estimate(&self) -> bool {
        self.settings_items.iter().any(|item| {
            self.setting_is_dirty(item)
                && matches!(item.kind, SettingKind::Env)
                && relevant_header_estimate_setting(item.key)
        })
    }

    fn record_proxy_model_sample(&mut self, telemetry: Option<&ProxyTelemetry>) {
        let Some(telemetry) = telemetry else {
            return;
        };
        let Some(response_at) = telemetry.last_response_at.as_deref() else {
            return;
        };
        if self.last_recorded_response_at.as_deref() == Some(response_at) {
            return;
        }
        let Some(model) = telemetry
            .active_model
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
        else {
            return;
        };
        let Some(tps) = telemetry.last_tokens_per_second else {
            self.last_recorded_response_at = Some(response_at.to_string());
            return;
        };
        let stats = self.model_perf_stats.entry(model.to_string()).or_default();
        let next_samples = stats.samples + 1;
        let weighted_sum = stats.avg_tokens_per_second * stats.samples as f64 + tps;
        stats.samples = next_samples;
        stats.avg_tokens_per_second = weighted_sum / next_samples as f64;
        stats.last_tokens_per_second = Some(tps);
        self.last_recorded_response_at = Some(response_at.to_string());
        let _ = save_model_perf_stats(&self.root, &self.model_perf_stats);
    }

    fn save_settings(&mut self) -> Result<()> {
        let previous_env = runtime_config::load_runtime_env_map(&self.root).unwrap_or_default();
        let mut env_map = previous_env.clone();
        for item in &self.settings_items {
            if item.kind != SettingKind::Env {
                continue;
            }
            let trimmed = item.value.trim();
            if trimmed.is_empty() {
                env_map.remove(item.key);
            } else {
                env_map.insert(item.key.to_string(), trimmed.to_string());
            }
        }
        if let Some(base_model) = env_map
            .get("CTOX_CHAT_MODEL")
            .cloned()
            .filter(|value| !value.trim().is_empty())
        {
            env_map.insert("CTOX_CHAT_MODEL_BASE".to_string(), base_model);
        } else {
            env_map.remove("CTOX_CHAT_MODEL_BASE");
        }
        normalize_runtime_model_settings(&mut env_map);
        let _ = runtime_planner::apply_chat_runtime_plan(&self.root, &mut env_map);
        runtime_config::save_runtime_env_map(&self.root, &env_map)?;
        channels::sync_prompt_identity(&self.root, &env_map)?;
        let previous_model = previous_env
            .get("CTOX_CHAT_MODEL")
            .cloned()
            .filter(|value| !value.trim().is_empty());
        let next_model = env_map
            .get("CTOX_CHAT_MODEL")
            .cloned()
            .filter(|value| !value.trim().is_empty());
        let next_source = infer_chat_source(&env_map);
        let next_preset = env_map.get("CTOX_CHAT_LOCAL_PRESET").map(String::as_str);
        let switch_status = if next_source.eq_ignore_ascii_case("local") {
            if let Some(next) = next_model.as_deref() {
                if local_runtime_apply_changed(&previous_env, &env_map) {
                    Some(switch_proxy_model(&self.root, next, next_preset))
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            match (&previous_model, &next_model) {
                (Some(previous), Some(next)) if previous != next => {
                    Some(switch_proxy_model(&self.root, next, None))
                }
                (None, Some(next)) => Some(switch_proxy_model(&self.root, next, None)),
                _ => None,
            }
        };
        self.settings_dirty = false;
        for item in &mut self.settings_items {
            item.saved_value = item.value.clone();
        }
        self.refresh_dynamic_setting_choices();
        if self.service_status.running {
            let _ = backend_manager::ensure_persistent_backends(&self.root);
        }
        self.status_line = match switch_status {
            Some(Ok(message)) => message,
            Some(Err(err)) => format!("Settings saved, but proxy switch failed: {err}"),
            None => "Settings saved to runtime/vllm_serve.env.".to_string(),
        };
        self.refresh_header();
        self.refresh_jami_qr();
        Ok(())
    }

    fn header_lines(&self, width: usize) -> Vec<String> {
        let model = compact_model_name(&self.header.model, width);
        let loop_state = if self.service_status.running {
            if self.service_status.busy {
                "busy"
            } else {
                "idle"
            }
        } else {
            "down"
        };
        let queue_state = if self.request_in_flight {
            if self.draft_queue.is_empty() && self.service_status.pending_count == 0 {
                "coil on".to_string()
            } else {
                format!(
                    "coil on q{}",
                    self.draft_queue.len() + self.service_status.pending_count
                )
            }
        } else if self.draft_queue.is_empty() {
            "coil off".to_string()
        } else {
            format!("coil off q{}", self.draft_queue.len())
        };
        let model_line = format!(
            "loaded {}   loop {}   {}   tok/s {}",
            model,
            loop_state,
            queue_state,
            self.header
                .tokens_per_second
                .map(|value| format!("{value:.1}"))
                .unwrap_or_else(|| "-".to_string()),
        );
        let usage_line = format!(
            "used {} / {}   compact {}   live {}   io {}/{}/{}",
            self.header.current_tokens,
            self.header.max_context,
            self.header.compact_at,
            self.header.realized_context,
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
                .unwrap_or_else(|| "-".to_string()),
        );
        let preview_line = truncate_for_ui(&self.header_preview_line(), width);
        let status_line = truncate_for_ui(&self.status_line, width);
        let bar_width = width.saturating_sub(18).max(12);
        let marker_index = if self.header.max_context == 0 {
            0
        } else {
            ((self.header.compact_at.min(self.header.max_context) as f64
                / self.header.max_context as f64)
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
        let realized_index = if self.header.max_context == 0 {
            0
        } else {
            ((self.header.realized_context.min(self.header.max_context) as f64
                / self.header.max_context as f64)
                * bar_width as f64)
                .floor() as usize
        };
        let mut bar = String::with_capacity(bar_width + 2);
        bar.push('▕');
        for idx in 0..bar_width {
            if idx == marker_index.min(bar_width.saturating_sub(1)) {
                bar.push('◆');
            } else if idx == realized_index.min(bar_width.saturating_sub(1)) {
                bar.push('╎');
            } else if idx < fill_index {
                bar.push('█');
            } else {
                bar.push('░');
            }
        }
        bar.push('▏');
        vec![
            truncate_for_ui(&model_line, width),
            truncate_for_ui(&usage_line, width),
            preview_line,
            truncate_for_ui(&format!("0 {bar} {}", self.header.max_context), width),
            status_line,
        ]
    }

    fn header_preview_line(&self) -> String {
        let model_item = self
            .settings_items
            .iter()
            .find(|item| item.key == "CTOX_CHAT_MODEL");
        let preset_item = self
            .settings_items
            .iter()
            .find(|item| item.key == "CTOX_CHAT_LOCAL_PRESET");
        let channel_item = self
            .settings_items
            .iter()
            .find(|item| item.key == "CTOX_OWNER_PREFERRED_CHANNEL");

        if let Some(bundle) = &self.chat_preset_bundle {
            if model_item
                .map(|item| self.setting_is_dirty(item))
                .unwrap_or(false)
                || preset_item
                    .map(|item| self.setting_is_dirty(item))
                    .unwrap_or(false)
            {
                let plan = &bundle.selected_plan;
                return format!(
                    "draft {}  {}  ctx {}  compact {}% min {}k  {} tok/s",
                    compact_model_name(&plan.model, 20),
                    plan.quantization,
                    plan.max_seq_len,
                    plan.compaction_threshold_percent,
                    plan.compaction_min_tokens / 1024,
                    plan.expected_tok_s.round() as i64
                );
            }
        }
        if let Some(channel_item) = channel_item {
            if self.setting_is_dirty(channel_item) {
                return format!("draft channel {}", channel_item.value.trim());
            }
        }
        "loaded state".to_string()
    }

    fn move_settings_menu(&mut self, delta: isize) {
        let Some(item) = self.current_setting() else {
            self.settings_menu_open = false;
            return;
        };
        if item.choices.is_empty() {
            self.settings_menu_open = false;
            return;
        }
        let len = item.choices.len() as isize;
        let next = (self.settings_menu_index as isize + delta).rem_euclid(len);
        self.settings_menu_index = next as usize;
    }

    fn commit_settings_menu_choice(&mut self) -> Result<()> {
        let selected_index = self.settings_menu_index;
        let Some(item) = self.current_setting_mut() else {
            self.settings_menu_open = false;
            return Ok(());
        };
        if item.choices.is_empty() {
            self.settings_menu_open = false;
            return Ok(());
        }
        let next = item
            .choices
            .get(selected_index)
            .copied()
            .unwrap_or(item.choices[0]);
        item.value = next.to_string();
        self.settings_dirty = true;
        self.settings_menu_open = false;
        self.refresh_jami_qr();
        Ok(())
    }

    fn refresh_jami_qr(&mut self) {
        let channel = self
            .value_for_setting("CTOX_OWNER_PREFERRED_CHANNEL")
            .unwrap_or("tui");
        if channel != "jami" {
            self.jami_qr_lines.clear();
            self.last_jami_qr_key.clear();
            self.jami_runtime_account = None;
            return;
        }
        let configured_jami_id = self
            .value_for_setting("CTO_JAMI_ACCOUNT_ID")
            .unwrap_or("")
            .trim()
            .to_string();
        let configured_profile_name = self
            .value_for_setting("CTO_JAMI_PROFILE_NAME")
            .unwrap_or("")
            .trim()
            .to_string();
        let resolved =
            resolve_jami_runtime_account(&self.root, &configured_jami_id, &configured_profile_name);
        let qr_payload = resolved
            .account
            .as_ref()
            .and_then(|account| {
                if !account.share_uri.trim().is_empty() {
                    Some(account.share_uri.trim().to_string())
                } else if !account.username.trim().is_empty() {
                    Some(format!("jami:{}", account.username.trim()))
                } else {
                    None
                }
            })
            .or_else(|| (!configured_jami_id.is_empty()).then(|| configured_jami_id.clone()))
            .unwrap_or_default();
        let qr_key = format!("{channel}:{qr_payload}");
        if !qr_payload.is_empty()
            && self.last_jami_qr_key == qr_key
            && !self.jami_qr_lines.is_empty()
        {
            self.jami_runtime_account = resolved.account;
            return;
        }
        self.jami_runtime_account = resolved.account.clone();
        if let Some(error) = resolved.error.as_deref() {
            self.jami_qr_lines = jami_error_lines(
                error,
                resolved.dbus_env_file.as_deref(),
                !configured_jami_id.is_empty() || !configured_profile_name.is_empty(),
            );
            self.last_jami_qr_key = qr_key;
            return;
        }
        if qr_payload.is_empty() {
            self.jami_qr_lines = jami_missing_account_lines(
                resolved.dbus_env_file.as_deref(),
                !configured_jami_id.is_empty() || !configured_profile_name.is_empty(),
            );
            self.last_jami_qr_key = qr_key;
            return;
        }
        self.jami_qr_lines = render_qr_lines(&qr_payload).unwrap_or_else(|| {
            vec![
                "Failed to render Jami QR.".to_string(),
                format!("uri {}", truncate_for_ui(&qr_payload, 40)),
            ]
        });
        self.last_jami_qr_key = qr_key;
    }

    fn push_local_activity(&mut self, event: String) {
        self.activity_log.push(event);
        if self.activity_log.len() > 32 {
            let overflow = self.activity_log.len() - 32;
            self.activity_log.drain(0..overflow);
        }
    }

    fn sync_activity_log(&mut self) {
        let mut merged = self.service_status.recent_events.clone();
        merged.extend(self.activity_log.iter().cloned());
        if merged.len() > 32 {
            merged = merged.split_off(merged.len() - 32);
        }
        self.activity_log = merged;
    }
}

fn load_skill_catalog(root: &Path) -> Vec<SkillCatalogEntry> {
    let mut catalog = Vec::new();
    let mut seen = HashSet::new();
    for base in skill_roots(root) {
        collect_skill_entries(&base, &mut seen, &mut catalog);
    }
    catalog.sort_by(|left, right| {
        left.name
            .cmp(&right.name)
            .then(left.source.cmp(&right.source))
            .then(left.skill_path.cmp(&right.skill_path))
    });
    catalog
}

fn skill_roots(root: &Path) -> Vec<PathBuf> {
    let mut roots = vec![root.join("skills")];
    if let Ok(codex_home) = std::env::var("CODEX_HOME") {
        roots.push(PathBuf::from(codex_home).join("skills"));
    } else if let Ok(home) = std::env::var("HOME") {
        roots.push(PathBuf::from(home).join(".codex/skills"));
    }
    roots
}

fn collect_skill_entries(
    base: &Path,
    seen: &mut HashSet<PathBuf>,
    catalog: &mut Vec<SkillCatalogEntry>,
) {
    if !base.exists() {
        return;
    }
    let Ok(base_canon) = std::fs::canonicalize(base) else {
        return;
    };
    let mut queue = vec![base_canon];
    while let Some(dir) = queue.pop() {
        let skill_md = dir.join("SKILL.md");
        if skill_md.is_file() {
            if let Ok(canon_skill) = std::fs::canonicalize(&skill_md) {
                if seen.insert(canon_skill) {
                    if let Some(entry) = parse_skill_catalog_entry(base, &dir, &skill_md) {
                        catalog.push(entry);
                    }
                }
            }
            continue;
        }
        let Ok(read_dir) = std::fs::read_dir(&dir) else {
            continue;
        };
        for child in read_dir.flatten() {
            let child_path = child.path();
            if child.file_type().map(|file_type| file_type.is_dir()).unwrap_or(false) {
                queue.push(child_path);
            }
        }
    }
}

fn parse_skill_catalog_entry(
    base: &Path,
    dir: &Path,
    skill_md: &Path,
) -> Option<SkillCatalogEntry> {
    let body = std::fs::read_to_string(skill_md).ok()?;
    Some(SkillCatalogEntry {
        name: dir.file_name()?.to_string_lossy().to_string(),
        source: classify_skill_source(base, skill_md),
        skill_path: skill_md.to_path_buf(),
        description: parse_skill_description(&body),
        helper_tools: list_named_children(&dir.join("scripts")),
        resources: collect_resource_summaries(dir),
    })
}

fn parse_skill_description(body: &str) -> String {
    let mut in_code_fence = false;
    for line in body.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("```") {
            in_code_fence = !in_code_fence;
            continue;
        }
        if in_code_fence
            || trimmed.is_empty()
            || trimmed.starts_with('#')
            || trimmed.starts_with('-')
            || trimmed.starts_with('*')
            || trimmed.starts_with('<')
        {
            continue;
        }
        return trimmed.to_string();
    }
    "No inline summary available.".to_string()
}

fn collect_resource_summaries(dir: &Path) -> Vec<String> {
    let mut resources = Vec::new();
    push_resource_summary(&mut resources, "references", &dir.join("references"));
    push_resource_summary(&mut resources, "assets", &dir.join("assets"));
    push_resource_summary(&mut resources, "templates", &dir.join("templates"));
    push_resource_summary(&mut resources, "agents", &dir.join("agents"));
    resources
}

fn push_resource_summary(out: &mut Vec<String>, label: &str, dir: &Path) {
    let entries = list_named_children(dir);
    if entries.is_empty() {
        return;
    }
    let preview = entries.iter().take(5).cloned().collect::<Vec<_>>().join(", ");
    let suffix = if entries.len() > 5 {
        format!(" (+{} more)", entries.len() - 5)
    } else {
        String::new()
    };
    out.push(format!("{label}: {preview}{suffix}"));
}

fn list_named_children(dir: &Path) -> Vec<String> {
    let Ok(read_dir) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut entries = read_dir
        .flatten()
        .filter_map(|entry| {
            let path = entry.path();
            let file_type = entry.file_type().ok()?;
            if !(file_type.is_file() || file_type.is_dir()) {
                return None;
            }
            Some(path.file_name()?.to_string_lossy().to_string())
        })
        .collect::<Vec<_>>();
    entries.sort();
    entries
}

fn classify_skill_source(base: &Path, skill_md: &Path) -> String {
    let base_text = base.to_string_lossy();
    let path_text = skill_md.to_string_lossy();
    if base_text.contains(".codex/skills") {
        if path_text.contains("/.system/") {
            "codex system".to_string()
        } else {
            "codex custom".to_string()
        }
    } else if path_text.contains("/skills/.system/") {
        "ctox system".to_string()
    } else {
        "repo custom".to_string()
    }
}

fn switch_proxy_model(root: &Path, model: &str, preset: Option<&str>) -> Result<String> {
    let mut env_map = runtime_config::load_runtime_env_map(root).unwrap_or_default();
    env_map.remove("CTOX_BOOST_ACTIVE_UNTIL_EPOCH");
    env_map.remove("CTOX_BOOST_REASON");
    runtime_config::save_runtime_env_map(root, &env_map)?;
    if !execution_baseline::uses_ctox_proxy_model(model) {
        return Ok("Settings saved to runtime/vllm_serve.env.".to_string());
    }
    let host = runtime_config::env_or_config(root, "CTOX_PROXY_HOST")
        .unwrap_or_else(|| "127.0.0.1".to_string());
    let port = runtime_config::env_or_config(root, "CTOX_PROXY_PORT")
        .unwrap_or_else(|| "12434".to_string());
    let url = format!("http://{host}:{port}/ctox/switch");
    let payload = match preset.map(str::trim).filter(|value| !value.is_empty()) {
        Some(preset) => format!(r#"{{"model":"{}","preset":"{}"}}"#, model, preset),
        None => format!(r#"{{"model":"{}"}}"#, model),
    };
    let response = ureq::post(&url)
        .set("content-type", "application/json")
        .send_string(&payload)
        .with_context(|| format!("failed to reach proxy switch endpoint at {url}"))?;
    let body = response
        .into_string()
        .context("failed to read proxy switch response body")?;
    Ok(format!(
        "Settings saved and proxy switched: {}",
        body.trim()
    ))
}

fn load_settings_items(root: &Path) -> Vec<SettingItem> {
    let env_map = runtime_config::load_runtime_env_map(root).unwrap_or_default();
    let inferred_chat_source = infer_chat_source(&env_map);
    let inferred_api_provider = infer_api_provider(&env_map);
    let mut choices_env_map = env_map.clone();
    choices_env_map.insert("CTOX_CHAT_SOURCE".to_string(), inferred_chat_source.clone());
    choices_env_map.insert(
        "CTOX_API_PROVIDER".to_string(),
        inferred_api_provider.clone(),
    );
    let is_api_source = inferred_chat_source.eq_ignore_ascii_case("api");
    let chat_model_choices = supported_chat_model_choices(&choices_env_map);
    let active_model = if is_api_source {
        runtime_config::configured_chat_model_from_map(&env_map)
            .filter(|value| execution_baseline::is_openai_api_chat_model(value))
            .or_else(|| chat_model_choices.first().map(|value| (*value).to_string()))
            .unwrap_or_default()
    } else {
        runtime_config::configured_chat_model_from_map(&env_map)
            .unwrap_or_else(|| DEFAULT_ACTIVE_MODEL.to_string())
    };
    let boost_model = env_map
        .get("CTOX_CHAT_MODEL_BOOST")
        .cloned()
        .unwrap_or_default();
    let boost_minutes = env_map
        .get("CTOX_BOOST_DEFAULT_MINUTES")
        .cloned()
        .unwrap_or_else(|| "20".to_string());
    let compaction_threshold = env_map
        .get("CTOX_CHAT_COMPACTION_THRESHOLD_PERCENT")
        .cloned()
        .unwrap_or_else(|| DEFAULT_COMPACTION_THRESHOLD_PERCENT.to_string());
    vec![
        SettingItem {
            key: "CTOX_SERVICE_TOGGLE",
            label: "CTOX Loop",
            value: String::new(),
            saved_value: String::new(),
            secret: false,
            choices: Vec::new(),
            help: "Start or stop the CTOX background loop from within the TUI.",
            kind: SettingKind::ServiceToggle,
        },
        SettingItem {
            key: "CTOX_CHAT_SOURCE",
            label: "Chat Source",
            value: inferred_chat_source.clone(),
            saved_value: inferred_chat_source,
            secret: false,
            choices: vec!["local", "api"],
            help: "Choose whether the chat model runs locally or through an API provider.",
            kind: SettingKind::Env,
        },
        SettingItem {
            key: "CTOX_API_PROVIDER",
            label: "API Provider",
            value: inferred_api_provider.clone(),
            saved_value: inferred_api_provider,
            secret: false,
            choices: API_PROVIDER_CHOICES.to_vec(),
            help: "API provider for remote chat models.",
            kind: SettingKind::Env,
        },
        SettingItem {
            key: "CTOX_CHAT_MODEL",
            label: if is_api_source { "API Model" } else { "Local Model" },
            value: active_model,
            saved_value: runtime_config::configured_chat_model_from_map(&env_map)
                .filter(|value| {
                    if is_api_source {
                        execution_baseline::is_openai_api_chat_model(value)
                    } else {
                        true
                    }
                })
                .unwrap_or_else(|| {
                    if is_api_source {
                        String::new()
                    } else {
                        DEFAULT_ACTIVE_MODEL.to_string()
                    }
                }),
            secret: false,
            choices: chat_model_choices,
            help: "Selected chat model for the chosen chat source.",
            kind: SettingKind::Env,
        },
        SettingItem {
            key: "CTOX_CHAT_MODEL_BOOST",
            label: "Boost Model",
            value: boost_model.clone(),
            saved_value: boost_model,
            secret: false,
            choices: supported_chat_model_choices(&choices_env_map),
            help: "Optional stronger model for temporary boost leases. CTOX can request this model when it is genuinely stuck and then fall back automatically after the lease expires.",
            kind: SettingKind::Env,
        },
        SettingItem {
            key: "CTOX_BOOST_DEFAULT_MINUTES",
            label: "Boost TTL",
            value: boost_minutes.clone(),
            saved_value: boost_minutes,
            secret: false,
            choices: vec!["10", "15", "20", "30", "45", "60"],
            help: "Default lifetime in minutes for an automatic boost lease before the proxy falls back to the base model.",
            kind: SettingKind::Env,
        },
        SettingItem {
            key: "OPENAI_API_KEY",
            label: "API Token",
            value: env_map.get("OPENAI_API_KEY").cloned().unwrap_or_default(),
            saved_value: env_map.get("OPENAI_API_KEY").cloned().unwrap_or_default(),
            secret: true,
            choices: Vec::new(),
            help: "API token for the selected remote provider.",
            kind: SettingKind::Env,
        },
        SettingItem {
            key: "CTOX_CHAT_LOCAL_PRESET",
            label: "Chat Preset",
            value: runtime_planner::ChatPreset::from_label(
                env_map
                    .get("CTOX_CHAT_LOCAL_PRESET")
                    .map(String::as_str)
                    .unwrap_or(DEFAULT_CHAT_PRESET),
            )
            .label()
            .to_string(),
            saved_value: runtime_planner::ChatPreset::from_label(
                env_map
                    .get("CTOX_CHAT_LOCAL_PRESET")
                    .map(String::as_str)
                    .unwrap_or(DEFAULT_CHAT_PRESET),
            )
            .label()
            .to_string(),
            secret: false,
            choices: CHAT_PRESET_CHOICES.to_vec(),
            help: "Preview the planned local runtime. Enter once selects, Enter again saves and applies it through the proxy.",
            kind: SettingKind::Env,
        },
        SettingItem {
            key: "CTOX_EMBEDDING_MODEL",
            label: "Embed Model",
            value: execution_baseline::auxiliary_model_selection(
                execution_baseline::AuxiliaryRole::Embedding,
                env_map.get("CTOX_EMBEDDING_MODEL").map(String::as_str),
            )
            .choice
            .to_string(),
            saved_value: execution_baseline::auxiliary_model_selection(
                execution_baseline::AuxiliaryRole::Embedding,
                env_map.get("CTOX_EMBEDDING_MODEL").map(String::as_str),
            )
            .choice
            .to_string(),
            secret: false,
            choices: supported_embedding_model_choices(),
            help: "Persistent embedding sidecar. GPU keeps the current fast path; CPU keeps embeddings available on hosts without CUDA.",
            kind: SettingKind::Env,
        },
        SettingItem {
            key: "CTOX_STT_MODEL",
            label: "STT Model",
            value: execution_baseline::auxiliary_model_selection(
                execution_baseline::AuxiliaryRole::Stt,
                env_map.get("CTOX_STT_MODEL").map(String::as_str),
            )
            .choice
            .to_string(),
            saved_value: execution_baseline::auxiliary_model_selection(
                execution_baseline::AuxiliaryRole::Stt,
                env_map.get("CTOX_STT_MODEL").map(String::as_str),
            )
            .choice
            .to_string(),
            secret: false,
            choices: supported_stt_model_choices(),
            help: "Speech-to-text sidecar. GPU keeps Voxtral; CPU switches to a faster-whisper path for installations without a GPU.",
            kind: SettingKind::Env,
        },
        SettingItem {
            key: "CTOX_TTS_MODEL",
            label: "TTS Model",
            value: execution_baseline::auxiliary_model_selection(
                execution_baseline::AuxiliaryRole::Tts,
                env_map.get("CTOX_TTS_MODEL").map(String::as_str),
            )
            .choice
            .to_string(),
            saved_value: execution_baseline::auxiliary_model_selection(
                execution_baseline::AuxiliaryRole::Tts,
                env_map.get("CTOX_TTS_MODEL").map(String::as_str),
            )
            .choice
            .to_string(),
            secret: false,
            choices: supported_tts_model_choices(),
            help: "Text-to-speech sidecar. GPU keeps the native Qwen path; CPU standardizes on Piper over Speaches, with language-specific voices such as German, French, and English.",
            kind: SettingKind::Env,
        },
        SettingItem {
            key: "CTOX_CHAT_COMPACTION_THRESHOLD_PERCENT",
            label: "Compact %",
            value: compaction_threshold,
            saved_value: env_map
                .get("CTOX_CHAT_COMPACTION_THRESHOLD_PERCENT")
                .cloned()
                .unwrap_or_else(|| DEFAULT_COMPACTION_THRESHOLD_PERCENT.to_string()),
            secret: false,
            choices: vec!["60", "70", "75", "80", "85"],
            help: "Compaction trigger percentage for the active context window. Save to apply.",
            kind: SettingKind::Env,
        },
        SettingItem {
            key: "CTOX_OWNER_NAME",
            label: "Owner Name",
            value: env_map
                .get("CTOX_OWNER_NAME")
                .cloned()
                .unwrap_or_else(|| "Michael Welsch".to_string()),
            saved_value: env_map
                .get("CTOX_OWNER_NAME")
                .cloned()
                .unwrap_or_else(|| "Michael Welsch".to_string()),
            secret: false,
            choices: Vec::new(),
            help: "Owner name used in prompts and communication identity.",
            kind: SettingKind::Env,
        },
        SettingItem {
            key: "CTOX_OWNER_EMAIL_ADDRESS",
            label: "Owner E-Mail",
            value: env_map
                .get("CTOX_OWNER_EMAIL_ADDRESS")
                .cloned()
                .unwrap_or_default(),
            saved_value: env_map
                .get("CTOX_OWNER_EMAIL_ADDRESS")
                .cloned()
                .unwrap_or_default(),
            secret: false,
            choices: Vec::new(),
            help: "Owner mailbox for full administrative mail authority. Additional admins and general domain access are configured in the next fields.",
            kind: SettingKind::Env,
        },
        SettingItem {
            key: "CTOX_ALLOWED_EMAIL_DOMAIN",
            label: "Allowed Domain",
            value: env_map
                .get("CTOX_ALLOWED_EMAIL_DOMAIN")
                .cloned()
                .unwrap_or_default(),
            saved_value: env_map
                .get("CTOX_ALLOWED_EMAIL_DOMAIN")
                .cloned()
                .unwrap_or_default(),
            secret: false,
            choices: Vec::new(),
            help: "Any sender from this mail domain may contact CTOX for support and account help. Leave blank to derive it from the owner e-mail domain.",
            kind: SettingKind::Env,
        },
        SettingItem {
            key: "CTOX_EMAIL_ADMIN_POLICIES",
            label: "Mail Admins",
            value: env_map
                .get("CTOX_EMAIL_ADMIN_POLICIES")
                .cloned()
                .unwrap_or_default(),
            saved_value: env_map
                .get("CTOX_EMAIL_ADMIN_POLICIES")
                .cloned()
                .unwrap_or_default(),
            secret: false,
            choices: Vec::new(),
            help: "Comma/newline list like admin@domain:sudo, helpdesk@domain:nosudo. Owner keeps full rights; listed admins may do admin work by mail, with optional sudo.",
            kind: SettingKind::Env,
        },
        SettingItem {
            key: "CTOX_OWNER_PREFERRED_CHANNEL",
            label: "Communication",
            value: env_map
                .get("CTOX_OWNER_PREFERRED_CHANNEL")
                .cloned()
                .unwrap_or_else(|| DEFAULT_COMMUNICATION_PATH.to_string()),
            saved_value: env_map
                .get("CTOX_OWNER_PREFERRED_CHANNEL")
                .cloned()
                .unwrap_or_else(|| DEFAULT_COMMUNICATION_PATH.to_string()),
            secret: false,
            choices: COMMUNICATION_PATH_CHOICES.to_vec(),
            help: "Preferred communication path for CTOX replies.",
            kind: SettingKind::Env,
        },
        SettingItem {
            key: "CTO_EMAIL_ADDRESS",
            label: "E-Mail Address",
            value: env_map.get("CTO_EMAIL_ADDRESS").cloned().unwrap_or_default(),
            saved_value: env_map.get("CTO_EMAIL_ADDRESS").cloned().unwrap_or_default(),
            secret: false,
            choices: Vec::new(),
            help: "Mailbox address for e-mail communication.",
            kind: SettingKind::Env,
        },
        SettingItem {
            key: "CTO_EMAIL_PASSWORD",
            label: "E-Mail Password",
            value: env_map.get("CTO_EMAIL_PASSWORD").cloned().unwrap_or_default(),
            saved_value: env_map.get("CTO_EMAIL_PASSWORD").cloned().unwrap_or_default(),
            secret: true,
            choices: Vec::new(),
            help: "Mailbox password for IMAP/SMTP communication.",
            kind: SettingKind::Env,
        },
        SettingItem {
            key: "CTO_EMAIL_PROVIDER",
            label: "E-Mail Protocol",
            value: env_map
                .get("CTO_EMAIL_PROVIDER")
                .cloned()
                .unwrap_or_else(|| "imap".to_string()),
            saved_value: env_map
                .get("CTO_EMAIL_PROVIDER")
                .cloned()
                .unwrap_or_else(|| "imap".to_string()),
            secret: false,
            choices: EMAIL_PROVIDER_CHOICES.to_vec(),
            help: "Select the e-mail access protocol.",
            kind: SettingKind::Env,
        },
        SettingItem {
            key: "CTO_EMAIL_IMAP_HOST",
            label: "IMAP Host",
            value: env_map.get("CTO_EMAIL_IMAP_HOST").cloned().unwrap_or_default(),
            saved_value: env_map.get("CTO_EMAIL_IMAP_HOST").cloned().unwrap_or_default(),
            secret: false,
            choices: Vec::new(),
            help: "IMAP hostname for mailbox sync.",
            kind: SettingKind::Env,
        },
        SettingItem {
            key: "CTO_EMAIL_IMAP_PORT",
            label: "IMAP Port",
            value: env_map
                .get("CTO_EMAIL_IMAP_PORT")
                .cloned()
                .unwrap_or_else(|| "993".to_string()),
            saved_value: env_map
                .get("CTO_EMAIL_IMAP_PORT")
                .cloned()
                .unwrap_or_else(|| "993".to_string()),
            secret: false,
            choices: Vec::new(),
            help: "IMAP port.",
            kind: SettingKind::Env,
        },
        SettingItem {
            key: "CTO_EMAIL_SMTP_HOST",
            label: "SMTP Host",
            value: env_map.get("CTO_EMAIL_SMTP_HOST").cloned().unwrap_or_default(),
            saved_value: env_map.get("CTO_EMAIL_SMTP_HOST").cloned().unwrap_or_default(),
            secret: false,
            choices: Vec::new(),
            help: "SMTP hostname for outgoing mail.",
            kind: SettingKind::Env,
        },
        SettingItem {
            key: "CTO_EMAIL_SMTP_PORT",
            label: "SMTP Port",
            value: env_map
                .get("CTO_EMAIL_SMTP_PORT")
                .cloned()
                .unwrap_or_else(|| "587".to_string()),
            saved_value: env_map
                .get("CTO_EMAIL_SMTP_PORT")
                .cloned()
                .unwrap_or_else(|| "587".to_string()),
            secret: false,
            choices: Vec::new(),
            help: "SMTP port.",
            kind: SettingKind::Env,
        },
        SettingItem {
            key: "CTO_EMAIL_GRAPH_USER",
            label: "Graph User",
            value: env_map.get("CTO_EMAIL_GRAPH_USER").cloned().unwrap_or_default(),
            saved_value: env_map.get("CTO_EMAIL_GRAPH_USER").cloned().unwrap_or_default(),
            secret: false,
            choices: Vec::new(),
            help: "Microsoft Graph mailbox user or principal.",
            kind: SettingKind::Env,
        },
        SettingItem {
            key: "CTO_EMAIL_EWS_URL",
            label: "EWS URL",
            value: env_map.get("CTO_EMAIL_EWS_URL").cloned().unwrap_or_default(),
            saved_value: env_map.get("CTO_EMAIL_EWS_URL").cloned().unwrap_or_default(),
            secret: false,
            choices: Vec::new(),
            help: "Exchange Web Services endpoint.",
            kind: SettingKind::Env,
        },
        SettingItem {
            key: "CTO_EMAIL_EWS_AUTH_TYPE",
            label: "EWS Auth",
            value: env_map
                .get("CTO_EMAIL_EWS_AUTH_TYPE")
                .cloned()
                .unwrap_or_else(|| "basic".to_string()),
            saved_value: env_map
                .get("CTO_EMAIL_EWS_AUTH_TYPE")
                .cloned()
                .unwrap_or_else(|| "basic".to_string()),
            secret: false,
            choices: EMAIL_EWS_AUTH_CHOICES.to_vec(),
            help: "Authentication mode for EWS.",
            kind: SettingKind::Env,
        },
        SettingItem {
            key: "CTO_EMAIL_EWS_USERNAME",
            label: "EWS User",
            value: env_map
                .get("CTO_EMAIL_EWS_USERNAME")
                .cloned()
                .unwrap_or_default(),
            saved_value: env_map
                .get("CTO_EMAIL_EWS_USERNAME")
                .cloned()
                .unwrap_or_default(),
            secret: false,
            choices: Vec::new(),
            help: "Username for EWS authentication.",
            kind: SettingKind::Env,
        },
        SettingItem {
            key: "CTO_JAMI_PROFILE_NAME",
            label: "Jami Name",
            value: env_map
                .get("CTO_JAMI_PROFILE_NAME")
                .cloned()
                .unwrap_or_default(),
            saved_value: env_map
                .get("CTO_JAMI_PROFILE_NAME")
                .cloned()
                .unwrap_or_default(),
            secret: false,
            choices: Vec::new(),
            help: "Displayed Jami profile name.",
            kind: SettingKind::Env,
        },
        SettingItem {
            key: "CTO_JAMI_ACCOUNT_ID",
            label: "Jami Account",
            value: env_map.get("CTO_JAMI_ACCOUNT_ID").cloned().unwrap_or_default(),
            saved_value: env_map.get("CTO_JAMI_ACCOUNT_ID").cloned().unwrap_or_default(),
            secret: false,
            choices: Vec::new(),
            help: "Jami account id or share URI.",
            kind: SettingKind::Env,
        },
    ]
}

fn settings_map_from_items(items: &[SettingItem]) -> BTreeMap<String, String> {
    let mut env_map = BTreeMap::new();
    for item in items {
        if item.kind != SettingKind::Env {
            continue;
        }
        let trimmed = item.value.trim();
        if trimmed.is_empty() {
            continue;
        }
        env_map.insert(item.key.to_string(), trimmed.to_string());
    }
    env_map
}

fn current_context_tokens(db_path: &Path, max_context: usize) -> Result<usize> {
    let engine = lcm::LcmEngine::open(db_path, lcm::LcmConfig::default())?;
    let decision =
        engine.evaluate_compaction(chat_runtime::CHAT_CONVERSATION_ID, max_context as i64)?;
    Ok(decision.current_tokens.max(0) as usize)
}

fn load_proxy_telemetry(root: &Path) -> Result<Option<ProxyTelemetry>> {
    let host = runtime_config::env_or_config(root, "CTOX_PROXY_HOST")
        .unwrap_or_else(|| "127.0.0.1".to_string());
    let port = runtime_config::env_or_config(root, "CTOX_PROXY_PORT")
        .unwrap_or_else(|| "12434".to_string());
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

fn compute_compaction_threshold(
    max_context: usize,
    threshold_percent: usize,
    min_tokens: usize,
) -> usize {
    let percent_threshold = max_context.saturating_mul(threshold_percent.clamp(1, 99)) / 100;
    percent_threshold.max(min_tokens.min(max_context))
}

fn relevant_header_estimate_setting(key: &str) -> bool {
    matches!(
        key,
        "CTOX_CHAT_SOURCE"
            | "CTOX_API_PROVIDER"
            | "OPENAI_API_KEY"
            | "CTOX_CHAT_MODEL"
            | "CTOX_CHAT_LOCAL_PRESET"
            | "CTOX_CHAT_COMPACTION_THRESHOLD_PERCENT"
    )
}

fn local_runtime_apply_changed(
    previous_env: &BTreeMap<String, String>,
    next_env: &BTreeMap<String, String>,
) -> bool {
    [
        "CTOX_CHAT_SOURCE",
        "CTOX_CHAT_MODEL",
        "CTOX_CHAT_LOCAL_PRESET",
        "CTOX_CHAT_RUNTIME_PLAN_DIGEST",
        "CTOX_VLLM_SERVE_CUDA_VISIBLE_DEVICES",
        "CTOX_VLLM_SERVE_DEVICE_LAYERS",
        "CTOX_VLLM_SERVE_MAX_SEQ_LEN",
        "CTOX_VLLM_SERVE_DISABLE_NCCL",
        "CTOX_VLLM_SERVE_TENSOR_PARALLEL_BACKEND",
        "CTOX_VLLM_SERVE_MN_LOCAL_WORLD_SIZE",
        "CTOX_VLLM_SERVE_ISQ",
    ]
    .iter()
    .any(|key| previous_env.get(*key).map(String::as_str) != next_env.get(*key).map(String::as_str))
}

fn is_model_runtime_setting(key: &str) -> bool {
    matches!(
        key,
        "CTOX_CHAT_SOURCE"
            | "CTOX_API_PROVIDER"
            | "OPENAI_API_KEY"
            | "CTOX_CHAT_MODEL"
            | "CTOX_CHAT_LOCAL_PRESET"
            | "CTOX_EMBEDDING_MODEL"
            | "CTOX_STT_MODEL"
            | "CTOX_TTS_MODEL"
            | "CTOX_CHAT_COMPACTION_THRESHOLD_PERCENT"
    )
}

fn normalize_runtime_model_settings(env_map: &mut BTreeMap<String, String>) {
    let chat_source = infer_chat_source(env_map);
    if chat_source.eq_ignore_ascii_case("api") {
        env_map.insert("CTOX_API_PROVIDER".to_string(), infer_api_provider(env_map));
        env_map.remove("CTOX_CHAT_LOCAL_PRESET");
        env_map.remove("CTOX_VLLM_SERVE_MAX_SEQ_LEN");
        env_map.remove("CTOX_VLLM_SERVE_ISQ");
        env_map.remove("CTOX_VLLM_SERVE_DISABLE_NCCL");
        env_map.remove("CTOX_VLLM_SERVE_CUDA_VISIBLE_DEVICES");
        env_map.remove("CTOX_VLLM_SERVE_DEVICE_LAYERS");
        env_map.remove("CTOX_VLLM_SERVE_MAX_SEQS");
        runtime_planner::clear_chat_plan_env(env_map);
        return;
    }

    env_map.insert("CTOX_API_PROVIDER".to_string(), "local".to_string());
    env_map
        .entry("CTOX_CHAT_LOCAL_PRESET".to_string())
        .or_insert_with(|| DEFAULT_CHAT_PRESET.to_string());
}

fn sample_gpu_cards() -> Result<Vec<GpuCardState>> {
    let gpu_output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .context("failed to run nvidia-smi gpu query")?;
    if !gpu_output.status.success() {
        anyhow::bail!("nvidia-smi gpu query failed");
    }

    let mut cards = Vec::new();
    let mut uuid_to_index = HashMap::new();
    for line in String::from_utf8_lossy(&gpu_output.stdout).lines() {
        let parts = line.split(',').map(|part| part.trim()).collect::<Vec<_>>();
        if parts.len() < 6 {
            continue;
        }
        let index = parts[0].parse::<usize>().unwrap_or(0);
        uuid_to_index.insert(parts[1].to_string(), index);
        cards.push(GpuCardState {
            index,
            name: parts[2].to_string(),
            used_mb: parts[3].parse::<u64>().unwrap_or(0),
            total_mb: parts[4].parse::<u64>().unwrap_or(0),
            utilization: parts[5].parse::<u64>().unwrap_or(0),
            allocations: Vec::new(),
        });
    }

    let proc_output = Command::new("nvidia-smi")
        .args([
            "--query-compute-apps=gpu_uuid,pid,used_memory",
            "--format=csv,noheader,nounits",
        ])
        .output();
    let Ok(proc_output) = proc_output else {
        return Ok(cards);
    };
    if !proc_output.status.success() {
        return Ok(cards);
    }

    let mut pid_to_gpu = Vec::new();
    let mut pids = Vec::new();
    for line in String::from_utf8_lossy(&proc_output.stdout).lines() {
        let parts = line.split(',').map(|part| part.trim()).collect::<Vec<_>>();
        if parts.len() < 3 {
            continue;
        }
        let Some(&gpu_index) = uuid_to_index.get(parts[0]) else {
            continue;
        };
        let Ok(pid) = parts[1].parse::<u32>() else {
            continue;
        };
        let used_mb = parts[2].parse::<u64>().unwrap_or(0);
        pid_to_gpu.push((pid, gpu_index, used_mb));
        pids.push(pid.to_string());
    }
    if pids.is_empty() {
        return Ok(cards);
    }

    let ps_output = Command::new("ps")
        .args(["-o", "pid=,command=", "-p", &pids.join(",")])
        .output()
        .context("failed to run ps for gpu processes")?;
    let mut pid_to_command = HashMap::new();
    for line in String::from_utf8_lossy(&ps_output.stdout).lines() {
        let trimmed = line.trim_start();
        let mut split = trimmed.splitn(2, ' ');
        let Some(pid_raw) = split.next() else {
            continue;
        };
        let Some(command) = split.next() else {
            continue;
        };
        if let Ok(pid) = pid_raw.trim().parse::<u32>() {
            pid_to_command.insert(pid, command.trim().to_string());
        }
    }

    for (pid, gpu_index, used_mb) in pid_to_gpu {
        let Some(command) = pid_to_command.get(&pid) else {
            continue;
        };
        let Some(model) = model_name_from_process_command(command) else {
            continue;
        };
        let short_label = compact_model_name(&model, 18);
        if let Some(card) = cards.iter_mut().find(|card| card.index == gpu_index) {
            if let Some(existing) = card
                .allocations
                .iter_mut()
                .find(|allocation| allocation.model == model)
            {
                existing.used_mb = existing.used_mb.saturating_add(used_mb);
            } else {
                card.allocations.push(GpuModelUsage {
                    model,
                    short_label,
                    used_mb,
                });
            }
        }
    }

    cards.sort_by_key(|card| card.index);
    for card in &mut cards {
        card.allocations
            .sort_by(|left, right| right.used_mb.cmp(&left.used_mb));
    }
    Ok(cards)
}

fn model_name_from_process_command(command: &str) -> Option<String> {
    [
        "openai/gpt-oss-20b",
        "Qwen/Qwen3.5-4B",
        "Qwen/Qwen3.5-9B",
        "Qwen/Qwen3.5-27B",
        "Qwen/Qwen3.5-35B-A3B",
        "zai-org/GLM-4.7-Flash",
        "Qwen/Qwen3-Embedding-0.6B",
        "mistralai/Voxtral-Mini-4B-Realtime-2602",
        "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    ]
    .iter()
    .find(|candidate| command.contains(**candidate))
    .map(|candidate| (*candidate).to_string())
}

fn estimated_tokens_per_second(
    model: &str,
    perf_stats: &BTreeMap<String, ModelPerfStats>,
) -> Option<f64> {
    perf_stats
        .get(model.trim())
        .map(|stats| stats.avg_tokens_per_second)
        .or_else(|| {
            Some(match model.trim() {
                "openai/gpt-oss-20b" => 90.0,
                "Qwen/Qwen3.5-4B" => 140.0,
                "Qwen/Qwen3.5-9B" => 95.0,
                "Qwen/Qwen3.5-27B" => 45.0,
                "Qwen/Qwen3.5-35B-A3B" => 38.0,
                "zai-org/GLM-4.7-Flash" => 48.0,
                _ => return None,
            })
        })
}

fn gpu_cards_from_plan(
    plan: &runtime_planner::ChatRuntimePlan,
    env_map: &BTreeMap<String, String>,
) -> Vec<GpuCardState> {
    plan.gpu_allocations
        .iter()
        .map(|allocation| {
            let mut allocations = Vec::new();
            if allocation.aux_reserve_mb > 0 {
                allocations.extend(estimated_aux_model_usages(
                    allocation.aux_reserve_mb,
                    env_map,
                ));
            }
            if allocation.chat_budget_mb > 0 {
                allocations.push(GpuModelUsage {
                    model: plan.model.clone(),
                    short_label: compact_model_name(&plan.model, 18),
                    used_mb: allocation.chat_budget_mb,
                });
            }
            GpuCardState {
                index: allocation.gpu_index,
                name: allocation.name.clone(),
                used_mb: allocation
                    .desktop_reserve_mb
                    .saturating_add(allocation.aux_reserve_mb)
                    .saturating_add(allocation.chat_budget_mb),
                total_mb: allocation.total_mb,
                utilization: 0,
                allocations,
            }
        })
        .collect()
}

fn estimated_aux_model_usages(
    total_aux_mb: u64,
    env_map: &BTreeMap<String, String>,
) -> Vec<GpuModelUsage> {
    if total_aux_mb == 0 {
        return Vec::new();
    }
    let models = [
        execution_baseline::auxiliary_model_selection(
            execution_baseline::AuxiliaryRole::Embedding,
            env_map.get("CTOX_EMBEDDING_MODEL").map(String::as_str),
        ),
        execution_baseline::auxiliary_model_selection(
            execution_baseline::AuxiliaryRole::Stt,
            env_map.get("CTOX_STT_MODEL").map(String::as_str),
        ),
        execution_baseline::auxiliary_model_selection(
            execution_baseline::AuxiliaryRole::Tts,
            env_map.get("CTOX_TTS_MODEL").map(String::as_str),
        ),
    ]
    .into_iter()
    .filter_map(|selection| {
        let reserve_mb = selection.gpu_reserve_mb();
        (reserve_mb > 0).then_some((selection.request_model, reserve_mb))
    })
    .collect::<Vec<_>>();
    if models.is_empty() {
        return Vec::new();
    }
    let total_weight = models.iter().map(|(_, weight)| *weight).sum::<u64>().max(1);
    let mut remaining = total_aux_mb;
    let mut usages = Vec::with_capacity(models.len());
    for (index, (model, weight)) in models.iter().enumerate() {
        let share = if index + 1 == models.len() {
            remaining
        } else {
            total_aux_mb.saturating_mul(*weight) / total_weight
        };
        remaining = remaining.saturating_sub(share);
        usages.push(GpuModelUsage {
            model: (*model).to_string(),
            short_label: compact_model_name(model, 18),
            used_mb: share,
        });
    }
    usages.retain(|usage| usage.used_mb > 0);
    usages
}

fn estimate_gpu_cards(
    live_cards: &[GpuCardState],
    loaded_model: &str,
    draft_model: &str,
    settings: &BTreeMap<String, String>,
    live_context: usize,
) -> Vec<GpuCardState> {
    let mut cards = live_cards.to_vec();
    let target_isq = settings
        .get("CTOX_VLLM_SERVE_ISQ")
        .map(String::as_str)
        .unwrap_or("Q4K");
    let target_context = settings
        .get("CTOX_VLLM_SERVE_MAX_SEQ_LEN")
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(live_context.max(2048));
    let estimated_total_mb = estimate_chat_model_memory_mb(draft_model, target_isq, target_context);

    let mut aux_by_gpu: HashMap<usize, u64> = HashMap::new();
    for card in &cards {
        let aux_used = card
            .allocations
            .iter()
            .filter(|allocation| allocation.model != loaded_model)
            .map(|allocation| allocation.used_mb)
            .sum::<u64>();
        aux_by_gpu.insert(card.index, aux_used);
    }

    let weights = parse_device_layer_weights(settings.get("CTOX_VLLM_SERVE_DEVICE_LAYERS"));
    let total_weight = weights.values().copied().sum::<u64>();
    let selected_gpu_indices = if weights.is_empty() {
        cards.iter().map(|card| card.index).collect::<Vec<_>>()
    } else {
        weights.keys().copied().collect::<Vec<_>>()
    };
    let gpu_count = selected_gpu_indices.len().max(1) as u64;

    for card in &mut cards {
        let aux_used = aux_by_gpu.get(&card.index).copied().unwrap_or(0);
        card.allocations
            .retain(|allocation| allocation.model != loaded_model);
        let chat_used = if let Some(weight) = weights.get(&card.index) {
            if total_weight == 0 {
                estimated_total_mb / gpu_count
            } else {
                estimated_total_mb.saturating_mul(*weight) / total_weight
            }
        } else if weights.is_empty() {
            estimated_total_mb / gpu_count
        } else {
            0
        };
        if chat_used > 0 {
            card.allocations.push(GpuModelUsage {
                model: draft_model.to_string(),
                short_label: compact_model_name(draft_model, 18),
                used_mb: chat_used,
            });
        }
        card.used_mb = aux_used.saturating_add(chat_used);
        card.allocations
            .sort_by(|left, right| right.used_mb.cmp(&left.used_mb));
    }
    cards
}

fn configured_runtime_models(env_map: &BTreeMap<String, String>) -> Vec<String> {
    let mut models = Vec::new();
    if let Some(value) = env_map
        .get("CTOX_CHAT_MODEL")
        .map(|value| value.trim())
        .filter(|value| !value.is_empty())
    {
        models.push(value.to_string());
    }
    for selection in [
        execution_baseline::auxiliary_model_selection(
            execution_baseline::AuxiliaryRole::Embedding,
            env_map.get("CTOX_EMBEDDING_MODEL").map(String::as_str),
        ),
        execution_baseline::auxiliary_model_selection(
            execution_baseline::AuxiliaryRole::Stt,
            env_map.get("CTOX_STT_MODEL").map(String::as_str),
        ),
        execution_baseline::auxiliary_model_selection(
            execution_baseline::AuxiliaryRole::Tts,
            env_map.get("CTOX_TTS_MODEL").map(String::as_str),
        ),
    ] {
        if !models
            .iter()
            .any(|existing| existing == selection.request_model)
        {
            models.push(selection.request_model.to_string());
        }
    }
    models
}

fn filter_gpu_cards_to_models(
    live_cards: &[GpuCardState],
    allowed_models: &[String],
) -> Vec<GpuCardState> {
    if allowed_models.is_empty() {
        return live_cards.to_vec();
    }
    let mut cards = live_cards.to_vec();
    for card in &mut cards {
        card.allocations.retain(|allocation| {
            allowed_models
                .iter()
                .any(|model| model == &allocation.model)
        });
        card.allocations
            .sort_by(|left, right| right.used_mb.cmp(&left.used_mb));
        card.used_mb = card
            .allocations
            .iter()
            .map(|allocation| allocation.used_mb)
            .sum();
    }
    cards
}

fn parse_device_layer_weights(raw: Option<&String>) -> BTreeMap<usize, u64> {
    let mut weights = BTreeMap::new();
    let Some(raw) = raw else {
        return weights;
    };
    for chunk in raw.trim_matches('\'').split(';') {
        let Some((gpu, weight)) = chunk.split_once(':') else {
            continue;
        };
        let Ok(gpu_index) = gpu.trim().parse::<usize>() else {
            continue;
        };
        let Ok(weight_value) = weight.trim().parse::<u64>() else {
            continue;
        };
        weights.insert(gpu_index, weight_value);
    }
    weights
}

fn estimate_chat_model_memory_mb(model: &str, isq: &str, target_context: usize) -> u64 {
    let base_mb = match model.trim() {
        "openai/gpt-oss-20b" => 18_500,
        "Qwen/Qwen3.5-4B" => 4_000,
        "Qwen/Qwen3.5-9B" => 7_000,
        "Qwen/Qwen3.5-27B" => 17_500,
        "Qwen/Qwen3.5-35B-A3B" => 20_500,
        "zai-org/GLM-4.7-Flash" => 21_000,
        _ => 12_000,
    } as f64;
    let isq_factor = match isq.trim().to_ascii_uppercase().as_str() {
        "Q2K" => 0.72,
        "Q3K" => 0.84,
        "Q4K" => 1.0,
        "Q5K" => 1.12,
        "Q6K" => 1.24,
        "Q8K" => 1.40,
        "FP8" => 1.30,
        _ => 1.0,
    };
    let context_factor = 0.88 + 0.12 * ((target_context.max(1024) as f64) / 4096.0).clamp(0.5, 4.0);
    (base_mb * isq_factor * context_factor).round() as u64
}

fn estimate_max_context_window(
    cards: &[GpuCardState],
    loaded_model: &str,
    draft_model: &str,
    settings: &BTreeMap<String, String>,
    live_context: usize,
) -> usize {
    if cards.is_empty() {
        return DEFAULT_MAX_CONTEXT;
    }
    let target_isq = settings
        .get("CTOX_VLLM_SERVE_ISQ")
        .map(String::as_str)
        .unwrap_or("Q4K");
    let base_context = live_context.max(2048);
    let base_memory_mb =
        estimate_chat_model_memory_mb(draft_model, target_isq, base_context).max(1);
    let weights = parse_device_layer_weights(settings.get("CTOX_VLLM_SERVE_DEVICE_LAYERS"));
    let selected_gpu_indices = if weights.is_empty() {
        cards.iter().map(|card| card.index).collect::<Vec<_>>()
    } else {
        weights.keys().copied().collect::<Vec<_>>()
    };
    let selected_cards = cards
        .iter()
        .filter(|card| selected_gpu_indices.contains(&card.index))
        .collect::<Vec<_>>();
    if selected_cards.is_empty() {
        return DEFAULT_MAX_CONTEXT;
    }
    let total_budget_mb = selected_cards
        .iter()
        .map(|card| ((card.total_mb as f64) * 0.96).floor() as u64)
        .sum::<u64>();
    let aux_usage_mb = selected_cards
        .iter()
        .map(|card| {
            card.allocations
                .iter()
                .filter(|allocation| allocation.model != loaded_model)
                .map(|allocation| allocation.used_mb)
                .sum::<u64>()
        })
        .sum::<u64>();
    if total_budget_mb <= aux_usage_mb {
        return 2048;
    }
    let usable_chat_budget_mb = total_budget_mb - aux_usage_mb;
    let low_context = 2048usize;
    let high_context = 8192usize;
    let low_memory_mb =
        estimate_chat_model_memory_mb(draft_model, target_isq, low_context).max(base_memory_mb);
    let high_memory_mb =
        estimate_chat_model_memory_mb(draft_model, target_isq, high_context).max(low_memory_mb + 1);
    let slope_mb_per_token =
        (high_memory_mb.saturating_sub(low_memory_mb)) as f64 / (high_context - low_context) as f64;
    let fixed_overhead_mb =
        (low_memory_mb as f64 - slope_mb_per_token * low_context as f64).max(0.0);
    let budget_f = usable_chat_budget_mb as f64;
    let estimated_context = if budget_f <= fixed_overhead_mb {
        low_context
    } else {
        ((budget_f - fixed_overhead_mb) / slope_mb_per_token).round() as usize
    };
    estimated_context.clamp(2048, DEFAULT_MAX_CONTEXT)
}

fn render_qr_lines(payload: &str) -> Option<Vec<String>> {
    let code = QrCode::new(payload.as_bytes()).ok()?;
    let width = code.width();
    let colors = code.to_colors();
    let pad = 4usize;
    let mut matrix = vec![vec![false; width + pad * 2]; width + pad * 2];
    for y in 0..width {
        for x in 0..width {
            matrix[y + pad][x + pad] = matches!(colors[y * width + x], QrColor::Dark);
        }
    }
    if matrix.len() % 2 != 0 {
        matrix.push(vec![false; matrix[0].len()]);
    }
    let mut lines = Vec::with_capacity(matrix.len() / 2);
    for y in (0..matrix.len()).step_by(2) {
        let top = &matrix[y];
        let bottom = &matrix[y + 1];
        let mut line = String::with_capacity(top.len());
        for x in 0..top.len() {
            let ch = match (top[x], bottom[x]) {
                (false, false) => ' ',
                (true, false) => '▀',
                (false, true) => '▄',
                (true, true) => '█',
            };
            line.push(ch);
        }
        lines.push(line);
    }
    Some(lines)
}

fn resolve_jami_runtime_account(
    root: &Path,
    configured_account_id: &str,
    configured_profile_name: &str,
) -> JamiResolveOutcome {
    let adapter = root.join("scripts/communication_jami_cli.mjs");
    let mut command = Command::new("node");
    command
        .current_dir(root)
        .arg(&adapter)
        .arg("resolve-account");
    if !configured_account_id.trim().is_empty() {
        command
            .arg("--account-id")
            .arg(configured_account_id.trim());
    }
    if !configured_profile_name.trim().is_empty() {
        command
            .arg("--profile-name")
            .arg(configured_profile_name.trim());
    }
    let output = match command.output() {
        Ok(output) => output,
        Err(err) => {
            return JamiResolveOutcome {
                account: None,
                error: Some(format!("failed to start jami adapter: {err}")),
                dbus_env_file: None,
            };
        }
    };
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let parsed: Result<JamiResolvedEnvelope, _> = serde_json::from_slice(&output.stdout);
    let fallback_error = (!stderr.trim().is_empty())
        .then(|| stderr.trim().to_string())
        .or_else(|| {
            if stdout.trim().is_empty() {
                None
            } else {
                Some(stdout.trim().to_string())
            }
        });
    if !output.status.success() {
        return match parsed {
            Ok(parsed) => JamiResolveOutcome {
                account: parsed.resolved_account,
                error: parsed.error.or(fallback_error),
                dbus_env_file: parsed.dbus_env_file,
            },
            Err(_) => JamiResolveOutcome {
                account: None,
                error: fallback_error.or_else(|| {
                    Some(format!("jami adapter exited with status {}", output.status))
                }),
                dbus_env_file: None,
            },
        };
    }
    match parsed {
        Ok(parsed) => JamiResolveOutcome {
            account: parsed.resolved_account,
            error: if parsed.ok { parsed.error } else { parsed.error.or(fallback_error) },
            dbus_env_file: parsed.dbus_env_file,
        },
        Err(err) => JamiResolveOutcome {
            account: None,
            error: Some(format!("failed to parse jami adapter output: {err}")),
            dbus_env_file: None,
        },
    }
}

fn jami_missing_account_lines(dbus_env_file: Option<&str>, has_config: bool) -> Vec<String> {
    let mut lines = vec!["No live Jami RING account is available yet.".to_string()];
    if has_config {
        lines.push("Configured account/profile could not be resolved to an active share URI.".to_string());
    } else {
        lines.push("No Jami account id or profile is configured yet, so the TUI cannot derive a QR target.".to_string());
    }
    if let Some(path) = dbus_env_file.filter(|value| !value.trim().is_empty()) {
        lines.push(format!("dbus {}", truncate_for_ui(path, 40)));
    }
    lines.push("Verify the Jami daemon is running and that a RING account exists.".to_string());
    lines
}

fn jami_error_lines(error: &str, dbus_env_file: Option<&str>, has_config: bool) -> Vec<String> {
    let mut lines = vec!["Jami runtime is not ready.".to_string()];
    lines.push(format!("blocker {}", truncate_for_ui(error, 68)));
    if error.contains("gdbus ENOENT") {
        lines.push("Missing `gdbus`: install GLib DBus tooling so CTOX can resolve the Jami account.".to_string());
    }
    if let Some(path) = dbus_env_file.filter(|value| !value.trim().is_empty()) {
        lines.push(format!("dbus {}", truncate_for_ui(path, 40)));
    } else {
        lines.push("No Jami DBus env file is loaded yet.".to_string());
    }
    if has_config {
        lines.push("Configured Jami account/profile is present, but runtime resolution still failed.".to_string());
    } else {
        lines.push("No configured Jami account/profile is available to fall back to.".to_string());
    }
    lines.push("Start or repair the Jami daemon first; then reopen the Jami settings view.".to_string());
    lines
}

fn model_perf_stats_path(root: &Path) -> PathBuf {
    root.join("runtime/model_perf_stats.json")
}

fn load_model_perf_stats(root: &Path) -> BTreeMap<String, ModelPerfStats> {
    let path = model_perf_stats_path(root);
    std::fs::read(&path)
        .ok()
        .and_then(|bytes| serde_json::from_slice::<BTreeMap<String, ModelPerfStats>>(&bytes).ok())
        .unwrap_or_default()
}

fn save_model_perf_stats(root: &Path, stats: &BTreeMap<String, ModelPerfStats>) -> Result<()> {
    let path = model_perf_stats_path(root);
    let bytes = serde_json::to_vec_pretty(stats).context("failed to encode model perf stats")?;
    std::fs::write(path, bytes).context("failed to write model perf stats")?;
    Ok(())
}

fn mask_secret(value: &str) -> String {
    if value.chars().count() <= 4 {
        return "*".repeat(value.chars().count());
    }
    let tail: String = value
        .chars()
        .rev()
        .take(4)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    format!(
        "{}{}",
        "*".repeat(value.chars().count().saturating_sub(4)),
        tail
    )
}

struct TerminalGuard;

impl TerminalGuard {
    fn enter(stdout: &mut io::Stdout) -> Result<Self> {
        terminal::enable_raw_mode().context("failed to enable raw mode")?;
        execute!(
            stdout,
            EnterAlternateScreen,
            terminal::Clear(ClearType::All),
            cursor::Hide
        )?;
        Ok(Self)
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        let mut stdout = io::stdout();
        let _ = terminal::disable_raw_mode();
        let _ = execute!(stdout, cursor::Show, LeaveAlternateScreen);
    }
}

fn summarize_inline(value: &str, max_chars: usize) -> String {
    truncate_for_ui(value, max_chars)
}

fn truncate_for_ui(value: &str, max_chars: usize) -> String {
    let collapsed = value.split_whitespace().collect::<Vec<_>>().join(" ");
    if collapsed.chars().count() <= max_chars {
        return collapsed;
    }
    let mut out = collapsed
        .chars()
        .take(max_chars.saturating_sub(1))
        .collect::<String>();
    out.push('…');
    out
}

fn compact_model_name(model: &str, width: usize) -> String {
    let short = model
        .rsplit('/')
        .next()
        .unwrap_or(model)
        .replace("openai/", "")
        .replace("Qwen/", "");
    if width < 72 {
        truncate_for_ui(&short, 18)
    } else {
        truncate_for_ui(&short, 30)
    }
}

fn signed_delta(delta: i64) -> String {
    if delta > 0 {
        format!("+{delta}")
    } else {
        delta.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime_config;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_root(label: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("ctox-tui-{label}-{nonce}"));
        fs::create_dir_all(&root).unwrap();
        root
    }

    #[test]
    fn supported_chat_models_only_include_openai_when_provider_and_key_are_present() {
        let local_only = supported_chat_model_choices(&BTreeMap::new());
        assert!(local_only.contains(&"openai/gpt-oss-20b"));
        assert!(!local_only.contains(&"gpt-5.4"));

        let mut openai_without_key = BTreeMap::new();
        openai_without_key.insert("CTOX_CHAT_SOURCE".to_string(), "api".to_string());
        openai_without_key.insert("CTOX_API_PROVIDER".to_string(), "openai".to_string());
        let still_local_only = supported_chat_model_choices(&openai_without_key);
        assert!(!still_local_only.contains(&"gpt-5.4"));

        openai_without_key.insert("OPENAI_API_KEY".to_string(), "sk-test".to_string());
        let with_openai = supported_chat_model_choices(&openai_without_key);
        assert!(with_openai.contains(&"gpt-5.4-nano"));
        assert!(with_openai.contains(&"gpt-5.4-mini"));
        assert!(with_openai.contains(&"gpt-5.4"));
    }

    #[test]
    fn model_settings_include_provider_and_communication_fields() {
        let root = temp_root("settings");
        let mut env_map = BTreeMap::new();
        env_map.insert(
            "CTOX_CHAT_MODEL".to_string(),
            "openai/gpt-oss-20b".to_string(),
        );
        runtime_config::save_runtime_env_map(&root, &env_map).unwrap();

        let items = load_settings_items(&root);
        let keys: Vec<_> = items.iter().map(|item| item.key).collect();

        assert!(keys.contains(&"CTOX_CHAT_MODEL"));
        assert!(keys.contains(&"CTOX_CHAT_MODEL_BASE"));
        assert!(keys.contains(&"CTOX_CHAT_MODEL_BOOST"));
        assert!(keys.contains(&"CTOX_BOOST_DEFAULT_MINUTES"));
        assert!(keys.contains(&"CTOX_API_PROVIDER"));
        assert!(keys.contains(&"OPENAI_API_KEY"));
        assert!(!keys.contains(&"CTOX_AUXILIARY_CUDA_VISIBLE_DEVICES"));
        assert!(!keys.contains(&"CTOX_EMBEDDING_PORT"));
        assert!(!keys.contains(&"CTOX_STT_PORT"));
        assert!(!keys.contains(&"CTOX_TTS_PORT"));
        assert!(keys.contains(&"CTOX_OWNER_NAME"));
        assert!(keys.contains(&"CTOX_OWNER_EMAIL_ADDRESS"));
        assert!(keys.contains(&"CTOX_OWNER_PREFERRED_CHANNEL"));
        assert!(keys.contains(&"CTO_EMAIL_PASSWORD"));
        assert!(keys.contains(&"CTO_EMAIL_PROVIDER"));
        assert!(keys.contains(&"CTO_JAMI_ACCOUNT_ID"));
    }

    #[test]
    fn app_defaults_to_model_settings_view_and_openai_key_visibility_follows_provider() {
        let root = temp_root("visibility");
        let db_path = root.join("runtime/test.sqlite3");
        let mut env_map = BTreeMap::new();
        env_map.insert(
            "CTOX_CHAT_MODEL".to_string(),
            "openai/gpt-oss-20b".to_string(),
        );
        env_map.insert("CTOX_API_PROVIDER".to_string(), "local".to_string());
        runtime_config::save_runtime_env_map(&root, &env_map).unwrap();

        let mut app = App::new(root.clone(), db_path);
        assert_eq!(app.settings_view, SettingsView::Model);

        let openai_key_row = app
            .settings_items
            .iter()
            .find(|item| item.key == "OPENAI_API_KEY")
            .unwrap()
            .clone();
        assert!(!app.setting_visible(&openai_key_row));

        let chat_source_idx = app
            .settings_items
            .iter()
            .position(|item| item.key == "CTOX_CHAT_SOURCE")
            .unwrap();
        let provider_idx = app
            .settings_items
            .iter()
            .position(|item| item.key == "CTOX_API_PROVIDER")
            .unwrap();
        app.settings_items[chat_source_idx].value = "api".to_string();
        app.settings_items[provider_idx].value = "openai".to_string();
        assert!(app.setting_visible(&openai_key_row));
    }

    #[test]
    fn visible_model_settings_follow_requested_minimal_flow() {
        let root = temp_root("minimal-flow");
        let db_path = root.join("runtime/test.sqlite3");
        let mut env_map = BTreeMap::new();
        env_map.insert("CTOX_CHAT_SOURCE".to_string(), "local".to_string());
        runtime_config::save_runtime_env_map(&root, &env_map).unwrap();

        let mut app = App::new(root.clone(), db_path);
        let local_keys = app
            .visible_setting_indices()
            .into_iter()
            .map(|idx| app.settings_items[idx].key)
            .collect::<Vec<_>>();
        assert_eq!(
            local_keys,
            vec![
                "CTOX_SERVICE_TOGGLE",
                "CTOX_CHAT_SOURCE",
                "CTOX_CHAT_MODEL",
                "CTOX_CHAT_MODEL_BOOST",
                "CTOX_BOOST_DEFAULT_MINUTES",
                "CTOX_CHAT_LOCAL_PRESET",
                "CTOX_EMBEDDING_MODEL",
                "CTOX_STT_MODEL",
                "CTOX_TTS_MODEL",
                "CTOX_CHAT_COMPACTION_THRESHOLD_PERCENT",
                "CTOX_OWNER_NAME",
                "CTOX_OWNER_EMAIL_ADDRESS",
                "CTOX_OWNER_PREFERRED_CHANNEL",
            ]
        );

        let chat_source_idx = app
            .settings_items
            .iter()
            .position(|item| item.key == "CTOX_CHAT_SOURCE")
            .unwrap();
        app.settings_items[chat_source_idx].value = "api".to_string();
        let api_keys = app
            .visible_setting_indices()
            .into_iter()
            .map(|idx| app.settings_items[idx].key)
            .collect::<Vec<_>>();
        assert_eq!(
            api_keys,
            vec![
                "CTOX_SERVICE_TOGGLE",
                "CTOX_CHAT_SOURCE",
                "CTOX_API_PROVIDER",
                "CTOX_CHAT_MODEL",
                "CTOX_CHAT_MODEL_BOOST",
                "CTOX_BOOST_DEFAULT_MINUTES",
                "OPENAI_API_KEY",
                "CTOX_EMBEDDING_MODEL",
                "CTOX_STT_MODEL",
                "CTOX_TTS_MODEL",
                "CTOX_CHAT_COMPACTION_THRESHOLD_PERCENT",
                "CTOX_OWNER_NAME",
                "CTOX_OWNER_EMAIL_ADDRESS",
                "CTOX_OWNER_PREFERRED_CHANNEL",
            ]
        );
    }

    #[test]
    fn communication_visibility_switches_email_and_jami_blocks() {
        let root = temp_root("comm-visibility");
        let db_path = root.join("runtime/test.sqlite3");
        let mut app = App::new(root, db_path);

        let owner_row = app
            .settings_items
            .iter()
            .find(|item| item.key == "CTOX_OWNER_NAME")
            .unwrap()
            .clone();
        let email_protocol_row = app
            .settings_items
            .iter()
            .find(|item| item.key == "CTO_EMAIL_PROVIDER")
            .unwrap()
            .clone();
        let email_password_row = app
            .settings_items
            .iter()
            .find(|item| item.key == "CTO_EMAIL_PASSWORD")
            .unwrap()
            .clone();
        let jami_row = app
            .settings_items
            .iter()
            .find(|item| item.key == "CTO_JAMI_ACCOUNT_ID")
            .unwrap()
            .clone();
        let ews_url_row = app
            .settings_items
            .iter()
            .find(|item| item.key == "CTO_EMAIL_EWS_URL")
            .unwrap()
            .clone();

        assert!(app.setting_visible(&owner_row));
        assert!(!app.setting_visible(&email_protocol_row));
        assert!(!app.setting_visible(&email_password_row));
        assert!(!app.setting_visible(&jami_row));

        app.settings_items
            .iter_mut()
            .find(|item| item.key == "CTOX_OWNER_PREFERRED_CHANNEL")
            .unwrap()
            .value = "email".to_string();
        assert!(app.setting_visible(&email_protocol_row));
        assert!(app.setting_visible(&email_password_row));
        assert!(!app.setting_visible(&jami_row));

        app.settings_items
            .iter_mut()
            .find(|item| item.key == "CTO_EMAIL_PROVIDER")
            .unwrap()
            .value = "ews".to_string();
        assert!(app.setting_visible(&ews_url_row));

        app.settings_items
            .iter_mut()
            .find(|item| item.key == "CTOX_OWNER_PREFERRED_CHANNEL")
            .unwrap()
            .value = "jami".to_string();
        assert!(!app.setting_visible(&email_protocol_row));
        assert!(app.setting_visible(&jami_row));
    }

    #[test]
    fn load_skill_catalog_discovers_skill_scripts_and_resources() {
        let root = temp_root("skills-catalog");
        let skill_dir = root.join("skills/.system/demo-skill");
        fs::create_dir_all(skill_dir.join("scripts")).unwrap();
        fs::create_dir_all(skill_dir.join("references")).unwrap();
        fs::write(
            skill_dir.join("SKILL.md"),
            "# Demo Skill\n\nUse this skill when the operator wants a demo workflow.\n",
        )
        .unwrap();
        fs::write(skill_dir.join("scripts/demo_tool.py"), "print('ok')\n").unwrap();
        fs::write(skill_dir.join("references/example.md"), "reference\n").unwrap();

        let catalog = load_skill_catalog(&root);
        let entry = catalog
            .iter()
            .find(|entry| entry.name == "demo-skill")
            .unwrap();
        assert_eq!(entry.description, "Use this skill when the operator wants a demo workflow.");
        assert!(entry.helper_tools.iter().any(|tool| tool == "demo_tool.py"));
        assert!(entry.resources.iter().any(|resource| resource.contains("references:")));
    }

    #[test]
    fn tab_cycles_chat_skills_settings() {
        let root = temp_root("page-cycle");
        let db_path = root.join("runtime/test.sqlite3");
        let mut app = App::new(root, db_path);
        assert_eq!(app.page, Page::Chat);
        app.handle_key_event(KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE))
            .unwrap();
        assert_eq!(app.page, Page::Skills);
        app.handle_key_event(KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE))
            .unwrap();
        assert_eq!(app.page, Page::Settings);
        app.handle_key_event(KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE))
            .unwrap();
        assert_eq!(app.page, Page::Chat);
    }

    #[test]
    fn jami_error_lines_surface_missing_gdbus_explicitly() {
        let lines = jami_error_lines("spawnSync gdbus ENOENT", None, false);
        let joined = lines.join("\n");
        assert!(joined.contains("Jami runtime is not ready."));
        assert!(joined.contains("Missing `gdbus`"));
        assert!(joined.contains("No configured Jami account/profile"));
    }

    #[test]
    fn jami_missing_account_lines_explain_missing_runtime_account() {
        let lines = jami_missing_account_lines(Some("/tmp/cto-jami-dbus.env"), true);
        let joined = lines.join("\n");
        assert!(joined.contains("No live Jami RING account is available yet."));
        assert!(joined.contains("Configured account/profile could not be resolved"));
        assert!(joined.contains("/tmp/cto-jami-dbus.env"));
    }

    #[test]
    fn save_settings_mirrors_chat_model_to_base_model() {
        let root = temp_root("boost-save");
        let db_path = root.join("runtime/test.sqlite3");
        let mut app = App::new(root.clone(), db_path);
        let chat_idx = app
            .settings_items
            .iter()
            .position(|item| item.key == "CTOX_CHAT_MODEL")
            .unwrap();
        app.settings_items[chat_idx].value = "gpt-5.4-mini".to_string();
        app.save_settings().unwrap();

        let saved = runtime_config::load_runtime_env_map(&root).unwrap();
        assert_eq!(
            saved.get("CTOX_CHAT_MODEL_BASE").map(String::as_str),
            Some("gpt-5.4-mini")
        );
    }
}
