// ref: internal/tui/app.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use super::auth_tab::AuthTabModel;
use super::config_tab::ConfigTabModel;
use super::dashboard::{DashboardModel, DashboardSnapshot};
use super::i18n::I18n;
use super::keys_tab::{KeysSnapshot, KeysTabModel};
use super::logs_tab::{LogFilter, LogsTabModel};
use super::oauth_tab::{OAuthPollMessage, OAuthStartMessage, OAuthTabModel};
use serde_json::Value;
use std::io;
use std::time::{Duration, SystemTime};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tab {
    Dashboard,
    Config,
    Auth,
    Keys,
    OAuth,
    Logs,
}
impl Tab {
    const ALL: [Self; 6] = [
        Self::Dashboard,
        Self::Config,
        Self::Auth,
        Self::Keys,
        Self::OAuth,
        Self::Logs,
    ];
    fn index(self) -> usize {
        Self::ALL.iter().position(|tab| *tab == self).unwrap_or(0)
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyCode {
    Char(char),
    Enter,
    Escape,
    Backspace,
    Up,
    Down,
    Left,
    Right,
    Tab,
    BackTab,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KeyEvent {
    pub code: KeyCode,
    pub control: bool,
}
#[derive(Debug, Clone, PartialEq)]
pub enum TerminalEvent {
    Key(KeyEvent),
    Resize {
        width: u16,
        height: u16,
    },
    Tick(SystemTime),
    Dashboard(DashboardSnapshot),
    Config(Value),
    AuthFiles(Vec<Value>),
    Keys(KeysSnapshot),
    LogLines {
        lines: Vec<String>,
        cursor: Option<i64>,
    },
    OAuthStart(OAuthStartMessage),
    OAuthPoll(OAuthPollMessage),
    Status(String),
    Shutdown,
}

pub trait TerminalClock: Send + Sync {
    fn now(&self) -> SystemTime;
}
#[derive(Debug, Default)]
pub struct SystemTerminalClock;
impl TerminalClock for SystemTerminalClock {
    fn now(&self) -> SystemTime {
        SystemTime::now()
    }
}
pub trait TerminalBackend {
    fn enter(&mut self) -> io::Result<()>;
    fn size(&self) -> io::Result<(u16, u16)>;
    fn read_event(&mut self, timeout: Duration) -> io::Result<Option<TerminalEvent>>;
    fn render(&mut self, snapshot: &str) -> io::Result<()>;
    fn leave(&mut self) -> io::Result<()>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AppAction {
    None,
    Quit,
    Refresh(Tab),
    ConfigPatch {
        path: String,
        value: Value,
    },
    DeleteAuth(String),
    ToggleAuth {
        name: String,
        disabled: bool,
    },
    StartOAuth {
        provider: super::oauth_tab::OAuthProvider,
        generation: u64,
    },
    CancelOAuth(String),
    SubmitOAuthCallback(String),
}

pub struct App {
    selected: Tab,
    width: u16,
    height: u16,
    i18n: I18n,
    dashboard: DashboardModel,
    config: ConfigTabModel,
    auth: AuthTabModel,
    keys: KeysTabModel,
    oauth: OAuthTabModel,
    logs: LogsTabModel,
    status: String,
    shutdown: bool,
}
impl Default for App {
    fn default() -> Self {
        Self {
            selected: Tab::Dashboard,
            width: 80,
            height: 24,
            i18n: I18n::default(),
            dashboard: DashboardModel::default(),
            config: ConfigTabModel::default(),
            auth: AuthTabModel::default(),
            keys: KeysTabModel::default(),
            oauth: OAuthTabModel::default(),
            logs: LogsTabModel::default(),
            status: String::new(),
            shutdown: false,
        }
    }
}
impl App {
    pub fn selected_tab(&self) -> Tab {
        self.selected
    }
    pub fn is_shutdown(&self) -> bool {
        self.shutdown
    }
    pub fn update(&mut self, event: TerminalEvent) -> AppAction {
        match event {
            TerminalEvent::Resize { width, height } => {
                self.set_size(width, height);
                AppAction::None
            }
            TerminalEvent::Dashboard(snapshot) => {
                self.dashboard.apply(snapshot);
                AppAction::None
            }
            TerminalEvent::Config(config) => {
                self.config.apply_config(&config);
                AppAction::None
            }
            TerminalEvent::AuthFiles(files) => {
                self.auth.apply_files(files);
                AppAction::None
            }
            TerminalEvent::Keys(keys) => {
                self.keys.apply(keys);
                AppAction::None
            }
            TerminalEvent::LogLines { lines, cursor } => {
                self.logs.append(lines, cursor);
                AppAction::None
            }
            TerminalEvent::OAuthStart(message) => {
                self.oauth.apply_start(message);
                AppAction::None
            }
            TerminalEvent::OAuthPoll(message) => {
                self.oauth.apply_poll(message);
                AppAction::None
            }
            TerminalEvent::Status(status) => {
                self.status = status;
                AppAction::None
            }
            TerminalEvent::Shutdown => {
                self.shutdown = true;
                AppAction::Quit
            }
            TerminalEvent::Tick(_) => AppAction::None,
            TerminalEvent::Key(key) => self.handle_key(key),
        }
    }

    fn handle_key(&mut self, key: KeyEvent) -> AppAction {
        if key.control && matches!(key.code, KeyCode::Char('c' | 'C'))
            || matches!(key.code, KeyCode::Char('q' | 'Q'))
        {
            self.shutdown = true;
            return AppAction::Quit;
        }
        match key.code {
            KeyCode::Tab | KeyCode::Right => {
                self.select_relative(1);
                return AppAction::Refresh(self.selected);
            }
            KeyCode::BackTab | KeyCode::Left => {
                self.select_relative(-1);
                return AppAction::Refresh(self.selected);
            }
            KeyCode::Char('l' | 'L') if key.control => {
                self.i18n.toggle();
                return AppAction::None;
            }
            _ => {}
        }
        match self.selected {
            Tab::Dashboard => {
                if matches!(key.code, KeyCode::Char('r' | 'R')) {
                    AppAction::Refresh(Tab::Dashboard)
                } else {
                    AppAction::None
                }
            }
            Tab::Config => self.handle_config_key(key.code),
            Tab::Auth => self.handle_auth_key(key.code),
            Tab::Keys => {
                match key.code {
                    KeyCode::Up => self.keys.move_cursor(-1),
                    KeyCode::Down => self.keys.move_cursor(1),
                    KeyCode::Char('v' | 'V') => self.keys.toggle_reveal(),
                    _ => {}
                }
                AppAction::None
            }
            Tab::OAuth => self.handle_oauth_key(key.code),
            Tab::Logs => {
                match key.code {
                    KeyCode::Up => self.logs.scroll(1),
                    KeyCode::Down => self.logs.scroll(-1),
                    KeyCode::Char('0') => self.logs.set_filter(LogFilter::All),
                    KeyCode::Char('1') => self.logs.set_filter(LogFilter::Info),
                    KeyCode::Char('2') => self.logs.set_filter(LogFilter::Warn),
                    KeyCode::Char('3') => self.logs.set_filter(LogFilter::Error),
                    KeyCode::Char('4') => self.logs.set_filter(LogFilter::Debug),
                    _ => {}
                }
                AppAction::None
            }
        }
    }
    fn handle_config_key(&mut self, key: KeyCode) -> AppAction {
        match key {
            KeyCode::Up => self.config.move_cursor(-1),
            KeyCode::Down => self.config.move_cursor(1),
            KeyCode::Char('e' | 'E') => self.config.begin_edit(),
            KeyCode::Char(' ') => {
                if let Some((path, value)) = self.config.toggle_bool() {
                    return AppAction::ConfigPatch { path, value };
                }
            }
            KeyCode::Char(character) => self.config.edit_push(character),
            KeyCode::Backspace => self.config.edit_backspace(),
            KeyCode::Escape => self.config.cancel_edit(),
            KeyCode::Enter => {
                if let Some((path, value)) = self.config.commit_edit() {
                    return AppAction::ConfigPatch { path, value };
                }
            }
            _ => {}
        }
        AppAction::None
    }
    fn handle_auth_key(&mut self, key: KeyCode) -> AppAction {
        match key {
            KeyCode::Up => self.auth.move_cursor(-1),
            KeyCode::Down => self.auth.move_cursor(1),
            KeyCode::Char('d' | 'D') => self.auth.request_delete(),
            KeyCode::Char('y' | 'Y') => {
                if let Some(name) = self.auth.confirm_delete() {
                    return AppAction::DeleteAuth(name);
                }
            }
            KeyCode::Char('t' | 'T') => {
                if let Some((name, disabled)) = self.auth.toggle_selected() {
                    return AppAction::ToggleAuth { name, disabled };
                }
            }
            KeyCode::Escape => self.auth.cancel_delete(),
            _ => {}
        }
        AppAction::None
    }
    fn handle_oauth_key(&mut self, key: KeyCode) -> AppAction {
        match key {
            KeyCode::Up => self.oauth.select(-1),
            KeyCode::Down => self.oauth.select(1),
            KeyCode::Enter => {
                if let Some(callback) = self.oauth.take_callback() {
                    return AppAction::SubmitOAuthCallback(callback);
                }
                if let Some((provider, generation)) = self.oauth.start() {
                    return AppAction::StartOAuth {
                        provider,
                        generation,
                    };
                }
            }
            KeyCode::Escape => {
                if let Some(state) = self.oauth.cancel() {
                    return AppAction::CancelOAuth(state);
                }
            }
            KeyCode::Backspace => self.oauth.callback_backspace(),
            KeyCode::Char(character) => self.oauth.callback_push(character),
            _ => {}
        }
        AppAction::None
    }
    fn select_relative(&mut self, delta: isize) {
        let len = Tab::ALL.len() as isize;
        let next = (self.selected.index() as isize + delta).rem_euclid(len) as usize;
        self.selected = Tab::ALL[next];
    }
    fn set_size(&mut self, width: u16, height: u16) {
        self.width = width.max(1);
        self.height = height.max(3);
        let content_height = self.height.saturating_sub(2);
        self.dashboard.set_size(self.width, content_height);
        self.config.set_size(self.width, content_height);
        self.auth.set_size(self.width, content_height);
        self.keys.set_size(self.width, content_height);
        self.oauth.set_size(self.width, content_height);
        self.logs.set_size(self.width, content_height);
    }
    pub fn view(&self) -> String {
        let tabs = self
            .i18n
            .tab_names()
            .into_iter()
            .enumerate()
            .map(|(index, name)| {
                if index == self.selected.index() {
                    format!("[{name}]")
                } else {
                    name.to_owned()
                }
            })
            .collect::<Vec<_>>()
            .join(" ");
        let body = match self.selected {
            Tab::Dashboard => self.dashboard.render(),
            Tab::Config => self.config.render(),
            Tab::Auth => self.auth.render(),
            Tab::Keys => self.keys.render(),
            Tab::OAuth => self.oauth.render(),
            Tab::Logs => self.logs.render(),
        };
        let status = if self.status.is_empty() {
            format!("Tab/←→ navigate · q {}", self.i18n.text("quit"))
        } else {
            self.status.clone()
        };
        format!(
            "{}\n{}\n{}",
            super::styles::clip_width(&tabs, self.width as usize),
            body,
            super::styles::clip_width(&status, self.width as usize)
        )
    }
}

pub fn run_terminal(
    app: &mut App,
    terminal: &mut dyn TerminalBackend,
    clock: &dyn TerminalClock,
) -> io::Result<()> {
    terminal.enter()?;
    let result = (|| {
        let (width, height) = terminal.size()?;
        app.update(TerminalEvent::Resize { width, height });
        while !app.is_shutdown() {
            let event = terminal
                .read_event(Duration::from_millis(100))?
                .unwrap_or_else(|| TerminalEvent::Tick(clock.now()));
            app.update(event);
            terminal.render(&app.view())?;
        }
        Ok(())
    })();
    let leave = terminal.leave();
    match (result, leave) {
        (Err(error), _) => Err(error),
        (Ok(()), Err(error)) => Err(error),
        (Ok(()), Ok(())) => Ok(()),
    }
}
