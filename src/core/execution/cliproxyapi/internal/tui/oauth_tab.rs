// ref: internal/tui/oauth_tab.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OAuthState {
    Idle,
    Starting,
    WaitingRemote,
    WaitingDevice,
    Success,
    Failed,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OAuthProvider {
    pub id: String,
    pub label: String,
    pub device_flow: bool,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OAuthStartMessage {
    pub generation: u64,
    pub state: String,
    pub url: String,
    pub user_code: String,
    pub error: Option<String>,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OAuthPollMessage {
    pub generation: u64,
    pub state: String,
    pub done: bool,
    pub error: Option<String>,
}
#[derive(Debug, Clone)]
pub struct OAuthTabModel {
    providers: Vec<OAuthProvider>,
    selected: usize,
    state: OAuthState,
    generation: u64,
    auth_state: String,
    auth_url: String,
    user_code: String,
    callback_input: String,
    consecutive_errors: usize,
    max_errors: usize,
    width: u16,
    height: u16,
}
impl Default for OAuthTabModel {
    fn default() -> Self {
        Self {
            providers: vec![
                OAuthProvider {
                    id: "claude".into(),
                    label: "Claude".into(),
                    device_flow: false,
                },
                OAuthProvider {
                    id: "codex".into(),
                    label: "Codex".into(),
                    device_flow: false,
                },
                OAuthProvider {
                    id: "gemini".into(),
                    label: "Gemini".into(),
                    device_flow: true,
                },
            ],
            selected: 0,
            state: OAuthState::Idle,
            generation: 0,
            auth_state: String::new(),
            auth_url: String::new(),
            user_code: String::new(),
            callback_input: String::new(),
            consecutive_errors: 0,
            max_errors: 3,
            width: 0,
            height: 0,
        }
    }
}
impl OAuthTabModel {
    pub fn set_size(&mut self, width: u16, height: u16) {
        self.width = width;
        self.height = height;
    }
    pub fn generation(&self) -> u64 {
        self.generation
    }
    pub fn state(&self) -> OAuthState {
        self.state
    }
    pub fn auth_state(&self) -> &str {
        &self.auth_state
    }
    pub fn select(&mut self, delta: isize) {
        self.selected = self
            .selected
            .saturating_add_signed(delta)
            .min(self.providers.len().saturating_sub(1));
    }
    pub fn start(&mut self) -> Option<(OAuthProvider, u64)> {
        let provider = self.providers.get(self.selected)?.clone();
        self.generation = self.generation.wrapping_add(1);
        self.state = OAuthState::Starting;
        self.clear_session();
        Some((provider, self.generation))
    }
    pub fn apply_start(&mut self, message: OAuthStartMessage) -> bool {
        if !should_accept_oauth_start(&message, self.generation) {
            return false;
        }
        if message.error.is_some() {
            self.state = OAuthState::Failed;
            return true;
        }
        self.auth_state = message.state;
        self.auth_url = message.url;
        self.user_code = message.user_code;
        self.state = if self.user_code.is_empty() {
            OAuthState::WaitingRemote
        } else {
            OAuthState::WaitingDevice
        };
        true
    }
    pub fn apply_poll(&mut self, message: OAuthPollMessage) -> bool {
        if !should_accept_oauth_poll(&message, &self.auth_state, self.generation, self.state) {
            return false;
        }
        if message.error.is_some() {
            self.consecutive_errors += 1;
            if should_fail_oauth_status_poll(self.consecutive_errors, self.max_errors) {
                self.state = OAuthState::Failed;
            }
            return true;
        }
        self.consecutive_errors = 0;
        if message.done {
            self.state = OAuthState::Success;
        }
        true
    }
    pub fn cancel(&mut self) -> Option<String> {
        self.generation = self.generation.wrapping_add(1);
        let remote = (!self.auth_state.is_empty()).then(|| self.auth_state.clone());
        self.state = OAuthState::Idle;
        self.clear_session();
        remote
    }
    pub fn callback_push(&mut self, character: char) {
        self.callback_input.push(character);
    }
    pub fn callback_backspace(&mut self) {
        self.callback_input.pop();
    }
    pub fn take_callback(&mut self) -> Option<String> {
        let value = self.callback_input.trim().to_owned();
        self.callback_input.clear();
        (!value.is_empty()).then_some(value)
    }
    pub fn render(&self) -> String {
        let mut lines = vec!["OAuth".to_owned()];
        match self.state {
            OAuthState::Idle => {
                for (index, provider) in self.providers.iter().enumerate() {
                    lines.push(format!(
                        "{} {}",
                        if index == self.selected { ">" } else { " " },
                        provider.label
                    ));
                }
            }
            OAuthState::Starting => lines.push("Starting authentication…".into()),
            OAuthState::WaitingRemote => {
                lines.push(format!("Open: {}", self.auth_url));
                lines.push(format!("Callback: {}", self.callback_input));
            }
            OAuthState::WaitingDevice => {
                lines.push(format!("Open: {}", self.auth_url));
                lines.push(format!("Code: {}", self.user_code));
            }
            OAuthState::Success => lines.push("Authentication successful".into()),
            OAuthState::Failed => lines.push("Authentication failed".into()),
        }
        lines
            .into_iter()
            .take(self.height.max(1) as usize)
            .map(|line| super::styles::clip_width(&line, self.width as usize))
            .collect::<Vec<_>>()
            .join("\n")
    }
    fn clear_session(&mut self) {
        self.auth_state.clear();
        self.auth_url.clear();
        self.user_code.clear();
        self.callback_input.clear();
        self.consecutive_errors = 0;
    }
}
pub fn should_accept_oauth_start(message: &OAuthStartMessage, generation: u64) -> bool {
    message.generation == generation
}
pub fn should_accept_oauth_poll(
    message: &OAuthPollMessage,
    auth_state: &str,
    generation: u64,
    state: OAuthState,
) -> bool {
    message.generation == generation
        && message.state == auth_state
        && matches!(state, OAuthState::WaitingRemote | OAuthState::WaitingDevice)
}
pub fn should_fail_oauth_status_poll(consecutive_errors: usize, max_errors: usize) -> bool {
    max_errors > 0 && consecutive_errors >= max_errors
}
pub fn wrap_text(text: &str, width: usize) -> Vec<String> {
    if width == 0 {
        return vec![String::new()];
    }
    let mut lines = Vec::new();
    let mut current = String::new();
    for word in text.split_whitespace() {
        if !current.is_empty() && current.chars().count() + word.chars().count() + 1 > width {
            lines.push(current);
            current = String::new();
        }
        if !current.is_empty() {
            current.push(' ');
        }
        current.push_str(word);
    }
    if !current.is_empty() || lines.is_empty() {
        lines.push(current);
    }
    lines
}
