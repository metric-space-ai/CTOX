// ref: internal/cmd/auth_manager.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use super::login_prompt::Prompt;
use std::collections::BTreeMap;
use std::fmt;
use std::io;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CommandConfig {
    pub config_path: PathBuf,
    pub auth_dir: PathBuf,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoginPlan {
    pub provider: String,
    pub no_browser: bool,
    pub callback_port: Option<u16>,
    pub metadata: BTreeMap<String, String>,
    pub config: CommandConfig,
}
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LoginRecord {
    pub id: String,
    pub label: String,
    pub saved_path: Option<PathBuf>,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoginFailureKind {
    PortInUse,
    Cancelled,
    InvalidConfig,
    Provider,
    Storage,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoginFailure {
    pub kind: LoginFailureKind,
    pub message: String,
}
impl fmt::Display for LoginFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}
impl std::error::Error for LoginFailure {}

#[derive(Debug, Default)]
pub struct CommandCancellation(AtomicBool);
impl CommandCancellation {
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Release);
    }
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
}

pub trait CommandOutput: Send + Sync {
    fn info(&self, message: &str);
    fn error(&self, message: &str);
}
pub trait LoginHandler: Send + Sync {
    fn login(
        &self,
        plan: &LoginPlan,
        prompt: &dyn Prompt,
        cancellation: &CommandCancellation,
    ) -> Result<LoginRecord, LoginFailure>;
}

pub struct AuthManager {
    handlers: BTreeMap<String, Arc<dyn LoginHandler>>,
}
impl AuthManager {
    pub fn new(handlers: impl IntoIterator<Item = (String, Arc<dyn LoginHandler>)>) -> Self {
        Self {
            handlers: handlers
                .into_iter()
                .map(|(provider, handler)| (provider.trim().to_ascii_lowercase(), handler))
                .collect(),
        }
    }
    pub fn login(
        &self,
        plan: &LoginPlan,
        prompt: &dyn Prompt,
        cancellation: &CommandCancellation,
    ) -> Result<LoginRecord, LoginFailure> {
        if cancellation.is_cancelled() {
            return Err(LoginFailure {
                kind: LoginFailureKind::Cancelled,
                message: "authentication cancelled".into(),
            });
        }
        let handler = self
            .handlers
            .get(&plan.provider.trim().to_ascii_lowercase())
            .ok_or_else(|| LoginFailure {
                kind: LoginFailureKind::Provider,
                message: format!("unsupported authentication provider: {}", plan.provider),
            })?;
        handler.login(plan, prompt, cancellation)
    }
}

pub struct LoginCommand<'a> {
    pub manager: &'a AuthManager,
    pub prompt: &'a dyn Prompt,
    pub output: &'a dyn CommandOutput,
    pub cancellation: &'a CommandCancellation,
}
impl LoginCommand<'_> {
    pub fn execute(&self, plan: &LoginPlan) -> Result<LoginRecord, LoginFailure> {
        match self.manager.login(plan, self.prompt, self.cancellation) {
            Ok(record) => {
                if let Some(path) = &record.saved_path {
                    self.output
                        .info(&format!("Authentication saved to {}", path.display()));
                }
                if !record.label.is_empty() {
                    self.output
                        .info(&format!("Authenticated as {}", record.label));
                }
                self.output
                    .info(&format!("{} authentication successful", plan.provider));
                Ok(record)
            }
            Err(error) => {
                self.output.error(&format!(
                    "{} authentication failed: {}",
                    plan.provider, error
                ));
                Err(error)
            }
        }
    }
}

pub fn validate_callback_port(port: Option<u16>) -> io::Result<Option<u16>> {
    if port == Some(0) {
        Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "callback port must be non-zero",
        ))
    } else {
        Ok(port)
    }
}
