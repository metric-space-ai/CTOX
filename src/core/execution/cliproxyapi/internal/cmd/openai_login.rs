// ref: internal/cmd/openai_login.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use super::auth_manager::{CommandConfig, LoginCommand, LoginFailure, LoginPlan, LoginRecord};
use std::collections::BTreeMap;
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LoginOptions {
    pub no_browser: bool,
    pub callback_port: Option<u16>,
}
pub fn login_plan(
    provider: &str,
    config: CommandConfig,
    options: &LoginOptions,
    metadata: BTreeMap<String, String>,
) -> LoginPlan {
    LoginPlan {
        provider: provider.trim().to_ascii_lowercase(),
        no_browser: options.no_browser,
        callback_port: options.callback_port,
        metadata,
        config,
    }
}
pub fn codex_login_plan(config: CommandConfig, options: &LoginOptions) -> LoginPlan {
    login_plan("codex", config, options, BTreeMap::new())
}
pub fn do_codex_login(
    command: &LoginCommand<'_>,
    config: CommandConfig,
    options: &LoginOptions,
) -> Result<LoginRecord, LoginFailure> {
    command.execute(&codex_login_plan(config, options))
}
