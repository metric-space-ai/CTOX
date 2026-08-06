// ref: internal/cmd/anthropic_login.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use super::auth_manager::{CommandConfig, LoginCommand, LoginFailure, LoginPlan, LoginRecord};
use super::openai_login::{login_plan, LoginOptions};
use std::collections::BTreeMap;
pub fn claude_login_plan(config: CommandConfig, options: &LoginOptions) -> LoginPlan {
    login_plan("claude", config, options, BTreeMap::new())
}
pub fn do_claude_login(
    command: &LoginCommand<'_>,
    config: CommandConfig,
    options: &LoginOptions,
) -> Result<LoginRecord, LoginFailure> {
    command.execute(&claude_login_plan(config, options))
}
