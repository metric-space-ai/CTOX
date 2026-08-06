// ref: internal/cmd/openai_device_login.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use super::auth_manager::{CommandConfig, LoginCommand, LoginFailure, LoginPlan, LoginRecord};
use super::openai_login::{login_plan, LoginOptions};
use std::collections::BTreeMap;
pub const CODEX_LOGIN_MODE_METADATA_KEY: &str = "codex_login_mode";
pub fn codex_device_login_plan(config: CommandConfig, options: &LoginOptions) -> LoginPlan {
    login_plan(
        "codex",
        config,
        options,
        BTreeMap::from([(CODEX_LOGIN_MODE_METADATA_KEY.into(), "device".into())]),
    )
}
pub fn do_codex_device_login(
    command: &LoginCommand<'_>,
    config: CommandConfig,
    options: &LoginOptions,
) -> Result<LoginRecord, LoginFailure> {
    command.execute(&codex_device_login_plan(config, options))
}
