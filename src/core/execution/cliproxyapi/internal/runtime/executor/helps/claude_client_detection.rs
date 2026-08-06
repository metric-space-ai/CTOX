// ref: internal/runtime/executor/helps/claude_client_detection.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use serde_json::Value;

use super::claude_code_session::header_value_case_insensitive;
use super::claude_device_profile::{
    parse_claude_code_user_agent_details, plausible_claude_code_user_agent, ClaudeHeaderDefaults,
};
use super::cloak_utils::is_valid_user_id;
use crate::sdk::api::handlers::header_filter::HeaderMap;

const CLAUDE_CODE_BETA: &str = "claude-code-20250219";

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ClaudeCodeRequestDetection {
    pub confirmed: bool,
    pub strong_signals: bool,
    pub native_client: bool,
    pub x_app_cli: bool,
    pub user_agent: bool,
    pub betas_present: bool,
    pub metadata_user_id: bool,
    pub entrypoint: String,
    pub subclient: String,
    pub agent_sdk_version: String,
}

pub fn detect_claude_code_request(
    headers: Option<&HeaderMap>,
    payload: &[u8],
    count_tokens: bool,
    defaults: &ClaudeHeaderDefaults,
) -> ClaudeCodeRequestDetection {
    let user_agent = header_value_case_insensitive(headers, "User-Agent");
    let (entrypoint, agent_sdk_version) = parse_claude_code_user_agent_details(&user_agent);
    let metadata_user_id = serde_json::from_slice::<Value>(payload)
        .ok()
        .and_then(|root| {
            root.get("metadata")?
                .get("user_id")?
                .as_str()
                .map(is_valid_user_id)
        })
        .unwrap_or(false);
    let x_app_cli = header_value_case_insensitive(headers, "X-App") == "cli";
    let user_agent_signal = plausible_claude_code_user_agent(&user_agent, defaults);
    let betas_present = header_contains_claude_code_beta(headers);
    let strong_signals =
        x_app_cli && user_agent_signal && betas_present && (count_tokens || metadata_user_id);
    let native_client = matches!(entrypoint.as_str(), "cli" | "sdk-cli" | "claude-vscode");
    ClaudeCodeRequestDetection {
        confirmed: strong_signals && native_client,
        strong_signals,
        native_client,
        x_app_cli,
        user_agent: user_agent_signal,
        betas_present,
        metadata_user_id,
        subclient: claude_code_subclient(&entrypoint).to_owned(),
        entrypoint,
        agent_sdk_version,
    }
}

fn header_contains_claude_code_beta(headers: Option<&HeaderMap>) -> bool {
    headers
        .into_iter()
        .flat_map(|headers| headers.iter())
        .filter(|(key, _)| key.eq_ignore_ascii_case("Anthropic-Beta"))
        .flat_map(|(_, values)| values)
        .flat_map(|value| value.split(','))
        .any(|beta| beta.trim() == CLAUDE_CODE_BETA)
}

fn claude_code_subclient(entrypoint: &str) -> &'static str {
    match entrypoint {
        "cli" => "claude-code-cli",
        "mcp" => "claude-code-mcp",
        "bench" => "claude-code-bench",
        "sdk-cli" => "claude-code-cli-sdk",
        "sdk-ts" => "claude-code-sdk-ts",
        "sdk-py" => "claude-code-sdk-py",
        "claude-vscode" => "claude-code-vscode",
        "claude-code-github-action" => "claude-code-gh-action",
        "local-agent" | "local_agent" => "claude-local-agent",
        "claude-desktop" => "claude-desktop",
        "claude-desktop-3p" => "claude-desktop-3p",
        "remote" => "claude-remote",
        "remote_baku" => "claude-remote-baku",
        "remote_cowork" => "claude-remote-cowork",
        "remote_trigger" => "claude-remote-trigger",
        "remote_desktop" => "claude-remote-desktop",
        "remote_mobile" => "claude-remote-mobile",
        "claude_in_slack" | "claude-in-slack" => "claude-in-slack",
        "claude-in-teams" => "claude-in-teams",
        "claude-security" => "claude-security",
        "ssh-remote" => "claude-ssh-remote",
        "claude-coworker" => "claude-coworker",
        "claude-coworker-terminal" => "claude-coworker-terminal",
        _ => "",
    }
}
