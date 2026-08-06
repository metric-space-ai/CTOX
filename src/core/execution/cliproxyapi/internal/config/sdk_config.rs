// ref: internal/config/sdk_config.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use serde::{Deserialize, Serialize};

use super::DisableImageGenerationMode;

/// Provider-neutral server settings from the public SDK configuration.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case", deny_unknown_fields)]
pub struct SdkConfig {
    #[serde(default)]
    pub proxy_url: String,
    #[serde(default)]
    pub disable_image_generation: DisableImageGenerationMode,
    #[serde(default)]
    pub gpt_image_2_base_model: String,
    #[serde(default)]
    pub video_result_auth_cache_ttl: String,
    #[serde(default)]
    pub force_model_prefix: bool,
    #[serde(default)]
    pub request_log: bool,
    /// Runtime-only mirror used by handlers; never accepted from serialized configuration.
    #[serde(skip)]
    pub codex_optimize_multi_agent_v2: bool,
    #[serde(default)]
    pub claude_code: ClaudeCodeConfig,
    #[serde(default)]
    pub api_keys: Vec<String>,
    #[serde(default)]
    pub passthrough_headers: bool,
    #[serde(default)]
    pub streaming: StreamingConfig,
    #[serde(default)]
    pub nonstream_keepalive_interval: i32,
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case", deny_unknown_fields)]
pub struct ClaudeCodeConfig {
    #[serde(default)]
    pub disable_cloaking_model_list: bool,
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case", deny_unknown_fields)]
pub struct StreamingConfig {
    #[serde(default)]
    pub keepalive_seconds: i32,
    #[serde(default)]
    pub bootstrap_retries: i32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sdk_surface_is_closed_and_preserves_streaming_values() {
        let config: SdkConfig = serde_yaml::from_str(
            "passthrough-headers: true\nstreaming:\n  keepalive-seconds: 5\n  bootstrap-retries: 2\n",
        )
        .unwrap();
        assert!(config.passthrough_headers);
        assert_eq!(config.streaming.keepalive_seconds, 5);
        assert_eq!(config.streaming.bootstrap_retries, 2);
        assert!(serde_yaml::from_str::<SdkConfig>("unknown: true\n").is_err());
    }
}
