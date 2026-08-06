// Origin: CTOX
// License: AGPL-3.0-only

mod antigravity_version;
mod claude_code_instructions;
#[path = "copy-example-config.rs"]
mod copy_example_config;
mod credentials;
mod header_utils;
#[path = "mime-type.rs"]
pub mod mime_type;
mod oauth;

#[cfg(test)]
mod antigravity_version_test;
#[cfg(test)]
mod claude_code_instructions_test;
#[cfg(test)]
mod header_utils_test;
#[cfg(test)]
mod oauth_test;

pub use antigravity_version::{
    AntigravityManifestFuture, AntigravityManifestRequest, AntigravityManifestTransport,
    AntigravityVersionCache, AntigravityVersionClock, AntigravityVersionError,
    SystemAntigravityVersionClock, ANTIGRAVITY_FALLBACK_VERSION, ANTIGRAVITY_FETCH_TIMEOUT,
    ANTIGRAVITY_GOOG_API_CLIENT_UA, ANTIGRAVITY_HUB_LATEST_MANIFEST_URL, ANTIGRAVITY_HUB_PLATFORM,
    ANTIGRAVITY_NODE_API_CLIENT_UA, ANTIGRAVITY_VERSION_CACHE_TTL,
};
pub use claude_code_instructions::CLAUDE_CODE_INSTRUCTIONS;
pub use copy_example_config::copy_config_template;
pub use credentials::{
    merge_metadata, saving_credentials_event, MergeMetadataError, SavingCredentialsEvent,
    CREDENTIAL_SEPARATOR,
};
pub use header_utils::{ensure_header, scrub_proxy_and_fingerprint_headers, Headers};
pub use oauth::{
    async_prompt, generate_random_state, parse_oauth_callback, OAuthCallback,
    ParseOAuthCallbackError, RandomStateError,
};
