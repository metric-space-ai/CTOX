// ref: internal/auth/antigravity/constants.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use std::time::Duration;

pub const CALLBACK_PORT: u16 = 51_121;
pub const AUTH_ENDPOINT: &str = "https://accounts.google.com/o/oauth2/v2/auth";
pub const TOKEN_ENDPOINT: &str = "https://oauth2.googleapis.com/token";
pub const USER_INFO_ENDPOINT: &str = "https://www.googleapis.com/oauth2/v2/userinfo?alt=json";
pub const API_ENDPOINT: &str = "https://cloudcode-pa.googleapis.com";
pub const DAILY_API_ENDPOINT: &str = "https://daily-cloudcode-pa.googleapis.com";
pub const API_VERSION: &str = "v1internal";
pub const REFRESH_SKEW: Duration = Duration::from_secs(3_000);

pub(crate) const ANTIGRAVITY_USER_AGENT: &str = "antigravity/hub/2.2.1 darwin/arm64";
pub(crate) const ANTIGRAVITY_NODE_API_CLIENT_USER_AGENT: &str = "google-api-nodejs-client/10.3.0";
pub(crate) const ANTIGRAVITY_GOOG_API_CLIENT_USER_AGENT: &str = "gl-node/22.21.1";

pub(crate) const CLIENT_ID: &str =
    "1071006060591-tmhssin2h21lcre235vtolojh4g403ep.apps.googleusercontent.com";
// Google treats this installed-application OAuth value as a public client
// credential. It is never accepted from runtime config or rendered by Debug.
pub(crate) const CLIENT_SECRET: &str = "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";
pub(crate) const SCOPES: &[&str] = &[
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/cclog",
    "https://www.googleapis.com/auth/experimentsandconfigs",
];
