// Origin: CTOX
// License: AGPL-3.0-only

pub mod anthropic;
pub mod anthropic_auth;
pub mod errors;
pub mod html_templates;
pub mod identity;
pub mod oauth_response;
pub mod oauth_server;
pub mod pkce;
pub mod token;
#[cfg(feature = "anthropic-fingerprint-transport")]
pub mod utls_transport;

#[cfg(all(test, feature = "anthropic-fingerprint-transport"))]
mod anthropic_auth_proxy_test;
#[cfg(test)]
mod anthropic_auth_test;
#[cfg(test)]
mod identity_test;
#[cfg(test)]
mod oauth_response_test;
#[cfg(all(test, feature = "anthropic-fingerprint-transport"))]
mod utls_transport_test;

pub use anthropic::{ClaudeAuthBundle, ClaudeUserInfo};
#[cfg(feature = "anthropic-fingerprint-transport")]
pub use anthropic_auth::{new_claude_auth_with_proxy, ClaudeProxyOverride};
pub use anthropic_auth::{
    AuthFlowError, ClaudeAuth, ClaudeCodeExchangeTransport, ClaudeRefreshCoordinator,
    ClaudeRefreshTransport, ExchangeHttpResponse, ExchangeRequest, OAuthInspectHttpResponse,
    OAuthInspectKind, OAuthInspectRequest, OAuthProfile, RefreshClock, RefreshError,
    RefreshHttpResponse, RefreshRequest, RefreshTransportFailure, SystemRefreshClock, AUTH_SCOPE,
    AUTH_URL, CLIENT_ID, EXCHANGE_TIMEOUT, PROFILE_URL, REDIRECT_URI, REFRESH_TIMEOUT,
    REFRESH_TOKEN_URL, ROLES_URL, TOKEN_URL,
};
pub use errors::{
    get_user_friendly_message, is_authentication_error, is_oauth_error, new_authentication_error,
    new_oauth_error, AuthenticationError, OAuthError, ERR_CALLBACK_TIMEOUT,
    ERR_CODE_EXCHANGE_FAILED, ERR_INVALID_STATE, ERR_PORT_IN_USE, ERR_SERVER_START_FAILED,
};
pub use html_templates::{
    render_login_success_html, HtmlTemplateError, LOGIN_SUCCESS_HTML, SETUP_NOTICE_HTML,
};
pub use identity::{
    ensure_device_id_pool, generate_device_id_pool, has_canonical_device_id_pool,
    normalize_device_id_pool, read_device_id_pool, read_metadata_string, select_device_id,
    store_device_id_pool, store_metadata_string, store_metadata_value, valid_device_id,
    ClaudeIdentityError, CLAUDE_DEVICE_IDS_METADATA_KEY, CLAUDE_DEVICE_POOL_SIZE,
};
pub use oauth_response::{decode_claude_oauth_response_body, ClaudeOAuthResponseError};
pub use oauth_server::{OAuthResult, OAuthServer, OAuthServerError};
pub use pkce::{generate_pkce_codes, PkceCodes, PkceError};
pub use token::{
    ClaudeCredentialHandles, ClaudeSecretHandle, ClaudeSecretKind, ClaudeSecretStore,
    ClaudeStoredCredentials, ClaudeTokenData, ClaudeTokenStorage, SecretStoreError, SecretString,
    TokenError, CLAUDE_REFRESH_LEAD,
};
#[cfg(feature = "anthropic-fingerprint-transport")]
pub use utls_transport::{
    AnthropicHttpTransport, AnthropicProxyMode, AnthropicTransportBuildError,
};
