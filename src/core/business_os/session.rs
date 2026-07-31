// Origin: CTOX
// License: AGPL-3.0-only

//! The Business OS session and its accessors.
//!
//! Filed below store.rs and policy.rs because both need it: the policy
//! overlays decide from a session, and moving them while this type lived in
//! store.rs would have made policy depend on store — the inversion S-CUT5
//! exists to remove. store.rs re-exports these so its own call sites and the
//! public API stay unchanged.

use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct BusinessOsSession {
    pub ok: bool,
    pub authenticated: bool,
    pub auth_required: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<BusinessOsSessionUser>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub login_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct BusinessOsSessionUser {
    pub id: String,
    pub display_name: String,
    pub role: String,
    pub is_admin: bool,
}

pub(super) fn session_user_id(session: &BusinessOsSession) -> Option<&str> {
    session.user.as_ref().map(|user| user.id.as_str())
}

pub(super) fn session_role(session: &BusinessOsSession) -> &str {
    session
        .user
        .as_ref()
        .map(|user| user.role.as_str())
        .unwrap_or("user")
}
