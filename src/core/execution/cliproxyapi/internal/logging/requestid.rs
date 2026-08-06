// ref: internal/logging/requestid.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use super::requestmeta::RequestContext;

pub const HANDLER_REQUEST_ID_KEY: &str = "__request_id__";

/// Creates the upstream eight-character lowercase hexadecimal request ID.
#[must_use]
pub fn generate_request_id() -> String {
    let mut bytes = [0_u8; 4];
    if getrandom::fill(&mut bytes).is_err() {
        return "00000000".to_owned();
    }
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[must_use]
pub fn with_request_id(
    context: Option<&RequestContext>,
    request_id: impl Into<String>,
) -> RequestContext {
    let mut derived = context.cloned().unwrap_or_default();
    derived.request_id = Some(request_id.into());
    derived
}

#[must_use]
pub fn get_request_id(context: Option<&RequestContext>) -> &str {
    context
        .and_then(|context| context.request_id.as_deref())
        .unwrap_or_default()
}

/// Framework-neutral equivalent of the upstream Gin request-local setter.
pub fn set_handler_request_id(context: Option<&mut RequestContext>, request_id: impl Into<String>) {
    if let Some(context) = context {
        context.handler_request_id = Some(request_id.into());
    }
}

#[must_use]
pub fn get_handler_request_id(context: Option<&RequestContext>) -> &str {
    context
        .and_then(|context| context.handler_request_id.as_deref())
        .unwrap_or_default()
}
