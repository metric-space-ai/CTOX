// ref: sdk/cliproxy/auth/conductor_fast_error_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// Port-Note: direct and wrapped typed errors share one no-retry classifier
// License: MIT (upstream); modifications AGPL-3.0-only

use std::error::Error;
use std::fmt;
use std::sync::Arc;

use crate::sdk::pluginapi::PluginExecutionError;

use super::{is_request_scoped_plugin_error, plugin_error_status, AuthError};

#[derive(Debug)]
struct Wrapped(AuthError);

impl fmt::Display for Wrapped {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("direct response rejected the request")
    }
}

impl Error for Wrapped {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        Some(&self.0)
    }
}

fn request_scoped() -> AuthError {
    AuthError {
        code: "request_scoped".into(),
        message: "local request failure".into(),
        http_status: 422,
        ..AuthError::default()
    }
}

#[test]
fn fast_local_error_does_not_enter_refresh_or_cooldown_path() {
    let error: PluginExecutionError = Arc::new(request_scoped());
    assert!(is_request_scoped_plugin_error(&error));
    assert_eq!(plugin_error_status(&error), 422);
}

#[test]
fn fast_wrapped_direct_error_preserves_request_scope() {
    let error: PluginExecutionError = Arc::new(Wrapped(request_scoped()));
    assert!(is_request_scoped_plugin_error(&error));
    assert_eq!(plugin_error_status(&error), 422);
}
