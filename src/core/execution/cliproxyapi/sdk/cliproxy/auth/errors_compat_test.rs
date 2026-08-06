// ref: sdk/cliproxy/auth/errors_compat_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use crate::sdk::cliproxy::executor::{RequestScopedError, StatusError};

use super::{AuthError, AuthStatus};

#[test]
fn auth_error_preserves_the_complete_legacy_field_contract() {
    let error = AuthError {
        code: "code".to_owned(),
        message: "message".to_owned(),
        retryable: false,
        http_status: 408,
    };

    assert_eq!(
        serde_json::to_value(&error).expect("serialize auth error"),
        serde_json::json!({
            "code": "code",
            "message": "message",
            "retryable": false,
            "http_status": 408
        })
    );
    assert_eq!(error.to_string(), "code: message");
    assert_eq!(StatusError::status_code(&error), 408);
}

#[test]
fn empty_optional_error_fields_match_go_omitempty() {
    let error = AuthError {
        message: "plain".to_owned(),
        ..AuthError::default()
    };

    assert_eq!(error.to_string(), "plain");
    assert_eq!(
        serde_json::to_value(&error).expect("serialize auth error"),
        serde_json::json!({"message": "plain", "retryable": false})
    );
    assert_eq!(error.status_code(), 0);
    assert!(!error.is_request_scoped());
}

#[test]
fn request_scoped_code_implements_the_shared_executor_trait() {
    let error = AuthError {
        code: "request_scoped".to_owned(),
        message: "request shape is invalid".to_owned(),
        retryable: false,
        http_status: 400,
    };

    assert!(error.is_request_scoped());
    assert!(RequestScopedError::is_request_scoped(&error));
    assert!(!format!("{error:?}").contains("request shape is invalid"));
}

#[test]
fn auth_status_preserves_known_and_future_string_values() {
    let known = [
        AuthStatus::Unknown,
        AuthStatus::Active,
        AuthStatus::Pending,
        AuthStatus::Refreshing,
        AuthStatus::Error,
        AuthStatus::Disabled,
    ];
    for status in known {
        let encoded = serde_json::to_string(&status).expect("serialize status");
        let decoded: AuthStatus = serde_json::from_str(&encoded).expect("deserialize status");
        assert_eq!(decoded, status);
        assert!(decoded.is_known());
    }

    let future: AuthStatus = serde_json::from_str(r#""rotating""#).expect("future status");
    assert_eq!(future.as_str(), "rotating");
    assert!(!future.is_known());
    assert_eq!(serde_json::to_string(&future).unwrap(), r#""rotating""#);
}
