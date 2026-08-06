// ref: internal/interfaces/error_message.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

pub type Headers = BTreeMap<String, Vec<String>>;

/// Error plus optional trusted direct downstream response.
#[derive(Clone, Default)]
pub struct ErrorMessage {
    pub status_code: isize,
    pub error: Option<Arc<dyn std::error::Error + Send + Sync + 'static>>,
    pub addon: Headers,
    pub direct_response: bool,
    pub body: Vec<u8>,
    pub headers: Headers,
}

impl fmt::Debug for ErrorMessage {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ErrorMessage")
            .field("status_code", &self.status_code)
            .field("error", &self.error.as_ref().map(|_| "[REDACTED]"))
            .field("addon_header_count", &self.addon.len())
            .field("direct_response", &self.direct_response)
            .field("body_len", &self.body.len())
            .field("header_count", &self.headers.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use std::io;

    use super::*;

    #[test]
    fn aggregate_preserves_multi_headers_and_redacts_body_and_error() {
        let message = ErrorMessage {
            status_code: 429,
            error: Some(Arc::new(io::Error::other("token-do-not-log"))),
            addon: BTreeMap::from([(
                "Retry-After".to_owned(),
                vec!["1".to_owned(), "2".to_owned()],
            )]),
            direct_response: true,
            body: b"secret-body".to_vec(),
            headers: BTreeMap::from([("Content-Type".to_owned(), vec!["x".to_owned()])]),
        };
        assert_eq!(message.addon["Retry-After"].len(), 2);
        assert_eq!(message.body, b"secret-body");
        let debug = format!("{message:?}");
        assert!(!debug.contains("token-do-not-log"));
        assert!(!debug.contains("secret-body"));
        assert!(debug.contains("body_len: 11"));
    }
}
