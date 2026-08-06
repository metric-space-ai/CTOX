// ref: internal/logging/request_logger_writer.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: partial
// License: MIT (upstream); modifications AGPL-3.0-only

use super::request_logger::{FileRequestLogger, RequestLogRecord, RequestLogger};
use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

fn temp_logs_dir(label: &str) -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    std::env::temp_dir().join(format!(
        "ctox-cliproxy-{label}-{}-{nonce}",
        std::process::id()
    ))
}

fn record(status_code: u16) -> RequestLogRecord {
    RequestLogRecord {
        url: "/v1/responses?api_key=supersecret&plain=value".to_owned(),
        method: "POST".to_owned(),
        request_headers: BTreeMap::from([
            (
                "Authorization".to_owned(),
                vec!["Bearer upstream-secret-token".to_owned()],
            ),
            ("X-Plain".to_owned(), vec!["visible".to_owned()]),
        ]),
        request_body: br#"{"model":"server-owned"}"#.to_vec(),
        status_code,
        response_headers: BTreeMap::new(),
        response_body: br#"{"error":"upstream failed"}"#.to_vec(),
        request_id: "req/1".to_owned(),
        streaming: false,
    }
}

#[test]
fn disabled_logger_writes_only_forced_error_with_redacted_credentials() {
    let logs_dir = temp_logs_dir("forced");
    let logger = FileRequestLogger::new(false, &logs_dir, "", 3);
    assert!(logger.log_request(&record(200), false).unwrap().is_none());

    let path = logger.log_request(&record(502), true).unwrap().unwrap();
    let content = fs::read_to_string(path).unwrap();
    assert!(content.contains("api_key=supe...cret"));
    assert!(content.contains("Authorization: Bearer upst...oken"));
    assert!(content.contains("X-Plain: visible"));
    assert!(!content.contains("supersecret"));
    assert!(!content.contains("upstream-secret-token"));
    assert!(fs::remove_dir_all(logs_dir).is_ok());
}

#[test]
fn forced_error_retention_is_bounded() {
    let logs_dir = temp_logs_dir("retention");
    let logger = FileRequestLogger::new(false, &logs_dir, "", 2);
    for _ in 0..4 {
        logger.log_request(&record(500), true).unwrap();
    }
    let files = fs::read_dir(&logs_dir).unwrap().count();
    assert_eq!(files, 2);
    assert!(fs::remove_dir_all(logs_dir).is_ok());
}
