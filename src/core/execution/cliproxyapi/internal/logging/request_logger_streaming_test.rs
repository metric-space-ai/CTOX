// ref: internal/logging/request_logger_streaming.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: partial
// License: MIT (upstream); modifications AGPL-3.0-only

use super::request_logger::{FileRequestLogger, RequestLogger};
use super::request_logger_body_source::FileBodySource;
use super::request_logger_streaming::saturated_queue_drops_without_blocking;
use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

fn temp_logs_dir(label: &str) -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    std::env::temp_dir().join(format!(
        "ctox-stream-{label}-{}-{nonce}",
        std::process::id()
    ))
}

#[test]
fn file_body_source_preserves_order_recreates_directory_and_cleans_idempotently() {
    let logs_dir = temp_logs_dir("body-source");
    let source = FileBodySource::new_in_dir(&logs_dir, "api/request").unwrap();
    source.append_part(b"  first  ").unwrap();
    let mut second = source.create_part("second").unwrap();
    second.write_all(b"second").unwrap();
    drop(second);
    assert_eq!(source.bytes().unwrap(), b"first\nsecond");

    for path in source.paths() {
        fs::remove_file(path).unwrap();
    }
    source.append_bytes(b"recreated").unwrap();
    assert_eq!(source.bytes().unwrap(), b"recreated");
    source.cleanup().unwrap();
    source.cleanup().unwrap();
    assert!(!source.has_payload());
    assert!(fs::remove_dir_all(logs_dir).is_ok());
}

#[test]
fn saturated_stream_queue_has_an_explicit_drop_path() {
    assert_eq!(saturated_queue_drops_without_blocking(), 1);
}

#[test]
fn streaming_writer_drains_before_close_redacts_and_removes_temp_body() {
    let logs_dir = temp_logs_dir("writer");
    let logger = FileRequestLogger::new(true, &logs_dir, "", 2);
    let mut writer = logger
        .log_streaming_request(
            "/v1/responses?token=streaming-secret",
            "POST",
            &BTreeMap::from([(
                "Authorization".to_owned(),
                vec!["Bearer streaming-auth-token".to_owned()],
            )]),
            br#"{"stream":true}"#,
            "req-stream",
        )
        .unwrap()
        .unwrap();
    writer.write_status(
        200,
        &BTreeMap::from([(
            "Content-Type".to_owned(),
            vec!["text/event-stream".to_owned()],
        )]),
    );
    writer.write_chunk_async(b"data: one\n\n");
    writer.write_chunk_async(b"data: two\n\n");
    let outcome = writer.close().unwrap();
    let content = fs::read_to_string(outcome.path).unwrap();
    assert!(content.contains("data: one\n\ndata: two"));
    assert!(content.contains("token=stre...cret"));
    assert!(content.contains("Authorization: Bearer stre...oken"));
    assert!(!content.contains("streaming-secret"));
    assert!(!content.contains("streaming-auth-token"));
    assert!(
        !fs::read_dir(&logs_dir).unwrap().flatten().any(|entry| entry
            .file_name()
            .to_string_lossy()
            .starts_with("response-body-"))
    );
    assert!(fs::remove_dir_all(logs_dir).is_ok());
}
