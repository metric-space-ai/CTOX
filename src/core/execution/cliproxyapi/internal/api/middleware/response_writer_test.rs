// ref: internal/api/middleware/response_writer_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use super::response_writer::{detect_streaming, RequestInfo, ResponseWriterWrapper};
use crate::internal::logging::request_logger::{
    FileRequestLogger, RequestLogRecord, RequestLogger,
};
use std::collections::BTreeMap;
use std::fs;
use std::io;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

#[derive(Default)]
struct MemoryLogger {
    enabled: bool,
    calls: Mutex<Vec<(RequestLogRecord, bool)>>,
}

impl RequestLogger for MemoryLogger {
    fn is_enabled(&self) -> bool {
        self.enabled
    }

    fn log_request(&self, record: &RequestLogRecord, force: bool) -> io::Result<Option<PathBuf>> {
        self.calls.lock().unwrap().push((record.clone(), force));
        Ok(None)
    }
}

fn request_info() -> RequestInfo {
    RequestInfo {
        url: "/v1/responses".to_owned(),
        method: "POST".to_owned(),
        headers: BTreeMap::new(),
        body: br#"{"stream":true}"#.to_vec(),
        request_id: "req-1".to_owned(),
    }
}

#[test]
fn content_type_takes_precedence_over_request_stream_hint() {
    assert!(!detect_streaming(
        Some("application/json"),
        br#"{"stream":true}"#
    ));
    assert!(detect_streaming(
        Some("text/event-stream; charset=utf-8"),
        b"{}"
    ));
    assert!(detect_streaming(None, br#"{"stream": true}"#));
}

#[test]
fn error_only_mode_forces_disabled_logger_and_captures_error_body() {
    let logger = Arc::new(MemoryLogger::default());
    let mut writer = ResponseWriterWrapper::new(logger.clone(), request_info());
    writer.set_log_on_error_only(true);
    writer.write_header(
        502,
        BTreeMap::from([(
            "Content-Type".to_owned(),
            vec!["application/json".to_owned()],
        )]),
    );
    writer.write(br#"{"error":"upstream failed"}"#);
    assert!(writer.finalize().unwrap().is_none());

    let calls = logger.calls.lock().unwrap();
    assert_eq!(calls.len(), 1);
    assert!(calls[0].1);
    assert_eq!(calls[0].0.status_code, 502);
    assert_eq!(calls[0].0.response_body, br#"{"error":"upstream failed"}"#);
}

#[test]
fn error_only_success_does_not_call_disabled_logger() {
    let logger = Arc::new(MemoryLogger::default());
    let mut writer = ResponseWriterWrapper::new(logger.clone(), request_info());
    writer.set_log_on_error_only(true);
    writer.write_header(200, BTreeMap::new());
    writer.write(b"not retained");
    assert!(writer.finalize().unwrap().is_none());
    assert!(logger.calls.lock().unwrap().is_empty());
}

#[test]
fn response_capture_and_file_logger_form_an_operational_error_path() {
    let logs_dir =
        std::env::temp_dir().join(format!("ctox-response-writer-{}", std::process::id()));
    let _ = fs::remove_dir_all(&logs_dir);
    let logger = Arc::new(FileRequestLogger::new(false, &logs_dir, "", 2));
    let mut writer = ResponseWriterWrapper::new(logger, request_info());
    writer.set_log_on_error_only(true);
    writer.write_header(500, BTreeMap::new());
    writer.write(b"failure");
    let path = writer.finalize().unwrap().unwrap();
    assert!(fs::read_to_string(path).unwrap().contains("failure"));
    assert!(fs::remove_dir_all(logs_dir).is_ok());
}
