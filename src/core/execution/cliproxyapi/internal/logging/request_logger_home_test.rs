// ref: internal/logging/request_logger_home_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use super::request_logger::{FileRequestLogger, RequestLogRecord, RequestLogger};
use super::request_logger_home::{HomeRequestLogPayload, HomeRequestLogSink};
use super::request_logger_writer::DetailedRequestLog;
use std::collections::BTreeMap;
use std::io;
use std::sync::{Arc, Mutex};
use std::time::SystemTime;

struct StubHomeRequestSink {
    healthy: bool,
    pushed: Mutex<Vec<HomeRequestLogPayload>>,
}
impl HomeRequestLogSink for StubHomeRequestSink {
    fn heartbeat_ok(&self) -> bool {
        self.healthy
    }
    fn push_request_log(&self, payload: &HomeRequestLogPayload) -> io::Result<()> {
        self.pushed.lock().unwrap().push(payload.clone());
        Ok(())
    }
}

fn record() -> RequestLogRecord {
    RequestLogRecord {
        url: "/v1/chat/completions".into(),
        method: "POST".into(),
        request_headers: BTreeMap::from([
            ("Content-Type".into(), vec!["application/json".into()]),
            ("Authorization".into(), vec!["Bearer secret".into()]),
        ]),
        request_body: br#"{"input":"hello"}"#.to_vec(),
        status_code: 200,
        response_headers: BTreeMap::from([(
            "Content-Type".into(),
            vec!["application/json".into()],
        )]),
        response_body: br#"{"ok":true}"#.to_vec(),
        request_id: "req-1".into(),
        streaming: false,
    }
}

#[test]
fn bound_home_sink_replaces_local_request_log_output() {
    let dir = tempfile::tempdir().unwrap();
    let logger = FileRequestLogger::new(true, dir.path(), "", 0);
    let sink = Arc::new(StubHomeRequestSink {
        healthy: true,
        pushed: Mutex::new(Vec::new()),
    });
    logger.bind_home_sink(sink.clone());
    assert!(logger.log_request(&record(), false).unwrap().is_none());
    assert_eq!(dir.path().read_dir().unwrap().count(), 0);
    let pushed = sink.pushed.lock().unwrap();
    assert_eq!(pushed.len(), 1);
    assert_eq!(pushed[0].request_id, "req-1");
    assert_eq!(pushed[0].headers["Authorization"][0], "Bearer secret");
    assert!(!pushed[0].request_log.is_empty());
}

#[test]
fn detailed_source_is_forwarded_and_cleaned() {
    let dir = tempfile::tempdir().unwrap();
    let logger = FileRequestLogger::new(true, dir.path(), "", 0);
    let sink = Arc::new(StubHomeRequestSink {
        healthy: true,
        pushed: Mutex::new(Vec::new()),
    });
    logger.bind_home_sink(sink.clone());
    let source = logger.new_file_body_source("websocket").unwrap();
    source.append_part(b"Event: websocket.request\n{}").unwrap();
    let paths = source.paths();
    let headers = BTreeMap::from([("Upgrade".into(), vec!["websocket".into()])]);
    let details = DetailedRequestLog {
        url: "/v1/responses/ws",
        method: "GET",
        request_headers: &headers,
        request_body: b"",
        status_code: 101,
        response_headers: &headers,
        response: b"",
        websocket_timeline: b"",
        websocket_timeline_source: Some(&source),
        api_request: b"",
        api_request_source: None,
        api_response: b"",
        api_response_source: None,
        api_websocket_timeline: b"",
        api_websocket_timeline_source: None,
        api_response_errors: &[],
        force: false,
        request_id: "ws-1",
        request_timestamp: SystemTime::UNIX_EPOCH,
        api_response_timestamp: None,
    };
    assert!(logger.log_detailed(&details).unwrap().is_none());
    assert!(paths.iter().all(|path| !path.exists()));
    let pushed = sink.pushed.lock().unwrap();
    assert!(String::from_utf8_lossy(&pushed[0].request_log).contains("Event: websocket.request"));
}

#[test]
fn streaming_home_writer_preserves_request_id_and_drains_chunks() {
    let dir = tempfile::tempdir().unwrap();
    let logger = FileRequestLogger::new(true, dir.path(), "", 0);
    let sink = Arc::new(StubHomeRequestSink {
        healthy: true,
        pushed: Mutex::new(Vec::new()),
    });
    logger.bind_home_sink(sink.clone());
    let mut writer = logger
        .log_streaming_request("/v1/responses", "POST", &BTreeMap::new(), b"{}", "stream-1")
        .unwrap()
        .unwrap();
    writer.write_status(200, &BTreeMap::new());
    writer.write_chunk_async(b"data: ok\n\n");
    let outcome = writer.close().unwrap();
    assert!(outcome.path.as_os_str().is_empty());
    let pushed = sink.pushed.lock().unwrap();
    assert_eq!(pushed[0].request_id, "stream-1");
    assert!(String::from_utf8_lossy(&pushed[0].request_log).contains("data: ok"));
}

#[test]
fn forced_error_stays_local_when_regular_logging_is_disabled() {
    let dir = tempfile::tempdir().unwrap();
    let logger = FileRequestLogger::new(false, dir.path(), "", 2);
    let sink = Arc::new(StubHomeRequestSink {
        healthy: true,
        pushed: Mutex::new(Vec::new()),
    });
    logger.bind_home_sink(sink.clone());
    assert!(logger.log_request(&record(), true).unwrap().is_some());
    assert!(sink.pushed.lock().unwrap().is_empty());
}
