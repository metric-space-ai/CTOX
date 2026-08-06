// ref: internal/logging/home_app_log_forwarder_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use super::global_logger::{LogEntry, LogLevel};
use super::home_app_log_forwarder::{HomeAppLogForwarder, HomeAppLogPayload, HomeAppLogSink};
use std::io;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, SystemTime};

struct StubSink {
    healthy: bool,
    unsupported: bool,
    pushed: Mutex<Vec<Vec<u8>>>,
}
impl StubSink {
    fn new(healthy: bool) -> Self {
        Self {
            healthy,
            unsupported: false,
            pushed: Mutex::new(Vec::new()),
        }
    }
}
impl HomeAppLogSink for StubSink {
    fn heartbeat_ok(&self) -> bool {
        self.healthy
    }
    fn push_app_log(&self, payload: &[u8]) -> io::Result<()> {
        if self.unsupported {
            return Err(io::Error::other("ERR unsupported key"));
        }
        self.pushed.lock().unwrap().push(payload.to_vec());
        Ok(())
    }
}

fn wait_count(sink: &StubSink, count: usize) {
    for _ in 0..100 {
        if sink.pushed.lock().unwrap().len() >= count {
            return;
        }
        thread::sleep(Duration::from_millis(2));
    }
}

#[test]
fn bound_healthy_owner_receives_formatted_payload() {
    let forwarder = HomeAppLogForwarder::start(4);
    let sink = Arc::new(StubSink::new(true));
    forwarder.bind(sink.clone());
    let mut entry = LogEntry::new(LogLevel::Debug, "debug details", SystemTime::UNIX_EPOCH);
    entry.fields.insert("request_id".into(), "req-app-1".into());
    forwarder.fire(&entry);
    wait_count(&sink, 1);
    let raw = sink.pushed.lock().unwrap()[0].clone();
    let payload: HomeAppLogPayload = serde_json::from_slice(&raw).unwrap();
    assert_eq!(payload.level, "debug");
    assert_eq!(payload.request_id, "req-app-1");
    assert!(payload.line.contains("debug details"));
    forwarder.stop();
}

#[test]
fn rebind_and_deactivate_are_owner_scoped() {
    let forwarder = HomeAppLogForwarder::start(4);
    let first = Arc::new(StubSink::new(true));
    let second = Arc::new(StubSink::new(true));
    forwarder.bind(first.clone());
    forwarder.bind(second.clone());
    let first_trait: Arc<dyn HomeAppLogSink> = first.clone();
    forwarder.deactivate(&first_trait);
    forwarder.fire(&LogEntry::new(
        LogLevel::Info,
        "new owner",
        SystemTime::UNIX_EPOCH,
    ));
    wait_count(&second, 1);
    assert_eq!(first.pushed.lock().unwrap().len(), 0);
    let second_trait: Arc<dyn HomeAppLogSink> = second.clone();
    forwarder.deactivate(&second_trait);
    forwarder.fire(&LogEntry::new(
        LogLevel::Info,
        "detached",
        SystemTime::UNIX_EPOCH,
    ));
    thread::sleep(Duration::from_millis(10));
    assert_eq!(second.pushed.lock().unwrap().len(), 1);
}

#[test]
fn unhealthy_unbound_and_placeholder_ids_are_not_forwarded() {
    let forwarder = HomeAppLogForwarder::start(1);
    forwarder.fire(&LogEntry::new(
        LogLevel::Info,
        "unbound",
        SystemTime::UNIX_EPOCH,
    ));
    let sink = Arc::new(StubSink::new(false));
    forwarder.bind(sink.clone());
    let mut entry = LogEntry::new(LogLevel::Info, "down", SystemTime::UNIX_EPOCH);
    entry.fields.insert("request_id".into(), "--------".into());
    forwarder.fire(&entry);
    assert!(sink.pushed.lock().unwrap().is_empty());
}

#[test]
fn unsupported_owner_disables_forwarding() {
    let forwarder = HomeAppLogForwarder::start(1);
    let sink = Arc::new(StubSink {
        healthy: true,
        unsupported: true,
        pushed: Mutex::new(Vec::new()),
    });
    forwarder.bind(sink);
    forwarder.fire(&LogEntry::new(
        LogLevel::Info,
        "legacy",
        SystemTime::UNIX_EPOCH,
    ));
    for _ in 0..100 {
        if !forwarder.is_enabled() {
            break;
        }
        thread::sleep(Duration::from_millis(2));
    }
    assert!(!forwarder.is_enabled());
}
