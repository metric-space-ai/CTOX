// ref: internal/home/plugin_status_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use super::plugin_status::*;
use crate::internal::homeplugins::{PluginInstallStatus, SyncReport};
use crate::sdk::pluginstore::Platform;
use chrono::{DateTime, Utc};
use std::sync::Mutex;
use std::time::Instant;

struct Sink {
    payload: Mutex<Vec<u8>>,
    fail: bool,
}
impl PluginStatusSink for Sink {
    fn push_plugin_status(
        &self,
        payload: &[u8],
        deadline: Instant,
    ) -> Result<(), PluginStatusError> {
        assert!(deadline > Instant::now());
        if self.fail {
            Err(PluginStatusError("push failed".into()))
        } else {
            *self.payload.lock().unwrap() = payload.into();
            Ok(())
        }
    }
}
fn report() -> SyncReport {
    let now = DateTime::<Utc>::from_timestamp(1_700_000_000, 0).unwrap();
    SyncReport {
        schema_version: 1,
        task_id: 0,
        task: "plugin-sync".into(),
        node_id: String::new(),
        status: "success".into(),
        phase: "install".into(),
        ok: true,
        started_at: now,
        finished_at: Some(now),
        updated_at: now,
        platform: Platform::default(),
        plugins: vec![PluginInstallStatus {
            id: "sample".into(),
            install_status: "installed".into(),
            ..PluginInstallStatus::default()
        }],
        error: String::new(),
    }
}

#[test]
fn report_sets_trimmed_node_and_fresh_time_then_pushes() {
    let sink = Sink {
        payload: Mutex::new(Vec::new()),
        fail: false,
    };
    let old = report().updated_at;
    report_plugin_status(&sink, " node-1 ", report()).unwrap();
    let payload: SyncReport = serde_json::from_slice(&sink.payload.lock().unwrap()).unwrap();
    assert_eq!(payload.node_id, "node-1");
    assert!(payload.updated_at > old);
    assert!(payload.ok);
    assert_eq!(payload.plugins.len(), 1);
}

#[test]
fn empty_report_is_valid_and_node_id_is_required() {
    let sink = Sink {
        payload: Mutex::new(Vec::new()),
        fail: false,
    };
    let mut empty = report();
    empty.plugins.clear();
    report_plugin_status(&sink, "node", empty).unwrap();
    let payload: SyncReport = serde_json::from_slice(&sink.payload.lock().unwrap()).unwrap();
    assert!(payload.plugins.is_empty());
    assert!(report_plugin_status(&sink, " ", report())
        .unwrap_err()
        .0
        .contains("node id"));
}

#[test]
fn push_error_is_propagated() {
    let sink = Sink {
        payload: Mutex::new(Vec::new()),
        fail: true,
    };
    assert_eq!(
        report_plugin_status(&sink, "node", report()).unwrap_err(),
        PluginStatusError("push failed".into())
    );
}
