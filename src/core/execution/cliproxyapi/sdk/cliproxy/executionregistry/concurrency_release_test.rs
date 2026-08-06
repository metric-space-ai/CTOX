// ref: sdk/cliproxy/executionregistry/concurrency_release_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use super::{Registry, ReleaseGroup, Scope, ScopeSpec};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

#[derive(Clone, Default)]
struct RecordingReleaseSink {
    sequences: Arc<Mutex<HashMap<ReleaseGroup, i64>>>,
}

impl RecordingReleaseSink {
    fn mark_dirty(&self, group: ReleaseGroup, sequence: i64) {
        let mut sequences = self.sequences.lock().unwrap();
        let current = sequences.entry(group).or_default();
        *current = (*current).max(sequence);
    }

    fn sequence(&self, credential_id: &str, model: &str) -> i64 {
        self.sequences
            .lock()
            .unwrap()
            .get(&ReleaseGroup {
                credential_id: credential_id.into(),
                model: model.into(),
            })
            .copied()
            .unwrap_or_default()
    }
}

fn install_accounted_scope(registry: &Registry, credential_id: &str, model: &str) -> Scope {
    let pending = registry.begin_dispatch().unwrap();
    registry
        .install(
            &pending,
            ScopeSpec {
                credential_id: credential_id.into(),
                model: model.into(),
                accounted: true,
                ..ScopeSpec::default()
            },
        )
        .unwrap()
}

#[test]
fn registry_end_marks_one_dirty_group() {
    let sink = RecordingReleaseSink::default();
    let registry = Registry::new();
    let sink_callback = sink.clone();
    registry.set_release_callback(move |group, sequence| {
        sink_callback.mark_dirty(group, sequence);
    });

    let scope = install_accounted_scope(&registry, "cred-1", "gpt");
    scope.end("complete");
    scope.end("duplicate");
    assert_eq!(sink.sequence("cred-1", "gpt"), 1);
}

#[test]
fn unaccounted_scope_does_not_release() {
    let sink = RecordingReleaseSink::default();
    let registry = Registry::new();
    let sink_callback = sink.clone();
    registry.set_release_callback(move |group, sequence| {
        sink_callback.mark_dirty(group, sequence);
    });
    let pending = registry.begin_dispatch().unwrap();
    let scope = registry
        .install(
            &pending,
            ScopeSpec {
                credential_id: "cred-1".into(),
                model: "gpt".into(),
                accounted: false,
                ..ScopeSpec::default()
            },
        )
        .unwrap();
    scope.end("observation_complete");
    assert_eq!(sink.sequence("cred-1", "gpt"), 0);
}

#[test]
fn set_release_sink_replays_existing_sequences() {
    let registry = Registry::new();
    install_accounted_scope(&registry, "cred-1", "gpt").end("complete");

    let sink = RecordingReleaseSink::default();
    let sink_callback = sink.clone();
    registry.set_release_callback(move |group, sequence| {
        sink_callback.mark_dirty(group, sequence);
    });
    assert_eq!(sink.sequence("cred-1", "gpt"), 1);
}
