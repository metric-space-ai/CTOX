// ref: sdk/cliproxy/executionregistry/observation_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use super::{Registry, ScopeSpec};
use chrono::{TimeZone, Utc};

#[test]
fn freeze_in_flight_waits_for_pending_barrier_and_copies_scopes() {
    let registry = Registry::new();
    let pending = registry.begin_dispatch().unwrap();
    registry.observe_barrier(14);

    let before = registry.freeze_in_flight(Utc.timestamp_opt(12, 0).unwrap());
    assert_eq!(before.barrier_revision, 0);

    let scope = registry
        .install(
            &pending,
            ScopeSpec {
                request_id: "req-a".into(),
                credential_id: "cred".into(),
                model: "gpt-5".into(),
                kind: "http".into(),
                started_at: Utc.timestamp_opt(10, 0).unwrap(),
                accounted: true,
            },
        )
        .unwrap();
    let mut after = registry.freeze_in_flight(Utc.timestamp_opt(13, 0).unwrap());
    assert_eq!(after.barrier_revision, 14);
    assert_eq!(after.executions.len(), 1);
    assert!(after.executions[0].accounted);
    after.executions[0].request_id = "mutated".into();

    let copied = registry.freeze_in_flight(Utc.timestamp_opt(13, 0).unwrap());
    assert_eq!(copied.executions[0].request_id, "req-a");
    scope.end("completed");
    let ended = registry.freeze_in_flight(Utc.timestamp_opt(14, 0).unwrap());
    assert!(ended.executions.is_empty());
    assert!(ended.revision > after.revision);
}
