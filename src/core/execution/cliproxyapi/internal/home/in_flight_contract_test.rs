// ref: internal/home/in_flight_contract_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use super::requests::*;
use serde::Deserialize;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Fixture {
    part: InFlightSnapshotFrame,
    overflow: InFlightSnapshotFrame,
}

fn fixture() -> &'static [u8] {
    br#"{
 "part":{"kind":"part","revision":7,"observed_at":"2026-01-01T00:00:00Z","barrier_revision":6,"part_index":0,"part_count":1,"details_truncated":true,"aggregates":[{"credential_id":"cred-a","model":"gpt","status":"accounted","count":2},{"credential_id":"cred-b","model":"gpt","status":"unaccounted","count":1}],"details":[{"request_id":"req-1","credential_id":"cred-a","model":"gpt","request_kind":"stream","started_at":"2026-01-01T00:00:00Z"}]},
 "overflow":{"kind":"overflow","revision":8,"observed_at":"2026-01-01T00:00:01Z","barrier_revision":7,"aggregate_group_count":100001}
}"#
}

#[test]
fn credential_in_flight_wire_contract_decodes_exact_fields() {
    let value: Fixture = serde_json::from_slice(fixture()).unwrap();
    assert_eq!(value.part.kind, InFlightFrameKind::Part);
    assert_eq!(value.part.part_index, Some(0));
    assert_eq!(
        value.part.aggregates[0].status,
        InFlightAccountedStatus::Accounted
    );
    assert_eq!(
        value.part.aggregates[1].status,
        InFlightAccountedStatus::Unaccounted
    );
    assert_eq!(value.overflow.kind, InFlightFrameKind::Overflow);
    assert_eq!(value.overflow.aggregate_group_count, 100001);
}

#[test]
fn credential_in_flight_wire_contract_rejects_owner_or_secret_extensions() {
    let text = std::str::from_utf8(fixture()).unwrap();
    for invalid in [
        text.replace(
            "\"kind\":\"part\"",
            "\"kind\":\"part\",\"node_id\":\"node\"",
        ),
        text.replace(
            "\"credential_id\":\"cred-a\"",
            "\"credential_id\":\"cred-a\",\"fingerprint\":\"owner\"",
        ),
        text.replace(
            "\"request_id\":\"req-1\"",
            "\"request_id\":\"req-1\",\"secret\":\"x\"",
        ),
        format!("{text} {{}}"),
    ] {
        assert!(serde_json::from_str::<Fixture>(&invalid).is_err());
    }
}
