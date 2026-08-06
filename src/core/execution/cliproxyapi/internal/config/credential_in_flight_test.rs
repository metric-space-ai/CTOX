// ref: internal/config/credential_in_flight_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only
use super::credential_in_flight::*;
#[test]
fn defaults_validate_and_parse() {
    let cfg = CredentialInFlightConfig::default();
    assert!(cfg.validate().is_ok());
    let (d, s, r) = cfg.durations().unwrap();
    assert_eq!(d.as_secs(), 2);
    assert_eq!(s.as_secs(), 10);
    assert_eq!(r.as_secs(), 60)
}
#[test]
fn invalid_timing_capacity_and_closed_schema_fail() {
    let cfg = CredentialInFlightConfig {
        stale_after: "5s".into(),
        ..CredentialInFlightConfig::default()
    };
    assert!(cfg.validate().is_err());
    let cfg = CredentialInFlightConfig {
        max_part_count: 1,
        ..CredentialInFlightConfig::default()
    };
    assert!(cfg.validate().is_err());
    assert!(serde_json::from_str::<CredentialInFlightConfig>(
        r#"{"snapshot-interval":"2s","unknown":1}"#
    )
    .is_err())
}
