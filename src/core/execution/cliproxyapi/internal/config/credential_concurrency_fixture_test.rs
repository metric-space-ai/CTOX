// ref: internal/config/credential_concurrency_fixture_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use std::time::Duration;

use super::credential_concurrency::*;

#[test]
fn wire_fixture_preserves_hot_duration_strings_and_validates() {
    let config: CredentialConcurrencyConfig = serde_yaml::from_str(
        r#"lifecycle-config-revision: 1
cpa-heartbeat-timeout: 3s
cpa-cancel-bound: 5s
reclaim-grace: 5s
cleanup-interval: 5s
release-flush-interval: 250ms
release-max-backoff: 2s
busy-retry-min: 250ms
busy-retry-max: 1s
max-limit: 1000000
"#,
    )
    .unwrap();
    assert_eq!(config.release_flush_interval, Duration::from_millis(250));
    assert_eq!(config.busy_retry_max, Duration::from_secs(1));
    validate_credential_concurrency(&config).unwrap();

    for (heartbeat, cpa) in [(3, 3), (20, 0)] {
        let mut invalid = config.clone();
        invalid.lifecycle_config_revision = 0;
        invalid.cpa_heartbeat_timeout = Duration::from_secs(cpa);
        assert!(validate_credential_concurrency_lifecycle(
            Duration::from_secs(heartbeat),
            &invalid
        )
        .is_err());
    }
}
