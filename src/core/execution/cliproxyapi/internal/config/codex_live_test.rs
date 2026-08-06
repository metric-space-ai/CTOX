// ref: internal/config/codex_live_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use super::codex_live::{CodexLiveIceServer, CodexLiveMediaRelayConfig};

#[test]
fn parses_validates_and_redacts_turn_credentials() {
    let relay: CodexLiveMediaRelayConfig = serde_yaml::from_str(
        r#"enabled: true
max-sessions: 64
disable-private-remote-ips: true
public-ip: "203.0.113.10"
udp-port-min: 40000
udp-port-max: 40150
ice-servers:
  - urls: ["stun:stun.example.com:3478"]
  - urls: ["turn:turn.example.com:3478?transport=udp"]
    username: "relay-user"
    credential: "relay-secret"
"#,
    )
    .unwrap();
    assert_eq!(relay.effective_max_sessions(), 64);
    assert!(relay.disable_private_remote_ips);
    assert_eq!(relay.ice_servers.len(), 2);
    relay.validate().unwrap();
    let encoded = serde_json::to_string(&relay).unwrap();
    for sensitive in ["relay-secret", "credential", "relay-user", "username"] {
        assert!(
            !encoded.contains(sensitive),
            "leaked {sensitive}: {encoded}"
        );
    }
}

#[test]
fn migrates_legacy_private_ip_setting_and_rejects_conflict() {
    for (source, expected) in [
        ("allow-private-remote-ips: true\n", false),
        ("allow-private-remote-ips: false\n", true),
        ("enabled: true\n", false),
    ] {
        let relay: CodexLiveMediaRelayConfig = serde_yaml::from_str(source).unwrap();
        assert_eq!(relay.disable_private_remote_ips, expected);
    }
    assert!(serde_yaml::from_str::<CodexLiveMediaRelayConfig>(
        "allow-private-remote-ips: true\ndisable-private-remote-ips: false\n"
    )
    .is_err());
}

#[test]
fn rejects_invalid_values() {
    let invalid = [
        CodexLiveMediaRelayConfig {
            enabled: true,
            max_sessions: -1,
            ..Default::default()
        },
        CodexLiveMediaRelayConfig {
            enabled: true,
            public_ip: "not-an-ip".into(),
            ..Default::default()
        },
        CodexLiveMediaRelayConfig {
            enabled: true,
            udp_port_min: 40_000,
            ..Default::default()
        },
        CodexLiveMediaRelayConfig {
            enabled: true,
            udp_port_min: 40_100,
            udp_port_max: 40_000,
            ..Default::default()
        },
        CodexLiveMediaRelayConfig {
            enabled: true,
            max_sessions: 2,
            udp_port_min: 40_000,
            udp_port_max: 40_002,
            ..Default::default()
        },
        CodexLiveMediaRelayConfig {
            enabled: true,
            ice_servers: vec![CodexLiveIceServer {
                username: "user".into(),
                ..Default::default()
            }],
            ..Default::default()
        },
        CodexLiveMediaRelayConfig {
            enabled: true,
            ice_servers: vec![CodexLiveIceServer {
                urls: vec!["https://example.com".into()],
                ..Default::default()
            }],
            ..Default::default()
        },
    ];
    assert!(invalid.iter().all(|relay| relay.validate().is_err()));
}
