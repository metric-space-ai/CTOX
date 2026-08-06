// ref: internal/config/home_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use super::CliproxyRuntimeConfig;

#[test]
fn yaml_home_control_plane_is_rejected_instead_of_silently_ignored() {
    let source = r#"
home:
  enabled: true
  host: home.example.com
  port: 444
  disable-cluster-discovery: true
  tls:
    enable: true
    server-name: home.example.com
    ca-cert: C:/certs/ca.pem
    insecure-skip-verify: true
"#;

    let error = serde_yaml::from_str::<CliproxyRuntimeConfig>(source).unwrap_err();
    assert!(error.to_string().contains("unknown field `home`"));
}

#[test]
fn json_home_control_plane_is_rejected_by_the_same_closed_schema() {
    let source = r#"{
        "request_timeout_ms": 30000,
        "home": {
            "enabled": true,
            "host": "home.example.com",
            "port": 444,
            "tls": {"enable": true, "client_key": "secret"}
        }
    }"#;

    let error = serde_json::from_str::<CliproxyRuntimeConfig>(source).unwrap_err();
    assert!(error.to_string().contains("unknown field `home`"));
    assert!(!error.to_string().contains("secret"));
}
