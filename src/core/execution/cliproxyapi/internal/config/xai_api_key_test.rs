// ref: internal/config/xai_api_key_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use super::parse::parse_provider_compat_config;

#[test]
fn parses_xai_config_default_and_enabled() {
    assert!(
        !parse_provider_compat_config(b"{}")
            .unwrap()
            .xai
            .inject_x_search
    );
    assert!(
        parse_provider_compat_config(b"xai:\n  inject-x-search: true\n")
            .unwrap()
            .xai
            .inject_x_search
    );
}

#[test]
fn xai_api_key_matches_codex_shape_and_normalization() {
    let config = parse_provider_compat_config(
        br#"xai-api-key:
  - api-key: " xai-key "
    priority: 3
    weight: 5
    prefix: " team-xai "
    base-url: " https://api.x.ai/v1 "
    websockets: true
    proxy-url: " http://proxy.local "
    headers:
      X-Custom: value
    models:
      - name: grok-4.5
        alias: grok-latest
        display-name: Grok Latest
        force-mapping: true
    excluded-models: [" grok-3-* "]
    disable-cooling: true
  - api-key: dropped
    base-url: " "
"#,
    )
    .unwrap();
    assert_eq!(config.xai_api_key.len(), 1);
    let entry = &config.xai_api_key[0];
    assert_eq!(entry.api_key, " xai-key ");
    assert_eq!(entry.priority, 3);
    assert_eq!(entry.weight, Some(5));
    assert_eq!(entry.prefix, "team-xai");
    assert_eq!(entry.base_url, "https://api.x.ai/v1");
    assert!(entry.websockets);
    assert_eq!(entry.proxy_url, " http://proxy.local ");
    assert!(entry.disable_cooling);
    assert_eq!(
        entry.headers.get("X-Custom").map(String::as_str),
        Some("value")
    );
    assert_eq!(entry.models[0].display_name, "Grok Latest");
    assert!(entry.models[0].force_mapping);
    assert_eq!(entry.excluded_models, ["grok-3-*"]);
}
