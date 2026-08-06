// ref: internal/config/model_display_name_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use super::ProviderCompatConfig;

#[test]
fn display_names_decode_for_every_compatible_provider() {
    let source = r#"codex-api-key: [{models: [{name: c, alias: c, display-name: Codex Name}]}]
xai-api-key: [{models: [{name: x, alias: x, display-name: xAI Name}]}]
claude-api-key: [{models: [{name: a, alias: a, display-name: Claude Name}]}]
gemini-api-key: [{models: [{name: g, alias: g, display-name: Gemini Name}]}]
vertex-api-key: [{api-key: k, models: [{name: v, alias: v, display-name: Vertex Name}]}]
openai-compatibility: [{models: [{name: o, alias: o, display-name: Compatibility Name}]}]
"#;
    let config: ProviderCompatConfig = serde_yaml::from_str(source).unwrap();
    assert_eq!(config.codex_api_key[0].models[0].display_name, "Codex Name");
    assert_eq!(config.xai_api_key[0].models[0].display_name, "xAI Name");
    assert_eq!(
        config.claude_api_key[0].models[0].display_name,
        "Claude Name"
    );
    assert_eq!(
        config.gemini_api_key[0].models[0].display_name,
        "Gemini Name"
    );
    assert_eq!(
        config.vertex_api_key[0].models[0].display_name,
        "Vertex Name"
    );
    assert_eq!(
        config.openai_compatibility[0].models[0].display_name,
        "Compatibility Name"
    );
}
