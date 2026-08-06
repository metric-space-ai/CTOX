// ref: sdk/cliproxy/config_model_max_context_length_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use super::service_models::{build_config_models, ConfiguredModel};

#[test]
fn configured_models_propagate_max_context_length() {
    let model = build_config_models(
        &[ConfiguredModel {
            name: "upstream".into(),
            alias: "alias".into(),
            max_context_length: 1_048_576,
            ..ConfiguredModel::default()
        }],
        "owner",
        "provider",
    )
    .remove(0);
    assert_eq!(model.context_length, 1_048_576);
    assert_eq!(model.max_context_length, 1_048_576);
}
