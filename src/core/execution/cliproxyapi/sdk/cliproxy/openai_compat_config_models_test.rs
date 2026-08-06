// ref: sdk/cliproxy/openai_compat_config_models_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use super::service_models::{
    build_openai_compatibility_config_models, ConfiguredModel, OpenAiCompatibilityConfig,
};
use crate::internal::registry::OPENAI_IMAGE_MODEL_TYPE;

#[test]
fn compatibility_models_normalize_modalities_and_image_type() {
    let models = build_openai_compatibility_config_models(&OpenAiCompatibilityConfig {
        name: "mimo".into(),
        models: vec![
            ConfiguredModel {
                name: "vision".into(),
                alias: "mimo-v2.5-pro".into(),
                display_name: "Mimo Vision".into(),
                input_modalities: vec!["TEXT".into(), "image".into(), "image".into()],
                ..ConfiguredModel::default()
            },
            ConfiguredModel {
                name: "image".into(),
                alias: "compat-image".into(),
                image: true,
                ..ConfiguredModel::default()
            },
        ],
        ..OpenAiCompatibilityConfig::default()
    });
    assert_eq!(models[0].supported_input_modalities, ["text", "image"]);
    assert!(models[0].thinking.is_some());
    assert_eq!(models[1].provider_type, OPENAI_IMAGE_MODEL_TYPE);
    assert!(models[1].thinking.is_none());
}
