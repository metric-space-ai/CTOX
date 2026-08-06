// ref: internal/watcher/synthesizer/config_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use super::config::ConfigSynthesizer;
use super::context::SynthesisContext;
use super::interface::AuthSynthesizer;
use crate::internal::watcher::config_reload::{
    ApiKeyConfig, ModelRoute, OpenAiCompatibility, WatcherConfig,
};
use crate::internal::watcher::NativeWatchFilesystem;
use std::collections::BTreeMap;
use std::sync::Arc;

#[test]
fn config_synthesizer_covers_providers_weights_headers_models_and_stable_ids() {
    let mut config = WatcherConfig::default();
    config.providers.insert(
        "claude".into(),
        vec![ApiKeyConfig {
            api_key: "key".into(),
            id: "primary".into(),
            weight: Some(2),
            headers: BTreeMap::from([("X-Test".into(), "yes".into())]),
            models: vec![ModelRoute {
                name: "claude-4".into(),
                alias: "sonnet".into(),
                ..ModelRoute::default()
            }],
            ..ApiKeyConfig::default()
        }],
    );
    config.openai_compatibility.push(OpenAiCompatibility {
        name: "local".into(),
        base_url: "https://example.test/v1".into(),
        api_keys: vec!["compat-key".into()],
        models: vec![ModelRoute {
            name: "model".into(),
            ..ModelRoute::default()
        }],
        ..OpenAiCompatibility::default()
    });
    let dir = tempfile::tempdir().unwrap();
    let context = SynthesisContext {
        config: &config,
        auth_dir: dir.path(),
        files: vec![],
        filesystem: Arc::new(NativeWatchFilesystem),
        parser: None,
    };
    let first = ConfigSynthesizer::new().synthesize(&context).unwrap();
    let second = ConfigSynthesizer::new().synthesize(&context).unwrap();
    assert_eq!(first, second);
    assert_eq!(first.len(), 2);
    assert_eq!(first[0].weight, Some(2));
    assert_eq!(first[0].attributes["header:X-Test"], "yes");
    assert!(!first[1].attributes["models_hash"].is_empty());
}

#[test]
fn invalid_or_empty_credentials_are_skipped() {
    let mut config = WatcherConfig::default();
    config.providers.insert(
        "codex".into(),
        vec![
            ApiKeyConfig::default(),
            ApiKeyConfig {
                api_key: "key".into(),
                weight: Some(crate::internal::credentialweight::MAX_CREDENTIAL_WEIGHT + 1),
                ..ApiKeyConfig::default()
            },
        ],
    );
    let dir = tempfile::tempdir().unwrap();
    let context = SynthesisContext {
        config: &config,
        auth_dir: dir.path(),
        files: vec![],
        filesystem: Arc::new(NativeWatchFilesystem),
        parser: None,
    };
    assert!(ConfigSynthesizer::new()
        .synthesize(&context)
        .unwrap()
        .is_empty());
}
