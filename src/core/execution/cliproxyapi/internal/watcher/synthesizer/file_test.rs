// ref: internal/watcher/synthesizer/file_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use super::context::SynthesisContext;
use super::file::{synthesize_auth_file, FileSynthesizer};
use super::interface::AuthSynthesizer;
use crate::internal::watcher::config_reload::WatcherConfig;
use crate::internal::watcher::NativeWatchFilesystem;
use std::sync::Arc;

#[test]
fn file_synthesizer_discovers_valid_json_and_uses_relative_stable_id() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("claude.json");
    std::fs::write(&path, br#"{"type":"claude","access_token":"secret","priority":2,"weight":3,"excluded_models":["old"]}"#).unwrap();
    let config = WatcherConfig::default();
    let context = SynthesisContext {
        config: &config,
        auth_dir: dir.path(),
        files: vec![path.clone()],
        filesystem: Arc::new(NativeWatchFilesystem),
        parser: None,
    };
    let auths = FileSynthesizer::new().synthesize(&context).unwrap();
    assert_eq!(auths.len(), 1);
    assert_eq!(auths[0].file_name, "claude.json");
    assert_eq!(auths[0].priority, 2);
    assert_eq!(auths[0].weight, Some(3));
    assert_eq!(auths, FileSynthesizer::new().synthesize(&context).unwrap());
}

#[test]
fn malformed_gemini_and_invalid_weight_files_are_ignored() {
    let dir = tempfile::tempdir().unwrap();
    let config = WatcherConfig::default();
    let context = SynthesisContext {
        config: &config,
        auth_dir: dir.path(),
        files: vec![],
        filesystem: Arc::new(NativeWatchFilesystem),
        parser: None,
    };
    assert!(synthesize_auth_file(&context, &dir.path().join("bad.json"), b"not-json").is_err());
    assert!(synthesize_auth_file(
        &context,
        &dir.path().join("gemini.json"),
        br#"{"type":"gemini"}"#
    )
    .unwrap()
    .is_empty());
    assert!(synthesize_auth_file(
        &context,
        &dir.path().join("bad-weight.json"),
        format!(
            r#"{{"type":"claude","weight":{}}}"#,
            crate::internal::credentialweight::MAX_CREDENTIAL_WEIGHT + 1
        )
        .as_bytes()
    )
    .unwrap()
    .is_empty());
}
