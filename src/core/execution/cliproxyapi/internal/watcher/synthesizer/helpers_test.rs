// ref: internal/watcher/synthesizer/helpers_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use super::context::SynthesizedAuth;
use super::helpers::{
    add_config_headers_to_attrs, apply_auth_excluded_models_meta, StableIdGenerator,
};
use std::collections::BTreeMap;
#[test]
fn stable_id_is_deterministic_and_collision_safe() {
    let mut first = StableIdGenerator::default();
    let one = first.next("claude", &["path"]);
    let two = first.next("claude", &["path"]);
    assert_ne!(one.0, two.0);
    assert_eq!(one.1, two.1);
    let mut second = StableIdGenerator::default();
    assert_eq!(one, second.next("claude", &["path"]));
}
#[test]
fn excluded_models_merge_and_headers_are_namespaced() {
    let mut auth = SynthesizedAuth::default();
    apply_auth_excluded_models_meta(&mut auth, &[" B ".into()], &["a".into(), "b".into()]);
    assert_eq!(auth.excluded_models, vec!["a", "b"]);
    let mut attrs = BTreeMap::new();
    add_config_headers_to_attrs(
        &BTreeMap::from([
            (" X-Test ".into(), " value ".into()),
            (" ".into(), "skip".into()),
        ]),
        &mut attrs,
    );
    assert_eq!(
        attrs,
        BTreeMap::from([("header:X-Test".into(), "value".into())])
    );
}
