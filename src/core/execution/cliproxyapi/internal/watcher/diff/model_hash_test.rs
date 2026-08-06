// ref: internal/watcher/diff/model_hash_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use super::model_hash::{compute_excluded_models_hash, compute_models_hash};
use crate::internal::watcher::config_reload::ModelRoute;

#[test]
fn model_hash_tracks_order_duplicates_and_capabilities() {
    let base = ModelRoute {
        name: "gpt".into(),
        alias: "alias".into(),
        ..ModelRoute::default()
    };
    let mut changed = base.clone();
    changed.image = true;
    assert_ne!(
        compute_models_hash(std::slice::from_ref(&base)),
        compute_models_hash(&[changed])
    );
    assert_ne!(
        compute_models_hash(&[base.clone(), base.clone()]),
        compute_models_hash(std::slice::from_ref(&base))
    );
    assert_eq!(compute_models_hash(&[]), "");
}
#[test]
fn excluded_hash_normalizes_order_case_and_duplicates() {
    assert_eq!(
        compute_excluded_models_hash(&[" B ".into(), "a".into(), "a".into()]),
        compute_excluded_models_hash(&["A".into(), "b".into()])
    );
}
