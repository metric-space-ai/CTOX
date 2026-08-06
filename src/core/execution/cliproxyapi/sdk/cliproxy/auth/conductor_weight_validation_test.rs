// ref: sdk/cliproxy/auth/conductor_weight_validation_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use super::conductor_availability_test::{auth, manager, register};

#[test]
fn register_rejects_invalid_typed_scheduler_weight_without_publishing_candidate() {
    let (manager, caps) = manager();
    caps.set("invalid", 1_000_001);
    assert!(register(&manager, auth("invalid", false)).is_err());
    assert!(manager.candidates().is_empty());
}

#[test]
fn nonpositive_weights_are_normalized_before_publication() {
    for (id, weight, published) in [("zero", 0, 0), ("positive", 2, 2), ("negative", -1, 0)] {
        let (manager, caps) = manager();
        caps.set(id, weight);
        register(&manager, auth(id, false)).unwrap();
        assert_eq!(manager.candidates()[0].weight, published);
    }
}
