// ref: sdk/cliproxy/auth/selector_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// Port-Note: selector facades share instance-owned eligibility and smooth-weight state
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::HashMap;
use std::sync::Arc;

use super::{FillFirstSelector, RoundRobinSelector, WeightedRoundRobinSelector};

#[test]
fn fill_first_and_round_robin_are_deterministic() {
    let candidates = [
        super::scheduler_test::candidate("b", "codex", 0, 1),
        super::scheduler_test::candidate("a", "codex", 0, 1),
    ];
    let fill = FillFirstSelector::default();
    assert_eq!(
        fill.pick("codex", None, 0, &candidates, &[])
            .unwrap()
            .auth_id,
        "a"
    );
    assert_eq!(
        fill.pick("codex", None, 0, &candidates, &[])
            .unwrap()
            .auth_id,
        "a"
    );
    let round = RoundRobinSelector::default();
    assert_eq!(
        round
            .pick("codex", None, 0, &candidates, &[])
            .unwrap()
            .auth_id,
        "a"
    );
    assert_eq!(
        round
            .pick("codex", None, 0, &candidates, &[])
            .unwrap()
            .auth_id,
        "b"
    );
}

#[test]
fn weighted_selector_distributes_and_resets_changed_vector() {
    let selector = WeightedRoundRobinSelector::default();
    let mut candidates = vec![
        super::scheduler_test::candidate("a", "codex", 0, 3),
        super::scheduler_test::candidate("b", "codex", 0, 1),
        super::scheduler_test::candidate("excluded", "codex", 0, 0),
    ];
    let mut counts = HashMap::new();
    for _ in 0..40 {
        *counts
            .entry(
                selector
                    .pick("codex", None, 0, &candidates, &[])
                    .unwrap()
                    .auth_id,
            )
            .or_insert(0) += 1;
    }
    assert_eq!(counts.get("a"), Some(&30));
    assert_eq!(counts.get("b"), Some(&10));
    assert!(!counts.contains_key("excluded"));
    candidates[0].weight = 1;
    let picks = (0..4)
        .map(|_| {
            selector
                .pick("codex", None, 0, &candidates, &[])
                .unwrap()
                .auth_id
        })
        .collect::<Vec<_>>();
    assert_eq!(picks, ["a", "b", "a", "b"]);
}

#[test]
fn concurrent_round_robin_keeps_one_instance_cursor_without_panics() {
    let selector = Arc::new(RoundRobinSelector::default());
    let candidates = Arc::new(vec![
        super::scheduler_test::candidate("a", "codex", 0, 1),
        super::scheduler_test::candidate("b", "codex", 0, 1),
    ]);
    let handles = (0..8)
        .map(|_| {
            let selector = selector.clone();
            let candidates = candidates.clone();
            std::thread::spawn(move || {
                (0..100)
                    .map(|_| {
                        selector
                            .pick("codex", Some("gpt"), 0, &candidates, &[])
                            .unwrap()
                            .auth_id
                    })
                    .collect::<Vec<_>>()
            })
        })
        .collect::<Vec<_>>();
    let mut counts = HashMap::new();
    for handle in handles {
        for id in handle.join().unwrap() {
            *counts.entry(id).or_insert(0) += 1;
        }
    }
    assert_eq!(counts.get("a"), Some(&400));
    assert_eq!(counts.get("b"), Some(&400));
}
