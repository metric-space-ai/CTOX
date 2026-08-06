// ref: sdk/cliproxy/auth/scheduler_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// Port-Note: instance-owned strategy, eligibility, cooldown and mixed-provider behavior
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::{HashMap, HashSet};

use super::{
    AccountCandidate, AccountSelectionError, AuthScheduler, CooldownStateRecord,
    SchedulerPickOptions, SchedulerStrategy,
};

pub(super) fn candidate(id: &str, provider: &str, priority: i32, weight: i64) -> AccountCandidate {
    AccountCandidate {
        auth_id: id.into(),
        provider: provider.into(),
        priority,
        weight,
        websocket_enabled: false,
        supported_models: Vec::new(),
        disabled: false,
    }
}

pub(super) fn cooldown(id: &str, model: Option<&str>, until: i64) -> CooldownStateRecord {
    CooldownStateRecord {
        provider: "codex".into(),
        auth_id: id.into(),
        model: model.map(str::to_owned),
        status: "cooling".into(),
        next_retry_after_ms: Some(until),
        reason: "quota".into(),
        quota: Default::default(),
        last_error: None,
        updated_at_ms: 1,
    }
}

#[test]
fn strategies_preserve_priority_and_expected_rotation() {
    let mut candidates = vec![
        candidate("low", "codex", 0, 100),
        candidate("a", "codex", 5, 3),
        candidate("b", "codex", 5, 1),
    ];
    let fill = AuthScheduler::new(SchedulerStrategy::FillFirst);
    assert_eq!(
        fill.pick_single("codex", None, 0, &candidates, &[], &Default::default())
            .unwrap()
            .auth_id,
        "a"
    );
    let round = AuthScheduler::new(SchedulerStrategy::RoundRobin);
    let ids = (0..3)
        .map(|_| {
            round
                .pick_single("codex", None, 0, &candidates, &[], &Default::default())
                .unwrap()
                .auth_id
        })
        .collect::<Vec<_>>();
    assert_eq!(ids, ["a", "b", "a"]);
    let weighted = AuthScheduler::new(SchedulerStrategy::WeightedRoundRobin);
    let mut counts = HashMap::new();
    for _ in 0..40 {
        *counts
            .entry(
                weighted
                    .pick_single("codex", None, 0, &candidates, &[], &Default::default())
                    .unwrap()
                    .auth_id,
            )
            .or_insert(0) += 1;
    }
    assert_eq!(counts["a"], 30);
    assert_eq!(counts["b"], 10);
    candidates[1].weight = 1;
    assert!(weighted
        .pick_single("codex", None, 0, &candidates, &[], &Default::default())
        .is_ok());
}

#[test]
fn pinned_tried_model_and_websocket_filters_are_applied_before_priority() {
    let scheduler = AuthScheduler::new(SchedulerStrategy::RoundRobin);
    let mut http = candidate("http", "codex", 10, 1);
    http.supported_models = vec!["gpt-5".into()];
    let mut websocket = candidate("ws", "codex", 0, 1);
    websocket.websocket_enabled = true;
    websocket.supported_models = vec!["gpt-5".into()];
    let options = SchedulerPickOptions {
        prefer_websocket: true,
        ..Default::default()
    };
    assert_eq!(
        scheduler
            .pick_single(
                "codex",
                Some("gpt-5(high)"),
                0,
                &[http.clone(), websocket.clone()],
                &[],
                &options
            )
            .unwrap()
            .auth_id,
        "ws"
    );
    let pinned = SchedulerPickOptions {
        pinned_auth_id: Some("http".into()),
        prefer_websocket: true,
        tried_auth_ids: HashSet::new(),
    };
    assert_eq!(
        scheduler
            .pick_single(
                "codex",
                Some("gpt-5"),
                0,
                &[http.clone(), websocket],
                &[],
                &pinned
            )
            .unwrap()
            .auth_id,
        "http"
    );
    let tried = SchedulerPickOptions {
        tried_auth_ids: HashSet::from(["http".into()]),
        ..Default::default()
    };
    assert_eq!(
        scheduler.pick_single("codex", Some("gpt-5"), 0, &[http], &[], &tried),
        Err(AccountSelectionError::NotFound)
    );
}

#[test]
fn cooldown_reports_earliest_retry_and_promotes_at_deadline() {
    let scheduler = AuthScheduler::new(SchedulerStrategy::RoundRobin);
    let candidates = [candidate("a", "codex", 0, 1), candidate("b", "codex", 0, 1)];
    let states = [cooldown("a", None, 5_000), cooldown("b", None, 4_000)];
    assert_eq!(
        scheduler.pick_single(
            "codex",
            None,
            3_000,
            &candidates,
            &states,
            &Default::default()
        ),
        Err(AccountSelectionError::Cooldown {
            retry_after_ms: 4_000
        })
    );
    assert!(scheduler
        .pick_single(
            "codex",
            None,
            4_000,
            &candidates,
            &states,
            &Default::default()
        )
        .is_ok());
}

#[test]
fn mixed_rotation_is_scoped_to_normalized_provider_set() {
    let scheduler = AuthScheduler::new(SchedulerStrategy::RoundRobin);
    let candidates = [
        candidate("g1", "gemini", 0, 1),
        candidate("g2", "gemini", 0, 1),
        candidate("c1", "claude", 0, 1),
    ];
    let providers = vec![" GEMINI ".into(), "claude".into(), "gemini".into()];
    let picks = (0..3)
        .map(|_| {
            scheduler
                .pick_mixed(&providers, None, 0, &candidates, &[], &Default::default())
                .unwrap()
                .candidate
                .auth_id
        })
        .collect::<Vec<_>>();
    assert_eq!(picks, ["g1", "g2", "c1"]);
}
