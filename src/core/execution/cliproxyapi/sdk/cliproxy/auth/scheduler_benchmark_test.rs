// ref: sdk/cliproxy/auth/scheduler_benchmark_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// Port-Note: bounded high-cardinality regression probe replaces unstable unit benchmarks
// License: MIT (upstream); modifications AGPL-3.0-only

use super::{AuthScheduler, SchedulerStrategy};

#[test]
fn thousand_candidate_scheduler_probe_stays_deterministic() {
    let scheduler = AuthScheduler::new(SchedulerStrategy::RoundRobin);
    let candidates = (0..1_000)
        .map(|index| super::scheduler_test::candidate(&format!("auth-{index:04}"), "codex", 0, 1))
        .collect::<Vec<_>>();
    for index in 0..2_000 {
        let selected = scheduler
            .pick_single(
                "codex",
                Some("gpt"),
                0,
                &candidates,
                &[],
                &Default::default(),
            )
            .unwrap();
        assert_eq!(selected.auth_id, format!("auth-{:04}", index % 1_000));
    }
}
