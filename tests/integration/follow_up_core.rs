#[path = "../harness/mod.rs"]
mod harness;

use harness::TestRoot;

#[test]
fn follow_up_reports_done_when_scope_is_closed() {
    let root = TestRoot::new("follow-up-done");
    let output = root.run(&[
        "follow-up",
        "evaluate",
        "--goal",
        "Ship feature",
        "--result",
        "Implemented parser and tests with no remaining open work.",
    ]);
    let json = output.success().json();

    assert_eq!(json["status"].as_str(), Some("done"));
    assert_eq!(
        json["owner_communication_recommended"].as_bool(),
        Some(false)
    );
}

#[test]
fn follow_up_reports_blocked_on_user_for_approval_blockers() {
    let root = TestRoot::new("follow-up-blocked");
    let output = root.run(&[
        "follow-up",
        "evaluate",
        "--goal",
        "Ship feature",
        "--result",
        "Implementation paused.",
        "--blocker",
        "Need owner approval for production rollout",
        "--owner-visible",
    ]);
    let json = output.success().json();

    assert_eq!(json["status"].as_str(), Some("blocked_on_user"));
    assert_eq!(
        json["owner_communication_recommended"].as_bool(),
        Some(true)
    );
}

#[test]
fn follow_up_reports_needs_replan_when_requirements_changed() {
    let root = TestRoot::new("follow-up-replan");
    let output = root.run(&[
        "follow-up",
        "evaluate",
        "--goal",
        "Ship feature",
        "--result",
        "Initial plan assumed SMTP delivery.",
        "--requirements-changed",
        "--owner-visible",
    ]);
    let json = output.success().json();

    assert_eq!(json["status"].as_str(), Some("needs_replan"));
    assert_eq!(
        json["owner_communication_recommended"].as_bool(),
        Some(true)
    );
}
