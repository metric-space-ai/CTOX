#[path = "../harness/mod.rs"]
mod harness;

use harness::TestRoot;

#[test]
fn status_reports_stopped_service_on_fresh_root() {
    let root = TestRoot::new("service-surface");
    let output = root.run(&["status"]);
    let json = output.success().json();

    assert_eq!(json["running"].as_bool(), Some(false));
    assert_eq!(json["busy"].as_bool(), Some(false));
    assert!(json["listen_addr"].as_str().is_some());
    assert_eq!(json["pending_count"].as_u64(), Some(0));
}

#[test]
fn clean_room_baseline_plan_returns_plan_json() {
    let root = TestRoot::new("baseline-plan");
    let output = root.run(&[
        "clean-room-baseline-plan",
        "gpt_oss",
        "Reply with CTOX_BASELINE_OK and nothing else.",
    ]);
    let json = output.success().json();

    assert_eq!(json["runtime"]["family"].as_str(), Some("gpt_oss"));
    assert!(
        json["engine_command"].is_array(),
        "expected engine_command array in {json:#}"
    );
    assert!(
        json["codex_exec_command"].is_array(),
        "expected codex_exec_command array in {json:#}"
    );
}
