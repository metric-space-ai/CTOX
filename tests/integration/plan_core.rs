#[path = "../harness/mod.rs"]
mod harness;

use harness::TestRoot;

#[test]
fn plan_emit_next_queues_first_step_and_creates_channel_work() {
    let root = TestRoot::new("plan-emit-next");
    let ingested = root.run(&[
        "plan",
        "ingest",
        "--title",
        "Ship release hardening",
        "--prompt",
        "Inspect the release path, implement the next slice, and verify the outcome.",
    ]);
    let ingested_json = ingested.success().json();
    let goal_id = ingested_json["plan"]["goal"]["goal_id"]
        .as_str()
        .expect("goal id should exist")
        .to_string();
    let first_step_id = ingested_json["plan"]["steps"][0]["step_id"]
        .as_str()
        .expect("first step id should exist")
        .to_string();

    let emitted = root.run(&["plan", "emit-next", "--goal-id", &goal_id]);
    let emitted_json = emitted.success().json();
    let message_key = emitted_json["emitted"]["message_key"]
        .as_str()
        .expect("message key should exist")
        .to_string();

    assert_eq!(
        emitted_json["emitted"]["step_id"].as_str(),
        Some(first_step_id.as_str())
    );

    let shown = root.run(&["plan", "show", "--goal-id", &goal_id]);
    let shown_json = shown.success().json();
    assert_eq!(
        shown_json["plan"]["goal"]["last_emitted_at"]
            .as_str()
            .map(str::is_empty),
        Some(false)
    );
    assert_eq!(
        shown_json["plan"]["steps"][0]["status"].as_str(),
        Some("queued")
    );
    assert_eq!(
        shown_json["plan"]["steps"][0]["last_message_key"].as_str(),
        Some(message_key.as_str())
    );

    let channels = root.run(&["channel", "list"]);
    let channels_json = channels.success().json();
    assert_eq!(channels_json["count"].as_u64(), Some(1));
    assert_eq!(
        channels_json["messages"][0]["message_key"].as_str(),
        Some(message_key.as_str())
    );
    assert_eq!(
        channels_json["messages"][0]["routing"]["route_status"].as_str(),
        Some("pending")
    );
}

#[test]
fn plan_auto_advance_completes_goal_across_all_steps() {
    let root = TestRoot::new("plan-auto-advance");
    let ingested = root.run(&[
        "plan",
        "ingest",
        "--title",
        "Ship queue hardening",
        "--prompt",
        "Inspect the queue path, implement the hardening slice, and verify the result.",
        "--auto-advance",
    ]);
    let ingested_json = ingested.success().json();
    let goal_id = ingested_json["plan"]["goal"]["goal_id"]
        .as_str()
        .expect("goal id should exist")
        .to_string();
    let first_step_id = ingested_json["plan"]["steps"][0]["step_id"]
        .as_str()
        .expect("first step id should exist")
        .to_string();
    let second_step_id = ingested_json["plan"]["steps"][1]["step_id"]
        .as_str()
        .expect("second step id should exist")
        .to_string();
    let third_step_id = ingested_json["plan"]["steps"][2]["step_id"]
        .as_str()
        .expect("third step id should exist")
        .to_string();

    root.run(&["plan", "emit-next", "--goal-id", &goal_id])
        .success();

    root.run(&[
        "plan",
        "complete-step",
        "--step-id",
        &first_step_id,
        "--result",
        "Inspected queue flow and confirmed the current constraints.",
    ])
    .success();
    let after_first = root.run(&["plan", "show", "--goal-id", &goal_id]);
    let after_first_json = after_first.success().json();
    assert_eq!(
        after_first_json["plan"]["steps"][0]["status"].as_str(),
        Some("completed")
    );
    assert_eq!(
        after_first_json["plan"]["steps"][1]["status"].as_str(),
        Some("queued")
    );
    assert_eq!(
        after_first_json["plan"]["steps"][1]["step_id"].as_str(),
        Some(second_step_id.as_str())
    );

    root.run(&[
        "plan",
        "complete-step",
        "--step-id",
        &second_step_id,
        "--result",
        "Applied the hardening slice and captured the concrete changes.",
    ])
    .success();
    let after_second = root.run(&["plan", "show", "--goal-id", &goal_id]);
    let after_second_json = after_second.success().json();
    assert_eq!(
        after_second_json["plan"]["steps"][1]["status"].as_str(),
        Some("completed")
    );
    assert_eq!(
        after_second_json["plan"]["steps"][2]["status"].as_str(),
        Some("queued")
    );
    assert_eq!(
        after_second_json["plan"]["steps"][2]["step_id"].as_str(),
        Some(third_step_id.as_str())
    );

    root.run(&[
        "plan",
        "complete-step",
        "--step-id",
        &third_step_id,
        "--result",
        "Verified the result and no open work remains.",
    ])
    .success();
    let finished = root.run(&["plan", "show", "--goal-id", &goal_id]);
    let finished_json = finished.success().json();
    assert_eq!(
        finished_json["plan"]["goal"]["status"].as_str(),
        Some("completed")
    );
    assert_eq!(
        finished_json["plan"]["goal"]["last_completed_at"]
            .as_str()
            .map(str::is_empty),
        Some(false)
    );
    assert_eq!(
        finished_json["plan"]["steps"][2]["status"].as_str(),
        Some("completed")
    );
}

#[test]
fn plan_block_fail_retry_and_unblock_preserve_goal_state() {
    let root = TestRoot::new("plan-recovery");
    let ingested = root.run(&[
        "plan",
        "ingest",
        "--title",
        "Ship proxy hardening",
        "--prompt",
        "Inspect the proxy path, patch it, and verify the final state.",
    ]);
    let ingested_json = ingested.success().json();
    let goal_id = ingested_json["plan"]["goal"]["goal_id"]
        .as_str()
        .expect("goal id should exist")
        .to_string();
    let first_step_id = ingested_json["plan"]["steps"][0]["step_id"]
        .as_str()
        .expect("first step id should exist")
        .to_string();

    root.run(&[
        "plan",
        "block-step",
        "--step-id",
        &first_step_id,
        "--reason",
        "Need owner approval before touching production proxy config.",
    ])
    .success();
    let blocked = root.run(&["plan", "show", "--goal-id", &goal_id]);
    let blocked_json = blocked.success().json();
    assert_eq!(
        blocked_json["plan"]["goal"]["status"].as_str(),
        Some("blocked")
    );
    assert_eq!(
        blocked_json["plan"]["steps"][0]["status"].as_str(),
        Some("blocked")
    );
    assert_eq!(
        blocked_json["plan"]["steps"][0]["blocked_reason"].as_str(),
        Some("Need owner approval before touching production proxy config.")
    );

    root.run(&[
        "plan",
        "unblock-step",
        "--step-id",
        &first_step_id,
        "--defer-minutes",
        "5",
    ])
    .success();
    let unblocked = root.run(&["plan", "show", "--goal-id", &goal_id]);
    let unblocked_json = unblocked.success().json();
    assert_eq!(
        unblocked_json["plan"]["goal"]["status"].as_str(),
        Some("active")
    );
    assert_eq!(
        unblocked_json["plan"]["steps"][0]["status"].as_str(),
        Some("pending")
    );
    assert_eq!(
        unblocked_json["plan"]["steps"][0]["defer_until"]
            .as_str()
            .map(str::is_empty),
        Some(false)
    );

    root.run(&[
        "plan",
        "fail-step",
        "--step-id",
        &first_step_id,
        "--reason",
        "Proxy smoke failed due to upstream mismatch.",
    ])
    .success();
    let failed = root.run(&["plan", "show", "--goal-id", &goal_id]);
    let failed_json = failed.success().json();
    assert_eq!(
        failed_json["plan"]["steps"][0]["status"].as_str(),
        Some("failed")
    );
    assert!(failed_json["plan"]["steps"][0]["last_result_excerpt"]
        .as_str()
        .unwrap_or_default()
        .contains("upstream mismatch"));

    root.run(&["plan", "retry-step", "--step-id", &first_step_id])
        .success();
    let retried = root.run(&["plan", "show", "--goal-id", &goal_id]);
    let retried_json = retried.success().json();
    assert_eq!(
        retried_json["plan"]["goal"]["status"].as_str(),
        Some("active")
    );
    assert_eq!(
        retried_json["plan"]["steps"][0]["status"].as_str(),
        Some("pending")
    );
    assert_eq!(
        retried_json["plan"]["steps"][0]["blocked_reason"].as_str(),
        None
    );
}
