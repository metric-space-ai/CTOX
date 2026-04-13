#[path = "../harness/mod.rs"]
mod harness;

use harness::TestRoot;

#[test]
fn schedule_run_now_emits_queue_work() {
    let root = TestRoot::new("schedule-run-now");
    let added = root.run(&[
        "schedule",
        "add",
        "--name",
        "refresh scrape fixture",
        "--cron",
        "*/5 * * * *",
        "--prompt",
        "Refresh the scrape fixture and record the outcome.",
    ]);
    let added_json = added.success().json();
    let task_id = added_json["task"]["task_id"]
        .as_str()
        .expect("schedule add should return task id")
        .to_string();

    let run_now = root.run(&["schedule", "run-now", "--task-id", &task_id]);
    let run_json = run_now.success().json();
    assert_eq!(run_json["run"]["status"].as_str(), Some("emitted"));
    assert_eq!(run_json["run"]["task_id"].as_str(), Some(task_id.as_str()));

    let schedule = root.run(&["schedule", "list"]);
    let schedule_json = schedule.success().json();
    assert_eq!(schedule_json["count"].as_u64(), Some(1));
    assert!(schedule_json["tasks"][0]["name"]
        .as_str()
        .unwrap_or_default()
        .contains("refresh scrape fixture"));
    assert!(schedule_json["tasks"][0]["last_run_at"].is_string());
}

#[test]
fn schedule_pause_and_resume_toggle_enabled_state() {
    let root = TestRoot::new("schedule-toggle");
    let added = root.run(&[
        "schedule",
        "add",
        "--name",
        "demo toggle",
        "--cron",
        "*/10 * * * *",
        "--prompt",
        "Do the thing.",
    ]);
    let added_json = added.success().json();
    let task_id = added_json["task"]["task_id"]
        .as_str()
        .expect("schedule add should return task id")
        .to_string();

    let paused = root.run(&["schedule", "pause", "--task-id", &task_id]);
    let paused_json = paused.success().json();
    assert_eq!(paused_json["task"]["enabled"].as_bool(), Some(false));

    let resumed = root.run(&["schedule", "resume", "--task-id", &task_id]);
    let resumed_json = resumed.success().json();
    assert_eq!(resumed_json["task"]["enabled"].as_bool(), Some(true));
}
