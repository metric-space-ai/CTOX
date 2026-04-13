#[path = "../harness/mod.rs"]
mod harness;

use harness::TestRoot;

#[test]
fn queue_add_and_list_persist_on_isolated_root() {
    let root = TestRoot::new("durable-queue");
    let added = root.run(&[
        "queue",
        "add",
        "--title",
        "Investigate queue path",
        "--prompt",
        "Trace the queue pipeline.",
    ]);
    let added_json = added.success().json();
    let message_key = added_json["task"]["message_key"]
        .as_str()
        .expect("queue add should return message key")
        .to_string();

    let listed = root.run(&["queue", "list"]);
    let listed_json = listed.success().json();

    assert_eq!(listed_json["count"].as_u64(), Some(1));
    assert_eq!(
        listed_json["tasks"][0]["message_key"].as_str(),
        Some(message_key.as_str())
    );
    assert_eq!(
        listed_json["tasks"][0]["route_status"].as_str(),
        Some("pending")
    );
}

#[test]
fn plan_ingest_persists_goal_on_isolated_root() {
    let root = TestRoot::new("plan-emit-now");
    let ingested = root.run(&[
        "plan",
        "ingest",
        "--title",
        "Ship queue hardening",
        "--prompt",
        "Plan and execute queue hardening for CTOX.",
    ]);
    let ingested_json = ingested.success().json();

    assert_eq!(
        ingested_json["plan"]["goal"]["title"].as_str(),
        Some("Ship queue hardening")
    );

    let listed = root.run(&["plan", "list"]);
    let listed_json = listed.success().json();

    assert_eq!(listed_json["count"].as_u64(), Some(1));
    assert!(listed_json["goals"][0]["title"]
        .as_str()
        .unwrap_or_default()
        .contains("Ship queue hardening"));
}
