#[path = "../harness/mod.rs"]
mod harness;

use harness::TestRoot;
use rusqlite::Connection;

#[test]
fn channel_ingest_list_and_history_preserve_thread_context() {
    let root = TestRoot::new("channels-thread");
    let ingest = root.run(&[
        "channel",
        "ingest-tui",
        "--account-key",
        "owner",
        "--thread-key",
        "thread/demo",
        "--body",
        "hello queue",
        "--sender-display",
        "Owner",
        "--sender-address",
        "owner@example.com",
        "--subject",
        "Demo",
    ]);
    let ingest_json = ingest.success().json();
    assert_eq!(
        ingest_json["stored"]["thread_key"].as_str(),
        Some("thread/demo")
    );

    let listed = root.run(&["channel", "list", "--limit", "5"]);
    let list_json = listed.success().json();
    assert_eq!(list_json["count"].as_u64(), Some(1));
    assert_eq!(
        list_json["messages"][0]["routing"]["route_status"].as_str(),
        Some("pending")
    );

    let history = root.run(&[
        "channel",
        "history",
        "--thread-key",
        "thread/demo",
        "--limit",
        "5",
    ]);
    let history_json = history.success().json();
    assert_eq!(history_json["thread_key"].as_str(), Some("thread/demo"));
    assert_eq!(history_json["count"].as_u64(), Some(1));
}

#[test]
fn channel_take_ack_and_context_surface_routing_state() {
    let root = TestRoot::new("channels-routing");
    let ingest = root.run(&[
        "channel",
        "ingest-tui",
        "--account-key",
        "owner",
        "--thread-key",
        "thread/demo",
        "--body",
        "redis rollout blocked on approval",
        "--sender-display",
        "Owner",
        "--sender-address",
        "owner@example.com",
        "--subject",
        "Demo",
    ]);
    let ingest_json = ingest.success().json();
    let message_key = ingest_json["stored"]["message_key"]
        .as_str()
        .expect("ingest should return message key")
        .to_string();

    let taken = root.run(&[
        "channel",
        "take",
        "--limit",
        "5",
        "--lease-owner",
        "worker-1",
    ]);
    let taken_json = taken.success().json();
    assert_eq!(taken_json["lease_owner"].as_str(), Some("worker-1"));
    assert_eq!(
        taken_json["messages"][0]["routing"]["route_status"].as_str(),
        Some("leased")
    );

    let acked = root.run(&["channel", "ack", "--status", "handled", &message_key]);
    let acked_json = acked.success().json();
    assert_eq!(acked_json["updated"].as_u64(), Some(1));

    let listed = root.run(&["channel", "list", "--limit", "5"]);
    let list_json = listed.success().json();
    assert_eq!(
        list_json["messages"][0]["routing"]["route_status"].as_str(),
        Some("handled")
    );
    assert!(list_json["messages"][0]["routing"]["acked_at"].is_string());

    let context = root.run(&[
        "channel",
        "context",
        "--thread-key",
        "thread/demo",
        "--query",
        "approval",
        "--limit",
        "5",
    ]);
    let context_json = context.success().json();
    assert_eq!(
        context_json["context"]["latest_inbound"]["summary"].as_str(),
        Some("redis rollout blocked on approval")
    );
    assert_eq!(
        context_json["context"]["thread_key"].as_str(),
        Some("thread/demo")
    );
}

#[test]
fn channel_ack_ignores_unknown_message_keys() {
    let root = TestRoot::new("channels-ack-missing");
    let ingest = root.run(&[
        "channel",
        "ingest-tui",
        "--account-key",
        "owner",
        "--thread-key",
        "thread/demo",
        "--body",
        "hello queue",
        "--sender-display",
        "Owner",
        "--sender-address",
        "owner@example.com",
        "--subject",
        "Demo",
    ]);
    let ingest_json = ingest.success().json();
    let message_key = ingest_json["stored"]["message_key"]
        .as_str()
        .expect("ingest should return message key")
        .to_string();

    let acked = root.run(&[
        "channel",
        "ack",
        "--status",
        "handled",
        "missing-message-key",
    ]);
    let acked_json = acked.success().json();
    assert_eq!(acked_json["updated"].as_u64(), Some(0));

    let conn = Connection::open(root.db_path()).expect("failed to open channel db");
    let missing_count: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM communication_routing_state WHERE message_key = ?1",
            ["missing-message-key"],
            |row| row.get(0),
        )
        .expect("failed to count missing routing rows");
    assert_eq!(missing_count, 0);

    let route_status: String = conn
        .query_row(
            "SELECT route_status FROM communication_routing_state WHERE message_key = ?1",
            [&message_key],
            |row| row.get(0),
        )
        .expect("missing real routing row");
    assert_eq!(route_status, "pending");
}
