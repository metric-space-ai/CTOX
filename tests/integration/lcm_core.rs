#[path = "../harness/mod.rs"]
mod harness;

use harness::TestRoot;

#[test]
fn lcm_dump_persists_messages() {
    let root = TestRoot::new("lcm-dump");
    let db_path = root.path("runtime/test_lcm.db");
    let db = db_path.to_string_lossy().to_string();

    root.run(&["lcm-init", &db]).success();
    root.run(&["lcm-add-message", &db, "1", "user", "hello world"])
        .success();
    let dump = root.run(&["lcm-dump", &db, "1"]);
    let json = dump.success().json();

    assert_eq!(json["conversation_id"].as_i64(), Some(1));
    assert_eq!(
        json["messages"].as_array().map(|items| items.len()),
        Some(1)
    );
    assert_eq!(json["messages"][0]["content"].as_str(), Some("hello world"));
}

#[test]
fn continuity_init_and_refresh_produce_core_documents() {
    let root = TestRoot::new("lcm-continuity");
    let db_path = root.path("runtime/test_continuity.db");
    let db = db_path.to_string_lossy().to_string();

    root.run(&["lcm-init", &db]).success();
    root.run(&["lcm-add-message", &db, "1", "user", "hello world"])
        .success();

    let init = root.run(&["continuity-init", &db, "1"]);
    let init_json = init.success().json();
    assert!(init_json["narrative"]["content"]
        .as_str()
        .unwrap_or_default()
        .contains("# CONTINUITY NARRATIVE"));

    let refreshed = root.run(&["lcm-refresh-continuity", &db, "1"]);
    let refreshed_json = refreshed.success().json();
    assert_eq!(refreshed_json["conversation_id"].as_i64(), Some(1));
    assert_eq!(
        refreshed_json["source_message_ids"]
            .as_array()
            .map(|items| items.len()),
        Some(1)
    );
}
