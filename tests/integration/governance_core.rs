#[path = "../harness/mod.rs"]
mod harness;

use harness::TestRoot;

#[test]
fn governance_inventory_exposes_core_mechanisms() {
    let root = TestRoot::new("governance-inventory");
    let output = root.run(&["governance", "inventory"]);
    let json = output.success().json();

    assert!(json["count"].as_u64().unwrap_or(0) >= 8);
    let mechanisms = json["mechanisms"].as_array().expect("mechanisms array");
    assert!(mechanisms
        .iter()
        .any(|entry| { entry["mechanism_id"].as_str() == Some("queue_pressure_guard") }));
    assert!(mechanisms
        .iter()
        .any(|entry| { entry["mechanism_id"].as_str() == Some("follow_up_evaluate") }));
}

#[test]
fn governance_snapshot_returns_mechanisms_and_recent_events_surface() {
    let root = TestRoot::new("governance-snapshot");
    let output = root.run(&["governance", "snapshot", "--conversation-id", "1"]);
    let json = output.success().json();

    assert_eq!(json["ok"].as_bool(), Some(true));
    assert!(json["snapshot"]["mechanisms"].is_array());
    assert!(json["snapshot"]["recent_events"].is_array());
}
