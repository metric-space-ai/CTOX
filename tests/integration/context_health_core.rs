#[path = "../harness/mod.rs"]
mod harness;

use harness::TestRoot;

#[test]
fn context_health_flags_bootstrap_continuity_gaps() {
    let root = TestRoot::new("context-health-bootstrap");
    let db = root.path("runtime/ctox.sqlite3");
    let db_str = db.to_string_lossy().to_string();

    root.run(&["lcm-init", &db_str]).success();
    root.run(&["continuity-init", &db_str, "1"]).success();

    let assessed = root.run(&[
        "context-health",
        &db_str,
        "1",
        "Need to keep deployment blockers and next slices in view.",
    ]);
    let assessed_json = assessed.success().json();

    assert_eq!(assessed_json["conversation_id"].as_i64(), Some(1));
    assert_eq!(assessed_json["repair_recommended"].as_bool(), Some(true));
    assert_eq!(assessed_json["status"].as_str(), Some("critical"));
    assert!(assessed_json["summary"]
        .as_str()
        .unwrap_or_default()
        .contains("critical"));
    let warnings = assessed_json["warnings"]
        .as_array()
        .expect("warnings should be an array");
    assert!(
        warnings
            .iter()
            .any(|warning| warning["code"].as_str() == Some("focus_document_thin")),
        "expected focus_document_thin warning in {warnings:?}"
    );
}
