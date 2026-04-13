#[path = "../harness/mod.rs"]
mod harness;

use harness::TestRoot;

#[test]
fn verification_surfaces_are_available_on_fresh_lcm_state() {
    let root = TestRoot::new("verification-empty");
    let db = root.path("runtime/ctox_lcm.db");
    let db_str = db.to_string_lossy().to_string();

    root.run(&["lcm-init", &db_str]).success();

    let assurance = root.run(&["verification", "assurance"]);
    let assurance_json = assurance.success().json();
    assert_eq!(
        assurance_json["assurance"]["conversation_id"].as_i64(),
        Some(1)
    );
    assert!(assurance_json["assurance"]["latest_run"].is_null());
    assert_eq!(
        assurance_json["assurance"]["open_claims"]
            .as_array()
            .map(|items| items.len()),
        Some(0)
    );
    assert_eq!(
        assurance_json["assurance"]["closure_blocking_claims"]
            .as_array()
            .map(|items| items.len()),
        Some(0)
    );

    let runs = root.run(&["verification", "runs"]);
    let runs_json = runs.success().json();
    assert_eq!(runs_json["count"].as_u64(), Some(0));

    let claims = root.run(&["verification", "claims"]);
    let claims_json = claims.success().json();
    assert_eq!(claims_json["count"].as_u64(), Some(0));
}
