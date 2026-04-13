#[path = "../harness/mod.rs"]
mod harness;

use harness::TestRoot;
use std::fs;

#[test]
fn scrape_target_registration_surfaces_api_contract_and_web_paths() {
    let root = TestRoot::new("scrape-target");
    let payload_path = root.path("target.json");
    let script_path = root.path("extractor.js");
    fs::write(
        &payload_path,
        r#"{
  "target_key": "Acme Jobs",
  "display_name": "Acme Jobs",
  "start_url": "https://example.com/jobs",
  "target_kind": "jobs",
  "output_schema": {"schema_key": "jobs.v1"},
  "config": {
    "skip_probe": true,
    "api": {
      "filter_fields": ["title", "classification.category"]
    }
  }
}
"#,
    )
    .expect("failed to write target payload");
    fs::write(
        &script_path,
        "module.exports = async function extract() { return { records: [] }; };\n",
    )
    .expect("failed to write extractor script");

    let payload = payload_path.to_string_lossy().to_string();
    let script = script_path.to_string_lossy().to_string();

    let upserted = root.run(&["scrape", "upsert-target", "--input", &payload]);
    let upserted_json = upserted.success().json();
    assert_eq!(
        upserted_json["target"]["target_key"].as_str(),
        Some("acme-jobs")
    );

    let registered = root.run(&[
        "scrape",
        "register-script",
        "--target-key",
        "acme-jobs",
        "--script-file",
        &script,
        "--change-reason",
        "initial_import",
    ]);
    let registered_json = registered.success().json();
    assert_eq!(registered_json["script"]["revision_no"].as_i64(), Some(1));

    let listed = root.run(&["scrape", "list-targets"]);
    let listed_json = listed.success().json();
    assert_eq!(
        listed_json["targets"].as_array().map(|items| items.len()),
        Some(1)
    );

    let api = root.run(&["scrape", "show-api", "--target-key", "acme-jobs"]);
    let api_json = api.success().json();
    assert_eq!(api_json["api"]["target_key"].as_str(), Some("acme-jobs"));
    assert_eq!(api_json["api"]["source_count"].as_u64(), Some(1));
    assert_eq!(
        api_json["api"]["endpoints"]["api"].as_str(),
        Some("/ctox/scrape/targets/acme-jobs/api")
    );
    assert_eq!(
        api_json["api"]["endpoints"]["records"].as_str(),
        Some("/ctox/scrape/targets/acme-jobs/records")
    );
    assert_eq!(
        api_json["api"]["endpoints"]["semantic"].as_str(),
        Some("/ctox/scrape/targets/acme-jobs/semantic")
    );
    assert_eq!(
        api_json["api"]["endpoints"]["latest"].as_str(),
        Some("/ctox/scrape/targets/acme-jobs/latest")
    );
    assert!(api_json["api"]["paths"]["api_contract"]
        .as_str()
        .unwrap_or_default()
        .ends_with("api/api_contract.json"));
    assert!(api_json["api"]["records_query"]["filter_fields"]
        .as_array()
        .expect("filter_fields should be array")
        .iter()
        .any(|item| item.as_str() == Some("classification.category")));
}

#[test]
fn scrape_source_module_registration_surfaces_in_api_contract() {
    let root = TestRoot::new("scrape-source-module");
    let payload_path = root.path("target.json");
    let module_path = root.path("board-a-source.js");
    fs::write(
        &payload_path,
        r#"{
  "target_key": "Aggregated Jobs",
  "display_name": "Aggregated Jobs",
  "start_url": "https://example.com/jobs",
  "target_kind": "jobs",
  "config": {
    "skip_probe": true,
    "sources": [
      {
        "source_key": "board-a",
        "display_name": "Board A",
        "start_url": "https://a.example/jobs",
        "source_kind": "rss"
      }
    ]
  }
}
"#,
    )
    .expect("failed to write target payload");
    fs::write(
        &module_path,
        "module.exports = async function extractSource() { return { records: [{ id: 'a-1' }] }; };\n",
    )
    .expect("failed to write source module");

    let payload = payload_path.to_string_lossy().to_string();
    let module = module_path.to_string_lossy().to_string();

    root.run(&["scrape", "upsert-target", "--input", &payload])
        .success();
    let registered = root.run(&[
        "scrape",
        "register-source-module",
        "--target-key",
        "aggregated-jobs",
        "--source-key",
        "board-a",
        "--module-file",
        &module,
        "--change-reason",
        "initial_source_import",
    ]);
    let registered_json = registered.success().json();
    assert_eq!(
        registered_json["source_module"]["source_key"].as_str(),
        Some("board-a")
    );
    assert_eq!(
        registered_json["source_module"]["revision_no"].as_i64(),
        Some(1)
    );

    let api = root.run(&["scrape", "show-api", "--target-key", "aggregated-jobs"]);
    let api_json = api.success().json();
    let source_modules = api_json["api"]["source_modules"]
        .as_array()
        .expect("source_modules should be array");
    assert_eq!(source_modules.len(), 1);
    assert_eq!(source_modules[0]["source_key"].as_str(), Some("board-a"));
}
