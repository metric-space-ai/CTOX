// ref: internal/runtime/executor/antigravity_executor_buildrequest_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use super::antigravity_executor_request::{
    prepare_antigravity_generate_body, AntigravityRequestError,
};
use serde_json::{json, Value};

fn prepared(model: &str, body: Value) -> Value {
    serde_json::from_slice(
        &prepare_antigravity_generate_body(&serde_json::to_vec(&body).unwrap(), model, "project-1")
            .unwrap(),
    )
    .unwrap()
}

#[test]
fn uses_auth_project_route_model_and_preserves_request_type() {
    let out = prepared(
        "route-model",
        json!({"model":"payload-model","requestType":"web_search","request":{"contents":[]}}),
    );
    assert_eq!(out["project"], "project-1");
    assert_eq!(out["model"], "route-model");
    assert_eq!(out["requestType"], "web_search");
}

#[test]
fn derives_stable_session_and_preserves_explicit_session() {
    let body = json!({"request":{"contents":[{"role":"user","parts":[{"text":"same"}]}]}});
    let first = prepared("gemini-3", body.clone());
    let second = prepared("gemini-3", body);
    assert_eq!(
        first.pointer("/request/sessionId"),
        second.pointer("/request/sessionId")
    );
    let explicit = prepared(
        "gemini-3",
        json!({"request":{"sessionId":"mine","contents":[]}}),
    );
    assert_eq!(explicit.pointer("/request/sessionId"), Some(&json!("mine")));
}

#[test]
fn rejects_missing_project_and_request() {
    assert_eq!(
        prepare_antigravity_generate_body(br#"{"request":{}}"#, "m", ""),
        Err(AntigravityRequestError::InvalidProjectId)
    );
    assert_eq!(
        prepare_antigravity_generate_body(br#"{}"#, "m", "p"),
        Err(AntigravityRequestError::MissingRequest)
    );
}

#[test]
fn sanitizes_tools_but_skips_absent_and_empty_tools() {
    let out = prepared(
        "claude-sonnet",
        json!({"request":{"contents":[],"tools":[{"functionDeclarations":[{"name":"f","parameters":{"type":"object","properties":{},"pattern":"x"}}]}]}}),
    );
    let schema = out
        .pointer("/request/tools/0/functionDeclarations/0/parameters")
        .unwrap();
    assert!(schema.get("pattern").is_none());
    assert!(schema.pointer("/properties/reason").is_some());
    assert!(prepared("gemini", json!({"request":{"contents":[]}}))
        .pointer("/request/tools")
        .is_none());
    assert_eq!(
        prepared("gemini", json!({"request":{"contents":[],"tools":[]}})).pointer("/request/tools"),
        Some(&json!([]))
    );
}
