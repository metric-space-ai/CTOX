// ref: sdk/api/handlers/gemini/interactions_handlers_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use super::*;

#[test]
fn parses_exactly_one_model_or_agent_and_boolean_stream() {
    let model = parse_interactions_request_target(
        br#"{"model":"models/gemini-3.5-flash","stream":true,"input":"hi"}"#,
    )
    .unwrap();
    assert_eq!(model.model, "models/gemini-3.5-flash");
    assert!(model.stream);

    let agent = parse_interactions_request_target(br#"{"agent":"agents/test-agent"}"#).unwrap();
    assert_eq!(agent.agent, "agents/test-agent");
    for invalid in [
        br#"{"input":"hi"}"#.as_slice(),
        br#"{"model":"m","agent":"a"}"#.as_slice(),
        br#"{"model":"m","stream":"true"}"#.as_slice(),
        br#"{"model":1}"#.as_slice(),
        b"{".as_slice(),
    ] {
        assert!(parse_interactions_request_target(invalid).is_err());
    }
}

#[test]
fn model_resource_normalization_preserves_unrelated_raw_json_bytes() {
    let raw = b"{ \"input\" : [1,2], \"model\" : \"models/gemini-3.5-flash\", \"x\":true }";
    let target = parse_interactions_request_target(raw).unwrap();
    let (model, body) = prepare_interactions_execution_target(raw, &target);
    assert_eq!(model, "gemini-3.5-flash");
    assert_eq!(
        body,
        b"{ \"input\" : [1,2], \"model\" : \"gemini-3.5-flash\", \"x\":true }"
    );
}

#[test]
fn model_resource_normalization_only_rewrites_the_top_level_field() {
    let raw =
        br#"{"metadata":{"model":"models/gemini-3.5-flash"},"model":"models/gemini-3.5-flash"}"#;
    let target = parse_interactions_request_target(raw).unwrap();
    let (_, body) = prepare_interactions_execution_target(raw, &target);
    assert_eq!(
        body,
        br#"{"metadata":{"model":"models/gemini-3.5-flash"},"model":"gemini-3.5-flash"}"#
    );
}

#[test]
fn agent_request_forces_native_provider_and_auth_selection_model() {
    let target = parse_interactions_request_target(br#"{"agent":"agents/test-agent"}"#).unwrap();
    let request = build_interactions_execution_request(
        &target,
        "agents/test-agent",
        br#"{"agent":"agents/test-agent"}"#.to_vec(),
        "",
    );
    assert_eq!(request.forced_provider, "gemini-interactions");
    assert_eq!(
        request.auth_selection_model,
        INTERACTIONS_AGENT_AUTH_SELECTION_MODEL
    );
    assert_eq!(request.entry_protocol, "interactions");
    assert_eq!(request.exit_protocol, "interactions");
}

#[test]
fn stream_framing_wraps_bare_json_and_preserves_existing_sse() {
    assert_eq!(
        frame_interactions_sse_chunk(br#"{"type":"interaction.completed"}"#),
        b"data: {\"type\":\"interaction.completed\"}\n\n"
    );
    assert_eq!(
        frame_interactions_sse_chunk(b"event: done\ndata: {}\n\n"),
        b"event: done\ndata: {}\n\n"
    );
}
