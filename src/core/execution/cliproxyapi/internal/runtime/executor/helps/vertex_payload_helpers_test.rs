// ref: internal/runtime/executor/helps/vertex_payload_helpers_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use std::borrow::Cow;

use super::strip_vertex_openai_responses_tool_call_ids;
use crate::internal::util::get_gjson_bytes_no_copy;

#[test]
fn payload_without_tool_call_ids_is_borrowed_byte_identically() {
    let input = br#"{"contents":[{"role":"model","parts":[{"functionCall":{"name":"lookup","args":{"id":9007199254740993}}}]}]}"#;
    let output = strip_vertex_openai_responses_tool_call_ids(input, "openai-response");

    assert!(matches!(&output, Cow::Borrowed(_)));
    assert_eq!(output.as_ptr(), input.as_ptr());
    assert_eq!(output.as_ref(), input);
}

#[test]
fn targeted_ids_are_removed_without_touching_nested_domain_ids_or_large_integers() {
    let input = br#"{"contents":[{"role":"model","parts":[{"functionCall":{"id":"call_1","name":"lookup","args":{"id":9007199254740993}}}]},{"role":"user","parts":[{"functionResponse":{"id":"call_1","name":"lookup","response":{"id":"keep"}}}]}]}"#;
    let output = strip_vertex_openai_responses_tool_call_ids(input, " OPENAI-RESPONSE ");

    assert!(matches!(&output, Cow::Owned(_)));
    assert!(!get_gjson_bytes_no_copy(&output, "contents.0.parts.0.functionCall.id").exists());
    assert!(!get_gjson_bytes_no_copy(&output, "contents.1.parts.0.functionResponse.id").exists());
    assert_eq!(
        get_gjson_bytes_no_copy(&output, "contents.1.parts.0.functionResponse.response.id").str(),
        "keep"
    );
    assert_eq!(
        get_gjson_bytes_no_copy(&output, "contents.0.parts.0.functionCall.args.id").json(),
        "9007199254740993"
    );
}

#[test]
fn unsupported_source_and_malformed_payload_are_borrowed_noops() {
    let valid = br#"{"contents":[{"parts":[{"functionCall":{"id":"call_1"}}]}]}"#;
    assert!(matches!(
        strip_vertex_openai_responses_tool_call_ids(valid, "openai"),
        Cow::Borrowed(_)
    ));

    let malformed = br#"{"contents":[{"parts":[{"functionCall":{"id":"call_1"}}]}"#;
    assert!(matches!(
        strip_vertex_openai_responses_tool_call_ids(malformed, "openai-response"),
        Cow::Borrowed(_)
    ));
}

#[test]
fn changed_payload_preserves_unrelated_raw_formatting_and_member_order() {
    let input = br#"{ "keep" : [ 1, 2 ], "contents" : [{"parts":[{"functionCall":{ "id" : "call_1", "name" : "lookup", "args" : { "z":2, "a":1 } }}]}], "tail":true }"#;
    let output = strip_vertex_openai_responses_tool_call_ids(input, "openai-response");

    assert_eq!(
        output.as_ref(),
        br#"{ "keep" : [ 1, 2 ], "contents" : [{"parts":[{"functionCall":{  "name" : "lookup", "args" : { "z":2, "a":1 } }}]}], "tail":true }"#
    );
}
