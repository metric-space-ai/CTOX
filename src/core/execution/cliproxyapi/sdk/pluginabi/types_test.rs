// ref: sdk/pluginabi/types_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use serde_json::value::to_raw_value;

use super::*;

#[test]
fn envelope_round_trip_preserves_raw_result() {
    let payload = to_raw_value(&serde_json::json!({"name": "example"})).expect("raw payload");
    let encoded = serde_json::to_vec(&Envelope::success(Some(payload))).expect("encode envelope");
    let decoded: Envelope = serde_json::from_slice(&encoded).expect("decode envelope");

    assert!(decoded.ok);
    assert_eq!(
        decoded
            .result
            .as_deref()
            .map(serde_json::value::RawValue::get),
        Some(r#"{"name":"example"}"#)
    );
    assert!(decoded.error.is_none());
}

#[test]
fn method_names_are_stable() {
    assert_eq!(SCHEMA_VERSION, 2);
    assert_eq!(METHOD_PLUGIN_REGISTER, "plugin.register");
    assert_eq!(METHOD_REQUEST_INTERCEPT_BEFORE, "request.intercept_before");
    assert_eq!(METHOD_REQUEST_INTERCEPT_AFTER, "request.intercept_after");
    assert_eq!(METHOD_REQUEST_COMPLETE, "request.complete");
    assert_eq!(METHOD_RESPONSE_INTERCEPT_AFTER, "response.intercept_after");
    assert_eq!(
        METHOD_RESPONSE_INTERCEPT_STREAM_CHUNK,
        "response.intercept_stream_chunk"
    );
    assert_eq!(METHOD_HOST_HTTP_DO, "host.http.do");
    assert_eq!(METHOD_HOST_HTTP_STREAM_READ, "host.http.stream_read");
    assert_eq!(METHOD_HOST_MODEL_EXECUTE, "host.model.execute");
    assert_eq!(
        METHOD_HOST_MODEL_EXECUTE_STREAM,
        "host.model.execute_stream"
    );
    assert_eq!(METHOD_HOST_MODEL_STREAM_READ, "host.model.stream_read");
    assert_eq!(METHOD_HOST_MODEL_STREAM_CLOSE, "host.model.stream_close");
    assert_eq!(METHOD_HOST_AUTH_LIST, "host.auth.list");
    assert_eq!(METHOD_HOST_AUTH_GET, "host.auth.get");
    assert_eq!(METHOD_HOST_AUTH_GET_RUNTIME, "host.auth.get_runtime");
    assert_eq!(METHOD_HOST_AUTH_SAVE, "host.auth.save");
    assert_eq!(METHOD_EXECUTOR_EXECUTE_STREAM, "executor.execute_stream");
}

#[test]
fn scheduler_and_router_method_names_are_stable() {
    assert_eq!(METHOD_SCHEDULER_PICK, "scheduler.pick");
    assert_eq!(METHOD_MODEL_ROUTE, "model.route");
}
