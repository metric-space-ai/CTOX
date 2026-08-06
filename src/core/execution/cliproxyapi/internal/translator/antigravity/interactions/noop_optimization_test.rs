// ref: internal/translator/antigravity/interactions/noop_optimization_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::HashMap;

use serde_json::Value;

use super::rewrite_interactions_function_names;

#[test]
fn normalized_function_names_reuse_the_input_allocation() {
    let input = br#"{"request":{"contents":[{"parts":[{"functionCall":{"name":"lookup","args":{}}}]},{"parts":[{"functionResponse":{"name":"lookup","response":{}}}]}],"toolConfig":{"functionCallingConfig":{"allowedFunctionNames":["lookup"]}}}}"#.to_vec();
    let pointer = input.as_ptr();
    let output = rewrite_interactions_function_names(input, &HashMap::new());
    assert_eq!(output.as_ptr(), pointer);
}

#[test]
fn non_string_names_are_normalized_to_strings() {
    let input = br#"{"request":{"contents":[{"parts":[{"functionCall":{"name":true,"args":{}}}]}],"toolConfig":{"functionCallingConfig":{"allowedFunctionNames":[true]}}}}"#.to_vec();
    let output = rewrite_interactions_function_names(input, &HashMap::new());
    let root: Value = serde_json::from_slice(&output).unwrap();
    assert_eq!(
        root.pointer("/request/contents/0/parts/0/functionCall/name"),
        Some(&Value::String("true".into()))
    );
    assert_eq!(
        root.pointer("/request/toolConfig/functionCallingConfig/allowedFunctionNames/0"),
        Some(&Value::String("true".into()))
    );
}
