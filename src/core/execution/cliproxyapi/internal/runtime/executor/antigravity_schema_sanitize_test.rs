// ref: internal/runtime/executor/antigravity_schema_sanitize_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Candidate delta evidence: propertyNames regressions added by a88197f845c979132c8978ea223c6af05cc81536.
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use super::antigravity_executor_request::prepare_antigravity_generate_body;
use serde_json::{json, Value};

fn clean(body: Value) -> Value {
    clean_for_model(body, "claude-sonnet")
}

fn clean_for_model(body: Value, model: &str) -> Value {
    serde_json::from_slice(
        &prepare_antigravity_generate_body(&serde_json::to_vec(&body).unwrap(), model, "p")
            .unwrap(),
    )
    .unwrap()
}

fn property_names_shapes() -> [(&'static str, Value); 2] {
    [
        (
            "array_item",
            json!({
                "type": "object",
                "properties": {
                    "records": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {"name": {"type": "string"}},
                            "propertyNames": {"type": "string"}
                        }
                    }
                }
            }),
        ),
        (
            "property_named_properties",
            json!({
                "type": "object",
                "properties": {
                    "properties": {
                        "type": "object",
                        "propertyNames": {"type": "string"}
                    }
                }
            }),
        ),
    ]
}

#[test]
fn preserves_history_and_cleans_every_declaration_schema_location() {
    let history = json!([{"role":"model","parts":[{"functionCall":{"name":"f","args":{"x":1}}}]},{"role":"user","parts":[{"functionResponse":{"name":"f","response":{"ok":true}}}]}]);
    let out = clean(
        json!({"request":{"contents":history.clone(),"tools":[{"functionDeclarations":[{"name":"f","parameters":{"type":"object","properties":{"x":{"type":["string","null"],"pattern":"x"}},"required":["x"]},"response":{"anyOf":[{"type":"null"},{"type":"object","properties":{"ok":{"type":"boolean"}}}]}}]}]}}),
    );
    assert_eq!(out.pointer("/request/contents"), Some(&history));
    assert!(out
        .pointer("/request/tools/0/functionDeclarations/0/parameters/properties/x/pattern")
        .is_none());
    assert_eq!(
        out.pointer("/request/tools/0/functionDeclarations/0/parameters/required"),
        Some(&json!(["_"]))
    );
    assert_eq!(
        out.pointer("/request/tools/0/functionDeclarations/0/parameters/properties/_/type"),
        Some(&json!("boolean"))
    );
    assert_eq!(
        out.pointer("/request/tools/0/functionDeclarations/0/response/type"),
        Some(&json!("object"))
    );
}

#[test]
fn generation_response_schema_preserves_union_enum_metadata_without_placeholders() {
    let out = clean(
        json!({"request":{"contents":[],"generationConfig":{"responseMimeType":"application/json","responseSchema":{"oneOf":[{"type":"string"},{"type":"number"}],"enum":[1,"x"],"title":"keep","format":"drop"}}}}),
    );
    let schema = out
        .pointer("/request/generationConfig/responseSchema")
        .unwrap();
    assert!(schema.get("oneOf").is_some());
    assert_eq!(schema["title"], "keep");
    assert!(schema.get("format").is_none());
    assert!(schema.pointer("/properties/reason").is_none());
    assert_eq!(
        out.pointer("/request/generationConfig/responseMimeType"),
        Some(&json!("application/json"))
    );
}

#[test]
fn snake_case_response_schema_is_cleaned_and_operation_is_idempotent() {
    let input = json!({"request":{"contents":[],"generation_config":{"response_json_schema":{"type":"object","properties":{"x":{"type":"string","default":"bad"}}}}}});
    let once = clean(input);
    let twice = clean(once.clone());
    assert_eq!(
        once.pointer("/request/generation_config/response_json_schema"),
        twice.pointer("/request/generation_config/response_json_schema")
    );
    assert!(once
        .pointer("/request/generation_config/response_json_schema/properties/x/default")
        .is_none());
}

#[test]
fn strips_property_names_from_every_schema_location() {
    const DECLARATION_SCHEMA_KEYS: &[&str] = &[
        "parameters",
        "parametersJsonSchema",
        "parameters_json_schema",
        "response",
        "responseJsonSchema",
        "response_json_schema",
    ];
    const GENERATION_SCHEMA_KEYS: &[&str] = &[
        "responseSchema",
        "responseJsonSchema",
        "response_schema",
        "response_json_schema",
    ];

    for (shape_name, schema) in property_names_shapes() {
        for declaration_container in ["functionDeclarations", "function_declarations"] {
            for generation_container in ["generationConfig", "generation_config"] {
                let mut declaration = serde_json::Map::new();
                declaration.insert("name".into(), json!("t"));
                for key in DECLARATION_SCHEMA_KEYS {
                    declaration.insert((*key).into(), schema.clone());
                }
                let mut tool = serde_json::Map::new();
                tool.insert(
                    declaration_container.into(),
                    Value::Array(vec![Value::Object(declaration)]),
                );
                let mut generation = serde_json::Map::new();
                for key in GENERATION_SCHEMA_KEYS {
                    generation.insert((*key).into(), schema.clone());
                }
                let mut request = serde_json::Map::new();
                request.insert("tools".into(), Value::Array(vec![Value::Object(tool)]));
                request.insert(generation_container.into(), Value::Object(generation));

                for model in ["gemini-2.5-flash", "claude-opus-4-6"] {
                    let out =
                        clean_for_model(json!({"request": Value::Object(request.clone())}), model);
                    let encoded = serde_json::to_string(&out["request"]).unwrap();
                    assert!(
                        !encoded.contains("\"propertyNames\""),
                        "shape={shape_name}, declarations={declaration_container}, generation={generation_container}, model={model}: {encoded}"
                    );
                }
            }
        }
    }
}

#[test]
fn keeps_property_names_in_history_and_as_an_authored_property_name() {
    let history = json!([{
        "role": "model",
        "parts": [{
            "functionCall": {
                "name": "t",
                "args": {
                    "propertyNames": "keep-me",
                    "properties": {"propertyNames": "keep-me-too"}
                }
            }
        }]
    }]);
    let input = json!({
        "request": {
            "contents": history.clone(),
            "tools": [{
                "functionDeclarations": [{
                    "name": "t",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "propertyNames": {"type": "string"},
                            "properties": {
                                "type": "object",
                                "propertyNames": {"type": "string"}
                            }
                        }
                    }
                }]
            }]
        }
    });

    for model in ["gemini-2.5-flash", "claude-opus-4-6"] {
        let out = clean_for_model(input.clone(), model);
        assert_eq!(out.pointer("/request/contents"), Some(&history), "{model}");
        let schema = &out["request"]["tools"][0]["functionDeclarations"][0]["parameters"];
        assert!(
            schema.pointer("/properties/propertyNames").is_some(),
            "property named propertyNames was removed for {model}: {schema}"
        );
        assert!(
            schema
                .pointer("/properties/properties/propertyNames")
                .is_none(),
            "propertyNames keyword survived for {model}: {schema}"
        );
    }
}

#[test]
fn outbound_body_strips_property_names_but_keeps_function_arguments() {
    for (shape_name, schema) in property_names_shapes() {
        for model in ["gemini-3.1-pro", "claude-opus-4-6"] {
            let out = clean_for_model(
                json!({
                    "request": {
                        "contents": [{
                            "role": "model",
                            "parts": [{
                                "functionCall": {
                                    "name": "t",
                                    "args": {"propertyNames": "keep-me"}
                                }
                            }]
                        }],
                        "tools": [{
                            "function_declarations": [{
                                "name": "t",
                                "parametersJsonSchema": schema.clone()
                            }]
                        }],
                        "generationConfig": {"responseSchema": schema.clone()}
                    }
                }),
                model,
            );
            for pointer in ["/request/tools", "/request/generationConfig"] {
                let encoded = serde_json::to_string(out.pointer(pointer).unwrap()).unwrap();
                assert!(
                    !encoded.contains("\"propertyNames\""),
                    "shape={shape_name}, model={model}, path={pointer}: {encoded}"
                );
            }
            assert_eq!(
                out.pointer("/request/contents/0/parts/0/functionCall/args/propertyNames"),
                Some(&json!("keep-me")),
                "shape={shape_name}, model={model}"
            );
        }
    }
}
