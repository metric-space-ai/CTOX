// ref: internal/translator/gemini/openai/responses/*_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

//! Reproducible 22-case snapshot gate against the pinned Go implementation.
//! The expected document was emitted by the Go converter at the ref above;
//! fixture construction remains Rust-local so carrier envelopes are generated
//! by the same typed codec exercised by the production request path.

use std::collections::HashMap;

use serde_json::{json, Value};

use super::{
    convert_openai_responses_request_to_gemini,
    signature_carrier::{encode_carrier, CarrierDirection, CarrierTarget},
};

const NATIVE: &str = "EjQKMgEMOdbHO0Gd+c9Mxk4ELwPGbpCEcp2mFfYYLix2UVtBH3fL8GECc4+JITVnHF4qZDsA";

#[test]
fn pinned_go_request_differential_is_22_of_22() {
    let expected: Vec<Value> = serde_json::from_str(include_str!(
        "fixtures/gemini_responses_request_upstream_ffdb9c9f.expected.json"
    ))
    .expect("pinned Go request snapshot");
    let expected = expected
        .into_iter()
        .map(|entry| {
            (
                entry["name"].as_str().unwrap().to_owned(),
                entry["output"].clone(),
            )
        })
        .collect::<HashMap<_, _>>();
    let fixtures = fixtures();
    assert_eq!(fixtures.len(), 22);
    assert_eq!(expected.len(), 22);

    for (name, input) in fixtures {
        let actual: Value = serde_json::from_slice(&convert_openai_responses_request_to_gemini(
            "gemini-test",
            &input,
            false,
        ))
        .unwrap_or_else(|error| panic!("{name}: invalid Rust output: {error}"));
        assert_eq!(actual, expected[&name], "pinned Go mismatch for {name}");
    }
}

fn fixtures() -> Vec<(String, Vec<u8>)> {
    let next_text = encode_carrier(NATIVE, CarrierDirection::Next, CarrierTarget::Text);
    let next_function = encode_carrier(NATIVE, CarrierDirection::Next, CarrierTarget::Function);
    let previous_text = encode_carrier(NATIVE, CarrierDirection::Previous, CarrierTarget::Text);
    let previous_function =
        encode_carrier(NATIVE, CarrierDirection::Previous, CarrierTarget::Function);
    let standalone = encode_carrier(NATIVE, CarrierDirection::Standalone, CarrierTarget::Any);
    vec![
        fixture("gemini-simple-string", json!({"input":"hello"})),
        fixture(
            "gemini-system-media-generation",
            json!({
                "input":[
                    {"type":"message","role":"system","content":[{"type":"input_text","text":"Be precise."}]},
                    {"type":"message","role":"developer","content":[{"type":"input_text","text":"Use evidence."}]},
                    {"type":"message","role":"user","content":[
                        {"type":"input_text","text":"inspect"},
                        {"type":"input_image","image_url":"data:image/png;base64,aW1n"},
                        {"type":"input_audio","format":"mp3","data":"YXVkaW8="}
                    ]}
                ],
                "max_output_tokens":42,"temperature":0.5,"top_p":0.9,
                "stop_sequences":["END","STOP"],"reasoning":{"effort":"auto"}
            }),
        ),
        fixture(
            "gemini-role-split-and-trailing-prefill",
            json!({"input":[
                {"type":"message","role":"user","content":[{"type":"input_text","text":"question"}]},
                {"type":"message","role":"assistant","content":[{"type":"output_text","text":"historical answer"}]},
                {"type":"message","role":"user","content":[{"type":"input_text","text":"follow-up"}]},
                {"type":"message","role":"assistant","content":[{"type":"output_text","text":"prefill"}]}
            ]}),
        ),
        fixture(
            "gemini-tools-and-structured-output",
            json!({
                "input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"return JSON"}]}],
                "reasoning":{"effort":"HIGH"},
                "text":{"format":{"type":"json_schema","schema":{"type":"object","properties":{"ok":{"type":"boolean"}},"required":["ok"]}}},
                "tools":[{"type":"function","name":"1 read file","description":"Read","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}]
            }),
        ),
        fixture(
            "gemini-parallel-functions-output-order",
            json!({"input":[
                {"type":"function_call","call_id":"call-a","name":"first","arguments":"{\"x\":1}"},
                {"type":"function_call","call_id":"call-b","name":"second","arguments":"{}"},
                {"type":"function_call_output","call_id":"call-b","output":"plain"},
                {"type":"function_call_output","call_id":"call-a","output":"{\"ok\":true}"}
            ]}),
        ),
        fixture(
            "gemini-reasoning-and-function",
            json!({"input":[
                {"type":"reasoning","summary":[{"type":"summary_text","text":"consider"}]},
                {"type":"function_call","call_id":"call-run","name":"run","arguments":"{}"},
                {"type":"function_call_output","call_id":"call-run","output":"done"}
            ]}),
        ),
        fixture(
            "gemini-native-leading-function-signature",
            json!({"input":[
                {"type":"reasoning","encrypted_content":NATIVE,"summary":[]},
                {"type":"function_call","call_id":"call-native","name":"run","arguments":"{}"},
                {"type":"function_call_output","call_id":"call-native","output":"ok"}
            ]}),
        ),
        fixture(
            "gemini-marked-leading-text-carrier-alias",
            json!({"input":[
                {"type":"reasoning","encrypted_content":next_text,"summary":[]},
                {"type":"message","role":"assistant","content":[{"type":"output_text","text":"answer"}]}
            ]}),
        ),
        fixture(
            "gemini-marked-leading-function-carrier-alias",
            json!({"input":[
                {"type":"reasoning","encrypted_content":next_function,"summary":[]},
                {"type":"function_call","call_id":"call-carrier","name":"run","arguments":"{}"},
                {"type":"function_call_output","call_id":"call-carrier","output":"ok"}
            ]}),
        ),
        fixture(
            "gemini-malformed-carrier-drops-without-crossing-message",
            json!({"input":[
                {"type":"reasoning","encrypted_content":"cpa-gemini-responses-carrier-v1:previous:text:not-base64!","summary":[]},
                {"type":"message","role":"user","content":[{"type":"input_text","text":"safe"}]}
            ]}),
        ),
        fixture(
            "gemini-marked-previous-function-carrier",
            json!({"input":[
                {"type":"function_call","call_id":"call-prev","name":"run","arguments":"{}"},
                {"type":"reasoning","encrypted_content":previous_function,"summary":[]},
                {"type":"function_call_output","call_id":"call-prev","output":"ok"}
            ]}),
        ),
        fixture(
            "gemini-unmarked-post-call-carrier-matching-output",
            json!({"input":[
                {"type":"function_call","call_id":"call-raw-prev","name":"run","arguments":"{}"},
                {"type":"reasoning","encrypted_content":NATIVE,"summary":[]},
                {"type":"function_call_output","call_id":"call-raw-prev","output":"ok"}
            ]}),
        ),
        fixture(
            "gemini-marked-previous-text-carrier",
            json!({"input":[
                {"type":"message","role":"assistant","content":[{"type":"output_text","text":"signed answer"}]},
                {"type":"reasoning","encrypted_content":previous_text,"summary":[]},
                {"type":"message","role":"user","content":[{"type":"input_text","text":"next"}]}
            ]}),
        ),
        fixture(
            "gemini-standalone-function-carrier-stays-separate",
            json!({"input":[
                {"type":"reasoning","encrypted_content":standalone,"summary":[]},
                {"type":"function_call","call_id":"call-standalone","name":"run","arguments":"{}"}
            ]}),
        ),
        fixture(
            "gemini-tool-schema-required-and-metadata",
            schema_fixture(
                "clean schema",
                "search",
                json!({
                    "type":"object","title":"Root","$schema":"draft",
                    "properties":{"country":{"type":"string","title":"Country"},"$id":{"type":"string"}},
                    "required":["country","$id","stale"]
                }),
            ),
        ),
        fixture(
            "gemini-tool-schema-enum-constraints",
            schema_fixture(
                "clean enum",
                "rank",
                json!({
                    "type":"object","additionalProperties":false,
                    "properties":{"priority":{"type":"integer","enum":[0,1,2],"minimum":0,"format":"int32"}}
                }),
            ),
        ),
        fixture(
            "gemini-tool-schema-union-nullable",
            schema_fixture(
                "clean union",
                "choose",
                json!({
                    "type":"object",
                    "properties":{
                        "choice":{"description":"pick","anyOf":[{"type":"string"},{"type":"object","properties":{"id":{"type":"string"}}},{"type":"null"}]},
                        "maybe":{"type":["string","null"]}
                    },
                    "required":["choice","maybe"]
                }),
            ),
        ),
        fixture(
            "gemini-tool-schema-allof-and-ref",
            schema_fixture(
                "clean composition",
                "compose",
                json!({
                    "type":"object","$defs":{"Nested":{"type":"object"}},
                    "allOf":[
                        {"properties":{"name":{"type":"string"}},"required":["name"]},
                        {"properties":{"nested":{"$ref":"#/$defs/Nested","description":"old"}},"required":["nested"]}
                    ]
                }),
            ),
        ),
        fixture(
            "gemini-tool-schema-removes-placeholders",
            json!({
                "input":"clean placeholders",
                "tools":[
                    {"type":"function","name":"placeholder","parameters":{"type":"object","properties":{"_":{"type":"boolean"}},"required":["_"]}},
                    {"type":"function","name":"reason","parameters":{"type":"object","properties":{"reason":{"type":"string","description":"Brief explanation of why you are calling this tool"}},"required":["reason"]}}
                ]
            }),
        ),
        fixture(
            "gemini-schema-cleaning-is-path-local",
            json!({
                "input":[
                    {"type":"function_call","call_id":"history","name":"write","arguments":"{\"title\":\"kept\",\"format\":\"markdown\",\"default\":\"x\",\"const\":\"c\",\"x-custom\":\"keep\"}"},
                    {"type":"function_call_output","call_id":"history","output":"ok"}
                ],
                "tools":[{"type":"function","name":"write","parameters":{"type":"object","title":"removed","properties":{"title":{"type":"string"}},"required":["title"]}}],
                "text":{"format":{"type":"json_schema","schema":{"type":"object","title":"response-kept","properties":{"value":{"type":"string","format":"uri"}}}}}
            }),
        ),
        fixture(
            "gemini-alternating-post-call-carriers",
            json!({"input":[
                {"type":"function_call","call_id":"post-a","name":"a","arguments":"{}"},
                {"type":"reasoning","encrypted_content":NATIVE,"summary":[]},
                {"type":"function_call","call_id":"post-b","name":"b","arguments":"{}"},
                {"type":"reasoning","encrypted_content":NATIVE,"summary":[]},
                {"type":"function_call_output","call_id":"post-a","output":"a-ok"},
                {"type":"function_call_output","call_id":"post-b","output":"b-ok"}
            ]}),
        ),
        fixture(
            "gemini-alternating-leading-carriers",
            json!({"input":[
                {"type":"reasoning","encrypted_content":next_function,"summary":[]},
                {"type":"function_call","call_id":"lead-a","name":"a","arguments":"{}"},
                {"type":"reasoning","encrypted_content":next_function,"summary":[]},
                {"type":"function_call","call_id":"lead-b","name":"b","arguments":"{}"},
                {"type":"function_call_output","call_id":"lead-a","output":"a-ok"},
                {"type":"function_call_output","call_id":"lead-b","output":"b-ok"}
            ]}),
        ),
    ]
}

fn schema_fixture(prompt: &str, name: &str, parameters: Value) -> Value {
    json!({
        "input":prompt,
        "tools":[{"type":"function","name":name,"parameters":parameters}]
    })
}

fn fixture(name: &str, input: Value) -> (String, Vec<u8>) {
    (
        name.to_owned(),
        serde_json::to_vec(&input).expect("fixture JSON"),
    )
}
