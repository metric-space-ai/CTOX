// ref: internal/translator/antigravity/openai/responses/antigravity_openai-responses_request.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use serde_json::{json, Map, Value};

use crate::internal::signature::{
    compatible_antigravity_claude_thinking_signature, signature_payload_without_provider_prefix,
    signature_provider_from_model_name, SignatureProvider,
};

const SAFETY_CATEGORIES: &[(&str, &str)] = &[
    ("HARM_CATEGORY_HARASSMENT", "OFF"),
    ("HARM_CATEGORY_HATE_SPEECH", "OFF"),
    ("HARM_CATEGORY_SEXUALLY_EXPLICIT", "OFF"),
    ("HARM_CATEGORY_DANGEROUS_CONTENT", "OFF"),
    ("HARM_CATEGORY_CIVIC_INTEGRITY", "BLOCK_NONE"),
];

pub fn convert_openai_responses_request_to_antigravity(
    model_name: &str,
    input_raw_json: &[u8],
    _stream: bool,
) -> Vec<u8> {
    let Ok(root) = serde_json::from_slice::<Value>(input_raw_json) else {
        return input_raw_json.to_vec();
    };
    let mut request = Map::new();
    request.insert(
        "contents".to_owned(),
        Value::Array(convert_contents(model_name, &root)),
    );
    if let Some(instructions) = root.get("instructions").and_then(Value::as_str) {
        request.insert(
            "systemInstruction".to_owned(),
            json!({"parts":[{"text":instructions}]}),
        );
    }
    if let Some(tools) = convert_tools(&root) {
        request.insert("tools".to_owned(), tools);
    }
    let mut generation = Map::new();
    if let Some(value) = root.get("max_output_tokens").and_then(Value::as_u64) {
        generation.insert("maxOutputTokens".to_owned(), Value::from(value));
    }
    for (source, target) in [("temperature", "temperature"), ("top_p", "topP")] {
        if let Some(value) = root.get(source).and_then(Value::as_f64) {
            generation.insert(target.to_owned(), Value::from(value));
        }
    }
    if !generation.is_empty() {
        request.insert("generationConfig".to_owned(), Value::Object(generation));
    }
    request.insert(
        "safetySettings".to_owned(),
        Value::Array(
            SAFETY_CATEGORIES
                .iter()
                .map(|(category, threshold)| json!({"category":category,"threshold":threshold}))
                .collect(),
        ),
    );
    serde_json::to_vec(&json!({"project":"","request":request,"model":model_name}))
        .unwrap_or_default()
}

fn convert_contents(model_name: &str, root: &Value) -> Vec<Value> {
    let mut contents = Vec::new();
    let Some(input) = root.get("input") else {
        return contents;
    };
    if let Some(text) = input.as_str() {
        contents.push(json!({"role":"user","parts":[{"text":text}]}));
        return contents;
    }
    let provider = signature_provider_from_model_name(model_name);
    for item in input.as_array().into_iter().flatten() {
        let kind = item.get("type").and_then(Value::as_str).unwrap_or_else(|| {
            if item.get("role").is_some() {
                "message"
            } else {
                ""
            }
        });
        match kind {
            "message" => {
                let role = match item.get("role").and_then(Value::as_str).unwrap_or("user") {
                    "assistant" | "model" => "model",
                    role => role,
                };
                let mut parts = Vec::new();
                if let Some(text) = item.get("content").and_then(Value::as_str) {
                    parts.push(json!({"text":text}));
                }
                for part in item
                    .get("content")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                {
                    match part
                        .get("type")
                        .and_then(Value::as_str)
                        .unwrap_or("input_text")
                    {
                        "input_text" | "output_text" => {
                            if let Some(text) = part.get("text").and_then(Value::as_str) {
                                parts.push(json!({"text":text}));
                            }
                        }
                        "input_image" => {
                            if let Some((mime, data)) = data_url(
                                part.get("image_url")
                                    .or_else(|| part.get("url"))
                                    .and_then(Value::as_str)
                                    .unwrap_or(""),
                            ) {
                                parts.push(json!({"inline_data":{"mime_type":mime,"data":data}}));
                            }
                        }
                        _ => {}
                    }
                }
                if !parts.is_empty() {
                    contents.push(json!({"role":role,"parts":parts}));
                }
            }
            "reasoning" => {
                let text = item
                    .pointer("/summary/0/text")
                    .and_then(Value::as_str)
                    .unwrap_or("");
                let raw = item
                    .get("encrypted_content")
                    .and_then(Value::as_str)
                    .unwrap_or("");
                let signature = match provider {
                    SignatureProvider::Claude => {
                        compatible_antigravity_claude_thinking_signature(raw)
                    }
                    SignatureProvider::Gemini => {
                        let payload = signature_payload_without_provider_prefix(raw);
                        (!payload.is_empty()).then(|| payload.to_owned())
                    }
                    _ => None,
                };
                if !text.trim().is_empty() {
                    if let Some(signature) = signature {
                        contents.push(json!({"role":"model","parts":[{"thought":true,"thoughtSignature":signature,"text":text}]}));
                    }
                }
            }
            "function_call" => {
                let name = item.get("name").and_then(Value::as_str).unwrap_or("");
                let call_id = item.get("call_id").and_then(Value::as_str).unwrap_or("");
                let args = item
                    .get("arguments")
                    .and_then(Value::as_str)
                    .and_then(|raw| serde_json::from_str::<Value>(raw).ok())
                    .unwrap_or_else(|| json!({}));
                contents.push(json!({"role":"model","parts":[{"functionCall":{"id":call_id,"name":name,"args":args}}]}));
            }
            "function_call_output" => {
                let call_id = item.get("call_id").and_then(Value::as_str).unwrap_or("");
                let output = item.get("output").cloned().unwrap_or(Value::Null);
                contents.push(json!({"role":"user","parts":[{"functionResponse":{"id":call_id,"name":"","response":{"output":output}}}]}));
            }
            _ => {}
        }
    }
    contents
}

fn convert_tools(root: &Value) -> Option<Value> {
    let declarations: Vec<Value> = root
        .get("tools")?
        .as_array()?
        .iter()
        .filter(|tool| tool.get("type").and_then(Value::as_str) == Some("function"))
        .map(|tool| {
            json!({
                "name": tool.get("name").and_then(Value::as_str).unwrap_or(""),
                "description": tool.get("description").and_then(Value::as_str).unwrap_or(""),
                "parametersJsonSchema": tool.get("parameters").cloned().unwrap_or_else(|| json!({}))
            })
        })
        .collect();
    (!declarations.is_empty()).then(|| json!([{"functionDeclarations":declarations}]))
}

fn data_url(raw: &str) -> Option<(&str, &str)> {
    let value = raw.strip_prefix("data:")?;
    value
        .split_once(";base64,")
        .or_else(|| value.split_once(','))
}

#[cfg(test)]
mod tests {
    use super::*;
    use base64::{engine::general_purpose, Engine as _};

    fn claude_signature() -> String {
        let mut channel = vec![0x08, 0x0c, 0x10, 0x02, 0x32, 0x11];
        channel.extend_from_slice(b"claude-sonnet-4-6");
        let mut container = vec![0x0a, channel.len() as u8];
        container.extend_from_slice(&channel);
        let mut payload = vec![0x12, container.len() as u8];
        payload.extend_from_slice(&container);
        payload.extend_from_slice(&[0x18, 0x01]);
        general_purpose::STANDARD.encode(payload)
    }

    #[test]
    fn wraps_text_tools_generation_and_safety() {
        let output: Value = serde_json::from_slice(&convert_openai_responses_request_to_antigravity(
            "gemini-3-flash-agent",
            br#"{"instructions":"precise","input":[{"role":"user","content":[{"type":"input_text","text":"hello"}]}],"tools":[{"type":"function","name":"read","description":"Read","parameters":{"type":"object"}}],"max_output_tokens":42,"temperature":0.5}"#,
            false,
        )).unwrap();
        assert_eq!(output["model"], "gemini-3-flash-agent");
        assert_eq!(
            output["request"]["contents"][0]["parts"][0]["text"],
            "hello"
        );
        assert_eq!(
            output["request"]["systemInstruction"]["parts"][0]["text"],
            "precise"
        );
        assert_eq!(
            output["request"]["tools"][0]["functionDeclarations"][0]["name"],
            "read"
        );
        assert_eq!(output["request"]["generationConfig"]["maxOutputTokens"], 42);
        assert_eq!(
            output["request"]["safetySettings"]
                .as_array()
                .unwrap()
                .len(),
            5
        );
    }

    #[test]
    fn claude_reasoning_is_wrapped_once_and_incompatible_or_empty_is_dropped() {
        let native = claude_signature();
        let input = json!({"input":[
            {"type":"reasoning","encrypted_content":native,"summary":[{"type":"summary_text","text":"think"}]},
            {"type":"reasoning","encrypted_content":"gpt#invalid","summary":[{"type":"summary_text","text":"leak"}]},
            {"type":"reasoning","encrypted_content":native,"summary":[]},
            {"role":"user","content":[{"type":"input_text","text":"continue"}]}
        ]});
        let output: Value =
            serde_json::from_slice(&convert_openai_responses_request_to_antigravity(
                "claude-opus-4-6-thinking",
                &serde_json::to_vec(&input).unwrap(),
                false,
            ))
            .unwrap();
        let contents = output["request"]["contents"].as_array().unwrap();
        assert_eq!(contents.len(), 2);
        assert_eq!(contents[0]["parts"][0]["text"], "think");
        let expected = general_purpose::STANDARD.encode(claude_signature().as_bytes());
        assert_eq!(contents[0]["parts"][0]["thoughtSignature"], expected);
        assert!(!serde_json::to_string(&output).unwrap().contains("leak"));
    }
}
