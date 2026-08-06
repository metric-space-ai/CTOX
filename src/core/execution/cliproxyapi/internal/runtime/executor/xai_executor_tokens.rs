// ref: internal/runtime/executor/xai_executor_tokens.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use serde_json::Value;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum XaiTokenCountError {
    InvalidJson,
    Tokenizer,
}

pub fn count_xai_input_tokens(body: &[u8]) -> Result<u64, XaiTokenCountError> {
    if body.is_empty() {
        return Ok(0);
    }
    let root: Value = serde_json::from_slice(body).map_err(|_| XaiTokenCountError::InvalidJson)?;
    let mut segments = Vec::new();
    append_string(&mut segments, root.get("instructions"));
    collect_input(root.get("input"), &mut segments);
    collect_tools(root.get("tools"), &mut segments);
    if let Some(format) = root.pointer("/text/format") {
        append_string(&mut segments, format.get("name"));
        append_json(&mut segments, format.get("schema"));
    }
    if segments.is_empty() {
        return Ok(0);
    }
    let bpe = tiktoken_rs::o200k_base().map_err(|_| XaiTokenCountError::Tokenizer)?;
    Ok(bpe.encode_with_special_tokens(&segments.join("\n")).len() as u64)
}

#[must_use]
pub fn xai_token_count_response(count: u64) -> Vec<u8> {
    serde_json::to_vec(&serde_json::json!({"response":{"usage":{"input_tokens":count,"output_tokens":0,"total_tokens":count}}})).expect("static response is serializable")
}

fn collect_input(input: Option<&Value>, segments: &mut Vec<String>) {
    if let Some(text) = input.and_then(Value::as_str) {
        append_text(segments, text);
        return;
    }
    let Some(input) = input.and_then(Value::as_array) else {
        return;
    };
    for item in input {
        match item.get("type").and_then(Value::as_str).unwrap_or_default() {
            "message" => collect_content(item.get("content"), segments),
            "function_call" => {
                append_string(segments, item.get("name"));
                append_json(segments, item.get("arguments"));
            }
            "function_call_output" => append_json(segments, item.get("output")),
            "reasoning" => {
                if let Some(summary) = item.get("summary").and_then(Value::as_array) {
                    for part in summary {
                        append_string(segments, part.get("text"));
                    }
                }
            }
            _ => {}
        }
    }
}

fn collect_content(content: Option<&Value>, segments: &mut Vec<String>) {
    if let Some(text) = content.and_then(Value::as_str) {
        append_text(segments, text);
        return;
    }
    let Some(content) = content.and_then(Value::as_array) else {
        return;
    };
    for part in content {
        match part.get("type").and_then(Value::as_str).unwrap_or_default() {
            "text" | "input_text" | "output_text" => append_string(segments, part.get("text")),
            "refusal" => append_string(segments, part.get("refusal")),
            "input_image" => {
                append_string(segments, part.get("image_url"));
                append_string(segments, part.get("file_id"));
            }
            "input_file" => {
                for key in ["file_data", "file_url", "file_id", "filename"] {
                    append_string(segments, part.get(key));
                }
            }
            "input_audio" => {
                append_string(segments, part.get("data"));
                append_string(segments, part.pointer("/input_audio/data"));
            }
            _ => {}
        }
    }
}

fn collect_tools(tools: Option<&Value>, segments: &mut Vec<String>) {
    let Some(tools) = tools.and_then(Value::as_array) else {
        return;
    };
    for tool in tools
        .iter()
        .filter(|tool| tool.get("type").and_then(Value::as_str) == Some("function"))
    {
        append_string(segments, tool.get("name"));
        append_string(segments, tool.get("description"));
        append_json(segments, tool.get("parameters"));
    }
}
fn append_string(segments: &mut Vec<String>, value: Option<&Value>) {
    if let Some(text) = value.and_then(Value::as_str) {
        append_text(segments, text);
    }
}
fn append_text(segments: &mut Vec<String>, text: &str) {
    let text = text.trim();
    if !text.is_empty() {
        segments.push(text.to_owned());
    }
}
fn append_json(segments: &mut Vec<String>, value: Option<&Value>) {
    if let Some(value) = value {
        if let Some(text) = value.as_str() {
            append_text(segments, text);
        } else {
            append_text(segments, &value.to_string());
        }
    }
}
