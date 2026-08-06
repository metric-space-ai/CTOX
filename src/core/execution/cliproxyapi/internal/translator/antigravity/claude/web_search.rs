// ref: internal/translator/antigravity/claude/web_search.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use serde_json::{json, Value};
use std::collections::HashSet;

pub const ANTIGRAVITY_WEB_SEARCH_SYSTEM_INSTRUCTION: &str = "You are a search engine bot. You will be given a query from a user. Your task is to search the web for relevant information that will help the user. You MUST perform a web search. Do not respond or interact with the user, please respond as if they typed the query into a search bar.";

pub fn is_claude_typed_web_search_tool_type(tool_type: &str) -> bool {
    matches!(tool_type, "web_search_20250305" | "web_search_20260209")
}

pub fn should_build_antigravity_web_search_request(
    supports_native_google_search: bool,
    payload: &Value,
) -> bool {
    supports_native_google_search
        && has_only_claude_typed_web_search_tools(payload)
        && allows_claude_web_search_tool_choice(payload)
}

pub fn build_antigravity_web_search_request(model: &str, payload: &Value) -> Value {
    let mut google_search = json!({
        "enhancedContent": {
            "imageSearch": {
                "maxResultCount": extract_claude_web_search_max_uses(payload)
            }
        }
    });
    let domains = extract_claude_web_search_allowed_domains(payload);
    if !domains.is_empty() {
        google_search["includedDomains"] =
            Value::Array(domains.into_iter().map(Value::String).collect());
    }
    json!({
        "model": model,
        "requestType": "web_search",
        "request": {
            "contents": [{
                "role": "user",
                "parts": [{"text": extract_claude_web_search_query(payload)}]
            }],
            "systemInstruction": {
                "role": "user",
                "parts": [{"text": ANTIGRAVITY_WEB_SEARCH_SYSTEM_INSTRUCTION}]
            },
            "tools": [{"googleSearch": google_search}],
            "generationConfig": {"candidateCount": 1}
        }
    })
}

fn has_only_claude_typed_web_search_tools(payload: &Value) -> bool {
    let Some(tools) = payload.get("tools").and_then(Value::as_array) else {
        return false;
    };
    let mut has_web_search = false;
    for tool in tools {
        if is_claude_typed_web_search_tool_type(
            tool.get("type").and_then(Value::as_str).unwrap_or_default(),
        ) {
            has_web_search = true;
        } else {
            return false;
        }
    }
    has_web_search
}

fn allows_claude_web_search_tool_choice(payload: &Value) -> bool {
    let Some(choice) = payload.get("tool_choice") else {
        return true;
    };
    match choice {
        Value::String(kind) => matches!(kind.as_str(), "" | "auto" | "any"),
        Value::Object(choice) => match choice
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or_default()
        {
            "" | "auto" | "any" => true,
            "tool" => choice.get("name").and_then(Value::as_str) == Some("web_search"),
            _ => false,
        },
        _ => false,
    }
}

fn extract_claude_web_search_max_uses(payload: &Value) -> i64 {
    for tool in web_search_tools(payload) {
        let max_uses = tool
            .get("max_uses")
            .and_then(|value| {
                value
                    .as_i64()
                    .or_else(|| value.as_str().and_then(|value| value.parse().ok()))
            })
            .unwrap_or_default();
        if max_uses > 0 {
            return max_uses;
        }
    }
    5
}

fn extract_claude_web_search_allowed_domains(payload: &Value) -> Vec<String> {
    let Some(tool) = web_search_tools(payload).next() else {
        return Vec::new();
    };
    tool.get("allowed_domains")
        .and_then(Value::as_array)
        .map(|domains| {
            domains
                .iter()
                .filter_map(Value::as_str)
                .map(str::trim)
                .filter(|domain| !domain.is_empty())
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default()
}

fn extract_claude_web_search_query(payload: &Value) -> String {
    let Some(messages) = payload.get("messages").and_then(Value::as_array) else {
        return String::new();
    };
    for message in messages.iter().rev() {
        let role = message
            .get("role")
            .and_then(Value::as_str)
            .unwrap_or_default();
        if !matches!(role, "" | "user") {
            continue;
        }
        let query = extract_claude_text_content(message.get("content"));
        if !query.is_empty() {
            return query;
        }
    }
    String::new()
}

fn extract_claude_text_content(content: Option<&Value>) -> String {
    match content {
        Some(Value::String(text)) => text.trim().to_owned(),
        Some(Value::Array(parts)) => parts
            .iter()
            .filter_map(|part| part.get("text").and_then(Value::as_str))
            .map(str::trim)
            .filter(|text| !text.is_empty())
            .collect::<Vec<_>>()
            .join("\n")
            .trim()
            .to_owned(),
        _ => String::new(),
    }
}

fn web_search_tools(payload: &Value) -> impl Iterator<Item = &Value> {
    payload
        .get("tools")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|tool| {
            is_claude_typed_web_search_tool_type(
                tool.get("type").and_then(Value::as_str).unwrap_or_default(),
            )
        })
}

pub fn should_translate_web_search_grounding(
    original_request: &Value,
    translated_request: &Value,
) -> bool {
    has_claude_typed_web_search_tool(original_request)
        && has_antigravity_google_search_tool(translated_request)
}

pub fn antigravity_grounding_metadata(root: &Value) -> Option<&Value> {
    root.pointer("/response/candidates/0/groundingMetadata")
        .or_else(|| root.pointer("/candidates/0/groundingMetadata"))
}

pub fn antigravity_text_content(root: &Value) -> String {
    root.pointer("/response/candidates/0/content/parts")
        .or_else(|| root.pointer("/candidates/0/content/parts"))
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|part| part.get("text"))
        .map(value_string)
        .collect()
}

pub fn build_claude_web_search_content(
    tool_use_id: &str,
    text_content: &str,
    grounding_metadata: &Value,
) -> Vec<Value> {
    let mut server_tool_use = json!({
        "type":"server_tool_use",
        "id":tool_use_id,
        "name":"web_search",
        "input":{}
    });
    if let Some(query) = grounding_metadata
        .get("webSearchQueries")
        .and_then(Value::as_array)
        .and_then(|queries| queries.first())
        .map(value_string)
        .filter(|query| !query.is_empty())
    {
        server_tool_use["input"]["query"] = Value::String(query);
    }

    let mut content = vec![server_tool_use];
    content.push(json!({
        "type":"web_search_tool_result",
        "tool_use_id":tool_use_id,
        "content":web_search_results_from_grounding(grounding_metadata)
    }));
    content.extend(
        build_web_search_cited_text_blocks(text_content, grounding_metadata)
            .into_iter()
            .filter(|block| !block.text.is_empty())
            .map(|block| {
                let mut value = json!({"type":"text","text":block.text});
                if !block.citations.is_empty() {
                    value["citations"] = Value::Array(block.citations);
                }
                value
            }),
    );
    content
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct CitedTextBlock {
    text: String,
    citations: Vec<Value>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct GroundingSupport {
    start_index: i64,
    end_index: i64,
    chunk_urls: Vec<String>,
    chunk_title: String,
}

fn has_claude_typed_web_search_tool(payload: &Value) -> bool {
    web_search_tools(payload).next().is_some()
}

fn has_antigravity_google_search_tool(payload: &Value) -> bool {
    payload
        .pointer("/request/tools")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .any(|tool| tool.get("googleSearch").is_some())
}

fn web_search_results_from_grounding(metadata: &Value) -> Vec<Value> {
    let mut seen = HashSet::new();
    metadata
        .get("groundingChunks")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|chunk| chunk.get("web"))
        .filter_map(|web| {
            let uri = web
                .get("uri")
                .map(value_string)
                .unwrap_or_default()
                .trim()
                .to_owned();
            if uri.is_empty() || !seen.insert(uri.clone()) {
                return None;
            }
            let mut result = json!({
                "type":"web_search_result",
                "page_age":null,
                "url":uri
            });
            if let Some(title) = web.get("title") {
                result["title"] = Value::String(value_string(title));
            }
            Some(result)
        })
        .collect()
}

fn build_web_search_cited_text_blocks(text: &str, metadata: &Value) -> Vec<CitedTextBlock> {
    let supports = parse_grounding_supports(metadata);
    if supports.is_empty() {
        return (!text.is_empty())
            .then(|| CitedTextBlock {
                text: text.to_owned(),
                citations: Vec::new(),
            })
            .into_iter()
            .collect();
    }
    let bytes = text.as_bytes();
    let mut blocks = Vec::with_capacity(supports.len() + 1);
    let mut last_end = 0_i64;
    for support in supports {
        if support.end_index <= last_end {
            continue;
        }
        if support.start_index > last_end {
            push_byte_block(
                &mut blocks,
                bytes,
                last_end,
                support.start_index,
                Vec::new(),
            );
        }
        let cited_start = support.start_index.max(last_end);
        let cited_text = byte_text(bytes, cited_start, support.end_index);
        if !cited_text.is_empty() && !support.chunk_urls.is_empty() {
            let citation = json!({
                "type":"web_search_result_location",
                "cited_text":cited_text,
                "url":support.chunk_urls[0],
                "title":support.chunk_title
            });
            blocks.push(CitedTextBlock {
                text: citation["cited_text"]
                    .as_str()
                    .unwrap_or_default()
                    .to_owned(),
                citations: vec![citation],
            });
        }
        last_end = support.end_index.max(last_end);
    }
    if last_end < i64::try_from(bytes.len()).unwrap_or(i64::MAX) {
        push_byte_block(
            &mut blocks,
            bytes,
            last_end,
            i64::try_from(bytes.len()).unwrap_or(i64::MAX),
            Vec::new(),
        );
    }
    blocks
}

fn parse_grounding_supports(metadata: &Value) -> Vec<GroundingSupport> {
    let chunks = metadata
        .get("groundingChunks")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let chunk_data = chunks
        .iter()
        .map(|chunk| {
            let web = chunk.get("web").unwrap_or(&Value::Null);
            (
                web.get("uri").map(value_string).unwrap_or_default(),
                web.get("title").map(value_string).unwrap_or_default(),
            )
        })
        .collect::<Vec<_>>();
    metadata
        .get("groundingSupports")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|support| {
            let segment = support.get("segment")?;
            let mut parsed = GroundingSupport {
                start_index: integer(segment.get("startIndex")),
                end_index: integer(segment.get("endIndex")),
                chunk_urls: Vec::new(),
                chunk_title: String::new(),
            };
            for index in support
                .get("groundingChunkIndices")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(|index| usize::try_from(integer(Some(index))).ok())
            {
                let Some((url, title)) = chunk_data.get(index) else {
                    continue;
                };
                parsed.chunk_urls.push(url.clone());
                if parsed.chunk_title.is_empty() {
                    parsed.chunk_title.clone_from(title);
                }
            }
            Some(parsed)
        })
        .collect()
}

fn push_byte_block(
    blocks: &mut Vec<CitedTextBlock>,
    bytes: &[u8],
    start: i64,
    end: i64,
    citations: Vec<Value>,
) {
    let text = byte_text(bytes, start, end);
    if !text.is_empty() {
        blocks.push(CitedTextBlock { text, citations });
    }
}

fn byte_text(bytes: &[u8], start: i64, end: i64) -> String {
    let start = usize::try_from(start.max(0))
        .unwrap_or(usize::MAX)
        .min(bytes.len());
    let end = usize::try_from(end.max(0))
        .unwrap_or(usize::MAX)
        .min(bytes.len());
    if start < end {
        String::from_utf8_lossy(&bytes[start..end]).into_owned()
    } else {
        String::new()
    }
}

fn integer(value: Option<&Value>) -> i64 {
    value
        .and_then(|value| {
            value
                .as_i64()
                .or_else(|| value.as_str().and_then(|value| value.parse().ok()))
        })
        .unwrap_or_default()
}

fn value_string(value: &Value) -> String {
    match value {
        Value::String(value) => value.clone(),
        Value::Null => String::new(),
        other => other.to_string(),
    }
}

// Stream state and SSE framing live beside the wider response state machine in
// antigravity_claude_response.rs. Capability discovery and collision-free tool
// IDs are injected at typed CTOX lifecycle boundaries instead of global lookup.
