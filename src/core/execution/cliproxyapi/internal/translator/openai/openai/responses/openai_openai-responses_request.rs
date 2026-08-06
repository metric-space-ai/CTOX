// ref: internal/translator/openai/openai/responses/openai_openai-responses_request.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

//! Converts OpenAI Responses requests into the OpenAI Chat Completions
//! shape consumed by the upstream provider adapter. The translation
//! preserves every Responses input item, flattens namespace/custom
//! tool declarations, attaches reasoning content to the assistant
//! message that follows, and keeps `function_call` adjacency strict
//! when there are pending tool outputs.

use crate::internal::translator::common::{join_raw_array, set_raw_array_items};
use serde_json::{json, Value};

use super::tools::{
    convert_responses_tool_to_openai_chat_tools, qualify_responses_namespace_tool_name,
    responses_tool_output_text,
};

pub fn convert_openai_responses_request_to_openai_chat_completions(
    model_name: &str,
    input_raw_json: &[u8],
    stream: bool,
) -> Vec<u8> {
    let mut out: Value = json!({"model":"","messages":[],"stream":false});
    let root: Value = serde_json::from_slice(input_raw_json).unwrap_or(Value::Null);

    out["model"] = Value::String(model_name.to_string());
    out["stream"] = Value::Bool(stream);

    if let Some(max_tokens) = root.get("max_output_tokens") {
        if let Some(value) = max_tokens.as_i64() {
            out["max_tokens"] = Value::Number(value.into());
        }
    }

    let mut messages: Vec<Vec<u8>> = Vec::new();

    if let Some(instructions) = root.get("instructions") {
        if let Some(text) = instructions.as_str() {
            let message = json!({"role":"system","content":text});
            push_message(
                &mut messages,
                serde_json::to_vec(&message).unwrap_or_default(),
            );
        }
    }

    if let Some(input) = root.get("input") {
        if let Some(items) = input.as_array() {
            collect_input_messages(items, &root, &mut messages);
        } else if let Some(text) = input.as_str() {
            let message = json!({"role":"user","content":text});
            push_message(
                &mut messages,
                serde_json::to_vec(&message).unwrap_or_default(),
            );
        }
    }

    if !messages.is_empty() {
        let joined = join_raw_array(&messages);
        if let Some(object) = out.as_object_mut() {
            if let Ok(value) = serde_json::from_slice::<Value>(&joined) {
                object.insert("messages".to_string(), value);
            }
        }
    }

    append_tools(&root, &mut out);

    if let Some(effort) = root.pointer("/reasoning/effort") {
        if let Some(value) = effort.as_str() {
            let lowered = value.trim().to_lowercase();
            if !lowered.is_empty() {
                if let Some(object) = out.as_object_mut() {
                    object.insert("reasoning_effort".to_string(), Value::String(lowered));
                }
            }
        }
    }

    serde_json::to_vec(&out).unwrap_or_default()
}

fn collect_input_messages(items: &[Value], root: &Value, messages: &mut Vec<Vec<u8>>) {
    let mut output_call_ids: std::collections::HashSet<String> = std::collections::HashSet::new();
    for item in items {
        let kind = item.get("type").and_then(Value::as_str).unwrap_or("");
        if kind != "function_call_output" && kind != "custom_tool_call_output" {
            continue;
        }
        if let Some(call_id) = item
            .get("call_id")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            output_call_ids.insert(call_id.to_string());
        }
    }

    let mut pending_tool_calls: Vec<Value> = Vec::new();
    let mut pending_tool_call_ids: Vec<String> = Vec::new();
    let mut pending_reasoning: String = String::new();
    let mut awaiting_tool_outputs: std::collections::HashSet<String> =
        std::collections::HashSet::new();
    let mut deferred_messages: Vec<Vec<u8>> = Vec::new();

    let take_reasoning =
        |pending_reasoning: &mut String| -> String { std::mem::take(pending_reasoning) };
    let flush_tool_calls = |pending_tool_calls: &mut Vec<Value>,
                            pending_tool_call_ids: &mut Vec<String>,
                            pending_reasoning: &mut String,
                            awaiting_tool_outputs: &mut std::collections::HashSet<String>,
                            messages: &mut Vec<Vec<u8>>| {
        if pending_tool_calls.is_empty() {
            return;
        }
        let reasoning = take_reasoning(pending_reasoning);
        let calls = std::mem::take(pending_tool_calls);
        let mut message = json!({"role":"assistant","tool_calls":Value::Array(calls)});
        if !reasoning.is_empty() {
            message["reasoning_content"] = Value::String(reasoning);
        }
        push_message(messages, serde_json::to_vec(&message).unwrap_or_default());
        for id in pending_tool_call_ids.drain(..) {
            let trimmed = id.trim();
            if !trimmed.is_empty() {
                awaiting_tool_outputs.insert(trimmed.to_string());
            }
        }
    };
    let flush_deferred = |deferred: &mut Vec<Vec<u8>>, messages: &mut Vec<Vec<u8>>| {
        for message in deferred.drain(..) {
            messages.push(message);
        }
    };
    let has_awaiting_tool_output =
        |awaiting_tool_outputs: &std::collections::HashSet<String>,
         output_call_ids: &std::collections::HashSet<String>| {
            awaiting_tool_outputs
                .iter()
                .any(|id| output_call_ids.contains(id))
        };
    let append_regular =
        |message: Vec<u8>,
         deferred: &mut Vec<Vec<u8>>,
         messages: &mut Vec<Vec<u8>>,
         awaiting_tool_outputs: &std::collections::HashSet<String>,
         output_call_ids: &std::collections::HashSet<String>| {
            if has_awaiting_tool_output(awaiting_tool_outputs, output_call_ids) {
                deferred.push(message);
            } else {
                messages.push(message);
            }
        };
    let append_pending_reasoning =
        |pending_reasoning: &mut String,
         deferred: &mut Vec<Vec<u8>>,
         messages: &mut Vec<Vec<u8>>,
         awaiting_tool_outputs: &std::collections::HashSet<String>,
         output_call_ids: &std::collections::HashSet<String>| {
            let reasoning = take_reasoning(pending_reasoning);
            if reasoning.is_empty() {
                return;
            }
            let message = json!({"role":"assistant","content":"","reasoning_content":reasoning});
            append_regular(
                serde_json::to_vec(&message).unwrap_or_default(),
                deferred,
                messages,
                awaiting_tool_outputs,
                output_call_ids,
            );
        };

    for item in items {
        let mut item_type = item
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        if item_type.is_empty() && item.get("role").and_then(Value::as_str).is_some() {
            item_type = "message".to_string();
        }
        if item_type != "function_call" && item_type != "custom_tool_call" {
            flush_tool_calls(
                &mut pending_tool_calls,
                &mut pending_tool_call_ids,
                &mut pending_reasoning,
                &mut awaiting_tool_outputs,
                messages,
            );
        }

        match item_type.as_str() {
            "message" | "" => {
                let mut role = item
                    .get("role")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                if role == "developer" {
                    role = "user".to_string();
                }
                if role != "assistant" {
                    append_pending_reasoning(
                        &mut pending_reasoning,
                        &mut deferred_messages,
                        messages,
                        &awaiting_tool_outputs,
                        &output_call_ids,
                    );
                }
                let mut message = json!({"role":role,"content":[]});
                if let Some(content) = item.get("content") {
                    if let Some(parts) = content.as_array() {
                        let mut content_items: Vec<Vec<u8>> = Vec::new();
                        for part in parts {
                            let mut content_type = part
                                .get("type")
                                .and_then(Value::as_str)
                                .unwrap_or("")
                                .to_string();
                            if content_type.is_empty() {
                                content_type = "input_text".to_string();
                            }
                            match content_type.as_str() {
                                "input_text" | "output_text" => {
                                    let text =
                                        part.get("text").and_then(Value::as_str).unwrap_or("");
                                    let item = json!({"type":"text","text":text});
                                    content_items
                                        .push(serde_json::to_vec(&item).unwrap_or_default());
                                }
                                "input_image" => {
                                    let url =
                                        part.get("image_url").and_then(Value::as_str).unwrap_or("");
                                    let mut item =
                                        json!({"type":"image_url","image_url":{"url":url}});
                                    if let Some(detail) =
                                        normalize_chat_image_detail(part.get("detail"))
                                    {
                                        if !detail.is_empty() {
                                            item["image_url"]["detail"] = Value::String(detail);
                                        }
                                    }
                                    content_items
                                        .push(serde_json::to_vec(&item).unwrap_or_default());
                                }
                                _ => {}
                            }
                        }
                        message = set_raw_array_items_value(&message, "content", &content_items);
                    } else if let Some(text) = content.as_str() {
                        message["content"] = Value::String(text.to_string());
                    }
                }

                if role == "assistant" {
                    let mut reasoning = item
                        .get("reasoning_content")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_string();
                    if reasoning.is_empty() {
                        reasoning = take_reasoning(&mut pending_reasoning);
                    } else {
                        pending_reasoning.clear();
                    }
                    if !reasoning.is_empty() {
                        message["reasoning_content"] = Value::String(reasoning);
                    }
                }

                append_regular(
                    serde_json::to_vec(&message).unwrap_or_default(),
                    &mut deferred_messages,
                    messages,
                    &awaiting_tool_outputs,
                    &output_call_ids,
                );
            }
            "reasoning" => {
                let text = collect_openai_responses_reasoning_content(item);
                if pending_reasoning.is_empty() {
                    pending_reasoning = text;
                } else {
                    pending_reasoning.push_str(&text);
                }
            }
            "function_call" => {
                let mut tool_call =
                    json!({"id":"","type":"function","function":{"name":"","arguments":""}});
                if let Some(call_id) = item.get("call_id") {
                    if let Some(value) = call_id.as_str() {
                        tool_call["id"] = Value::String(value.to_string());
                    }
                }
                if let Some(name) = item.get("name") {
                    if let Some(mut function_name) = name.as_str().map(str::to_string) {
                        if let Some(namespace) = item
                            .get("namespace")
                            .and_then(Value::as_str)
                            .map(str::trim)
                            .filter(|value| !value.is_empty())
                        {
                            function_name =
                                qualify_responses_namespace_tool_name(namespace, &function_name);
                        }
                        tool_call["function"]["name"] = Value::String(function_name);
                    }
                }
                if let Some(arguments) = item.get("arguments") {
                    if let Some(value) = arguments.as_str() {
                        tool_call["function"]["arguments"] = Value::String(value.to_string());
                    }
                }
                pending_tool_calls.push(tool_call);
                if let Some(call_id) = item
                    .get("call_id")
                    .and_then(Value::as_str)
                    .map(str::trim)
                    .filter(|value| !value.is_empty())
                {
                    pending_tool_call_ids.push(call_id.to_string());
                }
            }
            "function_call_output" => {
                let mut tool_message = json!({"role":"tool","tool_call_id":"","content":""});
                let mut call_id = String::new();
                if let Some(value) = item.get("call_id") {
                    if let Some(text) = value.as_str() {
                        call_id = text.trim().to_string();
                        tool_message["tool_call_id"] = Value::String(call_id.clone());
                    }
                }
                if let Some(output) = item.get("output") {
                    if let Some(value) = set_function_call_output_content(&tool_message, output) {
                        tool_message = value;
                    }
                }
                push_message(
                    messages,
                    serde_json::to_vec(&tool_message).unwrap_or_default(),
                );
                if !call_id.is_empty() {
                    awaiting_tool_outputs.remove(&call_id);
                }
                if awaiting_tool_outputs.is_empty() && !deferred_messages.is_empty() {
                    flush_deferred(&mut deferred_messages, messages);
                }
            }
            "custom_tool_call" => {
                let call_id = item.get("call_id").and_then(Value::as_str).unwrap_or("");
                let name = item.get("name").and_then(Value::as_str).unwrap_or("");
                let input = item.get("input").and_then(Value::as_str).unwrap_or("");
                let mut tool_call =
                    json!({"id":call_id,"type":"function","function":{"name":name,"arguments":""}});
                let wrapped = json!({"input":input});
                tool_call["function"]["arguments"] =
                    Value::String(serde_json::to_string(&wrapped).unwrap_or_default());
                pending_tool_calls.push(tool_call);
                if !call_id.trim().is_empty() {
                    pending_tool_call_ids.push(call_id.trim().to_string());
                }
            }
            "custom_tool_call_output" => {
                let call_id = item
                    .get("call_id")
                    .and_then(Value::as_str)
                    .map(str::trim)
                    .unwrap_or("");
                let text = responses_tool_output_text_owned(item.get("output"));
                let tool_message = json!({"role":"tool","tool_call_id":call_id,"content":text});
                push_message(
                    messages,
                    serde_json::to_vec(&tool_message).unwrap_or_default(),
                );
                if !call_id.is_empty() {
                    awaiting_tool_outputs.remove(call_id);
                }
                if awaiting_tool_outputs.is_empty() && !deferred_messages.is_empty() {
                    flush_deferred(&mut deferred_messages, messages);
                }
            }
            _ => {}
        }
    }

    flush_tool_calls(
        &mut pending_tool_calls,
        &mut pending_tool_call_ids,
        &mut pending_reasoning,
        &mut awaiting_tool_outputs,
        messages,
    );
    append_pending_reasoning(
        &mut pending_reasoning,
        &mut deferred_messages,
        messages,
        &awaiting_tool_outputs,
        &output_call_ids,
    );
    flush_deferred(&mut deferred_messages, messages);
    let _ = root;
}

fn push_message(messages: &mut Vec<Vec<u8>>, encoded: Vec<u8>) {
    if !encoded.is_empty() {
        messages.push(encoded);
    }
}

fn set_raw_array_items_value(base: &Value, path: &str, items: &[Vec<u8>]) -> Value {
    let raw = serde_json::to_vec(base).unwrap_or_default();
    let updated = set_raw_array_items(&raw, path, items);
    serde_json::from_slice(&updated).unwrap_or_else(|_| base.clone())
}

fn append_tools(root: &Value, out: &mut Value) {
    let mut chat_tools: Vec<Vec<u8>> = Vec::new();
    append_chat_tools(root.get("tools"), &mut chat_tools);
    if let Some(items) = root.get("input").and_then(Value::as_array) {
        for item in items {
            if item.get("type").and_then(Value::as_str) == Some("additional_tools") {
                append_chat_tools(item.get("tools"), &mut chat_tools);
            }
        }
    }
    if chat_tools.is_empty() {
        return;
    }
    let raw = serde_json::to_vec(out).unwrap_or_default();
    let updated = set_raw_array_items(&raw, "tools", &chat_tools);
    if let Ok(value) = serde_json::from_slice::<Value>(&updated) {
        *out = value;
    }
    if let Some(parallel) = root.get("parallel_tool_calls") {
        if let Some(object) = out.as_object_mut() {
            if let Some(value) = parallel.as_bool() {
                object.insert("parallel_tool_calls".to_string(), Value::Bool(value));
            }
        }
    }
    if let Some(choice) = root.get("tool_choice") {
        if let Ok(updated) = splice_raw_field(
            &serde_json::to_vec(out).unwrap_or_default(),
            "tool_choice",
            &serde_json::to_vec(choice).unwrap_or_default(),
        ) {
            if let Ok(value) = serde_json::from_slice::<Value>(&updated) {
                *out = value;
            }
        }
    }
}

fn append_chat_tools(tools: Option<&Value>, sink: &mut Vec<Vec<u8>>) {
    let Some(items) = tools.and_then(Value::as_array) else {
        return;
    };
    for tool in items {
        for entry in convert_responses_tool_to_openai_chat_tools(tool) {
            sink.push(entry);
        }
    }
}

fn splice_raw_field(base: &[u8], key: &str, raw_value: &[u8]) -> Result<Vec<u8>, ()> {
    let mut root: Value = serde_json::from_slice(base).map_err(|_| ())?;
    let object = root.as_object_mut().ok_or(())?;
    let parsed: Value = serde_json::from_slice(raw_value).map_err(|_| ())?;
    object.insert(key.to_string(), parsed);
    serde_json::to_vec(&root).map_err(|_| ())
}

fn set_function_call_output_content(base: &Value, output: &Value) -> Option<Value> {
    if let Some(text) = output.as_str() {
        if serde_json::from_str::<Value>(text).is_err() {
            let mut next = base.clone();
            next["content"] = Value::String(text.to_string());
            return Some(next);
        }
    }
    let structured = if let Some(text) = output.as_str() {
        serde_json::from_str::<Value>(text).unwrap_or_else(|_| output.clone())
    } else {
        output.clone()
    };

    if has_chat_tool_output_image_part(&structured) {
        let parts = if let Some(items) = structured.as_array() {
            items
                .iter()
                .map(chat_tool_output_content_part)
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        let raw = serde_json::to_vec(base).ok()?;
        let updated = set_raw_array_items(&raw, "content", &parts);
        return serde_json::from_slice(&updated).ok();
    }

    let mut next = base.clone();
    next["content"] = Value::String(
        output
            .as_str()
            .map(str::to_owned)
            .unwrap_or_else(|| serde_json::to_string(&structured).unwrap_or_default()),
    );
    Some(next)
}

fn has_chat_tool_output_image_part(content: &Value) -> bool {
    let Some(items) = content.as_array() else {
        return false;
    };
    let mut has_image = false;
    for item in items {
        let Some(item_type) = item.get("type").and_then(Value::as_str) else {
            continue;
        };
        match item_type {
            "text" | "input_text" | "output_text" => {
                if !item.get("text").is_some_and(Value::is_string) {
                    return false;
                }
            }
            "image_url" | "input_image" => {
                if chat_tool_output_image_fields(item).is_none() {
                    return false;
                }
                has_image = true;
            }
            _ => return false,
        }
    }
    has_image
}

fn chat_tool_output_content_part(item: &Value) -> Vec<u8> {
    let item_type = item.get("type").and_then(Value::as_str).unwrap_or("");
    match item_type {
        "text" | "input_text" | "output_text" => {
            let text = item.get("text").and_then(Value::as_str).unwrap_or("");
            let part = json!({"type":"text","text":text});
            serde_json::to_vec(&part).unwrap_or_default()
        }
        "image_url" | "input_image" => {
            if let Some((url, detail)) = chat_tool_output_image_fields(item) {
                let mut part = json!({"type":"image_url","image_url":{"url":url}});
                if !detail.is_empty() {
                    part["image_url"]["detail"] = Value::String(detail);
                }
                serde_json::to_vec(&part).unwrap_or_default()
            } else {
                chat_tool_output_fallback_part(item)
            }
        }
        _ => chat_tool_output_fallback_part(item),
    }
}

fn chat_tool_output_fallback_part(item: &Value) -> Vec<u8> {
    let text = if item.is_string() {
        item.as_str().unwrap_or("").to_string()
    } else {
        serde_json::to_string(item).unwrap_or_default()
    };
    let part = json!({"type":"text","text":text});
    serde_json::to_vec(&part).unwrap_or_default()
}

fn chat_tool_output_image_fields(item: &Value) -> Option<(String, String)> {
    let (url_path, detail_path) = match item.get("type").and_then(Value::as_str) {
        Some("image_url") => ("/image_url/url", "/image_url/detail"),
        Some("input_image") => ("/image_url", "/detail"),
        _ => return None,
    };
    let url = item.pointer(url_path)?.as_str()?.trim();
    if url.is_empty() {
        return None;
    }
    let detail = normalize_chat_image_detail(item.pointer(detail_path))?;
    Some((url.to_string(), detail))
}

fn normalize_chat_image_detail(detail: Option<&Value>) -> Option<String> {
    let Some(detail) = detail else {
        return Some(String::new());
    };
    let text = detail.as_str()?;
    let normalized = text.trim().to_lowercase();
    match normalized.as_str() {
        "auto" | "low" | "high" => Some(normalized),
        "original" => Some("high".to_string()),
        _ => Some(String::new()),
    }
}

fn responses_tool_output_text_owned(value: Option<&Value>) -> String {
    let Some(value) = value else {
        return String::new();
    };
    responses_tool_output_text(value)
}

fn collect_openai_responses_reasoning_content(item: &Value) -> String {
    let mut reasoning = String::new();
    if let Some(summary) = item.get("summary").and_then(Value::as_array) {
        for entry in summary {
            if entry.get("type").and_then(Value::as_str) == Some("summary_text") {
                if let Some(text) = entry.get("text").and_then(Value::as_str) {
                    reasoning.push_str(text);
                }
            }
        }
    }
    if reasoning.is_empty() {
        "[reasoning unavailable]".to_string()
    } else {
        reasoning
    }
}
