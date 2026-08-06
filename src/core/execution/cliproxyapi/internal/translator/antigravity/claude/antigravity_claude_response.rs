// ref: internal/translator/antigravity/claude/antigravity_claude_response.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use base64::{engine::general_purpose, Engine as _};
use serde_json::{json, Value};

use crate::internal::cache::{
    cache_signature_best_effort, get_model_group, signature_cache_enabled, SignatureKvStore,
};
use crate::internal::signature::{signature_provider_from_model_name, SignatureProvider};
use crate::internal::translator::common::{append_sse_event, claude_input_tokens_json};
use crate::internal::util::{
    disambiguated_tool_name_map, gemini_claude_tool_use_id, restore_sanitized_tool_name,
    sanitize_claude_tool_id,
};

use super::signature_validation::encode_gemini_claude_carrier_signature;
use super::web_search::{
    antigravity_grounding_metadata, antigravity_text_content, build_claude_web_search_content,
    should_translate_web_search_grounding,
};

/// Converts the native-grounding aggregate branch with an injected tool-use ID.
/// The caller owns collision-free ID generation; injecting it keeps the pure
/// format boundary deterministic and independently differential-testable.
pub fn convert_antigravity_web_search_response_to_claude_non_stream(
    original_request: &[u8],
    translated_request: &[u8],
    response: &[u8],
    tool_use_id: &str,
) -> Option<Vec<u8>> {
    let original: Value = serde_json::from_slice(original_request).ok()?;
    let translated: Value = serde_json::from_slice(translated_request).ok()?;
    if !should_translate_web_search_grounding(&original, &translated) {
        return None;
    }
    let root: Value = serde_json::from_slice(response).ok()?;
    let grounding = antigravity_grounding_metadata(&root)?;
    let usage = root.pointer("/response/usageMetadata");
    let input_tokens = integer(usage.and_then(|usage| usage.get("promptTokenCount")));
    let candidate_tokens = integer(usage.and_then(|usage| usage.get("candidatesTokenCount")));
    let thought_tokens = integer(usage.and_then(|usage| usage.get("thoughtsTokenCount")));
    let total_tokens = integer(usage.and_then(|usage| usage.get("totalTokenCount")));
    let cached_tokens = integer(usage.and_then(|usage| usage.get("cachedContentTokenCount")));
    let mut output_tokens = candidate_tokens.saturating_add(thought_tokens);
    if output_tokens == 0 && total_tokens > 0 {
        output_tokens = total_tokens.saturating_sub(input_tokens).max(0);
    }
    let mut output = json!({
        "id":root.pointer("/response/responseId").map(value_string).unwrap_or_default(),
        "type":"message",
        "role":"assistant",
        "model":root.pointer("/response/modelVersion").map(value_string).unwrap_or_default(),
        "content":build_claude_web_search_content(
            tool_use_id,
            &antigravity_text_content(&root),
            grounding,
        ),
        "stop_reason":"end_turn",
        "stop_sequence":null,
        "usage":{
            "input_tokens":input_tokens,
            "output_tokens":output_tokens,
            "server_tool_use":{"web_search_requests":1}
        }
    });
    if cached_tokens > 0 {
        output["usage"]["cache_read_input_tokens"] = cached_tokens.into();
    }
    serde_json::to_vec(&output).ok()
}

/// Converts a complete Antigravity response into the Claude Messages aggregate
/// shape. The Web Search tool ID is lifecycle state injected by the caller; it
/// is ignored for ordinary responses.
pub fn convert_antigravity_response_to_claude_non_stream(
    original_request: &[u8],
    translated_request: &[u8],
    response: &[u8],
    web_search_tool_use_id: &str,
) -> Vec<u8> {
    if let Some(output) = convert_antigravity_web_search_response_to_claude_non_stream(
        original_request,
        translated_request,
        response,
        web_search_tool_use_id,
    ) {
        return output;
    }
    let root = serde_json::from_slice::<Value>(response).unwrap_or(Value::Null);
    let provider_response = root.get("response").unwrap_or(&Value::Null);
    let usage = provider_response.get("usageMetadata");
    let prompt_tokens = integer(usage.and_then(|usage| usage.get("promptTokenCount")));
    let candidate_tokens = integer(usage.and_then(|usage| usage.get("candidatesTokenCount")));
    let thought_tokens = integer(usage.and_then(|usage| usage.get("thoughtsTokenCount")));
    let total_tokens = integer(usage.and_then(|usage| usage.get("totalTokenCount")));
    let cached_tokens = integer(usage.and_then(|usage| usage.get("cachedContentTokenCount")));
    let mut output_tokens = candidate_tokens.saturating_add(thought_tokens);
    if output_tokens == 0 && total_tokens > 0 {
        output_tokens = total_tokens.saturating_sub(prompt_tokens).max(0);
    }

    let translated = serde_json::from_slice::<Value>(translated_request).unwrap_or(Value::Null);
    let model_name = translated
        .get("model")
        .map(value_string)
        .unwrap_or_default();
    let tool_name_map = disambiguated_tool_name_map(original_request);
    let mut aggregate = ClaudeAggregate::default();
    let mut tool_id_counter = 0_usize;

    for part in provider_response
        .pointer("/candidates/0/content/parts")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let signature = part
            .get("thoughtSignature")
            .or_else(|| part.get("thought_signature"))
            .map(value_string)
            .unwrap_or_default();
        if let Some(function_call) = part.get("functionCall").and_then(Value::as_object) {
            let is_claude_target = get_model_group(&model_name) == "claude";
            let mut attached_to_thought = false;
            if !is_claude_target
                && !signature.is_empty()
                && !aggregate.thinking.is_empty()
                && aggregate.thinking_signature.is_empty()
            {
                aggregate.thinking_signature.clone_from(&signature);
                aggregate.thinking_direction = "next";
                aggregate.thinking_target = "function";
                attached_to_thought = true;
            }
            aggregate.flush_thinking(&model_name);
            aggregate.flush_text();
            aggregate.has_tool_call = true;

            let provider_name = function_call
                .get("name")
                .map(value_string)
                .unwrap_or_default();
            let name = restore_sanitized_tool_name(&tool_name_map, &provider_name);
            tool_id_counter += 1;
            if !is_claude_target && !signature.is_empty() && !attached_to_thought {
                aggregate.append_signature_carrier(&model_name, &signature, "next", "function");
            }
            let args_raw = function_call
                .get("args")
                .and_then(|args| serde_json::to_string(args).ok())
                .unwrap_or_default();
            let stable_id =
                if signature_provider_from_model_name(&model_name) == SignatureProvider::Gemini {
                    gemini_claude_tool_use_id(
                        &function_call
                            .get("id")
                            .map(value_string)
                            .unwrap_or_default(),
                        &provider_name,
                        &args_raw,
                    )
                } else {
                    String::new()
                };
            let id = if stable_id.is_empty() {
                sanitize_claude_tool_id(&format!("tool_{tool_id_counter}"))
            } else {
                stable_id
            };
            let mut block = json!({"type":"tool_use","id":id,"name":name,"input":{}});
            if is_claude_target && !signature.is_empty() {
                block["signature"] =
                    Value::String(format_claude_signature_value(&model_name, &signature));
            }
            if let Some(args) = function_call.get("args").filter(|args| args.is_object()) {
                block["input"] = args.clone();
            }
            aggregate.content.push(block);
            aggregate.has_semantic_content = true;
            aggregate.last_semantic_kind = "function";
            continue;
        }

        let text = part.get("text").map(value_string).unwrap_or_default();
        if part.get("thought").and_then(Value::as_bool) == Some(true) {
            aggregate.flush_text();
            if !aggregate.thinking_signature.is_empty() {
                aggregate.flush_thinking(&model_name);
            }
            if !text.is_empty() {
                aggregate.thinking.push_str(&text);
                aggregate.has_semantic_content = true;
                aggregate.last_semantic_kind = "text";
            }
            if !signature.is_empty() {
                if !aggregate.thinking.is_empty() {
                    aggregate.thinking_signature.clone_from(&signature);
                    aggregate.thinking_direction = "standalone";
                    aggregate.thinking_target = "text";
                    aggregate.flush_thinking(&model_name);
                } else if aggregate.has_semantic_content {
                    let target = aggregate.last_semantic_kind;
                    aggregate.append_signature_carrier(&model_name, &signature, "previous", target);
                } else {
                    aggregate.append_signature_carrier(&model_name, &signature, "next", "any");
                }
            }
            continue;
        }

        let mut visible_signature_carrier = false;
        if !signature.is_empty() {
            if !aggregate.thinking.is_empty() && aggregate.thinking_signature.is_empty() {
                aggregate.thinking_signature.clone_from(&signature);
                aggregate.thinking_direction = "next";
                aggregate.thinking_target = "text";
                aggregate.flush_thinking(&model_name);
            } else {
                aggregate.flush_thinking(&model_name);
                aggregate.flush_text();
                if !text.is_empty() {
                    aggregate.append_signature_carrier(&model_name, &signature, "next", "text");
                    visible_signature_carrier = true;
                } else if aggregate.has_semantic_content {
                    let target = aggregate.last_semantic_kind;
                    aggregate.append_signature_carrier(&model_name, &signature, "previous", target);
                } else {
                    aggregate.append_signature_carrier(&model_name, &signature, "next", "any");
                }
            }
        }
        if !text.is_empty() {
            aggregate.flush_thinking(&model_name);
            aggregate.text.push_str(&text);
            aggregate.has_semantic_content = true;
            aggregate.last_semantic_kind = "text";
            if visible_signature_carrier {
                aggregate.flush_text();
            }
        }
    }
    aggregate.flush_thinking(&model_name);
    aggregate.flush_text();

    let stop_reason = if aggregate.has_tool_call {
        "tool_use"
    } else if provider_response.pointer("/candidates/0/finishReason")
        == Some(&Value::String("MAX_TOKENS".to_owned()))
    {
        "max_tokens"
    } else {
        "end_turn"
    };
    let mut output = json!({
        "id":provider_response.get("responseId").map(value_string).unwrap_or_default(),
        "type":"message",
        "role":"assistant",
        "model":provider_response.get("modelVersion").map(value_string).unwrap_or_default(),
        "content":aggregate.content,
        "stop_reason":stop_reason,
        "stop_sequence":null,
        "usage":{"input_tokens":prompt_tokens,"output_tokens":output_tokens}
    });
    if cached_tokens > 0 {
        output["usage"]["cache_read_input_tokens"] = cached_tokens.into();
    }
    if usage.is_none() && prompt_tokens == 0 && output_tokens == 0 {
        output.as_object_mut().unwrap().remove("usage");
    }
    serde_json::to_vec(&output).unwrap_or_default()
}

#[derive(Default)]
struct ClaudeAggregate {
    content: Vec<Value>,
    text: String,
    thinking: String,
    thinking_signature: String,
    thinking_direction: &'static str,
    thinking_target: &'static str,
    has_tool_call: bool,
    has_semantic_content: bool,
    last_semantic_kind: &'static str,
}

impl ClaudeAggregate {
    fn flush_text(&mut self) {
        if self.text.is_empty() {
            return;
        }
        self.content
            .push(json!({"type":"text","text":std::mem::take(&mut self.text)}));
    }

    fn flush_thinking(&mut self, model_name: &str) {
        if self.thinking.is_empty() && self.thinking_signature.is_empty() {
            return;
        }
        let mut block = json!({
            "type":"thinking",
            "thinking":std::mem::take(&mut self.thinking)
        });
        if !self.thinking_signature.is_empty() {
            block["signature"] = Value::String(format_gemini_claude_carrier_value(
                model_name,
                &std::mem::take(&mut self.thinking_signature),
                self.thinking_direction,
                self.thinking_target,
            ));
        }
        self.content.push(block);
        self.thinking_direction = "standalone";
        self.thinking_target = "text";
    }

    fn append_signature_carrier(
        &mut self,
        model_name: &str,
        signature: &str,
        direction: &str,
        target: &str,
    ) {
        if signature.is_empty() {
            return;
        }
        self.content.push(json!({
            "type":"thinking",
            "thinking":"",
            "signature":format_gemini_claude_carrier_value(
                model_name,
                signature,
                direction,
                target,
            )
        }));
    }
}

fn format_gemini_claude_carrier_value(
    model_name: &str,
    signature: &str,
    direction: &str,
    target: &str,
) -> String {
    if signature_provider_from_model_name(model_name) == SignatureProvider::Gemini {
        encode_gemini_claude_carrier_signature(signature, direction, target)
    } else {
        format_claude_signature_value(model_name, signature)
    }
}

fn format_claude_signature_value(model_name: &str, signature: &str) -> String {
    let group = get_model_group(model_name);
    if group == "gemini" {
        return signature.to_owned();
    }
    if signature_cache_enabled() {
        return format!("{group}#{signature}");
    }
    if group == "claude" && signature.starts_with('R') {
        return general_purpose::STANDARD
            .decode(signature)
            .ok()
            .and_then(|decoded| String::from_utf8(decoded).ok())
            .unwrap_or_default();
    }
    signature.to_owned()
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct AntigravityClaudeWebSearchStreamState {
    has_first_response: bool,
    response_index: usize,
    response_type: u8,
    has_web_search_tool: bool,
    web_search_requests: i64,
    text_buffer: String,
    finish_reason: String,
    has_finish_reason: bool,
    has_usage: bool,
    prompt_tokens: i64,
    candidate_tokens: i64,
    thought_tokens: i64,
    total_tokens: i64,
    cached_tokens: i64,
    has_content: bool,
    final_sent: bool,
    terminal_sent: bool,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct AntigravityClaudeStreamState {
    web_search: AntigravityClaudeWebSearchStreamState,
    has_first_response: bool,
    response_type: u8,
    response_index: usize,
    has_finish_reason: bool,
    finish_reason: String,
    has_usage: bool,
    prompt_tokens: i64,
    candidate_tokens: i64,
    thought_tokens: i64,
    total_tokens: i64,
    cached_tokens: i64,
    final_sent: bool,
    terminal_sent: bool,
    has_tool_use: bool,
    has_content: bool,
    has_semantic_content: bool,
    last_semantic_kind: String,
    current_thinking_text: String,
    current_thinking_signed: bool,
}

pub fn convert_antigravity_web_search_response_to_claude_stream(
    original_request: &[u8],
    translated_request: &[u8],
    chunk: &[u8],
    state: &mut AntigravityClaudeWebSearchStreamState,
    tool_use_id: &str,
) -> Vec<Vec<u8>> {
    if state.terminal_sent {
        return Vec::new();
    }
    let original: Value = match serde_json::from_slice(original_request) {
        Ok(value) => value,
        Err(_) => return Vec::new(),
    };
    let translated: Value = match serde_json::from_slice(translated_request) {
        Ok(value) => value,
        Err(_) => return Vec::new(),
    };
    if !should_translate_web_search_grounding(&original, &translated) {
        return Vec::new();
    }
    let mut events = Vec::new();
    if chunk == b"[DONE]" {
        if state.has_first_response && !state.has_content {
            push_event(
                &mut events,
                "content_block_start",
                json!({"type":"content_block_start","index":state.response_index,"content_block":{"type":"text","text":""}}),
            );
            state.response_type = 1;
            state.has_content = true;
        }
        if state.has_content {
            append_final_events(&mut events, state, true);
            push_event(&mut events, "message_stop", json!({"type":"message_stop"}));
            state.terminal_sent = true;
        }
        return events;
    }
    let Ok(root) = serde_json::from_slice::<Value>(chunk) else {
        return events;
    };
    if !state.has_first_response {
        let mut message = json!({
            "type":"message_start",
            "message":{
                "id":"msg_1nZdL29xx5MUA1yADyHTEsnR8uuvGzszyY",
                "type":"message",
                "role":"assistant",
                "content":[],
                "model":"claude-3-5-sonnet-20241022",
                "stop_reason":null,
                "stop_sequence":null,
                "usage":{"input_tokens":0,"output_tokens":0}
            }
        });
        if let Some(value) = root.pointer("/response/cpaUsageMetadata/promptTokenCount") {
            message["message"]["usage"]["input_tokens"] = integer(Some(value)).into();
        }
        if let Some(value) = root.pointer("/response/modelVersion") {
            message["message"]["model"] = Value::String(value_string(value));
        }
        if let Some(value) = root.pointer("/response/responseId") {
            message["message"]["id"] = Value::String(value_string(value));
        }
        push_event(&mut events, "message_start", message);
        state.has_first_response = true;
    }

    let mut handled_grounding = false;
    if !state.has_web_search_tool {
        if let Some(grounding) = antigravity_grounding_metadata(&root) {
            let text = std::mem::take(&mut state.text_buffer) + &antigravity_text_content(&root);
            append_web_search_blocks(&mut events, state, tool_use_id, &text, grounding);
            state.has_web_search_tool = true;
            state.web_search_requests = 1;
            state.has_content = true;
            state.response_type = 0;
            handled_grounding = true;
        }
    }
    if !state.has_web_search_tool && !handled_grounding {
        if let Some(parts) = root
            .pointer("/response/candidates/0/content/parts")
            .and_then(Value::as_array)
        {
            for part in parts {
                if part.get("thought").and_then(Value::as_bool) == Some(true)
                    || part.get("functionCall").is_some()
                {
                    continue;
                }
                if let Some(text) = part.get("text") {
                    state.text_buffer.push_str(&value_string(text));
                }
            }
        }
    }
    if let Some(finish) = root.pointer("/response/candidates/0/finishReason") {
        state.has_finish_reason = true;
        state.finish_reason = value_string(finish);
    }
    if let Some(usage) = root.pointer("/response/usageMetadata") {
        state.has_usage = true;
        state.cached_tokens = integer(usage.get("cachedContentTokenCount"));
        state.prompt_tokens =
            integer(usage.get("promptTokenCount")).saturating_sub(state.cached_tokens);
        state.candidate_tokens = integer(usage.get("candidatesTokenCount"));
        state.thought_tokens = integer(usage.get("thoughtsTokenCount"));
        state.total_tokens = integer(usage.get("totalTokenCount"));
        if state.candidate_tokens == 0 && state.total_tokens > 0 {
            state.candidate_tokens = state
                .total_tokens
                .saturating_sub(state.prompt_tokens)
                .saturating_sub(state.thought_tokens)
                .max(0);
        }
    }
    if !state.has_web_search_tool && state.has_finish_reason && !state.text_buffer.is_empty() {
        let text = std::mem::take(&mut state.text_buffer);
        push_event(
            &mut events,
            "content_block_start",
            json!({"type":"content_block_start","index":state.response_index,"content_block":{"type":"text","text":""}}),
        );
        push_event(
            &mut events,
            "content_block_delta",
            json!({"type":"content_block_delta","index":state.response_index,"delta":{"type":"text_delta","text":text}}),
        );
        state.response_type = 1;
        state.has_content = true;
    }
    if state.has_usage && state.has_finish_reason {
        append_final_events(&mut events, state, false);
    }
    events
}

pub fn convert_antigravity_response_to_claude_stream(
    original_request: &[u8],
    translated_request: &[u8],
    chunk: &[u8],
    state: &mut AntigravityClaudeStreamState,
    web_search_tool_use_id: &str,
) -> Vec<Vec<u8>> {
    convert_antigravity_response_to_claude_stream_with_runtime(
        original_request,
        translated_request,
        chunk,
        state,
        web_search_tool_use_id,
        None,
    )
}

/// Runtime response boundary with an explicitly injected durable signature
/// store. Publication remains best-effort, matching upstream response
/// semantics, but the injected store is authoritative and never shadowed by
/// process-local state.
pub fn convert_antigravity_response_to_claude_stream_with_runtime(
    original_request: &[u8],
    translated_request: &[u8],
    chunk: &[u8],
    state: &mut AntigravityClaudeStreamState,
    web_search_tool_use_id: &str,
    signature_store: Option<&dyn SignatureKvStore>,
) -> Vec<Vec<u8>> {
    let original = serde_json::from_slice::<Value>(original_request).unwrap_or(Value::Null);
    let translated = serde_json::from_slice::<Value>(translated_request).unwrap_or(Value::Null);
    if should_translate_web_search_grounding(&original, &translated) {
        return convert_antigravity_web_search_response_to_claude_stream(
            original_request,
            translated_request,
            chunk,
            &mut state.web_search,
            web_search_tool_use_id,
        );
    }
    if state.terminal_sent {
        return Vec::new();
    }
    let mut events = Vec::new();
    if chunk == b"[DONE]" {
        if state.has_first_response && !state.has_content {
            push_event(
                &mut events,
                "content_block_start",
                json!({"type":"content_block_start","index":state.response_index,"content_block":{"type":"text","text":""}}),
            );
            state.response_type = 1;
            state.has_content = true;
        }
        if state.has_content {
            append_normal_final_events(&mut events, state, true);
            push_event(&mut events, "message_stop", json!({"type":"message_stop"}));
            state.terminal_sent = true;
        }
        return events;
    }
    let root = serde_json::from_slice::<Value>(chunk).unwrap_or(Value::Null);
    let model_name = translated
        .get("model")
        .map(value_string)
        .unwrap_or_default();
    if !state.has_first_response {
        let mut start = json!({
            "type":"message_start",
            "message":{
                "id":"msg_1nZdL29xx5MUA1yADyHTEsnR8uuvGzszyY",
                "type":"message","role":"assistant","content":[],
                "model":"claude-3-5-sonnet-20241022",
                "stop_reason":null,"stop_sequence":null,
                "usage":{"input_tokens":0,"output_tokens":0}
            }
        });
        if let Some(value) = root.pointer("/response/cpaUsageMetadata/promptTokenCount") {
            start["message"]["usage"]["input_tokens"] = integer(Some(value)).into();
        }
        if let Some(value) = root.pointer("/response/cpaUsageMetadata/candidatesTokenCount") {
            start["message"]["usage"]["output_tokens"] = integer(Some(value)).into();
        }
        if let Some(value) = root.pointer("/response/modelVersion") {
            start["message"]["model"] = Value::String(value_string(value));
        }
        if let Some(value) = root.pointer("/response/responseId") {
            start["message"]["id"] = Value::String(value_string(value));
        }
        push_event(&mut events, "message_start", start);
        state.has_first_response = true;
    }

    let tool_name_map = disambiguated_tool_name_map(original_request);
    let finish_exists = root
        .pointer("/response/candidates/0/finishReason")
        .is_some();
    for part in root
        .pointer("/response/candidates/0/content/parts")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let text = part.get("text").map(value_string);
        let function_call = part.get("functionCall").and_then(Value::as_object);
        let signature = part
            .get("thoughtSignature")
            .or_else(|| part.get("thought_signature"))
            .map(value_string)
            .unwrap_or_default();
        let has_signature = !signature.is_empty() && function_call.is_none();
        if has_signature && text.as_deref().unwrap_or_default().is_empty() {
            let (direction, target) = if state.has_semantic_content {
                ("previous", state.last_semantic_kind.clone())
            } else {
                ("next", "any".to_owned())
            };
            append_normal_part_signature(
                &mut events,
                state,
                &model_name,
                &signature,
                direction,
                &target,
                signature_store,
            );
            continue;
        }

        if let Some(text) = text {
            if part.get("thought").and_then(Value::as_bool) == Some(true) {
                if !text.is_empty() {
                    state.has_semantic_content = true;
                    state.last_semantic_kind = "text".to_owned();
                    if state.response_type == 2 && state.current_thinking_signed {
                        close_normal_block(&mut events, state);
                    }
                    if state.response_type != 2 {
                        close_normal_block(&mut events, state);
                        start_normal_thinking(&mut events, state);
                        state.current_thinking_text.clear();
                    }
                    state.current_thinking_text.push_str(&text);
                    push_event(
                        &mut events,
                        "content_block_delta",
                        json!({"type":"content_block_delta","index":state.response_index,"delta":{"type":"thinking_delta","thinking":text}}),
                    );
                    state.has_content = true;
                }
                if has_signature {
                    append_normal_thinking_signature(
                        &mut events,
                        state,
                        &model_name,
                        &signature,
                        "standalone",
                        "text",
                        signature_store,
                    );
                }
            } else {
                let signature_targets_text = if has_signature {
                    append_normal_part_signature(
                        &mut events,
                        state,
                        &model_name,
                        &signature,
                        "next",
                        "text",
                        signature_store,
                    )
                } else {
                    false
                };
                if !text.is_empty() || !finish_exists {
                    if state.response_type != 1 {
                        close_normal_block(&mut events, state);
                        if !text.is_empty() {
                            push_event(
                                &mut events,
                                "content_block_start",
                                json!({"type":"content_block_start","index":state.response_index,"content_block":{"type":"text","text":""}}),
                            );
                            state.response_type = 1;
                        }
                    }
                    if state.response_type == 1 {
                        push_event(
                            &mut events,
                            "content_block_delta",
                            json!({"type":"content_block_delta","index":state.response_index,"delta":{"type":"text_delta","text":text}}),
                        );
                        state.has_content = true;
                    }
                }
                if !text.is_empty() {
                    state.has_semantic_content = true;
                    state.last_semantic_kind = "text".to_owned();
                    if signature_targets_text {
                        close_normal_block(&mut events, state);
                    }
                }
            }
        } else if let Some(function_call) = function_call {
            if get_model_group(&model_name) != "claude" {
                append_normal_part_signature(
                    &mut events,
                    state,
                    &model_name,
                    &signature,
                    "next",
                    "function",
                    signature_store,
                );
            }
            state.has_tool_use = true;
            close_normal_block(&mut events, state);
            let provider_name = function_call
                .get("name")
                .map(value_string)
                .unwrap_or_default();
            let name = restore_sanitized_tool_name(&tool_name_map, &provider_name);
            let args_raw = function_call
                .get("args")
                .and_then(|args| serde_json::to_string(args).ok())
                .unwrap_or_default();
            let stable_id =
                if signature_provider_from_model_name(&model_name) == SignatureProvider::Gemini {
                    gemini_claude_tool_use_id(
                        &function_call
                            .get("id")
                            .map(value_string)
                            .unwrap_or_default(),
                        &provider_name,
                        &args_raw,
                    )
                } else {
                    String::new()
                };
            let id = if stable_id.is_empty() {
                sanitize_claude_tool_id("")
            } else {
                stable_id
            };
            let mut block = json!({"type":"tool_use","id":id,"name":name,"input":{}});
            if get_model_group(&model_name) == "claude" && !signature.is_empty() {
                block["signature"] =
                    Value::String(format_claude_signature_value(&model_name, &signature));
            }
            push_event(
                &mut events,
                "content_block_start",
                json!({"type":"content_block_start","index":state.response_index,"content_block":block}),
            );
            if !args_raw.is_empty() {
                push_event(
                    &mut events,
                    "content_block_delta",
                    json!({"type":"content_block_delta","index":state.response_index,"delta":{"type":"input_json_delta","partial_json":args_raw}}),
                );
            }
            state.response_type = 3;
            state.has_content = true;
            state.has_semantic_content = true;
            state.last_semantic_kind = "function".to_owned();
        }
    }
    if let Some(finish) = root.pointer("/response/candidates/0/finishReason") {
        state.has_finish_reason = true;
        state.finish_reason = value_string(finish);
    }
    if let Some(usage) = root.pointer("/response/usageMetadata") {
        state.has_usage = true;
        state.cached_tokens = integer(usage.get("cachedContentTokenCount"));
        state.prompt_tokens =
            integer(usage.get("promptTokenCount")).saturating_sub(state.cached_tokens);
        state.candidate_tokens = integer(usage.get("candidatesTokenCount"));
        state.thought_tokens = integer(usage.get("thoughtsTokenCount"));
        state.total_tokens = integer(usage.get("totalTokenCount"));
        if state.candidate_tokens == 0 && state.total_tokens > 0 {
            state.candidate_tokens = state
                .total_tokens
                .saturating_sub(state.prompt_tokens)
                .saturating_sub(state.thought_tokens)
                .max(0);
        }
    }
    if state.has_usage && state.has_finish_reason {
        append_normal_final_events(&mut events, state, false);
    }
    events
}

fn close_normal_block(events: &mut Vec<Vec<u8>>, state: &mut AntigravityClaudeStreamState) {
    if state.response_type == 0 {
        return;
    }
    push_event(
        events,
        "content_block_stop",
        json!({"type":"content_block_stop","index":state.response_index}),
    );
    state.response_index += 1;
    state.response_type = 0;
    state.current_thinking_signed = false;
}

fn start_normal_thinking(events: &mut Vec<Vec<u8>>, state: &mut AntigravityClaudeStreamState) {
    push_event(
        events,
        "content_block_start",
        json!({"type":"content_block_start","index":state.response_index,"content_block":{"type":"thinking","thinking":""}}),
    );
    state.response_type = 2;
    state.current_thinking_signed = false;
    state.has_content = true;
}

fn append_normal_thinking_signature(
    events: &mut Vec<Vec<u8>>,
    state: &mut AntigravityClaudeStreamState,
    model_name: &str,
    signature: &str,
    direction: &str,
    target: &str,
    signature_store: Option<&dyn SignatureKvStore>,
) {
    if signature.is_empty() || state.response_type != 2 {
        return;
    }
    if !state.current_thinking_text.is_empty() {
        cache_signature_best_effort(
            signature_store,
            model_name,
            &state.current_thinking_text,
            signature,
        );
        state.current_thinking_text.clear();
    }
    push_event(
        events,
        "content_block_delta",
        json!({"type":"content_block_delta","index":state.response_index,"delta":{"type":"signature_delta","signature":format_gemini_claude_carrier_value(model_name,signature,direction,target)}}),
    );
    state.current_thinking_signed = true;
    state.has_content = true;
}

fn append_normal_part_signature(
    events: &mut Vec<Vec<u8>>,
    state: &mut AntigravityClaudeStreamState,
    model_name: &str,
    signature: &str,
    direction: &str,
    target: &str,
    signature_store: Option<&dyn SignatureKvStore>,
) -> bool {
    if signature.is_empty() {
        return false;
    }
    if state.response_type == 2 && !state.current_thinking_signed {
        append_normal_thinking_signature(
            events,
            state,
            model_name,
            signature,
            direction,
            target,
            signature_store,
        );
        return false;
    }
    close_normal_block(events, state);
    start_normal_thinking(events, state);
    push_event(
        events,
        "content_block_delta",
        json!({"type":"content_block_delta","index":state.response_index,"delta":{"type":"signature_delta","signature":format_gemini_claude_carrier_value(model_name,signature,direction,target)}}),
    );
    state.current_thinking_signed = true;
    state.has_content = true;
    true
}

fn append_normal_final_events(
    events: &mut Vec<Vec<u8>>,
    state: &mut AntigravityClaudeStreamState,
    force: bool,
) {
    if state.final_sent || (!state.has_usage && !force) || !state.has_content {
        return;
    }
    close_normal_block(events, state);
    let output = state.candidate_tokens.saturating_add(state.thought_tokens);
    let output = if output == 0 && state.total_tokens > 0 {
        state
            .total_tokens
            .saturating_sub(state.prompt_tokens)
            .max(0)
    } else {
        output
    };
    let stop = if state.has_tool_use {
        "tool_use"
    } else if state.finish_reason == "MAX_TOKENS" {
        "max_tokens"
    } else {
        "end_turn"
    };
    let mut delta = json!({
        "type":"message_delta",
        "delta":{"stop_reason":stop,"stop_sequence":null},
        "usage":{"input_tokens":state.prompt_tokens,"output_tokens":output}
    });
    if state.cached_tokens > 0 {
        delta["usage"]["cache_read_input_tokens"] = state.cached_tokens.into();
    }
    push_event(events, "message_delta", delta);
    state.final_sent = true;
}

fn append_web_search_blocks(
    events: &mut Vec<Vec<u8>>,
    state: &mut AntigravityClaudeWebSearchStreamState,
    tool_use_id: &str,
    text: &str,
    grounding: &Value,
) {
    let blocks = build_claude_web_search_content(tool_use_id, text, grounding);
    let server = &blocks[0];
    push_event(
        events,
        "content_block_start",
        json!({"type":"content_block_start","index":state.response_index,"content_block":{"type":"server_tool_use","id":tool_use_id,"name":"web_search","input":{}}}),
    );
    if let Some(query) = server.pointer("/input/query").and_then(Value::as_str) {
        push_event(
            events,
            "content_block_delta",
            json!({"type":"content_block_delta","index":state.response_index,"delta":{"type":"input_json_delta","partial_json":serde_json::to_string(&json!({"query":query})).unwrap_or_default()}}),
        );
    }
    push_event(
        events,
        "content_block_stop",
        json!({"type":"content_block_stop","index":state.response_index}),
    );
    state.response_index += 1;

    push_event(
        events,
        "content_block_start",
        json!({"type":"content_block_start","index":state.response_index,"content_block":blocks[1]}),
    );
    push_event(
        events,
        "content_block_stop",
        json!({"type":"content_block_stop","index":state.response_index}),
    );
    state.response_index += 1;

    for block in blocks.into_iter().skip(2) {
        let citations = block
            .get("citations")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let start = if citations.is_empty() {
            json!({"type":"content_block_start","index":state.response_index,"content_block":{"type":"text","text":""}})
        } else {
            json!({"type":"content_block_start","index":state.response_index,"content_block":{"citations":[],"type":"text","text":""}})
        };
        push_event(events, "content_block_start", start);
        for citation in citations {
            push_event(
                events,
                "content_block_delta",
                json!({"type":"content_block_delta","index":state.response_index,"delta":{"type":"citations_delta","citation":citation}}),
            );
        }
        let text = block
            .get("text")
            .and_then(Value::as_str)
            .unwrap_or_default();
        for chunk in split_runes(text, 50) {
            push_event(
                events,
                "content_block_delta",
                json!({"type":"content_block_delta","index":state.response_index,"delta":{"type":"text_delta","text":chunk}}),
            );
        }
        push_event(
            events,
            "content_block_stop",
            json!({"type":"content_block_stop","index":state.response_index}),
        );
        state.response_index += 1;
    }
}

fn append_final_events(
    events: &mut Vec<Vec<u8>>,
    state: &mut AntigravityClaudeWebSearchStreamState,
    force: bool,
) {
    if state.final_sent || (!state.has_usage && !force) || !state.has_content {
        return;
    }
    if state.response_type != 0 {
        push_event(
            events,
            "content_block_stop",
            json!({"type":"content_block_stop","index":state.response_index}),
        );
        state.response_type = 0;
    }
    let output_tokens = state.candidate_tokens.saturating_add(state.thought_tokens);
    let output_tokens = if output_tokens == 0 && state.total_tokens > 0 {
        state
            .total_tokens
            .saturating_sub(state.prompt_tokens)
            .max(0)
    } else {
        output_tokens
    };
    let stop_reason = if state.finish_reason == "MAX_TOKENS" {
        "max_tokens"
    } else {
        "end_turn"
    };
    let mut delta = json!({
        "type":"message_delta",
        "delta":{"stop_reason":stop_reason,"stop_sequence":null},
        "usage":{"input_tokens":state.prompt_tokens,"output_tokens":output_tokens}
    });
    if state.web_search_requests > 0 {
        delta["usage"]["server_tool_use"]["web_search_requests"] = state.web_search_requests.into();
    }
    if state.cached_tokens > 0 {
        delta["usage"]["cache_read_input_tokens"] = state.cached_tokens.into();
    }
    push_event(events, "message_delta", delta);
    state.final_sent = true;
}

fn split_runes(text: &str, chunk_size: usize) -> Vec<String> {
    if chunk_size == 0 || text.is_empty() {
        return Vec::new();
    }
    let runes = text.chars().collect::<Vec<_>>();
    runes
        .chunks(chunk_size)
        .map(|chunk| chunk.iter().collect())
        .collect()
}

fn push_event(events: &mut Vec<Vec<u8>>, event: &str, payload: Value) {
    let Ok(payload) = serde_json::to_vec(&payload) else {
        return;
    };
    let mut encoded = Vec::new();
    append_sse_event(&mut encoded, event, &payload, 3);
    events.push(encoded);
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

pub fn claude_token_count(count: i64) -> Vec<u8> {
    claude_input_tokens_json(count)
}

// CTOX injects lifecycle IDs, keeps terminal emission idempotent and splits
// Web Search state from normal state internally; the external conversion
// behavior remains pinned-Go differential-gated.
