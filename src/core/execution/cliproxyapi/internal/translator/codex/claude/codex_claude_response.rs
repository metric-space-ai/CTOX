// ref: internal/translator/codex/claude/codex_claude_response.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::VecDeque;

use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::internal::translator::common::{append_sse_event, claude_input_tokens_json};
use crate::internal::translator::interactions::claude::{
    convert_interactions_response_to_claude, convert_interactions_response_to_claude_non_stream,
    InteractionsToClaudeStreamState,
};
use crate::sdk::translator::TranslationContext;

use super::codex_claude_response_web_search::{
    append_non_stream_web_search, stream_web_search_events,
};
use super::CodexToInteractionsState;

pub struct CodexToClaudeStreamState {
    interactions: CodexToInteractionsState,
    claude: InteractionsToClaudeStreamState,
    next_web_search_index: usize,
    function_calls: VecDeque<BufferedFunctionCall>,
    deferred_events: Vec<Vec<u8>>,
}

#[derive(Default)]
struct BufferedFunctionCall {
    output_index: i64,
    call_id: String,
    name: String,
    arguments: String,
    done: bool,
}

impl Default for CodexToClaudeStreamState {
    fn default() -> Self {
        Self::with_identity("msg_ctox_codex")
    }
}

impl CodexToClaudeStreamState {
    pub fn with_identity(identity: impl Into<String>) -> Self {
        Self {
            interactions: CodexToInteractionsState::default(),
            claude: InteractionsToClaudeStreamState::with_identity(identity),
            next_web_search_index: 0,
            function_calls: VecDeque::new(),
            deferred_events: Vec::new(),
        }
    }
}

pub fn deterministic_claude_message_id(
    model_name: &str,
    original_request: &[u8],
    request: &[u8],
) -> String {
    let mut hash = Sha256::new();
    hash.update(model_name.as_bytes());
    hash.update([0]);
    hash.update(original_request);
    hash.update([0]);
    hash.update(request);
    let suffix = hash
        .finalize()
        .iter()
        .take(12)
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    format!("msg_{suffix}")
}

pub fn convert_codex_response_to_claude_stream(
    context: &TranslationContext,
    model_name: &str,
    original_request: &[u8],
    request: &[u8],
    raw: &[u8],
    state: &mut CodexToClaudeStreamState,
) -> Vec<Vec<u8>> {
    if let Some(output) = buffer_parallel_function_call_event(
        context,
        model_name,
        original_request,
        request,
        raw,
        state,
    ) {
        return output;
    }
    translate_stream_event(context, model_name, original_request, request, raw, state)
}

fn translate_stream_event(
    context: &TranslationContext,
    model_name: &str,
    original_request: &[u8],
    request: &[u8],
    raw: &[u8],
    state: &mut CodexToClaudeStreamState,
) -> Vec<Vec<u8>> {
    let interactions = super::convert_codex_response_to_interactions_stream(
        context,
        model_name,
        original_request,
        request,
        raw,
        &mut state.interactions,
    );
    let mut out = Vec::new();
    for event in interactions {
        out.extend(convert_interactions_response_to_claude(
            model_name,
            original_request,
            request,
            &event,
            &mut state.claude,
        ));
    }
    observe_block_indexes(&out, &mut state.next_web_search_index);
    out.extend(stream_web_search_events(
        raw,
        &mut state.next_web_search_index,
    ));
    if out.is_empty() {
        if let Some(error) = codex_error_event(raw) {
            out.push(error);
        }
    }
    out
}

fn buffer_parallel_function_call_event(
    context: &TranslationContext,
    model_name: &str,
    original_request: &[u8],
    request: &[u8],
    raw: &[u8],
    state: &mut CodexToClaudeStreamState,
) -> Option<Vec<Vec<u8>>> {
    let payload = raw.strip_prefix(b"data: ").unwrap_or(raw).trim_ascii();
    let root = serde_json::from_slice::<Value>(payload).ok()?;
    let event_type = root.get("type").and_then(Value::as_str).unwrap_or_default();
    let index = root
        .get("output_index")
        .and_then(Value::as_i64)
        .unwrap_or(-1);
    match event_type {
        "response.output_item.added"
            if root.pointer("/item/type").and_then(Value::as_str) == Some("function_call") =>
        {
            update_buffered_call(state, index, root.get("item"), false, false);
            Some(Vec::new())
        }
        "response.function_call_arguments.delta" => {
            let delta = root
                .get("delta")
                .and_then(Value::as_str)
                .unwrap_or_default();
            update_buffered_call(state, index, Some(&root), false, false);
            if let Some(call) = find_buffered_call_mut(state, index, &root) {
                call.arguments.push_str(delta);
            }
            Some(Vec::new())
        }
        "response.function_call_arguments.done" => {
            update_buffered_call(state, index, Some(&root), false, false);
            if let Some(call) = find_buffered_call_mut(state, index, &root) {
                if let Some(arguments) = root.get("arguments").and_then(Value::as_str) {
                    call.arguments = arguments.to_owned();
                }
            }
            Some(Vec::new())
        }
        "response.output_item.done"
            if root.pointer("/item/type").and_then(Value::as_str) == Some("function_call") =>
        {
            update_buffered_call(state, index, root.get("item"), true, true);
            let mut output = drain_completed_function_calls(
                context,
                model_name,
                original_request,
                request,
                state,
            );
            if state.function_calls.is_empty() {
                for deferred in std::mem::take(&mut state.deferred_events) {
                    output.extend(translate_stream_event(
                        context,
                        model_name,
                        original_request,
                        request,
                        &deferred,
                        state,
                    ));
                }
            }
            Some(output)
        }
        _ if !state.function_calls.is_empty() => {
            state.deferred_events.push(raw.to_vec());
            Some(Vec::new())
        }
        _ => None,
    }
}

fn update_buffered_call(
    state: &mut CodexToClaudeStreamState,
    output_index: i64,
    item: Option<&Value>,
    replace_arguments: bool,
    done: bool,
) {
    let item = item.unwrap_or(&Value::Null);
    let key = item
        .get("call_id")
        .or_else(|| item.get("id"))
        .and_then(Value::as_str)
        .unwrap_or_default();
    let position = state.function_calls.iter().position(|call| {
        (output_index >= 0 && call.output_index == output_index)
            || (!key.is_empty() && call.call_id == key)
    });
    if position.is_none() {
        state.function_calls.push_back(BufferedFunctionCall {
            output_index,
            ..BufferedFunctionCall::default()
        });
    }
    let position = position.unwrap_or(state.function_calls.len() - 1);
    let call = &mut state.function_calls[position];
    if output_index >= 0 {
        call.output_index = output_index;
    }
    if !key.is_empty() {
        call.call_id = key.to_owned();
    }
    if let Some(name) = item
        .get("name")
        .and_then(Value::as_str)
        .filter(|name| !name.is_empty())
    {
        call.name = name.to_owned();
    }
    if let Some(arguments) = item.get("arguments").and_then(Value::as_str) {
        if replace_arguments || call.arguments.is_empty() {
            call.arguments = arguments.to_owned();
        }
    }
    call.done |= done;
}

fn find_buffered_call_mut<'a>(
    state: &'a mut CodexToClaudeStreamState,
    output_index: i64,
    root: &Value,
) -> Option<&'a mut BufferedFunctionCall> {
    let key = root
        .get("call_id")
        .or_else(|| root.get("item_id"))
        .and_then(Value::as_str)
        .unwrap_or_default();
    state.function_calls.iter_mut().find(|call| {
        (output_index >= 0 && call.output_index == output_index)
            || (!key.is_empty() && call.call_id == key)
    })
}

fn drain_completed_function_calls(
    context: &TranslationContext,
    model_name: &str,
    original_request: &[u8],
    request: &[u8],
    state: &mut CodexToClaudeStreamState,
) -> Vec<Vec<u8>> {
    let mut output = Vec::new();
    while state.function_calls.front().is_some_and(|call| call.done) {
        let call = state.function_calls.pop_front().expect("front was present");
        for event in [
            json!({"type":"response.output_item.added","output_index":call.output_index,"item":{"type":"function_call","call_id":call.call_id,"name":call.name,"arguments":""}}),
            json!({"type":"response.function_call_arguments.delta","output_index":call.output_index,"call_id":call.call_id,"delta":call.arguments}),
            json!({"type":"response.output_item.done","output_index":call.output_index,"item":{"type":"function_call","call_id":call.call_id,"name":call.name,"arguments":call.arguments}}),
        ] {
            let mut frame = b"data: ".to_vec();
            frame.extend(serde_json::to_vec(&event).unwrap_or_default());
            output.extend(translate_stream_event(
                context,
                model_name,
                original_request,
                request,
                &frame,
                state,
            ));
        }
    }
    output
}

pub fn convert_codex_response_to_claude_non_stream(
    context: &TranslationContext,
    model_name: &str,
    original_request: &[u8],
    request: &[u8],
    raw: &[u8],
) -> Vec<u8> {
    let interactions = super::convert_codex_response_to_interactions_non_stream(
        context,
        model_name,
        original_request,
        request,
        raw,
    );
    let translated = convert_interactions_response_to_claude_non_stream(
        model_name,
        original_request,
        request,
        &interactions,
    );
    append_non_stream_web_search(&translated, raw)
}

pub fn claude_token_count(count: i64) -> Vec<u8> {
    claude_input_tokens_json(count)
}

fn observe_block_indexes(events: &[Vec<u8>], next: &mut usize) {
    for event in events {
        for line in String::from_utf8_lossy(event).lines() {
            let Some(payload) = line.strip_prefix("data: ") else {
                continue;
            };
            if let Ok(value) = serde_json::from_str::<Value>(payload) {
                if let Some(index) = value.get("index").and_then(Value::as_u64) {
                    *next = (*next).max(index as usize + 1);
                }
            }
        }
    }
}

fn codex_error_event(raw: &[u8]) -> Option<Vec<u8>> {
    let payload = raw.strip_prefix(b"data: ").unwrap_or(raw).trim_ascii();
    let root: Value = serde_json::from_slice(payload).ok()?;
    if root.get("type").and_then(Value::as_str) != Some("error") {
        return None;
    }
    let error = root.get("error").unwrap_or(&root);
    let mut kind = error
        .get("type")
        .or_else(|| root.get("error_type"))
        .and_then(Value::as_str)
        .unwrap_or("api_error");
    let code = error
        .get("code")
        .and_then(Value::as_str)
        .unwrap_or_default();
    if code == "cyber_policy" || kind == "invalid_request" {
        kind = "invalid_request_error";
    }
    let message = error
        .get("message")
        .or_else(|| root.get("message"))
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .unwrap_or(if code.is_empty() { kind } else { code });
    let payload =
        serde_json::to_vec(&json!({"type":"error","error":{"type":kind,"message":message}}))
            .ok()?;
    let mut out = Vec::new();
    append_sse_event(&mut out, "error", &payload, 3);
    Some(out)
}
