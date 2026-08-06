// ref: internal/translator/antigravity/openai/responses/antigravity_openai-responses_response.go @ a88197f845c979132c8978ea223c6af05cc81536
// ref: internal/translator/gemini/openai/responses/gemini_openai-responses_response.go:102-831 @ a88197f845c979132c8978ea223c6af05cc81536
// ref: internal/translator/gemini/openai/responses/gemini_openai-responses_response.go:832-1226 @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use crate::internal::translator::common::sse_event_data;
use base64::{engine::general_purpose::STANDARD_NO_PAD, Engine as _};
use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, HashMap, HashSet};
use uuid::Uuid;

const CARRIER_PREFIX: &str = "cpa-gemini-responses-carrier-v1:";
const NEXT: &str = "next";
const PREVIOUS: &str = "previous";
const STANDALONE: &str = "standalone";
const TEXT: &str = "text";
const FUNCTION: &str = "function";
const ANY: &str = "any";

#[derive(Clone, Debug)]
struct ReasoningOutput {
    text: String,
    signature: String,
    direction: String,
    target_kind: String,
}

#[derive(Clone, Debug)]
struct FunctionOutput {
    item: Value,
    signature: String,
}

#[derive(Clone, Debug)]
struct DetachedOutput {
    signature: String,
    direction: String,
    target_kind: String,
}

#[derive(Clone, Debug)]
struct MessageOutput {
    text: String,
    signatures: Vec<String>,
}

#[derive(Clone, Copy, Debug)]
enum OutputOrder {
    Reasoning(usize),
    Function(usize),
    Detached(usize),
    Message(usize),
}

#[derive(Clone, Debug, Default)]
pub struct AntigravityToResponsesState {
    identity: ResponseIdentity,
    sequence: u64,
    response_id: String,
    created_at: i64,
    started: bool,
    completed: bool,
    next_index: usize,
    message_index: Option<usize>,
    message_id: String,
    message_text: String,
    reasoning_index: Option<usize>,
    reasoning_id: String,
    reasoning_text: String,
    reasoning_signature: String,
    reasoning_pending: Vec<String>,
    pending_reasoning_signature: String,
    seen_reasoning_signatures: HashSet<String>,
    last_semantic_kind: String,
    outputs: BTreeMap<usize, Value>,
    sanitized_names: HashMap<String, String>,
}

#[derive(Clone, Debug)]
struct ResponseIdentity {
    response_id: String,
    created_at: i64,
    function_sequence: u64,
}

impl Default for ResponseIdentity {
    fn default() -> Self {
        Self {
            response_id: format!("resp_{}", Uuid::new_v4().simple()),
            created_at: 0,
            function_sequence: 0,
        }
    }
}

impl ResponseIdentity {
    fn with_identity(response_id: impl Into<String>, created_at: i64) -> Self {
        Self {
            response_id: response_id.into(),
            created_at,
            function_sequence: 0,
        }
    }

    fn next_function_id(&mut self) -> String {
        self.function_sequence += 1;
        format!("call_{}_{}", self.response_id, self.function_sequence)
    }
}

impl AntigravityToResponsesState {
    pub fn with_identity(response_id: impl Into<String>, created_at: i64) -> Self {
        Self {
            identity: ResponseIdentity::with_identity(response_id, created_at),
            ..Self::default()
        }
    }

    fn next_sequence(&mut self) -> u64 {
        self.sequence += 1;
        self.sequence
    }

    fn emit(&mut self, event: &str, mut payload: Value, output: &mut Vec<Vec<u8>>) {
        payload["sequence_number"] = Value::from(self.next_sequence());
        output.push(sse_event_data(
            event,
            &serde_json::to_vec(&payload).unwrap_or_default(),
        ));
    }

    fn start(
        &mut self,
        root: &Value,
        original_request: &[u8],
        request: &[u8],
        output: &mut Vec<Vec<u8>>,
    ) {
        if self.started {
            return;
        }
        let provider_id = root
            .get("responseId")
            .and_then(Value::as_str)
            .filter(|value| !value.is_empty())
            .map(ToOwned::to_owned)
            .unwrap_or_else(|| self.identity.response_id.clone());
        self.response_id = if provider_id.starts_with("resp_") {
            provider_id
        } else {
            format!("resp_{provider_id}")
        };
        self.created_at = root
            .get("createTime")
            .and_then(Value::as_str)
            .and_then(parse_rfc3339_unix_seconds)
            .unwrap_or(self.identity.created_at);
        self.sanitized_names = stream_sanitized_tool_name_map(original_request, request);
        self.emit(
            "response.created",
            json!({"type":"response.created","sequence_number":0,"response":{"id":self.response_id,"object":"response","created_at":self.created_at,"status":"in_progress","background":false,"error":null,"output":[]}}),
            output,
        );
        self.emit(
            "response.in_progress",
            json!({"type":"response.in_progress","sequence_number":0,"response":{"id":self.response_id,"object":"response","created_at":self.created_at,"status":"in_progress"}}),
            output,
        );
        self.started = true;
    }

    fn open_reasoning(&mut self, output: &mut Vec<Vec<u8>>) {
        if self.reasoning_index.is_some()
            || (self.reasoning_text.is_empty() && self.reasoning_signature.is_empty())
        {
            return;
        }
        let index = self.next_index;
        self.next_index += 1;
        self.reasoning_index = Some(index);
        self.reasoning_id = format!("rs_{}_{}", self.response_id, index);
        let encrypted = encode_carrier(&self.reasoning_signature, STANDALONE, TEXT);
        self.emit(
            "response.output_item.added",
            json!({"type":"response.output_item.added","sequence_number":0,"output_index":index,"item":{"id":self.reasoning_id,"type":"reasoning","status":"in_progress","encrypted_content":encrypted,"summary":[]}}),
            output,
        );
        self.emit(
            "response.reasoning_summary_part.added",
            json!({"type":"response.reasoning_summary_part.added","sequence_number":0,"item_id":self.reasoning_id,"output_index":index,"summary_index":0,"part":{"type":"summary_text","text":""}}),
            output,
        );
        for delta in std::mem::take(&mut self.reasoning_pending) {
            self.emit(
                "response.reasoning_summary_text.delta",
                json!({"type":"response.reasoning_summary_text.delta","sequence_number":0,"item_id":self.reasoning_id,"output_index":index,"summary_index":0,"delta":delta}),
                output,
            );
        }
    }

    fn finalize_reasoning(&mut self, output: &mut Vec<Vec<u8>>) {
        self.open_reasoning(output);
        let Some(index) = self.reasoning_index.take() else {
            return;
        };
        let id = std::mem::take(&mut self.reasoning_id);
        let text = std::mem::take(&mut self.reasoning_text);
        let signature = encode_carrier(
            &std::mem::take(&mut self.reasoning_signature),
            STANDALONE,
            TEXT,
        );
        self.emit(
            "response.reasoning_summary_text.done",
            json!({"type":"response.reasoning_summary_text.done","sequence_number":0,"item_id":id,"output_index":index,"summary_index":0,"text":text}),
            output,
        );
        self.emit(
            "response.reasoning_summary_part.done",
            json!({"type":"response.reasoning_summary_part.done","sequence_number":0,"item_id":id,"output_index":index,"summary_index":0,"part":{"type":"summary_text","text":text}}),
            output,
        );
        let item = json!({"id":id,"type":"reasoning","encrypted_content":signature,"summary":[{"type":"summary_text","text":text}]});
        self.emit(
            "response.output_item.done",
            json!({"type":"response.output_item.done","sequence_number":0,"output_index":index,"item":item}),
            output,
        );
        self.outputs.insert(index, item);
        self.reasoning_pending.clear();
    }

    fn append_reasoning(&mut self, text: &str, signature: &str, output: &mut Vec<Vec<u8>>) {
        if self.reasoning_index.is_some()
            && !signature.is_empty()
            && !self.reasoning_signature.is_empty()
            && signature != self.reasoning_signature
        {
            self.finalize_reasoning(output);
        }
        self.finalize_message(output);
        if !signature.is_empty() {
            self.reasoning_signature = signature.to_owned();
            self.seen_reasoning_signatures.insert(signature.to_owned());
        }
        self.last_semantic_kind = TEXT.to_owned();
        self.reasoning_text.push_str(text);
        if self.reasoning_index.is_none() {
            self.reasoning_pending.push(text.to_owned());
            if !self.reasoning_signature.is_empty() {
                self.open_reasoning(output);
            }
        } else if let Some(index) = self.reasoning_index.filter(|_| !text.is_empty()) {
            let id = self.reasoning_id.clone();
            self.emit(
                "response.reasoning_summary_text.delta",
                json!({"type":"response.reasoning_summary_text.delta","sequence_number":0,"item_id":id,"output_index":index,"summary_index":0,"delta":text}),
                output,
            );
        }
    }

    fn open_message(&mut self, output: &mut Vec<Vec<u8>>) {
        if self.message_index.is_some() {
            return;
        }
        let index = self.next_index;
        self.next_index += 1;
        self.message_index = Some(index);
        self.message_id = format!("msg_{}_{}", self.response_id, index);
        self.emit(
            "response.output_item.added",
            json!({"type":"response.output_item.added","sequence_number":0,"output_index":index,"item":{"id":self.message_id,"type":"message","status":"in_progress","content":[],"role":"assistant"}}),
            output,
        );
        self.emit(
            "response.content_part.added",
            json!({"type":"response.content_part.added","sequence_number":0,"item_id":self.message_id,"output_index":index,"content_index":0,"part":{"type":"output_text","annotations":[],"logprobs":[],"text":""}}),
            output,
        );
    }

    fn append_text(&mut self, text: &str, output: &mut Vec<Vec<u8>>) {
        self.finalize_reasoning(output);
        self.open_message(output);
        self.message_text.push_str(text);
        self.last_semantic_kind = TEXT.to_owned();
        let index = self.message_index.expect("message opened");
        let id = self.message_id.clone();
        self.emit(
            "response.output_text.delta",
            json!({"type":"response.output_text.delta","sequence_number":0,"item_id":id,"output_index":index,"content_index":0,"delta":text,"logprobs":[]}),
            output,
        );
    }

    fn finalize_message(&mut self, output: &mut Vec<Vec<u8>>) {
        let Some(index) = self.message_index.take() else {
            return;
        };
        let id = std::mem::take(&mut self.message_id);
        let text = std::mem::take(&mut self.message_text);
        self.emit(
            "response.output_text.done",
            json!({"type":"response.output_text.done","sequence_number":0,"item_id":id,"output_index":index,"content_index":0,"text":text,"logprobs":[]}),
            output,
        );
        self.emit(
            "response.content_part.done",
            json!({"type":"response.content_part.done","sequence_number":0,"item_id":id,"output_index":index,"content_index":0,"part":{"type":"output_text","annotations":[],"logprobs":[],"text":text}}),
            output,
        );
        let item = json!({"id":id,"type":"message","status":"completed","content":[{"type":"output_text","text":text}],"role":"assistant"});
        self.emit(
            "response.output_item.done",
            json!({"type":"response.output_item.done","sequence_number":0,"output_index":index,"item":item}),
            output,
        );
        self.outputs.insert(
            index,
            json!({"id":id,"type":"message","status":"completed","content":[{"type":"output_text","annotations":[],"logprobs":[],"text":text}],"role":"assistant"}),
        );
    }

    fn append_function(&mut self, call: &Value, output: &mut Vec<Vec<u8>>) {
        self.finalize_reasoning(output);
        self.finalize_message(output);
        let index = self.next_index;
        self.next_index += 1;
        self.last_semantic_kind = FUNCTION.to_owned();
        let call_id = self.identity.next_function_id();
        let item_id = format!("fc_{call_id}");
        let sanitized = call.get("name").and_then(Value::as_str).unwrap_or("");
        let name = self
            .sanitized_names
            .get(sanitized)
            .cloned()
            .unwrap_or_else(|| sanitized.to_owned());
        let arguments = call
            .get("args")
            .map(|value| serde_json::to_string(value).unwrap_or_default())
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| "{}".to_owned());
        self.emit(
            "response.output_item.added",
            json!({"type":"response.output_item.added","sequence_number":0,"output_index":index,"item":{"id":item_id,"type":"function_call","status":"in_progress","arguments":"","call_id":call_id,"name":name}}),
            output,
        );
        self.emit(
            "response.function_call_arguments.delta",
            json!({"type":"response.function_call_arguments.delta","sequence_number":0,"item_id":item_id,"output_index":index,"delta":arguments}),
            output,
        );
        self.emit(
            "response.function_call_arguments.done",
            json!({"type":"response.function_call_arguments.done","sequence_number":0,"item_id":item_id,"output_index":index,"arguments":arguments}),
            output,
        );
        let item = json!({"id":item_id,"type":"function_call","status":"completed","arguments":arguments,"call_id":call_id,"name":name});
        self.emit(
            "response.output_item.done",
            json!({"type":"response.output_item.done","sequence_number":0,"output_index":index,"item":item}),
            output,
        );
        self.outputs.insert(index, item);
    }

    fn emit_detached(
        &mut self,
        signature: &str,
        direction: &str,
        target: &str,
        output: &mut Vec<Vec<u8>>,
    ) {
        let signature = signature.trim();
        if signature.is_empty() || self.seen_reasoning_signatures.contains(signature) {
            return;
        }
        self.finalize_reasoning(output);
        self.finalize_message(output);
        let index = self.next_index;
        self.next_index += 1;
        let placement = if direction == PREVIOUS {
            "after"
        } else {
            "before"
        };
        let id = format!("rs_{}_detached_{placement}_{index}", self.response_id);
        let encrypted = encode_carrier(signature, direction, target);
        self.emit(
            "response.output_item.added",
            json!({"type":"response.output_item.added","sequence_number":0,"output_index":index,"item":{"id":id,"type":"reasoning","status":"in_progress","encrypted_content":encrypted,"summary":[]}}),
            output,
        );
        let item = json!({"id":id,"type":"reasoning","encrypted_content":encrypted,"summary":[]});
        self.emit(
            "response.output_item.done",
            json!({"type":"response.output_item.done","sequence_number":0,"output_index":index,"item":item}),
            output,
        );
        self.outputs.insert(index, item);
        self.seen_reasoning_signatures.insert(signature.to_owned());
    }

    fn emit_trailing_detached(&mut self, signature: &str, output: &mut Vec<Vec<u8>>) {
        let (direction, target) = match self.last_semantic_kind.as_str() {
            TEXT => (PREVIOUS, TEXT),
            FUNCTION => (PREVIOUS, FUNCTION),
            _ => (STANDALONE, ANY),
        };
        self.emit_detached(signature, direction, target, output);
    }

    fn complete(
        &mut self,
        root: &Value,
        original_request: &[u8],
        request: &[u8],
        output: &mut Vec<Vec<u8>>,
    ) {
        self.finalize_reasoning(output);
        self.finalize_message(output);
        let mut response = Map::new();
        response.insert("id".to_owned(), Value::String(self.response_id.clone()));
        response.insert("object".to_owned(), Value::String("response".to_owned()));
        response.insert("created_at".to_owned(), Value::from(self.created_at));
        response.insert("status".to_owned(), Value::String("completed".to_owned()));
        response.insert("background".to_owned(), Value::Bool(false));
        response.insert("error".to_owned(), Value::Null);
        let selected = select_stream_request(original_request, request);
        echo_request_fields(&mut response, selected.as_ref(), root);
        if !self.outputs.is_empty() {
            response.insert(
                "output".to_owned(),
                Value::Array(self.outputs.values().cloned().collect()),
            );
        }
        if let Some(usage) = root.get("usageMetadata") {
            response.insert("usage".to_owned(), convert_stream_usage(usage));
        }
        self.emit(
            "response.completed",
            json!({"type":"response.completed","sequence_number":0,"response":Value::Object(response)}),
            output,
        );
        self.completed = true;
    }
}

/// Converts one Antigravity/Gemini streaming payload into zero or more
/// Responses SSE events while retaining per-request protocol state.
pub fn convert_antigravity_response_to_openai_responses_stream(
    original_request: &[u8],
    request: &[u8],
    raw_json: &[u8],
    state: &mut AntigravityToResponsesState,
) -> Vec<Vec<u8>> {
    let mut raw = raw_json.trim_ascii();
    if let Some(data) = raw.strip_prefix(b"data:") {
        raw = data.trim_ascii();
    }
    if raw.is_empty() || state.completed {
        return Vec::new();
    }
    if raw == b"[DONE]" {
        if !state.started {
            return Vec::new();
        }
        raw = br#"{"candidates":[{"finishReason":"STOP"}]}"#;
    }
    let Ok(raw_root) = serde_json::from_slice::<Value>(raw) else {
        return Vec::new();
    };
    let root = raw_root.get("response").unwrap_or(&raw_root);
    let mut output = Vec::new();
    state.start(root, original_request, request, &mut output);
    if let Some(parts) = root
        .pointer("/candidates/0/content/parts")
        .and_then(Value::as_array)
    {
        for part in parts {
            let signature = part
                .get("thoughtSignature")
                .or_else(|| part.get("thought_signature"))
                .and_then(Value::as_str)
                .unwrap_or("")
                .trim();
            let is_thought = part.get("thought").and_then(Value::as_bool) == Some(true);
            let function_call = part.get("functionCall");
            let text = part.get("text").and_then(Value::as_str);

            if function_call.is_some() && !state.pending_reasoning_signature.is_empty() {
                let pending = std::mem::take(&mut state.pending_reasoning_signature);
                if signature.is_empty() {
                    state.emit_detached(&pending, NEXT, FUNCTION, &mut output);
                } else {
                    state.emit_trailing_detached(&pending, &mut output);
                }
            }

            if !signature.is_empty() && !is_thought {
                if function_call.is_some() {
                    state.emit_detached(signature, NEXT, FUNCTION, &mut output);
                } else if text.is_some_and(|text| !text.is_empty()) {
                    if !state.pending_reasoning_signature.is_empty()
                        && state.pending_reasoning_signature != signature
                    {
                        let pending = std::mem::take(&mut state.pending_reasoning_signature);
                        state.emit_trailing_detached(&pending, &mut output);
                    }
                    if !state.seen_reasoning_signatures.contains(signature) {
                        state.pending_reasoning_signature = signature.to_owned();
                    }
                } else if text == Some("") {
                    if !state.pending_reasoning_signature.is_empty() {
                        let pending = std::mem::take(&mut state.pending_reasoning_signature);
                        if pending != signature {
                            state.emit_trailing_detached(&pending, &mut output);
                        }
                    }
                    if state.message_index.is_some() || state.last_semantic_kind == FUNCTION {
                        state.emit_trailing_detached(signature, &mut output);
                    } else if !state.seen_reasoning_signatures.contains(signature) {
                        state.pending_reasoning_signature = signature.to_owned();
                    }
                    continue;
                }
            }

            if is_thought {
                if !state.pending_reasoning_signature.is_empty() && state.message_index.is_some() {
                    let pending = std::mem::take(&mut state.pending_reasoning_signature);
                    state.emit_trailing_detached(&pending, &mut output);
                }
                state.append_reasoning(text.unwrap_or(""), signature, &mut output);
            } else if let Some(text) = text.filter(|text| !text.is_empty()) {
                if signature.is_empty()
                    && !state.pending_reasoning_signature.is_empty()
                    && state.message_index.is_some()
                {
                    let pending = std::mem::take(&mut state.pending_reasoning_signature);
                    state.emit_trailing_detached(&pending, &mut output);
                }
                state.append_text(text, &mut output);
            } else if let Some(call) = function_call {
                state.append_function(call, &mut output);
            }
        }
    }
    if root
        .pointer("/candidates/0/finishReason")
        .and_then(Value::as_str)
        .is_some_and(|value| !value.is_empty())
    {
        if !state.pending_reasoning_signature.is_empty() {
            let pending = std::mem::take(&mut state.pending_reasoning_signature);
            state.emit_trailing_detached(&pending, &mut output);
        }
        state.complete(root, original_request, request, &mut output);
    }
    output
}

fn select_stream_request(original_request: &[u8], request: &[u8]) -> Option<Value> {
    serde_json::from_slice(original_request)
        .ok()
        .or_else(|| serde_json::from_slice(request).ok())
        .map(|root: Value| root.get("request").cloned().unwrap_or(root))
}

fn stream_sanitized_tool_name_map(
    original_request: &[u8],
    request: &[u8],
) -> HashMap<String, String> {
    serde_json::from_slice(original_request)
        .ok()
        .or_else(|| serde_json::from_slice(request).ok())
        .map(|root: Value| sanitized_tool_name_map(&root))
        .unwrap_or_default()
}

fn convert_stream_usage(usage: &Value) -> Value {
    json!({
        "input_tokens":usage.get("promptTokenCount").and_then(Value::as_i64).unwrap_or(0),
        "input_tokens_details":{"cached_tokens":usage.get("cachedContentTokenCount").and_then(Value::as_i64).unwrap_or(0)},
        "output_tokens":usage.get("candidatesTokenCount").and_then(Value::as_i64).unwrap_or(0),
        "output_tokens_details":{"reasoning_tokens":usage.get("thoughtsTokenCount").and_then(Value::as_i64).unwrap_or(0)},
        "total_tokens":usage.get("totalTokenCount").and_then(Value::as_i64).unwrap_or(0)
    })
}

/// Converts the Antigravity `generateContent` envelope into one OpenAI
/// Responses object. The upstream Antigravity converter unwraps the envelope
/// and delegates to the Gemini implementation; this Rust port keeps both
/// operations together until the shared Gemini module is activated.
pub fn convert_antigravity_response_to_openai_responses_non_stream(
    original_request: &[u8],
    request: &[u8],
    raw_json: &[u8],
) -> Vec<u8> {
    let mut state = AntigravityToResponsesState::default();
    convert_antigravity_response_to_openai_responses_non_stream_with_state(
        original_request,
        request,
        raw_json,
        &mut state,
    )
}

/// Non-streaming conversion with injected request-local identity/clock state.
pub fn convert_antigravity_response_to_openai_responses_non_stream_with_state(
    original_request: &[u8],
    request: &[u8],
    raw_json: &[u8],
    state: &mut AntigravityToResponsesState,
) -> Vec<u8> {
    let raw_root = serde_json::from_slice::<Value>(raw_json).unwrap_or(Value::Null);
    let wrapped = raw_root.get("response").is_some();
    let root = raw_root.get("response").unwrap_or(&raw_root);

    let original = parse_request(original_request, wrapped);
    let translated = parse_request(request, wrapped);
    let selected_request = original.as_ref().or(translated.as_ref());

    let provider_id = root
        .get("responseId")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| state.identity.response_id.clone());
    let response_id = if provider_id.starts_with("resp_") {
        provider_id
    } else {
        format!("resp_{provider_id}")
    };

    let mut response = Map::new();
    response.insert("id".to_owned(), Value::String(response_id.clone()));
    response.insert("object".to_owned(), Value::String("response".to_owned()));
    response.insert(
        "created_at".to_owned(),
        Value::from(
            root.get("createTime")
                .and_then(Value::as_str)
                .and_then(parse_rfc3339_unix_seconds)
                .unwrap_or(state.identity.created_at),
        ),
    );
    response.insert("status".to_owned(), Value::String("completed".to_owned()));
    response.insert("background".to_owned(), Value::Bool(false));
    response.insert("error".to_owned(), Value::Null);
    response.insert("incomplete_details".to_owned(), Value::Null);
    echo_request_fields(&mut response, selected_request, root);

    let sanitized_names = selected_request
        .map(sanitized_tool_name_map)
        .unwrap_or_default();
    let outputs = convert_parts(root, &response_id, &sanitized_names, &mut state.identity);
    if !outputs.is_empty() {
        response.insert("output".to_owned(), Value::Array(outputs));
    }
    if let Some(usage) = root.get("usageMetadata") {
        response.insert("usage".to_owned(), convert_usage(usage));
    }
    serde_json::to_vec(&Value::Object(response)).unwrap_or_default()
}

fn parse_request(raw: &[u8], unwrap: bool) -> Option<Value> {
    let root = serde_json::from_slice::<Value>(raw).ok()?;
    if unwrap {
        root.get("request").cloned().or(Some(root))
    } else {
        Some(root)
    }
}

fn echo_request_fields(response: &mut Map<String, Value>, request: Option<&Value>, root: &Value) {
    if let Some(request) = request {
        let request = unwrap_openai_request(request);
        for key in [
            "max_output_tokens",
            "max_tool_calls",
            "parallel_tool_calls",
            "previous_response_id",
            "prompt_cache_key",
            "reasoning",
            "safety_identifier",
            "service_tier",
            "store",
            "temperature",
            "text",
            "tool_choice",
            "tools",
            "top_logprobs",
            "top_p",
            "truncation",
            "user",
            "metadata",
        ] {
            if let Some(value) = request.get(key) {
                response.insert(key.to_owned(), value.clone());
            }
        }
        if let Some(instructions) = request.get("instructions").and_then(Value::as_str) {
            response.insert(
                "instructions".to_owned(),
                Value::String(instructions.to_owned()),
            );
        }
        if let Some(model) = request.get("model").and_then(Value::as_str) {
            response.insert("model".to_owned(), Value::String(model.to_owned()));
        } else if let Some(model) = root.get("modelVersion").and_then(Value::as_str) {
            response.insert("model".to_owned(), Value::String(model.to_owned()));
        }
    } else if let Some(model) = root.get("modelVersion").and_then(Value::as_str) {
        response.insert("model".to_owned(), Value::String(model.to_owned()));
    }
}

fn unwrap_openai_request(root: &Value) -> &Value {
    let Some(request) = root.get("request") else {
        return root;
    };
    if request.get("model").is_some()
        || request.get("input").is_some()
        || request.get("instructions").is_some()
    {
        request
    } else {
        root
    }
}

fn convert_parts(
    root: &Value,
    response_id: &str,
    sanitized_names: &HashMap<String, String>,
    identity: &mut ResponseIdentity,
) -> Vec<Value> {
    let parts = root
        .pointer("/candidates/0/content/parts")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or(&[]);
    let mut reasoning_outputs = Vec::<ReasoningOutput>::new();
    let mut function_outputs = Vec::<FunctionOutput>::new();
    let mut detached_outputs = Vec::<DetachedOutput>::new();
    let mut message_outputs = Vec::<MessageOutput>::new();
    let mut order = Vec::<OutputOrder>::new();
    let mut reasoning_signatures = HashSet::<String>::new();

    let mut reasoning_text = String::new();
    let mut reasoning_encrypted = String::new();
    let mut reasoning_direction = String::new();
    let mut reasoning_target = String::new();
    let mut message_text = String::new();
    let mut message_signatures = Vec::<String>::new();

    for part in parts {
        let mut signature = part
            .get("thoughtSignature")
            .or_else(|| part.get("thought_signature"))
            .and_then(Value::as_str)
            .unwrap_or("")
            .trim()
            .to_owned();
        if part.get("thought").and_then(Value::as_bool) == Some(true) {
            flush_message(
                &mut message_text,
                &mut message_signatures,
                &mut message_outputs,
                &mut order,
            );
            if !signature.is_empty()
                && !reasoning_encrypted.is_empty()
                && signature != reasoning_encrypted
            {
                flush_reasoning(
                    &mut reasoning_text,
                    &mut reasoning_encrypted,
                    &mut reasoning_direction,
                    &mut reasoning_target,
                    &mut reasoning_outputs,
                    &mut reasoning_signatures,
                    &mut order,
                );
            }
            if let Some(text) = part.get("text").and_then(Value::as_str) {
                reasoning_text.push_str(text);
            }
            if !signature.is_empty() {
                reasoning_encrypted = signature;
                reasoning_direction = STANDALONE.to_owned();
                reasoning_target = TEXT.to_owned();
            }
            continue;
        }

        if let Some(text) = part
            .get("text")
            .and_then(Value::as_str)
            .filter(|v| !v.is_empty())
        {
            let mut message_signature = String::new();
            if !signature.is_empty() {
                if !reasoning_text.is_empty() && reasoning_encrypted.is_empty() {
                    reasoning_encrypted = signature;
                    reasoning_direction = NEXT.to_owned();
                    reasoning_target = TEXT.to_owned();
                } else {
                    message_signature = signature;
                }
            }
            flush_reasoning(
                &mut reasoning_text,
                &mut reasoning_encrypted,
                &mut reasoning_direction,
                &mut reasoning_target,
                &mut reasoning_outputs,
                &mut reasoning_signatures,
                &mut order,
            );
            if !message_signatures.is_empty()
                && (message_signature.is_empty()
                    || message_signatures.last() != Some(&message_signature))
            {
                flush_message(
                    &mut message_text,
                    &mut message_signatures,
                    &mut message_outputs,
                    &mut order,
                );
            }
            message_text.push_str(text);
            if !message_signature.is_empty()
                && message_signatures.last() != Some(&message_signature)
            {
                message_signatures.push(message_signature);
            }
            continue;
        }

        if let Some(call) = part.get("functionCall") {
            if !reasoning_text.is_empty() && reasoning_encrypted.is_empty() && !signature.is_empty()
            {
                reasoning_encrypted = signature;
                reasoning_direction = NEXT.to_owned();
                reasoning_target = FUNCTION.to_owned();
                signature = String::new();
            }
            flush_reasoning(
                &mut reasoning_text,
                &mut reasoning_encrypted,
                &mut reasoning_direction,
                &mut reasoning_target,
                &mut reasoning_outputs,
                &mut reasoning_signatures,
                &mut order,
            );
            flush_message(
                &mut message_text,
                &mut message_signatures,
                &mut message_outputs,
                &mut order,
            );
            let sanitized = call.get("name").and_then(Value::as_str).unwrap_or("");
            let name = sanitized_names
                .get(sanitized)
                .map(String::as_str)
                .unwrap_or(sanitized);
            let call_id = identity.next_function_id();
            let arguments = call
                .get("args")
                .map(|value| serde_json::to_string(value).unwrap_or_default())
                .unwrap_or_default();
            let item = json!({
                "id": format!("fc_{call_id}"),
                "type": "function_call",
                "status": "completed",
                "arguments": arguments,
                "call_id": call_id,
                "name": name
            });
            let index = function_outputs.len();
            function_outputs.push(FunctionOutput { item, signature });
            order.push(OutputOrder::Function(index));
            continue;
        }

        if signature.is_empty() {
            continue;
        }
        let detached = if !reasoning_text.is_empty() {
            if reasoning_encrypted.is_empty() {
                reasoning_encrypted = signature;
                reasoning_direction = STANDALONE.to_owned();
                reasoning_target = TEXT.to_owned();
                None
            } else if reasoning_encrypted != signature {
                flush_reasoning(
                    &mut reasoning_text,
                    &mut reasoning_encrypted,
                    &mut reasoning_direction,
                    &mut reasoning_target,
                    &mut reasoning_outputs,
                    &mut reasoning_signatures,
                    &mut order,
                );
                Some(DetachedOutput {
                    signature,
                    direction: PREVIOUS.to_owned(),
                    target_kind: TEXT.to_owned(),
                })
            } else {
                None
            }
        } else if !message_text.is_empty() {
            if message_signatures.is_empty() {
                message_signatures.push(signature);
                None
            } else if message_signatures.last() != Some(&signature) {
                flush_message(
                    &mut message_text,
                    &mut message_signatures,
                    &mut message_outputs,
                    &mut order,
                );
                Some(DetachedOutput {
                    signature,
                    direction: PREVIOUS.to_owned(),
                    target_kind: TEXT.to_owned(),
                })
            } else {
                None
            }
        } else if !function_outputs.is_empty() {
            Some(DetachedOutput {
                signature,
                direction: PREVIOUS.to_owned(),
                target_kind: FUNCTION.to_owned(),
            })
        } else {
            Some(DetachedOutput {
                signature,
                direction: NEXT.to_owned(),
                target_kind: ANY.to_owned(),
            })
        };
        if let Some(detached) = detached {
            let index = detached_outputs.len();
            detached_outputs.push(detached);
            order.push(OutputOrder::Detached(index));
        }
    }

    flush_reasoning(
        &mut reasoning_text,
        &mut reasoning_encrypted,
        &mut reasoning_direction,
        &mut reasoning_target,
        &mut reasoning_outputs,
        &mut reasoning_signatures,
        &mut order,
    );
    flush_message(
        &mut message_text,
        &mut message_signatures,
        &mut message_outputs,
        &mut order,
    );

    let stem = response_id.strip_prefix("resp_").unwrap_or(response_id);
    let multiple_reasoning = reasoning_outputs.len() > 1;
    let mut output = Vec::<Value>::new();
    let mut seen_detached = HashSet::<String>::new();
    let mut detached_index = 0_usize;
    for item in order {
        match item {
            OutputOrder::Reasoning(index) => {
                let reasoning = &reasoning_outputs[index];
                let id = if multiple_reasoning {
                    format!("rs_{stem}_{index}")
                } else {
                    format!("rs_{stem}")
                };
                let encrypted = encode_carrier(
                    &reasoning.signature,
                    &reasoning.direction,
                    &reasoning.target_kind,
                );
                let mut value = json!({"id":id,"type":"reasoning","encrypted_content":encrypted});
                if !reasoning.text.is_empty() {
                    value["summary"] = json!([{"type":"summary_text","text":reasoning.text}]);
                }
                output.push(value);
            }
            OutputOrder::Message(index) => {
                let message = &message_outputs[index];
                for signature in &message.signatures {
                    append_detached(
                        &mut output,
                        &mut seen_detached,
                        &mut detached_index,
                        stem,
                        signature,
                        NEXT,
                        TEXT,
                    );
                }
                output.push(json!({
                    "id":format!("msg_{stem}_{index}"),
                    "type":"message",
                    "status":"completed",
                    "content":[{"type":"output_text","annotations":[],"logprobs":[],"text":message.text}],
                    "role":"assistant"
                }));
            }
            OutputOrder::Function(index) => {
                let function = &function_outputs[index];
                append_detached(
                    &mut output,
                    &mut seen_detached,
                    &mut detached_index,
                    stem,
                    &function.signature,
                    NEXT,
                    FUNCTION,
                );
                output.push(function.item.clone());
            }
            OutputOrder::Detached(index) => {
                let detached = &detached_outputs[index];
                if !reasoning_signatures.contains(&detached.signature) {
                    append_detached(
                        &mut output,
                        &mut seen_detached,
                        &mut detached_index,
                        stem,
                        &detached.signature,
                        &detached.direction,
                        &detached.target_kind,
                    );
                }
            }
        }
    }
    output
}

#[allow(clippy::too_many_arguments)]
fn flush_reasoning(
    text: &mut String,
    signature: &mut String,
    direction: &mut String,
    target: &mut String,
    outputs: &mut Vec<ReasoningOutput>,
    signatures: &mut HashSet<String>,
    order: &mut Vec<OutputOrder>,
) {
    if text.is_empty() && signature.is_empty() {
        return;
    }
    let index = outputs.len();
    if !signature.is_empty() {
        signatures.insert(signature.clone());
    }
    outputs.push(ReasoningOutput {
        text: std::mem::take(text),
        signature: std::mem::take(signature),
        direction: std::mem::take(direction),
        target_kind: std::mem::take(target),
    });
    order.push(OutputOrder::Reasoning(index));
}

fn flush_message(
    text: &mut String,
    signatures: &mut Vec<String>,
    outputs: &mut Vec<MessageOutput>,
    order: &mut Vec<OutputOrder>,
) {
    if text.is_empty() {
        return;
    }
    let index = outputs.len();
    outputs.push(MessageOutput {
        text: std::mem::take(text),
        signatures: std::mem::take(signatures),
    });
    order.push(OutputOrder::Message(index));
}

#[allow(clippy::too_many_arguments)]
fn append_detached(
    output: &mut Vec<Value>,
    seen: &mut HashSet<String>,
    index: &mut usize,
    response_stem: &str,
    signature: &str,
    direction: &str,
    target: &str,
) {
    if signature.is_empty() || !seen.insert(signature.to_owned()) {
        return;
    }
    let placement = if direction == PREVIOUS {
        "after"
    } else {
        "before"
    };
    output.push(json!({
        "id":format!("rs_{response_stem}_detached_{placement}_{}", *index),
        "type":"reasoning",
        "encrypted_content":encode_carrier(signature, direction, target),
        "summary":[]
    }));
    *index += 1;
}

fn encode_carrier(signature: &str, direction: &str, target: &str) -> String {
    let signature = signature.trim();
    if signature.is_empty() {
        return String::new();
    }
    if direction.is_empty() {
        return signature.to_owned();
    }
    format!(
        "{CARRIER_PREFIX}{direction}:{target}:{}",
        STANDARD_NO_PAD.encode(signature.as_bytes())
    )
}

fn convert_usage(usage: &Value) -> Value {
    let mut output = Map::new();
    output.insert(
        "input_tokens".to_owned(),
        Value::from(
            usage
                .get("promptTokenCount")
                .and_then(Value::as_i64)
                .unwrap_or(0),
        ),
    );
    output.insert(
        "input_tokens_details".to_owned(),
        json!({"cached_tokens":usage.get("cachedContentTokenCount").and_then(Value::as_i64).unwrap_or(0)}),
    );
    if let Some(value) = usage.get("candidatesTokenCount").and_then(Value::as_i64) {
        output.insert("output_tokens".to_owned(), Value::from(value));
    }
    if let Some(value) = usage.get("thoughtsTokenCount").and_then(Value::as_i64) {
        output.insert(
            "output_tokens_details".to_owned(),
            json!({"reasoning_tokens":value}),
        );
    }
    if let Some(value) = usage.get("totalTokenCount").and_then(Value::as_i64) {
        output.insert("total_tokens".to_owned(), Value::from(value));
    }
    Value::Object(output)
}

fn sanitized_tool_name_map(request: &Value) -> HashMap<String, String> {
    request
        .get("tools")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|tool| tool.get("name").and_then(Value::as_str))
        .filter_map(|name| {
            let sanitized = sanitize_function_name(name);
            (sanitized != name).then(|| (sanitized, name.to_owned()))
        })
        .fold(HashMap::new(), |mut names, (sanitized, original)| {
            names.entry(sanitized).or_insert(original);
            names
        })
}

fn sanitize_function_name(name: &str) -> String {
    if name.is_empty() {
        return String::new();
    }
    let mut output: String = name
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || "_.:-".contains(character) {
                character
            } else {
                '_'
            }
        })
        .collect();
    if !output
        .as_bytes()
        .first()
        .is_some_and(|byte| byte.is_ascii_alphabetic() || *byte == b'_')
    {
        if output.len() >= 64 {
            output.truncate(63);
        }
        output.insert(0, '_');
    }
    output.truncate(output.len().min(64));
    output
}

// RFC3339 timestamps emitted by Google are UTC in the supported response
// surface. This parser intentionally accepts the provider's common `Z` form
// without pulling a time framework into the portable core.
fn parse_rfc3339_unix_seconds(raw: &str) -> Option<i64> {
    let (date, time) = raw.trim_end_matches('Z').split_once('T')?;
    let mut date = date.split('-').map(|part| part.parse::<i64>().ok());
    let year = date.next()??;
    let month = date.next()??;
    let day = date.next()??;
    let time = time.split('.').next().unwrap_or(time);
    let mut time = time.split(':').map(|part| part.parse::<i64>().ok());
    let hour = time.next()??;
    let minute = time.next()??;
    let second = time.next()??;
    Some(days_from_civil(year, month, day) * 86_400 + hour * 3_600 + minute * 60 + second)
}

fn days_from_civil(year: i64, month: i64, day: i64) -> i64 {
    let year = year - i64::from(month <= 2);
    let era = if year >= 0 { year } else { year - 399 } / 400;
    let year_of_era = year - era * 400;
    let month_prime = month + if month > 2 { -3 } else { 9 };
    let day_of_year = (153 * month_prime + 2) / 5 + day - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    era * 146_097 + day_of_era - 719_468
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converts_envelope_text_request_echo_and_usage() {
        let response: Value = serde_json::from_slice(
            &convert_antigravity_response_to_openai_responses_non_stream(
                br#"{"request":{"model":"gemini-3-flash-agent","instructions":"concise","temperature":0.25}}"#,
                b"",
                br#"{"response":{"responseId":"ag-1","createTime":"2026-08-03T12:34:56.123Z","modelVersion":"ignored","candidates":[{"content":{"parts":[{"text":"hello"}]}}],"usageMetadata":{"promptTokenCount":3,"cachedContentTokenCount":1,"candidatesTokenCount":2,"thoughtsTokenCount":0,"totalTokenCount":5}}}"#,
            ),
        )
        .unwrap();
        assert_eq!(response["id"], "resp_ag-1");
        assert_eq!(response["created_at"], 1_785_760_496_i64);
        assert_eq!(response["instructions"], "concise");
        assert_eq!(response["output"][0]["content"][0]["text"], "hello");
        assert_eq!(
            response["usage"]["input_tokens_details"]["cached_tokens"],
            1
        );
    }

    #[test]
    fn preserves_reasoning_message_function_order_and_carriers() {
        let response: Value = serde_json::from_slice(
            &convert_antigravity_response_to_openai_responses_non_stream(
                br#"{"tools":[{"type":"function","name":"read file"}]}"#,
                b"",
                br#"{"response":{"responseId":"ag-2","candidates":[{"content":{"parts":[{"thought":true,"text":"think","thoughtSignature":"sig-a"},{"text":"answer"},{"functionCall":{"name":"read_file","args":{"path":"x"}},"thoughtSignature":"sig-b"}]}}]}}"#,
            ),
        )
        .unwrap();
        assert_eq!(response["output"][0]["type"], "reasoning");
        assert!(response["output"][0]["encrypted_content"]
            .as_str()
            .unwrap()
            .starts_with(CARRIER_PREFIX));
        assert_eq!(response["output"][1]["type"], "message");
        assert_eq!(response["output"][3]["type"], "function_call");
        assert_eq!(response["output"][3]["name"], "read file");
    }

    #[test]
    fn stream_is_incremental_and_completes_with_ordered_output_and_usage() {
        let request = br#"{"request":{"model":"gemini-3-flash-agent"}}"#;
        let mut state = AntigravityToResponsesState::default();
        let first = convert_antigravity_response_to_openai_responses_stream(
            request,
            b"",
            br#"{"response":{"responseId":"stream-1","createTime":"2026-08-03T12:34:56Z","candidates":[{"content":{"parts":[{"thought":true,"text":"think","thoughtSignature":"sig"}]}}]}}"#,
            &mut state,
        );
        let first_types = first
            .iter()
            .map(|event| event_type(event))
            .collect::<Vec<_>>();
        assert_eq!(
            first_types,
            [
                "response.created",
                "response.in_progress",
                "response.output_item.added",
                "response.reasoning_summary_part.added",
                "response.reasoning_summary_text.delta",
            ]
        );

        let second = convert_antigravity_response_to_openai_responses_stream(
            request,
            b"",
            br#"{"response":{"candidates":[{"content":{"parts":[{"text":"answer"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":2,"thoughtsTokenCount":1,"totalTokenCount":7}}}"#,
            &mut state,
        );
        assert!(!second
            .iter()
            .any(|event| event_type(event) == "response.created"));
        let completed = second
            .iter()
            .map(|event| event_data(event))
            .find(|event| event["type"] == "response.completed")
            .unwrap();
        assert_eq!(completed["response"]["output"][0]["type"], "reasoning");
        assert_eq!(completed["response"]["output"][1]["type"], "message");
        assert_eq!(completed["response"]["usage"]["total_tokens"], 7);
        assert!(convert_antigravity_response_to_openai_responses_stream(
            request, b"", b"[DONE]", &mut state,
        )
        .is_empty());
    }

    fn event_data(raw: &[u8]) -> Value {
        let line = raw
            .split(|byte| *byte == b'\n')
            .find(|line| line.starts_with(b"data: "))
            .unwrap();
        serde_json::from_slice(&line[6..]).unwrap()
    }

    fn event_type(raw: &[u8]) -> &str {
        let line = raw
            .split(|byte| *byte == b'\n')
            .find(|line| line.starts_with(b"event: "))
            .unwrap();
        std::str::from_utf8(&line[7..]).unwrap()
    }
}
