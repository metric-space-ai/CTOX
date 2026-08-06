// ref: internal/runtime/executor/kimi_thinking_replay.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

//! Kimi signed-thinking replay.
//!
//! Upstream uses a package cache and request-context identity. CTOX injects a
//! bounded cache and clock, and derives the privacy scope only from trusted
//! execution metadata or a credential-scoped Claude session.

use std::collections::BTreeMap;
use std::sync::Arc;

use serde_json::Value;
use sha2::{Digest, Sha256};
use tokio::sync::mpsc;

use crate::internal::cache::{
    KimiThinkingReplayCache, KimiThinkingReplaySnapshot, KIMI_THINKING_REPLAY_MAX_BLOCKS_PER_ENTRY,
    KIMI_THINKING_REPLAY_MAX_BYTES_PER_ENTRY,
};
use crate::internal::thinking::parse_suffix;
use crate::sdk::pluginapi::{
    ExecutorRequest, ExecutorStreamChunk, ExecutorStreamResponse, PluginExecutionError,
};

use super::kimi_executor::{normalize_kimi_upstream_model, KimiExecutorError};

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct KimiThinkingReplayScope {
    pub model_family: String,
    pub session_key: String,
    pub snapshot: KimiThinkingReplaySnapshot,
    pub cache_ready: bool,
    pub replay_applied: bool,
}

impl KimiThinkingReplayScope {
    #[must_use]
    pub fn valid(&self) -> bool {
        !self.model_family.trim().is_empty() && !self.session_key.trim().is_empty()
    }
}

#[must_use]
pub fn kimi_thinking_replay_model_family(model: &str) -> String {
    let base = parse_suffix(model.trim()).model_name;
    let normalized = normalize_kimi_upstream_model(&base);
    match normalized.as_str() {
        "k3" | "k3-256k" => "k3".into(),
        _ => normalized,
    }
}

#[must_use]
pub fn kimi_thinking_replay_scope_from_request(
    request: &ExecutorRequest,
) -> KimiThinkingReplayScope {
    let session_key = trusted_session_key(request).unwrap_or_default();
    KimiThinkingReplayScope {
        model_family: kimi_thinking_replay_model_family(&request.model),
        session_key,
        ..KimiThinkingReplayScope::default()
    }
}

#[must_use]
pub fn prepare_kimi_thinking_replay_request(
    cache: &KimiThinkingReplayCache,
    now_ms: i64,
    mut request: ExecutorRequest,
) -> (ExecutorRequest, KimiThinkingReplayScope) {
    let mut scope = kimi_thinking_replay_scope_from_request(&request);
    if !scope.valid() {
        return (request, scope);
    }
    let Ok((content, snapshot, found)) =
        cache.read(&scope.model_family, &scope.session_key, now_ms)
    else {
        return (request, scope);
    };
    scope.snapshot = snapshot;
    scope.cache_ready = true;
    if found {
        let (updated, restored) = restore_kimi_thinking_replay_content(&request.payload, &content);
        if restored {
            request.payload = updated;
            scope.replay_applied = true;
        }
    }
    (request, scope)
}

pub fn cache_kimi_thinking_replay_response(
    cache: &KimiThinkingReplayCache,
    now_ms: i64,
    scope: &KimiThinkingReplayScope,
    response: &[u8],
) {
    let content = serde_json::from_slice::<Value>(response)
        .ok()
        .and_then(|root| root.get("content").and_then(Value::as_array).cloned())
        .and_then(|content| serde_json::to_vec(&content).ok());
    if let Some(content) = content {
        cache_kimi_thinking_replay_content(cache, now_ms, scope, &content);
    }
}

pub fn cache_kimi_thinking_replay_content(
    cache: &KimiThinkingReplayCache,
    now_ms: i64,
    scope: &KimiThinkingReplayScope,
    content: &[u8],
) {
    if !scope.valid() || !scope.cache_ready {
        return;
    }
    if kimi_thinking_replay_content_is_replayable(content) {
        let _ = cache.replace_if_unchanged(
            &scope.model_family,
            &scope.session_key,
            &scope.snapshot,
            content,
            now_ms,
        );
    } else {
        clear_kimi_thinking_replay_content(cache, now_ms, scope);
    }
}

#[must_use]
pub fn should_clear_kimi_thinking_replay_after_error(
    error: &(dyn std::error::Error + 'static),
) -> bool {
    error
        .downcast_ref::<KimiExecutorError>()
        .and_then(KimiExecutorError::status_code)
        .is_some_and(|status| matches!(status, 400 | 422))
}

pub fn clear_kimi_thinking_replay_content(
    cache: &KimiThinkingReplayCache,
    now_ms: i64,
    scope: &KimiThinkingReplayScope,
) {
    if scope.valid() && scope.cache_ready {
        let _ = cache.delete_if_unchanged(
            &scope.model_family,
            &scope.session_key,
            &scope.snapshot,
            now_ms,
        );
    }
}

#[must_use]
pub fn kimi_thinking_replay_content_is_replayable(content: &[u8]) -> bool {
    let Ok(parts) = serde_json::from_slice::<Vec<Value>>(content) else {
        return false;
    };
    let signed_thinking = parts.iter().any(|part| {
        part.get("type").and_then(Value::as_str) == Some("thinking")
            && part
                .get("signature")
                .and_then(Value::as_str)
                .is_some_and(|value| !value.trim().is_empty())
    });
    let tool_use = parts.iter().any(|part| {
        part.get("type").and_then(Value::as_str) == Some("tool_use")
            && part
                .get("id")
                .and_then(Value::as_str)
                .is_some_and(|value| !value.trim().is_empty())
    });
    signed_thinking && tool_use
}

#[must_use]
pub fn restore_kimi_thinking_replay_content(body: &[u8], cached_content: &[u8]) -> (Vec<u8>, bool) {
    let Ok(cached) = serde_json::from_slice::<Value>(cached_content) else {
        return (body.to_vec(), false);
    };
    let Some(cached_parts) = kimi_non_thinking_content_parts(&cached) else {
        return (body.to_vec(), false);
    };
    let Ok(mut root) = serde_json::from_slice::<Value>(body) else {
        return (body.to_vec(), false);
    };
    let Some(messages) = root.get_mut("messages").and_then(Value::as_array_mut) else {
        return (body.to_vec(), false);
    };
    for message in messages.iter_mut().rev() {
        if !message
            .get("role")
            .and_then(Value::as_str)
            .is_some_and(|role| role.trim().eq_ignore_ascii_case("assistant"))
        {
            continue;
        }
        let Some(current) = message.get("content") else {
            continue;
        };
        if current == &cached || kimi_content_has_thinking(current) {
            continue;
        }
        if kimi_non_thinking_content_parts(current).as_ref() != Some(&cached_parts) {
            continue;
        }
        if let Some(object) = message.as_object_mut() {
            object.insert("content".into(), cached);
            return (
                serde_json::to_vec(&root).unwrap_or_else(|_| body.to_vec()),
                true,
            );
        }
    }
    (body.to_vec(), false)
}

fn kimi_content_has_thinking(content: &Value) -> bool {
    content.as_array().is_some_and(|parts| {
        parts.iter().any(|part| {
            matches!(
                part.get("type").and_then(Value::as_str),
                Some("thinking") | Some("redacted_thinking")
            )
        })
    })
}

fn kimi_non_thinking_content_parts(content: &Value) -> Option<Vec<Value>> {
    let parts = content.as_array()?;
    let mut output = Vec::new();
    let mut has_tool_use = false;
    for part in parts {
        match part.get("type").and_then(Value::as_str) {
            Some("thinking") | Some("redacted_thinking") => continue,
            Some("tool_use") => {
                if part
                    .get("id")
                    .and_then(Value::as_str)
                    .is_none_or(|id| id.trim().is_empty())
                {
                    return None;
                }
                has_tool_use = true;
            }
            _ => {}
        }
        output.push(canonical_json(part));
    }
    has_tool_use.then_some(output)
}

fn canonical_json(value: &Value) -> Value {
    match value {
        Value::Array(values) => Value::Array(values.iter().map(canonical_json).collect()),
        Value::Object(object) => Value::Object(
            object
                .iter()
                .map(|(key, value)| (key.clone(), canonical_json(value)))
                .collect(),
        ),
        _ => value.clone(),
    }
}

#[derive(Clone, Debug, Default)]
struct KimiThinkingReplayStreamBlock {
    raw: Value,
    text: String,
    thinking: String,
    signature: String,
    input: String,
    text_initialized: bool,
    thinking_initialized: bool,
    signature_initialized: bool,
    has_input_delta: bool,
    finished: bool,
}

#[derive(Debug, Default)]
pub struct KimiThinkingReplayStreamAccumulator {
    blocks: BTreeMap<usize, KimiThinkingReplayStreamBlock>,
    observed: bool,
    complete: bool,
    pub upstream_error: bool,
    abandoned: bool,
    bytes_used: usize,
}

impl KimiThinkingReplayStreamAccumulator {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn observe(&mut self, chunk: &[u8]) {
        for line in chunk.split(|byte| *byte == b'\n') {
            let line = trim_ascii(line);
            let Some(payload) = line.strip_prefix(b"data:").map(trim_ascii) else {
                continue;
            };
            if payload.is_empty() || payload == b"[DONE]" {
                continue;
            }
            let Ok(root) = serde_json::from_slice::<Value>(payload) else {
                self.abandon();
                continue;
            };
            match root.get("type").and_then(Value::as_str) {
                Some("message_start") => self.observed = true,
                Some("content_block_start") if !self.abandoned => self.observe_block_start(&root),
                Some("content_block_delta") if !self.abandoned => self.observe_block_delta(&root),
                Some("content_block_stop") if !self.abandoned => {
                    self.finish_block(
                        root.get("index").and_then(Value::as_u64).unwrap_or(0) as usize
                    );
                }
                Some("message_stop") => self.complete = true,
                Some("error") => {
                    self.upstream_error = true;
                    self.abandon();
                }
                _ => {}
            }
        }
    }

    fn observe_block_start(&mut self, root: &Value) {
        let index = root.get("index").and_then(Value::as_u64).unwrap_or(0) as usize;
        let Some(block) = root.get("content_block").filter(|block| block.is_object()) else {
            self.abandon();
            return;
        };
        if self.blocks.len() >= KIMI_THINKING_REPLAY_MAX_BLOCKS_PER_ENTRY
            || self.blocks.contains_key(&index)
        {
            self.abandon();
            return;
        }
        let size = serde_json::to_vec(block).map_or(0, |raw| raw.len());
        if self.reserve_bytes(size) {
            self.blocks.insert(
                index,
                KimiThinkingReplayStreamBlock {
                    raw: block.clone(),
                    ..KimiThinkingReplayStreamBlock::default()
                },
            );
        }
    }

    fn observe_block_delta(&mut self, root: &Value) {
        let index = root.get("index").and_then(Value::as_u64).unwrap_or(0) as usize;
        let Some(mut block) = self.blocks.remove(&index) else {
            self.abandon();
            return;
        };
        let delta = root.get("delta").unwrap_or(&Value::Null);
        let valid = match delta.get("type").and_then(Value::as_str) {
            Some("text_delta") => self.append_block_text(
                &mut block,
                "text",
                delta.get("text").and_then(Value::as_str).unwrap_or(""),
            ),
            Some("thinking_delta") => self.append_block_text(
                &mut block,
                "thinking",
                delta.get("thinking").and_then(Value::as_str).unwrap_or(""),
            ),
            Some("signature_delta") => self.append_block_text(
                &mut block,
                "signature",
                delta.get("signature").and_then(Value::as_str).unwrap_or(""),
            ),
            Some("input_json_delta") => {
                let suffix = delta
                    .get("partial_json")
                    .and_then(Value::as_str)
                    .unwrap_or("");
                if self.reserve_bytes(suffix.len()) {
                    block.input.push_str(suffix);
                    block.has_input_delta = true;
                    true
                } else {
                    false
                }
            }
            _ => false,
        };
        if !valid {
            self.abandon();
        } else if !self.abandoned {
            self.blocks.insert(index, block);
        }
    }

    fn append_block_text(
        &mut self,
        block: &mut KimiThinkingReplayStreamBlock,
        field: &str,
        suffix: &str,
    ) -> bool {
        let (builder, initialized) = match field {
            "text" => (&mut block.text, &mut block.text_initialized),
            "thinking" => (&mut block.thinking, &mut block.thinking_initialized),
            "signature" => (&mut block.signature, &mut block.signature_initialized),
            _ => return false,
        };
        if !*initialized {
            let initial = block.raw.get(field).and_then(Value::as_str).unwrap_or("");
            if !self.reserve_bytes(initial.len()) {
                return false;
            }
            builder.push_str(initial);
            *initialized = true;
        }
        if !self.reserve_bytes(suffix.len()) {
            return false;
        }
        builder.push_str(suffix);
        true
    }

    fn finish_block(&mut self, index: usize) {
        let Some(block) = self.blocks.get_mut(&index) else {
            self.abandon();
            return;
        };
        if block.has_input_delta && serde_json::from_str::<Value>(&block.input).is_err() {
            self.abandon();
            return;
        }
        block.finished = true;
    }

    fn reserve_bytes(&mut self, count: usize) -> bool {
        let Some(total) = self.bytes_used.checked_add(count) else {
            self.abandon();
            return false;
        };
        if total > KIMI_THINKING_REPLAY_MAX_BYTES_PER_ENTRY {
            self.abandon();
            return false;
        }
        self.bytes_used = total;
        true
    }

    fn abandon(&mut self) {
        self.abandoned = true;
        self.blocks.clear();
        self.bytes_used = 0;
    }

    #[must_use]
    pub fn content(&mut self) -> Option<Vec<u8>> {
        if !self.observed || !self.complete || self.upstream_error || self.abandoned {
            return None;
        }
        let mut output = Vec::with_capacity(self.blocks.len());
        for block in self.blocks.values() {
            if !block.finished {
                self.abandon();
                return None;
            }
            let mut raw = block.raw.clone();
            let Some(object) = raw.as_object_mut() else {
                self.abandon();
                return None;
            };
            if block.text_initialized {
                object.insert("text".into(), Value::String(block.text.clone()));
            }
            if block.thinking_initialized {
                object.insert("thinking".into(), Value::String(block.thinking.clone()));
            }
            if block.signature_initialized {
                object.insert("signature".into(), Value::String(block.signature.clone()));
            }
            if block.has_input_delta {
                let Ok(input) = serde_json::from_str::<Value>(&block.input) else {
                    self.abandon();
                    return None;
                };
                object.insert("input".into(), input);
            }
            output.push(raw);
        }
        let encoded = serde_json::to_vec(&output).ok()?;
        if encoded.len() > KIMI_THINKING_REPLAY_MAX_BYTES_PER_ENTRY {
            self.abandon();
            None
        } else {
            Some(encoded)
        }
    }
}

#[must_use]
pub fn wrap_kimi_thinking_replay_stream(
    cache: Arc<KimiThinkingReplayCache>,
    now_ms: i64,
    response: ExecutorStreamResponse,
    scope: KimiThinkingReplayScope,
) -> ExecutorStreamResponse {
    if !scope.valid() {
        return response;
    }
    let headers = response.headers.clone();
    let mut source = response.chunks;
    let (sender, receiver) = mpsc::channel(16);
    tokio::spawn(async move {
        let mut accumulator = KimiThinkingReplayStreamAccumulator::new();
        let mut has_error = false;
        while let Some(chunk) = source.recv().await {
            if chunk.error.is_some() {
                has_error = true;
            } else {
                accumulator.observe(&chunk.payload);
            }
            if sender.send(chunk).await.is_err() {
                return;
            }
        }
        if has_error {
            return;
        }
        if let Some(content) = accumulator.content() {
            cache_kimi_thinking_replay_content(&cache, now_ms, &scope, &content);
        } else if accumulator.upstream_error && scope.replay_applied {
            clear_kimi_thinking_replay_content(&cache, now_ms, &scope);
        }
    });
    ExecutorStreamResponse {
        headers,
        chunks: receiver,
    }
}

fn trusted_session_key(request: &ExecutorRequest) -> Option<String> {
    for key in ["execution_session_id", "derived_session_id"] {
        if let Some(session) = request
            .metadata
            .get(key)
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|session| !session.is_empty())
        {
            return Some(format!("execution:{session}"));
        }
    }
    let session = serde_json::from_slice::<Value>(&request.payload)
        .ok()
        .and_then(|root| {
            root.pointer("/metadata/user_id")
                .and_then(Value::as_str)
                .map(str::to_owned)
        })
        .and_then(|raw| serde_json::from_str::<Value>(&raw).ok())
        .and_then(|metadata| {
            metadata
                .get("session_id")
                .and_then(Value::as_str)
                .map(str::to_owned)
        })?;
    let credential = request
        .auth_attributes
        .get("downstream_api_key")
        .or_else(|| request.auth_attributes.get("caller_credential"))?
        .trim();
    if session.trim().is_empty() || credential.is_empty() {
        return None;
    }
    let credential_scope = format!("{:x}", Sha256::digest(credential.as_bytes()));
    let agent = request
        .metadata
        .get("agent_id")
        .and_then(Value::as_str)
        .unwrap_or("main");
    Some(format!(
        "client:{credential_scope}:claude:{}:agent:{}",
        session.trim(),
        agent.trim()
    ))
}

fn trim_ascii(mut bytes: &[u8]) -> &[u8] {
    while bytes.first().is_some_and(|byte| byte.is_ascii_whitespace()) {
        bytes = &bytes[1..];
    }
    while bytes.last().is_some_and(|byte| byte.is_ascii_whitespace()) {
        bytes = &bytes[..bytes.len() - 1];
    }
    bytes
}

pub fn stream_error(error: KimiExecutorError) -> ExecutorStreamChunk {
    ExecutorStreamChunk {
        payload: Vec::new(),
        error: Some(Arc::new(error) as PluginExecutionError),
    }
}
