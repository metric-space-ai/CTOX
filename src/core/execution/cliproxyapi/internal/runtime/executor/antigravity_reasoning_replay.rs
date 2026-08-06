// ref: internal/runtime/executor/antigravity_reasoning_replay.go:800-1505 @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

use crate::internal::cache::antigravity_reasoning_replay_cache::{
    AntigravityReasoningReplayCache, AntigravityReasoningReplayError,
    AntigravityReasoningReplaySnapshot, ANTIGRAVITY_REPLAY_MAX_BYTES_PER_ENTRY,
    ANTIGRAVITY_REPLAY_MAX_ITEMS_PER_ENTRY,
};

const BYPASS_SIGNATURE: &str = "skip_thought_signature_validator";
const CLAUDE_PROVENANCE_PREFIX: &str = "cpa_gemini_";

/// Applies already-normalized replay items to a Gemini-shaped Antigravity
/// request. Every write is pinned either to the signed part fingerprint or to
/// an exact function-call semantic match; ambiguous items fail closed.
pub fn apply_antigravity_reasoning_replay_items(
    payload: &[u8],
    items: &[Vec<u8>],
) -> Result<(Vec<u8>, usize), AntigravityReplayApplyError> {
    let mut root = serde_json::from_slice::<Value>(payload)
        .map_err(|_| AntigravityReplayApplyError::InvalidPayload)?;
    let mut applied = 0_usize;
    for raw in items {
        let Ok(item) = serde_json::from_slice::<Value>(raw) else {
            continue;
        };
        let changed = match item.get("type").and_then(Value::as_str) {
            Some("thought_signature") => apply_thought_signature(&mut root, &item),
            Some("function_call_part") => apply_function_call(&mut root, &item),
            _ => false,
        };
        applied += usize::from(changed);
    }
    Ok((
        serde_json::to_vec(&root).unwrap_or_else(|_| payload.to_vec()),
        applied,
    ))
}

/// Reads one replay lane, applies its current chain to the outgoing request and
/// creates the terminal accumulator bound to the exact cache snapshot. The
/// returned payload must be the payload sent upstream; this guarantees that
/// the accumulator republishes the complete visible chain, not only the new
/// response tail.
pub fn prepare_antigravity_reasoning_replay(
    cache: Arc<AntigravityReasoningReplayCache>,
    model: &str,
    session_key: &str,
    payload: &[u8],
    now_ms: i64,
) -> Result<(Vec<u8>, AntigravityReasoningReplayAccumulator), AntigravityReplaySessionError> {
    let (items, snapshot, _) = cache
        .read(model, session_key, now_ms)
        .map_err(AntigravityReplaySessionError::Cache)?;
    let (payload, _) = apply_antigravity_reasoning_replay_items(payload, &items)
        .map_err(|_| AntigravityReplaySessionError::InvalidPayload)?;
    let root = serde_json::from_slice::<Value>(&payload)
        .map_err(|_| AntigravityReplaySessionError::InvalidPayload)?;
    let accumulator =
        AntigravityReasoningReplayAccumulator::new(cache, model, session_key, snapshot, root);
    Ok((payload, accumulator))
}

pub(crate) fn antigravity_replay_session_key(
    original_request: &[u8],
    provider_payload: &[u8],
) -> Option<String> {
    if let Ok(root) = serde_json::from_slice::<Value>(original_request) {
        for (path, prefix) in [
            ("/session_id", "responses:"),
            ("/metadata/session_id", "responses:"),
            ("/prompt_cache_key", "prompt-cache:"),
        ] {
            if let Some(value) = root
                .pointer(path)
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|value| !value.is_empty())
            {
                return Some(format!("{prefix}{value}"));
            }
        }
    }
    let root = serde_json::from_slice::<Value>(provider_payload).ok()?;
    [
        "/sessionId",
        "/session_id",
        "/request/sessionId",
        "/request/session_id",
    ]
    .into_iter()
    .find_map(|path| {
        root.pointer(path)
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
    })
    .map(|value| format!("session:{value}"))
}

pub(crate) fn replay_now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .ok()
        .and_then(|duration| i64::try_from(duration.as_millis()).ok())
        .unwrap_or_default()
}

/// Request-scoped replay publication state. It owns no credentials and its
/// Debug representation intentionally omits the unhashed session lane.
pub struct AntigravityReasoningReplayAccumulator {
    cache: Arc<AntigravityReasoningReplayCache>,
    model: String,
    session_key: String,
    snapshot: AntigravityReasoningReplaySnapshot,
    request: Value,
    items: Vec<Vec<u8>>,
    item_bytes: usize,
    content_index: usize,
    next_part_index: usize,
    function_occurrences: HashMap<String, usize>,
    segment_occurrences: HashMap<String, usize>,
    seen_function_parts: HashSet<String>,
    seen_signatures: HashSet<String>,
    last_function_item: Option<usize>,
    active_segment: Option<ReplaySegment>,
    overflow: bool,
    terminal: bool,
}

#[derive(Debug)]
struct ReplaySegment {
    kind: &'static str,
    part_index: usize,
    text: String,
    signatures: Vec<String>,
}

impl std::fmt::Debug for AntigravityReasoningReplayAccumulator {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("AntigravityReasoningReplayAccumulator")
            .field("model", &self.model)
            .field("session_key", &"[SHA-256 INDEXED]")
            .field("items", &self.items.len())
            .field("item_bytes", &self.item_bytes)
            .field("overflow", &self.overflow)
            .field("terminal", &self.terminal)
            .finish()
    }
}

impl AntigravityReasoningReplayAccumulator {
    fn new(
        cache: Arc<AntigravityReasoningReplayCache>,
        model: &str,
        session_key: &str,
        snapshot: AntigravityReasoningReplaySnapshot,
        request: Value,
    ) -> Self {
        let (content_index, next_part_index) = pending_model_content_index(&request);
        let items = replay_items_from_request(&request);
        let item_bytes = items.iter().map(Vec::len).sum();
        let mut function_occurrences = HashMap::new();
        let mut segment_occurrences = HashMap::new();
        if let Some(parts) = request
            .pointer(&format!("/request/contents/{content_index}/parts"))
            .and_then(Value::as_array)
        {
            for part in parts {
                if let Some(call) = part.get("functionCall") {
                    if let Some(key) = function_key(call) {
                        *function_occurrences.entry(key).or_default() += 1;
                    }
                } else if let Some((kind, hash)) = part_fingerprint(part) {
                    *segment_occurrences
                        .entry(format!("{kind}\0{hash}"))
                        .or_default() += 1;
                }
            }
        }
        let seen_signatures = items
            .iter()
            .filter_map(|item| serde_json::from_slice::<Value>(item).ok())
            .filter_map(|item| {
                item.get("thoughtSignature")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned)
            })
            .collect();
        Self {
            cache,
            model: model.trim().to_owned(),
            session_key: session_key.trim().to_owned(),
            snapshot,
            request,
            overflow: items.len() > ANTIGRAVITY_REPLAY_MAX_ITEMS_PER_ENTRY
                || item_bytes > ANTIGRAVITY_REPLAY_MAX_BYTES_PER_ENTRY,
            items,
            item_bytes,
            content_index,
            next_part_index,
            function_occurrences,
            segment_occurrences,
            seen_function_parts: HashSet::new(),
            seen_signatures,
            last_function_item: None,
            active_segment: None,
            terminal: false,
        }
    }

    /// Accepts either a raw provider JSON envelope or one complete SSE line.
    /// Framing/fragment assembly remains the transport decoder's responsibility.
    pub fn observe_sse_line(&mut self, line: &[u8]) {
        let mut payload = trim_ascii(line);
        if let Some(rest) = payload.strip_prefix(b"data:") {
            payload = trim_ascii(rest);
        }
        if payload.is_empty() || payload == b"[DONE]" {
            return;
        }
        self.observe_response_payload(payload);
    }

    pub fn observe_response_payload(&mut self, payload: &[u8]) {
        let Ok(root) = serde_json::from_slice::<Value>(payload) else {
            return;
        };
        let response = root.get("response").unwrap_or(&root);
        let Some(candidate) = response
            .get("candidates")
            .and_then(Value::as_array)
            .and_then(|candidates| candidates.first())
        else {
            return;
        };
        if candidate
            .get("finishReason")
            .and_then(Value::as_str)
            .is_some_and(|reason| !reason.trim().is_empty())
        {
            self.terminal = true;
        }
        let Some(parts) = candidate
            .pointer("/content/parts")
            .and_then(Value::as_array)
        else {
            if self.terminal {
                self.flush_segment();
            }
            return;
        };
        for part in parts {
            self.observe_part(part);
        }
        if self.terminal {
            self.flush_segment();
        }
    }

    pub fn commit(
        mut self,
        now_ms: i64,
    ) -> Result<AntigravityReplayCommitOutcome, AntigravityReplaySessionError> {
        if !self.terminal {
            return Ok(AntigravityReplayCommitOutcome::NotTerminal);
        }
        self.flush_segment();
        if self.overflow || self.items.is_empty() {
            let deleted = self
                .cache
                .delete_if_unchanged(&self.model, &self.session_key, &self.snapshot, now_ms)
                .map_err(AntigravityReplaySessionError::Cache)?;
            return Ok(if deleted {
                AntigravityReplayCommitOutcome::Invalidated
            } else {
                AntigravityReplayCommitOutcome::RejectedStale
            });
        }
        match self.cache.replace_if_unchanged(
            &self.model,
            &self.session_key,
            &self.snapshot,
            &self.items,
            now_ms,
        ) {
            Ok(true) => Ok(AntigravityReplayCommitOutcome::Published),
            Ok(false) => Ok(AntigravityReplayCommitOutcome::RejectedStale),
            Err(error) => {
                let _ = self.cache.delete_if_unchanged(
                    &self.model,
                    &self.session_key,
                    &self.snapshot,
                    now_ms,
                );
                Err(AntigravityReplaySessionError::Cache(error))
            }
        }
    }

    /// Clears only this replay lane when Antigravity rejects a native thought
    /// signature. Other 400 responses leave the durable chain intact.
    pub fn clear_on_invalid_signature(
        &self,
        status: u16,
        body: &[u8],
        now_ms: i64,
    ) -> Result<bool, AntigravityReasoningReplayError> {
        if status != 400 || !is_invalid_antigravity_signature_error(body) {
            return Ok(false);
        }
        self.cache
            .delete_if_unchanged(&self.model, &self.session_key, &self.snapshot, now_ms)
    }

    fn observe_part(&mut self, part: &Value) {
        let part_index = self.next_part_index;
        self.next_part_index += 1;
        let signature = native_part_signature(part)
            .filter(|signature| has_native_signature(signature))
            .map(ToOwned::to_owned);
        if let Some(call) = part.get("functionCall") {
            self.flush_segment();
            self.observe_function_call(call, signature.as_deref(), part_index);
            return;
        }
        let thought = part.get("thought").and_then(Value::as_bool) == Some(true);
        let text = part.get("text").and_then(Value::as_str).unwrap_or("");
        if text.is_empty() {
            if let Some(signature) = signature {
                if self.last_function_item.is_some() {
                    self.attach_signature_to_last_function(&signature);
                } else if let Some(segment) = self.active_segment.as_mut() {
                    if self.seen_signatures.insert(signature.clone()) {
                        segment.signatures.push(signature);
                    }
                }
            }
            return;
        }
        let kind = if thought { "thought" } else { "text" };
        if self
            .active_segment
            .as_ref()
            .is_some_and(|segment| segment.kind != kind)
        {
            self.flush_segment();
        }
        let segment = self.active_segment.get_or_insert_with(|| ReplaySegment {
            kind,
            part_index,
            text: String::new(),
            signatures: Vec::new(),
        });
        segment.text.push_str(text);
        if let Some(signature) = signature {
            if self.seen_signatures.insert(signature.clone()) {
                segment.signatures.push(signature);
            }
        }
        self.last_function_item = None;
    }

    fn observe_function_call(&mut self, call: &Value, signature: Option<&str>, part_index: usize) {
        let Some(name) = call
            .get("name")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|name| !name.is_empty())
        else {
            return;
        };
        let Some(args) = call.get("args") else {
            return;
        };
        let native_id = call
            .get("id")
            .and_then(Value::as_str)
            .map(str::trim)
            .unwrap_or("");
        let semantic_key = function_semantic_key(name, args);
        let dedupe_key = format!("{native_id}\0{semantic_key}\0{}", signature.unwrap_or(""));
        if !self.seen_function_parts.insert(dedupe_key) {
            return;
        }
        let occurrence = *self
            .function_occurrences
            .entry(semantic_key)
            .and_modify(|value| *value += 1)
            .or_insert(1)
            - 1;
        let mut item = Map::new();
        item.insert(
            "type".to_owned(),
            Value::String("function_call_part".to_owned()),
        );
        item.insert("contentIndex".to_owned(), Value::from(self.content_index));
        item.insert("partIndex".to_owned(), Value::from(part_index));
        item.insert("targetOccurrence".to_owned(), Value::from(occurrence));
        item.insert("name".to_owned(), Value::String(name.to_owned()));
        item.insert("args".to_owned(), args.clone());
        if !native_id.is_empty() {
            item.insert("call_id".to_owned(), Value::String(native_id.to_owned()));
        }
        if let Some(signature) = signature {
            item.insert(
                "thoughtSignature".to_owned(),
                Value::String(signature.to_owned()),
            );
            self.seen_signatures.insert(signature.to_owned());
        }
        attach_context_hash(&mut item, &self.request, self.content_index);
        self.last_function_item = self.append_value(Value::Object(item));
    }

    fn attach_signature_to_last_function(&mut self, signature: &str) {
        if !self.seen_signatures.insert(signature.to_owned()) {
            return;
        }
        let Some(index) = self.last_function_item else {
            return;
        };
        let Some(raw) = self.items.get(index) else {
            return;
        };
        let Ok(mut item) = serde_json::from_slice::<Value>(raw) else {
            return;
        };
        if item.get("thoughtSignature").is_some() {
            return;
        }
        item["thoughtSignature"] = Value::String(signature.to_owned());
        let Ok(updated) = serde_json::to_vec(&item) else {
            return;
        };
        let new_bytes = self
            .item_bytes
            .saturating_sub(self.items[index].len())
            .saturating_add(updated.len());
        if new_bytes > ANTIGRAVITY_REPLAY_MAX_BYTES_PER_ENTRY {
            self.overflow = true;
            return;
        }
        self.items[index] = updated;
        self.item_bytes = new_bytes;
    }

    fn flush_segment(&mut self) {
        let Some(segment) = self.active_segment.take() else {
            return;
        };
        if segment.text.is_empty() || segment.signatures.is_empty() {
            return;
        }
        let mut digest = Sha256::new();
        digest.update(segment.kind.as_bytes());
        digest.update(b"\0");
        digest.update(segment.text.as_bytes());
        let target_hash = format!("{:x}", digest.finalize());
        let occurrence_key = format!("{}\0{target_hash}", segment.kind);
        let occurrence = *self
            .segment_occurrences
            .entry(occurrence_key)
            .and_modify(|value| *value += 1)
            .or_insert(1)
            - 1;
        for signature in segment.signatures {
            let mut item = Map::new();
            item.insert(
                "type".to_owned(),
                Value::String("thought_signature".to_owned()),
            );
            item.insert("thoughtSignature".to_owned(), Value::String(signature));
            item.insert("contentIndex".to_owned(), Value::from(self.content_index));
            item.insert("partIndex".to_owned(), Value::from(segment.part_index));
            item.insert(
                "targetKind".to_owned(),
                Value::String(segment.kind.to_owned()),
            );
            item.insert("targetHash".to_owned(), Value::String(target_hash.clone()));
            item.insert("targetOccurrence".to_owned(), Value::from(occurrence));
            attach_context_hash(&mut item, &self.request, self.content_index);
            self.append_value(Value::Object(item));
        }
    }

    fn append_value(&mut self, item: Value) -> Option<usize> {
        if self.overflow {
            return None;
        }
        let Ok(item) = serde_json::to_vec(&item) else {
            return None;
        };
        if self.items.len() + 1 > ANTIGRAVITY_REPLAY_MAX_ITEMS_PER_ENTRY
            || self.item_bytes.saturating_add(item.len()) > ANTIGRAVITY_REPLAY_MAX_BYTES_PER_ENTRY
        {
            self.overflow = true;
            return None;
        }
        let index = self.items.len();
        self.item_bytes += item.len();
        self.items.push(item);
        Some(index)
    }
}

#[must_use]
pub fn is_invalid_antigravity_signature_error(body: &[u8]) -> bool {
    let message = String::from_utf8_lossy(body).to_ascii_lowercase();
    (message.contains("thought signature") || message.contains("thoughtsignature"))
        && (message.contains("invalid")
            || message.contains("mismatch")
            || message.contains("required"))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AntigravityReplayCommitOutcome {
    NotTerminal,
    Published,
    RejectedStale,
    Invalidated,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AntigravityReplaySessionError {
    InvalidPayload,
    Cache(AntigravityReasoningReplayError),
}

impl std::fmt::Display for AntigravityReplaySessionError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("Antigravity replay session is unavailable")
    }
}
impl std::error::Error for AntigravityReplaySessionError {}

fn pending_model_content_index(root: &Value) -> (usize, usize) {
    let Some(contents) = root.pointer("/request/contents").and_then(Value::as_array) else {
        return (0, 0);
    };
    let Some(last) = contents.last() else {
        return (0, 0);
    };
    let is_model = last
        .get("role")
        .and_then(Value::as_str)
        .is_some_and(|role| role.trim().eq_ignore_ascii_case("model"));
    let parts = last.get("parts").and_then(Value::as_array);
    let has_function_response = parts.is_some_and(|parts| {
        parts
            .iter()
            .any(|part| part.get("functionResponse").is_some())
    });
    if is_model && !has_function_response {
        (contents.len() - 1, parts.map_or(0, Vec::len))
    } else {
        (contents.len(), 0)
    }
}

fn replay_items_from_request(root: &Value) -> Vec<Vec<u8>> {
    let Some(contents) = root.pointer("/request/contents").and_then(Value::as_array) else {
        return Vec::new();
    };
    let mut items = Vec::new();
    for (content_index, content) in contents.iter().enumerate() {
        if content
            .get("role")
            .and_then(Value::as_str)
            .is_none_or(|role| !role.trim().eq_ignore_ascii_case("model"))
        {
            continue;
        }
        let Some(parts) = content.get("parts").and_then(Value::as_array) else {
            continue;
        };
        let mut function_occurrences = HashMap::<String, usize>::new();
        for (part_index, part) in parts.iter().enumerate() {
            let signature = native_part_signature(part).filter(|value| has_native_signature(value));
            if let Some(call) = part.get("functionCall") {
                let Some(name) = call
                    .get("name")
                    .and_then(Value::as_str)
                    .map(str::trim)
                    .filter(|name| !name.is_empty())
                else {
                    continue;
                };
                let Some(args) = call.get("args") else {
                    continue;
                };
                let key = function_semantic_key(name, args);
                let occurrence = *function_occurrences
                    .entry(key)
                    .and_modify(|value| *value += 1)
                    .or_insert(1)
                    - 1;
                let mut item = Map::new();
                item.insert(
                    "type".to_owned(),
                    Value::String("function_call_part".to_owned()),
                );
                item.insert("contentIndex".to_owned(), Value::from(content_index));
                item.insert("partIndex".to_owned(), Value::from(part_index));
                item.insert("targetOccurrence".to_owned(), Value::from(occurrence));
                item.insert("name".to_owned(), Value::String(name.to_owned()));
                item.insert("args".to_owned(), args.clone());
                if let Some(id) = call
                    .get("id")
                    .and_then(Value::as_str)
                    .map(str::trim)
                    .filter(|id| !id.is_empty())
                {
                    item.insert("call_id".to_owned(), Value::String(id.to_owned()));
                }
                if let Some(signature) = signature {
                    item.insert(
                        "thoughtSignature".to_owned(),
                        Value::String(signature.to_owned()),
                    );
                }
                attach_context_hash(&mut item, root, content_index);
                if let Ok(item) = serde_json::to_vec(&Value::Object(item)) {
                    items.push(item);
                }
                continue;
            }
            let Some(signature) = signature else {
                continue;
            };
            let mut target_index = part_index;
            let mut fingerprint = part_fingerprint(part);
            if fingerprint.is_none() && part_index > 0 {
                target_index -= 1;
                fingerprint = part_fingerprint(&parts[target_index]);
            }
            let Some((kind, hash)) = fingerprint else {
                continue;
            };
            let occurrence = parts
                .iter()
                .take(target_index)
                .filter_map(part_fingerprint)
                .filter(|(candidate_kind, candidate_hash)| {
                    *candidate_kind == kind && *candidate_hash == hash
                })
                .count();
            let mut item = Map::new();
            item.insert(
                "type".to_owned(),
                Value::String("thought_signature".to_owned()),
            );
            item.insert(
                "thoughtSignature".to_owned(),
                Value::String(signature.to_owned()),
            );
            item.insert("contentIndex".to_owned(), Value::from(content_index));
            item.insert("partIndex".to_owned(), Value::from(target_index));
            item.insert("targetKind".to_owned(), Value::String(kind.to_owned()));
            item.insert("targetHash".to_owned(), Value::String(hash));
            item.insert("targetOccurrence".to_owned(), Value::from(occurrence));
            attach_context_hash(&mut item, root, content_index);
            if let Ok(item) = serde_json::to_vec(&Value::Object(item)) {
                items.push(item);
            }
        }
    }
    items
}

fn attach_context_hash(item: &mut Map<String, Value>, request: &Value, content_index: usize) {
    let context_hash = context_fingerprint(request, content_index);
    if !context_hash.is_empty() {
        item.insert("contextHash".to_owned(), Value::String(context_hash));
    }
}

fn native_part_signature(part: &Value) -> Option<&str> {
    part.get("thoughtSignature")
        .or_else(|| part.get("thought_signature"))
        .or_else(|| part.pointer("/extra_content/google/thought_signature"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|signature| !signature.is_empty())
}

fn function_key(call: &Value) -> Option<String> {
    let name = call.get("name").and_then(Value::as_str)?.trim();
    let args = call.get("args")?;
    (!name.is_empty()).then(|| function_semantic_key(name, args))
}

fn function_semantic_key(name: &str, args: &Value) -> String {
    let mut digest = Sha256::new();
    digest.update(name.trim().as_bytes());
    digest.update(b"\0");
    digest.update(canonical(args));
    format!("{:x}", digest.finalize())
}

fn trim_ascii(mut value: &[u8]) -> &[u8] {
    while value.first().is_some_and(u8::is_ascii_whitespace) {
        value = &value[1..];
    }
    while value.last().is_some_and(u8::is_ascii_whitespace) {
        value = &value[..value.len() - 1];
    }
    value
}

fn apply_thought_signature(root: &mut Value, item: &Value) -> bool {
    let Some(signature) = item
        .get("thoughtSignature")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty() && *value != BYPASS_SIGNATURE)
    else {
        return false;
    };
    let Some(content_index) = index(item, "contentIndex") else {
        return false;
    };
    if root
        .pointer(&format!("/request/contents/{content_index}/role"))
        .and_then(Value::as_str)
        .is_none_or(|role| !role.trim().eq_ignore_ascii_case("model"))
    {
        return false;
    }
    let expected_context = item
        .get("contextHash")
        .and_then(Value::as_str)
        .unwrap_or("");
    let actual_context =
        (!expected_context.is_empty()).then(|| context_fingerprint(root, content_index));
    let Some(parts) = root
        .pointer_mut(&format!("/request/contents/{content_index}/parts"))
        .and_then(Value::as_array_mut)
    else {
        return false;
    };
    let target_kind = item.get("targetKind").and_then(Value::as_str).unwrap_or("");
    let target_hash = item.get("targetHash").and_then(Value::as_str).unwrap_or("");
    let target = if !target_hash.is_empty() {
        let wanted = index(item, "targetOccurrence");
        let mut occurrence = 0_usize;
        parts.iter().enumerate().find_map(|(part_index, part)| {
            let (kind, fingerprint) = part_fingerprint(part)?;
            if fingerprint != target_hash || (!target_kind.is_empty() && kind != target_kind) {
                return None;
            }
            if wanted.is_none_or(|value| value == occurrence) {
                return Some(part_index);
            }
            occurrence += 1;
            None
        })
    } else {
        if expected_context.is_empty() || actual_context.as_deref() != Some(expected_context) {
            return false;
        }
        index(item, "partIndex")
            .filter(|part_index| parts.get(*part_index).and_then(part_fingerprint).is_some())
            .or_else(|| {
                parts
                    .iter()
                    .rposition(|part| part_fingerprint(part).is_some())
            })
    };
    let Some(target) = target else { return false };
    let already_native = parts[target]
        .get("thoughtSignature")
        .and_then(Value::as_str)
        .is_some_and(has_native_signature);
    if already_native {
        return false;
    }
    remove_signature_from_other_parts(parts, target, signature);
    let Some(part) = parts[target].as_object_mut() else {
        return false;
    };
    part.remove("thought_signature");
    if let Some(extra) = part.get_mut("extra_content").and_then(Value::as_object_mut) {
        extra.remove("google");
        if extra.is_empty() {
            part.remove("extra_content");
        }
    }
    part.insert(
        "thoughtSignature".to_owned(),
        Value::String(signature.to_owned()),
    );
    true
}

fn apply_function_call(root: &mut Value, item: &Value) -> bool {
    let Some(name) = item
        .get("name")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|v| !v.is_empty())
    else {
        return false;
    };
    let Some(args) = item.get("args") else {
        return false;
    };
    let native_id = item
        .get("call_id")
        .and_then(Value::as_str)
        .map(str::trim)
        .unwrap_or("");
    let signature = item
        .get("thoughtSignature")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|v| has_native_signature(v));
    let Some(contents) = root
        .pointer_mut("/request/contents")
        .and_then(Value::as_array_mut)
    else {
        return false;
    };

    // Reusing the provider ID with different semantics poisons the whole item;
    // do not fall through to another call that happens to have matching args.
    if !native_id.is_empty()
        && contents.iter().flat_map(content_parts).any(|part| {
            part.pointer("/functionCall/id").and_then(Value::as_str) == Some(native_id)
                && !function_semantics_match(part.get("functionCall"), name, args)
        })
    {
        return false;
    }

    let mut candidates = Vec::new();
    for (ci, content) in contents.iter().enumerate() {
        if content
            .get("role")
            .and_then(Value::as_str)
            .is_none_or(|role| !role.trim().eq_ignore_ascii_case("model"))
        {
            continue;
        }
        for (pi, part) in content_parts(content).enumerate() {
            if function_semantics_match(part.get("functionCall"), name, args) {
                candidates.push((ci, pi));
            }
        }
    }
    let exact = index(item, "contentIndex").zip(index(item, "partIndex"));
    let selected = exact
        .filter(|pair| candidates.contains(pair))
        .or_else(|| (candidates.len() == 1).then(|| candidates[0]));
    let Some((ci, pi)) = selected else {
        return false;
    };
    let current_id = contents[ci]["parts"][pi]["functionCall"]
        .get("id")
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim()
        .to_owned();
    let may_restore_id = current_id == native_id
        || (!native_id.is_empty() && current_id == provenance_id(native_id, name, args));
    let mut changed = false;
    if may_restore_id && !native_id.is_empty() && current_id != native_id {
        contents[ci]["parts"][pi]["functionCall"]["id"] = Value::String(native_id.to_owned());
        for content in contents.iter_mut() {
            for part in content
                .get_mut("parts")
                .and_then(Value::as_array_mut)
                .into_iter()
                .flatten()
            {
                let Some(response) = part.get_mut("functionResponse") else {
                    continue;
                };
                if response.get("id").and_then(Value::as_str) == Some(current_id.as_str()) {
                    response["id"] = Value::String(native_id.to_owned());
                    response["name"] = Value::String(name.to_owned());
                }
            }
        }
        changed = true;
    }
    if let Some(signature) = signature {
        let parts = contents[ci]
            .get_mut("parts")
            .and_then(Value::as_array_mut)
            .expect("selected part came from array");
        if !parts[pi]
            .get("thoughtSignature")
            .and_then(Value::as_str)
            .is_some_and(has_native_signature)
        {
            remove_signature_from_other_parts(parts, pi, signature);
            parts[pi]["thoughtSignature"] = Value::String(signature.to_owned());
            changed = true;
        }
    }
    changed
}

fn content_parts(content: &Value) -> impl Iterator<Item = &Value> {
    content
        .get("parts")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
}

fn function_semantics_match(call: Option<&Value>, name: &str, args: &Value) -> bool {
    call.is_some_and(|call| {
        call.get("name")
            .and_then(Value::as_str)
            .is_some_and(|value| value.trim() == name)
            && call
                .get("args")
                .is_some_and(|value| canonical(value) == canonical(args))
    })
}

fn part_fingerprint(part: &Value) -> Option<(&'static str, String)> {
    if part.get("functionCall").is_some() || part.get("functionResponse").is_some() {
        return None;
    }
    let text = part.get("text").and_then(Value::as_str)?;
    let kind = if part.get("thought").and_then(Value::as_bool) == Some(true) {
        "thought"
    } else {
        "text"
    };
    let mut digest = Sha256::new();
    digest.update(kind.as_bytes());
    digest.update(b"\0");
    digest.update(text.as_bytes());
    Some((kind, format!("{:x}", digest.finalize())))
}

fn context_fingerprint(root: &Value, before_content: usize) -> String {
    let Some(request) = root.get("request") else {
        return String::new();
    };
    let mut digest = Sha256::new();
    let mut wrote = false;
    for key in ["systemInstruction", "tools", "toolConfig"] {
        if let Some(value) = request.get(key) {
            digest.update(format!("request.{key}").as_bytes());
            digest.update(b"\0");
            digest.update(canonical(value));
            digest.update(b"\0");
            wrote = true;
        }
    }
    if let Some(contents) = request.get("contents").and_then(Value::as_array) {
        if before_content > contents.len() {
            return String::new();
        }
        for content in contents.iter().take(before_content) {
            digest.update(
                content
                    .get("role")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .trim()
                    .to_ascii_lowercase(),
            );
            digest.update(b"\0");
            wrote = true;
            for part in content_parts(content) {
                let mut clean = part.clone();
                if let Some(object) = clean.as_object_mut() {
                    object.remove("thoughtSignature");
                    object.remove("thought_signature");
                    if let Some(extra) = object
                        .get_mut("extra_content")
                        .and_then(Value::as_object_mut)
                    {
                        if let Some(google) = extra.get_mut("google").and_then(Value::as_object_mut)
                        {
                            google.remove("thought_signature");
                            if google.is_empty() {
                                extra.remove("google");
                            }
                        }
                        if extra.is_empty() {
                            object.remove("extra_content");
                        }
                    }
                }
                digest.update(canonical(&clean));
                digest.update(b"\0");
                wrote = true;
            }
        }
    }
    if wrote {
        format!("{:x}", digest.finalize())
    } else {
        String::new()
    }
}

fn canonical(value: &Value) -> Vec<u8> {
    fn sorted(value: &Value) -> Value {
        match value {
            Value::Object(map) => {
                let mut keys = map.keys().collect::<Vec<_>>();
                keys.sort();
                Value::Object(
                    keys.into_iter()
                        .map(|key| (key.clone(), sorted(&map[key])))
                        .collect(),
                )
            }
            Value::Array(values) => Value::Array(values.iter().map(sorted).collect()),
            _ => value.clone(),
        }
    }
    serde_json::to_vec(&sorted(value)).unwrap_or_default()
}

fn provenance_id(call_id: &str, name: &str, args: &Value) -> String {
    let mut digest = Sha256::new();
    digest.update(call_id.trim().as_bytes());
    digest.update(b"\0");
    digest.update(name.trim().as_bytes());
    digest.update(b"\0");
    digest.update(canonical(args));
    format!(
        "{CLAUDE_PROVENANCE_PREFIX}{}",
        &format!("{:x}", digest.finalize())[..32]
    )
}

fn remove_signature_from_other_parts(parts: &mut [Value], keep: usize, signature: &str) {
    for (index, part) in parts.iter_mut().enumerate() {
        if index == keep {
            continue;
        }
        if part.get("thoughtSignature").and_then(Value::as_str) == Some(signature) {
            part.as_object_mut()
                .map(|object| object.remove("thoughtSignature"));
        }
        if part.get("thought_signature").and_then(Value::as_str) == Some(signature) {
            part.as_object_mut()
                .map(|object| object.remove("thought_signature"));
        }
    }
}

fn has_native_signature(value: &str) -> bool {
    let value = value.trim();
    !value.is_empty() && value != BYPASS_SIGNATURE
}

fn index(value: &Value, key: &str) -> Option<usize> {
    value
        .get(key)
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AntigravityReplayApplyError {
    InvalidPayload,
}

impl std::fmt::Display for AntigravityReplayApplyError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("Antigravity replay payload is invalid")
    }
}
impl std::error::Error for AntigravityReplayApplyError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn fingerprint(text: &str) -> String {
        part_fingerprint(&serde_json::json!({"text":text}))
            .unwrap()
            .1
    }

    #[test]
    fn fingerprint_replays_across_context_drift_but_rejects_edited_part() {
        let item = serde_json::to_vec(&serde_json::json!({"type":"thought_signature","contentIndex":1,"partIndex":0,"thoughtSignature":"fingerprinted-signature-123456","targetKind":"text","targetHash":fingerprint("same answer")})).unwrap();
        let payload = br#"{"request":{"contents":[{"role":"user","parts":[{"text":"new context"}]},{"role":"model","parts":[{"text":"same answer"}]}]}}"#;
        let (out, applied) =
            apply_antigravity_reasoning_replay_items(payload, std::slice::from_ref(&item)).unwrap();
        let value: Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(applied, 1);
        assert_eq!(
            value
                .pointer("/request/contents/1/parts/0/thoughtSignature")
                .unwrap(),
            "fingerprinted-signature-123456"
        );
        let edited = payload.replace_ascii(b"same answer", b"edit answer");
        let (_, applied) = apply_antigravity_reasoning_replay_items(&edited, &[item]).unwrap();
        assert_eq!(applied, 0);
    }

    #[test]
    fn moves_client_carrier_to_fingerprinted_native_part() {
        let signature = "client-carried-signature-123456";
        let item = serde_json::to_vec(&serde_json::json!({"type":"thought_signature","contentIndex":1,"partIndex":1,"thoughtSignature":signature,"targetKind":"text","targetHash":fingerprint("visible answer")})).unwrap();
        let payload = format!(
            r#"{{"request":{{"contents":[{{"role":"user","parts":[{{"text":"turn"}}]}},{{"role":"model","parts":[{{"text":"hidden","thought":true,"thoughtSignature":"{signature}"}},{{"text":"visible answer"}}]}}]}}}}"#
        );
        let (out, applied) =
            apply_antigravity_reasoning_replay_items(payload.as_bytes(), &[item]).unwrap();
        let value: Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(applied, 1);
        assert!(value
            .pointer("/request/contents/1/parts/0/thoughtSignature")
            .is_none());
        assert_eq!(
            value
                .pointer("/request/contents/1/parts/1/thoughtSignature")
                .unwrap(),
            signature
        );
    }

    #[test]
    fn restores_bound_provenance_id_and_matching_response() {
        let args = serde_json::json!({"command":"same"});
        let native = "native-call-id";
        let synthetic = provenance_id(native, "run_command", &args);
        let item = serde_json::to_vec(&serde_json::json!({"type":"function_call_part","contentIndex":1,"partIndex":0,"call_id":native,"name":"run_command","args":args,"thoughtSignature":"rewritten-id-signature-123456"})).unwrap();
        let payload = format!(
            r#"{{"request":{{"contents":[{{"role":"user","parts":[{{"text":"run"}}]}},{{"role":"model","parts":[{{"functionCall":{{"id":"{synthetic}","name":"run_command","args":{{"command":"same"}}}}}}]}},{{"role":"function","parts":[{{"functionResponse":{{"id":"{synthetic}","name":"run_command","response":{{"ok":true}}}}}}]}}]}}}}"#
        );
        let (out, applied) =
            apply_antigravity_reasoning_replay_items(payload.as_bytes(), &[item]).unwrap();
        let value: Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(applied, 1);
        assert_eq!(
            value
                .pointer("/request/contents/1/parts/0/functionCall/id")
                .unwrap(),
            native
        );
        assert_eq!(
            value
                .pointer("/request/contents/2/parts/0/functionResponse/id")
                .unwrap(),
            native
        );
    }

    #[test]
    fn reused_id_with_changed_args_fails_closed_without_fallthrough() {
        let item = serde_json::to_vec(&serde_json::json!({"type":"function_call_part","contentIndex":1,"partIndex":0,"call_id":"reused-id","name":"run_command","args":{"command":"old"},"thoughtSignature":"stale-signature-123456"})).unwrap();
        let payload = br#"{"request":{"contents":[{"role":"model","parts":[{"functionCall":{"id":"reused-id","name":"run_command","args":{"command":"new"}}},{"functionCall":{"id":"other","name":"run_command","args":{"command":"old"}}}]}]}}"#;
        let (out, applied) = apply_antigravity_reasoning_replay_items(payload, &[item]).unwrap();
        assert_eq!(applied, 0);
        assert_eq!(
            serde_json::from_slice::<Value>(&out)
                .unwrap()
                .pointer("/request/contents/0/parts/1/thoughtSignature"),
            None
        );
    }

    #[test]
    fn replay_never_mutates_client_authored_parts() {
        let thought = serde_json::to_vec(&serde_json::json!({"type":"thought_signature","contentIndex":0,"partIndex":0,"thoughtSignature":"server-only-signature-123456","targetKind":"text","targetHash":fingerprint("client text")})).unwrap();
        let call = serde_json::to_vec(&serde_json::json!({"type":"function_call_part","contentIndex":1,"partIndex":0,"call_id":"native-call","name":"run_command","args":{"command":"same"},"thoughtSignature":"server-only-signature-123456"})).unwrap();
        let payload = br#"{"request":{"contents":[{"role":"user","parts":[{"text":"client text"}]},{"role":"user","parts":[{"functionCall":{"id":"native-call","name":"run_command","args":{"command":"same"}}}]}]}}"#;
        let (out, applied) =
            apply_antigravity_reasoning_replay_items(payload, &[thought, call]).unwrap();
        assert_eq!(applied, 0);
        assert_eq!(
            serde_json::from_slice::<Value>(&out).unwrap(),
            serde_json::from_slice::<Value>(payload).unwrap()
        );
    }

    #[test]
    fn prepared_lane_applies_parent_and_publishes_terminal_extension() {
        let cache = Arc::new(AntigravityReasoningReplayCache::new());
        let (_, empty, _) = cache.read("gemini-3", "session:one", 1).unwrap();
        let parent = serde_json::to_vec(&serde_json::json!({
            "type":"thought_signature", "thoughtSignature":"parent-signature-123456",
            "contentIndex":1, "partIndex":0, "targetKind":"text",
            "targetHash":fingerprint("old answer"), "targetOccurrence":0
        }))
        .unwrap();
        assert!(cache
            .replace_if_unchanged("gemini-3", "session:one", &empty, &[parent], 2)
            .unwrap());
        let request = br#"{"request":{"contents":[{"role":"user","parts":[{"text":"ask"}]},{"role":"model","parts":[{"text":"old answer"}]}]}}"#;
        let (prepared, mut accumulator) = prepare_antigravity_reasoning_replay(
            cache.clone(),
            "gemini-3",
            "session:one",
            request,
            3,
        )
        .unwrap();
        assert_eq!(
            serde_json::from_slice::<Value>(&prepared)
                .unwrap()
                .pointer("/request/contents/1/parts/0/thoughtSignature"),
            Some(&Value::String("parent-signature-123456".to_owned()))
        );
        accumulator.observe_response_payload(br#"{"response":{"candidates":[{"content":{"parts":[{"text":"new answer","thoughtSignature":"child-signature-123456"}]},"finishReason":"STOP"}]}}"#);
        assert_eq!(
            accumulator.commit(4).unwrap(),
            AntigravityReplayCommitOutcome::Published
        );
        let (items, _, found) = cache.read("gemini-3", "session:one", 5).unwrap();
        assert!(found);
        assert_eq!(items.len(), 2);
        assert_eq!(
            serde_json::from_slice::<Value>(&items[1]).unwrap()["targetHash"],
            fingerprint("new answer")
        );
    }

    #[test]
    fn incomplete_stream_never_mutates_replay_lane() {
        let cache = Arc::new(AntigravityReasoningReplayCache::new());
        let request = br#"{"request":{"contents":[{"role":"user","parts":[{"text":"ask"}]}]}}"#;
        let (_, mut accumulator) = prepare_antigravity_reasoning_replay(
            cache.clone(),
            "gemini-3",
            "session:partial",
            request,
            1,
        )
        .unwrap();
        accumulator.observe_response_payload(br#"{"response":{"candidates":[{"content":{"parts":[{"functionCall":{"id":"partial-call","name":"run","args":{"x":1}}}]}}]}}"#);
        assert_eq!(
            accumulator.commit(2).unwrap(),
            AntigravityReplayCommitOutcome::NotTerminal
        );
        let (_, _, found) = cache.read("gemini-3", "session:partial", 3).unwrap();
        assert!(!found);
    }

    #[test]
    fn stale_terminal_sibling_cannot_overwrite_winner() {
        let cache = Arc::new(AntigravityReasoningReplayCache::new());
        let request = br#"{"request":{"contents":[{"role":"user","parts":[{"text":"ask"}]}]}}"#;
        let (_, mut winner) = prepare_antigravity_reasoning_replay(
            cache.clone(),
            "gemini-3",
            "session:race",
            request,
            1,
        )
        .unwrap();
        let (_, mut stale) = prepare_antigravity_reasoning_replay(
            cache.clone(),
            "gemini-3",
            "session:race",
            request,
            1,
        )
        .unwrap();
        winner.observe_response_payload(br#"{"response":{"candidates":[{"content":{"parts":[{"text":"winner","thoughtSignature":"winner-signature-123456"}]},"finishReason":"STOP"}]}}"#);
        stale.observe_response_payload(br#"{"response":{"candidates":[{"content":{"parts":[{"text":"sibling","thoughtSignature":"sibling-signature-123456"}]},"finishReason":"STOP"}]}}"#);
        assert_eq!(
            winner.commit(2).unwrap(),
            AntigravityReplayCommitOutcome::Published
        );
        assert_eq!(
            stale.commit(3).unwrap(),
            AntigravityReplayCommitOutcome::RejectedStale
        );
        let (items, _, _) = cache.read("gemini-3", "session:race", 4).unwrap();
        assert_eq!(items.len(), 1);
        assert!(String::from_utf8_lossy(&items[0]).contains("winner-signature"));
    }

    #[test]
    fn detached_terminal_signature_binds_to_last_function_call() {
        let cache = Arc::new(AntigravityReasoningReplayCache::new());
        let request = br#"{"request":{"contents":[{"role":"user","parts":[{"text":"run"}]}]}}"#;
        let (_, mut accumulator) = prepare_antigravity_reasoning_replay(
            cache.clone(),
            "gemini-3",
            "session:function",
            request,
            1,
        )
        .unwrap();
        accumulator.observe_sse_line(br#"data: {"response":{"candidates":[{"content":{"parts":[{"functionCall":{"id":"native-1","name":"run","args":{"x":1}}}]}}]}}"#);
        accumulator.observe_sse_line(br#"data: {"response":{"candidates":[{"content":{"parts":[{"thoughtSignature":"detached-signature-123456"}]},"finishReason":"STOP"}]}}"#);
        assert_eq!(
            accumulator.commit(2).unwrap(),
            AntigravityReplayCommitOutcome::Published
        );
        let (items, _, _) = cache.read("gemini-3", "session:function", 3).unwrap();
        let item: Value = serde_json::from_slice(&items[0]).unwrap();
        assert_eq!(item["type"], "function_call_part");
        assert_eq!(item["thoughtSignature"], "detached-signature-123456");
    }

    #[test]
    fn terminal_overflow_invalidates_exact_snapshot() {
        let cache = Arc::new(AntigravityReasoningReplayCache::new());
        let request = br#"{"request":{"contents":[{"role":"user","parts":[{"text":"ask"}]}]}}"#;
        let (_, mut accumulator) = prepare_antigravity_reasoning_replay(
            cache.clone(),
            "gemini-3",
            "session:overflow",
            request,
            1,
        )
        .unwrap();
        accumulator.overflow = true;
        accumulator
            .observe_response_payload(br#"{"response":{"candidates":[{"finishReason":"STOP"}]}}"#);
        assert_eq!(
            accumulator.commit(2).unwrap(),
            AntigravityReplayCommitOutcome::Invalidated
        );
        let (_, _, found) = cache.read("gemini-3", "session:overflow", 3).unwrap();
        assert!(!found);
    }

    trait ReplaceAscii {
        fn replace_ascii(&self, from: &[u8], to: &[u8]) -> Vec<u8>;
    }
    impl ReplaceAscii for [u8] {
        fn replace_ascii(&self, from: &[u8], to: &[u8]) -> Vec<u8> {
            let at = self
                .windows(from.len())
                .position(|window| window == from)
                .unwrap();
            let mut out = self.to_vec();
            out.splice(at..at + from.len(), to.iter().copied());
            out
        }
    }
}
