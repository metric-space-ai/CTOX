// ref: internal/runtime/executor/codex_executor_reasoning.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::{Mutex, MutexGuard};

use serde_json::Value;
use sha2::{Digest, Sha256};

const DEFAULT_MAX_SESSIONS: usize = 256;
const DEFAULT_MAX_ITEMS_PER_SESSION: usize = 32;
const CODEX_CALL_ID_LIMIT: usize = 64;
const CODEX_REPLAY_TURN_TYPE: &str = "cpa_codex_replay_turn";

#[derive(Debug, Clone)]
pub struct CodexReasoningReplayScope {
    provider: String,
    session_key: String,
    request_fingerprint: String,
}

impl CodexReasoningReplayScope {
    pub fn new(provider: &str, session_key: &str) -> Option<Self> {
        let provider = provider.trim();
        let session_key = session_key.trim();
        if provider.is_empty() || session_key.is_empty() {
            return None;
        }
        Some(Self {
            provider: provider.to_ascii_lowercase(),
            session_key: session_key.to_owned(),
            request_fingerprint: String::new(),
        })
    }

    pub fn from_request(provider: &str, session_key: &str, body: &[u8]) -> Option<Self> {
        let mut scope = Self::new(provider, session_key)?;
        let value = serde_json::from_slice::<Value>(body).ok()?;
        let input = value.get("input").and_then(Value::as_array)?;
        scope.request_fingerprint = input_prefix_fingerprint(input, input.len());
        Some(scope)
    }
}

impl PartialEq for CodexReasoningReplayScope {
    fn eq(&self, other: &Self) -> bool {
        self.provider == other.provider && self.session_key == other.session_key
    }
}

impl Eq for CodexReasoningReplayScope {}

impl Hash for CodexReasoningReplayScope {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.provider.hash(state);
        self.session_key.hash(state);
    }
}

#[derive(Clone)]
struct ReplayEntry {
    items: Vec<Value>,
}

/// Host-owned bounded reasoning replay cache. There is deliberately no global
/// cache: callers choose its lifetime and therefore its privacy boundary.
pub struct CodexReasoningReplayCache {
    state: Mutex<ReplayState>,
    max_sessions: usize,
    max_items_per_session: usize,
}

#[derive(Default)]
struct ReplayState {
    entries: HashMap<CodexReasoningReplayScope, ReplayEntry>,
    order: VecDeque<CodexReasoningReplayScope>,
}

impl CodexReasoningReplayCache {
    pub fn new(max_sessions: usize, max_items_per_session: usize) -> Self {
        Self {
            state: Mutex::new(ReplayState::default()),
            max_sessions: max_sessions.max(1),
            max_items_per_session: max_items_per_session.max(1),
        }
    }

    pub fn apply(&self, scope: &CodexReasoningReplayScope, body: &[u8]) -> Vec<u8> {
        let items = lock_recover(&self.state)
            .entries
            .get(scope)
            .map(|entry| entry.items.clone())
            .unwrap_or_default();
        insert_codex_reasoning_replay_turns(body, &items).unwrap_or_else(|| body.to_vec())
    }

    pub fn commit_completed(&self, scope: CodexReasoningReplayScope, completed: &[u8]) -> bool {
        let Ok(value) = serde_json::from_slice::<Value>(completed) else {
            return false;
        };
        let Some(output) = value
            .get("output")
            .or_else(|| value.pointer("/response/output"))
            .and_then(Value::as_array)
        else {
            return false;
        };
        let mut replay_items = Vec::new();
        let mut call_ids = Vec::new();
        let mut assistant_fingerprint = String::new();
        for item in output {
            match item.get("type").and_then(Value::as_str) {
                Some("reasoning") => replay_items.push(item.clone()),
                Some("function_call" | "custom_tool_call") => {
                    if let Some(call_id) = string_field(item, "call_id") {
                        call_ids.push(call_id.to_owned());
                    }
                    replay_items.push(item.clone());
                }
                Some("message") => {
                    if let Some(fingerprint) = assistant_message_fingerprint(item) {
                        assistant_fingerprint = fingerprint;
                    }
                }
                _ => {}
            }
        }
        if replay_items.is_empty() {
            return false;
        }
        let mut hasher = Sha256::new();
        hasher.update(scope.request_fingerprint.as_bytes());
        hasher.update(b"\0assistant\0");
        hasher.update(assistant_fingerprint.as_bytes());
        for call_id in &call_ids {
            hasher.update(b"\0call\0");
            hasher.update(call_id.as_bytes());
        }
        for item in &replay_items {
            hasher.update(b"\0item\0");
            if let Ok(encoded) = serde_json::to_vec(item) {
                hasher.update(encoded);
            }
        }
        let mut marker = serde_json::json!({
            "type": CODEX_REPLAY_TURN_TYPE,
            "id": hex_digest(hasher.finalize().as_slice()),
        });
        if !assistant_fingerprint.is_empty() {
            marker["assistant_fingerprint"] = Value::String(assistant_fingerprint);
        }
        if !scope.request_fingerprint.is_empty() {
            marker["request_fingerprint"] = Value::String(scope.request_fingerprint.clone());
        }
        if !call_ids.is_empty() {
            marker["call_ids"] = serde_json::to_value(call_ids).unwrap_or(Value::Null);
        }
        let mut turn = Vec::with_capacity(replay_items.len() + 1);
        turn.push(marker);
        turn.extend(replay_items);
        let mut state = lock_recover(&self.state);
        state.order.retain(|candidate| candidate != &scope);
        state.order.push_back(scope.clone());
        let entry = state
            .entries
            .entry(scope)
            .or_insert_with(|| ReplayEntry { items: Vec::new() });
        if entry.items.first().and_then(item_type) != Some(CODEX_REPLAY_TURN_TYPE) {
            entry.items.clear();
        }
        let marker_id = turn[0].get("id").and_then(Value::as_str);
        if marker_id.is_some_and(|id| {
            entry.items.iter().any(|item| {
                item_type(item) == Some(CODEX_REPLAY_TURN_TYPE)
                    && item.get("id").and_then(Value::as_str) == Some(id)
            })
        }) {
            return true;
        }
        entry.items.extend(turn);
        trim_complete_turns(&mut entry.items, self.max_items_per_session);
        while state.entries.len() > self.max_sessions {
            if let Some(evicted) = state.order.pop_front() {
                state.entries.remove(&evicted);
            }
        }
        true
    }

    pub fn clear(&self, scope: &CodexReasoningReplayScope) -> bool {
        let mut state = lock_recover(&self.state);
        state.order.retain(|candidate| candidate != scope);
        state.entries.remove(scope).is_some()
    }

    pub fn clear_on_invalid_signature(
        &self,
        scope: &CodexReasoningReplayScope,
        status: u16,
        error_body: &[u8],
    ) -> bool {
        if status != 400 {
            return false;
        }
        let Ok(value) = serde_json::from_slice::<Value>(error_body) else {
            return false;
        };
        let code = value
            .pointer("/error/code")
            .or_else(|| value.get("code"))
            .and_then(Value::as_str)
            .unwrap_or_default();
        if matches!(
            code,
            "invalid_encrypted_content" | "invalid_reasoning_signature" | "invalid_signature"
        ) {
            self.clear(scope)
        } else {
            false
        }
    }

    pub fn len(&self) -> usize {
        lock_recover(&self.state).entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Default for CodexReasoningReplayCache {
    fn default() -> Self {
        Self::new(DEFAULT_MAX_SESSIONS, DEFAULT_MAX_ITEMS_PER_SESSION)
    }
}

impl fmt::Debug for CodexReasoningReplayCache {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CodexReasoningReplayCache")
            .field("entry_count", &self.len())
            .field("max_sessions", &self.max_sessions)
            .field("max_items_per_session", &self.max_items_per_session)
            .finish()
    }
}

pub fn codex_reasoning_replay_session_key(
    body: &[u8],
    header_session: Option<&str>,
) -> Option<String> {
    let value = serde_json::from_slice::<Value>(body).ok();
    for candidate in [
        value
            .as_ref()
            .and_then(|value| value.get("prompt_cache_key"))
            .and_then(Value::as_str),
        value
            .as_ref()
            .and_then(|value| value.pointer("/metadata/session_id"))
            .and_then(Value::as_str),
        header_session,
    ] {
        if let Some(candidate) = candidate.map(str::trim).filter(|value| !value.is_empty()) {
            return Some(candidate.to_owned());
        }
    }
    None
}

pub fn insert_codex_reasoning_replay_items(body: &[u8], replay_items: &[Value]) -> Option<Vec<u8>> {
    if replay_items.is_empty() {
        return None;
    }
    let mut value = serde_json::from_slice::<Value>(body).ok()?;
    let input = value.get_mut("input")?.as_array_mut()?;
    let mut insert = filter_replay_items_for_input(input, replay_items);
    if insert.is_empty() {
        return None;
    }
    align_call_ids(input, &mut insert);
    let index = replay_insert_index(input, &insert);
    input.splice(index..index, insert);
    serde_json::to_vec(&value).ok()
}

fn insert_codex_reasoning_replay_turns(body: &[u8], replay_items: &[Value]) -> Option<Vec<u8>> {
    let mut value = serde_json::from_slice::<Value>(body).ok()?;
    let input = value.get_mut("input")?.as_array_mut()?;
    let turns = split_replay_turns(replay_items);
    let mut insertions: HashMap<usize, Vec<Value>> = HashMap::new();
    let mut used = HashSet::new();
    let mut fallback_end = input.len().checked_sub(1);
    for turn in turns.into_iter().rev() {
        if turn.items.is_empty() {
            continue;
        }
        if !turn.marked {
            let mut items = filter_replay_items_for_input(input, &turn.items);
            if items.is_empty() {
                continue;
            }
            let index = replay_insert_index(input, &items);
            align_call_ids(input, &mut items);
            insertions.entry(index).or_default().splice(0..0, items);
            continue;
        }
        let Some(anchor) = replay_turn_anchor(input, &turn, fallback_end, &used) else {
            continue;
        };
        used.insert(anchor);
        if turn.request_fingerprint.is_empty() {
            fallback_end = anchor.checked_sub(1);
        }
        let mut items = filter_replay_turn_items(input, &turn.items);
        if items.is_empty() {
            continue;
        }
        align_call_ids(input, &mut items);
        insertions.entry(anchor).or_default().splice(0..0, items);
    }
    if insertions.is_empty() {
        return None;
    }
    let mut merged = Vec::with_capacity(input.len() + replay_items.len());
    for (index, item) in input.iter().enumerate() {
        if let Some(items) = insertions.remove(&index) {
            merged.extend(items);
        }
        merged.push(item.clone());
    }
    if let Some(items) = insertions.remove(&input.len()) {
        merged.extend(items);
    }
    *input = merged;
    serde_json::to_vec(&value).ok()
}

#[derive(Default)]
struct ReplayTurn {
    marked: bool,
    assistant_fingerprint: String,
    request_fingerprint: String,
    call_ids: Vec<String>,
    items: Vec<Value>,
}

fn split_replay_turns(items: &[Value]) -> Vec<ReplayTurn> {
    let mut turns = Vec::new();
    let mut current = ReplayTurn::default();
    for item in items {
        if item_type(item) == Some(CODEX_REPLAY_TURN_TYPE) {
            if !current.items.is_empty() {
                turns.push(current);
            }
            current = ReplayTurn {
                marked: true,
                assistant_fingerprint: string_field(item, "assistant_fingerprint")
                    .unwrap_or_default()
                    .to_owned(),
                request_fingerprint: string_field(item, "request_fingerprint")
                    .unwrap_or_default()
                    .to_owned(),
                call_ids: item
                    .get("call_ids")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                    .filter_map(Value::as_str)
                    .map(str::trim)
                    .filter(|value| !value.is_empty())
                    .map(ToOwned::to_owned)
                    .collect(),
                items: Vec::new(),
            };
        } else {
            current.items.push(item.clone());
        }
    }
    if !current.items.is_empty() {
        turns.push(current);
    }
    turns
}

fn replay_turn_anchor(
    input: &[Value],
    turn: &ReplayTurn,
    fallback_end: Option<usize>,
    used: &HashSet<usize>,
) -> Option<usize> {
    let search_end = if turn.request_fingerprint.is_empty() {
        fallback_end?
    } else {
        input.len().checked_sub(1)?
    }
    .min(input.len().saturating_sub(1));
    let matches_prefix = |index: usize| {
        turn.request_fingerprint.is_empty()
            || input_prefix_fingerprint(input, index) == turn.request_fingerprint
    };
    if !turn.call_ids.is_empty() {
        let expected = turn
            .call_ids
            .iter()
            .flat_map(|id| comparable_call_ids(id))
            .collect::<HashSet<_>>();
        for index in (0..=search_end).rev() {
            if used.contains(&index) || !matches_prefix(index) || !is_tool_item(&input[index]) {
                continue;
            }
            if comparable_call_ids(string_field(&input[index], "call_id").unwrap_or_default())
                .into_iter()
                .any(|id| expected.contains(&id))
            {
                return Some(index);
            }
        }
    }
    if !turn.assistant_fingerprint.is_empty() {
        for index in (0..=search_end).rev() {
            if !used.contains(&index)
                && matches_prefix(index)
                && assistant_message_fingerprint(&input[index]).as_deref()
                    == Some(turn.assistant_fingerprint.as_str())
            {
                return Some(index);
            }
        }
    }
    if turn.call_ids.is_empty() && turn.assistant_fingerprint.is_empty() {
        return Some(replay_insert_index(input, &turn.items));
    }
    None
}

fn filter_replay_turn_items(input: &[Value], items: &[Value]) -> Vec<Value> {
    let existing_reasoning = input
        .iter()
        .filter(|item| item_type(item) == Some("reasoning"))
        .filter_map(|item| string_field(item, "encrypted_content"))
        .map(ToOwned::to_owned)
        .collect::<HashSet<_>>();
    filter_replay_items(input, items, |item| {
        item_type(item) != Some("reasoning")
            || string_field(item, "encrypted_content")
                .is_some_and(|value| !existing_reasoning.contains(value))
    })
}

fn filter_replay_items_for_input(input: &[Value], items: &[Value]) -> Vec<Value> {
    let has_reasoning = input.iter().any(|item| {
        item_type(item) == Some("reasoning") && string_field(item, "encrypted_content").is_some()
    });
    filter_replay_items(input, items, |item| {
        item_type(item) != Some("reasoning") || !has_reasoning
    })
}

fn filter_replay_items(
    input: &[Value],
    items: &[Value],
    keep_reasoning: impl Fn(&Value) -> bool,
) -> Vec<Value> {
    let mut existing_calls = input
        .iter()
        .flat_map(tool_call_keys)
        .collect::<HashSet<_>>();
    let existing_outputs = input
        .iter()
        .filter(|item| is_tool_output(item))
        .filter_map(|item| string_field(item, "call_id"))
        .flat_map(comparable_call_ids)
        .collect::<HashSet<_>>();
    let mut filtered = Vec::new();
    for item in items {
        match item_type(item) {
            Some("reasoning") if keep_reasoning(item) => filtered.push(item.clone()),
            Some("function_call" | "custom_tool_call") => {
                let keys = tool_call_keys(item);
                let has_output = string_field(item, "call_id")
                    .map(comparable_call_ids)
                    .is_some_and(|ids| ids.iter().any(|id| existing_outputs.contains(id)));
                if !keys.is_empty()
                    && !keys.iter().any(|key| existing_calls.contains(key))
                    && has_output
                {
                    existing_calls.extend(keys);
                    filtered.push(item.clone());
                }
            }
            _ => {}
        }
    }
    filtered
}

fn replay_insert_index(input: &[Value], replay: &[Value]) -> usize {
    let replay_calls = replay
        .iter()
        .filter(|item| is_tool_call(item))
        .filter_map(|item| string_field(item, "call_id"))
        .flat_map(comparable_call_ids)
        .collect::<HashSet<_>>();
    if !replay_calls.is_empty() {
        for (index, item) in input.iter().enumerate() {
            if is_tool_output(item) {
                let call_id = string_field(item, "call_id").unwrap_or_default();
                if call_id.is_empty() || replay_calls.contains(call_id) {
                    return index;
                }
            }
        }
    }
    if let Some(index) = input
        .iter()
        .rposition(|item| message_role(item) == Some("assistant"))
    {
        return index;
    }
    for (index, item) in input.iter().enumerate() {
        if !matches!(message_role(item), Some("developer" | "system")) {
            return index;
        }
    }
    input.len()
}

fn align_call_ids(input: &[Value], replay: &mut [Value]) {
    let mappings = input
        .iter()
        .filter(|item| is_tool_output(item))
        .filter_map(|item| string_field(item, "call_id"))
        .flat_map(|call_id| {
            comparable_call_ids(call_id)
                .into_iter()
                .map(move |candidate| (candidate, call_id.to_owned()))
        })
        .collect::<HashMap<_, _>>();
    for item in replay {
        if !is_tool_call(item) {
            continue;
        }
        let Some(object) = item.as_object_mut() else {
            continue;
        };
        let call_id = object
            .get("call_id")
            .and_then(Value::as_str)
            .unwrap_or_default();
        if let Some(replacement) = comparable_call_ids(call_id)
            .iter()
            .find_map(|candidate| mappings.get(candidate))
        {
            object.insert("call_id".to_owned(), Value::String(replacement.clone()));
        }
    }
}

fn item_type(item: &Value) -> Option<&str> {
    string_field(item, "type")
}

fn string_field<'a>(item: &'a Value, key: &str) -> Option<&'a str> {
    item.get(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
}

fn is_tool_call(item: &Value) -> bool {
    matches!(item_type(item), Some("function_call" | "custom_tool_call"))
}

fn is_tool_output(item: &Value) -> bool {
    matches!(
        item_type(item),
        Some("function_call_output" | "custom_tool_call_output")
    )
}

fn is_tool_item(item: &Value) -> bool {
    is_tool_call(item) || is_tool_output(item)
}

fn message_role(item: &Value) -> Option<&str> {
    let role = string_field(item, "role")?;
    if item_type(item).is_none_or(|kind| kind == "message") {
        Some(role)
    } else {
        None
    }
}

fn tool_call_keys(item: &Value) -> Vec<String> {
    if !is_tool_call(item) {
        return Vec::new();
    }
    let Some(call_id) = string_field(item, "call_id") else {
        return Vec::new();
    };
    let kind = item_type(item).unwrap_or_default();
    comparable_call_ids(call_id)
        .into_iter()
        .map(|id| format!("{kind}:{id}"))
        .collect()
}

fn comparable_call_ids(call_id: &str) -> Vec<String> {
    let call_id = call_id.trim();
    if call_id.is_empty() {
        return Vec::new();
    }
    let sanitized = crate::internal::util::sanitize_claude_tool_id(call_id);
    let visible = shorten_call_id(&sanitized);
    if visible.is_empty() || visible == call_id {
        vec![call_id.to_owned()]
    } else {
        vec![call_id.to_owned(), visible]
    }
}

fn assistant_message_fingerprint(item: &Value) -> Option<String> {
    if item_type(item).is_some_and(|kind| kind != "message")
        || !message_role(item).is_some_and(|role| role.eq_ignore_ascii_case("assistant"))
    {
        return None;
    }
    let mut text = String::new();
    match item.get("content")? {
        Value::String(value) => text.push_str(value),
        Value::Array(parts) => {
            for part in parts {
                match item_type(part) {
                    Some("input_text" | "output_text") => {
                        text.push_str(part.get("text")?.as_str()?);
                    }
                    Some("refusal") => {
                        text.push_str("\0refusal\0");
                        text.push_str(part.get("refusal")?.as_str()?);
                    }
                    _ => return None,
                }
            }
        }
        _ => return None,
    }
    if text.is_empty() {
        None
    } else {
        Some(hex_digest(&Sha256::digest(text.as_bytes())))
    }
}

fn input_prefix_fingerprint(items: &[Value], end: usize) -> String {
    if end > items.len() {
        return String::new();
    }
    let mut hasher = Sha256::new();
    for item in &items[..end] {
        hasher.update(b"\0item\0");
        let Ok(encoded) = serde_json::to_vec(item) else {
            return String::new();
        };
        hasher.update(encoded);
    }
    hex_digest(&hasher.finalize())
}

fn hex_digest(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn trim_complete_turns(items: &mut Vec<Value>, max_items: usize) {
    while items
        .iter()
        .filter(|item| item_type(item) != Some(CODEX_REPLAY_TURN_TYPE))
        .count()
        > max_items
    {
        let next_marker = items.iter().enumerate().skip(1).find_map(|(index, item)| {
            (item_type(item) == Some(CODEX_REPLAY_TURN_TYPE)).then_some(index)
        });
        match next_marker {
            Some(index) => {
                items.drain(..index);
            }
            None => {
                let marker_count = items
                    .iter()
                    .take_while(|item| item_type(item) == Some(CODEX_REPLAY_TURN_TYPE))
                    .count();
                let excess = items.len().saturating_sub(marker_count + max_items);
                if excess > 0 {
                    items.drain(marker_count..marker_count + excess);
                }
                break;
            }
        }
    }
}

fn shorten_call_id(value: &str) -> String {
    if value.chars().count() <= CODEX_CALL_ID_LIMIT {
        return value.to_owned();
    }
    let digest = Sha256::digest(value.as_bytes());
    let suffix = format!(
        "_{}",
        digest[..8]
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>()
    );
    let prefix = value
        .chars()
        .take(CODEX_CALL_ID_LIMIT - suffix.len())
        .collect::<String>();
    format!("{prefix}{suffix}")
}

fn lock_recover<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|error| error.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replay_is_bounded_session_owned_and_committed_only_from_completion() {
        let cache = CodexReasoningReplayCache::new(1, 2);
        let scope = CodexReasoningReplayScope::new("codex", "session-a").unwrap();
        assert!(cache.commit_completed(scope.clone(), br#"{"output":[{"type":"reasoning","id":"r1","encrypted_content":"cipher"},{"type":"message","role":"assistant","id":"m1"}]}"#));
        let body = cache.apply(
            &scope,
            br#"{"input":[{"type":"message","role":"user","content":[]}]}"#,
        );
        let value: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(value["input"].as_array().unwrap().len(), 2);
        let other = CodexReasoningReplayScope::new("codex", "session-b").unwrap();
        cache.commit_completed(
            other,
            br#"{"output":[{"type":"reasoning","encrypted_content":"other"}]}"#,
        );
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn invalid_signature_clears_only_matching_scope() {
        let cache = CodexReasoningReplayCache::default();
        let scope = CodexReasoningReplayScope::new("codex", "session").unwrap();
        cache.commit_completed(
            scope.clone(),
            br#"{"output":[{"type":"reasoning","encrypted_content":"cipher"}]}"#,
        );
        assert!(cache.clear_on_invalid_signature(
            &scope,
            400,
            br#"{"error":{"code":"invalid_reasoning_signature"}}"#
        ));
        assert!(cache.is_empty());
    }
}
