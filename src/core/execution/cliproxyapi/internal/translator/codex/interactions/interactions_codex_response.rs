// ref: internal/translator/codex/interactions/interactions_codex_response.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use crate::internal::translator::openai::interactions::responses::{
    convert_openai_responses_response_to_interactions_non_stream,
    convert_openai_responses_response_to_interactions_stream,
};
use crate::sdk::translator::{TranslationContext, TranslationState};
use serde_json::Value;

#[derive(Default)]
pub struct CodexToInteractionsState {
    inner: TranslationState,
}

pub fn convert_codex_response_to_interactions_stream(
    context: &TranslationContext,
    model_name: &str,
    original_request: &[u8],
    request: &[u8],
    raw: &[u8],
    state: &mut CodexToInteractionsState,
) -> Vec<Vec<u8>> {
    let normalized = normalize_incomplete_stream_event(raw);
    convert_openai_responses_response_to_interactions_stream(
        context,
        model_name,
        original_request,
        request,
        &normalized,
        &mut state.inner,
    )
}

pub fn convert_codex_response_to_interactions_non_stream(
    context: &TranslationContext,
    model_name: &str,
    original_request: &[u8],
    request: &[u8],
    raw: &[u8],
) -> Vec<u8> {
    let response = unwrap_terminal_response(raw);
    let mut state = None;
    let mut output = convert_openai_responses_response_to_interactions_non_stream(
        context,
        model_name,
        original_request,
        request,
        &response,
        &mut state,
    );
    if let (Ok(source), Ok(mut translated)) = (
        serde_json::from_slice::<Value>(&response),
        serde_json::from_slice::<Value>(&output),
    ) {
        if let Some(status) = source.get("status") {
            translated["status"] = status.clone();
            output = serde_json::to_vec(&translated).unwrap_or(output);
        }
    }
    output
}

fn unwrap_terminal_response(raw: &[u8]) -> Vec<u8> {
    let Ok(root) = serde_json::from_slice::<Value>(raw) else {
        return raw.to_vec();
    };
    root.get("response")
        .and_then(|response| serde_json::to_vec(response).ok())
        .unwrap_or_else(|| raw.to_vec())
}

fn normalize_incomplete_stream_event(raw: &[u8]) -> Vec<u8> {
    let (prefix, payload) = raw
        .strip_prefix(b"data: ")
        .map_or((&b""[..], raw), |payload| (&b"data: "[..], payload));
    let Ok(mut root) = serde_json::from_slice::<Value>(payload) else {
        return raw.to_vec();
    };
    if root.get("type").and_then(Value::as_str) != Some("response.incomplete") {
        return raw.to_vec();
    }
    root["type"] = Value::String("response.completed".into());
    let mut out = prefix.to_vec();
    out.extend(serde_json::to_vec(&root).unwrap_or_else(|_| payload.to_vec()));
    out
}
