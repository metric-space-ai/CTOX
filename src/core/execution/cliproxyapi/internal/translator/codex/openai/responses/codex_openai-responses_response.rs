// ref: internal/translator/codex/openai/responses/codex_openai-responses_response.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use std::borrow::Cow;

use serde::Deserialize;
use serde_json::value::RawValue;

pub fn convert_codex_response_to_openai_responses(raw_json: &[u8]) -> Vec<Vec<u8>> {
    let Some(payload) = raw_json.strip_prefix(b"data:") else {
        return vec![raw_json.to_vec()];
    };
    let trimmed = match std::str::from_utf8(payload) {
        Ok(text) => text.trim().as_bytes(),
        Err(_) => payload.trim_ascii(),
    };
    let mut output = Vec::with_capacity(6 + trimmed.len());
    output.extend_from_slice(b"data: ");
    output.extend_from_slice(trimmed);
    vec![output]
}

pub fn convert_codex_response_to_openai_responses_non_stream(raw_json: &[u8]) -> Vec<u8> {
    #[derive(Deserialize)]
    struct TerminalEvent<'a> {
        #[serde(rename = "type", borrow)]
        event_type: Cow<'a, str>,
        #[serde(borrow)]
        response: Option<&'a RawValue>,
    }

    let Ok(event) = serde_json::from_slice::<TerminalEvent<'_>>(raw_json) else {
        return Vec::new();
    };
    if !matches!(
        event.event_type.as_ref(),
        "response.completed" | "response.incomplete"
    ) {
        return Vec::new();
    }
    event
        .response
        .map_or_else(Vec::new, |response| response.get().as_bytes().to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stream_reframes_only_exact_data_prefix() {
        assert_eq!(
            convert_codex_response_to_openai_responses(b"data:\n {\"type\":\"x\"} \n"),
            vec![b"data: {\"type\":\"x\"}".to_vec()]
        );
        assert_eq!(
            convert_codex_response_to_openai_responses(b" data: {}"),
            vec![b" data: {}".to_vec()]
        );
    }

    #[test]
    fn non_stream_extracts_only_terminal_raw_response() {
        let raw =
            br#"{"type":"response.incomplete","response": { "id": "r", "status":"incomplete" }}"#;
        assert_eq!(
            convert_codex_response_to_openai_responses_non_stream(raw),
            br#"{ "id": "r", "status":"incomplete" }"#
        );
        assert!(convert_codex_response_to_openai_responses_non_stream(
            br#"{"type":"response.output_text.delta","response":{}}"#
        )
        .is_empty());
    }
}
