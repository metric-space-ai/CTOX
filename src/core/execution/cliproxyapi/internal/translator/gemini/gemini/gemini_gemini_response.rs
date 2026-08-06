// ref: internal/translator/gemini/gemini/gemini_gemini_response.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use crate::internal::translator::common::gemini_token_count_json;

pub fn passthrough_gemini_response_stream(raw_json: &[u8]) -> Vec<Vec<u8>> {
    let payload = raw_json
        .strip_prefix(b"data:")
        .map(trim_like_go_bytes)
        .unwrap_or(raw_json);
    if payload == b"[DONE]" {
        Vec::new()
    } else {
        vec![payload.to_vec()]
    }
}

pub fn passthrough_gemini_response_non_stream(raw_json: &[u8]) -> Vec<u8> {
    raw_json.to_vec()
}

pub fn gemini_token_count(count: i64) -> Vec<u8> {
    gemini_token_count_json(count)
}

fn trim_like_go_bytes(bytes: &[u8]) -> &[u8] {
    match std::str::from_utf8(bytes) {
        Ok(text) => text.trim().as_bytes(),
        Err(_) => bytes.trim_ascii(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn response_and_count_contracts_match_upstream_shape() {
        assert_eq!(
            passthrough_gemini_response_stream(b"data: {\"x\":1}\n"),
            vec![b"{\"x\":1}".to_vec()]
        );
        assert!(passthrough_gemini_response_stream(b"data: [DONE]").is_empty());
        assert_eq!(
            passthrough_gemini_response_non_stream(b" {\"x\":1} "),
            b" {\"x\":1} "
        );
        let count: serde_json::Value = serde_json::from_slice(&gemini_token_count(7)).unwrap();
        assert_eq!(count["totalTokens"], 7);
        assert_eq!(count["promptTokensDetails"][0]["modality"], "TEXT");
        assert_eq!(count["promptTokensDetails"][0]["tokenCount"], 7);
    }
}
