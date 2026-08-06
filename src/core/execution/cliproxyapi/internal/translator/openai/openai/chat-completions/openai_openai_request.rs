// ref: internal/translator/openai/openai/chat-completions/openai_openai_request.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use crate::internal::translator::common::set_top_level_string;

pub fn convert_openai_request_to_openai(
    model_name: &str,
    input_raw_json: &[u8],
    _stream: bool,
) -> Vec<u8> {
    set_top_level_string(input_raw_json, "model", model_name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn updates_only_model_and_preserves_noop_bytes() {
        let input = b" { \"model\" : \"old\", \"messages\" : [] } ";
        let output: serde_json::Value =
            serde_json::from_slice(&convert_openai_request_to_openai("new", input, false)).unwrap();
        assert_eq!(output["model"], "new");
        let normalized = b" { \"model\" : \"new\", \"messages\" : [] } ";
        assert_eq!(
            convert_openai_request_to_openai("new", normalized, true),
            normalized
        );
    }
}
