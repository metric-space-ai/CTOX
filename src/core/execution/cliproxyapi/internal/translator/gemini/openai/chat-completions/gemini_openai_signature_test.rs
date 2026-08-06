// ref: internal/translator/gemini/openai/chat-completions/gemini_openai_signature_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use serde_json::Value;

use super::convert_openai_chat_request_to_gemini;

const NATIVE: &str = "EjQKMgEMOdbHO0Gd+c9Mxk4ELwPGbpCEcp2mFfYYLix2UVtBH3fL8GECc4+JITVnHF4qZDsA";

#[test]
fn preserves_compatible_signature_and_replaces_unknown_signature() {
    for (raw, expected) in [
        (format!("gemini#{NATIVE}"), NATIVE),
        (
            "not-a-provider-signature".to_owned(),
            super::gemini_openai_request::GEMINI_THOUGHT_BYPASS,
        ),
    ] {
        let input = format!(
            r#"{{"messages":[{{"role":"assistant","tool_calls":[{{"id":"call","type":"function","function":{{"name":"lookup","arguments":"{{}}"}},"extra_content":{{"google":{{"thought_signature":"{raw}"}}}}}}]}}]}}"#
        );
        let output: Value = serde_json::from_slice(&convert_openai_chat_request_to_gemini(
            "gemini-test",
            input.as_bytes(),
            false,
        ))
        .unwrap();
        assert_eq!(
            output["contents"][0]["parts"][0]["thoughtSignature"],
            expected
        );
    }
}
