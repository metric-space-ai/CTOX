// ref: internal/translator/antigravity/openai/chat-completions/noop_optimization_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use super::normalize_antigravity_openai_thinking_config;

#[test]
fn canonical_thinking_config_reuses_the_input_allocation() {
    let input = br#"{"request":{"generationConfig":{"thinkingConfig":{"includeThoughts":true,"thinkingLevel":"high","thinkingBudget":8192}}}}"#.to_vec();
    let pointer = input.as_ptr();
    let output = normalize_antigravity_openai_thinking_config(input);
    assert_eq!(output.as_ptr(), pointer);
}
