// Origin: CTOX
// License: AGPL-3.0-only

mod claude;
mod claude_messages_sanitize;
mod claude_validation;
mod gemini_sanitize;
mod gemini_validation;
mod gpt_validation;
mod grok_validation;
mod provider_compatibility;

#[cfg(test)]
mod claude_test;
#[cfg(test)]
mod gemini_sanitize_test;
#[cfg(test)]
mod gemini_validation_test;
#[cfg(test)]
mod gpt_validation_test;
#[cfg(test)]
mod grok_validation_test;
#[cfg(test)]
mod provider_compatibility_test;

pub use claude::{
    strip_invalid_claude_thinking_blocks, strip_invalid_claude_thinking_blocks_and_empty_messages,
};

pub use claude_messages_sanitize::{
    sanitize_claude_messages_for_claude_upstream, ClaudeSignatureSanitizeReport,
};
pub use claude_validation::{
    has_claude_thinking_signature_prefix, has_decodable_claude_thinking_signature,
    inspect_claude_cais_signature, inspect_claude_double_layer_signature,
    inspect_claude_signature_payload, inspect_claude_single_layer_signature,
    is_valid_claude_cais_signature, is_valid_claude_thinking_signature,
    is_valid_claude_thinking_signature_with_options, normalize_claude_bypass_thinking_signature,
    normalize_claude_provider_native_thinking_signature,
    normalize_claude_provider_native_thinking_signature_with_options,
    normalize_claude_thinking_signature, validate_claude_thinking_signatures,
    ClaudeCaisSignatureInfo, ClaudeSignatureTree, ClaudeSignatureValidationOptions,
    MAX_CLAUDE_THINKING_SIGNATURE_LEN,
};
pub use gemini_sanitize::{
    sanitize_gemini_request_thought_signatures, GEMINI_SKIP_THOUGHT_SIGNATURE_VALIDATOR,
};
pub use gemini_validation::{
    inspect_gemini_thought_signature, is_gemini_thought_signature_bypass,
    is_valid_gemini_thought_signature, validate_gemini_function_call_pairing,
    validate_gemini_thought_signatures, GeminiThoughtSignatureEnvelope, GeminiThoughtSignatureInfo,
    GeminiThoughtSignatureValidationOptions, GEMINI_CONTEXT_ENGINEERING_BYPASS,
    MAX_GEMINI_THOUGHT_SIGNATURE_LEN,
};
pub use gpt_validation::{
    inspect_gpt_reasoning_signature, is_valid_gpt_reasoning_signature, GptReasoningSignatureInfo,
    MAX_GPT_REASONING_SIGNATURE_LEN,
};
pub use grok_validation::{
    inspect_grok_encrypted_content, is_valid_grok_encrypted_content, GrokEncryptedContentError,
    GrokEncryptedContentInfo, MAX_GROK_ENCRYPTED_CONTENT_LEN,
    MIN_GROK_ENCRYPTED_CONTENT_DECODED_LEN, MIN_GROK_ENCRYPTED_CONTENT_ENTROPY_RATIO,
};
pub use provider_compatibility::{
    compatible_antigravity_claude_thinking_signature, compatible_gemini_signature,
    compatible_signature_for_provider, compatible_signature_for_provider_block,
    decide_signature_compatibility, decide_signature_compatibility_for_model,
    detect_signature_provider, detect_signature_provider_for_block,
    signature_payload_without_provider_prefix, signature_provider_from_model_name,
    split_signature_provider_prefix, SignatureBlockKind, SignatureCompatibilityAction,
    SignatureCompatibilityDecision, SignatureProvider,
};
