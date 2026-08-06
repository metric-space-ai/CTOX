// ref: internal/signature/provider_compatibility.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use super::claude_validation::{
    is_valid_claude_cais_signature, is_valid_claude_thinking_signature_with_options,
    normalize_claude_bypass_thinking_signature,
    normalize_claude_provider_native_thinking_signature, ClaudeSignatureValidationOptions,
};
use super::gemini_validation::{
    is_gemini_thought_signature_bypass, recognized_gemini_provider_signature,
    GEMINI_SKIP_THOUGHT_SIGNATURE_VALIDATOR,
};
use super::gpt_validation::is_valid_gpt_reasoning_signature;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SignatureProvider {
    Unknown,
    Claude,
    Gemini,
    GeminiBypass,
    Gpt,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum SignatureBlockKind {
    #[default]
    Unknown,
    ClaudeThinking,
    GeminiModelPart,
    GeminiFunctionCall,
    GptReasoning,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SignatureCompatibilityAction {
    Preserve,
    DropBlock,
    DropSignature,
    ReplaceWithGeminiBypass,
    NoCompatibleReplacement,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SignatureCompatibilityDecision {
    pub target_provider: SignatureProvider,
    pub detected_provider: SignatureProvider,
    pub block_kind: SignatureBlockKind,
    pub compatible: bool,
    pub action: SignatureCompatibilityAction,
    pub replacement_signature: String,
    pub normalized_signature: String,
    pub reason: String,
}

pub fn signature_provider_from_model_name(model_name: &str) -> SignatureProvider {
    let model = model_name.trim().to_ascii_lowercase();
    if model.contains("claude") {
        SignatureProvider::Claude
    } else if model.contains("gemini") {
        SignatureProvider::Gemini
    } else if model.contains("gpt")
        || model.contains("openai")
        || model.contains("codex")
        || ["o1", "o3", "o4"]
            .iter()
            .any(|prefix| model.starts_with(prefix))
    {
        SignatureProvider::Gpt
    } else {
        SignatureProvider::Unknown
    }
}

pub fn detect_signature_provider(raw_signature: &str) -> SignatureProvider {
    detect_signature_provider_for_block(raw_signature, SignatureBlockKind::Unknown)
}

pub fn detect_signature_provider_for_block(
    raw_signature: &str,
    _block_kind: SignatureBlockKind,
) -> SignatureProvider {
    let signature = raw_signature.trim();
    if signature.is_empty() {
        return SignatureProvider::Unknown;
    }
    let (prefix, payload, prefixed) = split_signature_provider_prefix(signature);
    if prefixed {
        return match prefix {
            SignatureProvider::Claude
                if is_valid_claude_cais_signature(payload)
                    || is_strict_claude_thinking_signature(payload) =>
            {
                SignatureProvider::Claude
            }
            SignatureProvider::Gpt if is_valid_gpt_reasoning_signature(payload) => {
                SignatureProvider::Gpt
            }
            SignatureProvider::Gemini if is_gemini_thought_signature_bypass(payload) => {
                SignatureProvider::GeminiBypass
            }
            SignatureProvider::Gemini if recognized_gemini_provider_signature(payload) => {
                SignatureProvider::Gemini
            }
            _ => SignatureProvider::Unknown,
        };
    }
    if signature.contains('#') {
        return SignatureProvider::Unknown;
    }
    if is_gemini_thought_signature_bypass(signature) {
        SignatureProvider::GeminiBypass
    } else if is_valid_gpt_reasoning_signature(signature) {
        SignatureProvider::Gpt
    } else if is_valid_claude_cais_signature(signature)
        || is_strict_claude_thinking_signature(signature)
    {
        SignatureProvider::Claude
    } else if recognized_gemini_provider_signature(signature) {
        SignatureProvider::Gemini
    } else {
        SignatureProvider::Unknown
    }
}

pub fn decide_signature_compatibility(
    target_provider: SignatureProvider,
    raw_signature: &str,
    block_kind: SignatureBlockKind,
) -> SignatureCompatibilityDecision {
    decide_signature_compatibility_for_model(target_provider, "", raw_signature, block_kind)
}

pub fn decide_signature_compatibility_for_model(
    target_provider: SignatureProvider,
    target_model: &str,
    raw_signature: &str,
    block_kind: SignatureBlockKind,
) -> SignatureCompatibilityDecision {
    let target_provider = normalize_target_provider(target_provider);
    let detected_provider = detect_signature_provider_for_block(raw_signature, block_kind);
    let compatible = provider_matches_target(target_provider, detected_provider);
    let normalized_signature = if compatible {
        normalize_for_provider(target_provider, raw_signature).unwrap_or_default()
    } else {
        String::new()
    };
    let (action, replacement_signature, reason) = if compatible && !normalized_signature.is_empty()
    {
        (
            SignatureCompatibilityAction::Preserve,
            String::new(),
            claude_compatibility_reason(target_provider, raw_signature, target_model),
        )
    } else {
        match target_provider {
            SignatureProvider::Gemini
                if matches!(
                    block_kind,
                    SignatureBlockKind::Unknown
                        | SignatureBlockKind::GeminiModelPart
                        | SignatureBlockKind::GeminiFunctionCall
                ) => (
                    SignatureCompatibilityAction::ReplaceWithGeminiBypass,
                    GEMINI_SKIP_THOUGHT_SIGNATURE_VALIDATOR.to_owned(),
                    "Gemini can bypass synthetic or incompatible model-part signatures with the documented sentinel".to_owned(),
                ),
            SignatureProvider::Gemini => (
                SignatureCompatibilityAction::DropBlock,
                String::new(),
                "signature is not compatible with Gemini and this block is not a bypass-safe Gemini model part".to_owned(),
            ),
            SignatureProvider::Claude => (
                SignatureCompatibilityAction::DropBlock,
                String::new(),
                "Claude has no cross-provider bypass sentinel for thinking blocks".to_owned(),
            ),
            SignatureProvider::Gpt => (
                SignatureCompatibilityAction::DropBlock,
                String::new(),
                "GPT reasoning encrypted_content cannot be synthesized from another provider signature".to_owned(),
            ),
            _ => (
                SignatureCompatibilityAction::NoCompatibleReplacement,
                String::new(),
                "unknown target provider".to_owned(),
            ),
        }
    };
    SignatureCompatibilityDecision {
        target_provider,
        detected_provider,
        block_kind,
        compatible: compatible && !normalized_signature.is_empty(),
        action,
        replacement_signature,
        normalized_signature,
        reason,
    }
}

pub fn compatible_signature_for_provider(
    target_provider: SignatureProvider,
    raw_signature: &str,
) -> Option<String> {
    let decision =
        decide_signature_compatibility(target_provider, raw_signature, SignatureBlockKind::Unknown);
    decision.compatible.then_some(decision.normalized_signature)
}

pub fn compatible_signature_for_provider_block(
    target_provider: SignatureProvider,
    raw_signature: &str,
    block_kind: SignatureBlockKind,
) -> Option<String> {
    let decision = decide_signature_compatibility(target_provider, raw_signature, block_kind);
    decision.compatible.then_some(decision.normalized_signature)
}

/// Claude thought signatures sent through Antigravity use the double-layer
/// R form. Native E signatures are wrapped once; an already valid R form is
/// retained. CAIS envelopes are intentionally not accepted by this upstream
/// endpoint.
pub fn compatible_antigravity_claude_thinking_signature(raw_signature: &str) -> Option<String> {
    if detect_signature_provider_for_block(raw_signature, SignatureBlockKind::ClaudeThinking)
        != SignatureProvider::Claude
    {
        return None;
    }
    let payload = signature_payload_without_provider_prefix(raw_signature);
    normalize_claude_bypass_thinking_signature(payload, true)
}

pub fn signature_payload_without_provider_prefix(raw_signature: &str) -> &str {
    let (_, payload, prefixed) = split_signature_provider_prefix(raw_signature);
    if prefixed {
        payload
    } else {
        raw_signature.trim()
    }
}

/// Returns a replay-safe native Gemini 3.x signature after validating the
/// observed field-2 -> field-1 protobuf envelope. Synthetic/cross-provider
/// values are rejected; callers may deliberately substitute the documented
/// bypass only at the first function call of a model turn.
pub fn compatible_gemini_signature(raw_signature: &str) -> Option<String> {
    compatible_signature_for_provider(SignatureProvider::Gemini, raw_signature)
}

fn is_strict_claude_thinking_signature(signature: &str) -> bool {
    is_valid_claude_thinking_signature_with_options(
        signature,
        ClaudeSignatureValidationOptions {
            strict: true,
            ..ClaudeSignatureValidationOptions::default()
        },
    )
}

pub fn split_signature_provider_prefix(raw_signature: &str) -> (SignatureProvider, &str, bool) {
    let signature = raw_signature.trim();
    let Some((prefix, payload)) = signature.split_once('#') else {
        return (SignatureProvider::Unknown, raw_signature, false);
    };
    let provider = match prefix.trim().to_ascii_lowercase().as_str() {
        "claude" | "anthropic" | "cais" | "claude-cais" | "claude_cais" | "ccmax"
        | "claude-code-max" | "claude_code_max" => SignatureProvider::Claude,
        "gemini" | "google" => SignatureProvider::Gemini,
        "openai" | "gpt" | "codex" => SignatureProvider::Gpt,
        _ => SignatureProvider::Unknown,
    };
    if provider == SignatureProvider::Unknown {
        (SignatureProvider::Unknown, raw_signature, false)
    } else {
        (provider, payload.trim(), true)
    }
}

pub(crate) fn has_signature_provider_prefix(raw_signature: &str) -> bool {
    split_signature_provider_prefix(raw_signature).2
}

fn normalize_for_provider(
    target_provider: SignatureProvider,
    raw_signature: &str,
) -> Option<String> {
    let payload = signature_payload_without_provider_prefix(raw_signature);
    match target_provider {
        SignatureProvider::Claude if is_valid_claude_cais_signature(payload) => {
            Some(payload.to_owned())
        }
        SignatureProvider::Claude => normalize_claude_provider_native_thinking_signature(payload),
        SignatureProvider::Gemini | SignatureProvider::GeminiBypass
            if is_gemini_thought_signature_bypass(payload)
                || recognized_gemini_provider_signature(payload) =>
        {
            Some(payload.to_owned())
        }
        SignatureProvider::Gpt if is_valid_gpt_reasoning_signature(payload) => {
            Some(payload.to_owned())
        }
        _ => None,
    }
}

fn normalize_target_provider(provider: SignatureProvider) -> SignatureProvider {
    if provider == SignatureProvider::GeminiBypass {
        SignatureProvider::Gemini
    } else {
        provider
    }
}

fn provider_matches_target(target: SignatureProvider, detected: SignatureProvider) -> bool {
    match target {
        SignatureProvider::Gemini => matches!(
            detected,
            SignatureProvider::Gemini | SignatureProvider::GeminiBypass
        ),
        SignatureProvider::Claude => detected == SignatureProvider::Claude,
        SignatureProvider::Gpt => detected == SignatureProvider::Gpt,
        _ => false,
    }
}

fn claude_compatibility_reason(
    target: SignatureProvider,
    raw_signature: &str,
    target_model: &str,
) -> String {
    if target != SignatureProvider::Claude {
        return "signature provider matches target provider".to_owned();
    }
    let Ok(info) = super::inspect_claude_cais_signature(signature_payload_without_provider_prefix(
        raw_signature,
    )) else {
        return "signature provider matches target provider".to_owned();
    };
    let mut reason = format!(
        "valid Claude CAIS signature with embedded model {} is compatible with any Claude target",
        info.model_text
    );
    if !target_model.trim().is_empty() {
        reason.push_str(", including target model ");
        reason.push_str(target_model.trim());
    }
    reason
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::internal::signature::claude_validation::test_claude_signature;
    use base64::{engine::general_purpose, Engine as _};

    fn gpt_signature() -> String {
        let mut payload = vec![0_u8; 1 + 8 + 16 + 16 + 32];
        payload[0] = 0x80;
        payload[8] = 1;
        for (index, byte) in payload.iter_mut().enumerate().skip(9) {
            *byte = index as u8;
        }
        general_purpose::URL_SAFE.encode(payload)
    }

    #[test]
    fn detects_provider_and_strips_trusted_prefix() {
        let claude = test_claude_signature();
        assert_eq!(
            detect_signature_provider(&claude),
            SignatureProvider::Claude
        );
        assert_eq!(
            compatible_signature_for_provider(
                SignatureProvider::Claude,
                &format!("anthropic#{claude}")
            ),
            Some(claude)
        );
        assert_eq!(
            detect_signature_provider(&gpt_signature()),
            SignatureProvider::Gpt
        );
    }

    #[test]
    fn rejects_cross_provider_replay() {
        let decision = decide_signature_compatibility(
            SignatureProvider::Claude,
            &gpt_signature(),
            SignatureBlockKind::ClaudeThinking,
        );
        assert!(!decision.compatible);
        assert_eq!(decision.action, SignatureCompatibilityAction::DropBlock);
        assert!(decision.normalized_signature.is_empty());
    }

    #[test]
    fn maps_model_families_without_guessing_unknowns() {
        assert_eq!(
            signature_provider_from_model_name("claude-sonnet-4-6"),
            SignatureProvider::Claude
        );
        assert_eq!(
            signature_provider_from_model_name("openai/gpt-5"),
            SignatureProvider::Gpt
        );
        assert_eq!(
            signature_provider_from_model_name("minimax-m3"),
            SignatureProvider::Unknown
        );
    }

    #[test]
    fn validates_only_known_gemini_field_two_envelopes() {
        let opaque = [0x01, 0x0c, 0x39, 0xd6, 0xc7, 0xaa];
        let mut inner = vec![0x0a, opaque.len() as u8];
        inner.extend_from_slice(&opaque);
        let mut outer = vec![0x12, inner.len() as u8];
        outer.extend_from_slice(&inner);
        let signature = general_purpose::STANDARD.encode(outer);

        assert_eq!(
            compatible_gemini_signature(&signature),
            Some(signature.clone())
        );
        assert_eq!(
            compatible_gemini_signature(&format!("gemini#{signature}")),
            Some(signature)
        );
        assert_eq!(compatible_gemini_signature("claude#invalid"), None);
        assert_eq!(compatible_gemini_signature("not-base64"), None);
    }

    #[test]
    fn rejects_untrusted_or_mismatched_cache_prefixes() {
        let claude = test_claude_signature();
        assert_eq!(
            detect_signature_provider(&format!("unknown#{claude}")),
            SignatureProvider::Unknown
        );
        assert_eq!(
            detect_signature_provider(&format!("gpt#{claude}")),
            SignatureProvider::Unknown
        );
    }

    #[test]
    fn antigravity_wraps_native_claude_signature_exactly_once() {
        let native = test_claude_signature();
        let wrapped = general_purpose::STANDARD.encode(native.as_bytes());
        assert_eq!(
            compatible_antigravity_claude_thinking_signature(&native),
            Some(wrapped.clone())
        );
        assert_eq!(
            compatible_antigravity_claude_thinking_signature(&wrapped),
            Some(wrapped)
        );
        assert_eq!(
            compatible_antigravity_claude_thinking_signature(&gpt_signature()),
            None
        );
    }
}
