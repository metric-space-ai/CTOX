// ref: internal/signature/provider_compatibility_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use base64::{engine::general_purpose, Engine as _};

use super::{
    claude_validation::test_claude_signature, compatible_antigravity_claude_thinking_signature,
    compatible_signature_for_provider, compatible_signature_for_provider_block,
    decide_signature_compatibility, detect_signature_provider, detect_signature_provider_for_block,
    signature_payload_without_provider_prefix, signature_provider_from_model_name,
    split_signature_provider_prefix, SignatureBlockKind, SignatureCompatibilityAction,
    SignatureProvider, GEMINI_SKIP_THOUGHT_SIGNATURE_VALIDATOR,
};

fn gemini_signature(value: &[u8]) -> String {
    let mut inner = vec![0x0a, value.len() as u8];
    inner.extend_from_slice(value);
    let mut outer = vec![0x12, inner.len() as u8];
    outer.extend_from_slice(&inner);
    general_purpose::STANDARD.encode(outer)
}

fn gpt_signature() -> String {
    let mut payload = vec![0_u8; 73];
    payload[0] = 0x80;
    general_purpose::URL_SAFE_NO_PAD.encode(payload)
}

#[test]
fn classifies_each_replay_safe_family_and_bypass() {
    let fixtures = [
        (test_claude_signature(), SignatureProvider::Claude),
        (
            gemini_signature(&[0x01, 0x0c, 0x39, 0xd6, 0xc7]),
            SignatureProvider::Gemini,
        ),
        (gpt_signature(), SignatureProvider::Gpt),
        (
            GEMINI_SKIP_THOUGHT_SIGNATURE_VALIDATOR.to_owned(),
            SignatureProvider::GeminiBypass,
        ),
    ];
    for (signature, provider) in fixtures {
        assert_eq!(detect_signature_provider(&signature), provider);
    }
}

#[test]
fn prefixes_are_strict_and_cannot_override_bad_payloads() {
    let claude = test_claude_signature();
    assert_eq!(
        detect_signature_provider(&format!("anthropic#{claude}")),
        SignatureProvider::Claude
    );
    assert_eq!(
        detect_signature_provider(&format!("gpt#{claude}")),
        SignatureProvider::Unknown
    );
    assert_eq!(
        detect_signature_provider(&format!("claude-cache#{claude}")),
        SignatureProvider::Unknown
    );
    assert_eq!(
        split_signature_provider_prefix("unknown#raw"),
        (SignatureProvider::Unknown, "unknown#raw", false)
    );
}

#[test]
fn compatible_provider_strips_only_trusted_prefix() {
    let gemini = gemini_signature(&[0x01, 1, 2, 3, 4]);
    assert_eq!(
        compatible_signature_for_provider(SignatureProvider::Gemini, &format!("gemini#{gemini}")),
        Some(gemini.clone())
    );
    assert_eq!(
        signature_payload_without_provider_prefix(" unknown#raw "),
        "unknown#raw"
    );
    assert_eq!(
        compatible_signature_for_provider(SignatureProvider::Claude, &gemini),
        None
    );
}

#[test]
fn bare_uuid_is_unknown_and_uses_gemini_bypass_for_model_parts() {
    let uuid = gemini_signature(b"e24830a7-5cd6-42fe-998b-ee539e72b9c3");
    assert_eq!(detect_signature_provider(&uuid), SignatureProvider::Gemini);

    let bare = general_purpose::STANDARD.encode(b"e24830a7-5cd6-42fe-998b-ee539e72b9c3");
    assert_eq!(detect_signature_provider(&bare), SignatureProvider::Unknown);
    let decision = decide_signature_compatibility(
        SignatureProvider::Gemini,
        &bare,
        SignatureBlockKind::GeminiFunctionCall,
    );
    assert_eq!(
        decision.action,
        SignatureCompatibilityAction::ReplaceWithGeminiBypass
    );
    assert_eq!(
        decision.replacement_signature,
        GEMINI_SKIP_THOUGHT_SIGNATURE_VALIDATOR
    );
    assert_eq!(
        compatible_signature_for_provider_block(
            SignatureProvider::Gemini,
            &bare,
            SignatureBlockKind::GeminiFunctionCall
        ),
        None
    );
}

#[test]
fn antigravity_wraps_claude_once_and_rejects_gemini_e_prefix() {
    let claude = test_claude_signature();
    let wrapped = general_purpose::STANDARD.encode(claude.as_bytes());
    assert_eq!(
        compatible_antigravity_claude_thinking_signature(&claude),
        Some(wrapped.clone())
    );
    assert_eq!(
        compatible_antigravity_claude_thinking_signature(&wrapped),
        Some(wrapped)
    );
    let gemini_e = gemini_signature(&[0x01, 0x0c, 0x39, 0xd6, 0xc7]);
    assert!(gemini_e.starts_with('E'));
    assert_eq!(
        compatible_antigravity_claude_thinking_signature(&gemini_e),
        None
    );
}

#[test]
fn model_names_map_without_guessing_unknowns() {
    for (model, provider) in [
        ("claude-sonnet-4-6", SignatureProvider::Claude),
        ("google/gemini-3.1-pro", SignatureProvider::Gemini),
        ("openai/gpt-5", SignatureProvider::Gpt),
        ("codex-mini", SignatureProvider::Gpt),
        ("o3", SignatureProvider::Gpt),
        ("minimax-m3", SignatureProvider::Unknown),
    ] {
        assert_eq!(signature_provider_from_model_name(model), provider);
    }
}

#[test]
fn block_context_preserves_native_gemini_and_replaces_foreign() {
    let gemini = gemini_signature(&[0x01, 1, 2, 3, 4]);
    assert_eq!(
        detect_signature_provider_for_block(&gemini, SignatureBlockKind::GeminiFunctionCall),
        SignatureProvider::Gemini
    );
    let preserved = decide_signature_compatibility(
        SignatureProvider::Gemini,
        &gemini,
        SignatureBlockKind::GeminiFunctionCall,
    );
    assert!(preserved.compatible);
    assert_eq!(preserved.action, SignatureCompatibilityAction::Preserve);
    let foreign = decide_signature_compatibility(
        SignatureProvider::Claude,
        &gpt_signature(),
        SignatureBlockKind::ClaudeThinking,
    );
    assert_eq!(foreign.action, SignatureCompatibilityAction::DropBlock);
}
