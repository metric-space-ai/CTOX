// Origin: CTOX
// License: AGPL-3.0-only

use super::CLAUDE_CODE_INSTRUCTIONS;

#[test]
fn embeds_the_exact_pinned_instruction_payload() {
    assert_eq!(
        CLAUDE_CODE_INSTRUCTIONS,
        r#"[{"type":"text","text":"You are Claude Code, Anthropic's official CLI for Claude.","cache_control":{"type":"ephemeral"}}]"#
    );
}
