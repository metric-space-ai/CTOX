// ref: internal/util/claude_attribution.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

const CLAUDE_CODE_ATTRIBUTION_SYSTEM_PREFIX: &str = "x-anthropic-billing-header:";

pub fn is_claude_code_attribution_system_text(text: &str) -> bool {
    text.trim_start()
        .starts_with(CLAUDE_CODE_ATTRIBUTION_SYSTEM_PREFIX)
}
