// ref: internal/misc/claude_code_instructions.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

/// Claude Code system instructions embedded into the executable at compile
/// time, matching Go's `//go:embed claude_code_instructions.txt` payload. The
/// source resource remains next to this module for upstream diffs; a conventional
/// final text-file newline is excluded from the wire payload.
pub static CLAUDE_CODE_INSTRUCTIONS: &str =
    include_str!("claude_code_instructions.txt").trim_ascii_end();
