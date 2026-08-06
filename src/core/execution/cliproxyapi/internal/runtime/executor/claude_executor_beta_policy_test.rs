// ref: internal/runtime/executor/claude_executor_beta_policy_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use super::{claude_code_cli_betas, claude_count_tokens_betas};
use std::collections::HashSet;

#[test]
fn beta_profile_is_body_and_credential_scoped_in_wire_order() {
    let body =
        br#"{"tools":[{"name":"read"}],"speed":"fast","diagnostics":{"previous_message_id":null}}"#;
    let betas = claude_code_cli_betas(body, &HashSet::new(), true);
    assert!(betas.starts_with("claude-code-20250219,oauth-2025-04-20"));
    assert!(betas.contains("advanced-tool-use-2025-11-20"));
    assert!(betas.ends_with("extended-cache-ttl-2025-04-11,cache-diagnosis-2026-04-07"));
    assert_eq!(claude_count_tokens_betas(false), "claude-code-20250219,interleaved-thinking-2025-05-14,context-management-2025-06-27,token-counting-2024-11-01");
}
