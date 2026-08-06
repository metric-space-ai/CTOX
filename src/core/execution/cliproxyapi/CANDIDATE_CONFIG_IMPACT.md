# Candidate configuration impact

This note records the non-Go configuration impact of upstream candidate
`a88197f845c979132c8978ea223c6af05cc81536` relative to accepted pin
`ffdb9c9fbc78a6235d59c9ccbdc4243ba35ecdcd`. It is review evidence for
`config.example.yaml`; it is not a second CTOX configuration authority.

CTOX keeps runtime behavior in typed configuration and durable secret/runtime
stores. It does not copy upstream process-environment fallbacks or plaintext
credential JSON configuration.

| Candidate change | CTOX disposition | Authority / evidence |
| --- | --- | --- |
| `cloak.mode: always` still bypasses a confirmed native Claude Code client | Ported as request-scoped strong-signal detection and `ClaudeCloakPolicy::verified_claude_code`; a User-Agent alone is insufficient | `internal/runtime/executor/helps/claude_client_detection.rs`, `internal/runtime/executor/claude_executor_cloaking.rs` |
| Non-strict cloaking uses a legacy reminder only where required and otherwise a `role=system` turn; strict mode keeps only proxy-owned identity/billing material | Ported in the Claude cloaking pipeline with typed request-scoped failures for unsupported caller system blocks | `internal/runtime/executor/claude_executor_cloaking.rs` |
| Cloaked OAuth custom tools receive stable opaque MCP aliases | Ported with caller-secret-scoped aliases and reverse mapping for unary and streaming responses | `internal/runtime/executor/claude_executor_request.rs`, `internal/runtime/executor/claude_executor_execute.rs` |
| `experimental-cch-signing` is deprecated and CCH signing is automatic | The legacy toggle is deliberately not exposed. OAuth requests install a deterministic fallback billing block when needed and sign only after all body rewrites | `internal/runtime/executor/claude_executor_request.rs`, `internal/runtime/executor/claude_executor_signing.rs` |
| `Anthropic-Beta` is assembled per request and ordered like Claude Code | Ported as a request/body/model-aware beta policy; unsupported direct-Anthropic caller betas are not blindly forwarded | `internal/runtime/executor/claude_executor_request.rs`, `internal/runtime/executor/claude_executor_beta_policy_test.rs` |
| Claude fingerprint baseline becomes `2.1.220 / 0.94.0 / v26.3.0` | Ported as the typed default device profile and the Candidate helper baseline | `internal/runtime/executor/claude_executor.rs`, `internal/runtime/executor/helps/claude_device_profile.rs` |
| Native CLI, `sdk-cli`, and VSCode requests are confirmed only by the complete measured signal set | Ported by `detect_claude_code_request`; malformed or partial signals remain unconfirmed | `internal/runtime/executor/helps/claude_client_detection.rs` and its tests |
| Credential timezone overrides a global header-default timezone | Adapted to CTOX: every Claude subscription account has a validated IANA timezone; empty means UTC. There is intentionally no process-global or environment fallback, so account authority is unambiguous | `internal/config/config_types.rs`, `cliproxyapi_host.rs`, `internal/runtime/executor/claude_executor_cloaking.rs` |
| Device-profile stabilization is scoped per auth/API key | Ported helper semantics use an auth-scoped cache. CTOX may additionally pin an explicit typed per-account profile; explicit typed configuration wins over learned request material | `internal/runtime/executor/helps/claude_device_profile.rs`, `internal/config/config_types.rs` |

## Required promotion evidence

This impact review is complete only when the Candidate review ledger also has
production-path evidence for native-client detection, credential identity,
device-profile selection, final-body CCH signing, beta ordering, and account
timezone validation. Helper-only unit tests are insufficient when a helper is
not consumed by the active executor/host path.
