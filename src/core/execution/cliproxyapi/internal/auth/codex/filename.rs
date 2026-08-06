// ref: internal/auth/codex/filename.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: replaced_by_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

// CTOX deliberately does not derive filesystem paths from provider identity or
// persist Codex OAuth credentials in vendor-style JSON files. Account-unique
// `CodexSecretHandle` values identify encrypted records in the host secret
// store, and the store rotates the full ID/access/refresh snapshot atomically.
// See `internal/auth/codex/token.rs` and the root `CtoxCodexSecretStore`.
