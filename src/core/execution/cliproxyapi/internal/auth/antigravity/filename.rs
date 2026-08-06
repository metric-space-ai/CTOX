// ref: internal/auth/antigravity/filename.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: replaced_by_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

// CTOX does not persist Antigravity OAuth credentials in email-derived JSON
// files. `AntigravityCredentialHandles` address encrypted access, refresh and
// state records by account-scoped opaque names; `CtoxAntigravitySecretStore`
// rotates those records, expiry and project routing state in one transaction.
