// ref: internal/api/buffered_conn.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: replaced_by_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

// CTOX replacement: the supervised subscription socket is HTTP-only and the
// Tokio HTTP reader consumes the prefix directly. No protocol sniff therefore
// needs a buffered connection that replays bytes peeked by another owner.
