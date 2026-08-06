// ref: internal/api/protocol_multiplexer_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

// Evidence: internal/api/server.rs::host_style_accept_loop_is_not_blocked_by_idle_connection
// exercises the CTOX accept-then-spawn replacement with a real idle TCP peer
// followed by a bounded GET /v1/models request.
