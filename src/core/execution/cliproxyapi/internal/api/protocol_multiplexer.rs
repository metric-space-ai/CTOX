// ref: internal/api/protocol_multiplexer.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: replaced_by_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

// CTOX replacement: the provider gateway is a loopback HTTP control surface.
// Durable queue/persistence ownership stays in the CTOX daemon and is never
// exposed as upstream Redis RESP on the subscription HTTP port. The host accept
// loop spawns before reading, preserving the upstream idle-client liveness fix.
