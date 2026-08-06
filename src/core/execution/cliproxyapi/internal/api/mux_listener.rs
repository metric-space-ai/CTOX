// ref: internal/api/mux_listener.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: replaced_by_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

// CTOX replacement: cliproxyapi_host.rs owns one bounded loopback HTTP listener
// and spawns each accepted connection directly into internal/api/server.rs. It
// does not hand connections between synthetic net.Listener queues.
