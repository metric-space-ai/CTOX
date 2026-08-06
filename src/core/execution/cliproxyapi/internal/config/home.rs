// ref: internal/config/home.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: replaced_by_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

//! The upstream `HomeConfig` describes a second Redis-backed control plane
//! injected through `-home-jwt`. CTOX owns queueing, persistence, policy and
//! runtime state itself, so the Rust gateway deliberately has no Home/Redis
//! configuration surface.
//!
//! The replacement is enforced by `CliproxyRuntimeConfig`'s closed Serde
//! schema: a `home` field is rejected instead of being accepted and silently
//! ignored. Keeping this mirrored module in the active graph makes that
//! architectural disposition explicit without reintroducing the displaced
//! transport or its TLS secret fields.
