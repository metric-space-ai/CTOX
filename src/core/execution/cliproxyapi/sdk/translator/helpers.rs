// ref: sdk/translator/helpers.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

//! Rust uses explicit `Registry` receivers instead of duplicating upstream's
//! package-level `*ByFormatName` wrappers. `Format` already owns the identifier.
