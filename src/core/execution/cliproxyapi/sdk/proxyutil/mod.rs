// ref: sdk/proxyutil/proxy.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

mod proxy;

#[cfg(test)]
mod proxy_test;

pub use proxy::*;
