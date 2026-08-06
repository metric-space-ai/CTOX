// ref: sdk/cliproxy/rtprovider.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

//! Per-auth proxy transport provider.

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard};

use crate::sdk::cliproxy::auth::Auth;
use crate::sdk::proxyutil::{build_http_transport, HttpTransport, ProxyErrorKind};

/// Receives credential-safe proxy build failures. The raw proxy URL is never
/// passed to the sink.
pub trait ProxyBuildFailureSink: Send + Sync {
    fn on_proxy_build_failure(&self, kind: ProxyErrorKind);
}

/// Instance-owned cache of transports, keyed by the trimmed auth proxy value.
pub struct DefaultRoundTripperProvider {
    cache: Mutex<HashMap<String, Arc<HttpTransport>>>,
    failure_sink: Option<Arc<dyn ProxyBuildFailureSink>>,
}

impl fmt::Debug for DefaultRoundTripperProvider {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DefaultRoundTripperProvider")
            .field("cache_entries", &lock_unpoisoned(&self.cache).len())
            .field("has_failure_sink", &self.failure_sink.is_some())
            .finish()
    }
}

#[must_use]
pub fn new_default_round_tripper_provider() -> DefaultRoundTripperProvider {
    DefaultRoundTripperProvider {
        cache: Mutex::new(HashMap::new()),
        failure_sink: None,
    }
}

impl DefaultRoundTripperProvider {
    #[must_use]
    pub fn with_failure_sink(failure_sink: Arc<dyn ProxyBuildFailureSink>) -> Self {
        Self {
            cache: Mutex::new(HashMap::new()),
            failure_sink: Some(failure_sink),
        }
    }

    /// Returns a cached transport for this auth's explicit proxy setting.
    /// Missing/blank settings retain upstream's `nil` result.
    #[must_use]
    pub fn round_tripper_for(&self, auth: Option<&Auth>) -> Option<Arc<HttpTransport>> {
        let proxy = auth?.proxy_url.trim();
        if proxy.is_empty() {
            return None;
        }
        if let Some(transport) = lock_unpoisoned(&self.cache).get(proxy).cloned() {
            return Some(transport);
        }

        let transport = match build_http_transport(proxy) {
            Ok((transport, _)) => Arc::new(transport),
            Err(error) => {
                if let Some(sink) = &self.failure_sink {
                    sink.on_proxy_build_failure(error.kind());
                }
                return None;
            }
        };

        let mut cache = lock_unpoisoned(&self.cache);
        Some(
            cache
                .entry(proxy.to_owned())
                .or_insert_with(|| Arc::clone(&transport))
                .clone(),
        )
    }

    #[cfg(test)]
    pub(super) fn cached_transport_count(&self) -> usize {
        lock_unpoisoned(&self.cache).len()
    }
}

fn lock_unpoisoned<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}
