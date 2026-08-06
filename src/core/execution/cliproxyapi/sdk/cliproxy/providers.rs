// ref: sdk/cliproxy/providers.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

//! Default token/API-key client loaders.
//!
//! The Go implementation reaches into the package-global watcher to build API
//! key clients from a mutable config pointer. CTOX injects the already-bound,
//! typed builder instead. This keeps credentials and runtime authority out of
//! globals and ambient environment while retaining the upstream result and
//! post-build cancellation semantics.

use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Cooperative load cancellation owned by the caller.
#[derive(Clone, Debug, Default)]
pub struct LoadContext {
    cancelled: Arc<AtomicBool>,
}

impl LoadContext {
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }

    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }
}

/// Number of clients successfully created from persisted tokens.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct TokenClientResult {
    pub successful_authed: usize,
}

/// Per-provider counts produced by API-key client construction.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ApiKeyClientResult {
    pub gemini_key_count: usize,
    pub vertex_compat_key_count: usize,
    pub claude_key_count: usize,
    pub codex_key_count: usize,
    pub xai_key_count: usize,
    pub openai_compat_count: usize,
}

/// Credential-safe client-build failure category.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ClientProviderError {
    Cancelled,
    BuildFailed,
}

impl fmt::Display for ClientProviderError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Cancelled => "client loading cancelled",
            Self::BuildFailed => "API-key client construction failed",
        })
    }
}

impl std::error::Error for ClientProviderError {}

/// Injected replacement for `watcher.BuildAPIKeyClients`.
pub trait ApiKeyClientBuilder: Send + Sync {
    fn build_api_key_clients(&self) -> Result<ApiKeyClientResult, ClientProviderError>;
}

/// Stateless token-backed loader. Upstream executors own token handling, so
/// loading always succeeds with zero eagerly authenticated clients.
#[derive(Clone, Copy, Debug, Default)]
pub struct FileTokenClientProvider;

#[must_use]
pub const fn new_file_token_client_provider() -> FileTokenClientProvider {
    FileTokenClientProvider
}

impl FileTokenClientProvider {
    pub fn load(&self, _context: &LoadContext) -> TokenClientResult {
        TokenClientResult {
            successful_authed: 0,
        }
    }
}

/// API-key loader bound to one caller-owned builder/configuration snapshot.
#[derive(Clone)]
pub struct ConfiguredApiKeyClientProvider {
    builder: Arc<dyn ApiKeyClientBuilder>,
}

impl fmt::Debug for ConfiguredApiKeyClientProvider {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ConfiguredApiKeyClientProvider")
            .field("has_builder", &true)
            .finish()
    }
}

#[must_use]
pub fn new_api_key_client_provider(
    builder: Arc<dyn ApiKeyClientBuilder>,
) -> ConfiguredApiKeyClientProvider {
    ConfiguredApiKeyClientProvider { builder }
}

impl ConfiguredApiKeyClientProvider {
    pub fn load(&self, context: &LoadContext) -> Result<ApiKeyClientResult, ClientProviderError> {
        let result = self.builder.build_api_key_clients()?;
        if context.is_cancelled() {
            return Err(ClientProviderError::Cancelled);
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    struct Builder {
        calls: AtomicUsize,
        result: ApiKeyClientResult,
    }

    impl ApiKeyClientBuilder for Builder {
        fn build_api_key_clients(&self) -> Result<ApiKeyClientResult, ClientProviderError> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            Ok(self.result)
        }
    }

    fn counts() -> ApiKeyClientResult {
        ApiKeyClientResult {
            gemini_key_count: 1,
            vertex_compat_key_count: 2,
            claude_key_count: 3,
            codex_key_count: 4,
            xai_key_count: 5,
            openai_compat_count: 6,
        }
    }

    #[test]
    fn token_provider_matches_stateless_upstream_result_even_when_cancelled() {
        let context = LoadContext::default();
        context.cancel();
        assert_eq!(
            new_file_token_client_provider().load(&context),
            TokenClientResult {
                successful_authed: 0
            }
        );
    }

    #[test]
    fn api_key_provider_returns_all_builder_counts() {
        let builder = Arc::new(Builder {
            calls: AtomicUsize::new(0),
            result: counts(),
        });
        let provider = new_api_key_client_provider(builder.clone());

        assert_eq!(provider.load(&LoadContext::default()), Ok(counts()));
        assert_eq!(builder.calls.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn api_key_provider_checks_cancellation_after_client_build() {
        let builder = Arc::new(Builder {
            calls: AtomicUsize::new(0),
            result: counts(),
        });
        let provider = new_api_key_client_provider(builder.clone());
        let context = LoadContext::default();
        context.cancel();

        assert_eq!(provider.load(&context), Err(ClientProviderError::Cancelled));
        assert_eq!(builder.calls.load(Ordering::Relaxed), 1);
    }
}
