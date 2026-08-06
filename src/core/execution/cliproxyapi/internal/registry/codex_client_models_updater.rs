// ref: internal/registry/codex_client_models_updater.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::{future::Future, pin::Pin};

use super::{validate_codex_client_models_json, CodexClientModelsError, CodexClientModelsStore};

pub const MAX_CODEX_CLIENT_MODELS_SIZE: usize = 8 << 20;
pub const DEFAULT_CODEX_CLIENT_MODELS_SOURCES: [&str; 2] = [
    "https://raw.githubusercontent.com/router-for-me/models/refs/heads/main/codex_client_models.json",
    "https://models.router-for.me/codex_client_models.json",
];

pub fn default_codex_client_models_sources() -> Vec<String> {
    DEFAULT_CODEX_CLIENT_MODELS_SOURCES
        .iter()
        .map(|source| (*source).to_owned())
        .collect()
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CatalogFetchFailure {
    pub source: String,
    pub reason: String,
}

pub type CodexClientModelsFetchFuture<'a> =
    Pin<Box<dyn Future<Output = Result<Vec<u8>, String>> + Send + 'a>>;

pub trait CodexClientModelsSource: Send + Sync {
    fn fetch<'a>(&'a self, source: &'a str, max_bytes: usize) -> CodexClientModelsFetchFuture<'a>;
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CatalogRefresh {
    pub source: String,
    pub changed: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CatalogRefreshError {
    pub failures: Vec<CatalogFetchFailure>,
}

impl std::fmt::Display for CatalogRefreshError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "Codex client model fetch failed from all {} sources",
            self.failures.len()
        )
    }
}

impl std::error::Error for CatalogRefreshError {}

/// Performs one bounded refresh. CTOX owns recurrence and cancellation; this
/// port deliberately does not recreate upstream's process-global goroutine.
pub async fn refresh_codex_client_models(
    store: &CodexClientModelsStore,
    transport: &dyn CodexClientModelsSource,
    sources: &[String],
) -> Result<CatalogRefresh, CatalogRefreshError> {
    let mut failures = Vec::with_capacity(sources.len());
    for source in sources {
        match fetch_and_validate(transport, source).await {
            Ok(data) => match store.load(&data, source) {
                Ok(changed) => {
                    return Ok(CatalogRefresh {
                        source: source.clone(),
                        changed,
                    })
                }
                Err(error) => failures.push(failure(source, error)),
            },
            Err(error) => failures.push(failure(source, error)),
        }
    }
    Err(CatalogRefreshError { failures })
}

async fn fetch_and_validate(
    transport: &dyn CodexClientModelsSource,
    source: &str,
) -> Result<Vec<u8>, CodexClientModelsError> {
    let data = transport
        .fetch(source, MAX_CODEX_CLIENT_MODELS_SIZE)
        .await
        .map_err(CodexClientModelsError::new)?;
    if data.len() > MAX_CODEX_CLIENT_MODELS_SIZE {
        return Err(CodexClientModelsError::new(format!(
            "catalog exceeded {MAX_CODEX_CLIENT_MODELS_SIZE} bytes"
        )));
    }
    validate_codex_client_models_json(&data)?;
    Ok(data)
}

fn failure(source: &str, error: CodexClientModelsError) -> CatalogFetchFailure {
    CatalogFetchFailure {
        source: source.to_owned(),
        reason: error.to_string(),
    }
}

#[cfg(feature = "codex-http-transport")]
mod http {
    use std::time::Duration;

    use futures_util::StreamExt;
    use wreq::{Client, Proxy};

    use super::{CodexClientModelsFetchFuture, CodexClientModelsSource};

    const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
    const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);

    #[derive(Clone)]
    pub struct WreqCodexClientModelsSource {
        client: Client,
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub enum WreqCatalogSourceBuildError {
        InvalidProxy,
        Client,
    }

    impl WreqCodexClientModelsSource {
        /// Environment proxy discovery is disabled. Any proxy comes from typed
        /// host configuration, matching the rest of the CTOX gateway.
        pub fn new(proxy_url: Option<&str>) -> Result<Self, WreqCatalogSourceBuildError> {
            let mut builder = Client::builder()
                .connect_timeout(CONNECT_TIMEOUT)
                .retry(wreq::retry::Policy::never())
                .redirect(wreq::redirect::Policy::none());
            if let Some(proxy_url) = proxy_url.map(str::trim).filter(|url| !url.is_empty()) {
                builder = builder.proxy(
                    Proxy::all(proxy_url).map_err(|_| WreqCatalogSourceBuildError::InvalidProxy)?,
                );
            } else {
                builder = builder.no_proxy();
            }
            Ok(Self {
                client: builder
                    .build()
                    .map_err(|_| WreqCatalogSourceBuildError::Client)?,
            })
        }
    }

    impl CodexClientModelsSource for WreqCodexClientModelsSource {
        fn fetch<'a>(
            &'a self,
            source: &'a str,
            max_bytes: usize,
        ) -> CodexClientModelsFetchFuture<'a> {
            Box::pin(async move {
                let response = self
                    .client
                    .get(source)
                    .timeout(REQUEST_TIMEOUT)
                    .send()
                    .await
                    .map_err(|error| format!("request failed: {error}"))?;
                if response.status().as_u16() != 200 {
                    return Err(format!("HTTP status {}", response.status().as_u16()));
                }
                let mut output = Vec::new();
                let mut chunks = response.bytes_stream();
                while let Some(chunk) = chunks.next().await {
                    let chunk = chunk.map_err(|error| format!("response read failed: {error}"))?;
                    if output.len().saturating_add(chunk.len()) > max_bytes {
                        return Err(format!("catalog exceeded {max_bytes} bytes"));
                    }
                    output.extend_from_slice(&chunk);
                }
                Ok(output)
            })
        }
    }
}

#[cfg(feature = "codex-http-transport")]
pub use http::{WreqCatalogSourceBuildError, WreqCodexClientModelsSource};
