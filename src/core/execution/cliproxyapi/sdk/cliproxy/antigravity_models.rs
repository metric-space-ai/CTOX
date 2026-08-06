// ref: sdk/cliproxy/antigravity_models.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::BTreeSet;
use std::fmt;
use std::sync::RwLock;
use std::time::Duration;

use serde::Deserialize;

use crate::internal::auth::antigravity::SecretString;
use crate::internal::runtime::executor::{
    AntigravityGenerateRequest, AntigravityGenerateTransport, AntigravityUpstreamTarget,
};

pub const ANTIGRAVITY_MODEL_BASE_URL_DAILY: &str = "https://daily-cloudcode-pa.googleapis.com";
pub const ANTIGRAVITY_MODEL_BASE_URL_PROD: &str = "https://cloudcode-pa.googleapis.com";
pub const MAX_ANTIGRAVITY_MODEL_RESPONSE_BYTES: usize = 1024 * 1024;

/// Provider-discovered Antigravity capabilities. The upstream response also
/// contains a `models` object, but registration intentionally annotates only
/// models already present in the trusted static/runtime catalog.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct AntigravityModelCapabilityHints {
    web_search_model_ids: BTreeSet<String>,
}

impl AntigravityModelCapabilityHints {
    pub fn supports_web_search(&self, model_id: &str) -> bool {
        let model_id = normalize_antigravity_capability_model_id(model_id);
        !model_id.is_empty() && self.web_search_model_ids.contains(&model_id)
    }

    pub fn web_search_model_ids(&self) -> impl Iterator<Item = &str> {
        self.web_search_model_ids.iter().map(String::as_str)
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct FetchAvailableModelsResponse {
    #[serde(default)]
    web_search_model_ids: Vec<String>,
}

pub fn parse_antigravity_model_capability_hints(
    body: &[u8],
) -> Result<AntigravityModelCapabilityHints, AntigravityModelCapabilityError> {
    let parsed: FetchAvailableModelsResponse = serde_json::from_slice(body)
        .map_err(|_| AntigravityModelCapabilityError::InvalidResponse)?;
    Ok(AntigravityModelCapabilityHints {
        web_search_model_ids: parsed
            .web_search_model_ids
            .into_iter()
            .map(|model_id| normalize_antigravity_capability_model_id(&model_id))
            .filter(|model_id| !model_id.is_empty())
            .collect(),
    })
}

/// Builds the exact upstream candidate order: one configured account URL, or
/// daily followed by production when the account has no override.
pub fn antigravity_model_discovery_targets(
    configured_base_url: Option<&str>,
) -> Result<Vec<AntigravityUpstreamTarget>, AntigravityModelCapabilityError> {
    let urls = configured_base_url
        .map(str::trim)
        .filter(|url| !url.is_empty())
        .map_or_else(
            || {
                vec![
                    ANTIGRAVITY_MODEL_BASE_URL_DAILY,
                    ANTIGRAVITY_MODEL_BASE_URL_PROD,
                ]
            },
            |url| vec![url],
        );
    urls.into_iter()
        .map(|url| {
            AntigravityUpstreamTarget::new(url)
                .map_err(|_| AntigravityModelCapabilityError::InvalidTarget)
        })
        .collect()
}

/// Executes bounded authenticated discovery and replaces the catalog only
/// from the first successful, non-empty capability response. Transport,
/// status, parse and empty-hint failures fall through to the next upstream.
pub async fn refresh_antigravity_model_capability_catalog(
    transport: &dyn AntigravityGenerateTransport,
    targets: &[AntigravityUpstreamTarget],
    access_token: &SecretString,
    available_model_ids: &[String],
    catalog: &AntigravityModelCapabilityCatalog,
    timeout: Duration,
) -> Result<(), AntigravityModelCapabilityError> {
    if targets.is_empty() || timeout.is_zero() {
        catalog.clear();
        return Err(AntigravityModelCapabilityError::Unavailable);
    }
    for target in targets {
        let request = AntigravityGenerateRequest::new_model_discovery(target, access_token.clone());
        let Ok(response) = transport.execute(&request, timeout).await else {
            continue;
        };
        if !(200..300).contains(&response.status())
            || response.body().len() > MAX_ANTIGRAVITY_MODEL_RESPONSE_BYTES
        {
            continue;
        }
        let Ok(hints) = parse_antigravity_model_capability_hints(response.body()) else {
            continue;
        };
        if hints.web_search_model_ids.is_empty() {
            continue;
        }
        catalog.replace_from_response(
            available_model_ids.iter().map(String::as_str),
            response.body(),
        )?;
        return Ok(());
    }
    catalog.clear();
    Err(AntigravityModelCapabilityError::Unavailable)
}

/// Atomic, process-local projection of the last successful capability fetch.
/// CTOX keeps the mutable discovery snapshot separate from the translator and
/// injects reads through its typed registration resolver.
#[derive(Default)]
pub struct AntigravityModelCapabilityCatalog {
    web_search_model_ids: RwLock<BTreeSet<String>>,
}

impl AntigravityModelCapabilityCatalog {
    pub fn new() -> Self {
        Self::default()
    }

    /// Replaces the snapshot, intersecting provider hints with the models that
    /// the authenticated Antigravity runtime actually exposes. Fetched-only
    /// model IDs never create routes.
    pub fn replace_from_response<'a>(
        &self,
        available_model_ids: impl IntoIterator<Item = &'a str>,
        body: &[u8],
    ) -> Result<(), AntigravityModelCapabilityError> {
        let hints = match parse_antigravity_model_capability_hints(body) {
            Ok(hints) => hints,
            Err(error) => {
                self.clear();
                return Err(error);
            }
        };
        let available = available_model_ids
            .into_iter()
            .map(normalize_antigravity_capability_model_id)
            .filter(|model_id| !model_id.is_empty())
            .collect::<BTreeSet<_>>();
        let next = hints
            .web_search_model_ids
            .intersection(&available)
            .cloned()
            .collect();
        *self
            .web_search_model_ids
            .write()
            .expect("Antigravity capability catalog poisoned") = next;
        Ok(())
    }

    pub fn supports_web_search(&self, model_id: &str) -> bool {
        let model_id = normalize_antigravity_capability_model_id(model_id);
        !model_id.is_empty()
            && self
                .web_search_model_ids
                .read()
                .expect("Antigravity capability catalog poisoned")
                .contains(&model_id)
    }

    pub fn clear(&self) {
        self.web_search_model_ids
            .write()
            .expect("Antigravity capability catalog poisoned")
            .clear();
    }
}

pub fn normalize_antigravity_capability_model_id(model_id: &str) -> String {
    let mut normalized = model_id.trim().to_ascii_lowercase();
    if let Some(open) = normalized.rfind('(') {
        if normalized.ends_with(')') {
            normalized.truncate(open);
            normalized = normalized.trim().to_owned();
        }
    }
    normalized
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AntigravityModelCapabilityError {
    InvalidResponse,
    InvalidTarget,
    Unavailable,
}

impl fmt::Display for AntigravityModelCapabilityError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::InvalidResponse => "Antigravity model capability response is invalid",
            Self::InvalidTarget => "Antigravity model capability target is invalid",
            Self::Unavailable => "Antigravity model capability discovery is unavailable",
        })
    }
}

impl std::error::Error for AntigravityModelCapabilityError {}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::future::Future;
    use std::pin::Pin;
    use std::sync::Arc;
    use std::sync::Mutex;

    use super::*;
    use crate::internal::translator::antigravity::claude::{
        register_claude_antigravity_with_capability_resolver, AntigravityClaudeRequestCapabilities,
    };
    use crate::sdk::translator::{antigravity, claude, Registry, TranslationContext};

    use crate::internal::runtime::executor::{
        AntigravityGenerateResponse, AntigravityGenerateTransportFailure,
    };

    struct SequenceTransport {
        responses: Mutex<
            VecDeque<Result<AntigravityGenerateResponse, AntigravityGenerateTransportFailure>>,
        >,
        urls: Mutex<Vec<String>>,
    }

    impl AntigravityGenerateTransport for SequenceTransport {
        fn execute<'a>(
            &'a self,
            request: &'a AntigravityGenerateRequest,
            _timeout: Duration,
        ) -> Pin<
            Box<
                dyn Future<
                        Output = Result<
                            AntigravityGenerateResponse,
                            AntigravityGenerateTransportFailure,
                        >,
                    > + Send
                    + 'a,
            >,
        > {
            self.urls.lock().unwrap().push(request.url().to_owned());
            let response = self.responses.lock().unwrap().pop_front().unwrap();
            Box::pin(async move { response })
        }
    }

    #[test]
    fn parser_normalizes_deduplicates_and_ignores_unrelated_response_fields() {
        let hints = parse_antigravity_model_capability_hints(
            br#"{"models":{"ignored":{}},"webSearchModelIds":[" Gemini-Web ","gemini-web","", "GEMINI-OTHER"]}"#,
        )
        .unwrap();
        assert_eq!(
            hints.web_search_model_ids().collect::<Vec<_>>(),
            vec!["gemini-other", "gemini-web"]
        );
        assert!(hints.supports_web_search("gemini-web(high)"));
        assert!(!hints.supports_web_search("unknown"));
    }

    #[test]
    fn catalog_intersects_known_models_and_malformed_refresh_clears_stale_state() {
        let catalog = AntigravityModelCapabilityCatalog::new();
        catalog
            .replace_from_response(
                ["gemini-web", "gemini-agent"],
                br#"{"webSearchModelIds":["gemini-web","fetched-only"]}"#,
            )
            .unwrap();
        assert!(catalog.supports_web_search("gemini-web(low)"));
        assert!(!catalog.supports_web_search("fetched-only"));
        assert_eq!(
            catalog.replace_from_response(["gemini-web"], b"not-json"),
            Err(AntigravityModelCapabilityError::InvalidResponse)
        );
        assert!(!catalog.supports_web_search("gemini-web"));
    }

    #[test]
    fn catalog_drives_the_registered_exact_model_request_gate() {
        let catalog = Arc::new(AntigravityModelCapabilityCatalog::new());
        catalog
            .replace_from_response(
                ["gemini-web", "gemini-agent"],
                br#"{"webSearchModelIds":["gemini-web"]}"#,
            )
            .unwrap();
        let registry = Registry::new();
        let resolver_catalog = Arc::clone(&catalog);
        register_claude_antigravity_with_capability_resolver(
            &registry,
            Arc::new(move |model| AntigravityClaudeRequestCapabilities {
                native_google_search: resolver_catalog.supports_web_search(model),
            }),
        );
        let request = br#"{"messages":[{"role":"user","content":"weather"}],"tools":[{"type":"web_search_20250305","name":"web_search"}]}"#;
        let output = registry.translate_request(
            &TranslationContext::default(),
            &claude(),
            &antigravity(),
            "gemini-web(high)",
            request,
            false,
        );
        let output: serde_json::Value = serde_json::from_slice(&output).unwrap();
        assert_eq!(output["requestType"], "web_search");
        assert_eq!(output["model"], "gemini-web(high)");
    }

    #[tokio::test]
    async fn refresh_falls_through_and_uses_only_known_models() {
        let transport = SequenceTransport {
            responses: Mutex::new(VecDeque::from([
                Ok(AntigravityGenerateResponse::new(503, None, Vec::new())),
                Ok(AntigravityGenerateResponse::new(
                    200,
                    None,
                    br#"{"webSearchModelIds":["known","fetched-only"]}"#.to_vec(),
                )),
            ])),
            urls: Mutex::new(Vec::new()),
        };
        let targets = antigravity_model_discovery_targets(None).unwrap();
        let catalog = AntigravityModelCapabilityCatalog::new();
        refresh_antigravity_model_capability_catalog(
            &transport,
            &targets,
            &SecretString::new("access-token").unwrap(),
            &["known".to_owned()],
            &catalog,
            Duration::from_secs(3),
        )
        .await
        .unwrap();
        assert!(catalog.supports_web_search("known"));
        assert!(!catalog.supports_web_search("fetched-only"));
        assert_eq!(
            transport.urls.lock().unwrap().as_slice(),
            [
                "https://daily-cloudcode-pa.googleapis.com/v1internal:fetchAvailableModels",
                "https://cloudcode-pa.googleapis.com/v1internal:fetchAvailableModels"
            ]
        );
    }

    #[tokio::test]
    async fn exhausted_or_invalid_discovery_clears_stale_authority() {
        let catalog = AntigravityModelCapabilityCatalog::new();
        catalog
            .replace_from_response(["known"], br#"{"webSearchModelIds":["known"]}"#)
            .unwrap();
        let transport = SequenceTransport {
            responses: Mutex::new(VecDeque::from([Err(
                AntigravityGenerateTransportFailure::Timeout,
            )])),
            urls: Mutex::new(Vec::new()),
        };
        let targets =
            antigravity_model_discovery_targets(Some("https://example.com/base")).unwrap();
        assert_eq!(
            refresh_antigravity_model_capability_catalog(
                &transport,
                &targets,
                &SecretString::new("access-token").unwrap(),
                &["known".to_owned()],
                &catalog,
                Duration::from_secs(3),
            )
            .await,
            Err(AntigravityModelCapabilityError::Unavailable)
        );
        assert!(!catalog.supports_web_search("known"));
        assert_eq!(
            antigravity_model_discovery_targets(Some("https://user:pass@example.com")),
            Err(AntigravityModelCapabilityError::InvalidTarget)
        );
    }

    #[cfg(feature = "antigravity-http-transport")]
    #[tokio::test]
    async fn native_transport_posts_exact_authenticated_discovery_request() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio::net::TcpListener;
        use tokio::sync::oneshot;

        use crate::internal::runtime::executor::AntigravityGenerateHttpTransport;

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let (captured_sender, captured_receiver) = oneshot::channel();
        tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut request = vec![0_u8; 4096];
            let count = socket.read(&mut request).await.unwrap();
            request.truncate(count);
            captured_sender.send(request).unwrap();
            let body = br#"{"webSearchModelIds":["known"]}"#;
            socket
                .write_all(
                    format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                        body.len()
                    )
                    .as_bytes(),
                )
                .await
                .unwrap();
            socket.write_all(body).await.unwrap();
        });
        let target = AntigravityUpstreamTarget::new(format!("http://{address}")).unwrap();
        let catalog = AntigravityModelCapabilityCatalog::new();
        refresh_antigravity_model_capability_catalog(
            &AntigravityGenerateHttpTransport::new(None).unwrap(),
            &[target],
            &SecretString::new("secret-access").unwrap(),
            &["known".to_owned()],
            &catalog,
            Duration::from_secs(3),
        )
        .await
        .unwrap();
        let request = String::from_utf8_lossy(&captured_receiver.await.unwrap()).into_owned();
        let lower = request.to_ascii_lowercase();
        assert!(request.starts_with("POST /v1internal:fetchAvailableModels HTTP/1.1"));
        assert!(lower.contains("authorization: bearer secret-access"));
        assert!(lower.contains("content-type: application/json"));
        assert!(lower.contains("user-agent: antigravity/hub/2.2.1 darwin/arm64"));
        assert!(request.ends_with("{}"));
        assert!(catalog.supports_web_search("known"));
    }
}
