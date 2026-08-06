// ref: internal/runtime/executor/helps/antigravity_grounding_urls.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::HashMap;
use std::fmt;

use serde_json::Value;
use url::Url;

const VERTEX_SEARCH_HOST: &str = "vertexaisearch.cloud.google.com";
const VERTEX_SEARCH_PATH_PREFIX: &str = "/grounding-api-redirect/";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GroundingRedirectResponse {
    pub status: u16,
    pub location: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GroundingRedirectError {
    Transport,
}

impl fmt::Display for GroundingRedirectError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("grounding redirect lookup failed")
    }
}

impl std::error::Error for GroundingRedirectError {}

/// Host-injected HEAD transport. Implementations must not follow redirects;
/// credentials, proxy policy and cancellation stay owned by the selected CTOX
/// execution lane.
pub trait GroundingRedirectTransport: Send + Sync {
    fn head(&self, url: &Url) -> Result<GroundingRedirectResponse, GroundingRedirectError>;
}

#[must_use]
pub fn is_antigravity_vertex_search_redirect(raw_url: &str) -> bool {
    Url::parse(raw_url).is_ok_and(|parsed| {
        parsed.scheme() == "https"
            && parsed.host_str() == Some(VERTEX_SEARCH_HOST)
            && parsed.port().is_none()
            && parsed.username().is_empty()
            && parsed.password().is_none()
            && parsed.path().starts_with(VERTEX_SEARCH_PATH_PREFIX)
    })
}

/// Replaces Vertex Search redirect URLs in grounding chunks with their HTTPS
/// targets. Failed or invalid lookups are best-effort and retain the original
/// payload. A no-op path is byte-identical.
#[must_use]
pub fn resolve_antigravity_grounding_urls(
    transport: &dyn GroundingRedirectTransport,
    payload: &[u8],
) -> Vec<u8> {
    if payload.is_empty() {
        return payload.to_vec();
    }
    let Ok(mut root) = serde_json::from_slice::<Value>(payload) else {
        return payload.to_vec();
    };
    let chunks_pointer = if root
        .pointer("/response/candidates/0/groundingMetadata/groundingChunks")
        .is_some_and(Value::is_array)
    {
        "/response/candidates/0/groundingMetadata/groundingChunks"
    } else if root
        .pointer("/candidates/0/groundingMetadata/groundingChunks")
        .is_some_and(Value::is_array)
    {
        "/candidates/0/groundingMetadata/groundingChunks"
    } else {
        return payload.to_vec();
    };
    let chunks = root
        .pointer_mut(chunks_pointer)
        .and_then(Value::as_array_mut)
        .expect("array pointer was validated");
    let mut resolved = HashMap::<String, String>::new();
    let mut changed = false;
    for chunk in chunks {
        let Some(uri) = chunk
            .pointer("/web/uri")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|uri| !uri.is_empty())
            .map(ToOwned::to_owned)
        else {
            continue;
        };
        let resolved_uri = resolved
            .entry(uri.clone())
            .or_insert_with(|| resolve_antigravity_grounding_url(transport, &uri));
        if resolved_uri == &uri {
            continue;
        }
        if let Some(slot) = chunk.pointer_mut("/web/uri") {
            *slot = Value::String(resolved_uri.clone());
            changed = true;
        }
    }
    if !changed {
        return payload.to_vec();
    }
    serde_json::to_vec(&root).unwrap_or_else(|_| payload.to_vec())
}

fn resolve_antigravity_grounding_url(
    transport: &dyn GroundingRedirectTransport,
    raw_url: &str,
) -> String {
    if !is_antigravity_vertex_search_redirect(raw_url) {
        return raw_url.to_owned();
    }
    let Ok(url) = Url::parse(raw_url) else {
        return raw_url.to_owned();
    };
    let Ok(response) = transport.head(&url) else {
        return raw_url.to_owned();
    };
    if !(300..400).contains(&response.status) {
        return raw_url.to_owned();
    }
    let Some(location) = response.location.as_deref().map(str::trim) else {
        return raw_url.to_owned();
    };
    let Ok(parsed) = Url::parse(location) else {
        return raw_url.to_owned();
    };
    if parsed.scheme() != "https"
        || parsed.host_str().is_none()
        || !parsed.username().is_empty()
        || parsed.password().is_some()
    {
        return raw_url.to_owned();
    }
    location.to_owned()
}
