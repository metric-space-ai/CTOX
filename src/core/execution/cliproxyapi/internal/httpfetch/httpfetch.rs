// ref: internal/httpfetch/httpfetch.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::BTreeMap;
use std::fmt;
use std::future::Future;
use std::pin::Pin;

pub type Headers = BTreeMap<String, String>;
pub type BodyChunkFuture<'a> =
    Pin<Box<dyn Future<Output = Result<Option<Vec<u8>>, String>> + Send + 'a>>;
pub type FetchFuture<'a> = Pin<Box<dyn Future<Output = Result<FetchResponse, String>> + Send + 'a>>;

pub trait ResponseBody: Send {
    fn next_chunk(&mut self) -> BodyChunkFuture<'_>;
}

pub struct FetchResponse {
    pub status: u16,
    pub body: Box<dyn ResponseBody>,
}

pub trait HttpDoer: Send + Sync {
    fn get<'a>(&'a self, request_url: &'a str, headers: &'a Headers) -> FetchFuture<'a>;
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum HttpFetchError {
    CreateRequest(String),
    RequestFailed(String),
    UnexpectedStatus { status: u16, detail: String },
    ReadResponse(String),
    ResponseTooLarge { max_size: usize },
}

impl fmt::Display for HttpFetchError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CreateRequest(error) => write!(formatter, "create request: {error}"),
            Self::RequestFailed(error) => write!(formatter, "request failed: {error}"),
            Self::UnexpectedStatus { status, detail } if detail.is_empty() => {
                write!(formatter, "unexpected status {status}")
            }
            Self::UnexpectedStatus { status, detail } => {
                write!(formatter, "unexpected status {status}: {detail}")
            }
            Self::ReadResponse(error) => write!(formatter, "read response: {error}"),
            Self::ResponseTooLarge { max_size } => {
                write!(
                    formatter,
                    "response exceeds maximum allowed size of {max_size} bytes"
                )
            }
        }
    }
}

impl std::error::Error for HttpFetchError {}

/// Performs a GET, requires a 2xx status and returns the response bytes.
/// `max_size == 0` preserves upstream's unbounded mode; positive limits are
/// enforced while chunks arrive rather than after the full body is buffered.
pub async fn get_bytes(
    client: &dyn HttpDoer,
    request_url: &str,
    headers: &Headers,
    max_size: usize,
) -> Result<Vec<u8>, HttpFetchError> {
    validate_request_url(request_url)?;
    let mut response = client
        .get(request_url, headers)
        .await
        .map_err(HttpFetchError::RequestFailed)?;

    if !(200..300).contains(&response.status) {
        let detail = read_error_detail(&mut *response.body, 4_096).await?;
        return Err(HttpFetchError::UnexpectedStatus {
            status: response.status,
            detail: String::from_utf8_lossy(&detail).trim().to_owned(),
        });
    }

    read_body(&mut *response.body, max_size, max_size > 0).await
}

async fn read_error_detail(
    body: &mut dyn ResponseBody,
    limit: usize,
) -> Result<Vec<u8>, HttpFetchError> {
    let mut data = Vec::new();
    while data.len() < limit {
        let Some(chunk) = body
            .next_chunk()
            .await
            .map_err(HttpFetchError::ReadResponse)?
        else {
            break;
        };
        let remaining = limit - data.len();
        data.extend_from_slice(&chunk[..chunk.len().min(remaining)]);
    }
    Ok(data)
}

fn validate_request_url(request_url: &str) -> Result<(), HttpFetchError> {
    let parsed = url::Url::parse(request_url)
        .map_err(|error| HttpFetchError::CreateRequest(error.to_string()))?;
    if !matches!(parsed.scheme(), "http" | "https")
        || parsed.host().is_none()
        || !parsed.username().is_empty()
        || parsed.password().is_some()
        || parsed.fragment().is_some()
    {
        return Err(HttpFetchError::CreateRequest(
            "URL must be credential-free http(s), contain a host, and omit fragments".to_owned(),
        ));
    }
    Ok(())
}

async fn read_body(
    body: &mut dyn ResponseBody,
    limit: usize,
    bounded: bool,
) -> Result<Vec<u8>, HttpFetchError> {
    let mut data = Vec::new();
    while let Some(chunk) = body
        .next_chunk()
        .await
        .map_err(HttpFetchError::ReadResponse)?
    {
        if bounded && chunk.len() > limit.saturating_sub(data.len()) {
            return Err(HttpFetchError::ResponseTooLarge { max_size: limit });
        }
        data.extend_from_slice(&chunk);
    }
    Ok(data)
}

#[cfg(any(
    feature = "anthropic-fingerprint-transport",
    feature = "codex-http-transport"
))]
mod wreq_adapter {
    use super::*;
    use futures_util::stream::BoxStream;
    use futures_util::StreamExt;

    struct WreqBody {
        chunks: BoxStream<'static, Result<Vec<u8>, String>>,
    }

    impl ResponseBody for WreqBody {
        fn next_chunk(&mut self) -> BodyChunkFuture<'_> {
            Box::pin(async move { self.chunks.next().await.transpose() })
        }
    }

    impl HttpDoer for wreq::Client {
        fn get<'a>(&'a self, request_url: &'a str, headers: &'a Headers) -> FetchFuture<'a> {
            Box::pin(async move {
                let mut request = self.get(request_url);
                for (key, value) in headers {
                    if value.is_empty() {
                        continue;
                    }
                    let name = wreq::header::HeaderName::from_bytes(key.as_bytes())
                        .map_err(|error| format!("invalid header name: {error}"))?;
                    let value = wreq::header::HeaderValue::from_str(value)
                        .map_err(|error| format!("invalid header value: {error}"))?;
                    request = request.header(name, value);
                }
                let response = request.send().await.map_err(|error| error.to_string())?;
                let status = response.status().as_u16();
                let chunks = response
                    .bytes_stream()
                    .map(|chunk| {
                        chunk
                            .map(|bytes| bytes.to_vec())
                            .map_err(|error| error.to_string())
                    })
                    .boxed();
                Ok(FetchResponse {
                    status,
                    body: Box::new(WreqBody { chunks }),
                })
            })
        }
    }
}
