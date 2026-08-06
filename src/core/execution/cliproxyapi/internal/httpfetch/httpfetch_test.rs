// ref: internal/httpfetch/httpfetch_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use super::*;
use std::sync::Mutex;

type FakeChunks = Vec<Result<Vec<u8>, String>>;

struct FakeBody {
    chunks: std::collections::VecDeque<Result<Vec<u8>, String>>,
}

impl ResponseBody for FakeBody {
    fn next_chunk(&mut self) -> BodyChunkFuture<'_> {
        Box::pin(async move { self.chunks.pop_front().transpose() })
    }
}

struct FakeClient {
    status: u16,
    chunks: Mutex<Option<FakeChunks>>,
    expected_headers: Headers,
}

impl FakeClient {
    fn response(status: u16, chunks: Vec<&[u8]>) -> Self {
        Self {
            status,
            chunks: Mutex::new(Some(
                chunks.into_iter().map(|chunk| Ok(chunk.to_vec())).collect(),
            )),
            expected_headers: Headers::new(),
        }
    }
}

impl HttpDoer for FakeClient {
    fn get<'a>(&'a self, _request_url: &'a str, headers: &'a Headers) -> FetchFuture<'a> {
        Box::pin(async move {
            assert_eq!(headers, &self.expected_headers);
            let chunks = self
                .chunks
                .lock()
                .unwrap()
                .take()
                .expect("fake response consumed once");
            Ok(FetchResponse {
                status: self.status,
                body: Box::new(FakeBody {
                    chunks: chunks.into(),
                }),
            })
        })
    }
}

#[tokio::test]
async fn get_bytes_returns_body_and_sends_headers() {
    let mut headers = Headers::new();
    headers.insert("User-Agent".to_owned(), "agent".to_owned());
    headers.insert("Accept".to_owned(), "application/json".to_owned());
    let client = FakeClient {
        expected_headers: headers.clone(),
        ..FakeClient::response(200, vec![b"pay", b"load"])
    };

    let data = get_bytes(&client, "https://example.test/data", &headers, 0)
        .await
        .unwrap();

    assert_eq!(data, b"payload");
}

#[tokio::test]
async fn get_bytes_rejects_error_status_with_bounded_detail() {
    let client = FakeClient::response(404, vec![b" missing \n", &[b'x'; 5_000]]);

    let error = get_bytes(&client, "https://example.test/missing", &Headers::new(), 0)
        .await
        .unwrap_err();

    assert!(error.to_string().contains("unexpected status 404"));
    assert!(matches!(
        error,
        HttpFetchError::UnexpectedStatus { status: 404, .. }
    ));
}

#[tokio::test]
async fn get_bytes_enforces_max_size_before_buffering_later_chunks() {
    let client = FakeClient::response(200, vec![b"0123", b"456789"]);

    let error = get_bytes(&client, "https://example.test/large", &Headers::new(), 4)
        .await
        .unwrap_err();

    assert_eq!(error, HttpFetchError::ResponseTooLarge { max_size: 4 });
    assert!(error.to_string().contains("maximum allowed size"));
}

#[tokio::test]
async fn malformed_url_fails_before_transport() {
    let client = FakeClient::response(200, vec![b"unreachable"]);

    let error = get_bytes(&client, "not a URL", &Headers::new(), 0)
        .await
        .unwrap_err();

    assert!(matches!(error, HttpFetchError::CreateRequest(_)));
    assert!(client.chunks.lock().unwrap().is_some());
}

#[tokio::test]
async fn credentialed_or_fragmented_url_fails_before_transport() {
    for url in [
        "https://user:secret@example.test/data",
        "https://example.test/data#local-fragment",
    ] {
        let client = FakeClient::response(200, vec![b"unreachable"]);

        let error = get_bytes(&client, url, &Headers::new(), 0)
            .await
            .unwrap_err();

        assert!(matches!(error, HttpFetchError::CreateRequest(_)));
        assert!(client.chunks.lock().unwrap().is_some());
    }
}

#[tokio::test]
async fn read_failure_is_classified() {
    let client = FakeClient {
        status: 200,
        chunks: Mutex::new(Some(vec![Err("stream reset".to_owned())])),
        expected_headers: Headers::new(),
    };

    let error = get_bytes(&client, "https://example.test/reset", &Headers::new(), 0)
        .await
        .unwrap_err();

    assert_eq!(
        error,
        HttpFetchError::ReadResponse("stream reset".to_owned())
    );
}
