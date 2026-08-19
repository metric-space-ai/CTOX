// Origin: CTOX
// License: AGPL-3.0-only

//! The bounded loopback intake/outtake surface for Workjet mailbox envelopes.
//!
//! These three routes ride on the SAME listener as the Business OS MCP channel
//! (`mcp_channel::serve_mcp_channel`, bound to 127.0.0.1:8788 by the service),
//! so there is no second port, no second bind policy and no second auth story.
//!
//! Authentication reuses the channel's existing non-public mechanism verbatim:
//! `mcp_channel::mcp_request_authorized`, an `Authorization: Bearer <token>`
//! check in constant time (HMAC-SHA256 comparison) against the per-instance
//! secret `business_os/mcp_inbound_auth_token`. `/health` is the only
//! unauthenticated route on this listener and it stays that way — all three
//! mailbox routes are authenticated, exactly like `POST /mcp`.
//!
//! On top of that the handler enforces loopback itself: a request whose peer
//! address is not 127.0.0.0/8 or ::1 is refused before the body is read, so a
//! misconfigured bind address cannot expose the mailbox to the network.

use std::io::Read;
use std::net::IpAddr;
use std::path::Path;

use anyhow::Context;
use serde_json::json;
use serde_json::Value;
use tiny_http::Header;
use tiny_http::Method;
use tiny_http::Request;
use tiny_http::Response;

use super::mcp_channel::mcp_request_authorized;
use super::workjet_mailbox;

const MAILBOX_PATH_PREFIX: &str = "/workjet/mailbox/";
const PUBLISH_PATH: &str = "/workjet/mailbox/publish";
const PENDING_PATH: &str = "/workjet/mailbox/pending";
const CONSUMED_PATH: &str = "/workjet/mailbox/consumed";

/// Request bodies are bounded before parsing: an unbounded `read_to_string`
/// on a loopback socket is a trivial memory-exhaustion lever.
const MAX_REQUEST_BODY_BYTES: u64 = (workjet_mailbox::MAX_DOCUMENT_BYTES as u64) * 2;

/// Whether `path` belongs to the mailbox surface. Used by the MCP channel's
/// route table to delegate without knowing any mailbox detail.
pub(super) fn is_mailbox_path(path: &str) -> bool {
    path.starts_with(MAILBOX_PATH_PREFIX)
}

/// Answers one mailbox request. Always consumes `request` and always responds.
pub(super) fn handle_mailbox_request(
    root: &Path,
    method: &Method,
    path: &str,
    mut request: Request,
) -> anyhow::Result<()> {
    if !request_is_loopback(&request) {
        return respond(
            request,
            403,
            json!({
                "ok": false,
                "error": "forbidden",
                "message": "The Workjet mailbox surface accepts loopback callers only.",
            }),
        );
    }
    if !mcp_request_authorized(root, &request) {
        return respond(
            request,
            401,
            json!({
                "ok": false,
                "error": "unauthorized",
                "message": "The Workjet mailbox surface requires the same Authorization: Bearer \
                            token as the Business OS MCP endpoint (secret \
                            business_os/mcp_inbound_auth_token).",
            }),
        );
    }

    match (method, path) {
        (Method::Post, PUBLISH_PATH) => {
            let body = match read_bounded_json(&mut request) {
                Ok(body) => body,
                Err(error) => return respond_invalid(request, error),
            };
            match workjet_mailbox::publish_envelope(root, &body) {
                Ok(result) => respond(request, 200, result),
                Err(error) => respond_invalid(request, error),
            }
        }
        (Method::Get, PENDING_PATH) => {
            let query = query_pairs(request.url());
            let environment_id = query
                .iter()
                .find(|(key, _)| key == "environment_id")
                .map(|(_, value)| value.clone())
                .unwrap_or_default();
            let after = query
                .iter()
                .find(|(key, _)| key == "after")
                .map(|(_, value)| value.clone());
            let limit = match query.iter().find(|(key, _)| key == "limit") {
                Some((_, value)) => match value.parse::<usize>() {
                    Ok(limit) => Some(limit),
                    Err(_) => {
                        return respond_invalid(
                            request,
                            anyhow::anyhow!("`limit` must be a positive integer"),
                        )
                    }
                },
                None => None,
            };
            match workjet_mailbox::pending_envelopes(root, &environment_id, after.as_deref(), limit)
            {
                Ok(result) => respond(request, 200, result),
                Err(error) => respond_invalid(request, error),
            }
        }
        (Method::Post, CONSUMED_PATH) => {
            let body = match read_bounded_json(&mut request) {
                Ok(body) => body,
                Err(error) => return respond_invalid(request, error),
            };
            let environment_id = body
                .get("environment_id")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .trim()
                .to_string();
            let envelope_ids = match body.get("envelope_ids").and_then(Value::as_array) {
                Some(entries) => entries
                    .iter()
                    .filter_map(Value::as_str)
                    .map(str::to_string)
                    .collect::<Vec<_>>(),
                None => {
                    return respond_invalid(
                        request,
                        anyhow::anyhow!("`envelope_ids` must be an array of envelope ids"),
                    )
                }
            };
            match workjet_mailbox::mark_consumed(root, &environment_id, &envelope_ids) {
                Ok(result) => respond(request, 200, result),
                Err(error) => respond_invalid(request, error),
            }
        }
        _ => respond(
            request,
            404,
            json!({ "ok": false, "error": "not_found", "path": path }),
        ),
    }
}

/// tiny_http reports the peer address; `None` (a unix socket or a test-built
/// request) is treated as loopback because it cannot have come off the network.
fn request_is_loopback(request: &Request) -> bool {
    match request.remote_addr() {
        Some(addr) => match addr.ip() {
            IpAddr::V4(ip) => ip.is_loopback(),
            IpAddr::V6(ip) => ip.is_loopback(),
        },
        None => true,
    }
}

fn read_bounded_json(request: &mut Request) -> anyhow::Result<Value> {
    if let Some(length) = request.body_length() {
        if length as u64 > MAX_REQUEST_BODY_BYTES {
            anyhow::bail!("request body exceeds the {MAX_REQUEST_BODY_BYTES} byte ceiling");
        }
    }
    let mut body = String::new();
    request
        .as_reader()
        .take(MAX_REQUEST_BODY_BYTES + 1)
        .read_to_string(&mut body)?;
    if body.len() as u64 > MAX_REQUEST_BODY_BYTES {
        anyhow::bail!("request body exceeds the {MAX_REQUEST_BODY_BYTES} byte ceiling");
    }
    if body.trim().is_empty() {
        anyhow::bail!("a JSON request body is required");
    }
    serde_json::from_str(&body).context("invalid JSON request body")
}

/// Minimal `application/x-www-form-urlencoded` query parsing. The mailbox only
/// ever reads bounded ids, a cursor and an integer, so a dependency-free
/// percent-decoder is enough and keeps the surface auditable.
fn query_pairs(url: &str) -> Vec<(String, String)> {
    let Some(query) = url.split_once('?').map(|(_, query)| query) else {
        return Vec::new();
    };
    query
        .split('&')
        .filter(|pair| !pair.is_empty())
        .map(|pair| {
            let (key, value) = pair.split_once('=').unwrap_or((pair, ""));
            (percent_decode(key), percent_decode(value))
        })
        .collect()
}

fn percent_decode(raw: &str) -> String {
    let bytes = raw.replace('+', " ").into_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'%' && index + 2 < bytes.len() {
            let hex = std::str::from_utf8(&bytes[index + 1..index + 3]).unwrap_or("");
            if let Ok(byte) = u8::from_str_radix(hex, 16) {
                out.push(byte);
                index += 3;
                continue;
            }
        }
        out.push(bytes[index]);
        index += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn respond_invalid(request: Request, error: anyhow::Error) -> anyhow::Result<()> {
    respond(
        request,
        400,
        json!({
            "ok": false,
            "error": "invalid_request",
            "message": error.to_string(),
        }),
    )
}

fn respond(request: Request, status: u16, value: Value) -> anyhow::Result<()> {
    let body = serde_json::to_string_pretty(&value)?;
    let mut response = Response::from_string(body).with_status_code(status);
    response.add_header(
        Header::from_bytes(
            &b"Content-Type"[..],
            &b"application/json; charset=utf-8"[..],
        )
        .map_err(|_| anyhow::anyhow!("failed to build content-type header"))?,
    );
    // No CORS allow-origin, for the same reason the MCP endpoint omits it: a
    // page in a local browser must not be able to read mailbox responses.
    request
        .respond(response)
        .map_err(|error| anyhow::anyhow!("failed to send response: {error}"))
}

#[cfg(test)]
mod tests {
    use std::io::Write;
    use std::net::TcpListener;
    use std::net::TcpStream;

    use super::*;

    /// Drives the real listener the way the daemon does: one thread serving
    /// `handle_mailbox_request`, one client speaking HTTP over loopback. This
    /// is the only way to exercise the authentication gate end to end, because
    /// `tiny_http::Request` cannot be constructed outside the server.
    struct MailboxServer {
        addr: std::net::SocketAddr,
        handle: Option<std::thread::JoinHandle<()>>,
        server: std::sync::Arc<tiny_http::Server>,
    }

    impl MailboxServer {
        fn start(root: &Path) -> Self {
            let port = TcpListener::bind("127.0.0.1:0")
                .expect("reserve port")
                .local_addr()
                .expect("local addr")
                .port();
            let addr: std::net::SocketAddr =
                format!("127.0.0.1:{port}").parse().expect("parse addr");
            let server = std::sync::Arc::new(
                tiny_http::Server::http(addr).expect("bind mailbox test server"),
            );
            let thread_server = std::sync::Arc::clone(&server);
            let root = root.to_path_buf();
            let handle = std::thread::spawn(move || {
                for request in thread_server.incoming_requests() {
                    let method = request.method().clone();
                    let path = request.url().split('?').next().unwrap_or("/").to_string();
                    if !is_mailbox_path(&path) {
                        let _ = respond(request, 404, json!({ "ok": false }));
                        continue;
                    }
                    let _ = handle_mailbox_request(&root, &method, &path, request);
                }
            });
            Self {
                addr,
                handle: Some(handle),
                server,
            }
        }

        fn call(
            &self,
            request_line: &str,
            token: Option<&str>,
            body: Option<&str>,
        ) -> (u16, Value) {
            let mut stream = TcpStream::connect(self.addr).expect("connect");
            let mut raw = format!("{request_line} HTTP/1.1\r\nHost: 127.0.0.1\r\n");
            if let Some(token) = token {
                raw.push_str(&format!("Authorization: Bearer {token}\r\n"));
            }
            let body = body.unwrap_or("");
            raw.push_str(&format!("Content-Length: {}\r\n", body.len()));
            raw.push_str("Connection: close\r\n\r\n");
            raw.push_str(body);
            stream.write_all(raw.as_bytes()).expect("write request");
            stream.flush().expect("flush");
            let mut response = String::new();
            std::io::Read::read_to_string(&mut stream, &mut response).expect("read response");
            let status = response
                .split_whitespace()
                .nth(1)
                .and_then(|code| code.parse::<u16>().ok())
                .expect("status code");
            let payload = response
                .split_once("\r\n\r\n")
                .map(|(_, body)| body.to_string())
                .unwrap_or_default();
            let value = serde_json::from_str(&payload).unwrap_or(Value::Null);
            (status, value)
        }
    }

    impl Drop for MailboxServer {
        fn drop(&mut self) {
            self.server.unblock();
            if let Some(handle) = self.handle.take() {
                let _ = handle.join();
            }
        }
    }

    fn publish_body(id: &str, target: &str) -> String {
        json!({
            "id": id,
            "target_environment_id": target,
            "envelope_json": "{\"sig\":\"opaque\"}",
            "payload_json": "{\"body\":\"opaque\"}",
        })
        .to_string()
    }

    #[test]
    fn mailbox_paths_are_recognized() {
        assert!(is_mailbox_path(PUBLISH_PATH));
        assert!(is_mailbox_path(PENDING_PATH));
        assert!(is_mailbox_path(CONSUMED_PATH));
        assert!(!is_mailbox_path("/health"));
        assert!(!is_mailbox_path("/mcp"));
        assert!(!is_mailbox_path("/workjet/mailbox"));
    }

    #[test]
    fn query_pairs_decode_bounded_parameters() {
        let pairs =
            query_pairs("/workjet/mailbox/pending?environment_id=env%2D1&after=a_b&limit=5");
        assert_eq!(
            pairs[0],
            ("environment_id".to_string(), "env-1".to_string())
        );
        assert_eq!(pairs[1], ("after".to_string(), "a_b".to_string()));
        assert_eq!(pairs[2], ("limit".to_string(), "5".to_string()));
        assert!(query_pairs("/workjet/mailbox/pending").is_empty());
    }

    #[test]
    fn every_mailbox_route_is_authenticated_and_round_trips() {
        let root = tempfile::tempdir().expect("temp root");
        let root = root.path();
        let token = super::super::mcp_channel::mcp_operator_auth_token(root).expect("auth token");
        let server = MailboxServer::start(root);

        // Unauthenticated: all three routes deny, unlike /health on this
        // listener, which is the only public route.
        for (line, body) in [
            (
                "POST /workjet/mailbox/publish",
                Some(publish_body("env-1", "target-a")),
            ),
            ("GET /workjet/mailbox/pending?environment_id=target-a", None),
            ("POST /workjet/mailbox/consumed", Some("{}".to_string())),
        ] {
            let (status, value) = server.call(line, None, body.as_deref());
            assert_eq!(status, 401, "{line} must be authenticated");
            assert_eq!(value["error"].as_str(), Some("unauthorized"));
            let (status, _) = server.call(line, Some("wrong-token"), body.as_deref());
            assert_eq!(status, 401, "{line} must reject a wrong token");
        }

        let (status, value) = server.call(
            "POST /workjet/mailbox/publish",
            Some(&token),
            Some(&publish_body("env-1", "target-a")),
        );
        assert_eq!(status, 200);
        assert_eq!(value["duplicate"].as_bool(), Some(false));

        let (status, value) = server.call(
            "POST /workjet/mailbox/publish",
            Some(&token),
            Some(&publish_body("env-1", "target-a")),
        );
        assert_eq!(status, 200);
        assert_eq!(value["duplicate"].as_bool(), Some(true));

        let (status, value) = server.call(
            "POST /workjet/mailbox/publish",
            Some(&token),
            Some(&publish_body("env-2", "target-a")),
        );
        assert_eq!(status, 200);
        assert_eq!(value["duplicate"].as_bool(), Some(false));

        let (status, page) = server.call(
            "GET /workjet/mailbox/pending?environment_id=target-a&limit=1",
            Some(&token),
            None,
        );
        assert_eq!(status, 200);
        assert_eq!(page["count"].as_u64(), Some(1));
        assert_eq!(page["has_more"].as_bool(), Some(true));
        let cursor = page["next_cursor"].as_str().expect("cursor").to_string();

        let (status, page_two) = server.call(
            &format!("GET /workjet/mailbox/pending?environment_id=target-a&after={cursor}"),
            Some(&token),
            None,
        );
        assert_eq!(status, 200);
        assert_eq!(page_two["count"].as_u64(), Some(1));
        assert_ne!(
            page_two["envelopes"][0]["id"], page["envelopes"][0]["id"],
            "the cursor must advance past the first page"
        );

        let (status, consumed) = server.call(
            "POST /workjet/mailbox/consumed",
            Some(&token),
            Some(&json!({ "environment_id": "target-a", "envelope_ids": ["env-1"] }).to_string()),
        );
        assert_eq!(status, 200);
        assert_eq!(consumed["updated"], json!(["env-1"]));

        let (status, page) = server.call(
            "GET /workjet/mailbox/pending?environment_id=target-a",
            Some(&token),
            None,
        );
        assert_eq!(status, 200);
        assert_eq!(page["count"].as_u64(), Some(1));
        assert_eq!(page["envelopes"][0]["id"].as_str(), Some("env-2"));

        // Bad input is a bounded 400, never a panic or a 500.
        let (status, value) = server.call(
            "POST /workjet/mailbox/publish",
            Some(&token),
            Some(&json!({ "id": "bad id" }).to_string()),
        );
        assert_eq!(status, 400);
        assert_eq!(value["error"].as_str(), Some("invalid_request"));

        let (status, _) = server.call(
            "GET /workjet/mailbox/pending?environment_id=target-a&limit=abc",
            Some(&token),
            None,
        );
        assert_eq!(status, 400);

        let (status, _) = server.call(
            "POST /workjet/mailbox/consumed",
            Some(&token),
            Some(&json!({ "environment_id": "target-a" }).to_string()),
        );
        assert_eq!(status, 400);

        let (status, _) = server.call("GET /workjet/mailbox/unknown", Some(&token), None);
        assert_eq!(status, 404);
    }
}
