// ref: internal/httpwire/ordered_conn_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use std::io::{self, Write};
use std::sync::Arc;

use super::ordered_conn::OrderedRequestWriter;

fn order(method: &str, target: &str) -> Vec<String> {
    if method == "POST" && target == "/v1/messages?beta=true" {
        [
            "Accept",
            "Authorization",
            "Content-Type",
            "User-Agent",
            "Connection",
            "Host",
            "Accept-Encoding",
            "Content-Length",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect()
    } else {
        ["Accept", "Host", "Connection"]
            .into_iter()
            .map(str::to_owned)
            .collect()
    }
}

#[test]
fn reorders_keep_alive_requests_without_changing_bodies() {
    let mut writer = OrderedRequestWriter::new(Vec::new(), Arc::new(order));
    let first = "POST /v1/messages?beta=true HTTP/1.1\r\nHost: api.anthropic.com\r\nUser-Agent: claude-cli/2.1.220 (external, cli)\r\nContent-Length: 7\r\nAccept: application/json\r\nX-Unknown: keep\r\nAuthorization: Bearer placeholder\r\nContent-Type: application/json\r\nConnection: keep-alive\r\nAccept-Encoding: gzip, deflate, br, zstd\r\n\r\n{\"a\":1}";
    let second = "GET /api/oauth/profile HTTP/1.1\r\nConnection: close\r\nHost: api.anthropic.com\r\nAccept: application/json\r\n\r\n";
    let input = format!("{first}{second}");
    for chunk in input.as_bytes().chunks(17) {
        assert_eq!(
            writer.write_request_bytes(chunk).expect("write"),
            chunk.len()
        );
    }
    let expected = "POST /v1/messages?beta=true HTTP/1.1\r\nAccept: application/json\r\nAuthorization: Bearer placeholder\r\nContent-Type: application/json\r\nUser-Agent: claude-cli/2.1.220 (external, cli)\r\nConnection: keep-alive\r\nHost: api.anthropic.com\r\nAccept-Encoding: gzip, deflate, br, zstd\r\nContent-Length: 7\r\nX-Unknown: keep\r\n\r\n{\"a\":1}GET /api/oauth/profile HTTP/1.1\r\nAccept: application/json\r\nHost: api.anthropic.com\r\nConnection: close\r\n\r\n";
    assert_eq!(writer.into_inner(), expected.as_bytes());
}

#[test]
fn preserves_chunked_body_and_reorders_next_request() {
    let order = |_: &str, _: &str| {
        ["Host", "Transfer-Encoding"]
            .into_iter()
            .map(str::to_owned)
            .collect()
    };
    let mut writer = OrderedRequestWriter::new(Vec::new(), Arc::new(order));
    let first = "POST /upload HTTP/1.1\r\nTransfer-Encoding: chunked\r\nHost: example.com\r\n\r\n4\r\ntest\r\n0\r\nX-Trailer: done\r\n\r\n";
    let second = "GET /next HTTP/1.1\r\nTransfer-Encoding: identity\r\nHost: example.com\r\n\r\n";
    for byte in format!("{first}{second}").bytes() {
        writer.write_request_bytes(&[byte]).expect("write byte");
    }
    let expected = "POST /upload HTTP/1.1\r\nHost: example.com\r\nTransfer-Encoding: chunked\r\n\r\n4\r\ntest\r\n0\r\nX-Trailer: done\r\n\r\nGET /next HTTP/1.1\r\nHost: example.com\r\nTransfer-Encoding: identity\r\n\r\n";
    assert_eq!(writer.into_inner(), expected.as_bytes());
}

struct PartialWriter {
    bytes: Vec<u8>,
    limit: usize,
    fail: bool,
}

impl Write for PartialWriter {
    fn write(&mut self, data: &[u8]) -> io::Result<usize> {
        let written = data.len().min(self.limit);
        self.bytes.extend_from_slice(&data[..written]);
        if self.fail {
            Err(io::Error::other("injected"))
        } else {
            Ok(written)
        }
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[test]
fn reports_header_as_accepted_after_transformed_write_failure() {
    let inner = PartialWriter {
        bytes: Vec::new(),
        limit: 2,
        fail: true,
    };
    let mut writer = OrderedRequestWriter::new(inner, Arc::new(|_, _| vec!["Host".into()]));
    let header = b"GET / HTTP/1.1\r\nHost: example.com\r\n\r\n";
    let error = writer
        .write_request_bytes(header)
        .expect_err("injected failure");
    assert_eq!(error.accepted(), header.len());
}

struct FailAfterPartialBodyWriter {
    bytes: Vec<u8>,
    writes: usize,
    partial: usize,
}

impl Write for FailAfterPartialBodyWriter {
    fn write(&mut self, data: &[u8]) -> io::Result<usize> {
        self.writes += 1;
        match self.writes {
            1 => {
                self.bytes.extend_from_slice(data);
                Ok(data.len())
            }
            2 => {
                let written = data.len().min(self.partial);
                self.bytes.extend_from_slice(&data[..written]);
                Ok(written)
            }
            3 => Err(io::Error::other("injected after partial body")),
            _ => {
                self.bytes.extend_from_slice(data);
                Ok(data.len())
            }
        }
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[test]
fn partial_fixed_body_failure_advances_state_before_retry() {
    let first = b"POST /one HTTP/1.1\r\nHost: example.com\r\nContent-Length: 5\r\n\r\nabcde";
    let second = b"GET /two HTTP/1.1\r\nHost: example.com\r\n\r\n";
    let input = [first.as_slice(), second.as_slice()].concat();
    let inner = FailAfterPartialBodyWriter {
        bytes: Vec::new(),
        writes: 0,
        partial: 2,
    };
    let mut writer = OrderedRequestWriter::new(inner, Arc::new(|_, _| vec!["Host".into()]));
    let error = writer
        .write_request_bytes(&input)
        .expect_err("body write must fail once");
    assert_eq!(error.accepted(), first.len() - 3);
    writer
        .write_request_bytes(&input[error.accepted()..])
        .expect("retry remaining bytes");
    assert_eq!(writer.into_inner().bytes, input);
}

#[test]
fn partial_chunk_wire_failure_advances_tracker_before_retry() {
    let first = b"POST /one HTTP/1.1\r\nHost: example.com\r\nTransfer-Encoding: chunked\r\n\r\n4\r\ntest\r\n0\r\n\r\n";
    let second = b"GET /two HTTP/1.1\r\nHost: example.com\r\n\r\n";
    let input = [first.as_slice(), second.as_slice()].concat();
    let header_len = first
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .expect("header terminator")
        + 4;
    let inner = FailAfterPartialBodyWriter {
        bytes: Vec::new(),
        writes: 0,
        partial: 3,
    };
    let mut writer = OrderedRequestWriter::new(
        inner,
        Arc::new(|_, _| vec!["Host".into(), "Transfer-Encoding".into()]),
    );
    let error = writer
        .write_request_bytes(&input)
        .expect_err("chunk write must fail once");
    assert_eq!(error.accepted(), header_len + 3);
    writer
        .write_request_bytes(&input[error.accepted()..])
        .expect("retry remaining chunk and request");
    assert_eq!(writer.into_inner().bytes, input);
}
