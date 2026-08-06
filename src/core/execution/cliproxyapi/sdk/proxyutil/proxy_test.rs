// ref: sdk/proxyutil/proxy_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::time::Duration;

use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use base64::Engine as _;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::time::timeout;

use super::{
    build_dialer, build_http_transport, parse, redact, HttpTransportRoute, Mode, ProxyErrorKind,
};

#[test]
fn parse_modes_and_normalizes_whitespace() {
    let cases = [
        ("", Mode::Inherit),
        (" \n\t", Mode::Inherit),
        ("direct", Mode::Direct),
        ("DiReCt", Mode::Direct),
        (" NONE ", Mode::Direct),
        ("http://proxy.example.com:8080", Mode::Proxy),
        ("https://proxy.example.com:8443", Mode::Proxy),
        ("socks5://proxy.example.com:1080", Mode::Proxy),
        ("socks5h://proxy.example.com:1080", Mode::Proxy),
    ];
    for (raw, mode) in cases {
        let setting = parse(raw).expect("setting should parse");
        assert_eq!(setting.mode(), mode, "raw={raw:?}");
        assert_eq!(setting.raw(), raw.trim());
    }
}

#[test]
fn parse_rejects_missing_and_unsupported_components() {
    let cases = [
        ("bad-value", ProxyErrorKind::MissingSchemeOrHost),
        ("http:relative", ProxyErrorKind::MissingSchemeOrHost),
        ("ftp://proxy.example", ProxyErrorKind::UnsupportedScheme),
        ("HTTP://proxy.example", ProxyErrorKind::UnsupportedScheme),
        ("http://", ProxyErrorKind::ParseUrl),
        ("http://host:99999", ProxyErrorKind::ParseUrl),
    ];
    for (raw, kind) in cases {
        assert_eq!(parse(raw).unwrap_err().kind(), kind, "raw={raw:?}");
    }
}

#[test]
fn errors_and_debug_do_not_expose_credentials() {
    let raw = "http://user:secret%@proxy.example.com:8080";
    let error = parse(raw).unwrap_err();
    let rendered = format!("{error:?} {error}");
    assert!(!rendered.contains("user"));
    assert!(!rendered.contains("secret"));

    let valid = parse("http://user:secret@proxy.example.com/path?token=x").unwrap();
    let debug = format!("{valid:?}");
    assert!(!debug.contains("secret"));
    assert!(!debug.contains("token"));
    assert!(debug.contains("redacted"));
}

#[test]
fn redact_matches_upstream_and_handles_ipv6() {
    assert_eq!(
        redact("http://user:pass@proxy.example.com:8080/path?token=secret"),
        "http://redacted@proxy.example.com:8080"
    );
    assert_eq!(
        redact("socks5://proxy.example.com:1080"),
        "socks5://proxy.example.com:1080"
    );
    assert_eq!(redact("https://[::1]:8443/a"), "https://[::1]:8443");
    assert_eq!(
        redact("http://@proxy.example"),
        "http://redacted@proxy.example"
    );
    assert_eq!(redact("bad-value"), "<invalid proxy URL>");
    assert_eq!(redact("http:relative"), "<invalid proxy URL>");
    assert_eq!(
        redact("http://user:secret%@proxy.example"),
        "<invalid proxy URL>"
    );
    assert_eq!(redact(""), "");
}

#[test]
fn http_transport_never_uses_environment_fallback() {
    let (inherit, mode) = build_http_transport("").unwrap();
    assert_eq!(mode, Mode::Inherit);
    assert_eq!(inherit.route(), &HttpTransportRoute::Direct);
    assert!(!inherit.uses_environment_proxy());

    let (direct, mode) = build_http_transport("direct").unwrap();
    assert_eq!(mode, Mode::Direct);
    assert_eq!(direct.route(), &HttpTransportRoute::Direct);

    let (http, mode) = build_http_transport("http://proxy.example:8080").unwrap();
    assert_eq!(mode, Mode::Proxy);
    assert_eq!(http.route(), &HttpTransportRoute::HttpProxy);
    assert_eq!(
        http.proxy_url().unwrap().as_str(),
        "http://proxy.example:8080/"
    );

    let (socks, mode) = build_http_transport("socks5h://proxy.example:1080").unwrap();
    assert_eq!(mode, Mode::Proxy);
    assert_eq!(socks.route(), &HttpTransportRoute::Socks5);
}

#[tokio::test]
async fn inherited_dialer_is_direct_and_supports_tcp_families() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (mut connection, _) = listener.accept().await.unwrap();
        connection.write_all(b"direct").await.unwrap();
    });
    let (dialer, mode) = build_dialer("").unwrap();
    assert_eq!(mode, Mode::Inherit);
    let mut stream = dialer.dial("tcp4", &address.to_string()).await.unwrap();
    let mut payload = [0_u8; 6];
    stream.read_exact(&mut payload).await.unwrap();
    assert_eq!(&payload, b"direct");
    server.await.unwrap();
}

#[tokio::test]
async fn http_connect_preserves_buffered_tunnel_bytes_and_basic_auth() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (mut connection, _) = listener.accept().await.unwrap();
        let request = read_head(&mut connection).await;
        let text = String::from_utf8(request).unwrap();
        assert!(text.starts_with("CONNECT target.example.com:443 HTTP/1.1\r\n"));
        assert!(text.contains("\r\nHost: target.example.com:443\r\n"));
        let encoded = BASE64_STANDARD.encode("user:pass");
        assert!(text.contains(&format!("\r\nProxy-Authorization: Basic {encoded}\r\n")));
        connection
            .write_all(b"HTTP/1.1 200 Connection Established\r\nX-Test: yes\r\n\r\nok")
            .await
            .unwrap();
        let mut ping = [0_u8; 4];
        connection.read_exact(&mut ping).await.unwrap();
        assert_eq!(&ping, b"ping");
    });

    let (dialer, mode) = build_dialer(&format!("http://user:pass@{address}")).unwrap();
    assert_eq!(mode, Mode::Proxy);
    let mut tunnel = dialer.dial("tcp", "target.example.com:443").await.unwrap();
    let mut buffered = [0_u8; 2];
    tunnel.read_exact(&mut buffered).await.unwrap();
    assert_eq!(&buffered, b"ok");
    tunnel.write_all(b"ping").await.unwrap();
    server.await.unwrap();
}

#[tokio::test]
async fn http_connect_rejection_is_typed_and_credential_safe() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (mut connection, _) = listener.accept().await.unwrap();
        let _ = read_head(&mut connection).await;
        connection
            .write_all(b"HTTP/1.1 407 Proxy Authentication Required\r\n\r\n")
            .await
            .unwrap();
    });
    let (dialer, _) = build_dialer(&format!("http://user:secret@{address}")).unwrap();
    let error = match dialer.dial("tcp", "target.example:443").await {
        Ok(_) => panic!("CONNECT should be rejected"),
        Err(error) => error,
    };
    assert_eq!(error.kind(), ProxyErrorKind::ConnectRejected);
    assert!(!error.to_string().contains("secret"));
    server.await.unwrap();
}

#[tokio::test]
async fn malformed_http_connect_response_is_rejected() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (mut connection, _) = listener.accept().await.unwrap();
        let _ = read_head(&mut connection).await;
        connection
            .write_all(b"HTTP/1.1 200 OK\r\nMalformed Header\r\n\r\n")
            .await
            .unwrap();
    });
    let (dialer, _) = build_dialer(&format!("http://{address}")).unwrap();
    let error = match dialer.dial("tcp", "target.example:443").await {
        Ok(_) => panic!("malformed CONNECT response should fail"),
        Err(error) => error,
    };
    assert_eq!(error.kind(), ProxyErrorKind::InvalidConnectResponse);
    server.await.unwrap();
}

#[tokio::test]
async fn https_proxy_starts_tls_before_connect() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (mut connection, _) = listener.accept().await.unwrap();
        let first = timeout(Duration::from_secs(1), connection.read_u8())
            .await
            .expect("TLS client hello should arrive")
            .unwrap();
        assert_eq!(first, 0x16, "expected TLS handshake record");
    });
    let (dialer, _) = build_dialer(&format!("https://localhost:{}", address.port())).unwrap();
    let error = match dialer.dial("tcp", "target.example:443").await {
        Ok(_) => panic!("plain test endpoint cannot complete TLS"),
        Err(error) => error,
    };
    assert_eq!(error.kind(), ProxyErrorKind::TlsHandshake);
    server.await.unwrap();
}

#[tokio::test]
async fn dropping_canceled_http_connect_closes_proxy_socket() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let (request_seen_tx, request_seen_rx) = tokio::sync::oneshot::channel();
    let server = tokio::spawn(async move {
        let (mut connection, _) = listener.accept().await.unwrap();
        let _ = read_head(&mut connection).await;
        request_seen_tx.send(()).unwrap();
        let mut byte = [0_u8; 1];
        let read = timeout(Duration::from_secs(1), connection.read(&mut byte))
            .await
            .expect("canceled dial must close socket")
            .unwrap();
        assert_eq!(read, 0);
    });
    let (dialer, _) = build_dialer(&format!("http://{address}")).unwrap();
    let dial = tokio::spawn(async move { dialer.dial("tcp", "target.example:443").await });
    request_seen_rx.await.unwrap();
    dial.abort();
    let _ = dial.await;
    server.await.unwrap();
}

#[tokio::test]
async fn socks5h_performs_remote_domain_resolution_and_authentication() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (mut connection, _) = listener.accept().await.unwrap();
        let mut greeting = [0_u8; 4];
        connection.read_exact(&mut greeting).await.unwrap();
        assert_eq!(greeting, [5, 2, 0, 2]);
        connection.write_all(&[5, 2]).await.unwrap();

        let version = connection.read_u8().await.unwrap();
        let user_len = connection.read_u8().await.unwrap() as usize;
        let mut user = vec![0_u8; user_len];
        connection.read_exact(&mut user).await.unwrap();
        let pass_len = connection.read_u8().await.unwrap() as usize;
        let mut pass = vec![0_u8; pass_len];
        connection.read_exact(&mut pass).await.unwrap();
        assert_eq!(version, 1);
        assert_eq!(user, b"user name");
        assert_eq!(pass, b"p@ss");
        connection.write_all(&[1, 0]).await.unwrap();

        let mut fixed = [0_u8; 4];
        connection.read_exact(&mut fixed).await.unwrap();
        assert_eq!(fixed, [5, 1, 0, 3]);
        let host_len = connection.read_u8().await.unwrap() as usize;
        let mut host = vec![0_u8; host_len];
        connection.read_exact(&mut host).await.unwrap();
        assert_eq!(host, b"target.example");
        assert_eq!(connection.read_u16().await.unwrap(), 443);
        connection
            .write_all(&[5, 0, 0, 1, 127, 0, 0, 1, 0x1f, 0x90])
            .await
            .unwrap();
    });
    let (dialer, mode) = build_dialer(&format!("socks5h://user%20name:p%40ss@{address}")).unwrap();
    assert_eq!(mode, Mode::Proxy);
    let _stream = dialer.dial("tcp", "target.example:443").await.unwrap();
    server.await.unwrap();
}

#[tokio::test]
async fn socks5_rejection_and_invalid_targets_are_typed() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (mut connection, _) = listener.accept().await.unwrap();
        let mut greeting = [0_u8; 3];
        connection.read_exact(&mut greeting).await.unwrap();
        connection.write_all(&[5, 0]).await.unwrap();
        let mut request = [0_u8; 4];
        connection.read_exact(&mut request).await.unwrap();
        let length = connection.read_u8().await.unwrap() as usize;
        let mut rest = vec![0_u8; length + 2];
        connection.read_exact(&mut rest).await.unwrap();
        connection
            .write_all(&[5, 5, 0, 1, 0, 0, 0, 0, 0, 0])
            .await
            .unwrap();
    });
    let (dialer, _) = build_dialer(&format!("socks5://{address}")).unwrap();
    let rejection = match dialer.dial("tcp", "target.example:443").await {
        Ok(_) => panic!("SOCKS5 connection should be rejected"),
        Err(error) => error,
    };
    assert_eq!(rejection.kind(), ProxyErrorKind::SocksRejected);
    server.await.unwrap();

    let unsupported = match dialer.dial("udp", "target.example:443").await {
        Ok(_) => panic!("UDP should be unsupported"),
        Err(error) => error,
    };
    assert_eq!(unsupported.kind(), ProxyErrorKind::UnsupportedNetwork);
    let injected = match dialer
        .dial("tcp", "target.example:443\r\nInjected: yes")
        .await
    {
        Ok(_) => panic!("control characters should be rejected"),
        Err(error) => error,
    };
    assert_eq!(injected.kind(), ProxyErrorKind::InvalidTarget);
}

async fn read_head(stream: &mut tokio::net::TcpStream) -> Vec<u8> {
    let mut head = Vec::new();
    loop {
        let byte = stream.read_u8().await.unwrap();
        head.push(byte);
        if head.ends_with(b"\r\n\r\n") {
            return head;
        }
    }
}
