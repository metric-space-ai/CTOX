// ref: sdk/proxyutil/proxy.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

//! Explicit proxy parsing and connection setup.
//!
//! The upstream `ModeInherit` HTTP path delegates to Go's default transport,
//! which can read ambient proxy environment variables. CTOX deliberately
//! adapts that one behavior: inherit retains its classification but resolves
//! to a direct, environment-independent transport. Runtime proxy values must
//! come from typed configuration/the secret store.

use std::fmt;
use std::io;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use base64::Engine as _;
use percent_encoding::percent_decode_str;
use rustls::pki_types::ServerName;
use rustls::{ClientConfig, RootCertStore};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt, ReadBuf};
use tokio::net::TcpStream;
use tokio_rustls::TlsConnector;
use url::{Host, Url};

const MAX_CONNECT_RESPONSE_HEAD: usize = 64 * 1024;

/// How a configured proxy value is interpreted.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Mode {
    Inherit,
    Direct,
    Proxy,
    Invalid,
}

/// A credential-safe error category for proxy setup and I/O.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProxyErrorKind {
    ParseUrl,
    MissingSchemeOrHost,
    UnsupportedScheme,
    UnsupportedNetwork,
    InvalidEndpoint,
    InvalidTarget,
    CredentialsTooLong,
    Connect,
    TlsHandshake,
    WriteConnect,
    ReadConnect,
    ConnectResponseTooLarge,
    InvalidConnectResponse,
    ConnectRejected,
    SocksHandshake,
    SocksAuthentication,
    SocksRejected,
}

/// Typed proxy failure. Its display text never contains the proxy URL,
/// credentials, target payload, or an underlying error string.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProxyError {
    kind: ProxyErrorKind,
}

impl ProxyError {
    const fn new(kind: ProxyErrorKind) -> Self {
        Self { kind }
    }

    pub const fn kind(self) -> ProxyErrorKind {
        self.kind
    }
}

impl fmt::Display for ProxyError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self.kind {
            ProxyErrorKind::ParseUrl => "parse proxy URL failed",
            ProxyErrorKind::MissingSchemeOrHost => "proxy URL missing scheme/host",
            ProxyErrorKind::UnsupportedScheme => "unsupported proxy scheme",
            ProxyErrorKind::UnsupportedNetwork => "proxy dialer only supports TCP",
            ProxyErrorKind::InvalidEndpoint => "invalid proxy endpoint",
            ProxyErrorKind::InvalidTarget => "invalid proxy target",
            ProxyErrorKind::CredentialsTooLong => "proxy credentials exceed protocol limits",
            ProxyErrorKind::Connect => "connect to proxy failed",
            ProxyErrorKind::TlsHandshake => "HTTPS proxy TLS handshake failed",
            ProxyErrorKind::WriteConnect => "write CONNECT request failed",
            ProxyErrorKind::ReadConnect => "read CONNECT response failed",
            ProxyErrorKind::ConnectResponseTooLarge => "proxy CONNECT response headers too large",
            ProxyErrorKind::InvalidConnectResponse => "invalid proxy CONNECT response",
            ProxyErrorKind::ConnectRejected => "proxy CONNECT was rejected",
            ProxyErrorKind::SocksHandshake => "SOCKS5 handshake failed",
            ProxyErrorKind::SocksAuthentication => "SOCKS5 authentication failed",
            ProxyErrorKind::SocksRejected => "SOCKS5 connection was rejected",
        })
    }
}

impl std::error::Error for ProxyError {}

/// Normalized proxy configuration. Debug output intentionally omits `raw` and
/// the possibly credential-bearing URL.
#[derive(Clone, Eq, PartialEq)]
pub struct Setting {
    raw: String,
    mode: Mode,
    url: Option<Url>,
    has_userinfo: bool,
}

impl Setting {
    pub fn raw(&self) -> &str {
        &self.raw
    }

    pub const fn mode(&self) -> Mode {
        self.mode
    }

    pub fn url(&self) -> Option<&Url> {
        self.url.as_ref()
    }
}

impl fmt::Debug for Setting {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Setting")
            .field("mode", &self.mode)
            .field("proxy", &redact(&self.raw))
            .finish()
    }
}

/// Parse a proxy value without consulting process environment variables.
pub fn parse(raw: &str) -> Result<Setting, ProxyError> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Ok(Setting {
            raw: String::new(),
            mode: Mode::Inherit,
            url: None,
            has_userinfo: false,
        });
    }
    if trimmed.eq_ignore_ascii_case("direct") || trimmed.eq_ignore_ascii_case("none") {
        return Ok(Setting {
            raw: trimmed.to_owned(),
            mode: Mode::Direct,
            url: None,
            has_userinfo: false,
        });
    }

    let scheme_separator = trimmed
        .find("://")
        .ok_or_else(|| ProxyError::new(ProxyErrorKind::MissingSchemeOrHost))?;
    if scheme_separator == 0 {
        return Err(ProxyError::new(ProxyErrorKind::MissingSchemeOrHost));
    }
    let configured_scheme = &trimmed[..scheme_separator];
    validate_userinfo_percent_encoding(&trimmed[scheme_separator + 3..])?;
    let parsed = Url::parse(trimmed).map_err(|_| ProxyError::new(ProxyErrorKind::ParseUrl))?;
    if parsed.cannot_be_a_base() || parsed.host_str().is_none() {
        return Err(ProxyError::new(ProxyErrorKind::MissingSchemeOrHost));
    }
    if !matches!(configured_scheme, "socks5" | "socks5h" | "http" | "https") {
        return Err(ProxyError::new(ProxyErrorKind::UnsupportedScheme));
    }
    Ok(Setting {
        raw: trimmed.to_owned(),
        mode: Mode::Proxy,
        url: Some(parsed),
        has_userinfo: authority_has_userinfo(trimmed),
    })
}

/// A transport route consumable by an HTTP client adapter.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum HttpTransportRoute {
    Direct,
    HttpProxy,
    Socks5,
}

/// Explicit HTTP transport configuration. `Inherit` is deliberately direct
/// in CTOX, while `mode()` still reports that no value was configured.
#[derive(Clone)]
pub struct HttpTransport {
    mode: Mode,
    proxy_url: Option<Url>,
    dialer: ProxyDialer,
    route: HttpTransportRoute,
}

impl HttpTransport {
    pub const fn mode(&self) -> Mode {
        self.mode
    }

    pub fn route(&self) -> &HttpTransportRoute {
        &self.route
    }

    pub fn proxy_url(&self) -> Option<&Url> {
        self.proxy_url.as_ref()
    }

    /// This is always false by construction and is exposed as an audit hook.
    pub const fn uses_environment_proxy(&self) -> bool {
        false
    }

    /// Dial a TCP tunnel. For HTTP proxy routes this uses CONNECT; an HTTP
    /// client may instead use `proxy_url()` for forward-proxy requests.
    pub async fn dial(&self, network: &str, address: &str) -> Result<ProxyStream, ProxyError> {
        self.dialer.dial(network, address).await
    }
}

impl fmt::Debug for HttpTransport {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("HttpTransport")
            .field("mode", &self.mode)
            .field("route", &self.route)
            .field(
                "proxy",
                &self.proxy_url.as_ref().map(|url| redact(url.as_str())),
            )
            .field("uses_environment_proxy", &false)
            .finish()
    }
}

/// Construct an environment-independent HTTP transport.
pub fn build_http_transport(raw: &str) -> Result<(HttpTransport, Mode), ProxyError> {
    let setting = parse(raw)?;
    let mode = setting.mode;
    let (route, proxy_url, dialer) = match setting.url {
        None => (HttpTransportRoute::Direct, None, ProxyDialer::direct()),
        Some(url) if matches!(url.scheme(), "http" | "https") => (
            HttpTransportRoute::HttpProxy,
            Some(url.clone()),
            ProxyDialer::http_connect(url, setting.has_userinfo)?,
        ),
        Some(url) => (
            HttpTransportRoute::Socks5,
            Some(url.clone()),
            ProxyDialer::socks5(url, setting.has_userinfo)?,
        ),
    };
    Ok((
        HttpTransport {
            mode,
            proxy_url,
            dialer,
            route,
        },
        mode,
    ))
}

/// Construct a direct transport explicitly.
pub fn new_direct_transport() -> HttpTransport {
    HttpTransport {
        mode: Mode::Direct,
        proxy_url: None,
        dialer: ProxyDialer::direct(),
        route: HttpTransportRoute::Direct,
    }
}

/// Concrete async TCP proxy dialer.
#[derive(Clone)]
pub struct ProxyDialer {
    kind: Arc<DialerKind>,
}

enum DialerKind {
    Direct,
    HttpConnect {
        proxy_url: Url,
        has_userinfo: bool,
        tls_config: Arc<ClientConfig>,
    },
    Socks5 {
        endpoint: String,
        username: Option<Vec<u8>>,
        password: Option<Vec<u8>>,
    },
}

struct ProxyCredentials {
    username: Vec<u8>,
    password: Vec<u8>,
}

impl ProxyDialer {
    fn direct() -> Self {
        Self {
            kind: Arc::new(DialerKind::Direct),
        }
    }

    fn http_connect(proxy_url: Url, has_userinfo: bool) -> Result<Self, ProxyError> {
        Ok(Self {
            kind: Arc::new(DialerKind::HttpConnect {
                proxy_url,
                has_userinfo,
                tls_config: default_tls_config(),
            }),
        })
    }

    fn socks5(url: Url, has_userinfo: bool) -> Result<Self, ProxyError> {
        let endpoint = socks_dial_address(&url)?;
        let credentials = credentials(&url, has_userinfo);
        if credentials.as_ref().is_some_and(|value| {
            value.username.len() > u8::MAX as usize || value.password.len() > u8::MAX as usize
        }) {
            return Err(ProxyError::new(ProxyErrorKind::CredentialsTooLong));
        }
        let (username, password) = credentials
            .map(|value| (Some(value.username), Some(value.password)))
            .unwrap_or((None, None));
        Ok(Self {
            kind: Arc::new(DialerKind::Socks5 {
                endpoint,
                username,
                password,
            }),
        })
    }

    pub async fn dial(&self, network: &str, address: &str) -> Result<ProxyStream, ProxyError> {
        if !matches!(network, "tcp" | "tcp4" | "tcp6") {
            return Err(ProxyError::new(ProxyErrorKind::UnsupportedNetwork));
        }
        validate_target(address)?;
        match self.kind.as_ref() {
            DialerKind::Direct => TcpStream::connect(address)
                .await
                .map(|stream| Box::new(stream) as ProxyStream)
                .map_err(|_| ProxyError::new(ProxyErrorKind::Connect)),
            DialerKind::HttpConnect {
                proxy_url,
                has_userinfo,
                tls_config,
            } => dial_http_connect(proxy_url, *has_userinfo, tls_config, address).await,
            DialerKind::Socks5 {
                endpoint,
                username,
                password,
            } => dial_socks5(endpoint, username.as_deref(), password.as_deref(), address).await,
        }
    }
}

impl fmt::Debug for ProxyDialer {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let kind = match self.kind.as_ref() {
            DialerKind::Direct => "direct",
            DialerKind::HttpConnect { .. } => "http-connect",
            DialerKind::Socks5 { .. } => "socks5",
        };
        formatter
            .debug_struct("ProxyDialer")
            .field("kind", &kind)
            .finish()
    }
}

/// Build a connection-layer dialer. Inherit is direct in CTOX so this function
/// never falls back to an ambient system dialer.
pub fn build_dialer(raw: &str) -> Result<(ProxyDialer, Mode), ProxyError> {
    let setting = parse(raw)?;
    let mode = setting.mode;
    let dialer = match setting.url {
        None => ProxyDialer::direct(),
        Some(url) if matches!(url.scheme(), "http" | "https") => {
            ProxyDialer::http_connect(url, setting.has_userinfo)?
        }
        Some(url) => ProxyDialer::socks5(url, setting.has_userinfo)?,
    };
    Ok((dialer, mode))
}

/// An async byte stream returned by a proxy dialer.
pub type ProxyStream = Box<dyn AsyncProxyStream>;

pub trait AsyncProxyStream: AsyncRead + AsyncWrite + Unpin + Send {}
impl<T> AsyncProxyStream for T where T: AsyncRead + AsyncWrite + Unpin + Send {}

fn default_tls_config() -> Arc<ClientConfig> {
    let mut roots = RootCertStore::empty();
    roots.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
    Arc::new(
        ClientConfig::builder_with_provider(Arc::new(rustls::crypto::ring::default_provider()))
            .with_safe_default_protocol_versions()
            .expect("ring supports the configured TLS protocol versions")
            .with_root_certificates(roots)
            .with_no_client_auth(),
    )
}

async fn dial_http_connect(
    proxy_url: &Url,
    has_userinfo: bool,
    tls_config: &Arc<ClientConfig>,
    target: &str,
) -> Result<ProxyStream, ProxyError> {
    let endpoint = proxy_dial_address(proxy_url)?;
    let tcp = TcpStream::connect(endpoint)
        .await
        .map_err(|_| ProxyError::new(ProxyErrorKind::Connect))?;
    let mut stream: ProxyStream = if proxy_url.scheme() == "https" {
        let host = proxy_url
            .host_str()
            .ok_or_else(|| ProxyError::new(ProxyErrorKind::InvalidEndpoint))?;
        let server_name = ServerName::try_from(bare_host(host).to_owned())
            .map_err(|_| ProxyError::new(ProxyErrorKind::InvalidEndpoint))?;
        let tls = TlsConnector::from(Arc::clone(tls_config))
            .connect(server_name, tcp)
            .await
            .map_err(|_| ProxyError::new(ProxyErrorKind::TlsHandshake))?;
        Box::new(tls)
    } else {
        Box::new(tcp)
    };

    let mut request = format!("CONNECT {target} HTTP/1.1\r\nHost: {target}\r\n");
    if let Some(authorization) = proxy_authorization(proxy_url, has_userinfo) {
        request.push_str("Proxy-Authorization: ");
        request.push_str(&authorization);
        request.push_str("\r\n");
    }
    request.push_str("\r\n");
    stream
        .write_all(request.as_bytes())
        .await
        .map_err(|_| ProxyError::new(ProxyErrorKind::WriteConnect))?;
    stream
        .flush()
        .await
        .map_err(|_| ProxyError::new(ProxyErrorKind::WriteConnect))?;

    let mut received = Vec::with_capacity(1024);
    let head_end = loop {
        if received.len() >= MAX_CONNECT_RESPONSE_HEAD {
            return Err(ProxyError::new(ProxyErrorKind::ConnectResponseTooLarge));
        }
        let mut buffer = [0_u8; 1024];
        let read = stream
            .read(&mut buffer)
            .await
            .map_err(|_| ProxyError::new(ProxyErrorKind::ReadConnect))?;
        if read == 0 {
            return Err(ProxyError::new(ProxyErrorKind::ReadConnect));
        }
        received.extend_from_slice(&buffer[..read]);
        if received.len() > MAX_CONNECT_RESPONSE_HEAD {
            return Err(ProxyError::new(ProxyErrorKind::ConnectResponseTooLarge));
        }
        if let Some(position) = find_bytes(&received, b"\r\n\r\n") {
            break position + 4;
        }
    };
    let head = std::str::from_utf8(&received[..head_end])
        .map_err(|_| ProxyError::new(ProxyErrorKind::InvalidConnectResponse))?;
    let mut lines = head.split("\r\n");
    let status_line = lines
        .next()
        .ok_or_else(|| ProxyError::new(ProxyErrorKind::InvalidConnectResponse))?;
    let mut status = status_line.split_ascii_whitespace();
    let version = status
        .next()
        .ok_or_else(|| ProxyError::new(ProxyErrorKind::InvalidConnectResponse))?;
    let code = status
        .next()
        .and_then(|value| value.parse::<u16>().ok())
        .ok_or_else(|| ProxyError::new(ProxyErrorKind::InvalidConnectResponse))?;
    if !matches!(version, "HTTP/1.0" | "HTTP/1.1") {
        return Err(ProxyError::new(ProxyErrorKind::InvalidConnectResponse));
    }
    for header in lines.take_while(|line| !line.is_empty()) {
        let (name, value) = header
            .split_once(':')
            .ok_or_else(|| ProxyError::new(ProxyErrorKind::InvalidConnectResponse))?;
        if name.is_empty()
            || !name.bytes().all(is_http_token)
            || value
                .bytes()
                .any(|byte| byte.is_ascii_control() && byte != b'\t')
        {
            return Err(ProxyError::new(ProxyErrorKind::InvalidConnectResponse));
        }
    }
    if code != 200 {
        return Err(ProxyError::new(ProxyErrorKind::ConnectRejected));
    }
    let buffered = received.split_off(head_end);
    if buffered.is_empty() {
        Ok(stream)
    } else {
        Ok(Box::new(PrefixedStream {
            prefix: buffered,
            offset: 0,
            inner: stream,
        }))
    }
}

async fn dial_socks5(
    endpoint: &str,
    username: Option<&[u8]>,
    password: Option<&[u8]>,
    target: &str,
) -> Result<ProxyStream, ProxyError> {
    let mut stream = TcpStream::connect(endpoint)
        .await
        .map_err(|_| ProxyError::new(ProxyErrorKind::Connect))?;
    let has_auth = username.is_some();
    let greeting: &[u8] = if has_auth {
        &[0x05, 0x02, 0x00, 0x02]
    } else {
        &[0x05, 0x01, 0x00]
    };
    stream
        .write_all(greeting)
        .await
        .map_err(|_| ProxyError::new(ProxyErrorKind::SocksHandshake))?;
    let mut choice = [0_u8; 2];
    stream
        .read_exact(&mut choice)
        .await
        .map_err(|_| ProxyError::new(ProxyErrorKind::SocksHandshake))?;
    if choice[0] != 0x05 || choice[1] == 0xff {
        return Err(ProxyError::new(ProxyErrorKind::SocksHandshake));
    }
    if choice[1] == 0x02 {
        let username =
            username.ok_or_else(|| ProxyError::new(ProxyErrorKind::SocksAuthentication))?;
        let password = password.unwrap_or_default();
        if username.len() > u8::MAX as usize || password.len() > u8::MAX as usize {
            return Err(ProxyError::new(ProxyErrorKind::CredentialsTooLong));
        }
        let mut auth = Vec::with_capacity(username.len() + password.len() + 3);
        auth.extend_from_slice(&[0x01, username.len() as u8]);
        auth.extend_from_slice(username);
        auth.push(password.len() as u8);
        auth.extend_from_slice(password);
        stream
            .write_all(&auth)
            .await
            .map_err(|_| ProxyError::new(ProxyErrorKind::SocksAuthentication))?;
        let mut response = [0_u8; 2];
        stream
            .read_exact(&mut response)
            .await
            .map_err(|_| ProxyError::new(ProxyErrorKind::SocksAuthentication))?;
        if response != [0x01, 0x00] {
            return Err(ProxyError::new(ProxyErrorKind::SocksAuthentication));
        }
    } else if choice[1] != 0x00 {
        return Err(ProxyError::new(ProxyErrorKind::SocksHandshake));
    }

    let (host, port) = split_target(target)?;
    let mut request = vec![0x05, 0x01, 0x00];
    match Host::parse(&host).map_err(|_| ProxyError::new(ProxyErrorKind::InvalidTarget))? {
        Host::Ipv4(address) => {
            request.push(0x01);
            request.extend_from_slice(&address.octets());
        }
        Host::Ipv6(address) => {
            request.push(0x04);
            request.extend_from_slice(&address.octets());
        }
        Host::Domain(domain) => {
            let bytes = domain.as_bytes();
            if bytes.is_empty() || bytes.len() > u8::MAX as usize {
                return Err(ProxyError::new(ProxyErrorKind::InvalidTarget));
            }
            request.extend_from_slice(&[0x03, bytes.len() as u8]);
            request.extend_from_slice(bytes);
        }
    }
    request.extend_from_slice(&port.to_be_bytes());
    stream
        .write_all(&request)
        .await
        .map_err(|_| ProxyError::new(ProxyErrorKind::SocksHandshake))?;
    let mut response = [0_u8; 4];
    stream
        .read_exact(&mut response)
        .await
        .map_err(|_| ProxyError::new(ProxyErrorKind::SocksHandshake))?;
    if response[0] != 0x05 || response[2] != 0x00 {
        return Err(ProxyError::new(ProxyErrorKind::SocksHandshake));
    }
    if response[1] != 0x00 {
        return Err(ProxyError::new(ProxyErrorKind::SocksRejected));
    }
    let address_bytes = match response[3] {
        0x01 => 4,
        0x04 => 16,
        0x03 => {
            let length = stream
                .read_u8()
                .await
                .map_err(|_| ProxyError::new(ProxyErrorKind::SocksHandshake))?;
            length as usize
        }
        _ => return Err(ProxyError::new(ProxyErrorKind::SocksHandshake)),
    };
    let mut discard = vec![0_u8; address_bytes + 2];
    stream
        .read_exact(&mut discard)
        .await
        .map_err(|_| ProxyError::new(ProxyErrorKind::SocksHandshake))?;
    Ok(Box::new(stream))
}

fn proxy_dial_address(url: &Url) -> Result<String, ProxyError> {
    let host = url
        .host_str()
        .ok_or_else(|| ProxyError::new(ProxyErrorKind::InvalidEndpoint))?;
    let port = url
        .port()
        .unwrap_or(if url.scheme() == "https" { 443 } else { 80 });
    Ok(format_host_port(host, port))
}

fn socks_dial_address(url: &Url) -> Result<String, ProxyError> {
    let host = url
        .host_str()
        .ok_or_else(|| ProxyError::new(ProxyErrorKind::InvalidEndpoint))?;
    Ok(match url.port() {
        Some(port) => format_host_port(host, port),
        None => host.to_owned(),
    })
}

fn credentials(url: &Url, has_userinfo: bool) -> Option<ProxyCredentials> {
    if !has_userinfo {
        return None;
    }
    let username = percent_decode_str(url.username()).collect::<Vec<_>>();
    let password = percent_decode_str(url.password().unwrap_or_default()).collect::<Vec<_>>();
    Some(ProxyCredentials { username, password })
}

fn authority_has_userinfo(raw: &str) -> bool {
    let Some(authority_start) = raw.find("://").map(|position| position + 3) else {
        return false;
    };
    let authority_end = raw[authority_start..]
        .find(['/', '?', '#'])
        .map(|position| authority_start + position)
        .unwrap_or(raw.len());
    raw[authority_start..authority_end].contains('@')
}

fn proxy_authorization(url: &Url, has_userinfo: bool) -> Option<String> {
    let credentials = credentials(url, has_userinfo)?;
    let mut plain = credentials.username;
    plain.push(b':');
    plain.extend_from_slice(&credentials.password);
    Some(format!("Basic {}", BASE64_STANDARD.encode(plain)))
}

fn validate_target(target: &str) -> Result<(), ProxyError> {
    if target.bytes().any(|byte| byte.is_ascii_control()) {
        return Err(ProxyError::new(ProxyErrorKind::InvalidTarget));
    }
    let _ = split_target(target)?;
    Ok(())
}

fn split_target(target: &str) -> Result<(String, u16), ProxyError> {
    if let Ok(socket) = target.parse::<std::net::SocketAddr>() {
        return Ok((socket.ip().to_string(), socket.port()));
    }
    let (host, port) = target
        .rsplit_once(':')
        .ok_or_else(|| ProxyError::new(ProxyErrorKind::InvalidTarget))?;
    if host.is_empty() || host.contains(':') {
        return Err(ProxyError::new(ProxyErrorKind::InvalidTarget));
    }
    let port = port
        .parse::<u16>()
        .map_err(|_| ProxyError::new(ProxyErrorKind::InvalidTarget))?;
    Ok((host.to_owned(), port))
}

fn format_host_port(host: &str, port: u16) -> String {
    let host = bare_host(host);
    if host.contains(':') {
        format!("[{host}]:{port}")
    } else {
        format!("{host}:{port}")
    }
}

fn bare_host(host: &str) -> &str {
    host.strip_prefix('[')
        .and_then(|value| value.strip_suffix(']'))
        .unwrap_or(host)
}

fn validate_userinfo_percent_encoding(authority_and_rest: &str) -> Result<(), ProxyError> {
    let authority_end = authority_and_rest
        .find(['/', '?', '#'])
        .unwrap_or(authority_and_rest.len());
    let authority = &authority_and_rest[..authority_end];
    let Some((userinfo, _)) = authority.rsplit_once('@') else {
        return Ok(());
    };
    let bytes = userinfo.as_bytes();
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'%' {
            if index + 2 >= bytes.len()
                || !bytes[index + 1].is_ascii_hexdigit()
                || !bytes[index + 2].is_ascii_hexdigit()
            {
                return Err(ProxyError::new(ProxyErrorKind::ParseUrl));
            }
            index += 3;
        } else {
            index += 1;
        }
    }
    Ok(())
}

fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

fn is_http_token(byte: u8) -> bool {
    byte.is_ascii_alphanumeric()
        || matches!(
            byte,
            b'!' | b'#'
                | b'$'
                | b'%'
                | b'&'
                | b'\''
                | b'*'
                | b'+'
                | b'-'
                | b'.'
                | b'^'
                | b'_'
                | b'`'
                | b'|'
                | b'~'
        )
}

/// Return a log-safe proxy URL with credentials, path, query, and fragment
/// removed. Unsupported-but-well-formed schemes are still redacted like Go.
pub fn redact(raw: &str) -> String {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return String::new();
    }
    let Some(separator) = trimmed.find("://") else {
        return "<invalid proxy URL>".to_owned();
    };
    if separator == 0 || validate_userinfo_percent_encoding(&trimmed[separator + 3..]).is_err() {
        return "<invalid proxy URL>".to_owned();
    }
    let Ok(parsed) = Url::parse(trimmed) else {
        return "<invalid proxy URL>".to_owned();
    };
    let Some(host) = parsed.host_str() else {
        return "<invalid proxy URL>".to_owned();
    };
    let authority = if let Some(port) = parsed.port() {
        format_host_port(host, port)
    } else if host.contains(':') {
        let bare = host
            .strip_prefix('[')
            .and_then(|value| value.strip_suffix(']'))
            .unwrap_or(host);
        format!("[{bare}]")
    } else {
        host.to_owned()
    };
    let user = if authority_has_userinfo(trimmed) {
        "redacted@"
    } else {
        ""
    };
    format!("{}://{user}{authority}", &trimmed[..separator])
}

struct PrefixedStream {
    prefix: Vec<u8>,
    offset: usize,
    inner: ProxyStream,
}

impl AsyncRead for PrefixedStream {
    fn poll_read(
        mut self: Pin<&mut Self>,
        context: &mut Context<'_>,
        buffer: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        if self.offset < self.prefix.len() && buffer.remaining() > 0 {
            let available = &self.prefix[self.offset..];
            let count = available.len().min(buffer.remaining());
            buffer.put_slice(&available[..count]);
            self.offset += count;
            return Poll::Ready(Ok(()));
        }
        Pin::new(&mut self.inner).poll_read(context, buffer)
    }
}

impl AsyncWrite for PrefixedStream {
    fn poll_write(
        mut self: Pin<&mut Self>,
        context: &mut Context<'_>,
        buffer: &[u8],
    ) -> Poll<io::Result<usize>> {
        Pin::new(&mut self.inner).poll_write(context, buffer)
    }

    fn poll_flush(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(&mut self.inner).poll_flush(context)
    }

    fn poll_shutdown(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(&mut self.inner).poll_shutdown(context)
    }
}
