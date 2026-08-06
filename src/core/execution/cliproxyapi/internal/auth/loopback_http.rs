// Origin: CTOX
// License: AGPL-3.0-only

use std::future::Future;
use std::io;
use std::sync::Arc;
use std::time::Duration;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{mpsc, oneshot, Semaphore};
use tokio::task::JoinSet;

pub(crate) const IO_TIMEOUT: Duration = Duration::from_secs(10);
pub(crate) const MAX_REQUEST_HEAD_BYTES: usize = 16 * 1024;
pub(crate) const MAX_CONCURRENT_CONNECTIONS: usize = 16;
pub(crate) const MAX_TOTAL_CONNECTIONS: usize = 128;

pub(crate) async fn serve_bounded_loopback<E, MakeError, Handler, HandlerFuture>(
    listener: TcpListener,
    fatal_tx: mpsc::Sender<E>,
    mut shutdown_rx: oneshot::Receiver<()>,
    make_accept_error: MakeError,
    handler: Handler,
) where
    E: Send + 'static,
    MakeError: Fn() -> E,
    Handler: Fn(TcpStream) -> HandlerFuture + Clone + Send + Sync + 'static,
    HandlerFuture: Future<Output = ()> + Send + 'static,
{
    let permits = Arc::new(Semaphore::new(MAX_CONCURRENT_CONNECTIONS));
    let mut connections = JoinSet::new();

    for _ in 0..MAX_TOTAL_CONNECTIONS {
        let accepted = tokio::select! {
            _ = &mut shutdown_rx => {
                drain_connections(&mut connections).await;
                return;
            },
            accepted = listener.accept() => accepted,
        };
        let (stream, peer) = match accepted {
            Ok(value) => value,
            Err(_) => {
                let _ = fatal_tx.try_send(make_accept_error());
                return;
            }
        };
        if !peer.ip().is_loopback() {
            continue;
        }
        let Ok(permit) = Arc::clone(&permits).try_acquire_owned() else {
            drop(stream);
            continue;
        };
        let handler = handler.clone();
        connections.spawn(async move {
            let _permit = permit;
            handler(stream).await;
        });
        while connections.try_join_next().is_some() {}
    }

    let _ = fatal_tx.try_send(make_accept_error());
    drain_connections(&mut connections).await;
}

async fn drain_connections(connections: &mut JoinSet<()>) {
    while connections.join_next().await.is_some() {}
}

pub(crate) struct RequestHead {
    pub(crate) method: String,
    pub(crate) target: String,
}

pub(crate) async fn read_request_head(stream: &mut TcpStream) -> io::Result<RequestHead> {
    let mut bytes = Vec::with_capacity(1024);
    let mut chunk = [0_u8; 1024];
    loop {
        let read = stream.read(&mut chunk).await?;
        if read == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "request head ended",
            ));
        }
        bytes.extend_from_slice(&chunk[..read]);
        if bytes.len() > MAX_REQUEST_HEAD_BYTES {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "request head too large",
            ));
        }
        if bytes.windows(4).any(|window| window == b"\r\n\r\n") {
            break;
        }
    }

    let text = std::str::from_utf8(&bytes)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "request head is not UTF-8"))?;
    let request_line = text
        .split("\r\n")
        .next()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing request line"))?;
    let mut parts = request_line.split_whitespace();
    let method = parts.next().unwrap_or_default();
    let target = parts.next().unwrap_or_default();
    let version = parts.next().unwrap_or_default();
    if parts.next().is_some()
        || method.is_empty()
        || target.is_empty()
        || !matches!(version, "HTTP/1.0" | "HTTP/1.1")
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid request line",
        ));
    }
    Ok(RequestHead {
        method: method.to_owned(),
        target: target.to_owned(),
    })
}

pub(crate) struct HttpResponse {
    status: u16,
    reason: &'static str,
    content_type: &'static str,
    headers: Vec<(&'static str, &'static str)>,
    body: Vec<u8>,
}

impl HttpResponse {
    pub(crate) fn with_header(mut self, name: &'static str, value: &'static str) -> Self {
        self.headers.push((name, value));
        self
    }
}

pub(crate) fn response(
    status: u16,
    reason: &'static str,
    content_type: &'static str,
    body: Vec<u8>,
) -> HttpResponse {
    HttpResponse {
        status,
        reason,
        content_type,
        headers: Vec::new(),
        body,
    }
}

pub(crate) async fn write_response(
    stream: &mut TcpStream,
    response: HttpResponse,
) -> io::Result<()> {
    let mut head = format!(
        "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\nCache-Control: no-store\r\nX-Content-Type-Options: nosniff\r\nReferrer-Policy: no-referrer\r\n",
        response.status,
        response.reason,
        response.content_type,
        response.body.len(),
    );
    for (name, value) in response.headers {
        head.push_str(name);
        head.push_str(": ");
        head.push_str(value);
        head.push_str("\r\n");
    }
    head.push_str("\r\n");

    tokio::time::timeout(IO_TIMEOUT, async {
        stream.write_all(head.as_bytes()).await?;
        stream.write_all(&response.body).await?;
        stream.shutdown().await
    })
    .await
    .map_err(|_| io::Error::new(io::ErrorKind::TimedOut, "response write timed out"))?
}
