// ref: internal/httpwire/ordered_conn.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// Port-Note: exposes accepted-byte counts explicitly because std::io::Write errors cannot carry them.
// License: MIT (upstream); modifications AGPL-3.0-only

use std::fmt;
use std::io::{self, Write};
use std::sync::Arc;

const MAX_BUFFERED_REQUEST_HEADER: usize = 1 << 20;

pub type RequestHeaderOrder = dyn Fn(&str, &str) -> Vec<String> + Send + Sync + 'static;

#[derive(Debug)]
pub struct OrderedWriteError {
    accepted: usize,
    source: io::Error,
}

impl OrderedWriteError {
    #[must_use]
    pub fn accepted(&self) -> usize {
        self.accepted
    }

    #[must_use]
    pub fn source_error(&self) -> &io::Error {
        &self.source
    }
}

impl fmt::Display for OrderedWriteError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "ordered HTTP writer failed after accepting {} bytes: {}",
            self.accepted, self.source
        )
    }
}

impl std::error::Error for OrderedWriteError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.source)
    }
}

pub struct OrderedRequestWriter<W> {
    inner: W,
    order: Arc<RequestHeaderOrder>,
    header: Vec<u8>,
    body_remaining: i64,
    chunked: Option<ChunkedRequestTracker>,
}

impl<W: Write> OrderedRequestWriter<W> {
    #[must_use]
    pub fn new(inner: W, order: Arc<RequestHeaderOrder>) -> Self {
        Self {
            inner,
            order,
            header: Vec::new(),
            body_remaining: 0,
            chunked: None,
        }
    }

    #[must_use]
    pub fn get_ref(&self) -> &W {
        &self.inner
    }

    pub fn get_mut(&mut self) -> &mut W {
        &mut self.inner
    }

    pub fn into_inner(self) -> W {
        self.inner
    }

    pub fn write_request_bytes(&mut self, input: &[u8]) -> Result<usize, OrderedWriteError> {
        let original_len = input.len();
        let mut accepted = 0;
        let mut remaining = input.to_vec();

        while !remaining.is_empty() {
            if self.body_remaining > 0 {
                let body_bytes = remaining.len().min(self.body_remaining as usize);
                let written = match write_all_count(&mut self.inner, &remaining[..body_bytes]) {
                    Ok(written) => written,
                    Err((written, source)) => {
                        accepted += written;
                        self.body_remaining -= written as i64;
                        return Err(OrderedWriteError { accepted, source });
                    }
                };
                accepted += written;
                self.body_remaining -= written as i64;
                remaining.drain(..body_bytes);
                continue;
            }

            if let Some(tracker) = self.chunked.as_mut() {
                let mut preview = tracker.clone();
                let chunk_bytes = preview
                    .consume(&remaining)
                    .map_err(|source| OrderedWriteError { accepted, source })?
                    .0;
                let written = write_all_count(&mut self.inner, &remaining[..chunk_bytes]).map_err(
                    |(written, source)| {
                        if written > 0 {
                            let _ = tracker.consume(&remaining[..written]);
                        }
                        OrderedWriteError {
                            accepted: accepted + written,
                            source,
                        }
                    },
                )?;
                accepted += written;
                let completed = tracker
                    .consume(&remaining[..written])
                    .map_err(|source| OrderedWriteError { accepted, source })?
                    .1;
                if completed {
                    self.chunked = None;
                }
                remaining.drain(..chunk_bytes);
                continue;
            }

            let previous_header_len = self.header.len();
            self.header.extend_from_slice(&remaining);
            let Some(end) = find_bytes(&self.header, b"\r\n\r\n") else {
                if self.header.len() > MAX_BUFFERED_REQUEST_HEADER {
                    return Err(OrderedWriteError {
                        accepted,
                        source: invalid_data("request header exceeds 1048576 bytes"),
                    });
                }
                return Ok(original_len);
            };
            let header_end = end + 4;
            let body = self.header[header_end..].to_vec();
            let header = self.header[..header_end].to_vec();
            self.header.clear();
            let current_header_bytes = remaining
                .len()
                .min(header_end.saturating_sub(previous_header_len));
            let (ordered, content_length, chunked) =
                order_request_header(&header, self.order.as_ref());
            if let Err((_, source)) = write_all_count(&mut self.inner, &ordered) {
                return Err(OrderedWriteError {
                    accepted: original_len,
                    source,
                });
            }
            accepted += current_header_bytes;
            remaining = body;
            if chunked {
                self.chunked = Some(ChunkedRequestTracker::default());
            } else {
                self.body_remaining = content_length;
            }
        }
        Ok(original_len)
    }
}

impl<W: Write> Write for OrderedRequestWriter<W> {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        self.write_request_bytes(buffer)
            .map_err(|error| error.source)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

fn order_request_header(header: &[u8], order: &RequestHeaderOrder) -> (Vec<u8>, i64, bool) {
    let lines = split_crlf(&header[..header.len().saturating_sub(4)]);
    let header_lines = lines.get(1..).unwrap_or_default();
    let content_length = request_content_length(header_lines);
    let chunked = request_uses_chunked_encoding(header_lines);
    let Some(request_line) = lines.first() else {
        return (header.to_vec(), content_length, chunked);
    };
    let request_line = String::from_utf8_lossy(request_line);
    let mut parts = request_line.splitn(3, ' ');
    let (Some(method), Some(target), Some(_)) = (parts.next(), parts.next(), parts.next()) else {
        return (header.to_vec(), content_length, chunked);
    };
    let desired = order(method, target);
    if desired.is_empty() {
        return (header.to_vec(), content_length, chunked);
    }

    let mut used = vec![false; header_lines.len()];
    let mut output = Vec::with_capacity(header.len());
    output.extend_from_slice(lines[0]);
    output.extend_from_slice(b"\r\n");
    for name in desired {
        for (index, line) in header_lines.iter().enumerate() {
            if !used[index] && header_line_named(line, &name) {
                output.extend_from_slice(line);
                output.extend_from_slice(b"\r\n");
                used[index] = true;
            }
        }
    }
    for (index, line) in header_lines.iter().enumerate() {
        if !used[index] {
            output.extend_from_slice(line);
            output.extend_from_slice(b"\r\n");
        }
    }
    output.extend_from_slice(b"\r\n");
    (output, content_length, chunked)
}

fn header_line_named(line: &[u8], name: &str) -> bool {
    line.iter()
        .position(|byte| *byte == b':')
        .is_some_and(|colon| colon > 0 && line[..colon].eq_ignore_ascii_case(name.as_bytes()))
}

fn request_content_length(lines: &[&[u8]]) -> i64 {
    lines
        .iter()
        .find(|line| header_line_named(line, "Content-Length"))
        .and_then(|line| {
            line.iter()
                .position(|byte| *byte == b':')
                .map(|colon| &line[colon + 1..])
        })
        .and_then(|value| std::str::from_utf8(value).ok())
        .and_then(|value| value.trim().parse::<i64>().ok())
        .filter(|length| *length > 0)
        .unwrap_or(0)
}

fn request_uses_chunked_encoding(lines: &[&[u8]]) -> bool {
    lines.iter().any(|line| {
        header_line_named(line, "Transfer-Encoding")
            && line
                .iter()
                .position(|byte| *byte == b':')
                .and_then(|colon| std::str::from_utf8(&line[colon + 1..]).ok())
                .is_some_and(|value| {
                    value
                        .split(',')
                        .any(|part| part.trim().eq_ignore_ascii_case("chunked"))
                })
    })
}

#[derive(Clone, Default)]
struct ChunkedRequestTracker {
    state: ChunkState,
    line: Vec<u8>,
    data_remaining: i64,
    crlf_position: usize,
    trailers: Vec<u8>,
}

#[derive(Clone, Copy, Default)]
enum ChunkState {
    #[default]
    Size,
    Data,
    DataCrlf,
    Trailers,
}

impl ChunkedRequestTracker {
    fn consume(&mut self, data: &[u8]) -> io::Result<(usize, bool)> {
        let mut consumed = 0;
        while consumed < data.len() {
            match self.state {
                ChunkState::Size => {
                    self.line.push(data[consumed]);
                    consumed += 1;
                    if self.line.len() > MAX_BUFFERED_REQUEST_HEADER {
                        return Err(invalid_data("chunk size line exceeds limit"));
                    }
                    if !self.line.ends_with(b"\r\n") {
                        continue;
                    }
                    let raw = std::str::from_utf8(&self.line[..self.line.len() - 2])
                        .map_err(|_| invalid_data("invalid chunk size"))?;
                    let size_text = raw.split(';').next().unwrap_or_default().trim();
                    let size = i64::from_str_radix(size_text, 16)
                        .map_err(|_| invalid_data("invalid chunk size"))?;
                    if size < 0 {
                        return Err(invalid_data("invalid chunk size"));
                    }
                    self.line.clear();
                    if size == 0 {
                        self.state = ChunkState::Trailers;
                    } else {
                        self.data_remaining = size;
                        self.state = ChunkState::Data;
                    }
                }
                ChunkState::Data => {
                    let amount = (data.len() - consumed).min(self.data_remaining as usize);
                    consumed += amount;
                    self.data_remaining -= amount as i64;
                    if self.data_remaining == 0 {
                        self.crlf_position = 0;
                        self.state = ChunkState::DataCrlf;
                    }
                }
                ChunkState::DataCrlf => {
                    if data[consumed] != b"\r\n"[self.crlf_position] {
                        return Err(invalid_data("chunk data is missing CRLF terminator"));
                    }
                    consumed += 1;
                    self.crlf_position += 1;
                    if self.crlf_position == 2 {
                        self.state = ChunkState::Size;
                    }
                }
                ChunkState::Trailers => {
                    self.trailers.push(data[consumed]);
                    consumed += 1;
                    if self.trailers.len() > MAX_BUFFERED_REQUEST_HEADER {
                        return Err(invalid_data("chunk trailers exceed limit"));
                    }
                    if self.trailers == b"\r\n" || self.trailers.ends_with(b"\r\n\r\n") {
                        return Ok((consumed, true));
                    }
                }
            }
        }
        Ok((consumed, false))
    }
}

fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

fn split_crlf(mut bytes: &[u8]) -> Vec<&[u8]> {
    let mut lines = Vec::new();
    while let Some(index) = find_bytes(bytes, b"\r\n") {
        lines.push(&bytes[..index]);
        bytes = &bytes[index + 2..];
    }
    lines.push(bytes);
    lines
}

fn write_all_count(writer: &mut impl Write, mut data: &[u8]) -> Result<usize, (usize, io::Error)> {
    let mut total = 0;
    while !data.is_empty() {
        match writer.write(data) {
            Ok(0) => {
                return Err((
                    total,
                    io::Error::new(io::ErrorKind::WriteZero, "short write"),
                ))
            }
            Ok(written) => {
                total += written;
                data = &data[written..];
            }
            Err(error) => return Err((total, error)),
        }
    }
    Ok(total)
}

fn invalid_data(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}
