// ref: internal/logging/request_logger_format.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use super::request_logger::RequestLogRecord;
use crate::internal::api::middleware::request_logging::{
    mask_sensitive_header_value, mask_sensitive_query,
};
use flate2::read::{DeflateDecoder, GzDecoder, ZlibDecoder};
use std::collections::BTreeMap;
use std::io::{self, Cursor, Read, Write};
use std::time::SystemTime;

pub fn write_record(writer: &mut dyn Write, record: &RequestLogRecord) -> io::Result<()> {
    write_request_info(
        writer,
        &record.url,
        &record.method,
        &record.request_headers,
        &record.request_body,
        if record.streaming { "stream" } else { "http" },
        "http",
    )?;
    write_response_section(
        writer,
        record.status_code,
        &record.response_headers,
        &record.response_body,
        None,
        record.streaming,
    )
}

pub fn write_request_info(
    writer: &mut dyn Write,
    url: &str,
    method: &str,
    headers: &BTreeMap<String, Vec<String>>,
    body: &[u8],
    downstream_transport: &str,
    upstream_transport: &str,
) -> io::Result<()> {
    write_request_info_at(
        writer,
        url,
        method,
        headers,
        body,
        downstream_transport,
        upstream_transport,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn write_request_info_at(
    writer: &mut dyn Write,
    url: &str,
    method: &str,
    headers: &BTreeMap<String, Vec<String>>,
    body: &[u8],
    downstream_transport: &str,
    upstream_transport: &str,
    timestamp: Option<SystemTime>,
) -> io::Result<()> {
    let (path, query) = url.split_once('?').unwrap_or((url, ""));
    let masked_url = if query.is_empty() {
        path.to_owned()
    } else {
        format!("{path}?{}", mask_sensitive_query(query))
    };
    writeln!(writer, "=== REQUEST ===")?;
    if let Some(timestamp) = timestamp {
        let timestamp: chrono::DateTime<chrono::Utc> = timestamp.into();
        writeln!(writer, "Timestamp: {}", timestamp.to_rfc3339())?;
    }
    writeln!(writer, "{method} {masked_url}")?;
    writeln!(writer, "Downstream-Transport: {downstream_transport}")?;
    writeln!(writer, "Upstream-Transport: {upstream_transport}")?;
    write_headers(writer, headers)?;
    if !body.is_empty() {
        writeln!(writer)?;
        writer.write_all(body)?;
        write_section_spacing(writer, count_trailing_newlines(body))?;
    }
    Ok(())
}

pub fn write_api_section(
    writer: &mut dyn Write,
    header: &str,
    payload: &[u8],
    timestamp: Option<SystemTime>,
) -> io::Result<()> {
    if payload.is_empty() {
        return Ok(());
    }
    if !header.starts_with('\n') {
        writeln!(writer)?;
    }
    writeln!(writer, "{}", header.trim_end())?;
    if let Some(timestamp) = timestamp {
        let timestamp: chrono::DateTime<chrono::Utc> = timestamp.into();
        writeln!(writer, "Timestamp: {}", timestamp.to_rfc3339())?;
    }
    writer.write_all(payload)?;
    write_section_spacing(writer, count_trailing_newlines(payload))
}

pub fn write_response_section(
    writer: &mut dyn Write,
    status: u16,
    headers: &BTreeMap<String, Vec<String>>,
    response: &[u8],
    decompress_error: Option<&str>,
    streaming: bool,
) -> io::Result<()> {
    writeln!(writer, "=== RESPONSE ===")?;
    writeln!(writer, "Status: {status}")?;
    writeln!(writer, "Streaming: {streaming}")?;
    write_headers(writer, headers)?;
    if let Some(error) = decompress_error {
        writeln!(writer, "Decompression-Error: {error}")?;
    }
    if !response.is_empty() {
        writeln!(writer)?;
        writer.write_all(response)?;
        write_section_spacing(writer, count_trailing_newlines(response))?;
    }
    Ok(())
}

pub fn infer_downstream_transport(
    headers: &BTreeMap<String, Vec<String>>,
    websocket_timeline_has_payload: bool,
) -> &'static str {
    let upgrade = headers.iter().any(|(name, values)| {
        name.eq_ignore_ascii_case("upgrade")
            && values
                .iter()
                .any(|value| value.eq_ignore_ascii_case("websocket"))
    });
    if upgrade || websocket_timeline_has_payload {
        "websocket"
    } else {
        "http"
    }
}

pub fn infer_upstream_transport(
    api_request: &[u8],
    api_response: &[u8],
    api_websocket_timeline_has_payload: bool,
) -> &'static str {
    if api_websocket_timeline_has_payload {
        return "websocket";
    }
    let combined = [api_request, api_response].concat();
    if String::from_utf8_lossy(&combined)
        .to_ascii_lowercase()
        .contains("websocket")
    {
        "websocket"
    } else {
        "http"
    }
}

pub fn decompress_response(
    headers: &BTreeMap<String, Vec<String>>,
    response: &[u8],
) -> io::Result<Vec<u8>> {
    let encoding = headers
        .iter()
        .find(|(name, _)| name.eq_ignore_ascii_case("content-encoding"))
        .and_then(|(_, values)| values.first())
        .map(|value| value.trim().to_ascii_lowercase())
        .unwrap_or_default();
    let mut output = Vec::new();
    match encoding.as_str() {
        "" | "identity" => return Ok(response.to_vec()),
        "gzip" => {
            GzDecoder::new(response).read_to_end(&mut output)?;
        }
        "deflate" => {
            if ZlibDecoder::new(response).read_to_end(&mut output).is_err() {
                output.clear();
                DeflateDecoder::new(response).read_to_end(&mut output)?;
            }
        }
        "br" => {
            brotli::Decompressor::new(Cursor::new(response), 4096).read_to_end(&mut output)?;
        }
        "zstd" => {
            output = zstd::stream::decode_all(response)?;
        }
        other => {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                format!("unsupported content encoding: {other}"),
            ))
        }
    };
    Ok(output)
}

fn write_headers(
    writer: &mut dyn Write,
    headers: &BTreeMap<String, Vec<String>>,
) -> io::Result<()> {
    for (name, values) in headers {
        for value in values {
            writeln!(
                writer,
                "{name}: {}",
                mask_sensitive_header_value(name, value)
            )?;
        }
    }
    Ok(())
}

fn count_trailing_newlines(payload: &[u8]) -> usize {
    payload
        .iter()
        .rev()
        .take_while(|byte| **byte == b'\n')
        .count()
}

fn write_section_spacing(writer: &mut dyn Write, trailing_newlines: usize) -> io::Result<()> {
    for _ in trailing_newlines..2 {
        writer.write_all(b"\n")?;
    }
    Ok(())
}
