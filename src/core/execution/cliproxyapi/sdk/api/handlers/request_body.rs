// ref: sdk/api/handlers/request_body.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use std::fmt;
use std::io::Read;

pub const MAX_DECODED_REQUEST_BODY_BYTES: usize = 2 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestBodyDecodeError {
    UnsupportedEncoding,
    InvalidZstd,
    DecodedBodyTooLarge,
}

impl fmt::Display for RequestBodyDecodeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::UnsupportedEncoding => "request content encoding is unsupported",
            Self::InvalidZstd => "zstd request body is invalid",
            Self::DecodedBodyTooLarge => "decoded request body is too large",
        })
    }
}

impl std::error::Error for RequestBodyDecodeError {}

// ref: sdk/api/handlers/request_body.go:13-37
pub fn read_request_body(
    raw: &[u8],
    content_encoding: Option<&str>,
) -> Result<Vec<u8>, RequestBodyDecodeError> {
    let encoding = content_encoding.unwrap_or_default().trim();
    if encoding.is_empty() || encoding.eq_ignore_ascii_case("identity") {
        return Ok(raw.to_vec());
    }
    match decode_request_body(raw, encoding) {
        Ok(decoded) => Ok(decoded),
        Err(_) if serde_json::from_slice::<serde_json::Value>(raw).is_ok() => Ok(raw.to_vec()),
        Err(error) => Err(error),
    }
}

// ref: sdk/api/handlers/request_body.go:39-60
fn decode_request_body(
    raw: &[u8],
    content_encoding: &str,
) -> Result<Vec<u8>, RequestBodyDecodeError> {
    let mut body = raw.to_vec();
    for encoding in content_encoding.split(',').rev() {
        match encoding.trim().to_ascii_lowercase().as_str() {
            "" | "identity" => {}
            "zstd" => body = decode_zstd_request_body(&body)?,
            _ => return Err(RequestBodyDecodeError::UnsupportedEncoding),
        }
    }
    Ok(body)
}

fn decode_zstd_request_body(raw: &[u8]) -> Result<Vec<u8>, RequestBodyDecodeError> {
    let decoder =
        zstd::stream::read::Decoder::new(raw).map_err(|_| RequestBodyDecodeError::InvalidZstd)?;
    let mut bounded = decoder.take((MAX_DECODED_REQUEST_BODY_BYTES + 1) as u64);
    let mut decoded = Vec::new();
    bounded
        .read_to_end(&mut decoded)
        .map_err(|_| RequestBodyDecodeError::InvalidZstd)?;
    if decoded.len() > MAX_DECODED_REQUEST_BODY_BYTES {
        return Err(RequestBodyDecodeError::DecodedBodyTooLarge);
    }
    Ok(decoded)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_and_absent_encoding_preserve_bytes() {
        let raw = b"{ \"model\" : \"gpt\" }";
        assert_eq!(read_request_body(raw, None).unwrap(), raw);
        assert_eq!(read_request_body(raw, Some(" Identity ")).unwrap(), raw);
    }

    #[test]
    fn decodes_zstd_and_stacked_encodings_in_reverse_order() {
        let raw = br#"{"model":"gpt","input":"hello"}"#;
        let once = zstd::stream::encode_all(raw.as_slice(), 1).unwrap();
        assert_eq!(read_request_body(&once, Some("zstd")).unwrap(), raw);
        let twice = zstd::stream::encode_all(once.as_slice(), 1).unwrap();
        assert_eq!(
            read_request_body(&twice, Some(" zstd, ZSTD ")).unwrap(),
            raw
        );
    }

    #[test]
    fn valid_raw_json_survives_an_incorrect_encoding_label() {
        let raw = br#"{"model":"gpt","input":[]}"#;
        assert_eq!(read_request_body(raw, Some("gzip")).unwrap(), raw);
        assert_eq!(read_request_body(raw, Some("zstd")).unwrap(), raw);
    }

    #[test]
    fn invalid_or_unsupported_encoded_data_fails_closed() {
        assert_eq!(
            read_request_body(b"not-json", Some("gzip")),
            Err(RequestBodyDecodeError::UnsupportedEncoding)
        );
        assert_eq!(
            read_request_body(b"not-zstd", Some("zstd")),
            Err(RequestBodyDecodeError::InvalidZstd)
        );
    }

    #[test]
    fn decoded_body_limit_rejects_high_ratio_payload() {
        let oversized = vec![b'x'; MAX_DECODED_REQUEST_BODY_BYTES + 1];
        let compressed = zstd::stream::encode_all(oversized.as_slice(), 1).unwrap();
        assert_eq!(
            read_request_body(&compressed, Some("zstd")),
            Err(RequestBodyDecodeError::DecodedBodyTooLarge)
        );
    }
}
