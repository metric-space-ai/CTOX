// ref: internal/auth/claude/oauth_response.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// Candidate delta evidence: internal/auth/claude/oauth_response.go
// License: MIT (upstream); modifications AGPL-3.0-only

use std::fmt;
use std::io::{Cursor, Read};

use flate2::read::{DeflateDecoder, GzDecoder, ZlibDecoder};
use weezl::{decode::Decoder as LzwDecoder, BitOrder};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClaudeOAuthResponseError {
    UnsupportedEncoding(String),
    Decode { encoding: String },
}

impl fmt::Display for ClaudeOAuthResponseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedEncoding(encoding) => write!(
                formatter,
                "decode Claude OAuth response: unsupported content encoding {encoding:?}"
            ),
            Self::Decode { encoding } => {
                write!(formatter, "decode Claude OAuth {encoding} response")
            }
        }
    }
}

impl std::error::Error for ClaudeOAuthResponseError {}

pub fn decode_claude_oauth_response_body(
    mut encoded: Vec<u8>,
    content_encoding_headers: &[String],
) -> Result<Vec<u8>, ClaudeOAuthResponseError> {
    let encodings: Vec<&str> = content_encoding_headers
        .iter()
        .flat_map(|header| header.split(','))
        .map(str::trim)
        .filter(|encoding| !encoding.is_empty() && !encoding.eq_ignore_ascii_case("identity"))
        .collect();

    for encoding in encodings.into_iter().rev() {
        encoded = decode_claude_oauth_encoding(encoded, &encoding.to_ascii_lowercase())?;
    }
    Ok(encoded)
}

fn decode_claude_oauth_encoding(
    encoded: Vec<u8>,
    encoding: &str,
) -> Result<Vec<u8>, ClaudeOAuthResponseError> {
    match encoding {
        "gzip" => read_decoder(GzDecoder::new(Cursor::new(encoded)), encoding),
        "deflate" => {
            let mut zlib = ZlibDecoder::new(Cursor::new(encoded.as_slice()));
            let mut decoded = Vec::new();
            if zlib.read_to_end(&mut decoded).is_ok() {
                return Ok(decoded);
            }
            read_decoder(DeflateDecoder::new(Cursor::new(encoded)), encoding)
        }
        "br" => read_decoder(
            brotli::Decompressor::new(Cursor::new(encoded), 4096),
            encoding,
        ),
        "compress" => LzwDecoder::new(BitOrder::Msb, 8)
            .decode(&encoded)
            .map_err(|_| ClaudeOAuthResponseError::Decode {
                encoding: encoding.to_owned(),
            }),
        _ => Err(ClaudeOAuthResponseError::UnsupportedEncoding(
            encoding.to_owned(),
        )),
    }
}

fn read_decoder(
    mut reader: impl Read,
    encoding: &str,
) -> Result<Vec<u8>, ClaudeOAuthResponseError> {
    let mut decoded = Vec::new();
    reader
        .read_to_end(&mut decoded)
        .map_err(|_| ClaudeOAuthResponseError::Decode {
            encoding: encoding.to_owned(),
        })?;
    Ok(decoded)
}
