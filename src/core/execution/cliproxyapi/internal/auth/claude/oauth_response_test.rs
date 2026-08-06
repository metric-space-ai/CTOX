// ref: internal/auth/claude/oauth_response_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// Candidate delta evidence: internal/auth/claude/oauth_response_test.go
// License: MIT (upstream); modifications AGPL-3.0-only

use std::io::Write;

use flate2::write::{DeflateEncoder, GzEncoder, ZlibEncoder};
use flate2::Compression;
use weezl::{encode::Encoder as LzwEncoder, BitOrder};

use super::oauth_response::{decode_claude_oauth_response_body, ClaudeOAuthResponseError};

const PAYLOAD: &[u8] = br#"{"account":{"uuid":"test"}}"#;

fn gzip(input: &[u8]) -> Vec<u8> {
    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(input).unwrap();
    encoder.finish().unwrap()
}

fn zlib(input: &[u8]) -> Vec<u8> {
    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(input).unwrap();
    encoder.finish().unwrap()
}

fn raw_deflate(input: &[u8]) -> Vec<u8> {
    let mut encoder = DeflateEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(input).unwrap();
    encoder.finish().unwrap()
}

fn brotli(input: &[u8]) -> Vec<u8> {
    let mut encoded = Vec::new();
    {
        let mut encoder = brotli::CompressorWriter::new(&mut encoded, 4096, 5, 22);
        encoder.write_all(input).unwrap();
    }
    encoded
}

#[test]
fn decodes_stacked_repeated_headers_in_reverse_order() {
    let encoded = brotli(&gzip(PAYLOAD));
    let headers = vec!["gzip".to_owned(), "br".to_owned()];
    assert_eq!(
        decode_claude_oauth_response_body(encoded, &headers).unwrap(),
        PAYLOAD
    );
}

#[test]
fn decodes_all_advertised_encodings() {
    let cases = [
        ("gzip", gzip(PAYLOAD)),
        ("br", brotli(PAYLOAD)),
        ("deflate", zlib(PAYLOAD)),
        ("deflate", raw_deflate(PAYLOAD)),
        (
            "compress",
            LzwEncoder::new(BitOrder::Msb, 8).encode(PAYLOAD).unwrap(),
        ),
    ];
    for (encoding, encoded) in cases {
        assert_eq!(
            decode_claude_oauth_response_body(encoded, &[encoding.to_owned()]).unwrap(),
            PAYLOAD,
            "encoding {encoding}"
        );
    }
}

#[test]
fn flattens_comma_separated_values_and_ignores_identity() {
    let encoded = brotli(&gzip(PAYLOAD));
    assert_eq!(
        decode_claude_oauth_response_body(
            encoded,
            &["identity, gzip, br".to_owned(), "identity".to_owned()]
        )
        .unwrap(),
        PAYLOAD
    );
}

#[test]
fn unsupported_or_malformed_encodings_fail_closed() {
    assert_eq!(
        decode_claude_oauth_response_body(PAYLOAD.to_vec(), &["zstd".to_owned()]),
        Err(ClaudeOAuthResponseError::UnsupportedEncoding(
            "zstd".to_owned()
        ))
    );
    assert_eq!(
        decode_claude_oauth_response_body(PAYLOAD.to_vec(), &["gzip".to_owned()]),
        Err(ClaudeOAuthResponseError::Decode {
            encoding: "gzip".to_owned()
        })
    );
}
