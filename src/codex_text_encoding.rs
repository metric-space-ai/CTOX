use chardetng::EncodingDetector;
use encoding_rs::Encoding;
use encoding_rs::IBM866;
use encoding_rs::WINDOWS_1252;

const WINDOWS_1252_PUNCT_BYTES: [u8; 8] = [
    0x91, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x99,
];

pub fn bytes_to_string_smart(bytes: &[u8]) -> String {
    if bytes.is_empty() {
        return String::new();
    }

    if let Ok(utf8) = std::str::from_utf8(bytes) {
        return utf8.to_owned();
    }

    let encoding = detect_encoding(bytes);
    decode_bytes(bytes, encoding)
}

fn detect_encoding(bytes: &[u8]) -> &'static Encoding {
    let mut detector = EncodingDetector::new();
    detector.feed(bytes, true);
    let (encoding, _confident) = detector.guess_assess(None, true);

    if encoding == IBM866 && looks_like_windows_1252_punctuation(bytes) {
        return WINDOWS_1252;
    }

    encoding
}

fn decode_bytes(bytes: &[u8], encoding: &'static Encoding) -> String {
    let (decoded, _, had_errors) = encoding.decode(bytes);
    if had_errors {
        return String::from_utf8_lossy(bytes).into_owned();
    }
    decoded.into_owned()
}

fn looks_like_windows_1252_punctuation(bytes: &[u8]) -> bool {
    let mut saw_extended_punctuation = false;
    let mut saw_ascii_word = false;

    for &byte in bytes {
        if byte >= 0xA0 {
            return false;
        }
        if (0x80..=0x9F).contains(&byte) {
            if !WINDOWS_1252_PUNCT_BYTES.contains(&byte) {
                return false;
            }
            saw_extended_punctuation = true;
        }
        if byte.is_ascii_alphabetic() {
            saw_ascii_word = true;
        }
    }

    saw_extended_punctuation && saw_ascii_word
}
