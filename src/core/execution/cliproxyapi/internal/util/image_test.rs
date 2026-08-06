// ref: internal/util/image.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::io::Cursor;

use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use base64::Engine as _;

use super::create_white_image_base64;

#[test]
fn every_pinned_aspect_ratio_has_exact_dimensions_and_white_rgba_pixels() {
    for (aspect, width, height) in [
        ("1:1", 1024, 1024),
        ("2:3", 832, 1248),
        ("3:2", 1248, 832),
        ("3:4", 864, 1184),
        ("4:3", 1184, 864),
        ("4:5", 896, 1152),
        ("5:4", 1152, 896),
        ("9:16", 768, 1344),
        ("16:9", 1344, 768),
        ("21:9", 1536, 672),
    ] {
        verify_image(aspect, width, height);
    }
}

#[test]
fn unknown_and_empty_ratios_use_default_and_output_is_bare_deterministic_base64() {
    let unknown = create_white_image_base64("unknown").unwrap();
    let empty = create_white_image_base64("").unwrap();
    let square = create_white_image_base64("1:1").unwrap();
    assert_eq!(unknown, square);
    assert_eq!(empty, square);
    assert!(!square.starts_with("data:"));
    assert_eq!(
        BASE64_STANDARD.decode(&square).unwrap()[..8],
        *b"\x89PNG\r\n\x1a\n"
    );
}

fn verify_image(aspect: &str, expected_width: u32, expected_height: u32) {
    let encoded = create_white_image_base64(aspect).unwrap();
    let bytes = BASE64_STANDARD.decode(encoded).unwrap();
    let decoder = png::Decoder::new(Cursor::new(bytes));
    let mut reader = decoder.read_info().unwrap();
    assert_eq!(reader.info().width, expected_width, "aspect={aspect}");
    assert_eq!(reader.info().height, expected_height, "aspect={aspect}");
    assert_eq!(reader.info().color_type, png::ColorType::Rgba);
    assert_eq!(reader.info().bit_depth, png::BitDepth::Eight);
    let mut pixels = vec![0; reader.output_buffer_size().unwrap()];
    let frame = reader.next_frame(&mut pixels).unwrap();
    assert_eq!(
        frame.buffer_size(),
        expected_width as usize * expected_height as usize * 4
    );
    assert!(pixels[..frame.buffer_size()]
        .iter()
        .all(|byte| *byte == u8::MAX));
}
