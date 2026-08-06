// ref: internal/util/image.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::fmt;

use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use base64::Engine as _;

/// PNG encoding failure for [`create_white_image_base64`].
#[derive(Debug)]
pub struct WhiteImageEncodingError(png::EncodingError);

impl fmt::Display for WhiteImageEncodingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("encode white PNG failed")
    }
}

impl std::error::Error for WhiteImageEncodingError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.0)
    }
}

/// Creates a fully opaque white RGBA PNG and returns bare standard base64.
/// Unknown aspect-ratio strings intentionally fall back to the upstream 1:1
/// dimensions. The decoded image is equivalent to Go's `image/png` output;
/// compressed PNG bytes are encoder-specific and are not a stable wire
/// contract in the upstream API.
pub fn create_white_image_base64(aspect_ratio: &str) -> Result<String, WhiteImageEncodingError> {
    let (width, height) = dimensions(aspect_ratio);
    let pixels = vec![u8::MAX; width as usize * height as usize * 4];
    let mut encoded = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut encoded, width, height);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        encoder
            .write_header()
            .and_then(|mut writer| writer.write_image_data(&pixels))
            .map_err(WhiteImageEncodingError)?;
    }
    Ok(BASE64_STANDARD.encode(encoded))
}

fn dimensions(aspect_ratio: &str) -> (u32, u32) {
    match aspect_ratio {
        "1:1" => (1024, 1024),
        "2:3" => (832, 1248),
        "3:2" => (1248, 832),
        "3:4" => (864, 1184),
        "4:3" => (1184, 864),
        "4:5" => (896, 1152),
        "5:4" => (1152, 896),
        "9:16" => (768, 1344),
        "16:9" => (1344, 768),
        "21:9" => (1536, 672),
        _ => (1024, 1024),
    }
}
