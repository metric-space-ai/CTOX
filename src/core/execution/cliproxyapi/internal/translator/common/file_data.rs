// ref: internal/translator/common/file_data.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use crate::internal::misc::mime_type::mime_type_for_extension;

pub fn normalize_openai_file_data(
    filename: &str,
    fallback_mime_type: &str,
    file_data: &str,
) -> Option<(String, String)> {
    if file_data.is_empty() {
        return None;
    }
    let fallback = if fallback_mime_type.is_empty() {
        let extension = std::path::Path::new(filename)
            .extension()
            .and_then(std::ffi::OsStr::to_str)
            .unwrap_or("")
            .to_ascii_lowercase();
        mime_type_for_extension(&extension).unwrap_or("").to_owned()
    } else {
        fallback_mime_type.to_owned()
    };
    if !file_data
        .get(..5)
        .is_some_and(|prefix| prefix.eq_ignore_ascii_case("data:"))
    {
        return (!fallback.is_empty()).then(|| (fallback, file_data.to_owned()));
    }
    let (metadata, payload) = file_data[5..].split_once(',')?;
    if payload.is_empty() {
        return None;
    }
    let mut fields = metadata.split(';');
    let mime = fields.next()?.trim();
    if mime.is_empty() || !fields.any(|field| field.trim().eq_ignore_ascii_case("base64")) {
        return None;
    }
    Some((mime.to_owned(), payload.to_owned()))
}

#[cfg(test)]
mod tests {
    use super::normalize_openai_file_data;

    #[test]
    fn normalizes_data_urls_raw_data_and_failures() {
        assert_eq!(
            normalize_openai_file_data(
                "test.txt",
                "",
                "DATA:application/pdf;charset=binary;BASE64,JVBERi0xLjQK"
            ),
            Some(("application/pdf".into(), "JVBERi0xLjQK".into()))
        );
        assert_eq!(
            normalize_openai_file_data("TEST.PDF", "", "JVBERi0xLjQK"),
            Some(("application/pdf".into(), "JVBERi0xLjQK".into()))
        );
        assert!(normalize_openai_file_data("test", "", "JVBERi0xLjQK").is_none());
        assert!(normalize_openai_file_data("test.pdf", "", "data:application/pdf,x").is_none());
    }
}
