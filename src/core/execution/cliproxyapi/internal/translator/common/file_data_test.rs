// ref: internal/translator/common/file_data_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use super::normalize_openai_file_data;

#[test]
fn normalize_openai_file_data_matches_all_pinned_cases() {
    let valid = [
        (
            "test.pdf",
            "",
            "data:application/pdf;base64,JVBERi0xLjQK",
            "application/pdf",
        ),
        (
            "test.txt",
            "",
            "data:application/pdf;charset=binary;BASE64,JVBERi0xLjQK",
            "application/pdf",
        ),
        (
            "test.pdf",
            "",
            "DATA:application/pdf;base64,JVBERi0xLjQK",
            "application/pdf",
        ),
        ("TEST.PDF", "", "JVBERi0xLjQK", "application/pdf"),
        ("", "application/pdf", "JVBERi0xLjQK", "application/pdf"),
    ];
    for (filename, fallback, data, expected_mime) in valid {
        assert_eq!(
            normalize_openai_file_data(filename, fallback, data),
            Some((expected_mime.to_owned(), "JVBERi0xLjQK".to_owned()))
        );
    }

    for (filename, data) in [
        ("test.pdf", ""),
        ("test", "JVBERi0xLjQK"),
        ("test.pdf", "data:application/pdf,JVBERi0xLjQK"),
        ("test.pdf", "data:;base64,JVBERi0xLjQK"),
        ("test.pdf", "data:application/pdf;base64,"),
    ] {
        assert!(normalize_openai_file_data(filename, "", data).is_none());
    }
}
