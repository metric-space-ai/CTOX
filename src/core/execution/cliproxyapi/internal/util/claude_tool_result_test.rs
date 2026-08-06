// ref: internal/util/claude_tool_result_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use super::convert_claude_tool_result_content;

#[test]
fn pinned_content_shapes_match_upstream() {
    let cases = [
        (r#"{"content":"alpha"}"#, "alpha", false, 0),
        (
            r#"{"content":[{"type":"text","text":"alpha"}]}"#,
            r#"{"type":"text","text":"alpha"}"#,
            true,
            0,
        ),
        (
            r#"{"content":[{"type":"text","text":"alpha"},{"type":"text","text":"beta"}]}"#,
            r#"[{"type":"text","text":"alpha"},{"type":"text","text":"beta"}]"#,
            true,
            0,
        ),
        (
            r#"{"content":[{"type":"text","text":"alpha"},{"type":"image","source":{"type":"base64","media_type":"image/png","data":"aGVsbG8="}}]}"#,
            r#"{"type":"text","text":"alpha"}"#,
            true,
            1,
        ),
        (
            r#"{"content":[{"type":"image","source":{"type":"base64","media_type":"image/png","data":"aGVsbG8="}}]}"#,
            "",
            false,
            1,
        ),
        (
            r#"{"content":[{"type":"image","source":{"type":"base64","media_type":"image/png"}}]}"#,
            "",
            false,
            0,
        ),
        (r#"{"content":{"foo":"bar"}}"#, r#"{"foo":"bar"}"#, true, 0),
        (
            r#"{"content":{"type":"image","source":{"type":"base64","media_type":"image/png","data":"aGVsbG8="}}}"#,
            "",
            false,
            1,
        ),
        (r#"{}"#, "", false, 0),
    ];

    for (wrapper, expected_result, expected_raw, expected_images) in cases {
        let content = gjson::get(wrapper, "content");
        let result = convert_claude_tool_result_content(&content);
        assert_eq!(result.result, expected_result, "wrapper={wrapper}");
        assert_eq!(result.result_is_raw, expected_raw, "wrapper={wrapper}");
        assert_eq!(result.images.len(), expected_images, "wrapper={wrapper}");
    }
}

#[test]
fn image_fields_match_upstream() {
    let content = gjson::get(
        r#"{"content":[{"type":"image","source":{"type":"base64","media_type":"image/png","data":"aGVsbG8="}}]}"#,
        "content",
    );
    let result = convert_claude_tool_result_content(&content);
    assert_eq!(result.images[0].mime_type, "image/png");
    assert_eq!(result.images[0].data, "aGVsbG8=");
}

#[test]
fn preserves_raw_json_and_decodes_strings_once() {
    for (wrapper, expected, raw) in [
        (r#"{"content":"a\\n\"b"}"#, "a\\n\"b", false),
        (r#"{"content":null}"#, "null", true),
        (r#"{"content":false}"#, "false", true),
        (r#"{"content":1.2300}"#, "1.2300", true),
        (
            r#"{"content":["text",null,1.00]}"#,
            r#"["text",null,1.00]"#,
            true,
        ),
    ] {
        let content = gjson::get(wrapper, "content");
        let result = convert_claude_tool_result_content(&content);
        assert_eq!(result.result, expected, "wrapper={wrapper}");
        assert_eq!(result.result_is_raw, raw, "wrapper={wrapper}");
    }
}

#[test]
fn only_exact_base64_images_are_split_and_empty_data_is_dropped() {
    let wrapper = r#"{"content":[
      {"type":"image","source":{"type":"url","data":"keep"}},
      {"type":"Image","source":{"type":"base64","data":"keep"}},
      {"type":"image","source":{"type":"base64","media_type":"","data":123}},
      {"type":"image","source":{"type":"base64","media_type":"image/png","data":""}}
    ]}"#;
    let content = gjson::get(wrapper, "content");
    let result = convert_claude_tool_result_content(&content);
    assert_eq!(result.images.len(), 1);
    assert_eq!(result.images[0].mime_type, "");
    assert_eq!(result.images[0].data, "123");
    assert!(result.result_is_raw);
    let retained: serde_json::Value = serde_json::from_str(&result.result).unwrap();
    assert_eq!(retained.as_array().unwrap().len(), 2);
}
