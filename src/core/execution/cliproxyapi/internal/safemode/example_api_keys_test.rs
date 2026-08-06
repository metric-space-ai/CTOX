// ref: internal/safemode/example_api_keys_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use super::example_api_keys::{
    example_api_key_warning_page_html, example_api_keys, has_example_api_keys,
};

fn strings(values: &[&str]) -> Vec<String> {
    values.iter().map(|value| (*value).to_owned()).collect()
}

#[test]
fn detects_only_distinct_template_values_in_input_order() {
    let keys = strings(&[
        " real-key ",
        " your-api-key-1 ",
        "your-api-key",
        "change-me",
        "your-api-key-2",
        "your-api-key-2",
        "your-api-key-3",
    ]);
    assert_eq!(
        example_api_keys(&keys),
        strings(&["your-api-key-1", "your-api-key-2", "your-api-key-3"])
    );
    assert!(has_example_api_keys(&keys));
}

#[test]
fn similar_values_are_not_treated_as_examples() {
    let keys = strings(&[
        "your-api-key",
        "change-me",
        "changeme",
        "your-api-key-4",
        "my-your-api-key-1",
    ]);
    assert!(example_api_keys(&keys).is_empty());
    assert!(!has_example_api_keys(&keys));
}

#[test]
fn warning_page_includes_management_button_without_local_config_path() {
    let body = example_api_key_warning_page_html(
        &strings(&["your-api-key-1"]),
        "/management.html?safe-mode=configure",
    );
    for expected in [
        "Example API key detected",
        "your-api-key-1",
        "Open Management",
        r#"href="/management.html?safe-mode=configure""#,
        "Proxy API endpoints are disabled",
    ] {
        assert!(body.contains(expected), "missing {expected:?}: {body}");
    }
    assert!(!body.contains(r#"class="path""#));
}

#[test]
fn warning_page_escapes_keys_and_management_href_like_go_html() {
    let body = example_api_key_warning_page_html(
        &strings(&[r#"<&'">"#]),
        r#" /manage?next=<unsafe>&q="x" "#,
    );
    assert!(body.contains("&lt;&amp;&#39;&#34;&gt;"));
    assert!(body.contains(r#"href="/manage?next=&lt;unsafe&gt;&amp;q=&#34;x&#34;""#));
    assert!(!body.contains("<unsafe>"));
}

#[test]
fn empty_inputs_omit_optional_list_and_button() {
    let body = example_api_key_warning_page_html(&[], " \t");
    assert!(!body.contains(r#"<ul class="keys">"#));
    assert!(!body.contains("Open Management"));
    assert!(example_api_keys(&[]).is_empty());
}
