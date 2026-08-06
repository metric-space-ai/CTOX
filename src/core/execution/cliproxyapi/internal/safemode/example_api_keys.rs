// ref: internal/safemode/example_api_keys.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::HashSet;

use crate::internal::htmlsanitize;

const EXAMPLE_API_KEYS: [&str; 3] = ["your-api-key-1", "your-api-key-2", "your-api-key-3"];

pub fn example_api_keys(keys: &[String]) -> Vec<String> {
    let mut seen = HashSet::with_capacity(EXAMPLE_API_KEYS.len());
    let mut matches = Vec::new();
    for key in keys {
        let key = key.trim();
        if EXAMPLE_API_KEYS.contains(&key) && seen.insert(key.to_owned()) {
            matches.push(key.to_owned());
        }
    }
    matches
}

pub fn has_example_api_keys(keys: &[String]) -> bool {
    !example_api_keys(keys).is_empty()
}

pub fn example_api_key_warning_page_html(keys: &[String], management_path: &str) -> String {
    let mut page = String::from(
        r#"<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Example API key detected</title><style>body{margin:0;font-family:Arial,sans-serif;background:#f6f8fa;color:#1f2328}.wrap{max-width:760px;margin:12vh auto;padding:0 24px}.panel{background:#fff;border:1px solid #d0d7de;border-radius:8px;padding:28px;box-shadow:0 8px 24px rgba(140,149,159,.2)}h1{margin:0 0 12px;font-size:28px;line-height:1.25}p{font-size:16px;line-height:1.55}code{background:#f6f8fa;border:1px solid #d0d7de;border-radius:4px;padding:2px 5px}.keys{margin:16px 0;padding-left:22px}.actions{margin-top:24px}.button{display:inline-block;border-radius:6px;background:#0969da;color:#fff;text-decoration:none;font-weight:600;padding:10px 16px}.button:hover{background:#0759b8}</style></head><body><main class="wrap"><section class="panel"><h1>Example API key detected</h1><p>Proxy API endpoints are disabled because the top-level <code>api-keys</code> configuration still contains template values.</p>"#,
    );
    if !keys.is_empty() {
        page.push_str(r#"<p>Replace these values before using the proxy:</p><ul class="keys">"#);
        for key in keys {
            page.push_str("<li><code>");
            page.push_str(&htmlsanitize::string(key));
            page.push_str("</code></li>");
        }
        page.push_str("</ul>");
    }
    page.push_str("<p>Set strong random API keys, then retry the proxy endpoint.</p>");
    let management_path = management_path.trim();
    if !management_path.is_empty() {
        page.push_str(r#"<div class="actions"><a class="button" href=""#);
        page.push_str(&htmlsanitize::string(management_path));
        page.push_str(r#"">Open Management</a></div>"#);
    }
    page.push_str("</section></main></body></html>");
    page
}
