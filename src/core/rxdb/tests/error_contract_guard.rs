//! Guard: error handling in this crate must key on codes and structured
//! parameters — never on the human-readable `message` text.
//!
//! This exists because it was reintroduced in review. SYNC-A-C9a suppressed
//! expected peer-teardown errors by matching their message strings:
//!
//! ```ignore
//! matches!(
//!     error.parameters().get("message").and_then(Value::as_str),
//!     Some("unknown or unopened peer" | "WebRTC send queue result dropped")
//! )
//! ```
//!
//! That policy silently becomes wrong the moment somebody rewords a message —
//! the errors keep flowing, the suppression stops matching, and nothing fails.
//! It was replaced by a structured `expectedPeerTeardown` parameter. This guard
//! keeps the pattern from coming back.
//!
//! Reading `message` is fine — logging it, forwarding it, putting it in a
//! response. What is banned is *deciding* on it: comparing it against string
//! literals to classify an error.
//!
//! If you genuinely need a message comparison (a foreign library that offers
//! nothing else), put `// error-text-match: <reason>` on the line or in the four
//! lines above it. The marker makes the exception reviewable instead of silent.

use std::fs;
use std::path::{Path, PathBuf};

/// Window (in lines) around a `get("message")` hit that is inspected for a
/// literal comparison. Four lines covers the usual rustfmt-wrapped
/// `matches!(...)` / `== "..."` shapes without reaching into unrelated code.
const WINDOW: usize = 4;

const ESCAPE_MARKER: &str = "error-text-match:";

fn rust_sources(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            rust_sources(&path, out);
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            out.push(path);
        }
    }
}

/// Does this window decide something based on the message text?
fn compares_against_literal(window: &str) -> bool {
    // `matches!(msg, Some("literal"))`, `msg == "literal"`,
    // `msg.contains("literal")`, `msg.starts_with("literal")`.
    (window.contains("matches!(") && window.contains('"'))
        || window.contains("== \"")
        || window.contains("!= \"")
        || window.contains(".contains(\"")
        || window.contains(".starts_with(\"")
        || window.contains(".ends_with(\"")
}

#[test]
fn errors_are_classified_by_code_not_by_message_text() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut files = Vec::new();
    rust_sources(&root, &mut files);
    assert!(
        !files.is_empty(),
        "guard found no sources under {} — the walk is broken, not the code",
        root.display()
    );

    let mut violations = Vec::new();

    for file in &files {
        let Ok(text) = fs::read_to_string(file) else {
            continue;
        };
        let lines: Vec<&str> = text.lines().collect();

        for (index, line) in lines.iter().enumerate() {
            if !line.contains("get(\"message\")") {
                continue;
            }

            let start = index.saturating_sub(WINDOW);
            let end = (index + WINDOW + 1).min(lines.len());
            let window = lines[start..end].join("\n");

            if window.contains(ESCAPE_MARKER) {
                continue;
            }
            if !compares_against_literal(&window) {
                continue;
            }

            let relative = file
                .strip_prefix(env!("CARGO_MANIFEST_DIR"))
                .unwrap_or(file);
            violations.push(format!(
                "{}:{}: {}",
                relative.display(),
                index + 1,
                line.trim()
            ));
        }
    }

    assert!(
        violations.is_empty(),
        "error classification must use codes or structured parameters, not the \
         message text. A policy built on wording breaks silently when the wording \
         changes.\n\nOffending sites:\n  {}\n\nFix: add a structured parameter (see \
         EXPECTED_PEER_TEARDOWN_PARAM in connection_handler_rs.rs) or, if a text \
         comparison is genuinely unavoidable, annotate it with `// {} <reason>`.",
        violations.join("\n  "),
        ESCAPE_MARKER
    );
}
