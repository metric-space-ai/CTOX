// ref: sdk/cliproxy/auth/force_mapping_live_fixtures_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

pub(super) const LIVE_FIXTURES: &[(&str, &str, &str)] = &[
    (
        "gpt-5.4-fast",
        "gpt-5.4",
        r#"{"type":"response.created","response":{"model":"gpt-5.4"}}"#,
    ),
    (
        "claude-haiku-4-5",
        "gemini-3-flash",
        r#"{"type":"message_start","message":{"model":"gemini-3-flash"}}"#,
    ),
    (
        "k2.5",
        "kimi-k2.5",
        r#"{"type":"message_start","message":{"model":"kimi-k2.5"}}"#,
    ),
    (
        "grok-latest",
        "grok-4.3",
        r#"{"type":"message_start","message":{"model":"grok-4.3"}}"#,
    ),
];
