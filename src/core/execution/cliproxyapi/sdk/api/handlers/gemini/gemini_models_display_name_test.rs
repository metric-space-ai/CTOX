// ref: sdk/api/handlers/gemini/gemini_models_display_name_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use super::*;

#[test]
fn gemini_models_preserve_configured_display_name_and_fill_defaults() {
    let models = vec![serde_json::Map::from_iter([
        ("name".to_owned(), serde_json::json!("gemini-test")),
        (
            "displayName".to_owned(),
            serde_json::json!("Configured Gemini Name"),
        ),
    ])];
    let normalized = normalize_gemini_models(&models);
    assert_eq!(normalized[0]["name"], "models/gemini-test");
    assert_eq!(normalized[0]["displayName"], "Configured Gemini Name");
    assert_eq!(normalized[0]["description"], "gemini-test");
    assert_eq!(
        normalized[0]["supportedGenerationMethods"],
        serde_json::json!(["generateContent"])
    );
    assert_eq!(models[0]["name"], "gemini-test");
}

#[test]
fn action_parser_rejects_unknown_or_malformed_routes() {
    assert_eq!(
        GeminiAction::parse("/gemini-2.5-pro:countTokens", "/v1beta/models").unwrap(),
        GeminiAction::CountTokens {
            model: "gemini-2.5-pro".to_owned()
        }
    );
    let error = GeminiAction::parse("gemini-2.5-pro:unknown", "/bad").unwrap_err();
    assert_eq!(error.status, 404);
}
