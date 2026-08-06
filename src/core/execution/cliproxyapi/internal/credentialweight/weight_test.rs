// ref: internal/credentialweight/weight_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use serde_json::{json, Number, Value};

use super::weight::{
    normalize, parse_string, parse_value, CredentialWeightError, DEFAULT_CREDENTIAL_WEIGHT,
    MAX_CREDENTIAL_WEIGHT,
};

#[test]
fn parse_string_matches_upstream_defaults_and_integer_bounds() {
    assert_eq!(parse_string(""), Ok(DEFAULT_CREDENTIAL_WEIGHT));
    assert_eq!(parse_string("  -5 "), Ok(0));
    assert_eq!(
        parse_string(&MAX_CREDENTIAL_WEIGHT.to_string()),
        Ok(MAX_CREDENTIAL_WEIGHT)
    );
    assert_eq!(
        parse_string(&(MAX_CREDENTIAL_WEIGHT + 1).to_string()),
        Err(CredentialWeightError::AboveMaximum)
    );
    assert_eq!(parse_string("1.5"), Err(CredentialWeightError::NotInteger));
    assert_eq!(
        parse_string("9223372036854775808"),
        Err(CredentialWeightError::NotInteger)
    );
}

#[test]
fn parse_value_matches_upstream_number_and_string_semantics() {
    assert_eq!(parse_value(&json!(-5)), Ok(0));
    assert_eq!(parse_value(&json!(MAX_CREDENTIAL_WEIGHT)), Ok(1_000_000));
    assert_eq!(
        parse_value(&json!(MAX_CREDENTIAL_WEIGHT + 1)),
        Err(CredentialWeightError::AboveMaximum)
    );
    assert_eq!(
        parse_value(&json!(1.5)),
        Err(CredentialWeightError::NotInteger)
    );
    assert_eq!(parse_value(&json!(" 2 ")), Ok(2));
    assert_eq!(
        parse_value(&Value::Bool(true)),
        Err(CredentialWeightError::NotInteger)
    );

    let unsigned = Value::Number(Number::from(u64::MAX));
    assert_eq!(
        parse_value(&unsigned),
        Err(CredentialWeightError::AboveMaximum)
    );
}

#[test]
fn normalize_clamps_nonpositive_values_and_rejects_above_maximum() {
    assert_eq!(normalize(i64::MIN), Ok(0));
    assert_eq!(normalize(0), Ok(0));
    assert_eq!(normalize(1), Ok(1));
    assert_eq!(normalize(MAX_CREDENTIAL_WEIGHT), Ok(MAX_CREDENTIAL_WEIGHT));
    assert_eq!(
        normalize(MAX_CREDENTIAL_WEIGHT + 1),
        Err(CredentialWeightError::AboveMaximum)
    );
}
