// ref: internal/credentialweight/weight.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use std::fmt;

use serde_json::Value;

pub const DEFAULT_CREDENTIAL_WEIGHT: i64 = 1;
pub const MAX_CREDENTIAL_WEIGHT: i64 = 1_000_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CredentialWeightError {
    NotInteger,
    AboveMaximum,
}

impl fmt::Display for CredentialWeightError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotInteger => formatter.write_str("weight must be an integer"),
            Self::AboveMaximum => {
                write!(formatter, "weight must not exceed {MAX_CREDENTIAL_WEIGHT}")
            }
        }
    }
}

impl std::error::Error for CredentialWeightError {}

pub fn normalize(weight: i64) -> Result<i64, CredentialWeightError> {
    if weight <= 0 {
        return Ok(0);
    }
    if weight > MAX_CREDENTIAL_WEIGHT {
        return Err(CredentialWeightError::AboveMaximum);
    }
    Ok(weight)
}

pub fn parse_string(raw: &str) -> Result<i64, CredentialWeightError> {
    let raw = raw.trim();
    if raw.is_empty() {
        return Ok(DEFAULT_CREDENTIAL_WEIGHT);
    }
    let weight = raw
        .parse::<i64>()
        .map_err(|_| CredentialWeightError::NotInteger)?;
    normalize(weight)
}

pub fn parse_value(value: &Value) -> Result<i64, CredentialWeightError> {
    match value {
        Value::Number(number) => {
            if let Some(weight) = number.as_i64() {
                return normalize(weight);
            }
            if let Some(weight) = number.as_u64() {
                if weight > MAX_CREDENTIAL_WEIGHT as u64 {
                    return Err(CredentialWeightError::AboveMaximum);
                }
                return Ok(weight as i64);
            }
            Err(CredentialWeightError::NotInteger)
        }
        Value::String(raw) => parse_string(raw),
        _ => Err(CredentialWeightError::NotInteger),
    }
}
