// ref: internal/config/config_validation.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case", deny_unknown_fields)]
pub struct PayloadRule {
    #[serde(default)]
    pub params: BTreeMap<String, Value>,
}

#[must_use]
pub fn sanitize_payload_raw_rules(rules: Vec<PayloadRule>) -> Vec<PayloadRule> {
    rules
        .into_iter()
        .filter(|rule| {
            !rule.params.is_empty()
                && rule.params.values().all(|value| match value {
                    Value::String(raw) => {
                        let raw = raw.trim();
                        !raw.is_empty() && serde_json::from_str::<Value>(raw).is_ok()
                    }
                    _ => true,
                })
        })
        .collect()
}

#[must_use]
pub fn looks_like_bcrypt(value: &str) -> bool {
    value.starts_with("$2a$") || value.starts_with("$2b$") || value.starts_with("$2y$")
}

pub fn reject_plaintext_management_secret(value: &str) -> Result<(), &'static str> {
    if value.is_empty() || looks_like_bcrypt(value) {
        Ok(())
    } else {
        Err("plaintext management secrets must be imported through the CTOX secret store")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_rules_and_secret_policy_are_fail_closed() {
        let rules = vec![
            PayloadRule {
                params: BTreeMap::from([("ok".into(), Value::String(r#"{"a":1}"#.into()))]),
            },
            PayloadRule {
                params: BTreeMap::from([("bad".into(), Value::String("{".into()))]),
            },
        ];
        assert_eq!(sanitize_payload_raw_rules(rules).len(), 1);
        assert!(reject_plaintext_management_secret("plaintext").is_err());
        assert!(reject_plaintext_management_secret("$2b$hash").is_ok());
    }
}
