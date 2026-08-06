// ref: internal/runtime/executor/helps/cloak_utils.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use uuid::Uuid;

#[derive(Debug, Deserialize, Serialize)]
struct ClaudeMetadataUserId {
    device_id: String,
    account_uuid: String,
    session_id: String,
}

pub fn generate_fake_user_id() -> String {
    generate_fake_user_id_with_session_id(&Uuid::new_v4().to_string())
}

pub fn generate_fake_user_id_with_session_id(session_id: &str) -> String {
    let session_id = Uuid::parse_str(session_id)
        .unwrap_or_else(|_| Uuid::new_v4())
        .to_string();
    let mut random = [0_u8; 32];
    if getrandom::fill(&mut random).is_err() {
        random.copy_from_slice(Sha256::digest(Uuid::new_v4().as_bytes()).as_ref());
    }
    let device_id = random
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    serde_json::to_string(&ClaudeMetadataUserId {
        device_id,
        account_uuid: String::new(),
        session_id,
    })
    .expect("Claude metadata identity has an infallible JSON representation")
}

pub fn is_valid_user_id(value: &str) -> bool {
    let Ok(identity) = serde_json::from_str::<ClaudeMetadataUserId>(value) else {
        return false;
    };
    identity.device_id.len() == 64
        && identity
            .device_id
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        && valid_lower_uuid(&identity.session_id)
        && (identity.account_uuid.is_empty() || valid_lower_uuid(&identity.account_uuid))
}

/// Accepted-pin compatibility until the candidate executor-cloaking slice
/// replaces User-Agent-only policy with `ClaudeCodeRequestDetection`.
pub fn should_cloak(mode: &str, user_agent: &str) -> bool {
    match mode.to_ascii_lowercase().as_str() {
        "always" => true,
        "never" => false,
        _ => !is_claude_code_client(user_agent),
    }
}

/// Accepted-pin compatibility predicate; candidate request paths must use the
/// strong-signal detector instead of treating this as confirmation.
pub fn is_claude_code_client(user_agent: &str) -> bool {
    user_agent.starts_with("claude-cli")
}

fn valid_lower_uuid(value: &str) -> bool {
    value.len() == 36
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte) || byte == b'-')
        && Uuid::parse_str(value).is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_user_ids_match_claude_2_1_220_shape_and_are_unique() {
        let first = generate_fake_user_id();
        let second = generate_fake_user_id();
        assert!(is_valid_user_id(&first));
        assert!(is_valid_user_id(&second));
        assert_ne!(first, second);
        assert!(!is_valid_user_id("user_invalid"));
        let parsed: serde_json::Value = serde_json::from_str(&first).unwrap();
        assert_eq!(parsed["account_uuid"], "");
        assert_eq!(parsed["device_id"].as_str().unwrap().len(), 64);
    }

    #[test]
    fn supplied_session_is_preserved_or_repaired() {
        let session = "11111111-2222-4333-8444-555555555555";
        let parsed: serde_json::Value =
            serde_json::from_str(&generate_fake_user_id_with_session_id(session)).unwrap();
        assert_eq!(parsed["session_id"], session);
        let repaired: serde_json::Value =
            serde_json::from_str(&generate_fake_user_id_with_session_id("bad")).unwrap();
        assert!(Uuid::parse_str(repaired["session_id"].as_str().unwrap()).is_ok());
    }

    #[test]
    fn accepted_cloak_compatibility_remains_until_executor_cutover() {
        assert!(should_cloak("always", "claude-cli/2"));
        assert!(!should_cloak("never", "third-party"));
        assert!(!should_cloak("auto", "claude-cli/2"));
        assert!(should_cloak("", "third-party"));
    }
}
