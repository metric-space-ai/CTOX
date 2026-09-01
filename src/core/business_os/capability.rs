// Origin: CTOX
// License: AGPL-3.0-only
//
// Capability tokens for Business OS command authorization.
//
// Today native command authorization derives the actor from
// `client_context.actor` inside the replicated command document — a value
// asserted by the browser (see the SECURITY note on
// `store::rxdb_session_from_command`). A capability token closes that hole: the
// native side issues a short-lived, HMAC-signed token binding a user id to a
// role; the browser carries it on each command; the native verifies the
// signature and reads the role FROM THE TOKEN, never from the unsigned claim.
//
// This module is the pure cryptographic core (issue + verify). It takes the
// signing secret as a parameter and does no I/O, so it is unit-testable in
// isolation. Secret provisioning, the runtime enforcement flag, and the wiring
// into the command session live in `store.rs`.

use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine as _;
use ring::hmac;
use serde_json::Value;

/// Optional proof-of-possession binding carried by managed mobile grants.
/// The `cnf.jkt` value is the RFC 7638 thumbprint of the device's P-256 public
/// key. The private key never enters CTOX or the browser runtime.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapabilityDeviceBinding {
    pub device_pairing_id: String,
    pub device_id: String,
    pub proof_key_thumbprint: String,
}

/// The verified contents of a capability token.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapabilityClaims {
    pub user_id: String,
    pub email: Option<String>,
    pub role: String,
    pub actor_epoch: i64,
    pub issued_at_ms: i64,
    pub expires_at_ms: i64,
    pub device_binding: Option<CapabilityDeviceBinding>,
}

/// Issue an HMAC-SHA256 capability token of the form
/// `base64url(payload).base64url(sig)` binding `user_id` + `role` to a validity
/// window. Only a holder of `secret` (the native instance) can mint one.
pub fn issue_capability_token(
    secret: &[u8],
    user_id: &str,
    role: &str,
    issued_at_ms: i64,
    expires_at_ms: i64,
) -> String {
    issue_capability_token_with_epoch(secret, user_id, role, 0, issued_at_ms, expires_at_ms)
}

/// Issue a capability token bound to the actor's current revocation epoch.
/// Changing a role, disabling a user, or changing one of their grants bumps
/// this epoch in the native store and invalidates already issued tokens.
pub fn issue_capability_token_with_epoch(
    secret: &[u8],
    user_id: &str,
    role: &str,
    actor_epoch: i64,
    issued_at_ms: i64,
    expires_at_ms: i64,
) -> String {
    issue_capability_token_with_epoch_and_binding(
        secret,
        user_id,
        role,
        actor_epoch,
        issued_at_ms,
        expires_at_ms,
        None,
    )
}

/// Issue a capability token with an optional all-or-none managed-device
/// binding. Ordinary Business OS sessions continue to use unbound tokens.
pub fn issue_capability_token_with_epoch_and_binding(
    secret: &[u8],
    user_id: &str,
    role: &str,
    actor_epoch: i64,
    issued_at_ms: i64,
    expires_at_ms: i64,
    device_binding: Option<&CapabilityDeviceBinding>,
) -> String {
    issue_capability_token_with_epoch_and_identity(
        secret,
        user_id,
        None,
        role,
        actor_epoch,
        issued_at_ms,
        expires_at_ms,
        device_binding,
    )
}

/// Issue a capability token with an optional signed identity alias. Managed
/// control planes use this to bind the authenticated email to the stable user
/// id so native migrations never have to trust browser-asserted aliases.
pub fn issue_capability_token_with_epoch_and_identity(
    secret: &[u8],
    user_id: &str,
    email: Option<&str>,
    role: &str,
    actor_epoch: i64,
    issued_at_ms: i64,
    expires_at_ms: i64,
    device_binding: Option<&CapabilityDeviceBinding>,
) -> String {
    let mut payload = serde_json::json!({
        "uid": user_id,
        "role": role,
        "epoch": actor_epoch,
        "iat": issued_at_ms,
        "exp": expires_at_ms,
    });
    if let Some(email) = email.map(str::trim).filter(|email| !email.is_empty()) {
        payload["email"] = Value::String(email.to_ascii_lowercase());
    }
    if let Some(binding) = device_binding {
        payload["device_pairing_id"] = Value::String(binding.device_pairing_id.clone());
        payload["device_id"] = Value::String(binding.device_id.clone());
        payload["cnf"] = serde_json::json!({
            "jkt": binding.proof_key_thumbprint,
        });
    }
    let payload_b64 = URL_SAFE_NO_PAD.encode(serde_json::to_vec(&payload).unwrap_or_default());
    let key = hmac::Key::new(hmac::HMAC_SHA256, secret);
    let sig = hmac::sign(&key, payload_b64.as_bytes());
    let sig_b64 = URL_SAFE_NO_PAD.encode(sig.as_ref());
    format!("{payload_b64}.{sig_b64}")
}

/// Verify a capability token against `secret` at `now_ms`. Returns the claims
/// only when the signature is valid (constant-time) and the token has not
/// expired. Any malformed / tampered / expired token returns `None`.
pub fn verify_capability_token(
    secret: &[u8],
    token: &str,
    now_ms: i64,
) -> Option<CapabilityClaims> {
    let claims = verify_capability_token_allow_expired(secret, token)?;
    (now_ms < claims.expires_at_ms).then_some(claims)
}

/// Verify the signature and parse the complete capability without applying its
/// interactive-session expiry. This is intentionally crate-local: only the
/// native WebRTC peer may use it, and only after it has independently checked
/// the durable device binding plus the nonce-bound P-256 proof of possession.
pub(super) fn verify_capability_token_allow_expired(
    secret: &[u8],
    token: &str,
) -> Option<CapabilityClaims> {
    let (payload_b64, sig_b64) = token.split_once('.')?;
    let sig = URL_SAFE_NO_PAD.decode(sig_b64).ok()?;
    let key = hmac::Key::new(hmac::HMAC_SHA256, secret);
    // Constant-time verification (ring); a forged signature fails here.
    hmac::verify(&key, payload_b64.as_bytes(), &sig).ok()?;
    let payload: Value = serde_json::from_slice(&URL_SAFE_NO_PAD.decode(payload_b64).ok()?).ok()?;
    let expires_at_ms = payload.get("exp").and_then(Value::as_i64)?;
    let device_pairing_id = payload.get("device_pairing_id").and_then(Value::as_str);
    let device_id = payload.get("device_id").and_then(Value::as_str);
    let proof_key_thumbprint = payload.pointer("/cnf/jkt").and_then(Value::as_str);
    let device_binding = match (device_pairing_id, device_id, proof_key_thumbprint) {
        (None, None, None) => None,
        (Some(device_pairing_id), Some(device_id), Some(proof_key_thumbprint))
            if !device_pairing_id.is_empty()
                && !device_id.is_empty()
                && !proof_key_thumbprint.is_empty() =>
        {
            Some(CapabilityDeviceBinding {
                device_pairing_id: device_pairing_id.to_string(),
                device_id: device_id.to_string(),
                proof_key_thumbprint: proof_key_thumbprint.to_string(),
            })
        }
        // A signed-but-partial binding is malformed, never an unbound token.
        _ => return None,
    };
    Some(CapabilityClaims {
        user_id: payload.get("uid").and_then(Value::as_str)?.to_string(),
        email: payload
            .get("email")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|email| !email.is_empty())
            .map(str::to_ascii_lowercase),
        role: payload.get("role").and_then(Value::as_str)?.to_string(),
        actor_epoch: payload.get("epoch").and_then(Value::as_i64).unwrap_or(0),
        issued_at_ms: payload.get("iat").and_then(Value::as_i64).unwrap_or(0),
        expires_at_ms,
        device_binding,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const SECRET: &[u8] = b"instance-capability-secret-0123456789";
    const NOW: i64 = 1_750_000_000_000;
    const HOUR: i64 = 60 * 60 * 1000;

    #[test]
    fn issue_then_verify_round_trips() {
        let token = issue_capability_token_with_epoch(SECRET, "chef1", "chef", 7, NOW, NOW + HOUR);
        let claims = verify_capability_token(SECRET, &token, NOW + 1000).expect("valid");
        assert_eq!(claims.user_id, "chef1");
        assert_eq!(claims.email, None);
        assert_eq!(claims.role, "chef");
        assert_eq!(claims.actor_epoch, 7);
        assert_eq!(claims.expires_at_ms, NOW + HOUR);
        assert_eq!(claims.device_binding, None);
    }

    #[test]
    fn signed_email_alias_round_trips() {
        let token = issue_capability_token_with_epoch_and_identity(
            SECRET,
            "user-uuid",
            Some("Michael.Welsch@Metric-Space.AI"),
            "user",
            4,
            NOW,
            NOW + HOUR,
            None,
        );
        let claims = verify_capability_token(SECRET, &token, NOW).expect("valid");
        assert_eq!(claims.user_id, "user-uuid");
        assert_eq!(
            claims.email.as_deref(),
            Some("michael.welsch@metric-space.ai")
        );
    }

    #[test]
    fn device_binding_round_trips_as_tuple_and_cnf_jkt() {
        let binding = CapabilityDeviceBinding {
            device_pairing_id: "pairing-1".to_string(),
            device_id: "device-1".to_string(),
            proof_key_thumbprint: "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA".to_string(),
        };
        let token = issue_capability_token_with_epoch_and_binding(
            SECRET,
            "mobile-1",
            "user",
            3,
            NOW,
            NOW + HOUR,
            Some(&binding),
        );
        let (payload, _) = token.split_once('.').expect("token");
        let payload: Value =
            serde_json::from_slice(&URL_SAFE_NO_PAD.decode(payload).unwrap()).unwrap();
        assert_eq!(payload["device_pairing_id"], "pairing-1");
        assert_eq!(payload["device_id"], "device-1");
        assert_eq!(payload["cnf"]["jkt"], binding.proof_key_thumbprint);
        assert_eq!(
            verify_capability_token(SECRET, &token, NOW)
                .unwrap()
                .device_binding,
            Some(binding)
        );
    }

    #[test]
    fn wrong_secret_is_rejected() {
        let token = issue_capability_token(SECRET, "chef1", "chef", NOW, NOW + HOUR);
        assert!(verify_capability_token(b"other-secret", &token, NOW).is_none());
    }

    #[test]
    fn signed_partial_device_binding_is_rejected_not_downgraded() {
        let payload = serde_json::json!({
            "uid": "mobile-1",
            "role": "user",
            "epoch": 1,
            "iat": NOW,
            "exp": NOW + HOUR,
            "device_id": "device-only",
        });
        let payload_b64 = URL_SAFE_NO_PAD.encode(serde_json::to_vec(&payload).unwrap());
        let key = hmac::Key::new(hmac::HMAC_SHA256, SECRET);
        let signature = URL_SAFE_NO_PAD.encode(hmac::sign(&key, payload_b64.as_bytes()).as_ref());
        assert!(
            verify_capability_token(SECRET, &format!("{payload_b64}.{signature}"), NOW).is_none()
        );
    }

    #[test]
    fn tampered_payload_is_rejected() {
        // Forge a chef role over a token minted for a plain user — the signature
        // no longer matches the swapped payload.
        let token = issue_capability_token(SECRET, "u1", "user", NOW, NOW + HOUR);
        let sig = token.split_once('.').unwrap().1;
        let forged_payload = URL_SAFE_NO_PAD.encode(
            serde_json::to_vec(&serde_json::json!({
                "uid": "u1", "role": "chef", "iat": NOW, "exp": NOW + HOUR
            }))
            .unwrap(),
        );
        let forged = format!("{forged_payload}.{sig}");
        assert!(verify_capability_token(SECRET, &forged, NOW).is_none());
    }

    #[test]
    fn expired_token_is_rejected() {
        let token = issue_capability_token(SECRET, "chef1", "chef", NOW - 2 * HOUR, NOW - HOUR);
        assert!(verify_capability_token(SECRET, &token, NOW).is_none());
        let signed = verify_capability_token_allow_expired(SECRET, &token)
            .expect("native WebRTC verifier may inspect a correctly signed expired assertion");
        assert_eq!(signed.user_id, "chef1");
        assert_eq!(signed.expires_at_ms, NOW - HOUR);
    }

    #[test]
    fn garbage_token_is_rejected() {
        assert!(verify_capability_token(SECRET, "not-a-token", NOW).is_none());
        assert!(verify_capability_token(SECRET, "a.b.c", NOW).is_none());
        assert!(verify_capability_token(SECRET, "", NOW).is_none());
    }
}
