// Origin: CTOX
// License: AGPL-3.0-only

use anyhow::Context;
use base64::Engine;
use ring::rand::SecureRandom;
use rusqlite::{params, OptionalExtension};
use serde_json::{json, Value};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

pub const DEFAULT_TTL_SECONDS: i64 = 300;
pub const MIN_TTL_SECONDS: i64 = 60;
pub const MAX_TTL_SECONDS: i64 = 3_600;

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis() as i64)
        .unwrap_or(0)
}

fn ensure_table(conn: &rusqlite::Connection) -> anyhow::Result<()> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS business_mobile_invites (
            invite_id_hash TEXT PRIMARY KEY,
            user_id TEXT NOT NULL UNIQUE,
            created_at_ms INTEGER NOT NULL,
            expires_at_ms INTEGER NOT NULL,
            revoked_at_ms INTEGER,
            device_pairing_id TEXT,
            device_id TEXT,
            proof_key_thumbprint TEXT,
            CHECK (
                (device_pairing_id IS NULL AND device_id IS NULL AND proof_key_thumbprint IS NULL)
                OR
                (device_pairing_id IS NOT NULL AND device_id IS NOT NULL AND proof_key_thumbprint IS NOT NULL)
            )
        );",
    )?;
    for (column, definition) in [
        ("device_pairing_id", "TEXT"),
        ("device_id", "TEXT"),
        ("proof_key_thumbprint", "TEXT"),
    ] {
        if !table_has_column(conn, column)? {
            conn.execute(
                &format!("ALTER TABLE business_mobile_invites ADD COLUMN {column} {definition}"),
                [],
            )?;
        }
    }
    conn.execute_batch(
        "CREATE INDEX IF NOT EXISTS idx_business_mobile_invites_expiry
            ON business_mobile_invites(expires_at_ms, revoked_at_ms);
         CREATE UNIQUE INDEX IF NOT EXISTS idx_business_mobile_invites_device_pairing
            ON business_mobile_invites(device_pairing_id)
            WHERE device_pairing_id IS NOT NULL AND revoked_at_ms IS NULL;",
    )?;
    Ok(())
}

fn table_has_column(conn: &rusqlite::Connection, column: &str) -> anyhow::Result<bool> {
    let mut statement = conn.prepare("PRAGMA table_info(business_mobile_invites)")?;
    let mut rows = statement.query([])?;
    while let Some(row) = rows.next()? {
        if row.get::<_, String>(1)? == column {
            return Ok(true);
        }
    }
    Ok(false)
}

fn random_invite_id() -> anyhow::Result<String> {
    let mut bytes = [0_u8; 32];
    ring::rand::SystemRandom::new()
        .fill(&mut bytes)
        .map_err(|_| anyhow::anyhow!("failed to generate mobile invite id"))?;
    Ok(base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(bytes))
}

fn invite_hash(invite_id: &str) -> String {
    let digest = ring::digest::digest(&ring::digest::SHA256, invite_id.as_bytes());
    digest
        .as_ref()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn validate_display_name(value: Option<&str>) -> anyhow::Result<String> {
    let value = value.unwrap_or("Workjet mobile pairing").trim();
    anyhow::ensure!(!value.is_empty(), "mobile invite display name is required");
    anyhow::ensure!(
        value.chars().count() <= 256,
        "mobile invite display name exceeds 256 characters"
    );
    anyhow::ensure!(
        !value.chars().any(char::is_control),
        "mobile invite display name contains a control character"
    );
    Ok(value.to_string())
}

fn validate_binding_id(value: &str, label: &str) -> anyhow::Result<String> {
    let value = value.trim();
    anyhow::ensure!(!value.is_empty(), "mobile invite {label} is required");
    anyhow::ensure!(
        value.chars().count() <= 256,
        "mobile invite {label} exceeds 256 characters"
    );
    anyhow::ensure!(
        !value.chars().any(char::is_control),
        "mobile invite {label} contains a control character"
    );
    Ok(value.to_string())
}

/// Parse the three managed-device flags as an all-or-none tuple.
pub fn device_binding(
    device_pairing_id: Option<&str>,
    device_id: Option<&str>,
    proof_key_thumbprint: Option<&str>,
) -> anyhow::Result<Option<super::capability::CapabilityDeviceBinding>> {
    match (device_pairing_id, device_id, proof_key_thumbprint) {
        (None, None, None) => Ok(None),
        (Some(device_pairing_id), Some(device_id), Some(proof_key_thumbprint)) => {
            let proof_key_thumbprint = proof_key_thumbprint.trim();
            let decoded_thumbprint = base64::engine::general_purpose::URL_SAFE_NO_PAD
                .decode(proof_key_thumbprint)
                .ok();
            anyhow::ensure!(
                proof_key_thumbprint.len() == 43
                    && proof_key_thumbprint
                        .bytes()
                        .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-' || byte == b'_')
                    && decoded_thumbprint.as_deref().is_some_and(|value| value.len() == 32),
                "mobile invite proofKeyThumbprint must be a 43-character base64url SHA-256 thumbprint"
            );
            Ok(Some(super::capability::CapabilityDeviceBinding {
                device_pairing_id: validate_binding_id(device_pairing_id, "devicePairingId")?,
                device_id: validate_binding_id(device_id, "deviceId")?,
                proof_key_thumbprint: proof_key_thumbprint.to_string(),
            }))
        }
        _ => anyhow::bail!(
            "mobile invite device binding requires --device-pairing-id, --device-id, and --proof-key-thumbprint together"
        ),
    }
}

pub fn create(
    root: &Path,
    ttl_seconds: i64,
    display_name: Option<&str>,
    device_binding: Option<&super::capability::CapabilityDeviceBinding>,
) -> anyhow::Result<Value> {
    anyhow::ensure!(
        (MIN_TTL_SECONDS..=MAX_TTL_SECONDS).contains(&ttl_seconds),
        "mobile invite ttlSeconds must be between {MIN_TTL_SECONDS} and {MAX_TTL_SECONDS}"
    );
    let display_name = validate_display_name(display_name)?;
    let created_at_ms = now_ms();
    let expires_at_ms = created_at_ms
        .checked_add(ttl_seconds * 1_000)
        .context("mobile invite expiry overflow")?;
    let expires_at = chrono::DateTime::from_timestamp_millis(expires_at_ms)
        .context("mobile invite expiry is invalid")?
        .to_rfc3339_opts(chrono::SecondsFormat::Millis, true);
    let invite_id = random_invite_id()?;
    let invite_id_hash = invite_hash(&invite_id);
    let user_id = format!("workjet-mobile-invite-{}", &invite_id_hash[..24]);
    let (capability_token, capability_expires_at_ms) =
        super::store::issue_business_os_capability_token_for_managed_user_until_with_binding(
            root,
            &user_id,
            "Workjet Mobile",
            "user",
            created_at_ms,
            expires_at_ms,
            device_binding,
        )?;
    let config = super::store::sync_config(root)?;
    let mut conn = super::store::open_store(root)?;
    ensure_table(&conn)?;
    let persistence_result = (|| -> rusqlite::Result<()> {
        let tx = conn.transaction()?;
        if let Some(binding) = device_binding {
            let prior_user_id: Option<String> = tx
                .query_row(
                    "SELECT user_id FROM business_mobile_invites
                     WHERE device_pairing_id=?1 AND revoked_at_ms IS NULL",
                    params![binding.device_pairing_id.as_str()],
                    |row| row.get(0),
                )
                .optional()?;
            if let Some(prior_user_id) = prior_user_id {
                tx.execute(
                    "UPDATE business_mobile_invites
                     SET revoked_at_ms=?2
                     WHERE device_pairing_id=?1 AND revoked_at_ms IS NULL",
                    params![binding.device_pairing_id.as_str(), created_at_ms],
                )?;
                tx.execute(
                    "UPDATE business_users
                     SET active=0, capability_epoch=capability_epoch+1, updated_at_ms=?2
                     WHERE user_id=?1 AND active=1",
                    params![prior_user_id, created_at_ms],
                )?;
            }
        }
        tx.execute(
            "INSERT INTO business_mobile_invites
                (invite_id_hash, user_id, created_at_ms, expires_at_ms, revoked_at_ms,
                 device_pairing_id, device_id, proof_key_thumbprint)
             VALUES (?1, ?2, ?3, ?4, NULL, ?5, ?6, ?7)",
            params![
                invite_id_hash,
                user_id,
                created_at_ms,
                expires_at_ms,
                device_binding.map(|binding| binding.device_pairing_id.as_str()),
                device_binding.map(|binding| binding.device_id.as_str()),
                device_binding.map(|binding| binding.proof_key_thumbprint.as_str()),
            ],
        )?;
        tx.commit()
    })();
    if let Err(error) = persistence_result {
        let _ = conn.execute(
            "UPDATE business_users
             SET active=0, capability_epoch=capability_epoch+1, updated_at_ms=?2
             WHERE user_id=?1",
            params![user_id, created_at_ms],
        );
        return Err(error).context("failed to persist mobile invite");
    }

    let invite = json!({
        "type": "ctox-business-os-invite",
        "version": 1,
        "display_name": display_name,
        "instance_id": config.instance_id,
        "sync_room": config.sync_room,
        "native_peer_id": config.native_peer_id,
        "signaling_urls": config.signaling_urls,
        "signaling_room_password": config.signaling_room_password,
        "transport": "webrtc",
        "expires_at": expires_at,
        "data_plane": "rxdb-webrtc",
        "http_bridge_available": false,
        "secret_value_in_payload": true,
        "session": {
            "authenticated": true,
            "source": "mobile_invite",
            "capability_token": capability_token,
            "capability_expires_at_ms": capability_expires_at_ms,
            "user": {
                "id": user_id,
                "display_name": "Workjet Mobile",
                "role": "user",
                "is_admin": false
            }
        }
    });
    let grant_id = device_binding
        .map(|binding| binding.device_pairing_id.as_str())
        .unwrap_or(invite_id.as_str());
    Ok(json!({
        "businessOsInstanceId": config.instance_id,
        "deviceId": device_binding.map(|binding| binding.device_id.as_str()),
        "proofKeyThumbprint": device_binding.map(|binding| binding.proof_key_thumbprint.as_str()),
        "grantId": grant_id,
        "inviteId": invite_id,
        "invite": invite,
        "expiresAt": expires_at
    }))
}

pub fn revoke(root: &Path, invite_id: &str) -> anyhow::Result<Value> {
    let invite_id = invite_id.trim();
    anyhow::ensure!(
        !invite_id.is_empty() && invite_id.len() <= 256,
        "mobile invite id is invalid"
    );
    let invite_id_hash = invite_hash(invite_id);
    let now = now_ms();
    let mut conn = super::store::open_store(root)?;
    ensure_table(&conn)?;
    let tx = conn.transaction()?;
    let user_id: Option<String> = tx
        .query_row(
            "SELECT user_id FROM business_mobile_invites WHERE invite_id_hash=?1",
            params![invite_id_hash],
            |row| row.get(0),
        )
        .optional()?;
    if let Some(user_id) = user_id {
        tx.execute(
            "UPDATE business_mobile_invites
             SET revoked_at_ms=COALESCE(revoked_at_ms, ?2)
             WHERE invite_id_hash=?1",
            params![invite_id_hash, now],
        )?;
        tx.execute(
            "UPDATE business_users
             SET active=0, capability_epoch=capability_epoch+1, updated_at_ms=?2
             WHERE user_id=?1 AND active=1",
            params![user_id, now],
        )?;
    }
    tx.commit()?;
    Ok(json!({ "revoked": true }))
}

pub fn revoke_by_device_pairing_id(root: &Path, device_pairing_id: &str) -> anyhow::Result<Value> {
    let device_pairing_id = validate_binding_id(device_pairing_id, "devicePairingId")?;
    let now = now_ms();
    let mut conn = super::store::open_store(root)?;
    ensure_table(&conn)?;
    let tx = conn.transaction()?;
    let row: Option<(String, String)> = tx
        .query_row(
            "SELECT invite_id_hash, user_id
             FROM business_mobile_invites
             WHERE device_pairing_id=?1 AND revoked_at_ms IS NULL",
            params![device_pairing_id],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .optional()?;
    if let Some((invite_id_hash, user_id)) = row {
        tx.execute(
            "UPDATE business_mobile_invites
             SET revoked_at_ms=COALESCE(revoked_at_ms, ?2)
             WHERE invite_id_hash=?1",
            params![invite_id_hash, now],
        )?;
        tx.execute(
            "UPDATE business_users
             SET active=0, capability_epoch=capability_epoch+1, updated_at_ms=?2
             WHERE user_id=?1 AND active=1",
            params![user_id, now],
        )?;
    }
    tx.commit()?;
    Ok(json!({ "revoked": true }))
}

pub(super) fn is_active_device_binding(
    root: &Path,
    user_id: &str,
    binding: &super::capability::CapabilityDeviceBinding,
    at_ms: i64,
) -> bool {
    let Ok(conn) = super::store::open_store(root) else {
        return false;
    };
    if ensure_table(&conn).is_err() {
        return false;
    }
    conn.query_row(
        "SELECT 1
         FROM business_mobile_invites
         WHERE user_id=?1
           AND device_pairing_id=?2
           AND device_id=?3
           AND proof_key_thumbprint=?4
           AND revoked_at_ms IS NULL
           AND expires_at_ms>?5",
        params![
            user_id,
            binding.device_pairing_id,
            binding.device_id,
            binding.proof_key_thumbprint,
            at_ms,
        ],
        |_| Ok(()),
    )
    .optional()
    .ok()
    .flatten()
    .is_some()
}

#[cfg(test)]
mod tests {
    use super::{create, device_binding, revoke, revoke_by_device_pairing_id};
    use tempfile::tempdir;

    #[test]
    fn mobile_invite_is_short_lived_and_individually_revocable() -> anyhow::Result<()> {
        let root = tempdir()?;
        let created = create(root.path(), 300, Some("Operations"), None)?;
        let invite = &created["invite"];
        let token = invite["session"]["capability_token"]
            .as_str()
            .expect("capability token");
        assert_eq!(invite["data_plane"], "rxdb-webrtc");
        assert_eq!(invite["http_bridge_available"], false);
        assert_eq!(invite["session"]["user"]["role"], "user");
        assert!(super::super::store::verify_capability_role(root.path(), token).is_some());

        let invite_id = created["inviteId"].as_str().expect("invite id");
        assert_eq!(
            revoke(root.path(), invite_id)?,
            serde_json::json!({ "revoked": true })
        );
        assert_eq!(
            revoke(root.path(), invite_id)?,
            serde_json::json!({ "revoked": true })
        );
        assert!(super::super::store::verify_capability_role(root.path(), token).is_none());
        assert_eq!(
            revoke(root.path(), "unknown-invite")?,
            serde_json::json!({ "revoked": true })
        );
        Ok(())
    }

    #[test]
    fn mobile_invite_ttl_is_bounded() {
        let root = tempdir().unwrap();
        assert!(create(root.path(), 59, None, None).is_err());
        assert!(create(root.path(), 3_601, None, None).is_err());
    }

    #[test]
    fn managed_device_binding_is_all_or_none_and_revocable_by_pairing_id() -> anyhow::Result<()> {
        let root = tempdir()?;
        let thumbprint = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA";
        assert!(device_binding(Some("pairing-1"), None, Some(thumbprint)).is_err());
        let binding = device_binding(Some("pairing-1"), Some("device-1"), Some(thumbprint))?
            .expect("binding");
        let created = create(root.path(), 300, None, Some(&binding))?;
        assert_eq!(
            created["businessOsInstanceId"],
            created["invite"]["instance_id"]
        );
        assert_eq!(created["grantId"], "pairing-1");
        assert_eq!(created["deviceId"], "device-1");
        assert_eq!(created["proofKeyThumbprint"], thumbprint);
        let token = created["invite"]["session"]["capability_token"]
            .as_str()
            .expect("token");
        assert!(super::super::store::verify_capability_role(root.path(), token).is_some());
        let rotated = create(root.path(), 300, None, Some(&binding))?;
        let rotated_token = rotated["invite"]["session"]["capability_token"]
            .as_str()
            .expect("rotated token");
        assert!(super::super::store::verify_capability_role(root.path(), token).is_none());
        assert!(super::super::store::verify_capability_role(root.path(), rotated_token).is_some());
        assert_eq!(
            revoke_by_device_pairing_id(root.path(), "pairing-1")?,
            serde_json::json!({ "revoked": true })
        );
        assert!(super::super::store::verify_capability_role(root.path(), rotated_token).is_none());
        Ok(())
    }

    #[test]
    fn legacy_invite_table_migrates_nullable_binding_columns_and_index() -> anyhow::Result<()> {
        let root = tempdir()?;
        let conn = super::super::store::open_store(root.path())?;
        conn.execute_batch(
            "CREATE TABLE business_mobile_invites (
                invite_id_hash TEXT PRIMARY KEY,
                user_id TEXT NOT NULL UNIQUE,
                created_at_ms INTEGER NOT NULL,
                expires_at_ms INTEGER NOT NULL,
                revoked_at_ms INTEGER
            );",
        )?;
        drop(conn);
        let thumbprint = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA";
        let binding = device_binding(
            Some("pairing-migration"),
            Some("device-migration"),
            Some(thumbprint),
        )?
        .expect("binding");
        create(root.path(), 300, None, Some(&binding))?;

        let conn = super::super::store::open_store(root.path())?;
        for column in ["device_pairing_id", "device_id", "proof_key_thumbprint"] {
            assert!(super::table_has_column(&conn, column)?, "missing {column}");
        }
        let pairing_index_count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM sqlite_master
             WHERE type='index' AND name='idx_business_mobile_invites_device_pairing'",
            [],
            |row| row.get(0),
        )?;
        assert_eq!(pairing_index_count, 1);
        let stored: (String, String, String) = conn.query_row(
            "SELECT device_pairing_id, device_id, proof_key_thumbprint
             FROM business_mobile_invites
             WHERE device_pairing_id='pairing-migration'",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )?;
        assert_eq!(
            stored,
            (
                "pairing-migration".to_string(),
                "device-migration".to_string(),
                thumbprint.to_string(),
            )
        );
        Ok(())
    }
}
