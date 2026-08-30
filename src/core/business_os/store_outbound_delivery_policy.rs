// Origin: CTOX
// License: Apache-2.0

use super::store::{
    outbound_first_string, outbound_load_record, outbound_put_i64, outbound_put_string,
    outbound_string,
};
use super::store_outbound_commands::{outbound_email_account_key_from, outbound_payload_insert};
use super::store_projections::upsert_business_record;
use crate::mission::channels;
use rusqlite::Connection;
use serde_json::Value;

pub(super) fn outbound_has_matching_approval(
    conn: &Connection,
    message_id: &str,
    revision_id: &str,
) -> anyhow::Result<bool> {
    let mut stmt = conn.prepare(
        "SELECT payload_json
         FROM business_records
         WHERE collection = 'outbound_approvals'
           AND deleted = 0",
    )?;
    let rows = stmt.query_map([], |row| row.get::<_, String>(0))?;
    for row in rows {
        let payload: Value = serde_json::from_str(&row?)?;
        if outbound_string(&payload, &["message_id"]).as_deref() == Some(message_id)
            && outbound_string(&payload, &["revision_id"]).as_deref() == Some(revision_id)
            && outbound_string(&payload, &["decision"]).as_deref() == Some("approved")
        {
            return Ok(true);
        }
    }
    Ok(false)
}

pub(super) fn outbound_recipient_suppressed(
    conn: &Connection,
    recipient_email: &str,
) -> anyhow::Result<bool> {
    Ok(outbound_recipient_suppression_reason(conn, recipient_email)?.is_some())
}

/// Map a reply classification onto a canonical suppression reason, returning
/// `None` for classifications that should not block future sends (e.g.
/// `interested`, `unclear`, `auto_reply`). Hard stop signals — the recipient
/// asking to be removed, marking the message as spam, or bouncing — translate
/// into an active suppression so the send gate refuses any later outbound
/// message to the address.
pub(super) fn outbound_reply_suppression_reason(classification: &str) -> Option<&'static str> {
    match classification.trim().to_ascii_lowercase().as_str() {
        "unsubscribe" | "opt_out" | "opt-out" | "optout" | "remove" => Some("unsubscribe"),
        "complaint" | "spam_complaint" | "spam" | "abuse" => Some("complaint"),
        "bounce" | "hard_bounce" | "undeliverable" => Some("bounce"),
        _ => None,
    }
}

/// When a reply is classified as a hard stop signal, register an active
/// suppression entry for the engagement's recipient so future sends are blocked
/// by [`outbound_recipient_suppression_reason`]. Idempotent: a recipient that is
/// already actively suppressed is left untouched. Returns the suppression id
/// when a new entry is written.
/// Out-of-office is the one reply class that does not stop the sequence: it
/// schedules a wait/retry. The follow-up resumes after `ooo_until` (when the
/// auto-reply names a return date) or after a default hold otherwise. The
/// engagement keeps `status = reply_received` so the UI still surfaces the
/// reply; the scheduler honours the OOO exception and the resumed follow-up is
/// still approval-gated — no send happens without a fresh approval.
/// ref: skillbook reply_handling routing `out_of_office` -> "Follow-up nach OOO-Datum neu planen".
pub(super) fn outbound_apply_out_of_office_wait(engagement: &mut Value, payload: &Value, now: i64) {
    const OOO_DEFAULT_WAIT_MS: i64 = 3 * 24 * 60 * 60 * 1000;
    let resume_at = payload
        .get("ooo_until")
        .and_then(Value::as_i64)
        .filter(|until| *until > now)
        .unwrap_or(now + OOO_DEFAULT_WAIT_MS);
    outbound_put_i64(engagement, "next_action_at_ms", resume_at);
    outbound_payload_insert(
        engagement,
        "next_action_at_ms",
        Value::Number(serde_json::Number::from(resume_at)),
    );
    outbound_payload_insert(
        engagement,
        "reply_wait_reason",
        Value::String("out_of_office".to_string()),
    );
    outbound_payload_insert(
        engagement,
        "ooo_until",
        Value::Number(serde_json::Number::from(resume_at)),
    );
}

pub(super) fn outbound_apply_reply_suppression(
    conn: &Connection,
    engagement: &Value,
    engagement_id: &str,
    classification: &str,
    now: i64,
) -> anyhow::Result<Option<String>> {
    let Some(reason) = outbound_reply_suppression_reason(classification) else {
        return Ok(None);
    };
    let recipient = outbound_first_string(&[
        outbound_string(engagement, &["recipient_email"]),
        outbound_string(engagement, &["payload", "recipient_email"]),
        outbound_string(engagement, &["payload", "contact_email"]),
    ])
    .unwrap_or_default()
    .trim()
    .to_ascii_lowercase();
    if recipient.is_empty() || !recipient.contains('@') {
        return Ok(None);
    }
    // Idempotent: do not stack duplicate entries for an already-suppressed recipient.
    if outbound_recipient_suppression_reason(conn, &recipient)?.is_some() {
        return Ok(None);
    }
    let domain = recipient.split('@').nth(1).unwrap_or_default().to_string();
    let suppression_id = format!(
        "supp_reply_{}",
        channels::stable_digest(&format!("{recipient}|{reason}"))
    );
    let record = serde_json::json!({
        "id": suppression_id,
        "email": recipient,
        "domain": domain,
        "status": "active",
        "reason": reason,
        "suppression_reason": reason,
        "source": "reply_classification",
        "engagement_id": engagement_id,
        "created_at_ms": now,
        "updated_at_ms": now,
    });
    upsert_business_record(
        conn,
        "outbound_suppression_entries",
        &suppression_id,
        now,
        record,
    )?;

    // Hard-stop the engagement so the automation scheduler will not reapply the
    // sequence. The reply handler already cancels pending drafts; this records
    // the terminal stop reason on the engagement itself.
    if let Some(mut engagement) = outbound_load_record(conn, "outbound_engagements", engagement_id)?
    {
        outbound_put_string(&mut engagement, "status", "stopped".to_string());
        outbound_payload_insert(
            &mut engagement,
            "stop_reason",
            Value::String(reason.to_string()),
        );
        outbound_payload_insert(
            &mut engagement,
            "stopped_at_ms",
            Value::Number(serde_json::Number::from(now)),
        );
        outbound_put_i64(&mut engagement, "updated_at_ms", now);
        upsert_business_record(conn, "outbound_engagements", engagement_id, now, engagement)?;
    }

    Ok(Some(suppression_id))
}

/// Return the suppression reason (e.g. `bounce`, `opt_out`, `unsubscribe`,
/// `manual`) when the recipient or its domain is on an active suppression
/// entry, or `None` when the recipient is clear to receive. Email match takes
/// precedence over a domain-level block. The reason is surfaced so the send
/// gate can write a precise blocking reason instead of a generic message.
pub(super) fn outbound_recipient_suppression_reason(
    conn: &Connection,
    recipient_email: &str,
) -> anyhow::Result<Option<String>> {
    let recipient = recipient_email.trim().to_ascii_lowercase();
    let domain = recipient.split('@').nth(1).unwrap_or_default().to_string();
    let mut stmt = conn.prepare(
        "SELECT payload_json
         FROM business_records
         WHERE collection = 'outbound_suppression_entries'
           AND deleted = 0",
    )?;
    let rows = stmt.query_map([], |row| row.get::<_, String>(0))?;
    let mut domain_reason: Option<String> = None;
    for row in rows {
        let payload: Value = serde_json::from_str(&row?)?;
        let status = outbound_string(&payload, &["status"]).unwrap_or_else(|| "active".to_string());
        if matches!(status.as_str(), "inactive" | "deleted" | "expired") {
            continue;
        }
        let reason = outbound_first_string(&[
            outbound_string(&payload, &["reason"]),
            outbound_string(&payload, &["suppression_reason"]),
        ])
        .unwrap_or_else(|| "suppressed".to_string());
        let suppressed_email = outbound_string(&payload, &["email"])
            .or_else(|| outbound_string(&payload, &["recipient_email"]))
            .unwrap_or_default()
            .to_ascii_lowercase();
        let suppressed_domain = outbound_string(&payload, &["domain"])
            .unwrap_or_default()
            .to_ascii_lowercase();
        if !suppressed_email.is_empty() && suppressed_email == recipient {
            return Ok(Some(reason));
        }
        if !suppressed_domain.is_empty() && suppressed_domain == domain && domain_reason.is_none() {
            domain_reason = Some(reason);
        }
    }
    Ok(domain_reason)
}

pub(super) fn outbound_enforce_account_limit(
    conn: &Connection,
    sender_account_id: &str,
) -> anyhow::Result<()> {
    // Normalize the lookup key so bare email values like "user@example.com" still resolve
    // to the canonical `email:user@example.com` limit record instead of silently
    // bypassing the gate.
    let canonical = outbound_email_account_key_from(Some(sender_account_id.to_string()))
        .unwrap_or_else(|| sender_account_id.to_string());
    let limit = outbound_load_record(conn, "outbound_account_limits", &canonical)?.or(
        outbound_load_record(conn, "outbound_account_limits", sender_account_id)?,
    );
    let Some(limit) = limit else {
        return Ok(());
    };
    let blocked = limit
        .get("blocked")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    anyhow::ensure!(
        !blocked,
        "sender account is blocked for outbound communication"
    );
    let status = outbound_string(&limit, &["status"]).unwrap_or_else(|| "active".to_string());
    anyhow::ensure!(
        !matches!(
            status.as_str(),
            "blocked" | "locked" | "suspended" | "disabled"
        ),
        "sender account status `{status}` is not eligible to send"
    );
    if let Some(remaining) = limit.get("remaining_today").and_then(Value::as_i64) {
        anyhow::ensure!(remaining > 0, "sender account daily limit exhausted");
    }
    // daily_limit semantics: <= 0 means "no daily cap configured" (sane default for
    // newly linked mailboxes that have not yet been calibrated). Only enforce when an
    // operator has set a positive value.
    if let (Some(sent), Some(limit_value)) = (
        limit.get("daily_sent_count").and_then(Value::as_i64),
        limit.get("daily_limit").and_then(Value::as_i64),
    ) {
        if limit_value > 0 {
            anyhow::ensure!(sent < limit_value, "sender account daily limit exhausted");
        }
    }
    Ok(())
}
