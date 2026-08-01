// Origin: CTOX
// License: Apache-2.0

use super::store::{
    checked_scaled, first_string_field, hex_sha256, insert_business_event,
    materialize_business_chat_attachment, now_ms, open_store, rxdb_desktop_file_document,
    BusinessCommand, BusinessOsSession,
};
use super::store_projections::upsert_business_record;
use anyhow::Context;
use rusqlite::{params, Connection, OptionalExtension};
use serde_json::Value;
use std::path::Path;

/// Load every non-deleted record of `collection` whose JSON `field` equals
/// `value`, from the generic business_records store.
fn load_business_records_by_field(
    conn: &Connection,
    collection: &str,
    field: &str,
    value: &str,
) -> anyhow::Result<Vec<Value>> {
    let mut stmt = conn.prepare(
        "SELECT payload_json FROM business_records
         WHERE collection = ?1 AND deleted = 0
           AND json_extract(payload_json, '$.' || ?2) = ?3",
    )?;
    let rows = stmt.query_map(params![collection, field, value], |row| {
        row.get::<_, String>(0)
    })?;
    let mut out = Vec::new();
    for row in rows {
        let raw = row?;
        // Fail closed: a corrupt credential/consent row must not silently vanish
        // and let a server-authoritative gate pass as if it did not exist.
        let parsed = serde_json::from_str::<Value>(&raw)
            .with_context(|| format!("corrupt {collection} record while loading by {field}"))?;
        out.push(parsed);
    }
    Ok(out)
}

/// Server-authoritative ATS gate checks. Reads the subject's credentials/consents
/// from business_records and returns an allow/deny decision computed by the
/// native ats_gates primitives.
pub(super) fn handle_ats_active_command(
    root: &Path,
    _session: &BusinessOsSession,
    command: &BusinessCommand,
) -> anyhow::Result<Value> {
    let conn = open_store(root)?;
    let now = now_ms() as i64;
    match command.command_type.as_str() {
        "ats.deployment.check" => {
            let subject_id = command
                .payload
                .get("subject_id")
                .and_then(Value::as_str)
                .context("ats.deployment.check requires subject_id")?;
            let required: Vec<String> = command
                .payload
                .get("required_types")
                .and_then(Value::as_array)
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(ToOwned::to_owned))
                        .collect()
                })
                .unwrap_or_default();
            let required_refs: Vec<&str> = required.iter().map(String::as_str).collect();
            let credentials = load_business_records_by_field(
                &conn,
                "business_credentials",
                "subject_id",
                subject_id,
            )?;
            let readiness =
                super::ats_gates::evaluate_deployment_readiness(&credentials, &required_refs, now);
            Ok(serde_json::json!({
                "ok": true,
                "ready": readiness.ready,
                "blockers": readiness
                    .blockers
                    .iter()
                    .map(|(ty, reason)| serde_json::json!({ "credential_type": ty, "reason": reason }))
                    .collect::<Vec<_>>(),
            }))
        }
        "ats.consent.check" => {
            let subject_id = command
                .payload
                .get("subject_id")
                .and_then(Value::as_str)
                .context("ats.consent.check requires subject_id")?;
            let purpose = command.payload.get("purpose").and_then(Value::as_str);
            let consents = load_business_records_by_field(
                &conn,
                "business_consents",
                "subject_id",
                subject_id,
            )?;
            let require_evidence = crate::inference::runtime_env::env_or_config(
                root,
                "CTOX_BUSINESS_OS_REQUIRE_LEGAL_BASIS_EVIDENCE",
            )
            .as_deref()
                == Some("1");
            let allowed =
                super::ats_gates::evaluate_consent_gate(purpose, &consents, now, require_evidence);
            Ok(serde_json::json!({ "ok": true, "allowed": allowed, "purpose": purpose }))
        }
        "ats.retention.due" => {
            // Löschfristen: return the record ids of a collection that are past
            // their retention window, computed native-side. Read-only (the purge
            // itself is a separate admin action).
            let collection = command
                .payload
                .get("collection")
                .and_then(Value::as_str)
                .context("ats.retention.due requires collection")?;
            let retention_days = command
                .payload
                .get("retention_days")
                .and_then(Value::as_i64)
                .unwrap_or(0);
            let reference_field = command
                .payload
                .get("reference_field")
                .and_then(Value::as_str)
                .unwrap_or("created_at_ms");
            let mut stmt = conn.prepare(
                "SELECT record_id, payload_json FROM business_records
                 WHERE collection = ?1 AND deleted = 0",
            )?;
            let rows = stmt.query_map(params![collection], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            })?;
            let mut due_ids = Vec::new();
            for row in rows {
                let (id, payload) = row?;
                let reference = serde_json::from_str::<Value>(&payload)
                    .ok()
                    .and_then(|v| v.get(reference_field).and_then(Value::as_i64))
                    .unwrap_or(0);
                if super::ats_gates::retention_due(reference, retention_days, now) {
                    due_ids.push(id);
                }
            }
            Ok(serde_json::json!({
                "ok": true,
                "count": due_ids.len(),
                "due_ids": due_ids,
            }))
        }
        other => Err(anyhow::anyhow!("unsupported ats command type: {other}")),
    }
}

fn load_all_business_records(
    conn: &Connection,
    collection: &str,
) -> anyhow::Result<Vec<(String, Value)>> {
    let mut stmt = conn.prepare(
        "SELECT record_id, payload_json FROM business_records WHERE collection = ?1 AND deleted = 0",
    )?;
    let rows = stmt.query_map(params![collection], |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
    })?;
    let mut out = Vec::new();
    for row in rows {
        let (id, payload) = row?;
        if let Ok(parsed) = serde_json::from_str::<Value>(&payload) {
            out.push((id, parsed));
        }
    }
    Ok(out)
}

/// Server-authoritative ATS mutations: each writes the generic business_records
/// store (auto-projected to RxDB by rxdb_peer). Gates that protect a write
/// (consent, double-submission) run native-side via ats_gates.
// §5.3 ATS governance: stable rejection reason codes (one source of truth, not
// inline string literals) used by the present/deployment gates and surfaced to
// callers + the audit trail.
const ATS_REASON_CONSENT_MISSING: &str = "consent_to_present_missing";
const ATS_REASON_DOUBLE_SUBMISSION: &str = "double_submission";

/// §5.3: build the audit actor for an ATS governance event from the
/// already-authenticated command session (no extra store round-trip).
fn ats_actor_value(session: &BusinessOsSession) -> Value {
    match &session.user {
        Some(user) => serde_json::json!({
            "id": user.id,
            "display_name": user.display_name,
            "role": user.role,
            "trusted": true
        }),
        None => serde_json::json!({ "id": "rxdb-command", "trusted": false }),
    }
}

/// §5.3: append a DSGVO governance event to `business_events` recording an ATS
/// candidate-data decision — who (actor) shared which candidate with which
/// client, the decision, and any reason codes. This is the accountability trail
/// a Personalvermittler needs for candidate PII handling.
fn record_ats_governance_event(
    conn: &Connection,
    command: &BusinessCommand,
    actor: &Value,
    event_type: &str,
    summary: Value,
    observed_at_ms: i64,
) -> anyhow::Result<()> {
    let record_id = command.record_id.as_deref().unwrap_or("");
    insert_business_event(
        conn,
        "business_commands",
        record_id,
        event_type,
        serde_json::json!({
            "event_type": event_type,
            "command_type": command.command_type.as_str(),
            "module": command.module.as_str(),
            "actor": actor,
            "summary": summary,
            "observed_at_ms": observed_at_ms
        }),
        observed_at_ms,
    )
}

/// One canonical `accounting_invoices` line (postable shape): `quantity` in
/// thousandths and `unit_price_cents` in cents, with the revenue `account_code`
/// the invoice poster requires.
fn ats_invoice_line(position: i64, description: &str, amount_eur: f64, tax_rate: f64) -> Value {
    serde_json::json!({
        "position": position,
        "description": description,
        "quantity": 1000,
        "unit_price_cents": checked_scaled(amount_eur, 100.0),
        "tax_rate": tax_rate,
        "account_code": "8400"
    })
}

/// Build a canonical `accounting_invoices` DRAFT the invoices module can post.
/// ATS billing (placement fee, Leistungsnachweis, early-leave clawback) emits
/// these so the records satisfy the RxDB `accounting_invoices` schema (which
/// requires `state`/`currency`/`search_text`/`is_deleted`) and flow through the
/// invoices app — instead of a non-postable, schema-invalid shadow shape.
fn ats_canonical_invoice(
    id: &str,
    invoice_type: &str,
    party_id: &str,
    lines: Vec<Value>,
    now: i64,
    source_field: &str,
    source_id: &str,
) -> Value {
    let mut net_total: i64 = 0;
    let mut tax_total: i64 = 0;
    for line in &lines {
        let quantity = line.get("quantity").and_then(Value::as_i64).unwrap_or(0);
        let unit_price = line
            .get("unit_price_cents")
            .and_then(Value::as_i64)
            .unwrap_or(0);
        let net = ((unit_price as f64) * (quantity as f64) / 1000.0).round() as i64;
        let tax = ((net as f64) * line.get("tax_rate").and_then(Value::as_f64).unwrap_or(0.0))
            .round() as i64;
        net_total += net;
        tax_total += tax;
    }
    let total = net_total + tax_total;
    let mut invoice = serde_json::json!({
        "id": id,
        "invoice_type": invoice_type,
        "party_id": party_id,
        "currency": "EUR",
        "state": "draft",
        "invoice_date_ms": now,
        "lines": lines,
        "subtotal_cents": net_total,
        "tax_cents": tax_total,
        "total_cents": total,
        "paid_cents": 0,
        "open_cents": total,
        "search_text": party_id.to_lowercase(),
        "is_deleted": false,
        "created_at_ms": now,
        "updated_at_ms": now,
        "_deleted": false
    });
    if let Some(obj) = invoice.as_object_mut() {
        obj.insert(
            source_field.to_string(),
            Value::String(source_id.to_string()),
        );
    }
    invoice
}

/// Allowlist of ATS PII/record collections an ats.* mutation may target. A
/// payload-supplied `collection` outside this set is rejected so a malicious
/// RxDB command cannot redirect a retention purge / sign-off at arbitrary
/// business_records (invoices, credentials, commands, module records, …).
const ATS_WRITABLE_COLLECTIONS: &[&str] = &[
    "applications",
    "business_credentials",
    "business_consents",
    "submissions",
    "offers",
    "placements",
    "interview_scorecards",
    "interview_meetings",
    "signature_requests",
    "planning_time_records",
];

fn ensure_ats_collection_allowed(collection: &str) -> anyhow::Result<()> {
    anyhow::ensure!(
        ATS_WRITABLE_COLLECTIONS.contains(&collection),
        "collection {collection:?} is not an ATS-writable collection"
    );
    Ok(())
}

/// True when a placement type is a temp-staffing / Arbeitnehmerüberlassung
/// (Zeitarbeit) arrangement, which legally requires a deployment-readiness gate.
/// Direct/permanent placement (Personalvermittlung) does not.
fn placement_type_requires_deployment_gate(placement_type: &str) -> bool {
    matches!(
        placement_type.trim().to_ascii_lowercase().as_str(),
        "arbeitnehmerueberlassung"
            | "arbeitnehmerüberlassung"
            | "aue"
            | "anü"
            | "anue"
            | "temp"
            | "temp_staffing"
            | "zeitarbeit"
            | "ueberlassung"
            | "überlassung"
    )
}

/// The mandatory credential types an Arbeitnehmerüberlassung placement must
/// check, read from the runtime config (CTOX_BUSINESS_OS_AUE_REQUIRED_CREDENTIALS,
/// comma/space separated). The legal list stays in config so the recruiting
/// profile owns it (Baukasten) — but §9.2 makes the GATE itself mandatory
/// whenever placement_type is AÜG, never caller-optional.
fn aue_mandatory_required_types(root: &Path) -> Vec<String> {
    crate::inference::runtime_env::env_or_config(root, "CTOX_BUSINESS_OS_AUE_REQUIRED_CREDENTIALS")
        .map(|raw| {
            raw.split([',', ';', '\n', '\t', ' '])
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(ToOwned::to_owned)
                .collect()
        })
        .unwrap_or_default()
}

pub(super) fn handle_ats_mutating_command(
    root: &Path,
    session: &BusinessOsSession,
    command: &BusinessCommand,
) -> anyhow::Result<Value> {
    let conn = open_store(root)?;
    let now = now_ms() as i64;
    let p = &command.payload;
    let actor = ats_actor_value(session);
    match command.command_type.as_str() {
        "ats.retention.purge" => {
            let collection = p
                .get("collection")
                .and_then(Value::as_str)
                .context("ats.retention.purge requires collection")?;
            // Security: a payload-supplied collection must be an ATS PII
            // collection — never invoices/credentials/commands/module records.
            ensure_ats_collection_allowed(collection)?;
            let retention_days = p.get("retention_days").and_then(Value::as_i64).unwrap_or(0);
            let reference_field = p
                .get("reference_field")
                .and_then(Value::as_str)
                .unwrap_or("created_at_ms");
            let mut purged = Vec::new();
            for (id, payload) in load_all_business_records(&conn, collection)? {
                let reference = payload
                    .get(reference_field)
                    .and_then(Value::as_i64)
                    .unwrap_or(0);
                if super::ats_gates::retention_due(reference, retention_days, now) {
                    // DSGVO erasure: overwrite the PII payload with a non-PII
                    // tombstone before marking deleted — a soft-delete alone
                    // leaves candidate PII sitting in payload_json.
                    let tombstone = serde_json::json!({
                        "id": id,
                        "_deleted": true,
                        "redacted": true,
                        "redacted_at_ms": now,
                        "reference_field": reference_field,
                        "retention_days": retention_days,
                        "updated_at_ms": now
                    });
                    conn.execute(
                        "UPDATE business_records SET deleted = 1, updated_at_ms = ?3, payload_json = ?4 WHERE collection = ?1 AND record_id = ?2",
                        params![collection, id.as_str(), now, serde_json::to_string(&tombstone)?],
                    )?;
                    purged.push(id);
                }
            }
            // §5.3 DSGVO trail: record the retention deletion (PII erasure).
            record_ats_governance_event(
                &conn,
                command,
                &actor,
                "business_os.ats.retention_purged",
                serde_json::json!({
                    "collection": collection,
                    "purged_count": purged.len(),
                    "purged_ids": purged,
                    "retention_days": retention_days
                }),
                now,
            )?;
            Ok(serde_json::json!({ "ok": true, "purged": purged.len(), "purged_ids": purged }))
        }
        "ats.intake.capture" => {
            let channel =
                first_string_field(p, &["channel"]).unwrap_or_else(|| "email".to_string());
            let name = first_string_field(p, &["name"]).unwrap_or_default();
            let email = first_string_field(p, &["email"])
                .unwrap_or_default()
                .to_lowercase();
            let id = format!("appl_{now}");
            let dedupe_key = if !email.is_empty() {
                format!("email:{email}")
            } else {
                format!("name:{}", name.to_lowercase())
            };
            let payload = serde_json::json!({
                "id": id,
                "channel": channel,
                "vacancy_id": first_string_field(p, &["vacancy_id"]).unwrap_or_default(),
                "candidate": {
                    "name": name,
                    "email": email,
                    "phone": first_string_field(p, &["phone"]).unwrap_or_default()
                },
                "dedupe_key": dedupe_key,
                "status": "new",
                "received_at_ms": now,
                "created_at_ms": now,
                "updated_at_ms": now,
                "_deleted": false
            });
            upsert_business_record(&conn, "applications", &id, now, payload)?;
            // §5.3 DSGVO trail: record the inbound candidate-PII capture.
            record_ats_governance_event(
                &conn,
                command,
                &actor,
                "business_os.ats.intake_captured",
                serde_json::json!({
                    "application_id": id,
                    "channel": channel,
                    "dedupe_key": dedupe_key,
                    "has_email": !email.is_empty()
                }),
                now,
            )?;
            Ok(serde_json::json!({ "ok": true, "application_id": id, "dedupe_key": dedupe_key }))
        }
        "ats.placement.create" => {
            let candidate_id = first_string_field(p, &["candidate_id"])
                .context("ats.placement.create requires candidate_id")?;
            // §5.8/§9.2 deployment-readiness gate. The required credential set is
            // SERVER-DERIVED: the caller's required_types unioned with the
            // mandatory AÜG set when placement_type is an Arbeitnehmerüberlassung
            // arrangement. The gate is mandatory for AÜG (never caller-optional);
            // direct/permanent placement with no required types skips it.
            let placement_type = first_string_field(p, &["placement_type"]).unwrap_or_default();
            let is_aue = placement_type_requires_deployment_gate(&placement_type);
            let mut required: Vec<String> = p
                .get("required_types")
                .and_then(Value::as_array)
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(ToOwned::to_owned))
                        .collect()
                })
                .unwrap_or_default();
            if is_aue {
                for ty in aue_mandatory_required_types(root) {
                    if !required.iter().any(|r| r == &ty) {
                        required.push(ty);
                    }
                }
                // Fail closed: an AÜG placement must run a credential gate. An
                // empty set here is a misconfiguration, not a licence to skip.
                if required.is_empty() {
                    record_ats_governance_event(
                        &conn,
                        command,
                        &actor,
                        "business_os.ats.placement_denied",
                        serde_json::json!({
                            "candidate_id": candidate_id,
                            "decision": "denied",
                            "reason_codes": ["aue_required_credentials_unconfigured"]
                        }),
                        now,
                    )?;
                    return Ok(serde_json::json!({
                        "ok": true,
                        "allowed": false,
                        "blockers": [{ "reason": "aue_required_credentials_unconfigured" }],
                    }));
                }
            }
            if !required.is_empty() {
                let required_refs: Vec<&str> = required.iter().map(String::as_str).collect();
                let credentials = load_business_records_by_field(
                    &conn,
                    "business_credentials",
                    "subject_id",
                    &candidate_id,
                )?;
                let readiness = super::ats_gates::evaluate_deployment_readiness(
                    &credentials,
                    &required_refs,
                    now,
                );
                if !readiness.ready {
                    let blockers = readiness
                        .blockers
                        .iter()
                        .map(|(ty, reason)| serde_json::json!({ "credential_type": ty, "reason": reason }))
                        .collect::<Vec<_>>();
                    // §5.3 DSGVO trail: record the blocked deployment attempt.
                    record_ats_governance_event(
                        &conn,
                        command,
                        &actor,
                        "business_os.ats.placement_denied",
                        serde_json::json!({
                            "candidate_id": candidate_id,
                            "decision": "denied",
                            "reason_codes": ["deployment_not_ready"],
                            "blockers": blockers
                        }),
                        now,
                    )?;
                    return Ok(serde_json::json!({
                        "ok": true,
                        "allowed": false,
                        "blockers": blockers,
                    }));
                }
            }
            let client_account_id =
                first_string_field(p, &["client_account_id"]).unwrap_or_default();
            let fee = p.get("fee").and_then(Value::as_f64).unwrap_or(0.0);
            let id = format!("plac_{now}");
            // Deterministic fee-invoice id so a replay updates the same invoice
            // and the early-leave clawback can reference it (§17 credit note).
            let inv_id = format!("inv_placement_{id}");
            let payload = serde_json::json!({
                "id": id,
                "candidate_id": candidate_id,
                "vacancy_id": first_string_field(p, &["vacancy_id"]).unwrap_or_default(),
                "client_account_id": client_account_id,
                "fee_invoice_id": inv_id,
                "offer_id": first_string_field(p, &["offer_id"]).unwrap_or_default(),
                "start_ms": p.get("start_ms").and_then(Value::as_i64).unwrap_or(now),
                "guarantee_days": p.get("guarantee_days").and_then(Value::as_i64).unwrap_or(90),
                "fee": fee,
                "status": "confirmed",
                "created_at_ms": now,
                "updated_at_ms": now,
                "_deleted": false
            });
            upsert_business_record(&conn, "placements", &id, now, payload)?;
            // Emit the placement-fee draft invoice in the canonical, postable
            // accounting_invoices shape (flows through the invoices module).
            let invoice = ats_canonical_invoice(
                &inv_id,
                "sale_out",
                &client_account_id,
                vec![ats_invoice_line(1, "Vermittlungshonorar", fee, 0.19)],
                now,
                "source_placement_id",
                &id,
            );
            upsert_business_record(&conn, "accounting_invoices", &inv_id, now, invoice)?;
            // §5.3 DSGVO trail: record the placement lifecycle event.
            record_ats_governance_event(
                &conn,
                command,
                &actor,
                "business_os.ats.placement_created",
                serde_json::json!({
                    "candidate_id": candidate_id,
                    "client_account_id": client_account_id,
                    "placement_id": id,
                    "fee_invoice_id": inv_id,
                    "decision": "allowed"
                }),
                now,
            )?;
            Ok(
                serde_json::json!({ "ok": true, "allowed": true, "placement_id": id, "fee_invoice_id": inv_id }),
            )
        }
        "ats.submission.present" => {
            let candidate_id = first_string_field(p, &["candidate_id"])
                .context("ats.submission.present requires candidate_id")?;
            let client_account_id = first_string_field(p, &["client_account_id"])
                .context("ats.submission.present requires client_account_id")?;
            let consents = load_business_records_by_field(
                &conn,
                "business_consents",
                "subject_id",
                &candidate_id,
            )?;
            let require_evidence = crate::inference::runtime_env::env_or_config(
                root,
                "CTOX_BUSINESS_OS_REQUIRE_LEGAL_BASIS_EVIDENCE",
            )
            .as_deref()
                == Some("1");
            let has_consent = super::ats_gates::evaluate_consent_gate(
                Some("present_to_client"),
                &consents,
                now,
                require_evidence,
            );
            let existing: Vec<Value> = load_all_business_records(&conn, "submissions")?
                .into_iter()
                .map(|(_, v)| v)
                .collect();
            let conflict = super::ats_gates::find_double_submission(
                &existing,
                &candidate_id,
                &client_account_id,
                180,
                now,
            );
            let mut blockers = Vec::new();
            let mut reason_codes = Vec::new();
            if !has_consent {
                reason_codes.push(ATS_REASON_CONSENT_MISSING);
                blockers.push(serde_json::json!({ "reason": ATS_REASON_CONSENT_MISSING }));
            }
            if let Some(conflicting) = &conflict {
                reason_codes.push(ATS_REASON_DOUBLE_SUBMISSION);
                blockers.push(serde_json::json!({ "reason": ATS_REASON_DOUBLE_SUBMISSION, "conflicting_submission_id": conflicting }));
            }
            if !blockers.is_empty() {
                // §5.3 DSGVO trail: record the denied data-sharing attempt.
                record_ats_governance_event(
                    &conn,
                    command,
                    &actor,
                    "business_os.ats.candidate_present_denied",
                    serde_json::json!({
                        "candidate_id": candidate_id,
                        "client_account_id": client_account_id,
                        "decision": "denied",
                        "reason_codes": reason_codes
                    }),
                    now,
                )?;
                return Ok(
                    serde_json::json!({ "ok": true, "allowed": false, "blockers": blockers }),
                );
            }
            let id = format!("subm_{now}");
            let payload = serde_json::json!({
                "id": id,
                "candidate_id": candidate_id,
                "client_account_id": client_account_id,
                "vacancy_id": first_string_field(p, &["vacancy_id"]).unwrap_or_default(),
                "client_contact_id": first_string_field(p, &["client_contact_id"]).unwrap_or_default(),
                "sent_at_ms": now,
                "status": "sent",
                "created_at_ms": now,
                "updated_at_ms": now,
                "_deleted": false
            });
            upsert_business_record(&conn, "submissions", &id, now, payload)?;
            // §5.3 DSGVO trail: record which candidate was shared with which client.
            record_ats_governance_event(
                &conn,
                command,
                &actor,
                "business_os.ats.candidate_presented",
                serde_json::json!({
                    "candidate_id": candidate_id,
                    "client_account_id": client_account_id,
                    "submission_id": id,
                    "decision": "allowed"
                }),
                now,
            )?;
            Ok(serde_json::json!({ "ok": true, "allowed": true, "submission_id": id }))
        }
        "ats.leistungsnachweis.signoff" => {
            let collection = p
                .get("collection")
                .and_then(Value::as_str)
                .unwrap_or("planning_time_records");
            // Security: never let the signoff write its billing/signature fields
            // into a non-ATS collection (invoices, credentials, commands, …).
            ensure_ats_collection_allowed(collection)?;
            let record_id = first_string_field(p, &["record_id", "nachweis_id"])
                .context("ats.leistungsnachweis.signoff requires record_id")?;
            let existing = conn
                .query_row(
                    "SELECT payload_json FROM business_records WHERE collection = ?1 AND record_id = ?2 AND deleted = 0",
                    params![collection, record_id.as_str()],
                    |row| row.get::<_, String>(0),
                )
                .optional()?;
            let mut payload = existing
                .and_then(|raw| serde_json::from_str::<Value>(&raw).ok())
                .unwrap_or_else(|| serde_json::json!({ "id": record_id }));
            // §9.2 external sign-off proof: when enabled, the Entleiher signature
            // must be backed by a COMPLETED signature_request (by
            // signature_request_id or one whose document_id is this Nachweis) —
            // not just an internal admin assertion. Off by default (backward
            // compatible); on for hardened instances.
            let require_signature = crate::inference::runtime_env::env_or_config(
                root,
                "CTOX_BUSINESS_OS_REQUIRE_ENTLEIHER_SIGNATURE",
            )
            .as_deref()
                == Some("1");
            let signature_proven = if require_signature {
                let raw = match first_string_field(p, &["signature_request_id"]) {
                    Some(sig_id) => conn
                        .query_row(
                            "SELECT payload_json FROM business_records WHERE collection = 'signature_requests' AND record_id = ?1 AND deleted = 0",
                            params![sig_id.as_str()],
                            |row| row.get::<_, String>(0),
                        )
                        .optional()?,
                    None => conn
                        .query_row(
                            "SELECT payload_json FROM business_records WHERE collection = 'signature_requests' AND deleted = 0 AND json_extract(payload_json, '$.document_id') = ?1 ORDER BY updated_at_ms DESC LIMIT 1",
                            params![record_id.as_str()],
                            |row| row.get::<_, String>(0),
                        )
                        .optional()?,
                };
                raw.and_then(|raw| serde_json::from_str::<Value>(&raw).ok())
                    .and_then(|v| {
                        v.get("status")
                            .and_then(Value::as_str)
                            .map(|s| s == "completed")
                    })
                    .unwrap_or(false)
            } else {
                true
            };
            if signature_proven {
                if let Some(obj) = payload.as_object_mut() {
                    obj.insert("entleiher_signed".to_string(), Value::Bool(true));
                    obj.insert("signed_at_ms".to_string(), Value::from(now));
                }
            }
            // §5.9 Entleiher sign-off → billing gate. Billing requires the signed
            // Nachweis to carry positive billable hours AND a finite positive
            // charge_rate (Verrechnungssatz). billing_released is set true only
            // when an invoice is actually emitted, never on an empty/zero bill.
            let mut blockers: Vec<String> = super::ats_gates::evaluate_billing_gate(&payload)
                .iter()
                .map(|s| (*s).to_string())
                .collect();
            if require_signature && !signature_proven {
                blockers.push("entleiher_signature_proof_missing".to_string());
            }
            let charge_rate = p
                .get("charge_rate")
                .and_then(Value::as_f64)
                .or_else(|| payload.get("charge_rate").and_then(Value::as_f64))
                .unwrap_or(0.0);
            if !(charge_rate.is_finite() && charge_rate > 0.0) {
                blockers.push("missing_charge_rate".to_string());
            }
            let mut invoice_id = String::new();
            let mut net_total = 0.0;
            if blockers.is_empty() {
                let entries = payload
                    .get("entries")
                    .and_then(Value::as_array)
                    .cloned()
                    .unwrap_or_default();
                let surcharge = p
                    .get("surcharge_pct")
                    .cloned()
                    .or_else(|| payload.get("surcharge_pct").cloned())
                    .unwrap_or_else(|| serde_json::json!({}));
                let billing =
                    super::ats_gates::compute_nachweis_billing(&entries, charge_rate, &surcharge);
                net_total = billing.net_total;
                if net_total > 0.0 {
                    // Deterministic id: a replayed signoff updates the same
                    // invoice instead of minting a duplicate inv_{now}.
                    invoice_id = format!("inv_nachweis_{record_id}");
                    let account_id = payload
                        .get("entleiher_account_id")
                        .and_then(Value::as_str)
                        .map(ToOwned::to_owned)
                        .or_else(|| {
                            first_string_field(p, &["entleiher_account_id", "client_account_id"])
                        })
                        .unwrap_or_default();
                    let lines: Vec<Value> = billing
                        .lines
                        .iter()
                        .enumerate()
                        .map(|(i, l)| {
                            serde_json::json!({
                                "position": i + 1,
                                "description": format!("Arbeitsstunden ({})", l.category),
                                "quantity": checked_scaled(l.hours, 1000.0),
                                "unit_price_cents": checked_scaled(l.rate, 100.0),
                                "tax_rate": 0.19,
                                "account_code": "8400"
                            })
                        })
                        .collect();
                    let invoice = ats_canonical_invoice(
                        &invoice_id,
                        "sale_out",
                        &account_id,
                        lines,
                        now,
                        "source_nachweis_id",
                        &record_id,
                    );
                    upsert_business_record(
                        &conn,
                        "accounting_invoices",
                        &invoice_id,
                        now,
                        invoice,
                    )?;
                }
            }
            // Released only when an invoice was actually emitted.
            let billing_released = !invoice_id.is_empty();
            if let Some(obj) = payload.as_object_mut() {
                obj.insert(
                    "billing_released".to_string(),
                    Value::Bool(billing_released),
                );
            }
            upsert_business_record(&conn, collection, &record_id, now, payload)?;
            Ok(serde_json::json!({
                "ok": true,
                "record_id": record_id,
                "billing_released": billing_released,
                "blockers": blockers,
                "invoice_id": invoice_id,
                "net_total": net_total
            }))
        }
        "ats.signature.request" => {
            let document_id = first_string_field(p, &["document_id"])
                .context("ats.signature.request requires document_id")?;
            let signers = p
                .get("signers")
                .cloned()
                .unwrap_or_else(|| serde_json::json!([]));
            let id = format!("sig_{now}");
            let payload = serde_json::json!({
                "id": id,
                "document_id": document_id,
                "subject_kind": first_string_field(p, &["subject_kind"]).unwrap_or_default(),
                "signers": signers,
                "sent_at_ms": now,
                "status": "sent",
                "created_at_ms": now,
                "updated_at_ms": now,
                "_deleted": false
            });
            upsert_business_record(&conn, "signature_requests", &id, now, payload)?;
            record_ats_governance_event(
                &conn,
                command,
                &actor,
                "business_os.ats.signature_requested",
                serde_json::json!({ "request_id": id, "document_id": document_id }),
                now,
            )?;
            Ok(serde_json::json!({ "ok": true, "request_id": id, "status": "sent" }))
        }
        "ats.signature.sign" => {
            let request_id = first_string_field(p, &["request_id"])
                .context("ats.signature.sign requires request_id")?;
            let signer_id = first_string_field(p, &["signer_id"])
                .context("ats.signature.sign requires signer_id")?;
            // H11: signing must be performed by an authenticated actor; the event
            // is recorded as attributable (signed_by_actor_id) and integrity-
            // stamped (signed_artifact_id). NOTE: this still records an operator
            // acting on a signer's behalf — true signer self-authentication needs
            // a server-side signer identity model and is a separate change.
            let actor_id = session
                .user
                .as_ref()
                .map(|user| user.id.clone())
                .unwrap_or_default();
            anyhow::ensure!(
                !actor_id.is_empty() && actor_id != "rxdb-command",
                "ats.signature.sign requires an authenticated actor"
            );
            let existing = conn
                .query_row(
                    "SELECT payload_json FROM business_records WHERE collection = 'signature_requests' AND record_id = ?1 AND deleted = 0",
                    params![request_id.as_str()],
                    |row| row.get::<_, String>(0),
                )
                .optional()?
                .context("signature request not found")?;
            let mut payload: Value = serde_json::from_str(&existing)?;
            let mut signed_any = false;
            if let Some(signers) = payload.get_mut("signers").and_then(Value::as_array_mut) {
                for signer in signers.iter_mut() {
                    if signer.get("id").and_then(Value::as_str) == Some(signer_id.as_str()) {
                        if let Some(obj) = signer.as_object_mut() {
                            obj.insert("state".to_string(), Value::String("signed".to_string()));
                            obj.insert(
                                "signed_by_actor_id".to_string(),
                                Value::String(actor_id.clone()),
                            );
                            obj.insert("signed_at_ms".to_string(), serde_json::json!(now));
                        }
                        signed_any = true;
                    }
                }
            }
            anyhow::ensure!(
                signed_any,
                "signer_id does not match any signer on this request"
            );
            let signers = payload
                .get("signers")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default();
            let status = super::ats_gates::signature_request_status(
                &signers,
                payload.get("expires_at_ms").and_then(Value::as_i64),
                payload.get("sent_at_ms").and_then(Value::as_i64),
                now,
            );
            // Immutable signed artifact (content hash) for non-repudiation; the
            // signoff / AÜG path can require this on a completed request.
            let artifact_id = format!(
                "sig_{}",
                hex_sha256(
                    format!(
                        "{request_id}|{actor_id}|{now}|{}",
                        serde_json::to_string(&signers).unwrap_or_default()
                    )
                    .as_bytes()
                )
            );
            let completed = status == "completed";
            if let Some(obj) = payload.as_object_mut() {
                obj.insert("status".to_string(), Value::String(status.to_string()));
                if completed {
                    obj.insert(
                        "signed_artifact_id".to_string(),
                        Value::String(artifact_id.clone()),
                    );
                }
            }
            upsert_business_record(&conn, "signature_requests", &request_id, now, payload)?;
            record_ats_governance_event(
                &conn,
                command,
                &actor,
                "business_os.ats.signature_signed",
                serde_json::json!({
                    "request_id": request_id,
                    "signer_id": signer_id,
                    "status": status,
                    "signed_by_actor_id": actor_id,
                    "signed_artifact_id": completed.then(|| artifact_id.clone()),
                }),
                now,
            )?;
            Ok(serde_json::json!({
                "ok": true,
                "request_id": request_id,
                "status": status,
                "signed_artifact_id": completed.then_some(artifact_id),
            }))
        }
        "ats.credential.verify" => {
            // H12: the deployment / AÜG gate trusts the credential `verified`
            // flag, but no native command ever set it (the browser inserted
            // verified:false and there was no server-gated verification path).
            // This is the native verify — routed through is_ats_mutating_command
            // -> rxdb_command_session, so it is chef/admin + capability-token
            // gated — that stamps verified=true and the authenticated verifier.
            let credential_id = first_string_field(p, &["credential_id", "record_id", "id"])
                .context("ats.credential.verify requires credential_id")?;
            let verified_by = session
                .user
                .as_ref()
                .map(|user| user.id.clone())
                .unwrap_or_default();
            anyhow::ensure!(
                !verified_by.is_empty() && verified_by != "rxdb-command",
                "ats.credential.verify requires an authenticated verifier"
            );
            let existing = conn
                .query_row(
                    "SELECT payload_json FROM business_records WHERE collection = 'business_credentials' AND record_id = ?1 AND deleted = 0",
                    params![credential_id.as_str()],
                    |row| row.get::<_, String>(0),
                )
                .optional()?
                .context("credential not found")?;
            let mut payload: Value = serde_json::from_str(&existing)?;
            if let Some(obj) = payload.as_object_mut() {
                obj.insert("verified".to_string(), Value::Bool(true));
                obj.insert(
                    "verified_by".to_string(),
                    Value::String(verified_by.clone()),
                );
                obj.insert("verified_at_ms".to_string(), serde_json::json!(now));
            }
            upsert_business_record(&conn, "business_credentials", &credential_id, now, payload)?;
            record_ats_governance_event(
                &conn,
                command,
                &actor,
                "business_os.ats.credential_verified",
                serde_json::json!({ "credential_id": credential_id, "verified_by": verified_by }),
                now,
            )?;
            Ok(serde_json::json!({
                "ok": true,
                "credential_id": credential_id,
                "verified": true,
                "verified_by": verified_by,
            }))
        }
        "ats.placement.early_leave" => {
            // Guarantee/replacement: if the candidate leaves within the guarantee
            // window, draft a pro-rata clawback credit note (§5.11).
            let placement_id = first_string_field(p, &["placement_id"])
                .context("ats.placement.early_leave requires placement_id")?;
            let left_at_ms = p.get("left_at_ms").and_then(Value::as_i64).unwrap_or(now);
            let raw = conn
                .query_row(
                    "SELECT payload_json FROM business_records WHERE collection = 'placements' AND record_id = ?1 AND deleted = 0",
                    params![placement_id.as_str()],
                    |row| row.get::<_, String>(0),
                )
                .optional()?
                .context("placement not found")?;
            let mut placement: Value = serde_json::from_str(&raw)?;
            let start = placement
                .get("start_ms")
                .and_then(Value::as_i64)
                .unwrap_or(0);
            let days = placement
                .get("guarantee_days")
                .and_then(Value::as_i64)
                .unwrap_or(0);
            let fee = placement.get("fee").and_then(Value::as_f64).unwrap_or(0.0);
            let client_account_id = placement
                .get("client_account_id")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            let within = start > 0 && days > 0 && left_at_ms < start + days * 86_400_000;
            let mut clawback = 0.0;
            let mut credit_note_id = String::new();
            if within {
                let served = ((left_at_ms - start) / 86_400_000).clamp(0, days);
                let remaining_ratio = (days - served) as f64 / days as f64;
                clawback = (fee * remaining_ratio * 100.0).round() / 100.0;
                // Deterministic id (replay-safe) and reference the placement-fee
                // invoice so the clawback is a valid, postable §17 credit note.
                credit_note_id = format!("cn_earlyleave_{placement_id}");
                let fee_invoice_id = placement
                    .get("fee_invoice_id")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                let mut credit = ats_canonical_invoice(
                    &credit_note_id,
                    "credit_note_out",
                    &client_account_id,
                    vec![ats_invoice_line(
                        1,
                        "Anteilige Gutschrift (Garantie/Frühausstieg)",
                        clawback,
                        0.19,
                    )],
                    now,
                    "source_placement_id",
                    &placement_id,
                );
                if let Some(obj) = credit.as_object_mut() {
                    obj.insert(
                        "credit_note_for_id".to_string(),
                        Value::String(fee_invoice_id),
                    );
                }
                upsert_business_record(&conn, "accounting_invoices", &credit_note_id, now, credit)?;
            }
            if let Some(obj) = placement.as_object_mut() {
                obj.insert(
                    "status".to_string(),
                    Value::String("early_leave".to_string()),
                );
                obj.insert("left_at_ms".to_string(), Value::from(left_at_ms));
                obj.insert("updated_at_ms".to_string(), Value::from(now));
            }
            upsert_business_record(&conn, "placements", &placement_id, now, placement)?;
            record_ats_governance_event(
                &conn,
                command,
                &actor,
                "business_os.ats.placement_early_leave",
                serde_json::json!({
                    "placement_id": placement_id,
                    "within_guarantee": within,
                    "clawback": clawback,
                    "credit_note_id": credit_note_id
                }),
                now,
            )?;
            Ok(serde_json::json!({
                "ok": true,
                "placement_id": placement_id,
                "within_guarantee": within,
                "clawback": clawback,
                "credit_note_id": credit_note_id
            }))
        }
        "ats.interview.transcribe" => {
            // §5.10 Interview transcription: transcribe an interview recording via
            // the native STT runtime and write the transcript back onto the
            // interview_meetings record. The audio MUST be a Business OS desktop
            // file referenced by `source_file_id`, reconstructed and integrity-
            // verified from RxDB chunks (the browser-upload flow) — the data plane
            // stays WebRTC-only, no HTTP. A raw `audio_path` is intentionally NOT
            // accepted: an RxDB command must not be able to point the daemon at an
            // arbitrary local file. When the STT model weights are not installed
            // the runtime returns a clear error; surface it as a non-fatal outcome
            // so the command records a failed transcription instead of poisoning
            // the meeting record.
            let meeting_id = first_string_field(p, &["meeting_id", "record_id"])
                .context("ats.interview.transcribe requires meeting_id")?;
            // Security: the transcript may only ever be written onto the meeting
            // record — the target collection is fixed, never payload-controlled.
            let collection = "interview_meetings";
            let file_id = first_string_field(p, &["source_file_id"])
                .context("ats.interview.transcribe requires source_file_id")?;
            // DoS guard: bound the audio size before reconstructing/decoding every
            // chunk into memory.
            const MAX_AUDIO_BYTES: u64 = 256 * 1024 * 1024;
            let declared_size = rxdb_desktop_file_document(root, &file_id)
                .ok()
                .and_then(|doc| doc.get("size_bytes").and_then(Value::as_u64))
                .unwrap_or(0);
            anyhow::ensure!(
                declared_size <= MAX_AUDIO_BYTES,
                "interview audio {file_id} is too large ({declared_size} bytes, max {MAX_AUDIO_BYTES})"
            );
            let generation = first_string_field(p, &["generation_id"]);
            let materialized = materialize_business_chat_attachment(
                root,
                command.id.as_deref().unwrap_or("ats-transcribe"),
                &serde_json::json!({}),
                &file_id,
                generation.as_deref(),
            )?;
            let audio_path = std::path::PathBuf::from(materialized.local_path);
            match crate::execution::models::native_stt::transcribe_audio_file(root, &audio_path) {
                Ok(resp) => {
                    let transcript_id = format!("transcript_{now}");
                    let existing = conn
                        .query_row(
                            "SELECT payload_json FROM business_records WHERE collection = ?1 AND record_id = ?2 AND deleted = 0",
                            params![collection, meeting_id.as_str()],
                            |row| row.get::<_, String>(0),
                        )
                        .optional()?;
                    let mut payload = existing
                        .and_then(|raw| serde_json::from_str::<Value>(&raw).ok())
                        .unwrap_or_else(|| serde_json::json!({ "id": meeting_id }));
                    let char_count = resp.text.chars().count();
                    if let Some(obj) = payload.as_object_mut() {
                        obj.insert(
                            "transcript_id".to_string(),
                            Value::String(transcript_id.clone()),
                        );
                        obj.insert("transcript_text".to_string(), Value::String(resp.text));
                        obj.insert("transcript_model".to_string(), Value::String(resp.model));
                        obj.insert("transcribed_at_ms".to_string(), Value::from(now));
                    }
                    upsert_business_record(&conn, collection, &meeting_id, now, payload)?;
                    record_ats_governance_event(
                        &conn,
                        command,
                        &actor,
                        "business_os.ats.interview_transcribed",
                        serde_json::json!({
                            "meeting_id": meeting_id,
                            "transcript_id": transcript_id,
                            "transcript_chars": char_count,
                            "source_file_id": file_id
                        }),
                        now,
                    )?;
                    Ok(serde_json::json!({
                        "ok": true,
                        "meeting_id": meeting_id,
                        "transcript_id": transcript_id,
                        "transcript_chars": char_count
                    }))
                }
                Err(err) => Ok(serde_json::json!({
                    "ok": false,
                    "meeting_id": meeting_id,
                    "transcription_available": false,
                    "error": err.to_string()
                })),
            }
        }
        "ats.subject.export" => {
            // §9.2 DSGVO Art. 15 access: gather every PII record held about a
            // subject (candidate/worker) across the ATS collections plus the
            // governance audit trail, for a data-subject access request. chef/
            // admin gated (privileged read); the export itself is audited.
            let subject_id = first_string_field(p, &["subject_id", "candidate_id"])
                .context("ats.subject.export requires subject_id")?;
            let mut collections = serde_json::Map::new();
            for coll in ATS_WRITABLE_COLLECTIONS {
                let mut records =
                    load_business_records_by_field(&conn, coll, "subject_id", &subject_id)?;
                for rec in load_business_records_by_field(&conn, coll, "candidate_id", &subject_id)?
                {
                    let id = rec.get("id").and_then(Value::as_str).map(ToOwned::to_owned);
                    let dup = id
                        .as_deref()
                        .map(|rid| {
                            records
                                .iter()
                                .any(|r| r.get("id").and_then(Value::as_str) == Some(rid))
                        })
                        .unwrap_or(false);
                    if !dup {
                        records.push(rec);
                    }
                }
                if !records.is_empty() {
                    collections.insert((*coll).to_string(), Value::Array(records));
                }
            }
            // The audit trail rows that name this subject (who shared/processed it).
            let mut audit = Vec::new();
            {
                let mut stmt = conn.prepare(
                    "SELECT payload_json FROM business_events
                     WHERE command_type LIKE 'business_os.ats.%'
                       AND json_extract(payload_json, '$.summary.candidate_id') = ?1
                     ORDER BY observed_at_ms",
                )?;
                let rows =
                    stmt.query_map(params![subject_id.as_str()], |row| row.get::<_, String>(0))?;
                for row in rows {
                    if let Ok(value) = serde_json::from_str::<Value>(&row?) {
                        audit.push(value);
                    }
                }
            }
            let record_count: usize = collections
                .values()
                .filter_map(Value::as_array)
                .map(Vec::len)
                .sum();
            // The export is itself a processing event — record it.
            record_ats_governance_event(
                &conn,
                command,
                &actor,
                "business_os.ats.subject_exported",
                serde_json::json!({
                    "candidate_id": subject_id,
                    "record_count": record_count,
                    "audit_event_count": audit.len()
                }),
                now,
            )?;
            Ok(serde_json::json!({
                "ok": true,
                "subject_id": subject_id,
                "exported_at_ms": now,
                "record_count": record_count,
                "collections": collections,
                "audit_trail": audit
            }))
        }
        "ats.subject.erase" => {
            // §9.2 DSGVO Art. 17 erasure (right to be forgotten): redact +
            // tombstone every PII record about a subject across the ATS
            // collections, and return an erasure report. chef/admin gated; the
            // erasure is itself recorded as a (non-PII) processing event.
            let subject_id = first_string_field(p, &["subject_id", "candidate_id"])
                .context("ats.subject.erase requires subject_id")?;
            let mut erased = serde_json::Map::new();
            for coll in ATS_WRITABLE_COLLECTIONS {
                let mut ids: Vec<String> = Vec::new();
                for field in ["subject_id", "candidate_id"] {
                    for rec in load_business_records_by_field(&conn, coll, field, &subject_id)? {
                        if let Some(id) = rec.get("id").and_then(Value::as_str) {
                            if !ids.iter().any(|x| x == id) {
                                ids.push(id.to_string());
                            }
                        }
                    }
                }
                for id in &ids {
                    let tombstone = serde_json::json!({
                        "id": id,
                        "_deleted": true,
                        "redacted": true,
                        "redacted_at_ms": now,
                        "erasure_basis": "dsgvo_art17"
                    });
                    conn.execute(
                        "UPDATE business_records SET deleted = 1, updated_at_ms = ?3, payload_json = ?4 WHERE collection = ?1 AND record_id = ?2",
                        params![coll, id.as_str(), now, serde_json::to_string(&tombstone)?],
                    )?;
                }
                if !ids.is_empty() {
                    erased.insert(
                        (*coll).to_string(),
                        Value::Array(ids.into_iter().map(Value::String).collect()),
                    );
                }
            }
            let erased_count: usize = erased
                .values()
                .filter_map(Value::as_array)
                .map(Vec::len)
                .sum();
            record_ats_governance_event(
                &conn,
                command,
                &actor,
                "business_os.ats.subject_erased",
                serde_json::json!({ "candidate_id": subject_id, "erased_count": erased_count }),
                now,
            )?;
            Ok(serde_json::json!({
                "ok": true,
                "subject_id": subject_id,
                "erased_at_ms": now,
                "erased_count": erased_count,
                "erased": erased
            }))
        }
        other => Err(anyhow::anyhow!(
            "unsupported ats mutating command type: {other}"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::super::store::tests::test_session;
    use super::super::store::{
        now_ms, open_store, BusinessCommand, BusinessOsSession, BusinessOsSessionUser,
        CommandOrigin,
    };
    use super::super::store_projections::upsert_business_record;
    use super::handle_ats_mutating_command;
    use serde_json::Value;

    // H12: the native ats.credential.verify command stamps verified=true and the
    // authenticated verifier — the deployment/AÜG gate no longer trusts a flag
    // that only the browser could set.
    #[test]
    fn ats_credential_verify_stamps_verified_and_verifier() -> anyhow::Result<()> {
        let root = tempfile::tempdir()?;
        let now = now_ms() as i64;
        {
            let conn = open_store(root.path())?;
            upsert_business_record(
                &conn,
                "business_credentials",
                "cred-1",
                now,
                serde_json::json!({
                    "id": "cred-1", "subject_id": "cand-1",
                    "credential_type": "license", "verified": false
                }),
            )?;
        }
        let session = test_session("chef1", "chef");
        let command = BusinessCommand {
            origin: CommandOrigin::TrustedLocal,
            id: Some("vc1".into()),
            module: "ats".into(),
            command_type: "ats.credential.verify".into(),
            record_id: None,
            payload: serde_json::json!({ "credential_id": "cred-1" }),
            client_context: Value::Null,
        };
        let outcome = handle_ats_mutating_command(root.path(), &session, &command)?;
        assert_eq!(outcome.get("verified").and_then(Value::as_bool), Some(true));
        assert_eq!(
            outcome.get("verified_by").and_then(Value::as_str),
            Some("chef1")
        );

        let conn = open_store(root.path())?;
        let stored: String = conn.query_row(
            "SELECT payload_json FROM business_records WHERE collection = 'business_credentials' AND record_id = 'cred-1'",
            [],
            |row| row.get(0),
        )?;
        let v: Value = serde_json::from_str(&stored)?;
        assert_eq!(v.get("verified").and_then(Value::as_bool), Some(true));
        assert_eq!(v.get("verified_by").and_then(Value::as_str), Some("chef1"));
        Ok(())
    }

    // H11: ats.signature.sign requires an authenticated actor and stamps an
    // attributable signed_by_actor_id on the signer.
    #[test]
    fn ats_signature_sign_requires_actor_and_stamps_attribution() -> anyhow::Result<()> {
        let root = tempfile::tempdir()?;
        let now = now_ms() as i64;
        {
            let conn = open_store(root.path())?;
            upsert_business_record(
                &conn,
                "signature_requests",
                "req-1",
                now,
                serde_json::json!({
                    "id": "req-1", "document_id": "doc-1", "sent_at_ms": now,
                    "signers": [{ "id": "s1", "state": "pending" }]
                }),
            )?;
        }
        let command = BusinessCommand {
            origin: CommandOrigin::TrustedLocal,
            id: Some("sg1".into()),
            module: "ats".into(),
            command_type: "ats.signature.sign".into(),
            record_id: None,
            payload: serde_json::json!({ "request_id": "req-1", "signer_id": "s1" }),
            client_context: Value::Null,
        };

        // An unauthenticated actor (no session user) is rejected.
        let anon = BusinessOsSession {
            ok: true,
            authenticated: true,
            auth_required: false,
            user: None,
            login_url: None,
            reason: None,
        };
        assert!(
            handle_ats_mutating_command(root.path(), &anon, &command).is_err(),
            "unauthenticated signing must be rejected"
        );

        // An authenticated actor signs and is recorded on the signer.
        let session = test_session("chef1", "chef");
        handle_ats_mutating_command(root.path(), &session, &command)?;
        let conn = open_store(root.path())?;
        let stored: String = conn.query_row(
            "SELECT payload_json FROM business_records WHERE collection = 'signature_requests' AND record_id = 'req-1'",
            [],
            |row| row.get(0),
        )?;
        let v: Value = serde_json::from_str(&stored)?;
        let signer = &v["signers"][0];
        assert_eq!(signer.get("state").and_then(Value::as_str), Some("signed"));
        assert_eq!(
            signer.get("signed_by_actor_id").and_then(Value::as_str),
            Some("chef1"),
            "signer must record the authenticated signing actor: {v:?}"
        );
        Ok(())
    }
}
