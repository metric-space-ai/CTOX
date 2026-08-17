use anyhow::Context;
use ctox_web_stack::sources::{Country, FieldKey, ResearchMode};
use ctox_web_stack::PersonResearchRequest;
use serde::Deserialize;
use serde_json::Value;
use std::collections::HashSet;
use std::panic::{self, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

use super::store::{self, BusinessCommand};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ActiveResearchCommandKey {
    root: PathBuf,
    command_id: String,
}

static ACTIVE_RESEARCH_COMMANDS: OnceLock<Mutex<HashSet<ActiveResearchCommandKey>>> =
    OnceLock::new();

#[derive(Debug, Deserialize)]
struct PersonResearchCommandRequest {
    company: String,
    country: String,
    mode: String,
    #[serde(default)]
    fields: Vec<String>,
    #[serde(default)]
    include_private: Vec<String>,
    #[serde(default)]
    auto_browser_capture: bool,
}

pub(super) fn start(root: &Path, command: BusinessCommand) -> anyhow::Result<Value> {
    command
        .id
        .as_deref()
        .context("person-research command id is required")?;
    let mut running = store::write_rxdb_control_command_progress(
        root,
        &command,
        "running",
        serde_json::json!({
            "ok": true,
            "status": "running",
            "summary": "Recherche wurde gestartet."
        }),
    )?;
    match spawn_worker(root.to_path_buf(), command.clone()) {
        Ok(worker_started) => {
            if let Some(object) = running.as_object_mut() {
                object.insert("worker_started".to_string(), Value::Bool(worker_started));
            }
        }
        Err(error) => {
            let message = error.to_string();
            let failed = store::write_rxdb_failed_control_command_outcome(
                root,
                &command,
                "person_research_start",
                error,
            );
            if failed.is_ok() {
                log_lead_projection_error(
                    command.id.as_deref().unwrap_or_default(),
                    project_thesen_outbound_lead_state(
                        root,
                        &command,
                        "failed",
                        Some(message.as_str()),
                        None,
                    ),
                );
            }
            return failed;
        }
    }
    log_lead_projection_error(
        command.id.as_deref().unwrap_or_default(),
        project_thesen_outbound_lead_state(root, &command, "running", None, None),
    );
    Ok(running)
}

pub(crate) fn recover_once(root: &Path) -> anyhow::Result<usize> {
    let terminal_candidates = store::terminal_person_research_projection_candidates(root)?;
    let mut recovered = 0;
    for candidate in terminal_candidates {
        let command_id = candidate
            .command
            .id
            .as_deref()
            .context("terminal person-research candidate is missing command id")?;
        super::command_plane::complete_and_project_business_control_command(
            root,
            command_id,
            &candidate.terminal_status,
            &candidate.result,
            candidate.error_message.as_deref(),
        )?;
        let (lead_status, lead_error) = if candidate.terminal_status == "completed" {
            ("needs_review", None)
        } else {
            ("failed", candidate.error_message.as_deref())
        };
        log_lead_projection_error(
            command_id,
            project_thesen_outbound_lead_state(
                root,
                &candidate.command,
                lead_status,
                lead_error,
                (candidate.terminal_status == "completed").then_some(&candidate.result),
            ),
        );
        recovered += 1;
    }
    let commands = store::recoverable_person_research_commands(root)?;
    let mut started = recovered;
    for recoverable in commands {
        let worker_started = if recoverable.status == "accepted" {
            match store::authorize_recoverable_person_research_command(root, &recoverable.command) {
                Ok(command) => start(root, command)?
                    .get("worker_started")
                    .and_then(Value::as_bool)
                    .unwrap_or(false),
                Err(error) => {
                    store::write_rxdb_failed_control_command_outcome(
                        root,
                        &recoverable.command,
                        "person_research_authorization",
                        error,
                    )?;
                    false
                }
            }
        } else {
            // `running` is only persisted after the normal intake path completed
            // authentication and policy checks, so a crash-safe replay can resume
            // the worker without treating browser-controlled actor data as local.
            spawn_worker(root.to_path_buf(), recoverable.command)?
        };
        if worker_started {
            started += 1;
        }
    }
    Ok(started)
}

fn spawn_worker(root: PathBuf, command: BusinessCommand) -> anyhow::Result<bool> {
    let command_id = command
        .id
        .as_deref()
        .context("person-research command id is required")?
        .to_string();
    let Some(active_guard) = ActiveResearchCommandGuard::claim(&root, &command_id) else {
        return Ok(false);
    };
    let worker_command = command;
    let worker_command_id = command_id.clone();
    let spawn_result = std::thread::Builder::new()
        .name(format!(
            "ctox-person-research-{}",
            safe_workspace_segment(&command_id)
        ))
        .spawn(move || {
            let _active_guard = active_guard;
            let result =
                panic::catch_unwind(AssertUnwindSafe(|| execute(&root, &worker_command.payload)));
            let (persisted, lead_status, lead_error, lead_result) = match result {
                Ok(Ok(outcome)) => {
                    let lead_result = outcome.clone();
                    (
                        store::write_rxdb_control_command_outcome(
                            &root,
                            &worker_command,
                            "completed",
                            None,
                            Some("completed"),
                            outcome,
                        ),
                        "needs_review",
                        None,
                        Some(lead_result),
                    )
                }
                Ok(Err(error)) => {
                    let message = error.to_string();
                    (
                        store::write_rxdb_failed_control_command_outcome(
                            &root,
                            &worker_command,
                            "person_research",
                            error,
                        ),
                        "failed",
                        Some(message),
                        None,
                    )
                }
                Err(_) => {
                    let message = "person-research worker panicked".to_string();
                    (
                        store::write_rxdb_failed_control_command_outcome(
                            &root,
                            &worker_command,
                            "person_research",
                            anyhow::anyhow!(message.clone()),
                        ),
                        "failed",
                        Some(message),
                        None,
                    )
                }
            };
            if let Err(error) = &persisted {
                eprintln!(
                    "[business-os] person research `{worker_command_id}` outcome failed: {error:#}"
                );
            } else {
                log_lead_projection_error(
                    &worker_command_id,
                    project_thesen_outbound_lead_state(
                        &root,
                        &worker_command,
                        lead_status,
                        lead_error.as_deref(),
                        lead_result.as_ref(),
                    ),
                );
            }
        });
    if let Err(error) = spawn_result {
        return Err(error.into());
    }
    Ok(true)
}

fn project_thesen_outbound_lead_state(
    root: &Path,
    command: &BusinessCommand,
    research_status: &str,
    error: Option<&str>,
    result: Option<&Value>,
) -> anyhow::Result<()> {
    let Some(record_id) = thesen_outbound_writeback_record_id(command) else {
        return Ok(());
    };
    let now = now_ms();
    let command_id = command.id.as_deref().unwrap_or_default();
    let mut lead_document =
        thesen_outbound_lead_state_document(record_id, command_id, research_status, error, now);
    if let Some(result) = result {
        let existing =
            store::load_rxdb_collection_record(root, "thesen_outbound_leads", record_id)?
                .unwrap_or_else(|| serde_json::json!({ "id": record_id }));
        let outcome_patch = thesen_outbound_research_outcome_patch(&existing, result, now);
        merge_json_object_values(&mut lead_document, &outcome_patch);
    }
    store::upsert_rxdb_collection_record(
        root,
        "thesen_outbound_leads",
        record_id,
        now,
        lead_document,
    )
}

fn thesen_outbound_lead_state_document(
    record_id: &str,
    command_id: &str,
    research_status: &str,
    error: Option<&str>,
    now: i64,
) -> Value {
    let mut document = serde_json::json!({
        "id": record_id,
        "research_status": research_status,
        "command_id": command_id,
        "task_id": "",
        "payload": thesen_outbound_lead_state_patch(command_id, research_status, error, now),
    });
    if research_status != "running" {
        document["research_error"] = error
            .filter(|value| !value.trim().is_empty())
            .map(|value| Value::String(value.to_string()))
            .unwrap_or(Value::Null);
        document["research_updated_at_ms"] = Value::Number(now.into());
    }
    document
}

fn thesen_outbound_research_outcome_patch(
    existing: &Value,
    command_result: &Value,
    now: i64,
) -> Value {
    let outcome = command_result
        .get("result")
        .filter(|value| value.is_object())
        .unwrap_or(command_result);
    let mut data = existing
        .get("data")
        .filter(|value| value.is_object())
        .cloned()
        .unwrap_or_else(|| serde_json::json!({}));
    let mut contacts = existing
        .get("contacts")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let mut contact = contacts
        .first()
        .filter(|value| value.is_object())
        .cloned()
        .unwrap_or_else(|| serde_json::json!({}));
    let person_records = outcome
        .get("person_records")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let has_person_records = !person_records.is_empty();
    let mut evidence = existing
        .get("evidence")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let mut researched_field_keys = Vec::new();

    for (field_key, field) in outcome
        .get("fields")
        .and_then(Value::as_object)
        .into_iter()
        .flatten()
    {
        let Some(value) = field
            .get("value")
            .filter(|value| research_scalar_is_populated(value))
        else {
            continue;
        };
        researched_field_keys.push(field_key.clone());
        if field_key.starts_with("person_") && !has_person_records {
            contact[field_key] = value.clone();
        } else if !field_key.starts_with("person_") {
            data[field_key] = value.clone();
        }
        for candidate in field
            .get("candidates")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            let source_id = candidate
                .get("source_id")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let source_url = candidate
                .get("source_url")
                .and_then(Value::as_str)
                .unwrap_or_default();
            if source_id.trim().is_empty() && source_url.trim().is_empty() {
                continue;
            }
            evidence.push(serde_json::json!({
                "field_key": field_key,
                "value": candidate.get("value").cloned().unwrap_or(Value::Null),
                "confidence": candidate.get("confidence").cloned().unwrap_or(Value::Null),
                "source_id": source_id,
                "source_url": source_url,
                "tier": candidate.get("tier").cloned().unwrap_or(Value::Null),
                "via": candidate.get("via").cloned().unwrap_or(Value::Null),
                "label": if source_id.trim().is_empty() { source_url } else { source_id },
            }));
        }
    }

    evidence = deduplicate_research_evidence(evidence);
    if has_person_records {
        merge_researched_person_records(
            existing.get("id").and_then(Value::as_str).unwrap_or("lead"),
            &mut contacts,
            person_records,
        );
    } else if let Some(normalized) = normalize_researched_contact(contact) {
        let normalized = with_stable_contact_id(
            existing.get("id").and_then(Value::as_str).unwrap_or("lead"),
            normalized,
            0,
        );
        if contacts.is_empty() {
            contacts.push(normalized);
        } else {
            contacts[0] = normalized;
        }
    }
    let contact_ids = contacts
        .iter()
        .filter_map(|contact| contact.get("id").and_then(Value::as_str))
        .collect::<HashSet<_>>();
    let selected_contact_ids = existing
        .get("selected_contact_ids")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .filter(|id| contact_ids.contains(id))
        .map(|id| Value::String(id.to_string()))
        .collect::<Vec<_>>();
    let unverified_field_keys = researched_field_keys
        .iter()
        .filter(|field_key| independent_research_evidence_count(&evidence, field_key) < 2)
        .cloned()
        .collect::<Vec<_>>();
    let verified_field_keys = researched_field_keys
        .iter()
        .filter(|field_key| !unverified_field_keys.contains(field_key))
        .cloned()
        .collect::<Vec<_>>();

    serde_json::json!({
        "data": data,
        "contacts": contacts,
        "selected_contact_ids": selected_contact_ids,
        "evidence": evidence,
        "research_status": if !researched_field_keys.is_empty() && unverified_field_keys.is_empty() { "completed" } else { "needs_review" },
        "research_error": Value::Null,
        "research_updated_at_ms": now,
        "payload": {
            "researched_field_keys": researched_field_keys,
            "verified_field_keys": verified_field_keys,
            "unverified_field_keys": unverified_field_keys,
            "research_finished_at_ms": now,
            "research_tool": outcome.get("tool").cloned().unwrap_or(Value::Null),
            "browser_assist_tasks": outcome.get("browser_assist_tasks").cloned().unwrap_or_else(|| Value::Array(Vec::new())),
            "authenticated_source_capture_runs": outcome.get("authenticated_source_capture_runs").cloned().unwrap_or_else(|| Value::Array(Vec::new())),
        }
    })
}

fn merge_researched_person_records(
    lead_id: &str,
    contacts: &mut Vec<Value>,
    person_records: Vec<Value>,
) {
    for (index, record) in person_records.into_iter().enumerate() {
        let Some(mut normalized) = normalize_researched_contact(record) else {
            continue;
        };
        normalized = with_stable_contact_id(lead_id, normalized, index);
        let id = normalized.get("id").and_then(Value::as_str);
        let profile = contact_string(
            &normalized,
            &["person_linkedin", "person_xing", "linkedin", "xing"],
        );
        let existing_index = contacts.iter().position(|existing| {
            existing.get("id").and_then(Value::as_str) == id
                || (!profile.is_empty()
                    && contact_string(
                        existing,
                        &["person_linkedin", "person_xing", "linkedin", "xing"],
                    ) == profile)
        });
        if let Some(existing_index) = existing_index {
            merge_json_object_values(&mut contacts[existing_index], &normalized);
        } else {
            contacts.push(normalized);
        }
    }
}

fn research_scalar_is_populated(value: &Value) -> bool {
    match value {
        Value::String(value) => !value.trim().is_empty(),
        Value::Number(_) | Value::Bool(_) => true,
        _ => false,
    }
}

fn deduplicate_research_evidence(entries: Vec<Value>) -> Vec<Value> {
    let mut seen = HashSet::new();
    entries
        .into_iter()
        .filter(|entry| {
            let key = [
                entry
                    .get("field_key")
                    .or_else(|| entry.get("field"))
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
                entry
                    .get("source_id")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
                entry
                    .get("source_url")
                    .or_else(|| entry.get("url"))
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
                entry.get("value").map(Value::to_string).unwrap_or_default(),
            ]
            .join("|")
            .to_ascii_lowercase();
            !key.replace('|', "").is_empty() && seen.insert(key)
        })
        .collect()
}

fn independent_research_evidence_count(evidence: &[Value], field_key: &str) -> usize {
    evidence
        .iter()
        .filter(|entry| {
            entry
                .get("field_key")
                .or_else(|| entry.get("field"))
                .and_then(Value::as_str)
                == Some(field_key)
        })
        .filter_map(research_evidence_source_key)
        .collect::<HashSet<_>>()
        .len()
}

fn research_evidence_source_key(entry: &Value) -> Option<String> {
    if let Some(source_id) = entry
        .get("source_id")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        return Some(source_id.to_ascii_lowercase());
    }
    let source_url = entry
        .get("source_url")
        .or_else(|| entry.get("url"))
        .and_then(Value::as_str)?;
    url::Url::parse(source_url)
        .ok()
        .and_then(|url| {
            url.host_str()
                .map(|host| host.trim_start_matches("www.").to_string())
        })
        .filter(|host| !host.is_empty())
}

fn normalize_researched_contact(mut contact: Value) -> Option<Value> {
    let first_name = contact
        .get("person_vorname")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim();
    let last_name = contact
        .get("person_nachname")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim();
    let name = format!("{first_name} {last_name}").trim().to_string();
    let email = contact_string(&contact, &["person_email", "email"]);
    let phone = contact_string(&contact, &["person_telefon", "phone"]);
    let position = contact_string(&contact, &["person_position", "position"]);
    let profile = contact_string(
        &contact,
        &["person_linkedin", "person_xing", "linkedin", "xing"],
    );
    if name.is_empty()
        && email.is_empty()
        && phone.is_empty()
        && position.is_empty()
        && profile.is_empty()
    {
        return None;
    }
    let role = contact_string(&contact, &["person_funktion", "person_position", "role"]);
    if let Some(object) = contact.as_object_mut() {
        object.insert("name".to_string(), Value::String(name));
        object.insert("role".to_string(), Value::String(role));
        object.insert("position".to_string(), Value::String(position));
        object.insert("email".to_string(), Value::String(email));
        object.insert("phone".to_string(), Value::String(phone));
    }
    Some(contact)
}

fn contact_string(contact: &Value, keys: &[&str]) -> String {
    keys.iter()
        .find_map(|key| {
            contact
                .get(*key)
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|value| !value.is_empty())
        })
        .unwrap_or_default()
        .to_string()
}

fn with_stable_contact_id(lead_id: &str, mut contact: Value, index: usize) -> Value {
    if contact
        .get("id")
        .and_then(Value::as_str)
        .is_some_and(|id| !id.trim().is_empty())
    {
        return contact;
    }
    let identity = [
        "name",
        "first_name",
        "last_name",
        "person_vorname",
        "person_nachname",
        "email",
        "person_email",
        "phone",
        "person_telefon",
        "linkedin",
        "xing",
    ]
    .iter()
    .map(|key| {
        contact
            .get(*key)
            .and_then(Value::as_str)
            .unwrap_or_default()
            .trim()
            .to_lowercase()
    })
    .collect::<Vec<_>>()
    .join("|");
    let identity = if identity.replace('|', "").is_empty() {
        format!("position-{index}")
    } else {
        identity
    };
    let id = format!(
        "contact_{}",
        javascript_fingerprint(&format!("{lead_id}|{identity}"))
    );
    contact["id"] = Value::String(id);
    contact
}

fn javascript_fingerprint(value: &str) -> String {
    let mut hash = 2_166_136_261_u32;
    for unit in value.to_lowercase().encode_utf16() {
        hash ^= u32::from(unit);
        hash = hash.wrapping_mul(16_777_619);
    }
    base36(hash)
}

fn base36(mut value: u32) -> String {
    const DIGITS: &[u8; 36] = b"0123456789abcdefghijklmnopqrstuvwxyz";
    if value == 0 {
        return "0".to_string();
    }
    let mut output = Vec::new();
    while value > 0 {
        output.push(DIGITS[(value % 36) as usize] as char);
        value /= 36;
    }
    output.iter().rev().collect()
}

fn merge_json_object_values(target: &mut Value, patch: &Value) {
    let (Some(target), Some(patch)) = (target.as_object_mut(), patch.as_object()) else {
        *target = patch.clone();
        return;
    };
    for (key, value) in patch {
        match (target.get_mut(key), value) {
            (Some(existing), Value::Object(_)) if existing.is_object() => {
                merge_json_object_values(existing, value);
            }
            _ => {
                target.insert(key.clone(), value.clone());
            }
        }
    }
}

fn thesen_outbound_lead_state_patch(
    command_id: &str,
    research_status: &str,
    error: Option<&str>,
    now: i64,
) -> Value {
    let mut lead_payload = serde_json::json!({
        "last_research_command_id": command_id,
        "native_research_terminal_status": if research_status == "running" { Value::Null } else { Value::String(research_status.to_string()) },
    });
    if research_status == "running" {
        lead_payload["research_started_at_ms"] = Value::Number(now.into());
    } else {
        lead_payload["research_finished_at_ms"] = Value::Number(now.into());
    }
    if let Some(error) = error.filter(|value| !value.trim().is_empty()) {
        lead_payload["research_error"] = Value::String(error.to_string());
    } else if research_status != "running" {
        // Lifecycle patches are merged into the durable lead. A successful
        // retry must explicitly clear an error left by an older failed or
        // abandoned attempt instead of displaying both success and failure.
        lead_payload["research_error"] = Value::Null;
    }
    lead_payload
}

fn thesen_outbound_writeback_record_id(command: &BusinessCommand) -> Option<&str> {
    if command.module.trim() != "thesen-outbound" {
        return None;
    }
    let record_id = command.record_id.as_deref()?.trim();
    if record_id.is_empty() {
        return None;
    }
    let contract = command.payload.get("writeback_contract")?;
    let collection_allowed = contract
        .get("collection")
        .and_then(Value::as_str)
        .is_some_and(|collection| collection == "thesen_outbound_leads")
        || contract
            .get("allowed_collections")
            .and_then(Value::as_array)
            .is_some_and(|collections| {
                collections
                    .iter()
                    .filter_map(Value::as_str)
                    .any(|collection| collection == "thesen_outbound_leads")
            });
    if !collection_allowed {
        return None;
    }
    contract
        .get("record_ids")
        .and_then(Value::as_array)
        .is_some_and(|record_ids| {
            record_ids
                .iter()
                .filter_map(Value::as_str)
                .any(|candidate| candidate == record_id)
        })
        .then_some(record_id)
}

fn log_lead_projection_error(command_id: &str, result: anyhow::Result<()>) {
    if let Err(error) = result {
        eprintln!(
            "[business-os] person research `{command_id}` lead lifecycle projection failed: {error:#}"
        );
    }
}

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64
}

struct ActiveResearchCommandGuard {
    key: ActiveResearchCommandKey,
}

impl ActiveResearchCommandGuard {
    fn claim(root: &Path, command_id: &str) -> Option<Self> {
        let key = ActiveResearchCommandKey {
            root: std::fs::canonicalize(root).unwrap_or_else(|_| root.to_path_buf()),
            command_id: command_id.to_string(),
        };
        let mut active = active_commands();
        if active.insert(key.clone()) {
            Some(Self { key })
        } else {
            None
        }
    }
}

impl Drop for ActiveResearchCommandGuard {
    fn drop(&mut self) {
        active_commands().remove(&self.key);
    }
}

fn active_commands() -> std::sync::MutexGuard<'static, HashSet<ActiveResearchCommandKey>> {
    ACTIVE_RESEARCH_COMMANDS
        .get_or_init(|| Mutex::new(HashSet::new()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn execute(root: &Path, payload: &Value) -> anyhow::Result<Value> {
    let request: PersonResearchCommandRequest = serde_json::from_value(payload.clone())
        .context("invalid web_stack.person_research payload")?;
    let company = request.company.trim();
    anyhow::ensure!(!company.is_empty(), "company is required");
    let country = Country::from_iso(&request.country).with_context(|| {
        format!(
            "unsupported country `{}`; expected DE, AT or CH",
            request.country
        )
    })?;
    let mode = ResearchMode::from_str(&request.mode).with_context(|| {
        format!(
            "unsupported research mode `{}`; expected new_record, update_firm, update_person, update_inventory_general or have_data",
            request.mode
        )
    })?;
    let fields = request
        .fields
        .iter()
        .map(|field| {
            FieldKey::from_str(field)
                .with_context(|| format!("unsupported person-research field `{field}`"))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    let workspace =
        root.join("runtime")
            .join("research")
            .join("person")
            .join(safe_workspace_segment(
                payload
                    .get("command_id")
                    .and_then(Value::as_str)
                    .unwrap_or(company),
            ));
    let auto_browser_capture = request.auto_browser_capture;
    let research_request = PersonResearchRequest {
        company: company.to_string(),
        country,
        mode,
        fields,
        include_private: request.include_private,
        workspace: Some(workspace),
        persist_workspace: true,
    };
    let mut result = ctox_web_stack::run_ctox_person_research_tool(root, &research_request)?;
    if auto_browser_capture {
        let capture_tasks =
            crate::service::business_os::authenticated_person_research_capture_tasks(&result);
        let mut capture_runs = Vec::new();
        let mut remaining_tasks = Vec::new();
        for task in capture_tasks {
            let Some(source_id) = task
                .get("source_id")
                .and_then(Value::as_str)
                .map(str::to_string)
            else {
                remaining_tasks.push(task);
                continue;
            };
            let capture_args = vec![
                "source-capture".to_string(),
                "--source-id".to_string(),
                source_id.clone(),
                "--company".to_string(),
                company.to_string(),
                "--country".to_string(),
                country.as_iso().to_string(),
                "--task-id".to_string(),
                payload
                    .get("command_id")
                    .and_then(Value::as_str)
                    .unwrap_or(company)
                    .to_string(),
                "--timeout-ms".to_string(),
                "180000".to_string(),
            ];
            match crate::service::business_os::run_business_os_web_stack_source_capture(
                root,
                &capture_args,
            ) {
                Ok(capture) => {
                    let summary =
                        crate::service::business_os::merge_authenticated_person_research_capture(
                            &mut result,
                            &capture,
                        )?;
                    if !summary.get("ok").and_then(Value::as_bool).unwrap_or(false) {
                        remaining_tasks.push(task);
                    }
                    capture_runs.push(summary);
                }
                Err(error) => {
                    remaining_tasks.push(task);
                    capture_runs.push(serde_json::json!({
                        "ok": false,
                        "source_id": source_id,
                        "status": "failed",
                        "error": error.to_string(),
                        "stream": "rxdb",
                        "secret_value_in_payload": false,
                    }));
                }
            }
        }
        result["browser_assist_tasks"] = Value::Array(remaining_tasks);
        result["authenticated_source_capture_runs"] = Value::Array(capture_runs);
        crate::service::business_os::repersist_augmented_person_research(
            &research_request,
            &mut result,
        );
    }
    let populated = result
        .get("fields")
        .and_then(Value::as_object)
        .map(|fields| {
            fields
                .values()
                .filter(|field| field.get("value").is_some_and(|value| !value.is_null()))
                .count()
        })
        .unwrap_or(0);
    let requested = result
        .get("requested_fields")
        .and_then(Value::as_array)
        .map(Vec::len)
        .unwrap_or(0);
    let browser_assists = result
        .get("browser_assist_tasks")
        .and_then(Value::as_array)
        .map(Vec::len)
        .unwrap_or(0);
    if let Some(object) = result.as_object_mut() {
        object.insert(
            "summary".to_string(),
            Value::String(if browser_assists > 0 {
                format!(
                    "Recherche für {company} abgeschlossen: {populated} von {requested} Feldern gefunden. {browser_assists} Quelle(n) benötigen eine Browser-Autorisierung."
                )
            } else {
                format!(
                    "Recherche für {company} abgeschlossen: {populated} von {requested} Feldern gefunden."
                )
            }),
        );
    }
    Ok(result)
}

fn safe_workspace_segment(value: &str) -> String {
    let segment = value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_') {
                ch
            } else {
                '-'
            }
        })
        .collect::<String>();
    let trimmed = segment.trim_matches('-');
    if trimmed.is_empty() {
        "research".to_string()
    } else {
        trimmed.chars().take(120).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn active_worker_deduplication_is_scoped_to_the_ctox_root() {
        let first = tempfile::tempdir().unwrap();
        let second = tempfile::tempdir().unwrap();
        let command_id = format!("cmd-root-scope-{}", std::process::id());

        let first_guard = ActiveResearchCommandGuard::claim(first.path(), &command_id).unwrap();
        assert!(ActiveResearchCommandGuard::claim(first.path(), &command_id).is_none());
        let second_guard = ActiveResearchCommandGuard::claim(second.path(), &command_id).unwrap();

        drop(first_guard);
        assert!(ActiveResearchCommandGuard::claim(first.path(), &command_id).is_some());
        drop(second_guard);
    }

    #[test]
    fn active_worker_claim_is_released_during_unwind() {
        let root = tempfile::tempdir().unwrap();
        let command_id = format!("cmd-unwind-{}", std::process::id());
        let guard = ActiveResearchCommandGuard::claim(root.path(), &command_id).unwrap();

        let _ = panic::catch_unwind(AssertUnwindSafe(move || {
            let _guard = guard;
            panic!("worker panic");
        }));

        assert!(ActiveResearchCommandGuard::claim(root.path(), &command_id).is_some());
    }

    #[test]
    fn recovery_selects_only_nonterminal_person_research_commands() -> anyhow::Result<()> {
        let temp = tempfile::tempdir()?;
        let root = temp.path();
        let conn = store::open_store(root)?;
        for (id, status) in [
            ("accepted-research", "accepted"),
            ("running-research", "running"),
            ("completed-research", "completed"),
        ] {
            conn.execute(
                "INSERT INTO business_commands
                    (command_id, module, command_type, record_id, status, payload_json, client_context_json, observed_at_ms)
                 VALUES (?1, 'research', 'web_stack.person_research', 'company', ?2, ?3, '{}', 1)",
                rusqlite::params![
                    id,
                    status,
                    serde_json::json!({
                        "company": "Example GmbH",
                        "country": "DE",
                        "mode": "have_data"
                    })
                    .to_string(),
                ],
            )?;
        }
        drop(conn);

        let commands = store::recoverable_person_research_commands(root)?;
        let ids = commands
            .iter()
            .filter_map(|recoverable| recoverable.command.id.as_deref())
            .collect::<HashSet<_>>();
        assert_eq!(
            ids,
            HashSet::from(["accepted-research", "running-research"])
        );
        assert_eq!(commands[0].status, "accepted");
        assert_eq!(commands[1].status, "running");
        Ok(())
    }

    #[test]
    fn browser_capture_requires_explicit_typed_request_flag() {
        let disabled: PersonResearchCommandRequest = serde_json::from_value(serde_json::json!({
            "company": "Example AG",
            "country": "DE",
            "mode": "new_record"
        }))
        .unwrap();
        assert!(!disabled.auto_browser_capture);

        let enabled: PersonResearchCommandRequest = serde_json::from_value(serde_json::json!({
            "company": "Example AG",
            "country": "DE",
            "mode": "new_record",
            "auto_browser_capture": true
        }))
        .unwrap();
        assert!(enabled.auto_browser_capture);
    }

    #[test]
    fn workspace_segment_is_bounded_and_safe() {
        let segment = safe_workspace_segment("cmd /Example Industrial:2026");
        assert_eq!(segment, "cmd--Example-Industrial-2026");
        assert!(segment
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_')));
    }

    #[test]
    fn thesen_outbound_lifecycle_projection_requires_bounded_writeback_contract() {
        let command = BusinessCommand {
            id: Some("cmd-research".to_string()),
            module: "thesen-outbound".to_string(),
            command_type: "web_stack.person_research".to_string(),
            record_id: Some("lead-1".to_string()),
            payload: serde_json::json!({
                "writeback_contract": {
                    "collection": "thesen_outbound_leads",
                    "allowed_collections": ["thesen_outbound_leads"],
                    "record_ids": ["lead-1"]
                }
            }),
            client_context: Value::Null,
            origin: store::CommandOrigin::TrustedLocal,
        };
        assert_eq!(
            thesen_outbound_writeback_record_id(&command),
            Some("lead-1")
        );

        let mut wrong_module = command.clone();
        wrong_module.module = "research".to_string();
        assert_eq!(thesen_outbound_writeback_record_id(&wrong_module), None);

        let mut wrong_record = command.clone();
        wrong_record.record_id = Some("lead-2".to_string());
        assert_eq!(thesen_outbound_writeback_record_id(&wrong_record), None);

        let mut wrong_collection = command;
        wrong_collection.payload["writeback_contract"]["collection"] =
            Value::String("sellify_companies".to_string());
        wrong_collection.payload["writeback_contract"]["allowed_collections"] =
            serde_json::json!(["sellify_companies"]);
        assert_eq!(thesen_outbound_writeback_record_id(&wrong_collection), None);
    }

    #[test]
    fn successful_thesen_outbound_retry_clears_stale_research_error() {
        let failed = thesen_outbound_lead_state_patch(
            "cmd-failed",
            "failed",
            Some("prior attempt failed"),
            1_000,
        );
        assert_eq!(failed["research_error"], "prior attempt failed");

        let completed =
            thesen_outbound_lead_state_patch("cmd-completed", "needs_review", None, 2_000);
        assert_eq!(completed["research_error"], Value::Null);
        assert_eq!(completed["native_research_terminal_status"], "needs_review");
        assert_eq!(completed["research_finished_at_ms"], 2_000);

        let document = thesen_outbound_lead_state_document(
            "lead-1",
            "cmd-completed",
            "needs_review",
            None,
            2_000,
        );
        assert_eq!(document["research_error"], Value::Null);
        assert_eq!(document["research_updated_at_ms"], 2_000);
        assert_eq!(document["payload"]["research_error"], Value::Null);
    }

    #[test]
    fn terminal_research_outcome_projects_contact_data_and_evidence_without_browser() {
        let existing = serde_json::json!({
            "id": "lead-1",
            "data": {"legacy": "kept"},
            "contacts": [],
            "selected_contact_ids": [],
            "evidence": [],
            "research_error": "old failure"
        });
        let outcome = serde_json::json!({
            "tool": "ctox_person_research",
            "fields": {
                "firma_domain": {
                    "value": "example.test",
                    "candidates": [
                        {"value": "example.test", "source_id": "source-a", "source_url": "https://a.test/company"},
                        {"value": "example.test", "source_id": "source-b", "source_url": "https://b.test/company"}
                    ]
                },
                "person_vorname": {
                    "value": "Ada",
                    "candidates": [{"value": "Ada", "source_id": "xing.com", "source_url": "https://www.xing.com/profile/Ada_Lovelace"}]
                },
                "person_nachname": {
                    "value": "Lovelace",
                    "candidates": [{"value": "Lovelace", "source_id": "xing.com", "source_url": "https://www.xing.com/profile/Ada_Lovelace"}]
                },
                "person_xing": {
                    "value": "https://www.xing.com/profile/Ada_Lovelace",
                    "candidates": [{"value": "https://www.xing.com/profile/Ada_Lovelace", "source_id": "xing.com", "source_url": "https://www.xing.com/profile/Ada_Lovelace"}]
                }
            },
            "browser_assist_tasks": []
        });

        let patch = thesen_outbound_research_outcome_patch(&existing, &outcome, 2_000);

        assert_eq!(patch["data"]["legacy"], "kept");
        assert_eq!(patch["data"]["firma_domain"], "example.test");
        assert_eq!(patch["contacts"][0]["name"], "Ada Lovelace");
        assert_eq!(
            patch["contacts"][0]["person_xing"],
            "https://www.xing.com/profile/Ada_Lovelace"
        );
        assert!(patch["contacts"][0]["id"]
            .as_str()
            .is_some_and(|id| id.starts_with("contact_")));
        assert_eq!(patch["research_status"], "needs_review");
        assert_eq!(patch["research_error"], Value::Null);
        assert_eq!(patch["payload"]["research_tool"], "ctox_person_research");
        assert_eq!(
            patch["payload"]["verified_field_keys"],
            serde_json::json!(["firma_domain"])
        );
    }

    #[test]
    fn terminal_research_outcome_projects_every_profile_bound_person_record() {
        let existing = serde_json::json!({
            "id": "lead-multi",
            "data": {},
            "contacts": [{"id": "manual-contact", "name": "Manual Contact", "email": "manual@example.test"}],
            "selected_contact_ids": ["manual-contact"],
            "evidence": []
        });
        let outcome = serde_json::json!({
            "tool": "ctox_person_research",
            "fields": {
                "person_vorname": {"value": "Ada", "candidates": []},
                "person_nachname": {"value": "Lovelace", "candidates": []},
                "person_xing": {"value": "https://www.xing.com/profile/Ada_Lovelace", "candidates": []}
            },
            "person_records": [
                {
                    "person_vorname": "Ada",
                    "person_nachname": "Lovelace",
                    "person_funktion": "Geschäftsführung",
                    "person_xing": "https://www.xing.com/profile/Ada_Lovelace",
                    "source_id": "xing.com",
                    "source_url": "https://www.xing.com/profile/Ada_Lovelace"
                },
                {
                    "person_vorname": "Grace",
                    "person_nachname": "Hopper",
                    "person_funktion": "Leitung Vertrieb",
                    "person_xing": "https://www.xing.com/profile/Grace_Hopper",
                    "source_id": "xing.com",
                    "source_url": "https://www.xing.com/profile/Grace_Hopper"
                }
            ]
        });

        let patch = thesen_outbound_research_outcome_patch(&existing, &outcome, 3_000);

        assert_eq!(patch["contacts"].as_array().map(Vec::len), Some(3));
        assert_eq!(patch["contacts"][0]["id"], "manual-contact");
        assert_eq!(patch["contacts"][1]["name"], "Ada Lovelace");
        assert_eq!(patch["contacts"][2]["name"], "Grace Hopper");
        assert_eq!(patch["contacts"][2]["role"], "Leitung Vertrieb");
        assert_eq!(
            patch["selected_contact_ids"],
            serde_json::json!(["manual-contact"])
        );
    }
}
