// Origin: CTOX
// License: AGPL-3.0-only

use anyhow::Context;
use rusqlite::{params, OptionalExtension};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use super::person_research_command::{
    independent_research_evidence_count, merge_json_object_values,
    outbound_lead_generation_research_outcome_patch, parse_requested_fields,
    ActiveResearchCommandGuard,
};
use super::store::{self, BusinessCommand};
use crate::mission::channels;

const LEAD_COLLECTION: &str = "outbound_lead_generation_leads";
const GAP_METADATA_KEY: &str = "person_research_gap_closure";
const TERMINAL_FIELD_STATUSES: &[&str] =
    &["verified", "no_match", "unsupported", "action_required"];

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ResearchWritebackRequest {
    record_id: String,
    module: String,
    research_command_id: String,
    gap_task_id: String,
    field_status: BTreeMap<String, FieldStatus>,
    result: ResearchWritebackResult,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct FieldStatus {
    status: String,
    #[serde(default)]
    value: Value,
    #[serde(default)]
    sources: Vec<FieldSource>,
    #[serde(default)]
    attempts: Vec<FieldAttempt>,
    #[serde(default)]
    reason: String,
    #[serde(flatten)]
    extra: BTreeMap<String, Value>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct FieldSource {
    source_id: String,
    #[serde(default)]
    url: String,
    #[serde(default)]
    quote: String,
    #[serde(default)]
    person_key: Option<String>,
    #[serde(default)]
    requires_credential: bool,
    #[serde(default)]
    task_id: String,
    #[serde(default)]
    command_id: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct FieldAttempt {
    kind: String,
    query_or_url: String,
    result: Value,
    artifact_path: String,
    at: Value,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ResearchWritebackResult {
    fields: Value,
    #[serde(default)]
    person_records: Vec<Value>,
    #[serde(default)]
    evidence: Vec<Value>,
}

pub(super) fn build_gap_closure_prompt(contract: &Value) -> anyhow::Result<String> {
    let company = required_string(contract, "company")?;
    let record_id = required_string(contract, "record_id")?;
    let research_command_id = required_string(contract, "research_command_id")?;
    let module = required_string(contract, "module")?;
    let open_fields = contract
        .get("open_fields")
        .and_then(Value::as_array)
        .context("gap closure contract open_fields array is required")?;
    let instructions = contract
        .get("research_instructions")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let priorities = serde_json::to_string_pretty(
        contract
            .get("person_priorities")
            .unwrap_or(&Value::Array(Vec::new())),
    )?;
    let known_people = serde_json::to_string_pretty(
        contract
            .get("known_person_records")
            .unwrap_or(&Value::Array(Vec::new())),
    )?;
    let attempted = serde_json::to_string_pretty(
        contract
            .get("attempted_sources")
            .unwrap_or(&Value::Array(Vec::new())),
    )?;
    let terminal_fields = serde_json::to_string_pretty(
        contract
            .get("terminal_fields")
            .unwrap_or(&Value::Object(Map::new())),
    )?;
    let writeback_contract = serde_json::to_string_pretty(
        contract
            .get("writeback_contract")
            .context("gap closure writeback_contract is required")?,
    )?;
    let mut field_lines = Vec::new();
    for field in open_fields {
        let key = field
            .as_str()
            .context("gap closure open field names must be strings")?;
        field_lines.push(format!("- `{key}`: {}", field_definition(key)));
    }

    Ok(format!(
        r#"# Lückenschluss Personen-/Firmenrecherche

Firma: {company}
Lead/Record: {record_id}
Modul: {module}
Phase-A-Kommando: {research_command_id}

## Offene Felder mit Definition
{fields}

## Bereits versuchte Quellen aus Phase A
{attempted}

## Bereits terminale Phase-A-Felder (unverändert mitführen)
{terminal_fields}

## Owner-research_instructions (wörtlich, unverändert)
--- BEGIN OWNER INSTRUCTIONS ---
{instructions}
--- END OWNER INSTRUCTIONS ---

## Personen-Prioritäten
{priorities}

## Bereits bekannte Sellify-Personen
{known_people}

## Verbindlicher Feldvertrag für Phase B
- Bearbeite jedes offene Feld einzeln mit `ctox web search` und `ctox web read`; `ctox web browser-capture` ist optional.
- Vor `no_match` sind mindestens 1 dokumentierte Websuche und mindestens 2 dokumentierte Seitenlektüren (`web_read` oder `browser_capture`) Pflicht.
- Schreibe JEDEN Versuch als JSON-Datei unter `gap_closure/attempts/<feld>/<n>.json` in diesem Workspace. Jeder Versuch enthält kind, query_or_url, result, artifact_path und at.
- Halte den fortlaufenden Sammelstand nach jedem Versuch in `gap_closure/field_status.json` fest. Diese Datei ist der Checkpoint für einen Folgeturn.
- Terminale Feldstatus sind ausschließlich `verified`, `no_match`, `unsupported` und `action_required`.
- `verified` verlangt einen Wert und mindestens zwei unabhängige Belege von verschiedenen Hosts. Jeder Wert braucht source_id, URL und wörtlichen Belegtext.
- Personenbezogene Ergebnisse und Belege tragen einen stabilen `person_key`.
- Schreibe in `result.fields` nur strukturierte Feldobjekte, keine freien Texte.
- `action_required` ist ausschließlich für Login/Freigabe zulässig und verweist auf einen Auth-Assist (source_id plus Task-/Command-ID) oder eine Quelle mit `requires_credential=true`.
- Abschluss erfolgt AUSSCHLIESSLICH mit `ctox business-os commands dispatch` und dem typisierten Befehl `outbound.lead.research_writeback`.
- Der Task darf sich nicht als fertig melden und darf nicht beendet werden, bevor dieser Dispatch vom Daemon angenommen wurde.

## Writeback-Vertrag
{writeback_contract}

Sende beim Abschluss exakt die Payload-Felder `record_id`, `module`, `research_command_id`, `gap_task_id`, `field_status` und `result`. `field_status` muss ALLE angeforderten Felder abdecken; bereits terminale Phase-A-Felder werden unverändert mitgeführt."#,
        fields = field_lines.join("\n")
    ))
}

pub(super) fn enqueue_gap_closure_if_needed(
    root: &Path,
    command: &BusinessCommand,
    phase_a_result: &mut Value,
) -> anyhow::Result<Option<channels::QueueTaskView>> {
    let command_id = command
        .id
        .as_deref()
        .context("person-research command id is required for gap closure")?;
    let Some(record_id) =
        super::person_research_command::outbound_lead_generation_writeback_record_id(command)
    else {
        return Ok(None);
    };
    let requested_fields = canonical_requested_fields(command, phase_a_result)?;
    let phase_a_evidence = phase_a_projection_evidence(record_id, phase_a_result);
    let mut terminal_fields = Map::new();
    let mut open_fields = Vec::new();
    for field in &requested_fields {
        let field_result = phase_a_result.pointer(&format!("/fields/{field}"));
        let populated = field_result
            .and_then(|entry| entry.get("value"))
            .is_some_and(research_value_is_populated);
        let sources = phase_a_verified_sources(&phase_a_evidence, field);
        let independent = independent_research_evidence_count(&sources, field);
        let person_keys = sources
            .iter()
            .filter_map(|source| source.get("person_key").and_then(Value::as_str))
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .collect::<BTreeSet<_>>();
        let person_evidence_complete = !field.starts_with("person_")
            || (person_keys.len() == 1
                && sources.iter().all(|source| {
                    source
                        .get("person_key")
                        .and_then(Value::as_str)
                        .is_some_and(|value| !value.trim().is_empty())
                }));
        if populated && independent >= 2 && person_evidence_complete {
            terminal_fields.insert(
                field.clone(),
                serde_json::json!({
                    "status": "verified",
                    "value": field_result.and_then(|entry| entry.get("value")).cloned().unwrap_or(Value::Null),
                    "sources": sources,
                    "attempts": [],
                    "independent_sources": independent,
                }),
            );
        } else {
            open_fields.push(Value::String(field.clone()));
        }
    }
    if open_fields.is_empty() {
        phase_a_result["gap_closure"] = serde_json::json!({
            "required": false,
            "requested_fields": requested_fields,
            "terminal_fields": terminal_fields,
        });
        return Ok(None);
    }

    let company = command
        .payload
        .get("company")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .context("gap closure requires company")?;
    let workspace = phase_a_workspace(root, command)?;
    std::fs::create_dir_all(&workspace)
        .with_context(|| format!("create gap closure workspace {}", workspace.display()))?;
    let attempted_sources = phase_a_attempted_sources(phase_a_result);
    let writeback_contract = serde_json::json!({
        "command_type": "outbound.lead.research_writeback",
        "record_id": record_id,
        "module": "outbound-lead-generation",
        "research_command_id": command_id,
        "gap_task_id": "<current queue task id>",
        "field_status": {
            "statuses": TERMINAL_FIELD_STATUSES,
            "verified": "value plus at least two sources on different hosts",
            "no_match": "at least one web_search plus two web_read/browser_capture artifacts",
            "unsupported": "field cannot be researched by the available contract",
            "action_required": "login or approval with auth-assist reference"
        },
        "result": {"fields": {}, "person_records": [], "evidence": []}
    });
    let contract = serde_json::json!({
        "lead_id": record_id,
        "record_id": record_id,
        "company": company,
        "module": command.module,
        "research_command_id": command_id,
        "requested_fields": requested_fields,
        "open_fields": open_fields,
        "terminal_fields": terminal_fields,
        "attempted_sources": attempted_sources,
        "research_instructions": command.payload.get("research_instructions").cloned().unwrap_or_else(|| Value::String(String::new())),
        "known_person_records": command.payload.get("known_person_records").cloned().unwrap_or_else(|| Value::Array(Vec::new())),
        "person_priorities": command.payload.get("person_priorities").cloned().unwrap_or_else(|| Value::Array(Vec::new())),
        "writeback_contract": writeback_contract,
    });
    let prompt = build_gap_closure_prompt(&contract)?;
    let idempotency_key = format!("gap:{command_id}");
    let mut task = if let Some(existing) = load_gap_task_by_idempotency_key(root, &idempotency_key)?
    {
        existing
    } else {
        channels::create_queue_task(
            root,
            channels::QueueTaskCreateRequest {
                title: format!("Lückenschluss: {company}"),
                prompt,
                thread_key: format!("person-research-gap/{record_id}"),
                workspace_root: Some(workspace.to_string_lossy().into_owned()),
                priority: "high".to_string(),
                suggested_skill: None,
                parent_message_key: None,
                extra_metadata: Some(serde_json::json!({
                    "idempotency_key": idempotency_key,
                    GAP_METADATA_KEY: contract,
                })),
            },
        )?
    };
    // The gap task inherits the research command's human owner through the
    // `business_os_command_id` metadatum: owner resolution for
    // `--task-id <gap task>` and the harness command session follow it to the
    // command's verified actor. It is deliberately NOT a
    // `business_command_task_links` row — that link marks a task as the
    // command's own execution, and for an already-completed research command
    // the queue would settle the gap task to `handled` on lease as an
    // "orphaned lease of a terminal command" without running it.
    channels::set_queue_task_metadata_value(
        root,
        &task.message_key,
        "business_os_command_id",
        Value::String(command_id.to_string()),
    )?;
    let mut final_contract = contract;
    final_contract["gap_task_id"] = Value::String(task.message_key.clone());
    final_contract["writeback_contract"]["gap_task_id"] = Value::String(task.message_key.clone());
    channels::set_queue_task_metadata_value(
        root,
        &task.message_key,
        GAP_METADATA_KEY,
        final_contract.clone(),
    )?;
    task = channels::update_queue_task(
        root,
        channels::QueueTaskUpdateRequest {
            message_key: task.message_key.clone(),
            title: Some(format!("Lückenschluss: {company}")),
            prompt: Some(build_gap_closure_prompt(&final_contract)?),
            thread_key: Some(format!("person-research-gap/{record_id}")),
            workspace_root: Some(workspace.to_string_lossy().into_owned()),
            priority: Some("high".to_string()),
            ..Default::default()
        },
    )?;
    phase_a_result["gap_closure"] = serde_json::json!({
        "required": true,
        "owner_command_id": command_id,
        "task_id": task.message_key.clone(),
        "workspace_root": task.workspace_root.clone(),
        "requested_fields": task.metadata.pointer(&format!("/{GAP_METADATA_KEY}/requested_fields")).cloned().unwrap_or(Value::Null),
        "open_fields": task.metadata.pointer(&format!("/{GAP_METADATA_KEY}/open_fields")).cloned().unwrap_or(Value::Null),
        "terminal_fields": task.metadata.pointer(&format!("/{GAP_METADATA_KEY}/terminal_fields")).cloned().unwrap_or(Value::Null),
    });
    Ok(Some(task))
}

pub(super) fn handle_research_writeback(
    root: &Path,
    command: &BusinessCommand,
) -> anyhow::Result<Value> {
    let request: ResearchWritebackRequest = serde_json::from_value(command.payload.clone())
        .context("invalid outbound.lead.research_writeback payload")?;
    anyhow::ensure!(
        request.module == "outbound-lead-generation" && command.module == request.module,
        "research writeback module must be outbound-lead-generation"
    );
    anyhow::ensure!(
        command.record_id.as_deref() == Some(request.record_id.as_str()),
        "research writeback record_id does not match command record_id"
    );
    let _guard = ActiveResearchCommandGuard::claim(root, &request.record_id)
        .context("another person research or research writeback is active for this record_id")?;

    validate_original_research_command(root, &request)?;
    let mut lead = store::load_rxdb_collection_record(root, LEAD_COLLECTION, &request.record_id)?
        .context("research writeback lead record does not exist")?;
    anyhow::ensure!(
        lead.get("research_phase").and_then(Value::as_str) == Some("gap_closure"),
        "lead is not in research_phase gap_closure"
    );
    anyhow::ensure!(
        lead.get("gap_task_id").and_then(Value::as_str) == Some(request.gap_task_id.as_str()),
        "research writeback gap_task_id does not match lead"
    );
    let task = channels::load_queue_task(root, &request.gap_task_id)?
        .context("research writeback gap task does not exist")?;
    let contract = task
        .metadata
        .get(GAP_METADATA_KEY)
        .context("gap task is missing person_research_gap_closure metadata")?;
    validate_task_correlation(contract, &request)?;
    let requested_fields = contract
        .get("requested_fields")
        .and_then(Value::as_array)
        .context("gap task requested_fields array is missing")?
        .iter()
        .map(|value| {
            value
                .as_str()
                .map(str::to_string)
                .context("gap task requested_fields must contain strings")
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    validate_field_status_keys(&requested_fields, &request.field_status)?;
    validate_result_shape(&request.result, &requested_fields)?;
    validate_result_field_status_consistency(&request.result, &request.field_status)?;
    let workspace = task_workspace(root, &task, contract)?;
    for (field, status) in &request.field_status {
        validate_terminal_field(field, status, &workspace, contract)?;
    }

    let mut projection_result = serde_json::json!({
        "fields": request.result.fields,
        "person_records": request.result.person_records,
        "evidence": request.result.evidence,
        "known_person_records": contract.get("known_person_records").cloned().unwrap_or_else(|| Value::Array(Vec::new())),
    });
    add_field_status_evidence(&mut projection_result, &request.field_status)?;
    let now = super::person_research_command::now_ms();
    let patch = outbound_lead_generation_research_outcome_patch(&lead, &projection_result, now);
    merge_json_object_values(&mut lead, &patch);
    lead["field_status"] = serde_json::to_value(&request.field_status)?;
    lead["gap_task_id"] = Value::String(request.gap_task_id.clone());
    lead["research_phase"] = Value::Null;
    let needs_review = request
        .field_status
        .values()
        .any(|field| matches!(field.status.as_str(), "no_match" | "action_required"));
    let terminal_lead_status = if needs_review {
        "needs_review"
    } else {
        "completed"
    };
    lead["research_status"] = Value::String(terminal_lead_status.to_string());
    lead["payload"]["native_research_terminal_status"] =
        Value::String(terminal_lead_status.to_string());
    lead["payload"]["research_finished_at_ms"] = Value::Number(now.into());
    lead["research_error"] = Value::Null;
    lead["research_updated_at_ms"] = Value::Number(now.into());
    store::upsert_rxdb_collection_record(root, LEAD_COLLECTION, &request.record_id, now, lead)?;

    Ok(serde_json::json!({
        "ok": true,
        "record_id": request.record_id,
        "research_command_id": request.research_command_id,
        "gap_task_id": request.gap_task_id,
        "research_status": if needs_review { "needs_review" } else { "completed" },
        "field_status": request.field_status,
    }))
}

pub(super) fn cancel_open_gap_task_for_new_research(
    root: &Path,
    command: &BusinessCommand,
) -> anyhow::Result<bool> {
    let Some(record_id) = command.record_id.as_deref() else {
        return Ok(false);
    };
    let Some(lead) = store::load_rxdb_collection_record(root, LEAD_COLLECTION, record_id)? else {
        return Ok(false);
    };
    if lead.get("research_phase").and_then(Value::as_str) != Some("gap_closure") {
        return Ok(false);
    }
    let Some(task_id) = lead
        .get("gap_task_id")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
    else {
        return Ok(false);
    };
    let Some(task) = channels::load_queue_task(root, task_id)? else {
        return Ok(false);
    };
    if channels::route_status_is_terminal(&task.route_status) {
        return Ok(false);
    }
    channels::update_queue_task(
        root,
        channels::QueueTaskUpdateRequest {
            message_key: task.message_key,
            route_status: Some("cancelled".to_string()),
            status_note: Some(format!(
                "Cancelled because a new person research command `{}` superseded this gap closure.",
                command.id.as_deref().unwrap_or_default()
            )),
            ..Default::default()
        },
    )?;
    Ok(true)
}

fn load_gap_task_by_idempotency_key(
    root: &Path,
    idempotency_key: &str,
) -> anyhow::Result<Option<channels::QueueTaskView>> {
    let task_count = channels::count_queue_tasks(root, &[])?;
    if task_count == 0 {
        return Ok(None);
    }
    Ok(channels::list_queue_tasks(root, &[], task_count)?
        .into_iter()
        .find(|task| {
            task.metadata.get("idempotency_key").and_then(Value::as_str) == Some(idempotency_key)
        }))
}

fn validate_original_research_command(
    root: &Path,
    request: &ResearchWritebackRequest,
) -> anyhow::Result<()> {
    let conn = store::open_store(root)?;
    let original = conn
        .query_row(
            "SELECT command_type, module, record_id FROM business_commands WHERE command_id = ?1",
            params![request.research_command_id],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                ))
            },
        )
        .optional()?;
    let (command_type, module, record_id) = original.with_context(|| {
        format!(
            "research_command_id `{}` does not exist",
            request.research_command_id
        )
    })?;
    anyhow::ensure!(
        command_type == "web_stack.person_research",
        "research_command_id must reference web_stack.person_research"
    );
    anyhow::ensure!(
        module == request.module && record_id == request.record_id,
        "research_command_id does not belong to the same module and record_id"
    );
    Ok(())
}

fn validate_task_correlation(
    contract: &Value,
    request: &ResearchWritebackRequest,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        contract.get("record_id").and_then(Value::as_str) == Some(request.record_id.as_str()),
        "gap task record_id does not match writeback"
    );
    anyhow::ensure!(
        contract.get("research_command_id").and_then(Value::as_str)
            == Some(request.research_command_id.as_str()),
        "gap task research_command_id does not match writeback"
    );
    anyhow::ensure!(
        contract.get("module").and_then(Value::as_str) == Some(request.module.as_str()),
        "gap task module does not match writeback"
    );
    anyhow::ensure!(
        contract.get("gap_task_id").and_then(Value::as_str) == Some(request.gap_task_id.as_str()),
        "gap task id does not match writeback"
    );
    Ok(())
}

fn validate_field_status_keys(
    requested_fields: &[String],
    field_status: &BTreeMap<String, FieldStatus>,
) -> anyhow::Result<()> {
    parse_requested_fields(requested_fields)?;
    let submitted = field_status.keys().cloned().collect::<Vec<_>>();
    parse_requested_fields(&submitted)?;
    let requested = requested_fields.iter().cloned().collect::<BTreeSet<_>>();
    let submitted = submitted.into_iter().collect::<BTreeSet<_>>();
    let missing = requested
        .difference(&submitted)
        .cloned()
        .collect::<Vec<_>>();
    let unexpected = submitted
        .difference(&requested)
        .cloned()
        .collect::<Vec<_>>();
    anyhow::ensure!(
        missing.is_empty() && unexpected.is_empty(),
        "field_status must cover all requested_fields exactly; missing={missing:?}, unexpected={unexpected:?}"
    );
    Ok(())
}

fn validate_terminal_field(
    field: &str,
    status: &FieldStatus,
    workspace: &Path,
    contract: &Value,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        TERMINAL_FIELD_STATUSES.contains(&status.status.as_str()),
        "field `{field}` has non-terminal or unsupported status `{}`",
        status.status
    );
    for attempt in &status.attempts {
        validate_attempt(field, attempt, workspace)?;
    }
    match status.status.as_str() {
        "verified" => {
            anyhow::ensure!(
                research_value_is_populated(&status.value),
                "verified field `{field}` requires a populated scalar value"
            );
            let evidence = status
                .sources
                .iter()
                .map(|source| {
                    serde_json::json!({
                        "field_key": field,
                        "source_id": source.source_id,
                        "source_url": source.url,
                        "quote": source.quote,
                        "person_key": source.person_key,
                    })
                })
                .collect::<Vec<_>>();
            for source in &status.sources {
                validate_source(field, source)?;
                if field.starts_with("person_") {
                    anyhow::ensure!(
                        source
                            .person_key
                            .as_deref()
                            .is_some_and(|value| !value.trim().is_empty()),
                        "verified person field `{field}` evidence requires person_key"
                    );
                }
            }
            anyhow::ensure!(
                independent_research_evidence_count(&evidence, field) >= 2,
                "verified field `{field}` requires at least 2 independent sources on different hosts"
            );
        }
        "no_match" => validate_no_match(field, status)?,
        "unsupported" => {},
        "action_required" => anyhow::ensure!(
            action_required_has_auth_reference(status, contract),
            "action_required field `{field}` requires an auth-assist reference or requires_credential source"
        ),
        _ => unreachable!(),
    }
    Ok(())
}

fn validate_source(field: &str, source: &FieldSource) -> anyhow::Result<()> {
    anyhow::ensure!(
        !source.source_id.trim().is_empty(),
        "verified field `{field}` source_id must not be empty"
    );
    anyhow::ensure!(
        !source.quote.trim().is_empty(),
        "verified field `{field}` quote must not be empty"
    );
    let url = url::Url::parse(source.url.trim())
        .with_context(|| format!("verified field `{field}` has invalid source URL"))?;
    anyhow::ensure!(
        matches!(url.scheme(), "http" | "https") && url.host_str().is_some(),
        "verified field `{field}` source URL must be HTTP(S)"
    );
    Ok(())
}

fn validate_no_match(field: &str, status: &FieldStatus) -> anyhow::Result<()> {
    let searches = status
        .attempts
        .iter()
        .filter(|attempt| attempt.kind == "web_search")
        .count();
    let reads = status
        .attempts
        .iter()
        .filter(|attempt| matches!(attempt.kind.as_str(), "web_read" | "browser_capture"))
        .count();
    anyhow::ensure!(
        searches >= 1 && reads >= 2,
        "no_match field `{field}` requires at least 1 web_search and 2 web_read/browser_capture attempts"
    );
    Ok(())
}

fn validate_attempt(field: &str, attempt: &FieldAttempt, workspace: &Path) -> anyhow::Result<()> {
    anyhow::ensure!(
        matches!(
            attempt.kind.as_str(),
            "web_search" | "web_read" | "browser_capture" | "adapter"
        ),
        "field `{field}` has unsupported attempt kind `{}`",
        attempt.kind
    );
    anyhow::ensure!(
        !attempt.query_or_url.trim().is_empty(),
        "field `{field}` attempt query_or_url must not be empty"
    );
    anyhow::ensure!(
        !attempt.at.is_null(),
        "field `{field}` attempt at must not be null"
    );
    validate_artifact_path(field, workspace, &attempt.artifact_path)
}

fn validate_artifact_path(
    field: &str,
    workspace: &Path,
    artifact_path: &str,
) -> anyhow::Result<()> {
    let relative = Path::new(artifact_path.trim());
    let required_prefix = format!("gap_closure/attempts/{field}/");
    anyhow::ensure!(
        !artifact_path.trim().is_empty()
            && !relative.is_absolute()
            && artifact_path.trim().starts_with(&required_prefix),
        "field `{field}` artifact_path must be under `{required_prefix}`"
    );
    let file_name = relative
        .file_name()
        .and_then(|value| value.to_str())
        .context("attempt artifact_path must end in a UTF-8 file name")?;
    let sequence = file_name
        .strip_suffix(".json")
        .and_then(|value| value.parse::<usize>().ok());
    anyhow::ensure!(
        sequence.is_some_and(|value| value > 0),
        "field `{field}` artifact_path must end in a positive numeric `<n>.json`"
    );
    let workspace = std::fs::canonicalize(workspace)
        .with_context(|| format!("canonicalize gap task workspace {}", workspace.display()))?;
    let field_attempts =
        std::fs::canonicalize(workspace.join("gap_closure").join("attempts").join(field))
            .with_context(|| format!("canonicalize attempt directory for field `{field}`"))?;
    anyhow::ensure!(
        field_attempts.starts_with(&workspace),
        "field `{field}` attempt directory escapes the gap task workspace"
    );
    let artifact = std::fs::canonicalize(workspace.join(relative)).with_context(|| {
        format!("field `{field}` artifact_path `{artifact_path}` does not exist")
    })?;
    anyhow::ensure!(
        artifact.parent() == Some(field_attempts.as_path()),
        "field `{field}` artifact_path must remain directly under `{required_prefix}`"
    );
    let metadata = std::fs::metadata(&artifact)?;
    anyhow::ensure!(
        metadata.is_file() && metadata.len() > 0,
        "field `{field}` artifact_path must be an existing non-empty file"
    );
    Ok(())
}

fn action_required_has_auth_reference(status: &FieldStatus, contract: &Value) -> bool {
    if status.sources.iter().any(|source| {
        !source.source_id.trim().is_empty()
            && (source.requires_credential
                || !source.task_id.trim().is_empty()
                || !source.command_id.trim().is_empty())
    }) {
        return true;
    }
    let attempted_sources = contract
        .get("attempted_sources")
        .and_then(Value::as_array)
        .into_iter()
        .flatten();
    if status.sources.iter().any(|source| {
        attempted_sources.clone().any(|attempted| {
            attempted.get("source_id").and_then(Value::as_str) == Some(source.source_id.as_str())
                && attempted
                    .get("requires_credential")
                    .and_then(Value::as_bool)
                    == Some(true)
        })
    }) {
        return true;
    }
    status.attempts.iter().any(|attempt| {
        let source_id = attempt
            .result
            .get("source_id")
            .or_else(|| attempt.result.pointer("/auth_assist/source_id"))
            .and_then(Value::as_str)
            .unwrap_or_default();
        let task_or_command = [
            "/task_id",
            "/command_id",
            "/auth_assist/task_id",
            "/auth_assist/command_id",
        ]
        .into_iter()
        .any(|pointer| {
            attempt
                .result
                .pointer(pointer)
                .and_then(Value::as_str)
                .is_some_and(|value| !value.trim().is_empty())
        });
        !source_id.trim().is_empty() && task_or_command
    })
}

fn validate_result_shape(
    result: &ResearchWritebackResult,
    requested_fields: &[String],
) -> anyhow::Result<()> {
    let fields = result
        .fields
        .as_object()
        .context("research writeback result.fields must be an object")?;
    let keys = fields.keys().cloned().collect::<Vec<_>>();
    parse_requested_fields(&keys)?;
    let requested = requested_fields.iter().collect::<BTreeSet<_>>();
    anyhow::ensure!(
        keys.iter().all(|key| requested.contains(key)),
        "research writeback result.fields contains a field that was not requested"
    );
    for (field, value) in fields {
        anyhow::ensure!(
            value.is_object(),
            "research writeback result.fields.{field} must be a structured object, not free text"
        );
        if let Some(field_value) = value.get("value") {
            anyhow::ensure!(
                field_value.is_null() || research_value_is_populated(field_value),
                "research writeback result.fields.{field}.value must be a populated scalar or null"
            );
            if field.starts_with("person_") && research_value_is_populated(field_value) {
                anyhow::ensure!(
                    value
                        .get("person_key")
                        .and_then(Value::as_str)
                        .is_some_and(|value| !value.trim().is_empty()),
                    "research writeback result.fields.{field} requires person_key"
                );
            }
        }
    }
    for (index, person) in result.person_records.iter().enumerate() {
        anyhow::ensure!(
            person
                .get("person_key")
                .and_then(Value::as_str)
                .is_some_and(|value| !value.trim().is_empty()),
            "research writeback result.person_records[{index}] requires person_key"
        );
    }
    for (index, evidence) in result.evidence.iter().enumerate() {
        let field = evidence
            .get("field_key")
            .or_else(|| evidence.get("field"))
            .and_then(Value::as_str)
            .with_context(|| {
                format!("research writeback result.evidence[{index}] requires field_key")
            })?;
        parse_requested_fields(&[field.to_string()])?;
        anyhow::ensure!(
            requested_fields.iter().any(|requested| requested == field),
            "research writeback result.evidence[{index}] field was not requested"
        );
        anyhow::ensure!(
            evidence
                .get("source_id")
                .and_then(Value::as_str)
                .is_some_and(|value| !value.trim().is_empty()),
            "research writeback result.evidence[{index}] requires source_id"
        );
        let source_url = evidence
            .get("url")
            .or_else(|| evidence.get("source_url"))
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .with_context(|| format!("research writeback result.evidence[{index}] requires URL"))?;
        let source_url = url::Url::parse(source_url).with_context(|| {
            format!("research writeback result.evidence[{index}] has invalid URL")
        })?;
        anyhow::ensure!(
            matches!(source_url.scheme(), "http" | "https") && source_url.host_str().is_some(),
            "research writeback result.evidence[{index}] URL must be HTTP(S)"
        );
        anyhow::ensure!(
            evidence
                .get("quote")
                .and_then(Value::as_str)
                .is_some_and(|value| !value.trim().is_empty()),
            "research writeback result.evidence[{index}] requires quote"
        );
        if field.starts_with("person_") {
            anyhow::ensure!(
                evidence
                    .get("person_key")
                    .and_then(Value::as_str)
                    .is_some_and(|value| !value.trim().is_empty()),
                "research writeback person evidence[{index}] requires person_key"
            );
        }
    }
    Ok(())
}

fn validate_result_field_status_consistency(
    result: &ResearchWritebackResult,
    field_status: &BTreeMap<String, FieldStatus>,
) -> anyhow::Result<()> {
    let fields = result
        .fields
        .as_object()
        .context("research writeback result.fields must be an object")?;
    for (field, status) in field_status {
        let result_entry = fields.get(field);
        let result_value = result_entry.and_then(|entry| entry.get("value"));
        if status.status == "verified" {
            if let Some(result_value) = result_value.filter(|value| !value.is_null()) {
                anyhow::ensure!(
                    result_value == &status.value,
                    "verified result field `{field}` value does not match field_status value"
                );
            }
            if field.starts_with("person_") {
                let person_key = result_entry
                    .and_then(|entry| entry.get("person_key"))
                    .and_then(Value::as_str)
                    .map(str::trim)
                    .filter(|value| !value.is_empty())
                    .with_context(|| {
                        format!("verified person result field `{field}` requires person_key")
                    })?;
                anyhow::ensure!(
                    status
                        .sources
                        .iter()
                        .all(|source| source.person_key.as_deref().map(str::trim)
                            == Some(person_key)),
                    "verified person field `{field}` evidence person_key must match result.fields"
                );
            }
        } else {
            anyhow::ensure!(
                !research_value_is_populated(&status.value),
                "non-verified field `{field}` must not carry a populated field_status value"
            );
            anyhow::ensure!(
                !result_value.is_some_and(research_value_is_populated),
                "non-verified result field `{field}` must not carry a populated value"
            );
        }
    }
    Ok(())
}

fn add_field_status_evidence(
    result: &mut Value,
    field_status: &BTreeMap<String, FieldStatus>,
) -> anyhow::Result<()> {
    let fields = result
        .get_mut("fields")
        .and_then(Value::as_object_mut)
        .context("projection result fields must be an object")?;
    for (field, status) in field_status {
        if status.status != "verified" {
            continue;
        }
        let entry = fields
            .entry(field.clone())
            .or_insert_with(|| serde_json::json!({}));
        anyhow::ensure!(
            entry.is_object(),
            "verified result field `{field}` must be an object"
        );
        if entry.get("value").is_none_or(Value::is_null) {
            entry["value"] = status.value.clone();
        }
        let candidates = status
            .sources
            .iter()
            .map(|source| {
                serde_json::json!({
                    "value": status.value,
                    "source_id": source.source_id,
                    "source_url": source.url,
                    "quote": source.quote,
                    "person_key": source
                        .person_key
                        .as_deref()
                        .map(str::trim)
                        .filter(|value| !value.is_empty())
                        .map(|value| Value::String(value.to_string()))
                        .unwrap_or_else(|| entry.get("person_key").cloned().unwrap_or(Value::Null)),
                    "via": "gap_closure",
                })
            })
            .collect::<Vec<_>>();
        entry["candidates"] = Value::Array(candidates);
    }
    Ok(())
}

fn canonical_requested_fields(
    command: &BusinessCommand,
    result: &Value,
) -> anyhow::Result<Vec<String>> {
    let fields = result
        .get("requested_fields")
        .and_then(Value::as_array)
        .map(|values| {
            values
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect::<Vec<_>>()
        })
        .filter(|values| !values.is_empty())
        .unwrap_or_else(|| {
            command
                .payload
                .get("fields")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect()
        });
    let fields = if fields.is_empty() {
        ctox_web_stack::sources::OUTBOUND_RESEARCH_FIELDS
            .iter()
            .map(|field| field.as_str().to_string())
            .collect()
    } else {
        fields
    };
    parse_requested_fields(&fields)?;
    Ok(fields)
}

fn phase_a_verified_sources(evidence: &[Value], field: &str) -> Vec<Value> {
    evidence
        .iter()
        .filter(|entry| {
            entry
                .get("field_key")
                .or_else(|| entry.get("field"))
                .and_then(Value::as_str)
                == Some(field)
        })
        .filter_map(|entry| {
            let url = entry
                .get("source_url")
                .or_else(|| entry.get("url"))
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|value| !value.is_empty())?;
            let parsed = url::Url::parse(url).ok()?;
            if !matches!(parsed.scheme(), "http" | "https") || parsed.host_str().is_none() {
                return None;
            }
            let source_id = entry
                .get("source_id")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|value| !value.is_empty())?;
            let quote = entry
                .get("quote")
                .or_else(|| entry.get("note"))
                .or_else(|| entry.get("value"))
                .and_then(|value| {
                    value
                        .as_str()
                        .map(str::to_string)
                        .or_else(|| (!value.is_null()).then(|| value.to_string()))
                })
                .filter(|value| !value.trim().is_empty())?;
            Some(serde_json::json!({
                "field_key": field,
                "source_id": source_id,
                "source_url": url,
                "url": url,
                "quote": quote,
                "person_key": entry.get("person_key").cloned().unwrap_or(Value::Null),
            }))
        })
        .collect()
}

fn phase_a_attempted_sources(result: &Value) -> Value {
    let mut attempted = Vec::new();
    for (key, attempt_kind) in [
        ("plan", "adapter"),
        ("search_runs", "web_search"),
        ("read_runs", "web_read"),
        ("scrape_runs", "web_read"),
        ("browser_extract_runs", "browser_capture"),
        ("authenticated_source_capture_runs", "browser_capture"),
    ] {
        for entry in result
            .get(key)
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            let mut entry = entry.clone();
            if let Some(object) = entry.as_object_mut() {
                object
                    .entry("attempt_kind".to_string())
                    .or_insert_with(|| Value::String(attempt_kind.to_string()));
            }
            attempted.push(entry);
        }
    }
    Value::Array(attempted)
}

fn phase_a_projection_evidence(record_id: &str, result: &Value) -> Vec<Value> {
    let patch = outbound_lead_generation_research_outcome_patch(
        &serde_json::json!({"id": record_id, "data": {}, "contacts": [], "evidence": []}),
        result,
        0,
    );
    patch
        .get("evidence")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
}

fn phase_a_workspace(root: &Path, command: &BusinessCommand) -> anyhow::Result<PathBuf> {
    let command_id = command
        .id
        .as_deref()
        .context("person-research command id is required for the Phase-A workspace")?;
    Ok(root.join("runtime").join("research").join("person").join(
        super::person_research_command::safe_workspace_segment(command_id),
    ))
}

fn task_workspace(
    root: &Path,
    task: &channels::QueueTaskView,
    contract: &Value,
) -> anyhow::Result<PathBuf> {
    let research_command_id = required_string(contract, "research_command_id")?;
    let expected = root.join("runtime").join("research").join("person").join(
        super::person_research_command::safe_workspace_segment(research_command_id),
    );
    let actual = task
        .workspace_root
        .as_deref()
        .map(PathBuf::from)
        .context("gap task workspace_root is missing")?;
    anyhow::ensure!(
        actual == expected,
        "gap task workspace_root does not match the Phase-A workspace"
    );
    Ok(expected)
}

fn research_value_is_populated(value: &Value) -> bool {
    match value {
        Value::String(value) => !value.trim().is_empty(),
        Value::Number(_) | Value::Bool(_) => true,
        _ => false,
    }
}

fn required_string<'a>(value: &'a Value, key: &str) -> anyhow::Result<&'a str> {
    value
        .get(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .with_context(|| format!("gap closure contract {key} is required"))
}

fn field_definition(field: &str) -> &'static str {
    match field {
        "firma_name" => "aktuelle rechtliche Firmierung",
        "firma_fruehere_namen" => "belegte frühere Firmierungen",
        "firma_aktivitaetsstatus" => "aktueller Register-/Geschäftsstatus",
        "firma_anschrift" => "offizielle Hauptanschrift",
        "firma_besucheranschrift" => "öffentlich ausgewiesene Besucheranschrift",
        "firma_postanschrift" => "offizielle Postanschrift",
        "firma_postfach" => "Postfachangabe",
        "firma_plz" => "Postleitzahl der maßgeblichen Firmenanschrift",
        "firma_ort" => "Ort der maßgeblichen Firmenanschrift",
        "firma_land" => "Land der maßgeblichen Firmenanschrift",
        "firma_email" => "veröffentlichte zentrale Firmen-E-Mail-Adresse",
        "firma_domain" => "kanonische Unternehmensdomain",
        "firma_telefon" => "veröffentlichte zentrale Firmentelefonnummer",
        "firma_fax" => "veröffentlichte zentrale Faxnummer",
        "firma_geschaeftstaetigkeit" => "sachliche Beschreibung der Geschäftstätigkeit",
        "firma_homepage_fact_sheet" => "strukturierte Kernaussagen der offiziellen Homepage",
        "firma_geschaeftsfuehrung" => "aktuell belegte Geschäftsführung",
        "firma_prokura" => "aktuell belegte Prokuristinnen und Prokuristen",
        "wz_code" => "belegter Wirtschaftszweig-/WZ-Code",
        "umsatz" => "aktuellster belegbarer Umsatz mit Zeitraum und Einheit",
        "mitarbeiter" => "aktuellste belegbare Mitarbeiterzahl mit Zeitraum",
        "crm_record_number" => "Sellify-/CRM-Datensatznummer",
        "person_geschlecht" => "belegbare Anrede-/Geschlechtsangabe der Person",
        "person_titel" => "belegter akademischer oder beruflicher Titel",
        "person_vorname" => "belegter Vorname der priorisierten Person",
        "person_nachname" => "belegter Nachname der priorisierten Person",
        "person_funktion" => "belegte organisatorische Funktion der Person",
        "person_position" => "belegte konkrete Stellen-/Positionsbezeichnung",
        "person_email" => "belegte geschäftliche E-Mail-Adresse der Person",
        "person_email_validation" => "belegter Validierungsstatus der Personen-E-Mail",
        "person_telefon" => "belegte geschäftliche Telefonnummer der Person",
        "person_linkedin" => "kanonische LinkedIn-Profil-URL der Person",
        "person_xing" => "kanonische XING-Profil-URL der Person",
        _ => "Wert gemäß dem kanonischen CTOX-Personenrecherche-Feldvokabular",
    }
}

/// Test-only: materialize an RxDB collection table in the tenant RxDB store so
/// `store::upsert_rxdb_collection_record` / `load_rxdb_collection_record` see
/// a real table. In production the browser peer creates these tables during
/// replication; without one, the writer silently skips the upsert.
#[cfg(test)]
pub(super) fn seed_rxdb_collection_table_for_tests(
    root: &Path,
    collection: &str,
) -> anyhow::Result<()> {
    let path = store::rxdb_store_path(root);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let conn = rusqlite::Connection::open(&path)?;
    conn.execute_batch(&format!(
        "CREATE TABLE IF NOT EXISTS ctox_business_os__{collection}__v0 (
            id TEXT PRIMARY KEY NOT NULL,
            revision TEXT,
            deleted INTEGER NOT NULL DEFAULT 0,
            lastWriteTime REAL NOT NULL,
            data TEXT NOT NULL
        );"
    ))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_gap_fixture(
        root: &Path,
        research_command_id: &str,
        record_id: &str,
        field: &str,
    ) -> anyhow::Result<(BusinessCommand, channels::QueueTaskView)> {
        let research_command = BusinessCommand {
            id: Some(research_command_id.to_string()),
            module: "outbound-lead-generation".to_string(),
            command_type: "web_stack.person_research".to_string(),
            record_id: Some(record_id.to_string()),
            payload: serde_json::json!({
                "company": "Example AG",
                "country": "DE",
                "mode": "new_record",
                "fields": [field],
                "research_instructions": "Nur aktuelle Quellen.",
                "known_person_records": [],
                "person_priorities": [],
                "writeback_contract": {
                    "collection": "outbound_lead_generation_leads",
                    "allowed_collections": ["outbound_lead_generation_leads"],
                    "record_ids": [record_id]
                }
            }),
            client_context: Value::Null,
            origin: store::CommandOrigin::TrustedLocal,
        };
        let mut fields = Map::new();
        fields.insert(
            field.to_string(),
            serde_json::json!({"value": null, "candidates": []}),
        );
        let mut phase_a_result = serde_json::json!({
            "requested_fields": [field],
            "fields": fields,
            "plan": []
        });
        // Production always has the Business OS store (with its RxDB collection
        // tables) before a research command runs; open it first so the lead
        // upsert below lands in a real collection table instead of a no-op.
        drop(store::open_store(root)?);
        seed_rxdb_collection_table_for_tests(root, LEAD_COLLECTION)?;
        let task = enqueue_gap_closure_if_needed(root, &research_command, &mut phase_a_result)?
            .context("expected gap task")?;
        let conn = store::open_store(root)?;
        conn.execute(
            "INSERT INTO business_commands
                (command_id, module, command_type, record_id, status, payload_json, client_context_json, observed_at_ms)
             VALUES (?1, ?2, ?3, ?4, 'completed', ?5, '{}', 1)",
            rusqlite::params![
                research_command_id,
                &research_command.module,
                &research_command.command_type,
                record_id,
                serde_json::to_string(&research_command.payload)?,
            ],
        )?;
        drop(conn);
        store::upsert_rxdb_collection_record(
            root,
            LEAD_COLLECTION,
            record_id,
            1,
            serde_json::json!({
                "id": record_id,
                "data": {},
                "contacts": [],
                "evidence": [],
                "research_status": "running",
                "research_phase": "gap_closure",
                "gap_task_id": task.message_key,
                "payload": {"last_research_command_id": research_command_id}
            }),
        )?;
        Ok((research_command, task))
    }

    fn writeback_command(record_id: &str, payload: Value) -> BusinessCommand {
        BusinessCommand {
            id: Some(format!("writeback-{record_id}")),
            module: "outbound-lead-generation".to_string(),
            command_type: "outbound.lead.research_writeback".to_string(),
            record_id: Some(record_id.to_string()),
            payload,
            client_context: Value::Null,
            origin: store::CommandOrigin::TrustedLocal,
        }
    }

    #[test]
    fn phase_a_with_eight_of_thirty_two_fields_queues_exactly_one_complete_gap_task(
    ) -> anyhow::Result<()> {
        let temp = tempfile::tempdir()?;
        let requested = ctox_web_stack::sources::OUTBOUND_RESEARCH_FIELDS
            .iter()
            .map(|field| field.as_str().to_string())
            .collect::<Vec<_>>();
        let mut fields = Map::new();
        for field in requested.iter().take(8) {
            fields.insert(
                field.clone(),
                serde_json::json!({
                    "value": format!("value-{field}"),
                    "candidates": [
                        {"value": format!("value-{field}"), "source_id": "source-a", "source_url": format!("https://a.test/{field}"), "quote": "Beleg A"},
                        {"value": format!("value-{field}"), "source_id": "source-b", "source_url": format!("https://b.test/{field}"), "quote": "Beleg B"}
                    ]
                }),
            );
        }
        let command = BusinessCommand {
            id: Some("research-8-of-32".to_string()),
            module: "outbound-lead-generation".to_string(),
            command_type: "web_stack.person_research".to_string(),
            record_id: Some("lead-8-of-32".to_string()),
            payload: serde_json::json!({
                "company": "Example AG",
                "country": "DE",
                "mode": "new_record",
                "fields": requested,
                "research_instructions": "Aktuelle Quellen bevorzugen.",
                "known_person_records": [{"person_key": "sellify-1"}],
                "person_priorities": ["Geschäftsführung"],
                "writeback_contract": {
                    "collection": "outbound_lead_generation_leads",
                    "allowed_collections": ["outbound_lead_generation_leads"],
                    "record_ids": ["lead-8-of-32"]
                }
            }),
            client_context: Value::Null,
            origin: store::CommandOrigin::TrustedLocal,
        };
        let mut first_result = serde_json::json!({
            "requested_fields": requested,
            "fields": fields,
            "plan": [{"source_id": "phase-a-source"}]
        });
        let first = enqueue_gap_closure_if_needed(temp.path(), &command, &mut first_result)?
            .context("expected gap task")?;
        let mut replay_result = first_result.clone();
        replay_result["search_runs"] = serde_json::json!([{
            "query": "a changed retry result that would otherwise change the task digest"
        }]);
        let replay = enqueue_gap_closure_if_needed(temp.path(), &command, &mut replay_result)?
            .context("expected replayed gap task")?;

        assert_eq!(first.message_key, replay.message_key);
        assert_eq!(channels::list_queue_tasks(temp.path(), &[], 10)?.len(), 1);
        assert_eq!(first.title, "Lückenschluss: Example AG");
        assert_eq!(first.thread_key, "person-research-gap/lead-8-of-32");
        assert_eq!(first.priority, "high");
        let expected_workspace = temp
            .path()
            .join("runtime/research/person/research-8-of-32")
            .to_string_lossy()
            .into_owned();
        assert_eq!(
            first.workspace_root.as_deref(),
            Some(expected_workspace.as_str())
        );
        assert_eq!(first.metadata["idempotency_key"], "gap:research-8-of-32");
        let metadata = &first.metadata[GAP_METADATA_KEY];
        assert_eq!(metadata["gap_task_id"], first.message_key);
        assert_eq!(
            metadata["writeback_contract"]["gap_task_id"],
            first.message_key
        );
        for key in [
            "lead_id",
            "record_id",
            "module",
            "research_command_id",
            "requested_fields",
            "open_fields",
            "terminal_fields",
            "attempted_sources",
            "research_instructions",
            "known_person_records",
            "person_priorities",
            "writeback_contract",
        ] {
            assert!(metadata.get(key).is_some(), "missing metadata key {key}");
        }
        assert_eq!(
            metadata["requested_fields"].as_array().map(Vec::len),
            Some(32)
        );
        assert_eq!(metadata["open_fields"].as_array().map(Vec::len), Some(24));
        assert_eq!(
            metadata["terminal_fields"].as_object().map(Map::len),
            Some(8)
        );
        assert_eq!(first_result["gap_closure"]["required"], true);
        Ok(())
    }

    #[test]
    fn gap_closure_prompt_contains_owner_fields_and_contract_sentences() {
        let prompt = build_gap_closure_prompt(&serde_json::json!({
            "company": "Example AG",
            "record_id": "lead-1",
            "module": "outbound-lead-generation",
            "research_command_id": "research-1",
            "open_fields": ["firma_domain", "person_email"],
            "attempted_sources": [{"source_id": "handelsregister"}],
            "research_instructions": "Nur aktuell bestellte Personen recherchieren.",
            "person_priorities": ["Geschäftsführung"],
            "known_person_records": [{"person_key": "sellify-1", "nachname": "Muster"}],
            "writeback_contract": {"command_type": "outbound.lead.research_writeback"}
        }))
        .unwrap();
        assert!(prompt.contains("Nur aktuell bestellte Personen recherchieren."));
        assert!(prompt.contains("`firma_domain`"));
        assert!(prompt.contains("`person_email`"));
        assert!(prompt.contains("`ctox web search`"));
        assert!(prompt.contains("`ctox web read`"));
        assert!(prompt.contains("`ctox web browser-capture`"));
        assert!(prompt.contains("mindestens 1 dokumentierte Websuche"));
        assert!(prompt.contains("mindestens 2 dokumentierte Seitenlektüren"));
        assert!(prompt.contains("gap_closure/field_status.json"));
        assert!(prompt.contains("outbound.lead.research_writeback"));
        assert!(prompt.contains("bevor dieser Dispatch vom Daemon angenommen wurde"));
    }

    #[test]
    fn no_match_requires_search_two_reads_and_nonempty_workspace_artifacts() -> anyhow::Result<()> {
        let temp = tempfile::tempdir()?;
        std::fs::create_dir_all(temp.path().join("gap_closure/attempts/firma_domain"))?;
        for number in 1..=3 {
            std::fs::write(
                temp.path()
                    .join(format!("gap_closure/attempts/firma_domain/{number}.json")),
                b"{}",
            )?;
        }
        let attempt = |kind: &str, number: usize| FieldAttempt {
            kind: kind.to_string(),
            query_or_url: "query-or-url".to_string(),
            result: Value::Null,
            artifact_path: format!("gap_closure/attempts/firma_domain/{number}.json"),
            at: serde_json::json!(1),
        };
        let mut status = FieldStatus {
            status: "no_match".to_string(),
            value: Value::Null,
            sources: Vec::new(),
            attempts: vec![attempt("web_search", 1), attempt("web_read", 2)],
            reason: "not found".to_string(),
            extra: BTreeMap::new(),
        };
        assert!(validate_terminal_field(
            "firma_domain",
            &status,
            temp.path(),
            &serde_json::json!({})
        )
        .is_err());
        status.attempts.push(attempt("browser_capture", 3));
        validate_terminal_field("firma_domain", &status, temp.path(), &serde_json::json!({}))?;
        std::fs::write(
            temp.path().join("gap_closure/attempts/firma_domain/3.json"),
            b"",
        )?;
        assert!(validate_terminal_field(
            "firma_domain",
            &status,
            temp.path(),
            &serde_json::json!({})
        )
        .unwrap_err()
        .to_string()
        .contains("non-empty"));
        Ok(())
    }

    #[test]
    fn writeback_with_thirty_one_of_thirty_two_terminal_fields_is_rejected() {
        let requested = ctox_web_stack::sources::OUTBOUND_RESEARCH_FIELDS
            .iter()
            .map(|field| field.as_str().to_string())
            .collect::<Vec<_>>();
        let status = FieldStatus {
            status: "unsupported".to_string(),
            value: Value::Null,
            sources: Vec::new(),
            attempts: Vec::new(),
            reason: "unsupported".to_string(),
            extra: BTreeMap::new(),
        };
        let submitted = requested
            .iter()
            .take(31)
            .map(|field| (field.clone(), status.clone()))
            .collect::<BTreeMap<_, _>>();

        let error = validate_field_status_keys(&requested, &submitted).unwrap_err();
        assert!(error.to_string().contains("missing"));
    }

    #[test]
    fn field_status_rejects_missing_and_unknown_vocabulary_fields() {
        let mut submitted = BTreeMap::new();
        submitted.insert(
            "firma_domain".to_string(),
            FieldStatus {
                status: "unsupported".to_string(),
                value: Value::Null,
                sources: Vec::new(),
                attempts: Vec::new(),
                reason: "unsupported".to_string(),
                extra: BTreeMap::new(),
            },
        );
        assert!(validate_field_status_keys(
            &["firma_domain".to_string(), "firma_email".to_string()],
            &submitted
        )
        .unwrap_err()
        .to_string()
        .contains("missing"));
        let unknown = submitted["firma_domain"].clone();
        submitted.insert("firma_unbekannt".to_string(), unknown);
        assert!(
            validate_field_status_keys(&["firma_domain".to_string()], &submitted)
                .unwrap_err()
                .to_string()
                .contains("unsupported person-research field")
        );
    }

    #[test]
    fn writeback_rejects_wrong_record_and_research_command_correlation() {
        let request: ResearchWritebackRequest = serde_json::from_value(serde_json::json!({
            "record_id": "lead-wrong",
            "module": "outbound-lead-generation",
            "research_command_id": "research-wrong",
            "gap_task_id": "gap-wrong",
            "field_status": {},
            "result": {"fields": {}, "person_records": [], "evidence": []}
        }))
        .unwrap();
        let error = validate_task_correlation(
            &serde_json::json!({
                "record_id": "lead-right",
                "module": "outbound-lead-generation",
                "research_command_id": "research-right",
                "gap_task_id": "gap-right"
            }),
            &request,
        )
        .unwrap_err();
        assert!(error.to_string().contains("record_id"));
        let error = validate_task_correlation(
            &serde_json::json!({
                "record_id": "lead-wrong",
                "module": "outbound-lead-generation",
                "research_command_id": "research-right",
                "gap_task_id": "gap-wrong"
            }),
            &request,
        )
        .unwrap_err();
        assert!(error.to_string().contains("research_command_id"));
        let error = validate_task_correlation(
            &serde_json::json!({
                "record_id": "lead-wrong",
                "module": "outbound-lead-generation",
                "research_command_id": "research-wrong",
                "gap_task_id": "gap-right"
            }),
            &request,
        )
        .unwrap_err();
        assert!(error.to_string().contains("task id"));
    }

    #[test]
    fn verified_writeback_completes_lead_and_retains_gap_audit_id() -> anyhow::Result<()> {
        let temp = tempfile::tempdir()?;
        let record_id = "lead-verified";
        let research_command_id = "research-verified";
        let (_, task) =
            create_gap_fixture(temp.path(), research_command_id, record_id, "firma_domain")?;
        let command = writeback_command(
            record_id,
            serde_json::json!({
                "record_id": record_id,
                "module": "outbound-lead-generation",
                "research_command_id": research_command_id,
                "gap_task_id": task.message_key,
                "field_status": {
                    "firma_domain": {
                        "status": "verified",
                        "value": "example.test",
                        "sources": [
                            {"source_id": "official", "url": "https://example.test/imprint", "quote": "Example AG"},
                            {"source_id": "register", "url": "https://register.test/example", "quote": "example.test"}
                        ],
                        "attempts": []
                    }
                },
                "result": {
                    "fields": {"firma_domain": {"value": "example.test"}},
                    "person_records": [],
                    "evidence": []
                }
            }),
        );

        let result = handle_research_writeback(temp.path(), &command)?;
        let lead = store::load_rxdb_collection_record(temp.path(), LEAD_COLLECTION, record_id)?
            .context("lead missing after writeback")?;
        assert_eq!(result["research_status"], "completed");
        assert_eq!(lead["research_status"], "completed");
        assert!(lead["research_phase"].is_null());
        assert_eq!(lead["gap_task_id"], task.message_key);
        assert_eq!(lead["data"]["firma_domain"], "example.test");
        assert_eq!(lead["field_status"]["firma_domain"]["status"], "verified");
        Ok(())
    }

    #[test]
    fn no_match_writeback_finishes_lead_as_needs_review() -> anyhow::Result<()> {
        let temp = tempfile::tempdir()?;
        let record_id = "lead-no-match";
        let research_command_id = "research-no-match";
        let (_, task) =
            create_gap_fixture(temp.path(), research_command_id, record_id, "firma_domain")?;
        let attempts_root = temp
            .path()
            .join("runtime/research/person/research-no-match/gap_closure/attempts/firma_domain");
        std::fs::create_dir_all(&attempts_root)?;
        for number in 1..=3 {
            std::fs::write(attempts_root.join(format!("{number}.json")), b"{}")?;
        }
        let command = writeback_command(
            record_id,
            serde_json::json!({
                "record_id": record_id,
                "module": "outbound-lead-generation",
                "research_command_id": research_command_id,
                "gap_task_id": task.message_key,
                "field_status": {
                    "firma_domain": {
                        "status": "no_match",
                        "value": null,
                        "reason": "Keine belastbare Domain gefunden.",
                        "sources": [],
                        "attempts": [
                            {"kind": "web_search", "query_or_url": "Example AG", "result": {}, "artifact_path": "gap_closure/attempts/firma_domain/1.json", "at": 1},
                            {"kind": "web_read", "query_or_url": "https://one.test", "result": {}, "artifact_path": "gap_closure/attempts/firma_domain/2.json", "at": 2},
                            {"kind": "browser_capture", "query_or_url": "https://two.test", "result": {}, "artifact_path": "gap_closure/attempts/firma_domain/3.json", "at": 3}
                        ]
                    }
                },
                "result": {
                    "fields": {"firma_domain": {"value": null}},
                    "person_records": [],
                    "evidence": []
                }
            }),
        );

        handle_research_writeback(temp.path(), &command)?;
        let lead = store::load_rxdb_collection_record(temp.path(), LEAD_COLLECTION, record_id)?
            .context("lead missing after writeback")?;
        assert_eq!(lead["research_status"], "needs_review");
        assert!(lead["research_phase"].is_null());
        assert_eq!(lead["gap_task_id"], task.message_key);
        assert_eq!(lead["field_status"]["firma_domain"]["status"], "no_match");
        Ok(())
    }

    #[test]
    fn writeback_rejects_wrong_gap_task_id_and_keeps_lead_running() -> anyhow::Result<()> {
        let temp = tempfile::tempdir()?;
        let record_id = "lead-wrong-gap";
        let research_command_id = "research-wrong-gap";
        create_gap_fixture(temp.path(), research_command_id, record_id, "firma_domain")?;
        let command = writeback_command(
            record_id,
            serde_json::json!({
                "record_id": record_id,
                "module": "outbound-lead-generation",
                "research_command_id": research_command_id,
                "gap_task_id": "queue::wrong",
                "field_status": {"firma_domain": {"status": "unsupported"}},
                "result": {"fields": {"firma_domain": {"value": null}}}
            }),
        );

        let error = handle_research_writeback(temp.path(), &command).unwrap_err();
        assert!(error
            .to_string()
            .contains("gap_task_id does not match lead"));
        let lead = store::load_rxdb_collection_record(temp.path(), LEAD_COLLECTION, record_id)?
            .context("lead missing after rejected writeback")?;
        assert_eq!(lead["research_status"], "running");
        assert_eq!(lead["research_phase"], "gap_closure");
        Ok(())
    }

    #[test]
    fn manual_rerun_cancels_the_open_gap_task() -> anyhow::Result<()> {
        let temp = tempfile::tempdir()?;
        let record_id = "lead-rerun";
        let (_, task) = create_gap_fixture(
            temp.path(),
            "research-before-rerun",
            record_id,
            "firma_domain",
        )?;
        let rerun = BusinessCommand {
            id: Some("research-after-rerun".to_string()),
            module: "outbound-lead-generation".to_string(),
            command_type: "web_stack.person_research".to_string(),
            record_id: Some(record_id.to_string()),
            payload: serde_json::json!({"company": "Example AG"}),
            client_context: Value::Null,
            origin: store::CommandOrigin::TrustedLocal,
        };

        assert!(cancel_open_gap_task_for_new_research(temp.path(), &rerun)?);
        let cancelled = channels::load_queue_task(temp.path(), &task.message_key)?
            .context("cancelled gap task missing")?;
        assert_eq!(cancelled.route_status, "cancelled");
        assert!(cancelled
            .status_note
            .as_deref()
            .is_some_and(|note| note.contains("research-after-rerun")));
        Ok(())
    }

    #[test]
    fn result_fields_reject_free_text_and_nonverified_values() {
        let free_text: ResearchWritebackResult = serde_json::from_value(serde_json::json!({
            "fields": {"firma_domain": "example.test"}
        }))
        .unwrap();
        assert!(
            validate_result_shape(&free_text, &["firma_domain".to_string()])
                .unwrap_err()
                .to_string()
                .contains("structured object")
        );

        let result: ResearchWritebackResult = serde_json::from_value(serde_json::json!({
            "fields": {"firma_domain": {"value": "example.test"}}
        }))
        .unwrap();
        let statuses = BTreeMap::from([(
            "firma_domain".to_string(),
            FieldStatus {
                status: "unsupported".to_string(),
                value: Value::Null,
                sources: Vec::new(),
                attempts: Vec::new(),
                reason: "unsupported".to_string(),
                extra: BTreeMap::new(),
            },
        )]);
        assert!(validate_result_field_status_consistency(&result, &statuses)
            .unwrap_err()
            .to_string()
            .contains("non-verified"));
    }

    #[test]
    fn action_required_needs_auth_assist_or_credential_source() {
        let mut status = FieldStatus {
            status: "action_required".to_string(),
            value: Value::Null,
            sources: Vec::new(),
            attempts: Vec::new(),
            reason: "Login erforderlich".to_string(),
            extra: BTreeMap::new(),
        };
        let temp = tempfile::tempdir().unwrap();
        assert!(validate_terminal_field(
            "firma_domain",
            &status,
            temp.path(),
            &serde_json::json!({"attempted_sources": []})
        )
        .is_err());
        status.sources.push(FieldSource {
            source_id: "credential-source".to_string(),
            url: String::new(),
            quote: String::new(),
            person_key: None,
            requires_credential: true,
            task_id: String::new(),
            command_id: String::new(),
        });
        assert!(validate_terminal_field(
            "firma_domain",
            &status,
            temp.path(),
            &serde_json::json!({"attempted_sources": []})
        )
        .is_ok());
        status.sources[0].requires_credential = false;
        status.sources[0].task_id = "auth-assist-task".to_string();
        assert!(validate_terminal_field(
            "firma_domain",
            &status,
            temp.path(),
            &serde_json::json!({"attempted_sources": []})
        )
        .is_ok());
    }

    #[test]
    fn verified_person_evidence_requires_matching_person_key() {
        let mut status = FieldStatus {
            status: "verified".to_string(),
            value: serde_json::json!("Ada"),
            sources: vec![
                FieldSource {
                    source_id: "official".to_string(),
                    url: "https://example.test/ada".to_string(),
                    quote: "Ada Example".to_string(),
                    person_key: None,
                    requires_credential: false,
                    task_id: String::new(),
                    command_id: String::new(),
                },
                FieldSource {
                    source_id: "register".to_string(),
                    url: "https://register.test/ada".to_string(),
                    quote: "Ada".to_string(),
                    person_key: None,
                    requires_credential: false,
                    task_id: String::new(),
                    command_id: String::new(),
                },
            ],
            attempts: Vec::new(),
            reason: String::new(),
            extra: BTreeMap::new(),
        };
        let temp = tempfile::tempdir().unwrap();
        assert!(validate_terminal_field(
            "person_vorname",
            &status,
            temp.path(),
            &serde_json::json!({})
        )
        .unwrap_err()
        .to_string()
        .contains("person_key"));
        for source in &mut status.sources {
            source.person_key = Some("person-ada".to_string());
        }
        assert!(validate_terminal_field(
            "person_vorname",
            &status,
            temp.path(),
            &serde_json::json!({})
        )
        .is_ok());
    }

    #[test]
    fn verified_sources_must_use_different_hosts() {
        let status = |second_url: &str| FieldStatus {
            status: "verified".to_string(),
            value: serde_json::json!("example.test"),
            sources: vec![
                FieldSource {
                    source_id: "a".to_string(),
                    url: "https://www.example.test/a".to_string(),
                    quote: "A".to_string(),
                    person_key: None,
                    requires_credential: false,
                    task_id: String::new(),
                    command_id: String::new(),
                },
                FieldSource {
                    source_id: "b".to_string(),
                    url: second_url.to_string(),
                    quote: "B".to_string(),
                    person_key: None,
                    requires_credential: false,
                    task_id: String::new(),
                    command_id: String::new(),
                },
            ],
            attempts: Vec::new(),
            reason: String::new(),
            extra: BTreeMap::new(),
        };
        let temp = tempfile::tempdir().unwrap();
        assert!(validate_terminal_field(
            "firma_domain",
            &status("https://example.test/b"),
            temp.path(),
            &serde_json::json!({})
        )
        .is_err());
        assert!(validate_terminal_field(
            "firma_domain",
            &status("https://other.test/b"),
            temp.path(),
            &serde_json::json!({})
        )
        .is_ok());
    }
}
