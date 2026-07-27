//! Bridge between source modules and the CTOX scrape registry
//! (`src/capabilities/scrape.rs` + the `universal-scraping` skill).
//!
//! When a [`SourceModule`] sets [`SourceModule::scrape_target_key`] to
//! `Some(<key>)`, this bridge dispatches extraction to the registered
//! script revision under `runtime/scraping/targets/<key>/scripts/current.*`
//! instead of running the module's in-tree Rust `extract_fields`. The
//! shell-out invokes:
//!
//! ```text
//! ctox scrape execute --target-key <key> \
//!     --trigger-kind <person_research|repair|manual> \
//!     --input-json '{"company":"<name>","country":"<DE|AT|CH>"}' \
//!     --allow-heal --runtime-root <root>
//! ```
//!
//! and parses the resulting envelope into typed `(FieldKey, FieldEvidence)`
//! pairs that the [`person_research`](crate::person_research) orchestrator
//! aggregates exactly like Rust-native results.
//!
//! The bridge fails soft: if the target is not registered yet (e.g. early
//! in the migration of a source from Rust to scrape-target), it returns an
//! empty record list with a `target_not_registered` reason so the caller
//! can fall back to the Rust path.

use std::path::Path;
use std::path::PathBuf;
use std::process::Command;

use chrono::DateTime;
use rusqlite::params;
use rusqlite::Connection;
use rusqlite::OpenFlags;
use serde_json::json;
use serde_json::Value;

use super::Confidence;
use super::Country;
use super::FieldEvidence;
use super::FieldKey;
use super::SourceModule;
use crate::unlock::PublicUnlockOutcome;
use crate::unlock::PublicUnlockRequest;

const MAX_EVIDENCE_AGE_MS: u64 = 24 * 60 * 60 * 1_000;
const MAX_CLOCK_SKEW_MS: u64 = 5 * 60 * 1_000;

/// What the bridge produced for one delegated scrape call.
#[derive(Debug, Clone)]
pub struct ScrapeBridgeResult {
    pub target_key: &'static str,
    /// Typed records the scrape script emitted, mapped onto our vocab.
    pub fields: Vec<(FieldKey, FieldEvidence)>,
    /// `ctox scrape execute` classification:
    /// `succeeded | temporary_unreachable | portal_drift | blocked | partial_output`,
    /// or `target_not_registered` when the bridge skipped the call.
    pub classification: String,
    /// Compact reason string from the executor.
    pub reason: Option<String>,
    /// Whether a CTOX repair queue task was enqueued by `--allow-heal`.
    pub repair_queued: bool,
    /// Run id assigned by the scrape executor, if the call ran.
    pub run_id: Option<String>,
    /// Reasons why individual records were rejected before they could become
    /// person-research evidence.
    pub evidence_rejections: Vec<String>,
    /// Total registered-adapter executions for this source. Public browser
    /// fallback is bounded to one retry, so this is always one or two.
    pub attempts: usize,
    /// First classification when a public browser fallback caused a retry.
    pub initial_classification: Option<String>,
    /// Redacted result of the generic public-browser warm-up.
    pub public_browser_fallback: Option<PublicUnlockOutcome>,
}

/// Drive a scrape-target extraction for the given source module.
///
/// `root` is the CTOX state root (the dir containing `runtime/ctox.sqlite3`).
/// `ctox_bin` is the path of the `ctox` binary to invoke; pass
/// [`default_ctox_bin`] in the orchestrator.
pub fn run_via_scrape_target(
    module: &dyn SourceModule,
    company: &str,
    country: Country,
    root: &Path,
    ctox_bin: &Path,
) -> ScrapeBridgeResult {
    let Some(target_key) = module.scrape_target_key() else {
        // Not opted in — caller should use the Rust path.
        return ScrapeBridgeResult {
            target_key: "",
            fields: Vec::new(),
            classification: "no_scrape_target".to_string(),
            reason: Some("module.scrape_target_key() == None".to_string()),
            repair_queued: false,
            run_id: None,
            evidence_rejections: Vec::new(),
            attempts: 0,
            initial_classification: None,
            public_browser_fallback: None,
        };
    };

    let input = json!({
        "company": company,
        "country": country.as_iso(),
        "source_id": module.id(),
    });

    let current = run_with_public_browser_fallback(
        module,
        input,
        |input| execute_scrape_target_once(module, company, root, ctox_bin, target_key, input),
        |request| crate::unlock::run_public_browser_fallback(root, ctox_bin, request),
    );
    if !requires_public_browser_fallback(&current.classification) {
        return current;
    }
    recent_successful_result(root, target_key, module, company, &current).unwrap_or(current)
}

fn run_with_public_browser_fallback<Execute, Unlock>(
    module: &dyn SourceModule,
    mut input: Value,
    mut execute: Execute,
    mut unlock: Unlock,
) -> ScrapeBridgeResult
where
    Execute: FnMut(&Value) -> ScrapeBridgeResult,
    Unlock: FnMut(&PublicUnlockRequest) -> PublicUnlockOutcome,
{
    let mut initial = execute(&input);
    initial.attempts = 1;
    if !requires_public_browser_fallback(&initial.classification) {
        return initial;
    }

    let trigger = initial.classification.clone();
    let allowed_domains = public_source_domains(module);
    let request = PublicUnlockRequest {
        source_id: module.id().to_string(),
        browser_session_id: public_source_browser_session_id(module),
        target_url: public_source_start_url(module),
        allowed_domains,
        trigger: trigger.clone(),
        timeout_ms: 45_000,
    };
    let fallback = unlock(&request);
    if !fallback.ok {
        initial.initial_classification = Some(trigger);
        initial.public_browser_fallback = Some(fallback);
        return initial;
    }

    input["web_fallback"] = json!({
        "kind": "public_browser_unlock",
        "attempt": 1,
        "trigger": trigger,
        "source_id": module.id(),
        "browser_session_id": fallback.browser_session_id.clone(),
        "target_url": fallback.target_url.clone(),
        "final_url": fallback.final_url.clone(),
        "browser_status": fallback.status.clone(),
        "secret_value_in_payload": false,
    });
    input["browser_session_id"] = json!(fallback.browser_session_id.clone());
    let mut retried = execute(&input);
    retried.attempts = 2;
    retried.initial_classification = Some(initial.classification);
    retried.public_browser_fallback = Some(fallback);
    retried
}

fn requires_public_browser_fallback(classification: &str) -> bool {
    matches!(classification, "blocked" | "temporary_unreachable")
}

fn public_source_domains(module: &dyn SourceModule) -> Vec<String> {
    let mut domains = Vec::with_capacity(1 + module.host_suffixes().len());
    domains.push(
        module
            .id()
            .trim()
            .trim_start_matches("www.")
            .to_ascii_lowercase(),
    );
    domains.extend(module.host_suffixes().iter().map(|domain| {
        domain
            .trim()
            .trim_start_matches("www.")
            .to_ascii_lowercase()
    }));
    domains.retain(|domain| !domain.is_empty());
    domains.sort();
    domains.dedup();
    domains
}

fn public_source_start_url(module: &dyn SourceModule) -> String {
    format!(
        "https://www.{}/",
        module.id().trim().trim_start_matches("www.")
    )
}

fn public_source_browser_session_id(module: &dyn SourceModule) -> String {
    let source = module
        .id()
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() {
                ch.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect::<String>();
    format!("browser_session_web_stack_public_{source}")
}

fn execute_scrape_target_once(
    module: &dyn SourceModule,
    company: &str,
    root: &Path,
    ctox_bin: &Path,
    target_key: &'static str,
    input: &Value,
) -> ScrapeBridgeResult {
    let output = Command::new(ctox_bin)
        .arg("scrape")
        .arg("execute")
        .arg("--target-key")
        .arg(target_key)
        .arg("--trigger-kind")
        .arg("manual")
        .arg("--allow-heal")
        .arg("--input-json")
        .arg(input.to_string())
        .arg("--runtime-root")
        .arg(root)
        .output();

    let output = match output {
        Ok(o) => o,
        Err(err) => {
            return ScrapeBridgeResult {
                target_key,
                fields: Vec::new(),
                classification: "subprocess_failed".to_string(),
                reason: Some(format!("failed to spawn ctox: {err}")),
                repair_queued: false,
                run_id: None,
                evidence_rejections: Vec::new(),
                attempts: 1,
                initial_classification: None,
                public_browser_fallback: None,
            };
        }
    };

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    // Distinguish "target not registered" from other failures so the
    // caller can fall back to Rust extraction cleanly.
    if !output.status.success() {
        let classification =
            if stderr.contains("target_key not found") || stdout.contains("target_key not found") {
                "target_not_registered"
            } else {
                "executor_error"
            };
        return ScrapeBridgeResult {
            target_key,
            fields: Vec::new(),
            classification: classification.to_string(),
            reason: Some(stderr.into_owned()),
            repair_queued: false,
            run_id: None,
            evidence_rejections: Vec::new(),
            attempts: 1,
            initial_classification: None,
            public_browser_fallback: None,
        };
    }

    let envelope: Value = match serde_json::from_str(&stdout) {
        Ok(v) => v,
        Err(err) => {
            return ScrapeBridgeResult {
                target_key,
                fields: Vec::new(),
                classification: "parse_error".to_string(),
                reason: Some(format!("invalid JSON from scrape execute: {err}")),
                repair_queued: false,
                run_id: None,
                evidence_rejections: Vec::new(),
                attempts: 1,
                initial_classification: None,
                public_browser_fallback: None,
            };
        }
    };

    parse_scrape_envelope(target_key, module, company, &envelope)
}

/// Best-effort default for the `ctox` binary path: the currently running
/// executable. Falls back to plain `"ctox"` if `current_exe` is unavailable
/// (test contexts, sandboxes).
pub fn default_ctox_bin() -> PathBuf {
    std::env::current_exe().unwrap_or_else(|_| PathBuf::from("ctox"))
}

fn parse_scrape_envelope(
    target_key: &'static str,
    module: &dyn SourceModule,
    company: &str,
    envelope: &Value,
) -> ScrapeBridgeResult {
    let classification = envelope
        .get("classification")
        .and_then(|v| v.get("status"))
        .or_else(|| envelope.get("status"))
        .and_then(Value::as_str)
        .unwrap_or("unknown")
        .to_string();
    let reason = envelope
        .get("classification")
        .and_then(|v| v.get("reason"))
        .and_then(Value::as_str)
        .or_else(|| envelope.get("reason").and_then(Value::as_str))
        .map(|s| s.to_string());
    let repair_queued = envelope
        .get("repair_queue_task")
        .map(|v| !v.is_null())
        .unwrap_or(false);
    let run_id = envelope
        .get("run_id")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(|s| s.to_string());

    let mut evidence_rejections = Vec::new();
    if envelope.get("ok").and_then(Value::as_bool) != Some(true) {
        evidence_rejections.push("executor_not_ok".to_string());
    }
    if classification != "succeeded" {
        evidence_rejections.push(format!("classification_not_succeeded:{classification}"));
    }
    if run_id.is_none() {
        evidence_rejections.push("missing_run_id".to_string());
    }
    if let Some(manifest_path) = envelope.get("run_manifest_path").and_then(Value::as_str) {
        if !manifest_matches_run(Path::new(manifest_path), target_key, run_id.as_deref()) {
            evidence_rejections.push("run_manifest_mismatch".to_string());
        }
    }

    // Records: ctox scrape execute persists them to
    // `<run_dir>/outputs/records.json` and reports the run_manifest_path
    // in the envelope. The script's raw stdout is also tail-truncated
    // into `result.stdout_excerpt`; for short runs the records may be
    // recoverable from there, but we prefer the on-disk records.json
    // because it is the durable contract documented in
    // skills/.../universal-scraping/references/storage-layout.md.
    let records: Vec<Value> = if let Some(records_path) = locate_records_file(envelope) {
        load_records_file(&records_path).unwrap_or_default()
    } else {
        envelope
            .get("records")
            .or_else(|| envelope.get("result").and_then(|v| v.get("records")))
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default()
    };

    let fields = if evidence_rejections.is_empty() {
        records
            .into_iter()
            .enumerate()
            .filter_map(|(index, record)| {
                match record_to_field_evidence(&record, module, company, run_id.as_deref()) {
                    Ok(field) => field,
                    Err(reason) => {
                        evidence_rejections.push(format!("record_{index}:{reason}"));
                        None
                    }
                }
            })
            .collect()
    } else {
        Vec::new()
    };

    ScrapeBridgeResult {
        target_key,
        fields,
        classification,
        reason,
        repair_queued,
        run_id,
        evidence_rejections,
        attempts: 1,
        initial_classification: None,
        public_browser_fallback: None,
    }
}

fn manifest_matches_run(path: &Path, target_key: &str, run_id: Option<&str>) -> bool {
    let Some(expected_run_id) = run_id else {
        return false;
    };
    let Ok(raw) = std::fs::read_to_string(path) else {
        return false;
    };
    let Ok(manifest) = serde_json::from_str::<Value>(&raw) else {
        return false;
    };
    manifest.get("run_id").and_then(Value::as_str) == Some(expected_run_id)
        && manifest.get("target_key").and_then(Value::as_str) == Some(target_key)
        && manifest.get("status").and_then(Value::as_str) == Some("succeeded")
}

/// Try to find the on-disk records.json that the scrape executor wrote
/// for this run. Derived from `run_manifest_path` when present.
fn locate_records_file(envelope: &Value) -> Option<PathBuf> {
    let manifest = envelope.get("run_manifest_path").and_then(Value::as_str)?;
    let manifest_path = PathBuf::from(manifest);
    let run_dir = manifest_path.parent()?;
    let candidate = run_dir.join("outputs").join("records.json");
    Some(candidate)
}

fn load_records_file(path: &Path) -> Option<Vec<Value>> {
    let raw = std::fs::read_to_string(path).ok()?;
    let value: Value = serde_json::from_str(&raw).ok()?;
    match value {
        Value::Array(items) => Some(items),
        Value::Object(map) => {
            let recs = map.get("records")?;
            recs.as_array().cloned()
        }
        _ => None,
    }
}

fn recent_successful_result(
    root: &Path,
    target_key: &'static str,
    module: &dyn SourceModule,
    company: &str,
    current: &ScrapeBridgeResult,
) -> Option<ScrapeBridgeResult> {
    let relative = Path::new("scraping")
        .join("targets")
        .join(target_key)
        .join("runs");
    let run_roots = [root.join(&relative), root.join("runtime").join(&relative)];
    let mut cached_candidates = database_success_candidates(root, target_key, company);
    for runs in run_roots {
        let Ok(entries) = std::fs::read_dir(runs) else {
            continue;
        };
        for entry in entries.flatten() {
            let manifest_path = entry.path().join("run.json");
            let Ok(raw) = std::fs::read_to_string(&manifest_path) else {
                continue;
            };
            let Ok(manifest) = serde_json::from_str::<Value>(&raw) else {
                continue;
            };
            if manifest.get("target_key").and_then(Value::as_str) != Some(target_key) {
                continue;
            }
            if manifest.get("status").and_then(Value::as_str) == Some("succeeded")
                && cached_run_company_matches(&manifest, &manifest_path, company)
            {
                let finished_at = manifest
                    .get("finished_at_ms")
                    .or_else(|| manifest.get("started_at_ms"))
                    .or_else(|| manifest.get("scheduled_for"))
                    .and_then(timestamp_ms)
                    .or_else(|| file_modified_ms(&manifest_path));
                if let (Some(finished_at), Some(run_id), Some(records_path)) = (
                    finished_at.filter(|timestamp| is_fresh(*timestamp)),
                    manifest.get("run_id").and_then(Value::as_str),
                    manifest_records_path(&manifest_path),
                ) {
                    cached_candidates.push((finished_at, run_id.to_string(), records_path));
                }
            }
            if let Some(candidate) = embedded_last_success_candidate(&manifest, target_key, company)
            {
                cached_candidates.push(candidate);
            }
        }
    }
    cached_candidates.sort_by(|left, right| right.0.cmp(&left.0));
    cached_candidates.dedup_by(|left, right| left.1 == right.1);

    for (_, run_id, records_path) in cached_candidates {
        let Some(records) = load_records_file(&records_path) else {
            continue;
        };
        let envelope = json!({
            "ok": true,
            "status": "succeeded",
            "run_id": run_id,
            "records": records,
        });
        let mut cached = parse_scrape_envelope(target_key, module, company, &envelope);
        if cached.fields.is_empty() || !cached.evidence_rejections.is_empty() {
            continue;
        }
        cached.reason = Some(format!(
            "fresh_cached_success_after_{}",
            current.classification
        ));
        cached.attempts = current.attempts;
        cached.initial_classification = Some(current.classification.clone());
        cached.public_browser_fallback = current.public_browser_fallback.clone();
        return Some(cached);
    }
    None
}

fn database_success_candidates(
    root: &Path,
    target_key: &str,
    company: &str,
) -> Vec<(u64, String, PathBuf)> {
    let mut candidates = Vec::new();
    for db_path in [root.join("ctox.sqlite3"), root.join("runtime/ctox.sqlite3")] {
        if !db_path.is_file() {
            continue;
        }
        let Ok(connection) = Connection::open_with_flags(
            &db_path,
            OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
        ) else {
            continue;
        };
        let Ok(mut statement) = connection.prepare(
            "SELECT r.run_id, r.finished_at, r.started_at, r.output_dir
             FROM scrape_run r
             JOIN scrape_target t ON t.target_id = r.target_id
             WHERE t.target_key = ?1 AND r.status = 'succeeded'
             ORDER BY COALESCE(r.finished_at, r.started_at) DESC
             LIMIT 24",
        ) else {
            continue;
        };
        let Ok(rows) = statement.query_map(params![target_key], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, Option<String>>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, Option<String>>(3)?,
            ))
        }) else {
            continue;
        };
        for row in rows.flatten() {
            let (run_id, finished_at, started_at, output_dir) = row;
            let Some(timestamp) = finished_at
                .as_deref()
                .or(Some(started_at.as_str()))
                .and_then(|value| timestamp_ms(&Value::String(value.to_string())))
                .filter(|timestamp| is_fresh(*timestamp))
            else {
                continue;
            };
            let Some(output_dir) = output_dir.filter(|value| !value.trim().is_empty()) else {
                continue;
            };
            let records_path = PathBuf::from(output_dir).join("outputs/records.json");
            if cached_records_company_matches(&records_path, company) {
                candidates.push((timestamp, run_id, records_path));
            }
        }
    }
    candidates
}

fn manifest_records_path(manifest_path: &Path) -> Option<PathBuf> {
    Some(manifest_path.parent()?.join("outputs").join("records.json"))
}

fn embedded_last_success_candidate(
    manifest: &Value,
    target_key: &str,
    company: &str,
) -> Option<(u64, String, PathBuf)> {
    let previous = manifest.pointer("/run_context/last_successful_run")?;
    let finished_at = previous.get("finished_at").and_then(timestamp_ms)?;
    if !is_fresh(finished_at) {
        return None;
    }
    let run_id = previous.get("run_id").and_then(Value::as_str)?.to_string();
    let materialization = previous.pointer("/result/materialization")?;
    if materialization.get("target_key").and_then(Value::as_str) != Some(target_key) {
        return None;
    }
    let records_path = PathBuf::from(
        materialization
            .get("latest_records_path")
            .and_then(Value::as_str)?,
    );
    if !records_path
        .components()
        .any(|component| component.as_os_str() == target_key)
        || !cached_records_company_matches(&records_path, company)
    {
        return None;
    }
    Some((finished_at, run_id, records_path))
}

fn cached_run_company_matches(manifest: &Value, manifest_path: &Path, company: &str) -> bool {
    if manifest
        .pointer("/input/company")
        .or_else(|| manifest.pointer("/run_context/input/company"))
        .and_then(Value::as_str)
        .is_some_and(|identity| company_identity_matches(company, identity))
    {
        return true;
    }
    let records_path = manifest_path
        .parent()
        .map(|run_dir| run_dir.join("outputs").join("records.json"));
    records_path
        .as_deref()
        .is_some_and(|path| cached_records_company_matches(path, company))
}

fn cached_records_company_matches(records_path: &Path, company: &str) -> bool {
    load_records_file(records_path).is_some_and(|records| {
        records.iter().any(|record| {
            let field = record
                .get("field")
                .and_then(Value::as_str)
                .unwrap_or_default();
            if !matches!(field, "firma_name" | "company_name") {
                return false;
            }
            record
                .get("value")
                .and_then(Value::as_str)
                .is_some_and(|identity| company_identity_matches(company, identity))
        })
    })
}

fn file_modified_ms(path: &Path) -> Option<u64> {
    path.metadata()
        .ok()?
        .modified()
        .ok()?
        .duration_since(std::time::UNIX_EPOCH)
        .ok()
        .map(|duration| duration.as_millis() as u64)
}

fn record_to_field_evidence(
    record: &Value,
    module: &dyn SourceModule,
    company: &str,
    run_id: Option<&str>,
) -> Result<Option<(FieldKey, FieldEvidence)>, String> {
    let field_str = record
        .get("field")
        .and_then(Value::as_str)
        .ok_or_else(|| "missing_field".to_string())?;
    let field = FieldKey::from_str(field_str).ok_or_else(|| "unknown_field".to_string())?;
    let value = record
        .get("value")
        .and_then(Value::as_str)
        .ok_or_else(|| "missing_value".to_string())?
        .trim()
        .to_string();
    if value.trim().is_empty() {
        return Ok(None);
    }
    let expected_run_id = run_id.ok_or_else(|| "missing_run_id".to_string())?;
    let record_run_id = first_string(record, &["run_id", "scrape_run_id"])
        .or_else(|| {
            record
                .pointer("/provenance/run_id")
                .and_then(Value::as_str)
                .map(str::to_string)
        })
        .ok_or_else(|| "missing_record_run_id".to_string())?;
    if record_run_id != expected_run_id {
        return Err("record_run_id_mismatch".to_string());
    }
    let record_source_id = first_string(record, &["source_id", "source_key", "source"])
        .or_else(|| {
            record
                .pointer("/provenance/source_id")
                .and_then(Value::as_str)
                .map(str::to_string)
        })
        .ok_or_else(|| "missing_record_source_id".to_string())?;
    if !source_id_matches(&record_source_id, module) {
        return Err("record_source_id_mismatch".to_string());
    }

    let canonical_url = record.get("canonical_url");
    let source_url = record.get("source_url");
    let canonical_url = canonical_url
        .map(required_string)
        .transpose()
        .map_err(|_| "invalid_canonical_url".to_string())?;
    let source_url = source_url
        .map(required_string)
        .transpose()
        .map_err(|_| "invalid_source_url".to_string())?;
    let evidence_url = canonical_url
        .as_deref()
        .or(source_url.as_deref())
        .ok_or_else(|| "missing_source_url".to_string())?;
    if !valid_source_url(evidence_url, module)
        || canonical_url
            .as_deref()
            .is_some_and(|url| !valid_source_url(url, module))
        || source_url
            .as_deref()
            .is_some_and(|url| !valid_source_url(url, module))
    {
        return Err("source_url_not_canonical_for_source".to_string());
    }

    validate_record_evidence_gate(record)?;
    let identity = record_company_identity(record, field, &value)
        .ok_or_else(|| "missing_company_identity".to_string())?;
    if !company_identity_matches(company, &identity) {
        return Err("company_identity_mismatch".to_string());
    }

    let confidence = match record.get("confidence").and_then(Value::as_str) {
        Some("high") => Confidence::High,
        Some("low") => Confidence::Low,
        Some("user_provided") => Confidence::UserProvided,
        _ => Confidence::Medium,
    };
    let source_url = evidence_url.to_string();
    let note = record
        .get("note")
        .and_then(Value::as_str)
        .map(|s| s.to_string());
    Ok(Some((
        field,
        FieldEvidence {
            value,
            confidence,
            source_url,
            note,
        },
    )))
}

fn validate_record_evidence_gate(record: &Value) -> Result<(), String> {
    let gate = record
        .get("evidence_gate")
        .or_else(|| record.get("evidence"))
        .unwrap_or(record);
    if gate.get("evidence_eligible").and_then(Value::as_bool) != Some(true) {
        return Err("evidence_not_eligible".to_string());
    }
    if gate.get("verification_status").and_then(Value::as_str) != Some("verified") {
        return Err("evidence_not_verified".to_string());
    }
    if !gate
        .get("http_status")
        .and_then(Value::as_u64)
        .is_some_and(|status| (200..300).contains(&status))
    {
        return Err("evidence_http_status_invalid".to_string());
    }
    let checked_at = gate
        .get("checked_at")
        .or_else(|| gate.get("checked_at_ms"))
        .and_then(timestamp_ms)
        .ok_or_else(|| "missing_evidence_timestamp".to_string())?;
    if !is_fresh(checked_at) {
        return Err("evidence_stale".to_string());
    }
    if !gate
        .get("snapshot_hash")
        .and_then(Value::as_str)
        .is_some_and(|hash| !hash.trim().is_empty())
    {
        return Err("missing_snapshot_hash".to_string());
    }
    if gate.get("fresh").and_then(Value::as_bool) == Some(false) {
        return Err("evidence_marked_stale".to_string());
    }
    Ok(())
}

fn first_string(value: &Value, keys: &[&str]) -> Option<String> {
    keys.iter().find_map(|key| {
        value
            .get(*key)
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_string)
    })
}

fn required_string(value: &Value) -> Result<String, ()> {
    let value = value.as_str().ok_or(())?.trim();
    (!value.is_empty()).then(|| value.to_string()).ok_or(())
}

fn source_id_matches(raw: &str, module: &dyn SourceModule) -> bool {
    raw.eq_ignore_ascii_case(module.id())
        || module
            .aliases()
            .iter()
            .any(|alias| raw.eq_ignore_ascii_case(alias))
}

fn valid_source_url(raw: &str, module: &dyn SourceModule) -> bool {
    let Ok(url) = url::Url::parse(raw) else {
        return false;
    };
    matches!(url.scheme(), "http" | "https")
        && url.host_str().is_some()
        && (url.host_str().is_some_and(|host| {
            let host = host
                .trim_start_matches("www.")
                .trim_start_matches("app.")
                .trim_start_matches("api.")
                .to_ascii_lowercase();
            let matches_host = |candidate: &str| {
                let candidate = candidate.to_ascii_lowercase();
                host == candidate || host.ends_with(&format!(".{candidate}"))
            };
            matches_host(module.id())
                || module.aliases().iter().any(|alias| matches_host(alias))
                || module
                    .host_suffixes()
                    .iter()
                    .any(|suffix| matches_host(suffix))
        }))
}

fn record_company_identity(record: &Value, field: FieldKey, value: &str) -> Option<String> {
    first_string(
        record,
        &["company", "company_name", "firma_name", "company_identity"],
    )
    .or_else(|| {
        record
            .pointer("/provenance/company")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_string)
    })
    .or_else(|| (field == FieldKey::FirmaName).then(|| value.to_string()))
}

fn company_identity_matches(company: &str, identity: &str) -> bool {
    let expected = company_tokens(company);
    let actual: Vec<String> = identity
        .split(|ch: char| !ch.is_ascii_alphanumeric())
        .map(str::to_ascii_lowercase)
        .filter(|token| !token.is_empty())
        .collect();
    !expected.is_empty()
        && expected
            .iter()
            .all(|token| actual.iter().any(|actual| actual == token))
}

fn company_tokens(company: &str) -> Vec<String> {
    company
        .split(|ch: char| !ch.is_ascii_alphanumeric())
        .map(str::to_ascii_lowercase)
        .filter(|token| token.len() >= 3)
        .filter(|token| {
            !matches!(
                token.as_str(),
                "ag" | "at"
                    | "ch"
                    | "co"
                    | "de"
                    | "gmbh"
                    | "inc"
                    | "kg"
                    | "ltd"
                    | "mbh"
                    | "sa"
                    | "sarl"
                    | "se"
                    | "the"
                    | "ug"
                    | "und"
            )
        })
        .collect()
}

fn timestamp_ms(value: &Value) -> Option<u64> {
    let raw = value
        .as_u64()
        .or_else(|| {
            value
                .as_i64()
                .filter(|value| *value >= 0)
                .map(|value| value as u64)
        })
        .or_else(|| {
            let text = value.as_str()?.trim();
            text.parse::<u64>().ok()
        });
    if let Some(raw) = raw {
        return Some(if raw < 10_000_000_000 {
            raw * 1_000
        } else {
            raw
        });
    }
    value
        .as_str()
        .and_then(|text| DateTime::parse_from_rfc3339(text.trim()).ok())
        .and_then(|date| u64::try_from(date.timestamp_millis()).ok())
}

fn is_fresh(timestamp: u64) -> bool {
    let now = now_ms();
    timestamp <= now.saturating_add(MAX_CLOCK_SKEW_MS)
        && now.saturating_sub(timestamp) <= MAX_EVIDENCE_AGE_MS
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_cached_run(root: &Path, checked_at: u64, finished_at: u64) {
        let run_id = "scrape_run-cached";
        let run = root.join("scraping/targets/northdata-de/runs").join(run_id);
        std::fs::create_dir_all(run.join("outputs")).unwrap();
        std::fs::write(
            run.join("run.json"),
            serde_json::to_vec(&json!({
                "run_id": run_id,
                "target_key": "northdata-de",
                "status": "succeeded",
                "finished_at_ms": finished_at,
                "input": { "company": "Example Industrial GmbH", "country": "DE" }
            }))
            .unwrap(),
        )
        .unwrap();
        std::fs::write(
            run.join("outputs/records.json"),
            serde_json::to_vec(&json!([{
                "field": "firma_name",
                "value": "Example Industrial GmbH",
                "confidence": "high",
                "source_url": "https://www.northdata.de/Example+Industrial+GmbH",
                "run_id": run_id,
                "source_id": "northdata.de",
                "company_name": "Example Industrial GmbH",
                "evidence_eligible": true,
                "verification_status": "verified",
                "http_status": 200,
                "checked_at": checked_at,
                "snapshot_hash": "sha256:cached"
            }]))
            .unwrap(),
        )
        .unwrap();
    }

    #[test]
    fn fresh_identity_matched_success_can_cover_a_transient_access_failure() {
        let root = std::env::temp_dir().join(format!(
            "ctox-scrape-cache-{}-{}",
            std::process::id(),
            now_ms()
        ));
        write_cached_run(&root, now_ms(), now_ms());
        let current = stub_result("temporary_unreachable");
        let result = recent_successful_result(
            &root,
            "northdata-de",
            crate::sources::find("northdata.de").unwrap(),
            "Example Industrial GmbH",
            &current,
        )
        .expect("fresh cached result");
        assert_eq!(result.classification, "succeeded");
        assert_eq!(result.fields.len(), 1);
        assert_eq!(
            result.reason.as_deref(),
            Some("fresh_cached_success_after_temporary_unreachable")
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn registry_database_resolves_release_bound_success_output() {
        let root = std::env::temp_dir().join(format!(
            "ctox-scrape-cache-registry-{}-{}",
            std::process::id(),
            now_ms()
        ));
        let release_run = root
            .join("releases/release-previous/runtime/scraping/targets/northdata-de/runs")
            .join("scrape_run-registry");
        std::fs::create_dir_all(release_run.join("outputs")).unwrap();
        std::fs::write(
            release_run.join("outputs/records.json"),
            serde_json::to_vec(&json!([{
                "field": "firma_name",
                "value": "Example Industrial GmbH",
                "confidence": "high",
                "source_url": "https://www.northdata.de/Example+Industrial+GmbH",
                "run_id": "scrape_run-registry",
                "source_id": "northdata.de",
                "company_name": "Example Industrial GmbH",
                "evidence_eligible": true,
                "verification_status": "verified",
                "http_status": 200,
                "checked_at": now_ms(),
                "snapshot_hash": "sha256:registry"
            }]))
            .unwrap(),
        )
        .unwrap();
        std::fs::create_dir_all(&root).unwrap();
        let connection = Connection::open(root.join("ctox.sqlite3")).unwrap();
        connection
            .execute_batch(
                "CREATE TABLE scrape_target (target_id TEXT PRIMARY KEY, target_key TEXT NOT NULL);
                 CREATE TABLE scrape_run (
                   run_id TEXT PRIMARY KEY,
                   target_id TEXT NOT NULL,
                   started_at TEXT NOT NULL,
                   finished_at TEXT,
                   status TEXT NOT NULL,
                   output_dir TEXT
                 );",
            )
            .unwrap();
        let now = chrono::Utc::now().to_rfc3339();
        connection
            .execute(
                "INSERT INTO scrape_target (target_id, target_key) VALUES (?1, ?2)",
                params!["target-northdata", "northdata-de"],
            )
            .unwrap();
        connection
            .execute(
                "INSERT INTO scrape_run
                 (run_id, target_id, started_at, finished_at, status, output_dir)
                 VALUES (?1, ?2, ?3, ?4, 'succeeded', ?5)",
                params![
                    "scrape_run-registry",
                    "target-northdata",
                    &now,
                    &now,
                    release_run.to_string_lossy().as_ref()
                ],
            )
            .unwrap();
        drop(connection);

        let result = recent_successful_result(
            &root,
            "northdata-de",
            crate::sources::find("northdata.de").unwrap(),
            "Example Industrial GmbH",
            &stub_result("blocked"),
        )
        .expect("registry output should be reusable across release paths");
        assert_eq!(result.classification, "succeeded");
        assert_eq!(result.run_id.as_deref(), Some("scrape_run-registry"));
        assert_eq!(result.fields.len(), 1);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn stale_success_is_not_reused() {
        let root = std::env::temp_dir().join(format!(
            "ctox-scrape-cache-stale-{}-{}",
            std::process::id(),
            now_ms()
        ));
        let stale = now_ms().saturating_sub(MAX_EVIDENCE_AGE_MS + 1_000);
        write_cached_run(&root, stale, stale);
        assert!(recent_successful_result(
            &root,
            "northdata-de",
            crate::sources::find("northdata.de").unwrap(),
            "Example Industrial GmbH",
            &stub_result("blocked"),
        )
        .is_none());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn real_scrape_manifest_shape_uses_verified_record_identity_and_file_time() {
        let root = std::env::temp_dir().join(format!(
            "ctox-scrape-cache-real-shape-{}-{}",
            std::process::id(),
            now_ms()
        ));
        write_cached_run(&root, now_ms(), now_ms());
        let manifest = root.join("scraping/targets/northdata-de/runs/scrape_run-cached/run.json");
        std::fs::write(
            &manifest,
            serde_json::to_vec(&json!({
                "run_id": "scrape_run-cached",
                "target_key": "northdata-de",
                "status": "succeeded",
                "result": { "records_found": 1 }
            }))
            .unwrap(),
        )
        .unwrap();
        let result = recent_successful_result(
            &root,
            "northdata-de",
            crate::sources::find("northdata.de").unwrap(),
            "Example Industrial GmbH",
            &stub_result("blocked"),
        )
        .expect("real scrape manifest should be reusable");
        assert_eq!(result.classification, "succeeded");
        assert_eq!(result.fields.len(), 1);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn embedded_last_success_uses_verified_snapshot_and_ignores_legal_form_tokens() {
        let root = std::env::temp_dir().join(format!(
            "ctox-scrape-cache-embedded-{}-{}",
            std::process::id(),
            now_ms()
        ));
        let current_run = root.join("scraping/targets/moneyhouse-ch/runs/scrape_run-current");
        let records_path =
            root.join("archive/scraping/targets/moneyhouse-ch/state/latest_records.json");
        std::fs::create_dir_all(&current_run).unwrap();
        std::fs::create_dir_all(records_path.parent().unwrap()).unwrap();
        std::fs::write(
            &records_path,
            serde_json::to_vec(&json!([{
                "field": "firma_name",
                "value": "Nests Sàrl",
                "confidence": "high",
                "source_url": "https://www.moneyhouse.ch/de/company/nests-sarl-2272997971",
                "run_id": "scrape_run-previous",
                "source_id": "moneyhouse.ch",
                "evidence_eligible": true,
                "verification_status": "verified",
                "http_status": 200,
                "checked_at": now_ms(),
                "snapshot_hash": "sha256:embedded"
            }]))
            .unwrap(),
        )
        .unwrap();
        std::fs::write(
            current_run.join("run.json"),
            serde_json::to_vec(&json!({
                "run_id": "scrape_run-current",
                "target_key": "moneyhouse-ch",
                "status": "temporary_unreachable",
                "run_context": {
                    "last_successful_run": {
                        "run_id": "scrape_run-previous",
                        "finished_at": chrono::Utc::now().to_rfc3339(),
                        "result": {
                            "materialization": {
                                "target_key": "moneyhouse-ch",
                                "latest_records_path": records_path
                            }
                        }
                    }
                }
            }))
            .unwrap(),
        )
        .unwrap();
        let result = recent_successful_result(
            &root,
            "moneyhouse-ch",
            crate::sources::find("moneyhouse.ch").unwrap(),
            "Nests Sarl",
            &stub_result("temporary_unreachable"),
        )
        .expect("embedded fresh success should be reusable");
        assert_eq!(result.classification, "succeeded");
        assert_eq!(result.fields.len(), 1);
        std::fs::remove_dir_all(root).unwrap();
    }

    fn stub_result(classification: &str) -> ScrapeBridgeResult {
        ScrapeBridgeResult {
            target_key: "northdata.de",
            fields: Vec::new(),
            classification: classification.to_string(),
            reason: None,
            repair_queued: false,
            run_id: Some(format!("scrape_run-{classification}")),
            evidence_rejections: Vec::new(),
            attempts: 1,
            initial_classification: None,
            public_browser_fallback: None,
        }
    }

    fn successful_unlock(request: &PublicUnlockRequest) -> PublicUnlockOutcome {
        PublicUnlockOutcome {
            attempted: true,
            ok: true,
            status: "browser_ready".to_string(),
            source_id: request.source_id.clone(),
            browser_session_id: request.browser_session_id.clone(),
            trigger: request.trigger.clone(),
            target_url: request.target_url.clone(),
            final_url: Some(request.target_url.clone()),
            challenge_detected: false,
            markers: Vec::new(),
            error: None,
            secret_value_in_payload: false,
        }
    }

    #[test]
    fn access_failures_run_one_public_unlock_and_one_adapter_retry() {
        for initial_status in ["blocked", "temporary_unreachable"] {
            let module = crate::sources::find("northdata.de").unwrap();
            let mut executions = 0;
            let mut retry_input = None;
            let mut unlocks = 0;
            let result = run_with_public_browser_fallback(
                module,
                json!({
                    "company": "North Data GmbH",
                    "country": "DE",
                    "source_id": "northdata.de",
                }),
                |input| {
                    executions += 1;
                    if executions == 1 {
                        stub_result(initial_status)
                    } else {
                        retry_input = Some(input.clone());
                        stub_result("succeeded")
                    }
                },
                |request| {
                    unlocks += 1;
                    assert_eq!(request.source_id, "northdata.de");
                    assert_eq!(request.trigger, initial_status);
                    assert_eq!(request.allowed_domains, ["northdata.de"]);
                    assert_eq!(
                        request.browser_session_id,
                        "browser_session_web_stack_public_northdata-de"
                    );
                    successful_unlock(request)
                },
            );

            assert_eq!(executions, 2, "{initial_status}");
            assert_eq!(unlocks, 1, "{initial_status}");
            assert_eq!(result.attempts, 2);
            assert_eq!(
                result.initial_classification.as_deref(),
                Some(initial_status)
            );
            assert_eq!(result.classification, "succeeded");
            assert_eq!(
                result
                    .public_browser_fallback
                    .as_ref()
                    .map(|outcome| outcome.status.as_str()),
                Some("browser_ready")
            );
            let retry_input = retry_input.expect("retry input");
            assert_eq!(
                retry_input.pointer("/web_fallback/trigger"),
                Some(&json!(initial_status))
            );
            assert_eq!(
                retry_input.pointer("/web_fallback/secret_value_in_payload"),
                Some(&json!(false))
            );
            assert_eq!(
                retry_input.pointer("/browser_session_id"),
                Some(&json!("browser_session_web_stack_public_northdata-de"))
            );
        }
    }

    #[test]
    fn non_access_failures_never_open_the_browser_fallback() {
        for status in ["succeeded", "portal_drift", "partial_output"] {
            let module = crate::sources::find("northdata.de").unwrap();
            let mut executions = 0;
            let mut unlocks = 0;
            let result = run_with_public_browser_fallback(
                module,
                json!({}),
                |_| {
                    executions += 1;
                    stub_result(status)
                },
                |request| {
                    unlocks += 1;
                    successful_unlock(request)
                },
            );

            assert_eq!(executions, 1, "{status}");
            assert_eq!(unlocks, 0, "{status}");
            assert_eq!(result.attempts, 1);
            assert!(result.initial_classification.is_none());
            assert!(result.public_browser_fallback.is_none());
        }
    }

    #[test]
    fn failed_public_unlock_does_not_retry_the_adapter() {
        let module = crate::sources::find("companyhouse.de").unwrap();
        let mut executions = 0;
        let mut unlocks = 0;
        let result = run_with_public_browser_fallback(
            module,
            json!({}),
            |_| {
                executions += 1;
                stub_result("blocked")
            },
            |request| {
                unlocks += 1;
                PublicUnlockOutcome {
                    attempted: true,
                    ok: false,
                    status: "challenge_persisted".to_string(),
                    source_id: request.source_id.clone(),
                    browser_session_id: request.browser_session_id.clone(),
                    trigger: request.trigger.clone(),
                    target_url: request.target_url.clone(),
                    final_url: Some(request.target_url.clone()),
                    challenge_detected: true,
                    markers: vec!["captcha".to_string()],
                    error: Some("challenge_persisted".to_string()),
                    secret_value_in_payload: false,
                }
            },
        );

        assert_eq!(executions, 1);
        assert_eq!(unlocks, 1);
        assert_eq!(result.attempts, 1);
        assert_eq!(result.initial_classification.as_deref(), Some("blocked"));
        assert_eq!(
            result
                .public_browser_fallback
                .as_ref()
                .map(|outcome| outcome.status.as_str()),
            Some("challenge_persisted")
        );
    }

    #[test]
    fn parse_envelope_decodes_typed_records() {
        let envelope = json!({
            "ok": true,
            "run_id": "scrape_run-abc",
            "classification": { "status": "succeeded", "reason": "" },
            "records": [
                {
                    "field": "firma_name",
                    "value": "Roche Holding AG",
                    "confidence": "high",
                    "source_url": "https://www.zefix.ch/en/search/entity/list/firm/1",
                    "run_id": "scrape_run-abc",
                    "source_id": "zefix.ch",
                    "company_name": "Roche Holding AG",
                    "evidence_eligible": true,
                    "verification_status": "verified",
                    "http_status": 200,
                    "checked_at": now_ms(),
                    "snapshot_hash": "sha256:abc",
                    "note": "h1 selector"
                },
                {
                    "field": "person_funktion",
                    "value": "Mitglied des Verwaltungsrates",
                    "confidence": "medium",
                    "source_url": "https://www.zefix.ch/en/search/entity/list/firm/1",
                    "run_id": "scrape_run-abc",
                    "source_id": "zefix.ch",
                    "company_name": "Roche Holding AG",
                    "evidence_eligible": true,
                    "verification_status": "verified",
                    "http_status": 200,
                    "checked_at": now_ms(),
                    "snapshot_hash": "sha256:def"
                },
                {
                    "field": "person_email_validation",
                    "value": "valid",
                    "confidence": "high",
                    "source_url": "https://www.experte.de/email-pruefen"
                },
                {
                    "field": "umsatz",
                    "value": "",
                    "confidence": "high"
                }
            ]
        });
        let result = parse_scrape_envelope(
            "zefix.ch",
            crate::sources::find("zefix.ch").unwrap(),
            "Roche Holding AG",
            &envelope,
        );
        assert_eq!(result.classification, "succeeded");
        assert_eq!(result.run_id.as_deref(), Some("scrape_run-abc"));
        assert!(!result.repair_queued);
        // Empty values and cross-source records without the required run,
        // source, identity, and evidence gate are dropped.
        assert_eq!(result.fields.len(), 2);
        let (firma_key, firma_ev) = &result.fields[0];
        assert_eq!(*firma_key, FieldKey::FirmaName);
        assert_eq!(firma_ev.value, "Roche Holding AG");
        assert_eq!(firma_ev.confidence, Confidence::High);
        assert_eq!(firma_ev.note.as_deref(), Some("h1 selector"));
        assert!(result
            .evidence_rejections
            .iter()
            .any(|reason| reason == "record_2:missing_record_run_id"));
    }

    #[test]
    fn parse_envelope_marks_drift_with_repair_queue() {
        let envelope = json!({
            "ok": true,
            "run_id": "scrape_run-drift",
            "classification": { "status": "portal_drift", "reason": "reachable_empty_output" },
            "records": [],
            "repair_queue_task": {
                "queue_item_id": "queue_abc",
                "thread_key": "scrape/northdata.de"
            }
        });
        let result = parse_scrape_envelope(
            "northdata.de",
            crate::sources::find("northdata.de").unwrap(),
            "Northdata AG",
            &envelope,
        );
        assert_eq!(result.classification, "portal_drift");
        assert!(result.repair_queued);
        assert!(result.fields.is_empty());
        assert_eq!(result.reason.as_deref(), Some("reachable_empty_output"));
    }

    #[test]
    fn parse_envelope_handles_alt_record_shapes() {
        // Some scripts emit under result.records (legacy shape from
        // task-contracts.md).
        let envelope = json!({
            "ok": true,
            "classification": { "status": "succeeded" },
            "result": {
                "records": [
                    {
                        "field": "firma_ort",
                        "value": "Basel",
                        "confidence": "high",
                        "source_url": "https://www.zefix.ch/en/search/entity/list/firm/1",
                        "run_id": "scrape_run-alt",
                        "source_id": "zefix.ch",
                        "company_name": "Roche Holding AG",
                        "evidence_eligible": true,
                        "verification_status": "verified",
                        "http_status": 200,
                        "checked_at": now_ms(),
                        "snapshot_hash": "sha256:ghi"
                    }
                ]
            }
        });
        let mut envelope = envelope;
        envelope["ok"] = json!(true);
        envelope["run_id"] = json!("scrape_run-alt");
        let result = parse_scrape_envelope(
            "zefix.ch",
            crate::sources::find("zefix.ch").unwrap(),
            "Roche Holding AG",
            &envelope,
        );
        assert_eq!(result.fields.len(), 1);
        assert_eq!(result.fields[0].0, FieldKey::FirmaOrt);
    }

    #[test]
    fn parse_envelope_rejects_failed_runs_and_adversarial_records() {
        let statuses = ["blocked", "portal_drift", "partial_output"];
        for status in statuses {
            let envelope = json!({
                "ok": true,
                "run_id": "scrape_run-failed",
                "classification": { "status": status },
                "records": [{
                    "field": "firma_name",
                    "value": "ACME GmbH",
                    "source_url": "https://www.zefix.ch/en/search/entity/list/firm/1",
                    "run_id": "scrape_run-failed",
                    "source_id": "zefix.ch",
                    "company_name": "ACME GmbH",
                    "evidence_eligible": true,
                    "verification_status": "verified",
                    "http_status": 200,
                    "checked_at": now_ms(),
                    "snapshot_hash": "sha256:failed"
                }]
            });
            let result = parse_scrape_envelope(
                "zefix.ch",
                crate::sources::find("zefix.ch").unwrap(),
                "ACME GmbH",
                &envelope,
            );
            assert!(
                result.fields.is_empty(),
                "status {status} must not emit evidence"
            );
            assert!(!result.evidence_rejections.is_empty());
        }

        let base = json!({
            "field": "firma_name",
            "value": "ACME GmbH",
            "source_url": "https://www.zefix.ch/en/search/entity/list/firm/1",
            "run_id": "scrape_run-adversarial",
            "source_id": "zefix.ch",
            "company_name": "ACME GmbH",
            "evidence_eligible": true,
            "verification_status": "verified",
            "http_status": 200,
            "checked_at": now_ms(),
            "snapshot_hash": "sha256:good"
        });
        for mutation in [
            "missing_url",
            "wrong_run",
            "wrong_source",
            "stale",
            "blocked_gate",
        ] {
            let mut record = base.clone();
            match mutation {
                "missing_url" => {
                    record["source_url"] = Value::Null;
                }
                "wrong_run" => {
                    record["run_id"] = json!("scrape_run-other");
                }
                "wrong_source" => {
                    record["source_id"] = json!("evil.example");
                }
                "stale" => {
                    record["checked_at"] = json!(1_u64);
                }
                "blocked_gate" => {
                    record["evidence_eligible"] = json!(false);
                }
                _ => unreachable!(),
            }
            let envelope = json!({
                "ok": true,
                "run_id": "scrape_run-adversarial",
                "classification": { "status": "succeeded" },
                "records": [record]
            });
            let result = parse_scrape_envelope(
                "zefix.ch",
                crate::sources::find("zefix.ch").unwrap(),
                "ACME GmbH",
                &envelope,
            );
            assert!(
                result.fields.is_empty(),
                "mutation {mutation} must be rejected"
            );
            assert!(result
                .evidence_rejections
                .iter()
                .any(|reason| reason.starts_with("record_")));
        }
    }
}
