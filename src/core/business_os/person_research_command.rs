use anyhow::Context;
use ctox_web_stack::sources::{Country, FieldKey, ResearchMode};
use ctox_web_stack::PersonResearchRequest;
use serde::Deserialize;
use serde_json::Value;
use std::collections::HashSet;
use std::panic::{self, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

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
            return store::write_rxdb_failed_control_command_outcome(
                root,
                &command,
                "person_research_start",
                error,
            );
        }
    }
    Ok(running)
}

pub(crate) fn recover_once(root: &Path) -> anyhow::Result<usize> {
    let commands = store::recoverable_person_research_commands(root)?;
    let mut started = 0;
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
            let persisted = match result {
                Ok(Ok(outcome)) => store::write_rxdb_control_command_outcome(
                    &root,
                    &worker_command,
                    "completed",
                    None,
                    Some("completed"),
                    outcome,
                ),
                Ok(Err(error)) => store::write_rxdb_failed_control_command_outcome(
                    &root,
                    &worker_command,
                    "person_research",
                    error,
                ),
                Err(_) => store::write_rxdb_failed_control_command_outcome(
                    &root,
                    &worker_command,
                    "person_research",
                    anyhow::anyhow!("person-research worker panicked"),
                ),
            };
            if let Err(error) = persisted {
                eprintln!(
                    "[business-os] person research `{worker_command_id}` outcome failed: {error:#}"
                );
            }
        });
    if let Err(error) = spawn_result {
        return Err(error.into());
    }
    Ok(true)
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
}
