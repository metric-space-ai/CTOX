use crate::browser_agent_bridge::browser_agent_runtime_config;
use crate::browser_agent_bridge::create_browser_agent_job;
use crate::browser_agent_bridge::wait_for_browser_agent_job;
use crate::contracts::Paths;
use crate::contracts::load_browser_subworker_policy;
use crate::contracts::now_iso;
use crate::runtime_db::TaskRecord;
use crate::runtime_db::WorkerJobRecord;
use crate::runtime_db::claim_next_worker_job;
use crate::runtime_db::emit_worker_review_task;
use crate::runtime_db::enqueue_internal_task;
use crate::runtime_db::finalize_worker_job_for_review;
use crate::runtime_db::load_task_by_id;
use crate::runtime_db::record_agent_event;
use crate::storage::save_json;
use anyhow::Context;
use serde::Deserialize;
use serde_json::Value;
use serde_json::json;
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Clone)]
struct WorkerExecutionOutcome {
    result_summary: String,
    result_detail: String,
    artifact_paths: Vec<String>,
    spawned_task_ids: Vec<i64>,
}

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
struct WorkerBrowserContract {
    objective: Option<String>,
    target_url: Option<String>,
    capability_title: Option<String>,
    #[serde(default)]
    repeated_task: bool,
    #[serde(default)]
    train_specialist_model: bool,
    #[serde(default)]
    request_repair: bool,
    bridge_kind: Option<String>,
    browser_action: Option<WorkerBrowserActionSpec>,
    runtime_config: Option<Value>,
    recipe_payload: Option<Value>,
    task_spec: Option<Value>,
    code: Option<String>,
    timeout_ms: Option<u64>,
    workspace_path_hint: Option<String>,
    failing_tool: Option<String>,
    error_text: Option<String>,
    #[serde(default)]
    patch_targets: Vec<String>,
    #[serde(default)]
    validation_targets: Vec<String>,
    coding_prompt: Option<String>,
    note: Option<String>,
    handoff_artifact_path: Option<String>,
    preferred_model: Option<String>,
    dataset_contract: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
struct WorkerBrowserActionSpec {
    action: String,
    url: Option<String>,
    output_path: Option<String>,
    wait_ms: Option<u64>,
    width: Option<u32>,
    height: Option<u32>,
    justification: Option<String>,
}

pub fn advance_browser_subworkers(
    paths: &Paths,
    max_items: usize,
) -> anyhow::Result<Vec<WorkerJobRecord>> {
    let mut completed = Vec::new();
    for _ in 0..max_items {
        let Some(job) = claim_next_worker_job(paths)? else {
            break;
        };
        let outcome = match execute_worker_job(paths, &job) {
            Ok(outcome) => outcome,
            Err(err) => WorkerExecutionOutcome {
                result_summary: format!(
                    "Worker {} konnte {} nicht sauber abschliessen",
                    job.worker_kind, job.contract_title
                ),
                result_detail: format!(
                    "Der Worker-Lauf ist fehlgeschlagen und braucht Review.\n\nVertrag: {}\nFehler: {}",
                    job.contract_detail,
                    err
                ),
                artifact_paths: Vec::new(),
                spawned_task_ids: Vec::new(),
            },
        };
        let review_task = emit_worker_review_task(
            paths,
            &job,
            &outcome.result_summary,
            &outcome.result_detail,
        )?;
        let updated = finalize_worker_job_for_review(
            paths,
            job.id,
            &outcome.result_summary,
            &outcome.result_detail,
            review_task.id,
        )?;
        let _ = record_agent_event(
            paths,
            "worker/runtimeCompleted",
            Some(updated.parent_task_id),
            &updated.parent_task_title,
            &format!(
                "Worker-Job {} ({}) wurde ausgefuehrt und als Review zurueckgestellt.",
                updated.id, updated.worker_kind
            ),
            &serde_json::to_string(&json!({
                "workerJobId": updated.id,
                "workerKind": updated.worker_kind,
                "reviewTaskId": review_task.id,
                "artifactPaths": outcome.artifact_paths,
                "spawnedTaskIds": outcome.spawned_task_ids,
            }))
            .unwrap_or_else(|_| "{}".to_string()),
        );
        completed.push(updated);
    }
    Ok(completed)
}

fn execute_worker_job(paths: &Paths, job: &WorkerJobRecord) -> anyhow::Result<WorkerExecutionOutcome> {
    match job.worker_kind.as_str() {
        "browser_agent" => execute_browser_agent_job(paths, job),
        "repair_agent" => execute_repair_agent_job(paths, job),
        "specialist_worker" => execute_specialist_worker_job(paths, job),
        other => Ok(WorkerExecutionOutcome {
            result_summary: format!("Worker {} hat einen generischen Vertrag abgelegt", other),
            result_detail: format!(
                "Fuer den Worker-Kind `{}` ist noch kein spezieller Executor hinterlegt.\n\nContract Title: {}\nContract Detail:\n{}\n\nBitte im Review entscheiden, ob der Vertrag auf browser_agent, repair_agent oder specialist_worker umgestellt werden soll.",
                other,
                job.contract_title,
                job.contract_detail
            ),
            artifact_paths: Vec::new(),
            spawned_task_ids: Vec::new(),
        }),
    }
}

fn execute_browser_agent_job(
    paths: &Paths,
    job: &WorkerJobRecord,
) -> anyhow::Result<WorkerExecutionOutcome> {
    let policy = load_browser_subworker_policy(paths);
    let contract = parse_browser_contract(job);
    let artifact_dir = ensure_job_artifact_dir(paths, job)?;
    let objective = contract
        .objective
        .clone()
        .unwrap_or_else(|| job.contract_title.clone());
    let capability_title = contract
        .capability_title
        .clone()
        .unwrap_or_else(|| "browser_capability".to_string());
    let mut artifact_paths = Vec::new();
    let mut spawned_task_ids = Vec::new();
    let mut detail_blocks = Vec::new();
    let mut action_error: Option<String> = None;
    let mut repair_requested_by_extension = false;

    let bridge_request = build_browser_agent_bridge_request(
        paths,
        job,
        &contract,
        &objective,
        &capability_title,
    );
    let bridge_request_path = artifact_dir.join("browser-agent-request.json");
    save_json(&bridge_request_path, &bridge_request)?;
    artifact_paths.push(bridge_request_path.display().to_string());

    let bridge_job = create_browser_agent_job(paths, bridge_request.clone())?;
    detail_blocks.push(format!(
        "Browser-Agent-Bridge-Job `{}` wurde fuer die Chrome-Extension eingereiht.",
        bridge_job.job_id
    ));

    let bridge_timeout_ms = contract.timeout_ms.unwrap_or(120_000);
    match wait_for_browser_agent_job(paths, &bridge_job.job_id, bridge_timeout_ms)? {
        Some(bridge_result)
            if matches!(bridge_result.status.as_str(), "completed" | "failed") =>
        {
            let result_artifact_path = artifact_dir.join("browser-agent-result.json");
            save_json(&result_artifact_path, &bridge_result)?;
            artifact_paths.push(result_artifact_path.display().to_string());
            let summary = bridge_result
                .result
                .as_ref()
                .and_then(|value| value.get("summary"))
                .and_then(Value::as_str)
                .unwrap_or("");
            detail_blocks.push(format!(
                "Chrome-Extension-Job endete mit Status `{}`{}",
                bridge_result.status,
                if summary.trim().is_empty() {
                    String::new()
                } else {
                    format!(": {}", summary.trim())
                }
            ));
            if bridge_result.status == "failed" {
                action_error = bridge_result
                    .result
                    .as_ref()
                    .and_then(|value| value.get("error"))
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned)
                    .or_else(|| Some("Browser-Agent-Extension meldete Fehler.".to_string()));
            }

            if let Some(repair_contract) = bridge_result
                .result
                .as_ref()
                .and_then(extract_bridge_repair_contract)
                .map(|repair| merge_worker_browser_contract(&contract, &repair))
            {
                repair_requested_by_extension = true;
                let repair_task = queue_workspace_repair_from_contract(
                    paths,
                    job,
                    &repair_contract,
                    action_error.clone(),
                    artifact_dir.join("coding-handoff.json"),
                    artifact_paths.clone(),
                )?;
                artifact_paths.push(repair_task.1.clone());
                spawned_task_ids.push(repair_task.0);
                detail_blocks.push(format!(
                    "Chrome-Extension hat CTO-Repair-Task #{} angefordert.",
                    repair_task.0
                ));
            }
        }
        Some(bridge_result) => {
            detail_blocks.push(format!(
                "Browser-Agent-Bridge-Job `{}` blieb auf Status `{}` stehen.",
                bridge_result.job_id, bridge_result.status
            ));
            action_error = Some(format!(
                "browser agent job {} did not reach terminal state before timeout",
                bridge_result.job_id
            ));
        }
        None => {
            detail_blocks.push(
                "Browser-Agent-Bridge-Job konnte nicht wieder geladen werden.".to_string(),
            );
            action_error = Some("browser agent bridge job vanished".to_string());
        }
    }

    let should_request_repair = contract.request_repair
        || contract.coding_prompt.is_some()
        || !contract.patch_targets.is_empty()
        || !contract.validation_targets.is_empty();
    if should_request_repair && !repair_requested_by_extension {
        let repair_task = queue_workspace_repair_from_contract(
            paths,
            job,
            &contract,
            action_error.clone(),
            artifact_dir.join("coding-handoff.json"),
            artifact_paths.clone(),
        )?;
        artifact_paths.push(repair_task.1.clone());
        spawned_task_ids.push(repair_task.0);
        detail_blocks.push(format!(
            "CTO-Repair-Task #{} wurde fuer Browserdiagnose eingereiht.",
            repair_task.0
        ));
    }

    if contract.repeated_task || contract.train_specialist_model {
        let accepted_record_path = resolve_root_relative_path(
            paths,
            &policy.specialist_runtime.accepted_records_dir,
        )
        .join(format!("job-{}-accepted-record.json", job.id));
        let training_request_path = resolve_root_relative_path(
            paths,
            &policy.specialist_runtime.training_requests_dir,
        )
        .join(format!("job-{}-training-request.json", job.id));
        let specialist_task = queue_specialist_factory_request(
            paths,
            job,
            &contract,
            accepted_record_path,
            training_request_path,
            &policy.specialist_runtime.preferred_small_model,
            &policy.specialist_runtime.dataset_contract,
            &objective,
            &capability_title,
            artifact_paths.clone(),
        )?;
        artifact_paths.push(specialist_task.1.clone());
        artifact_paths.push(specialist_task.2.clone());
        spawned_task_ids.push(specialist_task.0);
        detail_blocks.push(format!(
            "Specialist-Fabrik-Task #{} fuer wiederkehrende Browserfaehigkeit eingereiht.",
            specialist_task.0
        ));
    }

    if let Some(note) = contract.note.as_deref().filter(|value| !value.trim().is_empty()) {
        detail_blocks.push(format!("Notiz: {}", note.trim()));
    }

    Ok(WorkerExecutionOutcome {
        result_summary: format!(
            "Browser-Agent hat `{}` bearbeitet{}{}",
            objective,
            if spawned_task_ids.is_empty() { "" } else { " und Folgearbeit erzeugt" },
            if action_error.is_some() { " (mit Fehlerdiagnose)" } else { "" },
        ),
        result_detail: format!(
            "Browser-Agent Policy: {}\n\nObjective: {}\nCapability: {}\nTarget URL: {}\n\n{}",
            policy.browser_agent.role,
            objective,
            capability_title,
            contract.target_url.as_deref().unwrap_or("none"),
            detail_blocks.join("\n\n"),
        ),
        artifact_paths,
        spawned_task_ids,
    })
}

fn execute_repair_agent_job(
    paths: &Paths,
    job: &WorkerJobRecord,
) -> anyhow::Result<WorkerExecutionOutcome> {
    let contract = parse_browser_contract(job);
    let artifact_dir = ensure_job_artifact_dir(paths, job)?;
    let error_text = contract.error_text.clone();
    let repair_task = queue_workspace_repair_from_contract(
        paths,
        job,
        &contract,
        error_text,
        artifact_dir.join("repair-handoff.json"),
        Vec::new(),
    )?;
    Ok(WorkerExecutionOutcome {
        result_summary: format!(
            "Repair-Agent hat Repair-Handoff fuer `{}` an den CTO-Agenten uebergeben",
            contract
                .objective
                .as_deref()
                .unwrap_or(job.contract_title.as_str())
        ),
        result_detail: format!(
            "Der Repair-Agent hat keine Root-Patches selbst ausgefuehrt, sondern CTO-eigene Reparaturarbeit eingereiht.\n\nWorkspace Repair Task: #{}\nHandoff Artifact: {}",
            repair_task.0, repair_task.1
        ),
        artifact_paths: vec![repair_task.1],
        spawned_task_ids: vec![repair_task.0],
    })
}

fn execute_specialist_worker_job(
    paths: &Paths,
    job: &WorkerJobRecord,
) -> anyhow::Result<WorkerExecutionOutcome> {
    let policy = load_browser_subworker_policy(paths);
    let contract = parse_browser_contract(job);
    let objective = contract
        .objective
        .clone()
        .unwrap_or_else(|| job.contract_title.clone());
    let capability_title = contract
        .capability_title
        .clone()
        .unwrap_or_else(|| "browser_capability".to_string());
    let accepted_record_path = resolve_root_relative_path(
        paths,
        &policy.specialist_runtime.accepted_records_dir,
    )
    .join(format!("job-{}-accepted-record.json", job.id));
    let training_request_path = resolve_root_relative_path(
        paths,
        &policy.specialist_runtime.training_requests_dir,
    )
    .join(format!("job-{}-training-request.json", job.id));
    let specialist_task = queue_specialist_factory_request(
        paths,
        job,
        &contract,
        accepted_record_path,
        training_request_path,
        contract
            .preferred_model
            .as_deref()
            .unwrap_or(&policy.specialist_runtime.preferred_small_model),
        contract
            .dataset_contract
            .as_deref()
            .unwrap_or(&policy.specialist_runtime.dataset_contract),
        &objective,
        &capability_title,
        Vec::new(),
    )?;
    Ok(WorkerExecutionOutcome {
        result_summary: format!(
            "Specialist-Worker hat eine Trainingsanfrage fuer `{}` vorbereitet",
            objective
        ),
        result_detail: format!(
            "Ein kontrollierter Specialist-Fabrik-Pfad wurde eingereiht.\n\nFactory Task: #{}\nAccepted Record: {}\nTraining Request: {}\nTarget Model: {}",
            specialist_task.0,
            specialist_task.1,
            specialist_task.2,
            contract
                .preferred_model
                .as_deref()
                .unwrap_or(&policy.specialist_runtime.preferred_small_model)
        ),
        artifact_paths: vec![specialist_task.1, specialist_task.2],
        spawned_task_ids: vec![specialist_task.0],
    })
}

fn queue_workspace_repair_from_contract(
    paths: &Paths,
    job: &WorkerJobRecord,
    contract: &WorkerBrowserContract,
    action_error: Option<String>,
    artifact_path: PathBuf,
    supporting_artifacts: Vec<String>,
) -> anyhow::Result<(i64, String)> {
    let objective = contract
        .objective
        .clone()
        .unwrap_or_else(|| job.contract_title.clone());
    let prompt = build_repair_prompt(job, contract, action_error.as_deref());
    let workspace_hint = contract
        .workspace_path_hint
        .clone()
        .unwrap_or_else(|| paths.root.display().to_string());
    save_json(
        &artifact_path,
        &json!({
            "format": "patch_handoff_v1",
            "createdAt": now_iso(),
            "workerJobId": job.id,
            "workerKind": job.worker_kind,
            "objective": objective,
            "repoPathHint": workspace_hint,
            "patchTargets": contract.patch_targets.clone(),
            "validationTargets": contract.validation_targets.clone(),
            "failingTool": contract.failing_tool.clone(),
            "errorText": action_error.or_else(|| contract.error_text.clone()),
            "prompt": prompt,
            "supportingArtifacts": supporting_artifacts,
            "sourceHandoffPath": contract.handoff_artifact_path.clone(),
        }),
    )?;
    let detail = format!(
        "Browserdiagnose hat eine CTO-eigene Reparaturaufgabe erzeugt.\n\nObjective: {}\nWorkspace: {}\nPatch Targets: {}\nValidation Targets: {}\nHandoff Artifact: {}\n\nRepair Prompt:\n{}",
        objective,
        workspace_hint,
        join_or_none(&contract.patch_targets),
        join_or_none(&contract.validation_targets),
        artifact_path.display(),
        prompt,
    );
    let task = enqueue_internal_task(
        paths,
        Some(job.parent_task_id),
        "workspace_repair",
        &format!("Browser-Repair fuer {}", short_label(&objective, 72)),
        &detail,
        930,
    )?;
    Ok((task.id, artifact_path.display().to_string()))
}

#[allow(clippy::too_many_arguments)]
fn queue_specialist_factory_request(
    paths: &Paths,
    job: &WorkerJobRecord,
    contract: &WorkerBrowserContract,
    accepted_record_path: PathBuf,
    training_request_path: PathBuf,
    preferred_model: &str,
    dataset_contract: &str,
    objective: &str,
    capability_title: &str,
    artifact_paths: Vec<String>,
) -> anyhow::Result<(i64, String, String)> {
    save_json(
        &accepted_record_path,
        &json!({
            "format": "browser_accepted_record_v1",
            "createdAt": now_iso(),
            "workerJobId": job.id,
            "objective": objective,
            "capabilityTitle": capability_title,
            "targetUrl": contract.target_url.clone(),
            "artifacts": artifact_paths,
            "note": contract.note.clone(),
        }),
    )?;
    save_json(
        &training_request_path,
        &json!({
            "format": "browser_specialist_training_request_v1",
            "createdAt": now_iso(),
            "workerJobId": job.id,
            "objective": objective,
            "capabilityTitle": capability_title,
            "preferredSmallModel": preferred_model,
            "datasetContract": dataset_contract,
            "acceptedRecordPaths": [accepted_record_path.display().to_string()],
            "requestedBy": job.worker_kind,
            "sourceContractTitle": job.contract_title,
        }),
    )?;
    let detail = format!(
        "Wiederkehrende Browserarbeit soll in die kontrollierte Specialist-Fabrik gehen.\n\nObjective: {}\nCapability: {}\nPreferred Small Model: {}\nDataset Contract: {}\nAccepted Record: {}\nTraining Request: {}",
        objective,
        capability_title,
        preferred_model,
        dataset_contract,
        accepted_record_path.display(),
        training_request_path.display(),
    );
    let task = enqueue_internal_task(
        paths,
        Some(job.parent_task_id),
        "specialist_model_factory",
        &format!("Browser-Specialist fuer {}", short_label(objective, 72)),
        &detail,
        760,
    )?;
    Ok((
        task.id,
        accepted_record_path.display().to_string(),
        training_request_path.display().to_string(),
    ))
}

fn load_parent_task_or_fallback(paths: &Paths, job: &WorkerJobRecord) -> anyhow::Result<TaskRecord> {
    if let Some(task) = load_task_by_id(paths, job.parent_task_id)? {
        return Ok(task);
    }
    Ok(TaskRecord {
        id: job.parent_task_id,
        created_at: now_iso(),
        updated_at: now_iso(),
        parent_task_id: None,
        worker_job_id: Some(job.id),
        source_interrupt_id: None,
        source_channel: "worker_runtime".to_string(),
        speaker: format!("worker::{}", job.worker_kind),
        task_kind: "delegated_worker_context".to_string(),
        title: job.parent_task_title.clone(),
        detail: job.contract_detail.clone(),
        trust_level: "system".to_string(),
        priority_score: 0,
        status: "await_review".to_string(),
        run_count: 0,
        last_checkpoint_summary: None,
        last_checkpoint_at: None,
        last_output: None,
    })
}

fn parse_browser_contract(job: &WorkerJobRecord) -> WorkerBrowserContract {
    serde_json::from_str::<WorkerBrowserContract>(&job.contract_detail).unwrap_or_else(|_| {
        WorkerBrowserContract {
            objective: Some(job.contract_title.clone()),
            note: Some(job.contract_detail.clone()),
            ..WorkerBrowserContract::default()
        }
    })
}

fn build_browser_agent_bridge_request(
    paths: &Paths,
    job: &WorkerJobRecord,
    contract: &WorkerBrowserContract,
    objective: &str,
    capability_title: &str,
) -> Value {
    let bridge_kind = contract
        .bridge_kind
        .clone()
        .or_else(|| {
            if contract.repeated_task || contract.train_specialist_model {
                Some("browser_capability_craft".to_string())
            } else if contract.code.as_deref().map(str::trim).filter(|value| !value.is_empty()).is_some() {
                Some("browser_action_test".to_string())
            } else {
                Some("browser_collection".to_string())
            }
        })
        .unwrap_or_else(|| "browser_collection".to_string());
    let mut runtime_config = contract
        .runtime_config
        .clone()
        .unwrap_or_else(|| json!({}));
    if let Some(object) = runtime_config.as_object_mut() {
        if let Some(url) = contract
            .target_url
            .as_deref()
            .filter(|value| !value.trim().is_empty())
        {
            object
                .entry("startUrl".to_string())
                .or_insert_with(|| Value::String(url.trim().to_string()));
            object
                .entry("targetUrl".to_string())
                .or_insert_with(|| Value::String(url.trim().to_string()));
            object
                .entry("baseUrl".to_string())
                .or_insert_with(|| Value::String(url.trim().to_string()));
        }
        if let Some(code) = contract
            .code
            .as_deref()
            .filter(|value| !value.trim().is_empty())
        {
            object
                .entry("code".to_string())
                .or_insert_with(|| Value::String(code.trim().to_string()));
        }
        if let Some(spec) = contract.browser_action.as_ref() {
            if let Some(action_url) = spec
                .url
                .as_deref()
                .filter(|value| !value.trim().is_empty())
            {
                object
                    .entry("url".to_string())
                    .or_insert_with(|| Value::String(action_url.trim().to_string()));
            }
        }
        object.insert(
            "capabilityTitle".to_string(),
            Value::String(capability_title.to_string()),
        );
        object.insert(
            "workspacePathHint".to_string(),
            Value::String(
                contract
                    .workspace_path_hint
                    .clone()
                    .unwrap_or_else(|| paths.root.display().to_string()),
            ),
        );
        let bridge_config = browser_agent_runtime_config(paths);
        if let Some(model_ref) = bridge_config
            .get("plannerModelRef")
            .and_then(Value::as_str)
            .filter(|value| !value.trim().is_empty())
        {
            object
                .entry("plannerModelRef".to_string())
                .or_insert_with(|| Value::String(model_ref.to_string()));
        }
        if let Some(model_ref) = bridge_config
            .get("preferredVisionModelRef")
            .and_then(Value::as_str)
            .filter(|value| !value.trim().is_empty())
        {
            object
                .entry("visionModelRef".to_string())
                .or_insert_with(|| Value::String(model_ref.to_string()));
        }
    }

    json!({
        "requestId": format!("worker-job-{}", job.id),
        "kind": bridge_kind,
        "objective": objective,
        "capability_title": capability_title,
        "prompt": contract.note.clone().unwrap_or_else(|| objective.to_string()),
        "description": contract.note.clone().unwrap_or_else(|| job.contract_title.clone()),
        "runtime_config": runtime_config,
        "task_spec": contract.task_spec.clone().unwrap_or_else(|| {
            json!({
                "task_name": job.contract_title.as_str(),
                "task_goal": objective,
            })
        }),
        "recipe_payload": contract.recipe_payload.clone().unwrap_or_else(|| json!({})),
        "code": contract.code.clone(),
        "request_cto_repair": contract.request_repair,
    })
}

fn ensure_job_artifact_dir(paths: &Paths, job: &WorkerJobRecord) -> anyhow::Result<PathBuf> {
    let dir = paths
        .browser_artifacts_dir
        .join("worker-jobs")
        .join(format!("job-{}", job.id));
    fs::create_dir_all(&dir)
        .with_context(|| format!("failed to create {}", dir.display()))?;
    Ok(dir)
}

fn resolve_root_relative_path(paths: &Paths, raw: &str) -> PathBuf {
    let candidate = PathBuf::from(raw);
    if candidate.is_absolute() {
        candidate
    } else {
        paths.root.join(candidate)
    }
}

fn extract_bridge_repair_contract(value: &Value) -> Option<WorkerBrowserContract> {
    value
        .get("repairRequest")
        .cloned()
        .and_then(|payload| serde_json::from_value::<WorkerBrowserContract>(payload).ok())
}

fn merge_worker_browser_contract(
    base: &WorkerBrowserContract,
    overlay: &WorkerBrowserContract,
) -> WorkerBrowserContract {
    WorkerBrowserContract {
        objective: overlay.objective.clone().or_else(|| base.objective.clone()),
        target_url: overlay.target_url.clone().or_else(|| base.target_url.clone()),
        capability_title: overlay
            .capability_title
            .clone()
            .or_else(|| base.capability_title.clone()),
        repeated_task: overlay.repeated_task || base.repeated_task,
        train_specialist_model: overlay.train_specialist_model || base.train_specialist_model,
        request_repair: overlay.request_repair || base.request_repair,
        bridge_kind: overlay.bridge_kind.clone().or_else(|| base.bridge_kind.clone()),
        browser_action: overlay
            .browser_action
            .clone()
            .or_else(|| base.browser_action.clone()),
        runtime_config: overlay
            .runtime_config
            .clone()
            .or_else(|| base.runtime_config.clone()),
        recipe_payload: overlay
            .recipe_payload
            .clone()
            .or_else(|| base.recipe_payload.clone()),
        task_spec: overlay.task_spec.clone().or_else(|| base.task_spec.clone()),
        code: overlay.code.clone().or_else(|| base.code.clone()),
        timeout_ms: overlay.timeout_ms.or(base.timeout_ms),
        workspace_path_hint: overlay
            .workspace_path_hint
            .clone()
            .or_else(|| base.workspace_path_hint.clone()),
        failing_tool: overlay
            .failing_tool
            .clone()
            .or_else(|| base.failing_tool.clone()),
        error_text: overlay.error_text.clone().or_else(|| base.error_text.clone()),
        patch_targets: if overlay.patch_targets.is_empty() {
            base.patch_targets.clone()
        } else {
            overlay.patch_targets.clone()
        },
        validation_targets: if overlay.validation_targets.is_empty() {
            base.validation_targets.clone()
        } else {
            overlay.validation_targets.clone()
        },
        coding_prompt: overlay
            .coding_prompt
            .clone()
            .or_else(|| base.coding_prompt.clone()),
        note: overlay.note.clone().or_else(|| base.note.clone()),
        handoff_artifact_path: overlay
            .handoff_artifact_path
            .clone()
            .or_else(|| base.handoff_artifact_path.clone()),
        preferred_model: overlay
            .preferred_model
            .clone()
            .or_else(|| base.preferred_model.clone()),
        dataset_contract: overlay
            .dataset_contract
            .clone()
            .or_else(|| base.dataset_contract.clone()),
    }
}

fn build_repair_prompt(
    job: &WorkerJobRecord,
    contract: &WorkerBrowserContract,
    action_error: Option<&str>,
) -> String {
    let mut lines = vec![
        format!(
            "Repair the workspace issue behind browser worker job #{}: {}.",
            job.id, job.contract_title
        ),
        format!(
            "Objective: {}",
            contract
                .objective
                .as_deref()
                .unwrap_or(job.contract_title.as_str())
        ),
    ];
    if let Some(workspace) = contract
        .workspace_path_hint
        .as_deref()
        .filter(|value| !value.trim().is_empty())
    {
        lines.push(format!("Workspace hint: {}", workspace.trim()));
    }
    if let Some(tool) = contract
        .failing_tool
        .as_deref()
        .filter(|value| !value.trim().is_empty())
    {
        lines.push(format!("Failing tool: {}", tool.trim()));
    }
    if let Some(error_text) = action_error
        .or(contract.error_text.as_deref())
        .filter(|value| !value.trim().is_empty())
    {
        lines.push(format!("Observed error: {}", error_text.trim()));
    }
    if !contract.patch_targets.is_empty() {
        lines.push(format!(
            "Patch targets: {}",
            contract.patch_targets.join(", ")
        ));
    }
    if !contract.validation_targets.is_empty() {
        lines.push(format!(
            "Validation targets: {}",
            contract.validation_targets.join(", ")
        ));
    }
    if let Some(prompt) = contract
        .coding_prompt
        .as_deref()
        .filter(|value| !value.trim().is_empty())
    {
        lines.push(String::new());
        lines.push("Requested repair prompt:".to_string());
        lines.push(prompt.trim().to_string());
    } else {
        lines.push(String::new());
        lines.push(
            "Diagnose the root cause, patch the workspace, rerun the smallest relevant validation and leave a compact summary."
                .to_string(),
        );
    }
    lines.join("\n")
}

fn short_label(value: &str, max_chars: usize) -> String {
    let trimmed = value.trim();
    if trimmed.chars().count() <= max_chars {
        return trimmed.to_string();
    }
    let shortened = trimmed
        .chars()
        .take(max_chars.saturating_sub(3))
        .collect::<String>();
    format!("{}...", shortened)
}

fn join_or_none(values: &[String]) -> String {
    if values.is_empty() {
        "none".to_string()
    } else {
        values.join(", ")
    }
}
