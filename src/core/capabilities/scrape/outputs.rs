// Post-run outputs: record materialization with insert/update/delete
// deltas, template capture/promotion, and repair-request bundles.

use super::classify::ScrapeRunStatus;
use super::execute::{CommandExecution, ProbeResult};
use super::registry::open_db;
use super::{
    artifact_record, canonical_json, compute_sha256, latest_source_revision_map,
    latest_state_paths_for_target, load_active_record_index, load_latest_active_records_sample,
    load_target_view, now_iso_string, probe_to_json, record_identity_fields, record_identity_key,
    resolve_input_path, resolve_workspace_dir, slugify, stable_digest, tail_excerpt,
    target_sources, MaterializationOutcome, RegisteredTarget, DEFAULT_REPAIR_SKILL,
    MIN_TEMPLATE_CODE_LEN, MIN_TEMPLATE_RESULTS, MIN_TEMPLATE_TARGETS,
};
use anyhow::{Context, Result};
use rusqlite::{params, Connection, OptionalExtension};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

pub(super) fn repair_skill_for_status(status: ScrapeRunStatus) -> &'static str {
    if status == ScrapeRunStatus::Blocked {
        "web-unlock"
    } else {
        DEFAULT_REPAIR_SKILL
    }
}

pub(super) fn record_template_example(
    root: &Path,
    target_key: &str,
    template_key_raw: &str,
    script_file_arg: &str,
    language: &str,
    result_count: Option<i64>,
    challenge_score: i64,
    nomination_reason: Option<&str>,
) -> Result<Value> {
    let conn = open_db(root)?;
    let target = load_target_view(&conn, target_key)?.context("target_key not found")?;
    let script_path = resolve_input_path(root, script_file_arg);
    let script_body = fs::read_to_string(&script_path)
        .with_context(|| format!("failed to read script file {}", script_path.display()))?;
    let script_sha256 = compute_sha256(script_body.trim());
    let template_key = slugify(template_key_raw);
    let now = now_iso_string();
    if let Some((existing_result_count, existing_challenge)) = conn
        .query_row(
            r#"
            SELECT result_count, challenge_score
            FROM scrape_template_example
            WHERE template_key = ?1 AND target_id = ?2 AND script_sha256 = ?3
            "#,
            params![template_key, target.target_id, script_sha256],
            |row| Ok((row.get::<_, Option<i64>>(0)?, row.get::<_, i64>(1)?)),
        )
        .optional()?
    {
        let merged_result_count = match (existing_result_count, result_count) {
            (Some(left), Some(right)) => Some(left.max(right)),
            (Some(left), None) => Some(left),
            (None, Some(right)) => Some(right),
            (None, None) => None,
        };
        conn.execute(
            r#"
            UPDATE scrape_template_example
            SET script_body = ?4,
                language = ?5,
                result_count = ?6,
                challenge_score = ?7,
                nomination_reason = ?8,
                updated_at = ?9
            WHERE template_key = ?1 AND target_id = ?2 AND script_sha256 = ?3
            "#,
            params![
                template_key,
                target.target_id,
                script_sha256,
                script_body,
                language,
                merged_result_count,
                existing_challenge.max(challenge_score.clamp(0, 3)),
                nomination_reason,
                now,
            ],
        )?;
    } else {
        conn.execute(
            r#"
            INSERT INTO scrape_template_example (
                template_key, target_id, script_sha256, script_body, language, result_count,
                challenge_score, nomination_reason, created_at, updated_at
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?9)
            "#,
            params![
                template_key,
                target.target_id,
                script_sha256,
                script_body,
                language,
                result_count,
                challenge_score.clamp(0, 3),
                nomination_reason,
                now,
            ],
        )?;
    }
    let aggregate = conn.query_row(
        r#"
        SELECT
            COUNT(*) AS example_count,
            COUNT(DISTINCT target_id) AS target_count,
            MAX(COALESCE(result_count, 0)) AS best_result_count,
            MAX(challenge_score) AS best_challenge_score
        FROM scrape_template_example
        WHERE template_key = ?1 AND script_sha256 = ?2
        "#,
        params![template_key, script_sha256],
        |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, i64>(1)?,
                row.get::<_, i64>(2)?,
                row.get::<_, i64>(3)?,
            ))
        },
    )?;
    let (promoted, promotion_reason) =
        should_auto_promote_template(&script_body, aggregate.2, aggregate.1, aggregate.3);
    if promoted {
        upsert_promoted_template(
            &conn,
            &template_key,
            &script_sha256,
            &script_body,
            language,
            aggregate.0,
            aggregate.1,
            aggregate.2,
            &promotion_reason,
        )?;
    }
    Ok(json!({
        "template_key": template_key,
        "target_key": target.target_key,
        "script_sha256": script_sha256,
        "example_count": aggregate.0,
        "target_count": aggregate.1,
        "best_result_count": aggregate.2,
        "best_challenge_score": aggregate.3,
        "promoted": promoted,
        "promotion_reason": promotion_reason,
    }))
}

pub(super) fn promote_template(
    root: &Path,
    template_key_raw: &str,
    script_file_arg: &str,
    language: &str,
    reason: &str,
) -> Result<Value> {
    let conn = open_db(root)?;
    let template_key = slugify(template_key_raw);
    let script_path = resolve_input_path(root, script_file_arg);
    let script_body = fs::read_to_string(&script_path)
        .with_context(|| format!("failed to read script file {}", script_path.display()))?;
    let script_sha256 = compute_sha256(script_body.trim());
    let aggregate = conn
        .query_row(
            r#"
            SELECT
                COUNT(*) AS example_count,
                COUNT(DISTINCT target_id) AS target_count,
                MAX(COALESCE(result_count, 0)) AS best_result_count
            FROM scrape_template_example
            WHERE template_key = ?1 AND script_sha256 = ?2
            "#,
            params![template_key, script_sha256],
            |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, i64>(2)?,
                ))
            },
        )
        .unwrap_or((0, 0, 0));
    upsert_promoted_template(
        &conn,
        &template_key,
        &script_sha256,
        &script_body,
        language,
        aggregate.0.max(1),
        aggregate.1.max(1),
        aggregate.2,
        reason,
    )?;
    Ok(json!({
        "template_key": template_key,
        "script_sha256": script_sha256,
        "source_example_count": aggregate.0.max(1),
        "source_target_count": aggregate.1.max(1),
        "best_result_count": aggregate.2,
        "promotion_reason": reason,
    }))
}

pub(super) fn maybe_record_template_from_target(
    root: &Path,
    target: &RegisteredTarget,
    records_found: i64,
) -> Result<Option<Value>> {
    let template_key = target
        .view
        .config
        .get("template_key")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty());
    let Some(template_key) = template_key else {
        return Ok(None);
    };
    let challenge_score = target
        .view
        .config
        .get("template_challenge_score")
        .and_then(Value::as_i64)
        .unwrap_or(0);
    Ok(Some(record_template_example(
        root,
        &target.view.target_key,
        template_key,
        &target.script.script_path,
        &target.script.language,
        Some(records_found),
        challenge_score,
        Some("successful_ctox_execute"),
    )?))
}

pub(super) fn build_repair_prompt(
    repair_workspace: &Path,
    target: &RegisteredTarget,
    run_id: &str,
    status: ScrapeRunStatus,
    records_found: i64,
    repair_request_path: Option<&PathBuf>,
) -> String {
    let repair_request = repair_request_path
        .map(|path| path.to_string_lossy().to_string())
        .unwrap_or_default();
    let source_keys = target_sources(&target.view)
        .into_iter()
        .map(|source| source.source_key)
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "Repair CTOX scrape target `{}` inside the isolated workspace `{}`. Do not modify the workspace parent, the active CTOX release, any `runtime` symlink, or any database file. Read the repair bundle at `{}` first. This was run `{}` with status `{}` and records_found `{}`. The configured sources are: [{}]. Inspect `sources/` and `scripts/` before changing extraction logic. If the failure is real portal drift or partial extraction, revise only files below `scripts/` and/or `sources/<source_key>/`. Register root changes with `ctox scrape register-script --target-key {} --script-file <path> --change-reason script_relearned` and source-local changes with `ctox scrape register-source-module --target-key {} --source-key <source_key> --module-file <path> --change-reason source_relearned`, then rerun `ctox scrape execute --target-key {} --trigger-kind repair --allow-heal`. These three `ctox scrape` commands run from the sandbox through the active daemon, so invoke them unchanged. Do not rewrite the script if the evidence shows only temporary upstream downtime or blocking.",
        target.view.target_key,
        repair_workspace.display(),
        repair_request,
        run_id,
        status.as_str(),
        records_found,
        source_keys,
        target.view.target_key,
        target.view.target_key,
        target.view.target_key,
    )
}

pub(super) fn write_repair_request(
    conn: &Connection,
    run_dir: &Path,
    target: &RegisteredTarget,
    status: ScrapeRunStatus,
    reason: &str,
    probe: &ProbeResult,
    execution: &CommandExecution,
    records_found: i64,
    last_successful_run: Option<&Value>,
    materialization: Option<&MaterializationOutcome>,
    reauthorization: Option<&Value>,
) -> Result<PathBuf> {
    let path = run_dir.join("repair_request.json");
    let latest_state_paths = latest_state_paths_for_target(&target.view);
    let latest_sample = load_latest_active_records_sample(conn, &target.view.target_id, 10)?;
    let source_modules = latest_source_revision_map(conn, &target.view.target_id)?
        .into_values()
        .collect::<Vec<_>>();
    fs::write(
        &path,
        serde_json::to_string_pretty(&json!({
            "target_key": target.view.target_key,
            "display_name": target.view.display_name,
            "start_url": target.view.start_url,
            "status": status.as_str(),
            "reason": reason,
            "probe": probe_to_json(probe),
            "records_found": records_found,
            "reauthorization": reauthorization.cloned().unwrap_or(Value::Null),
            "workspace_dir": target.view.workspace_dir,
            "manifest_path": resolve_workspace_dir(Path::new(""), &target.view.workspace_dir).join("manifest.json"),
            "current_script_path": target.script.script_path,
            "current_revision_no": target.script.revision_no,
            "current_script_sha256": target.script.script_sha256,
            "sources": target_sources(&target.view),
            "source_modules": source_modules,
            "last_successful_run": last_successful_run,
            "latest_state_paths": {
                "latest_records": latest_state_paths.0,
                "latest_summary": latest_state_paths.1,
            },
            "latest_materialized_sample": latest_sample,
            "current_run_materialization": materialization.as_ref().map(|item| item.summary.clone()),
            "stdout_excerpt": tail_excerpt(&execution.stdout_text, 4000),
            "stderr_excerpt": tail_excerpt(&execution.stderr_text, 4000),
        }))?,
    )?;
    Ok(path)
}

pub(super) fn materialize_latest_records(
    conn: &Connection,
    target: &RegisteredTarget,
    run_id: &str,
    finished_at: &str,
    records: &[Value],
    output_dir: &Path,
    default_schema_key: Option<&str>,
) -> Result<MaterializationOutcome> {
    let identity_fields = record_identity_fields(target);
    let schema_key = default_schema_key.map(ToOwned::to_owned).or_else(|| {
        target
            .view
            .output_schema
            .get("schema_key")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned)
    });
    let existing = load_active_record_index(conn, &target.view.target_id)?;
    let mut next_records: BTreeMap<String, (String, Value)> = BTreeMap::new();
    let mut duplicate_keys = Vec::new();

    for record in records {
        let key = record_identity_key(record, &identity_fields)
            .unwrap_or_else(|| format!("hash:{}", stable_digest(&canonical_json(record))));
        let hash = compute_sha256(&canonical_json(record));
        if next_records
            .insert(key.clone(), (hash, record.clone()))
            .is_some()
        {
            duplicate_keys.push(key);
        }
    }

    let mut inserted_count = 0_i64;
    let mut updated_count = 0_i64;
    let mut unchanged_count = 0_i64;
    for (record_key, (record_hash, record)) in &next_records {
        match existing.get(record_key) {
            Some(existing_hash) if existing_hash == record_hash => unchanged_count += 1,
            Some(_) => updated_count += 1,
            None => inserted_count += 1,
        }
        conn.execute(
            r#"
            INSERT INTO scrape_record_latest (
                target_id, record_key, record_hash, record_json, schema_key,
                first_seen_at, last_seen_at, last_run_id, deleted_at
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?6, ?7, NULL)
            ON CONFLICT(target_id, record_key) DO UPDATE SET
                record_hash = excluded.record_hash,
                record_json = excluded.record_json,
                schema_key = excluded.schema_key,
                last_seen_at = excluded.last_seen_at,
                last_run_id = excluded.last_run_id,
                deleted_at = NULL
            "#,
            params![
                target.view.target_id,
                record_key,
                record_hash,
                canonical_json(record),
                schema_key,
                finished_at,
                run_id,
            ],
        )?;
    }

    let mut deleted_count = 0_i64;
    for record_key in existing.keys() {
        if !next_records.contains_key(record_key) {
            deleted_count += 1;
            conn.execute(
                r#"
                UPDATE scrape_record_latest
                SET deleted_at = ?3,
                    last_seen_at = ?3,
                    last_run_id = ?4
                WHERE target_id = ?1 AND record_key = ?2 AND deleted_at IS NULL
                "#,
                params![target.view.target_id, record_key, finished_at, run_id],
            )?;
        }
    }

    let active_record_count = conn.query_row(
        "SELECT COUNT(*) FROM scrape_record_latest WHERE target_id = ?1 AND deleted_at IS NULL",
        params![target.view.target_id],
        |row| row.get::<_, i64>(0),
    )?;
    let state_dir = resolve_workspace_dir(Path::new(""), &target.view.workspace_dir).join("state");
    fs::create_dir_all(&state_dir)?;
    let latest_records_path = state_dir.join("latest_records.json");
    let latest_summary_path = state_dir.join("latest_summary.json");
    let delta_path = output_dir.join("delta.json");
    let latest_records = next_records
        .values()
        .map(|(_, record)| record.clone())
        .collect::<Vec<_>>();
    let summary = json!({
        "run_id": run_id,
        "target_key": target.view.target_key,
        "schema_key": schema_key,
        "identity_fields": identity_fields,
        "inserted_count": inserted_count,
        "updated_count": updated_count,
        "unchanged_count": unchanged_count,
        "deleted_count": deleted_count,
        "active_record_count": active_record_count,
        "duplicate_key_count": duplicate_keys.len(),
        "duplicate_keys": duplicate_keys,
        "latest_records_path": latest_records_path,
        "latest_summary_path": latest_summary_path,
    });
    fs::write(
        &latest_records_path,
        serde_json::to_string_pretty(&latest_records)?,
    )?;
    fs::write(
        &latest_summary_path,
        serde_json::to_string_pretty(&summary)?,
    )?;
    fs::write(&delta_path, serde_json::to_string_pretty(&summary)?)?;
    Ok(MaterializationOutcome {
        summary,
        delta_artifact: artifact_record("delta_json", &delta_path, schema_key.as_deref(), None)?,
    })
}

pub(super) fn should_auto_promote_template(
    script_body: &str,
    best_result_count: i64,
    target_count: i64,
    challenge_score: i64,
) -> (bool, String) {
    if script_body.trim().len() < MIN_TEMPLATE_CODE_LEN {
        return (false, "script_too_short".to_string());
    }
    if best_result_count < MIN_TEMPLATE_RESULTS {
        return (false, "result_count_below_threshold".to_string());
    }
    if target_count >= MIN_TEMPLATE_TARGETS {
        return (true, "multi_target_template".to_string());
    }
    if target_count >= 1 && challenge_score >= 3 {
        return (true, "manual_or_high_challenge_override".to_string());
    }
    (false, "insufficient_cross_target_evidence".to_string())
}

pub(super) fn upsert_promoted_template(
    conn: &Connection,
    template_key: &str,
    script_sha256: &str,
    script_body: &str,
    language: &str,
    source_example_count: i64,
    source_target_count: i64,
    best_result_count: i64,
    promotion_reason: &str,
) -> Result<()> {
    let now = now_iso_string();
    conn.execute(
        r#"
        INSERT INTO scrape_template_promoted (
            template_key, script_sha256, script_body, language, source_example_count,
            source_target_count, best_result_count, promotion_reason, is_active, created_at, updated_at
        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, 1, ?9, ?9)
        ON CONFLICT(template_key) DO UPDATE SET
            script_sha256 = excluded.script_sha256,
            script_body = excluded.script_body,
            language = excluded.language,
            source_example_count = excluded.source_example_count,
            source_target_count = excluded.source_target_count,
            best_result_count = excluded.best_result_count,
            promotion_reason = excluded.promotion_reason,
            is_active = 1,
            updated_at = excluded.updated_at
        "#,
        params![
            template_key,
            script_sha256,
            script_body,
            language,
            source_example_count,
            source_target_count,
            best_result_count,
            promotion_reason,
            now,
        ],
    )?;
    Ok(())
}
