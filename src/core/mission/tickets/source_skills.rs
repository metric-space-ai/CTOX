// Source-skill machinery: skill bundles (import/validate/show/query),
// skillbooks and runbook items, embeddings-backed retrieval, reply
// composition and the review-note grounding checks.

use super::{
    load_active_ticket_source_skill_binding_from_conn, load_case, load_ticket, now_iso_string,
    open_ticket_db, parse_json_column, parse_json_string_column,
    put_ticket_source_skill_binding_with_conn, record_audit, suggested_skill_for_routed_event,
    AuditRequest, RoutedTicketEvent, TicketCaseView, TicketItemView,
    TicketSourceKnowledgeResourceRecord, TicketSourceMainSkillRecord, TicketSourceRunbookBundle,
    TicketSourceRunbookItemRecord, TicketSourceRunbookRecord, TicketSourceSkillBindingView,
    TicketSourceSkillMatchView, TicketSourceSkillNoteReviewFinding,
    TicketSourceSkillNoteReviewView, TicketSourceSkillReplyView, TicketSourceSkillShowView,
    TicketSourceSkillbookRecord, DEFAULT_TICKET_SKILL_EMBEDDING_MODEL, WORKFLOW_CASE_KIND,
    WORKFLOW_ORCHESTRATOR_SKILL, WORKFLOW_STEP_KIND,
};
use crate::inference::engine;
use crate::inference::local_transport::LocalTransport;
use crate::inference::model_registry;
use crate::inference::runtime_kernel;
use crate::inference::supervisor;
use anyhow::{anyhow, bail, Context, Result};
use regex::Regex;
use rusqlite::{params, Connection, OptionalExtension};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::io::{BufRead as _, BufReader, Read as _, Write as _};
use std::path::Path;
use std::path::PathBuf;
use std::process::Command;
use std::time::Duration;

pub(super) fn resolve_source_skill_artifact_path(
    root: &Path,
    binding: &TicketSourceSkillBindingView,
) -> Option<std::path::PathBuf> {
    if let Ok(Some(path)) =
        crate::skill_store::resolve_materialized_skill_dir(root, &binding.skill_name)
    {
        return Some(path);
    }
    let raw = binding.artifact_path.as_deref()?.trim();
    resolve_skill_bundle_dir_hint(root, raw)
}

pub(super) fn resolve_skill_bundle_dir_hint(root: &Path, raw: &str) -> Option<std::path::PathBuf> {
    if raw.trim().is_empty() {
        return None;
    }
    let path = Path::new(raw.trim());
    let candidate = if path.is_absolute() {
        path.to_path_buf()
    } else {
        root.join(path)
    };
    candidate.exists().then_some(candidate)
}

pub(super) fn resolve_repo_script_path(root: &Path, relative: &str) -> Option<std::path::PathBuf> {
    let root_candidate = root.join(relative);
    if root_candidate.exists() {
        return Some(root_candidate);
    }
    if let Ok(current_dir) = std::env::current_dir() {
        let cwd_candidate = current_dir.join(relative);
        if cwd_candidate.exists() {
            return Some(cwd_candidate);
        }
    }
    None
}

pub(crate) fn show_ticket_source_skill(
    root: &Path,
    system: &str,
) -> Result<TicketSourceSkillShowView> {
    let conn = open_ticket_db(root)?;
    let binding = load_active_ticket_source_skill_binding_from_conn(&conn, system)?
        .context("active source skill binding not found")?;
    let artifact_path = resolve_source_skill_artifact_path(root, &binding);
    let skill_markdown_path = artifact_path
        .as_ref()
        .map(|path| path.join("SKILL.md"))
        .filter(|path| path.exists());
    let skill_preview = skill_markdown_path
        .as_ref()
        .map(std::fs::read_to_string)
        .transpose()?
        .map(|content| {
            content
                .lines()
                .filter(|line| !line.trim_start().starts_with("---"))
                .filter(|line| !line.trim().is_empty())
                .take(14)
                .collect::<Vec<_>>()
                .join("\n")
        })
        .filter(|text| !text.trim().is_empty());
    Ok(TicketSourceSkillShowView {
        binding,
        artifact_path: artifact_path.map(|path| path.display().to_string()),
        skill_markdown_path: skill_markdown_path.map(|path| path.display().to_string()),
        skill_preview,
    })
}

pub(crate) fn query_ticket_source_skill(
    root: &Path,
    system: &str,
    query: &str,
    top_k: usize,
) -> Result<Value> {
    let conn = open_ticket_db(root)?;
    let binding = load_active_ticket_source_skill_binding_from_conn(&conn, system)?
        .context("active source skill binding not found")?;
    match binding.archetype.as_str() {
        "operating-model" => {
            let artifact_path = resolve_source_skill_artifact_path(root, &binding)
                .context("active source skill binding does not have a usable artifact path")?;
            let script = resolve_repo_script_path(
                root,
                "skills/system/knowledge_bootstrap/ticket-operating-model-bootstrap/scripts/query_ticket_operating_model.py",
            )
            .context("ticket operating-model query helper is not available in this runtime root")?;
            if !script.exists() {
                anyhow::bail!(
                    "ticket operating-model query helper not found at {}",
                    script.display()
                );
            }
            let output = Command::new("python3")
                .arg(&script)
                .arg("--model-dir")
                .arg(&artifact_path)
                .arg("--query")
                .arg(query)
                .arg("--top-k")
                .arg(top_k.to_string())
                .output()
                .with_context(|| format!("failed to run {}", script.display()))?;
            if !output.status.success() {
                anyhow::bail!(
                    "source skill query failed: {}",
                    String::from_utf8_lossy(&output.stderr).trim()
                );
            }
            let payload: Value = serde_json::from_slice(&output.stdout)
                .context("source skill query returned invalid json")?;
            Ok(json!({
                "ok": true,
                "source_system": system,
                "binding": binding,
                "artifact_path": artifact_path.display().to_string(),
                "result": payload,
            }))
        }
        "skillbook-runbook" => {
            let (main_skill, retrieval_mode, matches) =
                query_ticket_skillbook_runbook_bundle(root, &conn, &binding, query, top_k)?;
            Ok(json!({
                "ok": true,
                "source_system": system,
                "binding": binding,
                "result": {
                    "retrieval_mode": retrieval_mode,
                    "main_skill": {
                        "main_skill_id": main_skill.main_skill_id,
                        "title": main_skill.title,
                        "primary_channel": main_skill.primary_channel,
                    },
                    "count": matches.len(),
                    "matches": matches,
                },
            }))
        }
        other => anyhow::bail!("source skill query is not supported for archetype {other}"),
    }
}

pub(crate) fn import_ticket_source_skill_bundle(
    root: &Path,
    system: &str,
    bundle_dir: &str,
    embedding_model_override: Option<&str>,
    skip_embeddings: bool,
) -> Result<Value> {
    let bundle_path = resolve_bundle_dir(root, bundle_dir)?;
    let main_skill: TicketSourceMainSkillRecord =
        read_json_file(&bundle_path.join("main_skill.json"))?;
    let skillbook: TicketSourceSkillbookRecord =
        read_json_file(&bundle_path.join("skillbook.json"))?;
    let runbooks = read_json_file::<TicketSourceRunbookBundle>(&bundle_path.join("runbook.json"))?
        .into_runbooks();
    let items: Vec<TicketSourceRunbookItemRecord> =
        read_jsonl_file(&bundle_path.join("runbook_items.jsonl"))?;
    let resources_path = bundle_path.join("resources.jsonl");
    let resources: Option<Vec<TicketSourceKnowledgeResourceRecord>> = if resources_path.is_file() {
        Some(read_jsonl_file(&resources_path)?)
    } else {
        None
    };
    anyhow::ensure!(
        !runbooks.is_empty(),
        "bundle {} does not contain runbooks",
        bundle_path.display()
    );
    anyhow::ensure!(
        !items.is_empty(),
        "bundle {} does not contain runbook items",
        bundle_path.display()
    );
    validate_ticket_source_skill_bundle(
        &main_skill,
        &skillbook,
        &runbooks,
        &items,
        resources.as_deref().unwrap_or_default(),
    )?;

    let now = now_iso_string();
    let embedding_model = embedding_model_override
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .unwrap_or_else(default_ticket_skill_embedding_model);

    let embeddings = if skip_embeddings {
        Vec::new()
    } else {
        let inputs = items
            .iter()
            .map(|item| item.chunk_text.clone())
            .collect::<Vec<_>>();
        embed_texts_for_ticket_skills(root, &inputs, &embedding_model)?
    };

    let runbooks_by_id = runbooks
        .iter()
        .map(|runbook| (runbook.runbook_id.as_str(), runbook))
        .collect::<BTreeMap<_, _>>();
    let runbook_ids = runbooks
        .iter()
        .map(|runbook| runbook.runbook_id.clone())
        .collect::<Vec<_>>();
    let mut conn = open_ticket_db(root)?;
    let binding;
    {
        let tx = conn.transaction()?;
        validate_ticket_source_skill_id_ownership(
            &tx,
            &skillbook.skillbook_id,
            &runbooks,
            &items,
            resources.as_deref(),
        )?;
        let incoming_item_ids = items
            .iter()
            .map(|item| item.item_id.as_str())
            .collect::<BTreeSet<_>>();
        let mut existing_items =
            tx.prepare("SELECT item_id FROM knowledge_runbook_items WHERE skillbook_id = ?1")?;
        let old_item_ids = existing_items
            .query_map(params![skillbook.skillbook_id], |row| {
                row.get::<_, String>(0)
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        drop(existing_items);
        for old_item_id in old_item_ids {
            if !incoming_item_ids.contains(old_item_id.as_str()) {
                tx.execute(
                    "DELETE FROM knowledge_embeddings WHERE item_id = ?1",
                    params![old_item_id],
                )?;
            }
        }
        tx.execute(
            "DELETE FROM knowledge_runbook_items WHERE skillbook_id = ?1",
            params![skillbook.skillbook_id],
        )?;
        tx.execute(
            "DELETE FROM knowledge_runbooks WHERE skillbook_id = ?1",
            params![skillbook.skillbook_id],
        )?;
        if resources.is_some() {
            tx.execute(
                "DELETE FROM knowledge_resources WHERE skillbook_id = ?1",
                params![skillbook.skillbook_id],
            )?;
        }
        upsert_ticket_source_main_skill(&tx, &main_skill, &now)?;
        upsert_ticket_source_skillbook(&tx, &skillbook, &now)?;
        for runbook in &runbooks {
            upsert_ticket_source_runbook(&tx, runbook, &now)?;
        }
        for (index, item) in items.iter().enumerate() {
            let runbook = runbooks_by_id
                .get(item.runbook_id.as_str())
                .context("validated runbook parent missing during import")?;
            upsert_ticket_source_runbook_item(&tx, item, &runbook.version, &runbook.status, &now)?;
            if let Some(vector) = embeddings.get(index) {
                upsert_ticket_source_embedding(&tx, &item.item_id, &embedding_model, vector, &now)?;
            }
        }
        if let Some(resources) = &resources {
            for resource in resources {
                upsert_ticket_source_knowledge_resource(
                    &tx,
                    &skillbook.skillbook_id,
                    resource,
                    &now,
                )?;
            }
        }
        binding = put_ticket_source_skill_binding_with_conn(
            &tx,
            system,
            &main_skill.main_skill_id,
            "skillbook-runbook",
            "active",
            "bundle-import",
            Some(bundle_dir),
            Some(&format!(
                "Imported main skill {}, skillbook {}, {} runbooks",
                main_skill.main_skill_id,
                skillbook.skillbook_id,
                runbooks.len()
            )),
            &now,
        )?;
        record_audit(
            &tx,
            AuditRequest {
                ticket_key: &format!("*ticket-source:{}*", system),
                case_id: None,
                actor_type: "knowledge_importer",
                action_type: "source_skill_bundle_import",
                label: None,
                bundle_label: None,
                bundle_version: None,
                details: json!({
                    "system": system,
                    "main_skill_id": main_skill.main_skill_id,
                    "skillbook_id": skillbook.skillbook_id,
                    "runbook_ids": runbook_ids.clone(),
                    "runbook_count": runbooks.len(),
                    "item_count": items.len(),
                    "resource_count": resources.as_ref().map(Vec::len),
                    "embedding_model": if skip_embeddings { None::<String> } else { Some(embedding_model.clone()) },
                    "bundle_dir": bundle_path.display().to_string(),
                }),
            },
        )?;
        tx.commit()?;
    }
    Ok(json!({
        "ok": true,
        "binding": binding,
        "bundle_dir": bundle_path.display().to_string(),
        "main_skill_id": main_skill.main_skill_id,
        "skillbook_id": skillbook.skillbook_id,
        "runbook_ids": runbook_ids,
        "runbook_count": runbooks.len(),
        "item_count": items.len(),
        "resource_count": resources.as_ref().map(Vec::len),
        "embedding_model": if skip_embeddings { Value::Null } else { json!(embedding_model) },
        "embeddings_indexed": !skip_embeddings,
    }))
}

pub(crate) fn import_ticket_source_skill_resources(
    root: &Path,
    skillbook_id: &str,
    resources_file: &str,
    replace: bool,
) -> Result<Value> {
    let skillbook_id = skillbook_id.trim();
    anyhow::ensure!(!skillbook_id.is_empty(), "skillbook_id must not be empty");
    let resources_path = {
        let path = Path::new(resources_file);
        if path.is_absolute() {
            path.to_path_buf()
        } else {
            root.join(path)
        }
    };
    anyhow::ensure!(
        resources_path.is_file(),
        "resources file not found at {}",
        resources_path.display()
    );
    let resources: Vec<TicketSourceKnowledgeResourceRecord> = read_jsonl_file(&resources_path)?;
    anyhow::ensure!(
        !resources.is_empty(),
        "resources file {} is empty",
        resources_path.display()
    );

    let now = now_iso_string();
    let mut conn = open_ticket_db(root)?;
    let imported_resource_ids;
    {
        let tx = conn.transaction()?;
        let skillbook_exists = tx.query_row(
            "SELECT EXISTS(SELECT 1 FROM knowledge_skillbooks WHERE skillbook_id = ?1)",
            params![skillbook_id],
            |row| row.get::<_, bool>(0),
        )?;
        anyhow::ensure!(
            skillbook_exists,
            "knowledge skillbook {} does not exist",
            skillbook_id
        );

        let known_item_ids = tx
            .prepare("SELECT item_id FROM knowledge_runbook_items WHERE skillbook_id = ?1")?
            .query_map(params![skillbook_id], |row| row.get::<_, String>(0))?
            .collect::<rusqlite::Result<BTreeSet<_>>>()?;
        let mut resource_ids = BTreeSet::new();
        for resource in &resources {
            anyhow::ensure!(
                !resource.resource_id.trim().is_empty(),
                "resources file contains an empty resource_id"
            );
            anyhow::ensure!(
                !resource.title.trim().is_empty(),
                "resource {} has an empty title",
                resource.resource_id
            );
            anyhow::ensure!(
                resource_ids.insert(resource.resource_id.as_str()),
                "resources file contains duplicate resource {}",
                resource.resource_id
            );
            for item_id in &resource.linked_runbook_items {
                anyhow::ensure!(
                    known_item_ids.contains(item_id),
                    "resource {} references runbook item {} outside skillbook {}",
                    resource.resource_id,
                    item_id,
                    skillbook_id
                );
            }
        }
        validate_ticket_source_skill_id_ownership(&tx, skillbook_id, &[], &[], Some(&resources))?;

        if replace {
            tx.execute(
                "DELETE FROM knowledge_resources WHERE skillbook_id = ?1",
                params![skillbook_id],
            )?;
        }
        for resource in &resources {
            upsert_ticket_source_knowledge_resource(&tx, skillbook_id, resource, &now)?;
        }
        imported_resource_ids = resources
            .iter()
            .map(|resource| resource.resource_id.clone())
            .collect::<Vec<_>>();
        record_audit(
            &tx,
            AuditRequest {
                ticket_key: &format!("*knowledge-skillbook:{}*", skillbook_id),
                case_id: None,
                actor_type: "knowledge_importer",
                action_type: "skillbook_resource_import",
                label: None,
                bundle_label: None,
                bundle_version: None,
                details: json!({
                    "skillbook_id": skillbook_id,
                    "resource_count": resources.len(),
                    "replace": replace,
                    "resources_file": resources_path.display().to_string(),
                }),
            },
        )?;
        tx.commit()?;
    }

    Ok(json!({
        "ok": true,
        "skillbook_id": skillbook_id,
        "resource_count": imported_resource_ids.len(),
        "resource_ids": imported_resource_ids,
        "replace": replace,
        "resources_file": resources_path.display().to_string(),
    }))
}

pub(super) fn validate_ticket_source_skill_bundle(
    main_skill: &TicketSourceMainSkillRecord,
    skillbook: &TicketSourceSkillbookRecord,
    runbooks: &[TicketSourceRunbookRecord],
    items: &[TicketSourceRunbookItemRecord],
    resources: &[TicketSourceKnowledgeResourceRecord],
) -> Result<()> {
    anyhow::ensure!(
        main_skill
            .linked_skillbooks
            .iter()
            .any(|id| id == &skillbook.skillbook_id),
        "main skill {} does not link skillbook {}",
        main_skill.main_skill_id,
        skillbook.skillbook_id
    );

    let mut runbook_ids = BTreeSet::new();
    for runbook in runbooks {
        anyhow::ensure!(
            !runbook.runbook_id.trim().is_empty(),
            "bundle contains a runbook with an empty runbook_id"
        );
        anyhow::ensure!(
            runbook.skillbook_id == skillbook.skillbook_id,
            "runbook {} links skillbook {}, expected {}",
            runbook.runbook_id,
            runbook.skillbook_id,
            skillbook.skillbook_id
        );
        anyhow::ensure!(
            runbook_ids.insert(runbook.runbook_id.as_str()),
            "bundle contains duplicate runbook {}",
            runbook.runbook_id
        );
    }

    let linked_runbooks = skillbook
        .linked_runbooks
        .iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    anyhow::ensure!(
        linked_runbooks == runbook_ids,
        "skillbook linked_runbooks do not match the runbook catalog"
    );
    let main_linked_runbooks = main_skill
        .linked_runbooks
        .iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    anyhow::ensure!(
        runbook_ids.is_subset(&main_linked_runbooks),
        "main skill linked_runbooks omit runbooks from the catalog"
    );

    let mut item_ids = BTreeSet::new();
    for item in items {
        anyhow::ensure!(
            item_ids.insert(item.item_id.as_str()),
            "bundle contains duplicate runbook item {}",
            item.item_id
        );
        anyhow::ensure!(
            runbook_ids.contains(item.runbook_id.as_str()),
            "runbook item {} references missing runbook {}",
            item.item_id,
            item.runbook_id
        );
        anyhow::ensure!(
            item.skillbook_id == skillbook.skillbook_id,
            "runbook item {} links skillbook {}, expected {}",
            item.item_id,
            item.skillbook_id,
            skillbook.skillbook_id
        );
    }

    let known_item_ids = items
        .iter()
        .map(|item| item.item_id.as_str())
        .collect::<BTreeSet<_>>();
    let mut resource_ids = BTreeSet::new();
    for resource in resources {
        anyhow::ensure!(
            !resource.resource_id.trim().is_empty(),
            "bundle contains a resource with an empty resource_id"
        );
        anyhow::ensure!(
            !resource.title.trim().is_empty(),
            "resource {} has an empty title",
            resource.resource_id
        );
        anyhow::ensure!(
            resource_ids.insert(resource.resource_id.as_str()),
            "bundle contains duplicate resource {}",
            resource.resource_id
        );
        for item_id in &resource.linked_runbook_items {
            anyhow::ensure!(
                known_item_ids.contains(item_id.as_str()),
                "resource {} references missing runbook item {}",
                resource.resource_id,
                item_id
            );
        }
    }
    Ok(())
}

pub(super) fn validate_ticket_source_skill_id_ownership(
    conn: &Connection,
    skillbook_id: &str,
    runbooks: &[TicketSourceRunbookRecord],
    items: &[TicketSourceRunbookItemRecord],
    resources: Option<&[TicketSourceKnowledgeResourceRecord]>,
) -> Result<()> {
    for runbook in runbooks {
        let owner = conn
            .query_row(
                "SELECT skillbook_id FROM knowledge_runbooks WHERE runbook_id = ?1",
                params![runbook.runbook_id],
                |row| row.get::<_, String>(0),
            )
            .optional()?;
        anyhow::ensure!(
            owner.as_deref().is_none_or(|owner| owner == skillbook_id),
            "runbook {} is already owned by skillbook {}",
            runbook.runbook_id,
            owner.unwrap_or_default()
        );
    }
    for item in items {
        let owner = conn
            .query_row(
                "SELECT skillbook_id FROM knowledge_runbook_items WHERE item_id = ?1",
                params![item.item_id],
                |row| row.get::<_, String>(0),
            )
            .optional()?;
        anyhow::ensure!(
            owner.as_deref().is_none_or(|owner| owner == skillbook_id),
            "runbook item {} is already owned by skillbook {}",
            item.item_id,
            owner.unwrap_or_default()
        );
    }
    for resource in resources.unwrap_or_default() {
        let owner = conn
            .query_row(
                "SELECT skillbook_id FROM knowledge_resources WHERE resource_id = ?1",
                params![resource.resource_id],
                |row| row.get::<_, String>(0),
            )
            .optional()?;
        anyhow::ensure!(
            owner.as_deref().is_none_or(|owner| owner == skillbook_id),
            "resource {} is already owned by skillbook {}",
            resource.resource_id,
            owner.unwrap_or_default()
        );
    }
    Ok(())
}

pub(crate) fn resolve_ticket_source_skill_for_target(
    root: &Path,
    ticket_key: Option<&str>,
    case_id: Option<&str>,
    top_k: usize,
) -> Result<Value> {
    let (ticket, case) = resolve_ticket_and_case(root, ticket_key, case_id)?;
    let query = build_ticket_source_skill_query_text(&ticket);
    let result = query_ticket_source_skill(root, &ticket.source_system, &query, top_k)?;
    Ok(json!({
        "ok": true,
        "ticket_key": ticket.ticket_key,
        "case_id": case.as_ref().map(|item| item.case_id.clone()),
        "query": query,
        "resolution": result.get("result").cloned().unwrap_or_else(|| json!({})),
    }))
}

pub(crate) fn compose_ticket_source_skill_reply(
    root: &Path,
    ticket_key: Option<&str>,
    case_id: Option<&str>,
    send_policy: &str,
    subject_override: Option<&str>,
    body_only: bool,
) -> Result<Value> {
    let canonical_send_policy = canonical_source_skill_send_policy(send_policy)?;
    let (ticket, case) = resolve_ticket_and_case(root, ticket_key, case_id)?;
    let query = build_ticket_source_skill_query_text(&ticket);
    let conn = open_ticket_db(root)?;
    let binding = load_active_ticket_source_skill_binding_from_conn(&conn, &ticket.source_system)?
        .context("active source skill binding not found")?;
    anyhow::ensure!(
        binding.archetype == "skillbook-runbook",
        "reply composition is only supported for skillbook-runbook bindings"
    );
    let (main_skill, retrieval_mode, matches) =
        query_ticket_skillbook_runbook_bundle(root, &conn, &binding, &query, 3)?;
    let best = matches
        .first()
        .cloned()
        .context("no runbook item match found for reply composition")?;
    let second_score = matches.get(1).map(|item| item.score).unwrap_or(0.0);
    let score_gap = best.score - second_score;
    let confidence_clear = match retrieval_mode.as_str() {
        "embedding" => best.score >= 0.35 && score_gap >= 0.02,
        _ => best.score >= 0.08 && score_gap >= 0.02,
    };
    if !confidence_clear {
        return Ok(json!({
            "decision": "needs_review",
            "ticket_key": ticket.ticket_key,
            "case_id": case.as_ref().map(|item| item.case_id.clone()),
            "retrieval_mode": retrieval_mode,
            "matches": matches,
        }));
    }
    let skillbook = load_ticket_source_skillbook_from_conn(
        &conn,
        main_skill
            .linked_skillbooks
            .first()
            .map(String::as_str)
            .context("main skill has no linked skillbook")?,
    )?
    .context("linked skillbook not found in runtime db")?;
    let reply = compose_reply_from_runbook_item(
        &ticket,
        case.as_ref(),
        &main_skill,
        &skillbook,
        &best,
        canonical_send_policy,
        subject_override,
    )?;
    if body_only {
        return Ok(Value::String(reply.reply_body));
    }
    Ok(serde_json::to_value(reply)?)
}

pub(super) fn resolve_ticket_and_case(
    root: &Path,
    ticket_key: Option<&str>,
    case_id: Option<&str>,
) -> Result<(TicketItemView, Option<TicketCaseView>)> {
    match (ticket_key, case_id) {
        (Some(ticket_key), None) => Ok((
            load_ticket(root, ticket_key)?.context("ticket not found")?,
            None,
        )),
        (None, Some(case_id)) => {
            let case = load_case(root, case_id)?.context("ticket case not found")?;
            let ticket =
                load_ticket(root, &case.ticket_key)?.context("ticket not found for case")?;
            Ok((ticket, Some(case)))
        }
        (Some(_), Some(_)) => anyhow::bail!("provide either --ticket-key or --case-id, not both"),
        (None, None) => anyhow::bail!("provide --ticket-key or --case-id"),
    }
}

pub(super) fn query_ticket_skillbook_runbook_bundle(
    root: &Path,
    conn: &Connection,
    binding: &TicketSourceSkillBindingView,
    query: &str,
    top_k: usize,
) -> Result<(
    TicketSourceMainSkillRecord,
    String,
    Vec<TicketSourceSkillMatchView>,
)> {
    let main_skill = load_ticket_source_main_skill_from_conn(conn, &binding.skill_name)?
        .context("bound main skill is not present in runtime db; import the bundle first")?;
    anyhow::ensure!(
        !main_skill.linked_runbooks.is_empty(),
        "bound main skill does not link any runbooks"
    );
    let items = load_ticket_source_runbook_items_for_runbooks(conn, &main_skill.linked_runbooks)?;
    anyhow::ensure!(
        !items.is_empty(),
        "no runbook items are stored for the linked source skill runbooks"
    );
    let embeddings = load_ticket_source_embeddings_for_items(
        conn,
        &items
            .iter()
            .map(|item| item.item_id.clone())
            .collect::<Vec<_>>(),
    )?;
    let embedding_model = embeddings
        .values()
        .find_map(|(model, _)| Some(model.clone()));
    let (retrieval_mode, scored_matches) = if let Some(model) = embedding_model {
        let query_embedding = embed_texts_for_ticket_skills(root, &[query.to_string()], &model)?
            .into_iter()
            .next()
            .context("embedding service returned no query vector")?;
        let mut matches = items
            .iter()
            .filter_map(|item| {
                let (_, embedding) = embeddings.get(&item.item_id)?;
                Some(TicketSourceSkillMatchView {
                    item_id: item.item_id.clone(),
                    label: item.label.clone(),
                    title: item.title.clone(),
                    problem_class: item.problem_class.clone(),
                    score: cosine_similarity(&query_embedding, embedding),
                    expected_guidance: item.expected_guidance.clone(),
                    earliest_blocker: item.earliest_blocker.clone(),
                    escalate_when: item.escalate_when.clone(),
                    pages: item.pages.clone(),
                    tool_actions: item.tool_actions.clone(),
                    writeback_policy: item.writeback_policy.clone(),
                })
            })
            .collect::<Vec<_>>();
        matches.sort_by(|left, right| {
            right
                .score
                .partial_cmp(&left.score)
                .unwrap_or(Ordering::Equal)
        });
        ("embedding".to_string(), matches)
    } else {
        let mut matches = items
            .iter()
            .map(|item| TicketSourceSkillMatchView {
                item_id: item.item_id.clone(),
                label: item.label.clone(),
                title: item.title.clone(),
                problem_class: item.problem_class.clone(),
                score: lexical_overlap_ratio(query, &item.chunk_text),
                expected_guidance: item.expected_guidance.clone(),
                earliest_blocker: item.earliest_blocker.clone(),
                escalate_when: item.escalate_when.clone(),
                pages: item.pages.clone(),
                tool_actions: item.tool_actions.clone(),
                writeback_policy: item.writeback_policy.clone(),
            })
            .collect::<Vec<_>>();
        matches.sort_by(|left, right| {
            right
                .score
                .partial_cmp(&left.score)
                .unwrap_or(Ordering::Equal)
        });
        ("lexical_fallback".to_string(), matches)
    };
    let mut matches = scored_matches;
    matches.truncate(top_k.max(1));
    Ok((main_skill, retrieval_mode, matches))
}

pub(super) fn compose_reply_from_runbook_item(
    ticket: &TicketItemView,
    case: Option<&TicketCaseView>,
    _main_skill: &TicketSourceMainSkillRecord,
    _skillbook: &TicketSourceSkillbookRecord,
    item: &TicketSourceSkillMatchView,
    send_policy: &str,
    subject_override: Option<&str>,
) -> Result<TicketSourceSkillReplyView> {
    let language = detect_ticket_reply_language(&format!("{}\n{}", ticket.title, ticket.body_text));
    let salutation = if language == "en" { "Hello," } else { "Hallo," };
    let manual_reference = if item.pages.is_empty() {
        None
    } else {
        Some(format!("Manual reference: {}", item.pages.join(", ")))
    };
    let mut paragraphs = vec![
        salutation.to_string(),
        item.expected_guidance.trim().to_string(),
    ];
    if let Some(reference) = manual_reference.clone() {
        paragraphs.push(reference);
    }
    let reply_body = paragraphs.join("\n\n");
    Ok(TicketSourceSkillReplyView {
        decision: send_policy.to_string(),
        source_system: ticket.source_system.clone(),
        ticket_key: ticket.ticket_key.clone(),
        case_id: case.map(|item| item.case_id.clone()),
        matched_label: item.label.clone(),
        item_id: item.item_id.clone(),
        reply_subject: subject_override
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(|value| format!("Re: {value}"))
            .unwrap_or_else(|| format!("Re: {}", ticket.title.trim())),
        reply_body,
        manual_reference,
        writeback_policy: item.writeback_policy.clone(),
    })
}

pub(super) fn detect_ticket_reply_language(text: &str) -> &'static str {
    let lowered = text.to_lowercase();
    let english_markers = [
        "hello",
        "please",
        "password",
        "support",
        "registration",
        "login",
    ];
    if english_markers
        .iter()
        .filter(|marker| lowered.contains(**marker))
        .count()
        >= 2
    {
        "en"
    } else {
        "de"
    }
}

pub(super) fn canonical_source_skill_send_policy(value: &str) -> Result<&'static str> {
    match value.trim().to_ascii_lowercase().as_str() {
        "suggestion" | "suggest" => Ok("suggestion"),
        "draft" => Ok("draft"),
        "send" => Ok("send"),
        other => anyhow::bail!("unsupported send policy: {other}"),
    }
}

pub(super) fn resolve_bundle_dir(root: &Path, raw: &str) -> Result<PathBuf> {
    let candidate = Path::new(raw.trim());
    let path = if candidate.is_absolute() {
        candidate.to_path_buf()
    } else {
        root.join(candidate)
    };
    anyhow::ensure!(
        path.exists(),
        "bundle path does not exist: {}",
        path.display()
    );
    Ok(path)
}

pub(super) fn read_json_file<T: DeserializeOwned>(path: &Path) -> Result<T> {
    let body = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    serde_json::from_str(&body).with_context(|| format!("invalid json in {}", path.display()))
}

pub(super) fn read_jsonl_file<T: DeserializeOwned>(path: &Path) -> Result<Vec<T>> {
    let body = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    body.lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| serde_json::from_str(line).map_err(anyhow::Error::from))
        .collect::<Result<Vec<_>>>()
        .with_context(|| format!("invalid jsonl in {}", path.display()))
}

pub(super) fn default_ticket_skill_embedding_model() -> String {
    model_registry::default_auxiliary_model(engine::AuxiliaryRole::Embedding)
        .unwrap_or(DEFAULT_TICKET_SKILL_EMBEDDING_MODEL)
        .to_string()
}

pub(super) fn embed_texts_for_ticket_skills(
    root: &Path,
    inputs: &[String],
    model: &str,
) -> Result<Vec<Vec<f64>>> {
    if inputs.is_empty() {
        return Ok(Vec::new());
    }
    supervisor::ensure_auxiliary_backend_launchable(root, engine::AuxiliaryRole::Embedding)
        .context("embedding backend is not launchable for ticket skill retrieval")?;
    supervisor::ensure_auxiliary_backend_ready(root, engine::AuxiliaryRole::Embedding, false)
        .context("failed to ensure managed embedding backend for ticket skill retrieval")?;
    let resolved_runtime = runtime_kernel::InferenceRuntimeKernel::resolve(root)
        .context("failed to resolve runtime kernel for ticket skill retrieval")?;
    if let Some(binding) =
        resolved_runtime.binding_for_auxiliary_role(engine::AuxiliaryRole::Embedding)
    {
        if !binding.transport.is_private_ipc() {
            anyhow::bail!(
                "ctox_core_local requires private IPC for local embedding inference; loopback HTTP transport is not allowed"
            );
        }
        let label = binding.transport.display_label();
        return embed_texts_for_ticket_skills_via_local_socket(&binding.transport, inputs, model)
            .with_context(|| format!("failed to reach embedding transport {label}"));
    }
    let base_url = resolved_runtime
        .auxiliary_base_url(engine::AuxiliaryRole::Embedding)
        .filter(|value| !value.trim().is_empty())
        .map(str::to_string)
        .ok_or_else(|| anyhow::anyhow!("embedding runtime is not resolved"))?;
    let response = ureq::post(&format!("{}/v1/embeddings", base_url.trim_end_matches('/')))
        .set("content-type", "application/json")
        .timeout(Duration::from_secs(30))
        .send_string(&serde_json::to_string(&json!({
            "model": model,
            "input": inputs,
        }))?)
        .with_context(|| format!("failed to reach embedding service at {}", base_url))?;
    let body = response
        .into_string()
        .context("failed to read embedding response")?;
    let payload: Value =
        serde_json::from_str(&body).context("failed to parse embedding response")?;
    let mut indexed = payload
        .get("data")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    indexed.sort_by_key(|item| item.get("index").and_then(Value::as_u64).unwrap_or(0));
    let vectors = indexed
        .into_iter()
        .map(|item| {
            item.get("embedding")
                .and_then(Value::as_array)
                .map(|values| values.iter().filter_map(Value::as_f64).collect::<Vec<_>>())
                .filter(|values| !values.is_empty())
                .context("embedding response missing vectors")
        })
        .collect::<Result<Vec<_>>>()?;
    anyhow::ensure!(
        vectors.len() == inputs.len(),
        "embedding response count mismatch: expected {}, got {}",
        inputs.len(),
        vectors.len()
    );
    Ok(vectors)
}

#[derive(Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum TicketSkillEmbeddingSocketRequest<'a> {
    EmbeddingsCreate {
        model: &'a str,
        inputs: &'a [String],
        truncate_sequence: bool,
    },
}

#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum TicketSkillEmbeddingSocketResponse {
    Embeddings {
        #[allow(dead_code)]
        model: String,
        data: Vec<Vec<f32>>,
        #[serde(rename = "prompt_tokens")]
        _prompt_tokens: u32,
        #[serde(rename = "total_tokens")]
        _total_tokens: u32,
    },
    Error {
        code: String,
        message: String,
    },
}

pub(super) fn embed_texts_for_ticket_skills_via_local_socket(
    transport: &LocalTransport,
    inputs: &[String],
    model: &str,
) -> Result<Vec<Vec<f64>>> {
    let timeout = Duration::from_secs(30);
    let label = transport.display_label();
    let mut stream = transport
        .connect_blocking(timeout)
        .with_context(|| format!("failed to connect via {label}"))?;
    let request = TicketSkillEmbeddingSocketRequest::EmbeddingsCreate {
        model,
        inputs,
        truncate_sequence: false,
    };
    let mut payload =
        serde_json::to_vec(&request).context("failed to encode ticket skill embedding request")?;
    payload.push(b'\n');
    stream
        .write_all(&payload)
        .with_context(|| format!("failed to write request via {label}"))?;
    stream
        .flush()
        .with_context(|| format!("failed to flush request via {label}"))?;
    let mut reader = BufReader::new(stream);
    let mut line = String::new();
    reader
        .read_line(&mut line)
        .with_context(|| format!("failed to read response via {label}"))?;
    anyhow::ensure!(
        !line.trim().is_empty(),
        "embedding socket returned an empty response"
    );
    match serde_json::from_str::<TicketSkillEmbeddingSocketResponse>(line.trim())
        .context("failed to parse embedding socket response")?
    {
        TicketSkillEmbeddingSocketResponse::Embeddings { data, .. } => Ok(data
            .into_iter()
            .map(|values| values.into_iter().map(|value| value as f64).collect())
            .collect()),
        TicketSkillEmbeddingSocketResponse::Error { code, message } => {
            anyhow::bail!("{code}: {message}")
        }
    }
}

pub(super) fn cosine_similarity(left: &[f64], right: &[f64]) -> f64 {
    if left.is_empty() || right.is_empty() || left.len() != right.len() {
        return 0.0;
    }
    let mut dot = 0.0;
    let mut left_norm = 0.0;
    let mut right_norm = 0.0;
    for (l, r) in left.iter().zip(right.iter()) {
        dot += l * r;
        left_norm += l * l;
        right_norm += r * r;
    }
    if left_norm <= f64::EPSILON || right_norm <= f64::EPSILON {
        0.0
    } else {
        dot / (left_norm.sqrt() * right_norm.sqrt())
    }
}

pub(super) fn upsert_ticket_source_main_skill(
    conn: &Connection,
    record: &TicketSourceMainSkillRecord,
    now: &str,
) -> Result<()> {
    conn.execute(
        r#"
        INSERT INTO knowledge_main_skills (
            main_skill_id, title, primary_channel, entry_action, resolver_contract_json,
            execution_contract_json, resolve_flow_json, writeback_flow_json,
            linked_skillbooks_json, linked_runbooks_json, created_at, updated_at
        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?11)
        ON CONFLICT(main_skill_id) DO UPDATE SET
            title=excluded.title,
            primary_channel=excluded.primary_channel,
            entry_action=excluded.entry_action,
            resolver_contract_json=excluded.resolver_contract_json,
            execution_contract_json=excluded.execution_contract_json,
            resolve_flow_json=excluded.resolve_flow_json,
            writeback_flow_json=excluded.writeback_flow_json,
            linked_skillbooks_json=excluded.linked_skillbooks_json,
            linked_runbooks_json=excluded.linked_runbooks_json,
            updated_at=excluded.updated_at
        "#,
        params![
            record.main_skill_id,
            record.title,
            record.primary_channel,
            record.entry_action,
            serde_json::to_string(&record.resolver_contract)?,
            serde_json::to_string(&record.execution_contract)?,
            serde_json::to_string(&record.resolve_flow)?,
            serde_json::to_string(&record.writeback_flow)?,
            serde_json::to_string(&record.linked_skillbooks)?,
            serde_json::to_string(&record.linked_runbooks)?,
            now,
        ],
    )?;
    Ok(())
}

pub(super) fn upsert_ticket_source_skillbook(
    conn: &Connection,
    record: &TicketSourceSkillbookRecord,
    now: &str,
) -> Result<()> {
    conn.execute(
        r#"
        INSERT INTO knowledge_skillbooks (
            skillbook_id, title, version, status, summary, mission, non_negotiable_rules_json,
            runtime_policy, answer_contract, workflow_backbone_json, routing_taxonomy_json,
            linked_runbooks_json, created_at, updated_at
        ) VALUES (?1, ?2, ?3, 'active', ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?12)
        ON CONFLICT(skillbook_id) DO UPDATE SET
            title=excluded.title,
            version=excluded.version,
            status=excluded.status,
            summary=excluded.summary,
            mission=excluded.mission,
            non_negotiable_rules_json=excluded.non_negotiable_rules_json,
            runtime_policy=excluded.runtime_policy,
            answer_contract=excluded.answer_contract,
            workflow_backbone_json=excluded.workflow_backbone_json,
            routing_taxonomy_json=excluded.routing_taxonomy_json,
            linked_runbooks_json=excluded.linked_runbooks_json,
            updated_at=excluded.updated_at
        "#,
        params![
            record.skillbook_id,
            record.title,
            record.version,
            summarize_text(&record.mission, 220),
            record.mission,
            serde_json::to_string(&record.non_negotiable_rules)?,
            record.runtime_policy,
            record.answer_contract,
            serde_json::to_string(&record.workflow_backbone)?,
            serde_json::to_string(&record.routing_taxonomy)?,
            serde_json::to_string(&record.linked_runbooks)?,
            now,
        ],
    )?;
    Ok(())
}

pub(super) fn upsert_ticket_source_runbook(
    conn: &Connection,
    record: &TicketSourceRunbookRecord,
    now: &str,
) -> Result<()> {
    conn.execute(
        r#"
        INSERT INTO knowledge_runbooks (
            runbook_id, skillbook_id, title, version, status, summary, problem_domain,
            item_labels_json, created_at, updated_at
        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?9)
        ON CONFLICT(runbook_id) DO UPDATE SET
            skillbook_id=excluded.skillbook_id,
            title=excluded.title,
            version=excluded.version,
            status=excluded.status,
            summary=excluded.summary,
            problem_domain=excluded.problem_domain,
            item_labels_json=excluded.item_labels_json,
            updated_at=excluded.updated_at
        "#,
        params![
            record.runbook_id,
            record.skillbook_id,
            record.title,
            record.version,
            record.status,
            summarize_text(&record.title, 220),
            record.problem_domain,
            serde_json::to_string(&record.item_labels)?,
            now,
        ],
    )?;
    Ok(())
}

pub(super) fn upsert_ticket_source_runbook_item(
    conn: &Connection,
    record: &TicketSourceRunbookItemRecord,
    version: &str,
    status: &str,
    now: &str,
) -> Result<()> {
    conn.execute(
        r#"
        INSERT INTO knowledge_runbook_items (
            item_id, runbook_id, skillbook_id, label, title, problem_class, chunk_text,
            structured_json, status, version, created_at, updated_at
        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?11)
        ON CONFLICT(item_id) DO UPDATE SET
            runbook_id=excluded.runbook_id,
            skillbook_id=excluded.skillbook_id,
            label=excluded.label,
            title=excluded.title,
            problem_class=excluded.problem_class,
            chunk_text=excluded.chunk_text,
            structured_json=excluded.structured_json,
            status=excluded.status,
            version=excluded.version,
            updated_at=excluded.updated_at
        "#,
        params![
            record.item_id,
            record.runbook_id,
            record.skillbook_id,
            record.label,
            record.title,
            record.problem_class,
            record.chunk_text,
            serde_json::to_string(record)?,
            status,
            version,
            now,
        ],
    )?;
    Ok(())
}

pub(super) fn upsert_ticket_source_knowledge_resource(
    conn: &Connection,
    skillbook_id: &str,
    record: &TicketSourceKnowledgeResourceRecord,
    now: &str,
) -> Result<()> {
    conn.execute(
        r#"
        INSERT INTO knowledge_resources (
            resource_id, skillbook_id, title, kind, source_id, role, canonical_url,
            snapshot_hash, evidence_eligible, linked_runbook_items_json, metadata_json,
            created_at, updated_at
        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?12)
        ON CONFLICT(resource_id) DO UPDATE SET
            skillbook_id=excluded.skillbook_id,
            title=excluded.title,
            kind=excluded.kind,
            source_id=excluded.source_id,
            role=excluded.role,
            canonical_url=excluded.canonical_url,
            snapshot_hash=excluded.snapshot_hash,
            evidence_eligible=excluded.evidence_eligible,
            linked_runbook_items_json=excluded.linked_runbook_items_json,
            metadata_json=excluded.metadata_json,
            updated_at=excluded.updated_at
        "#,
        params![
            record.resource_id,
            skillbook_id,
            record.title,
            record.kind,
            record.source_id,
            record.role,
            record.canonical_url,
            record.snapshot_hash,
            i64::from(record.evidence_eligible),
            serde_json::to_string(&record.linked_runbook_items)?,
            serde_json::to_string(record)?,
            now,
        ],
    )?;
    Ok(())
}

pub(super) fn upsert_ticket_source_embedding(
    conn: &Connection,
    item_id: &str,
    embedding_model: &str,
    vector: &[f64],
    now: &str,
) -> Result<()> {
    conn.execute(
        r#"
        INSERT INTO knowledge_embeddings (
            item_id, embedding_model, embedding_json, updated_at
        ) VALUES (?1, ?2, ?3, ?4)
        ON CONFLICT(item_id, embedding_model) DO UPDATE SET
            embedding_json=excluded.embedding_json,
            updated_at=excluded.updated_at
        "#,
        params![
            item_id,
            embedding_model,
            serde_json::to_string(vector)?,
            now
        ],
    )?;
    Ok(())
}

// ----- Incremental procedural-knowledge writers ---------------------------
//
// Builder-style entry points exposed for `ctox knowledge skill new /
// add-skillbook / add-runbook / add-item` (Tier 4). They construct the
// canonical `TicketSource*Record` shapes from primitive parameters and call
// the same upsert helpers used by `import_ticket_source_skill_bundle`. This
// is the only way to grow a skillbook / runbook turn-by-turn without
// preparing a full bundle directory on disk first.

/// Create or upsert a `knowledge_main_skills` row. Returns the
/// `TicketSourceMainSkillRecord` JSON view including the stored fields.
///
/// Optional contract fields default to empty / null when the caller cannot
/// fill them yet. The agent fills them progressively over later turns by
/// calling this same function again with the same `main_skill_id`.
pub(crate) fn create_or_update_main_skill(
    root: &Path,
    main_skill_id: &str,
    title: &str,
    primary_channel: &str,
    entry_action: &str,
    resolver_contract: Option<Value>,
    execution_contract: Option<Value>,
    resolve_flow: Vec<String>,
    writeback_flow: Vec<String>,
    linked_skillbooks: Vec<String>,
    linked_runbooks: Vec<String>,
) -> Result<Value> {
    anyhow::ensure!(!main_skill_id.trim().is_empty(), "main_skill_id required");
    anyhow::ensure!(!title.trim().is_empty(), "title required");
    anyhow::ensure!(
        !primary_channel.trim().is_empty(),
        "primary_channel required"
    );
    anyhow::ensure!(!entry_action.trim().is_empty(), "entry_action required");
    let record = TicketSourceMainSkillRecord {
        main_skill_id: main_skill_id.to_string(),
        title: title.to_string(),
        primary_channel: primary_channel.to_string(),
        entry_action: entry_action.to_string(),
        resolver_contract: resolver_contract.unwrap_or_else(|| Value::Object(Default::default())),
        execution_contract: execution_contract.unwrap_or_else(|| Value::Object(Default::default())),
        resolve_flow,
        writeback_flow,
        linked_skillbooks,
        linked_runbooks,
    };
    let now = now_iso_string();
    let conn = open_ticket_db(root)?;
    upsert_ticket_source_main_skill(&conn, &record, &now)?;
    Ok(serde_json::to_value(&record)?)
}

/// Create or upsert a `knowledge_skillbooks` row. The summary column is
/// auto-derived from `mission` via `summarize_text`; callers don't have to
/// supply it. Status is always `active` (matching the bundle-import path).
pub(crate) fn create_or_update_skillbook(
    root: &Path,
    skillbook_id: &str,
    title: &str,
    version: &str,
    mission: &str,
    runtime_policy: &str,
    answer_contract: &str,
    non_negotiable_rules: Vec<String>,
    workflow_backbone: Vec<String>,
    routing_taxonomy: Vec<String>,
    linked_runbooks: Vec<String>,
) -> Result<Value> {
    anyhow::ensure!(!skillbook_id.trim().is_empty(), "skillbook_id required");
    anyhow::ensure!(!title.trim().is_empty(), "title required");
    anyhow::ensure!(!version.trim().is_empty(), "version required");
    anyhow::ensure!(!mission.trim().is_empty(), "mission required");
    let record = TicketSourceSkillbookRecord {
        skillbook_id: skillbook_id.to_string(),
        title: title.to_string(),
        version: version.to_string(),
        mission: mission.to_string(),
        non_negotiable_rules,
        runtime_policy: runtime_policy.to_string(),
        answer_contract: answer_contract.to_string(),
        workflow_backbone,
        routing_taxonomy,
        linked_runbooks,
    };
    let now = now_iso_string();
    let conn = open_ticket_db(root)?;
    upsert_ticket_source_skillbook(&conn, &record, &now)?;
    Ok(serde_json::to_value(&record)?)
}

/// Create or upsert a `knowledge_runbooks` row.
pub(crate) fn create_or_update_runbook(
    root: &Path,
    runbook_id: &str,
    skillbook_id: &str,
    title: &str,
    version: &str,
    status: &str,
    problem_domain: &str,
    item_labels: Vec<String>,
) -> Result<Value> {
    anyhow::ensure!(!runbook_id.trim().is_empty(), "runbook_id required");
    anyhow::ensure!(!skillbook_id.trim().is_empty(), "skillbook_id required");
    anyhow::ensure!(!title.trim().is_empty(), "title required");
    anyhow::ensure!(!version.trim().is_empty(), "version required");
    anyhow::ensure!(!status.trim().is_empty(), "status required");
    let record = TicketSourceRunbookRecord {
        runbook_id: runbook_id.to_string(),
        skillbook_id: skillbook_id.to_string(),
        title: title.to_string(),
        version: version.to_string(),
        status: status.to_string(),
        problem_domain: problem_domain.to_string(),
        item_labels,
    };
    let now = now_iso_string();
    let conn = open_ticket_db(root)?;
    upsert_ticket_source_runbook(&conn, &record, &now)?;
    Ok(serde_json::to_value(&record)?)
}

/// Add (or update) a single labeled runbook item, and — unless
/// `skip_embedding` — refresh its embedding row through the standard
/// auxiliary embedding backend. The runbook's `item_labels` list is also
/// refreshed defensively so listing the runbook surfaces the new label.
pub(crate) fn add_or_update_runbook_item(
    root: &Path,
    item_id: &str,
    runbook_id: &str,
    skillbook_id: &str,
    label: &str,
    title: &str,
    problem_class: &str,
    chunk_text: &str,
    version: &str,
    status: &str,
    embedding_model_override: Option<&str>,
    skip_embedding: bool,
) -> Result<Value> {
    anyhow::ensure!(!item_id.trim().is_empty(), "item_id required");
    anyhow::ensure!(!runbook_id.trim().is_empty(), "runbook_id required");
    anyhow::ensure!(!skillbook_id.trim().is_empty(), "skillbook_id required");
    anyhow::ensure!(!label.trim().is_empty(), "label required");
    anyhow::ensure!(!title.trim().is_empty(), "title required");
    anyhow::ensure!(!problem_class.trim().is_empty(), "problem_class required");
    anyhow::ensure!(!chunk_text.trim().is_empty(), "chunk_text required");
    anyhow::ensure!(!version.trim().is_empty(), "version required");
    anyhow::ensure!(!status.trim().is_empty(), "status required");

    let record = TicketSourceRunbookItemRecord {
        item_id: item_id.to_string(),
        runbook_id: runbook_id.to_string(),
        skillbook_id: skillbook_id.to_string(),
        label: label.to_string(),
        title: title.to_string(),
        problem_class: problem_class.to_string(),
        trigger_phrases: Vec::new(),
        entry_conditions: Vec::new(),
        earliest_blocker: String::new(),
        expected_guidance: String::new(),
        tool_actions: Value::Object(Default::default()),
        verification: Vec::new(),
        writeback_policy: Value::Object(Default::default()),
        escalate_when: Vec::new(),
        sources: Value::Object(Default::default()),
        pages: Vec::new(),
        chunk_text: chunk_text.to_string(),
    };

    let now = now_iso_string();
    let conn = open_ticket_db(root)?;
    upsert_ticket_source_runbook_item(&conn, &record, version, status, &now)?;
    let labels = refresh_runbook_item_labels(&conn, runbook_id, &now)?;

    let (embedding_status, embedding_model_used) = if skip_embedding {
        ("skipped", None)
    } else {
        let model = embedding_model_override
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(ToOwned::to_owned)
            .unwrap_or_else(default_ticket_skill_embedding_model);
        let inputs = vec![chunk_text.to_string()];
        match embed_texts_for_ticket_skills(root, &inputs, &model) {
            Ok(vectors) => {
                if let Some(vector) = vectors.first() {
                    upsert_ticket_source_embedding(&conn, item_id, &model, vector, &now)?;
                }
                ("indexed", Some(model))
            }
            Err(err) => {
                // Surface the error to the caller without rolling back the
                // item write — the row is durable, only the embedding is
                // missing. The agent can rerun `refresh-item-embedding` once
                // the backend is back.
                eprintln!("warning: embedding refresh for runbook item {item_id} failed: {err:#}");
                ("error", None)
            }
        }
    };

    Ok(json!({
        "item": record,
        "embedding": {
            "status": embedding_status,
            "model": embedding_model_used,
        },
        "runbook": {
            "runbook_id": runbook_id,
            "item_labels": labels,
        },
    }))
}

/// Recompute the embedding for an existing runbook item. Idempotent.
pub(crate) fn refresh_runbook_item_embedding(
    root: &Path,
    item_id: &str,
    embedding_model_override: Option<&str>,
) -> Result<Value> {
    anyhow::ensure!(!item_id.trim().is_empty(), "item_id required");
    let conn = open_ticket_db(root)?;
    let chunk_text: String = conn
        .query_row(
            "SELECT chunk_text FROM knowledge_runbook_items WHERE item_id = ?1",
            params![item_id],
            |row| row.get(0),
        )
        .optional()
        .context("query chunk_text for runbook item")?
        .with_context(|| format!("runbook item not found: {item_id}"))?;
    let model = embedding_model_override
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .unwrap_or_else(default_ticket_skill_embedding_model);
    let inputs = vec![chunk_text];
    let vectors = embed_texts_for_ticket_skills(root, &inputs, &model)?;
    let now = now_iso_string();
    if let Some(vector) = vectors.first() {
        upsert_ticket_source_embedding(&conn, item_id, &model, vector, &now)?;
    } else {
        anyhow::bail!("embedding backend returned no vector for item {item_id}");
    }
    Ok(json!({
        "ok": true,
        "item_id": item_id,
        "embedding_model": model,
        "updated_at": now,
    }))
}

/// Pull the current set of labels for the runbook from
/// `knowledge_runbook_items` and write the deduplicated, label-sorted list
/// back into `knowledge_runbooks.item_labels_json`. Returns the new list.
pub(super) fn refresh_runbook_item_labels(
    conn: &Connection,
    runbook_id: &str,
    now: &str,
) -> Result<Vec<String>> {
    let mut stmt = conn.prepare(
        "SELECT DISTINCT label FROM knowledge_runbook_items WHERE runbook_id = ?1 ORDER BY label ASC",
    )?;
    let labels: Vec<String> = stmt
        .query_map(params![runbook_id], |row| row.get::<_, String>(0))?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    drop(stmt);
    conn.execute(
        "UPDATE knowledge_runbooks SET item_labels_json = ?1, updated_at = ?2 WHERE runbook_id = ?3",
        params![serde_json::to_string(&labels)?, now, runbook_id],
    )?;
    Ok(labels)
}

pub(super) fn load_ticket_source_main_skill_from_conn(
    conn: &Connection,
    main_skill_id: &str,
) -> Result<Option<TicketSourceMainSkillRecord>> {
    conn.query_row(
        r#"
        SELECT main_skill_id, title, primary_channel, entry_action, resolver_contract_json,
               execution_contract_json, resolve_flow_json, writeback_flow_json,
               linked_skillbooks_json, linked_runbooks_json
        FROM knowledge_main_skills
        WHERE main_skill_id = ?1
        LIMIT 1
        "#,
        params![main_skill_id],
        |row| {
            Ok(TicketSourceMainSkillRecord {
                main_skill_id: row.get(0)?,
                title: row.get(1)?,
                primary_channel: row.get(2)?,
                entry_action: row.get(3)?,
                resolver_contract: parse_json_column(row.get::<_, String>(4)?),
                execution_contract: parse_json_column(row.get::<_, String>(5)?),
                resolve_flow: parse_json_string_column(row.get::<_, String>(6)?),
                writeback_flow: parse_json_string_column(row.get::<_, String>(7)?),
                linked_skillbooks: parse_json_string_column(row.get::<_, String>(8)?),
                linked_runbooks: parse_json_string_column(row.get::<_, String>(9)?),
            })
        },
    )
    .optional()
    .map_err(anyhow::Error::from)
}

pub(super) fn load_ticket_source_skillbook_from_conn(
    conn: &Connection,
    skillbook_id: &str,
) -> Result<Option<TicketSourceSkillbookRecord>> {
    conn.query_row(
        r#"
        SELECT skillbook_id, title, version, mission, non_negotiable_rules_json, runtime_policy,
               answer_contract, workflow_backbone_json, routing_taxonomy_json, linked_runbooks_json
        FROM knowledge_skillbooks
        WHERE skillbook_id = ?1
        LIMIT 1
        "#,
        params![skillbook_id],
        |row| {
            Ok(TicketSourceSkillbookRecord {
                skillbook_id: row.get(0)?,
                title: row.get(1)?,
                version: row.get(2)?,
                mission: row.get(3)?,
                non_negotiable_rules: parse_json_string_column(row.get::<_, String>(4)?),
                runtime_policy: row.get(5)?,
                answer_contract: row.get(6)?,
                workflow_backbone: parse_json_string_column(row.get::<_, String>(7)?),
                routing_taxonomy: parse_json_string_column(row.get::<_, String>(8)?),
                linked_runbooks: parse_json_string_column(row.get::<_, String>(9)?),
            })
        },
    )
    .optional()
    .map_err(anyhow::Error::from)
}

pub(super) fn load_ticket_source_runbook_items_for_runbooks(
    conn: &Connection,
    runbook_ids: &[String],
) -> Result<Vec<TicketSourceRunbookItemRecord>> {
    let mut statement = conn.prepare(
        r#"
        SELECT structured_json
        FROM knowledge_runbook_items
        ORDER BY runbook_id ASC, label ASC
        "#,
    )?;
    let rows = statement.query_map([], |row| row.get::<_, String>(0))?;
    let filter = runbook_ids.iter().cloned().collect::<BTreeSet<_>>();
    let mut items = Vec::new();
    for row in rows {
        let raw = row?;
        let item: TicketSourceRunbookItemRecord = serde_json::from_str(&raw)?;
        if filter.contains(&item.runbook_id) {
            items.push(item);
        }
    }
    Ok(items)
}

pub(super) fn load_ticket_source_embeddings_for_items(
    conn: &Connection,
    item_ids: &[String],
) -> Result<std::collections::BTreeMap<String, (String, Vec<f64>)>> {
    let mut statement = conn.prepare(
        r#"
        SELECT item_id, embedding_model, embedding_json
        FROM knowledge_embeddings
        ORDER BY updated_at DESC
        "#,
    )?;
    let rows = statement.query_map([], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, String>(2)?,
        ))
    })?;
    let filter = item_ids.iter().cloned().collect::<BTreeSet<_>>();
    let mut map = std::collections::BTreeMap::new();
    for row in rows {
        let (item_id, model, raw_embedding) = row?;
        if !filter.contains(&item_id) || map.contains_key(&item_id) {
            continue;
        }
        let vector = serde_json::from_str::<Vec<f64>>(&raw_embedding).unwrap_or_default();
        if !vector.is_empty() {
            map.insert(item_id, (model, vector));
        }
    }
    Ok(map)
}

pub(super) fn summarize_text(text: &str, limit: usize) -> String {
    let compact = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if compact.chars().count() <= limit {
        compact
    } else {
        compact.chars().take(limit).collect()
    }
}

pub(super) fn build_ticket_source_skill_query_text(ticket: &TicketItemView) -> String {
    let title = ticket.title.trim();
    let body = ticket.body_text.trim();
    if body.is_empty() {
        return title.to_string();
    }
    let compact_body = body.split_whitespace().collect::<Vec<_>>().join(" ");
    let clipped = compact_body.chars().take(260).collect::<String>();
    format!("{title}. {clipped}")
}

pub(super) fn shorten_review_excerpt(text: &str, limit: usize) -> String {
    let compact = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if compact.chars().count() <= limit {
        compact
    } else {
        compact
            .chars()
            .take(limit.saturating_sub(3))
            .collect::<String>()
            + "..."
    }
}

pub(super) fn lexical_overlap_ratio(left: &str, right: &str) -> f64 {
    let token_re = Regex::new(r"[A-Za-zÄÖÜäöüß0-9._/-]{3,}").expect("static token regex");
    let left_tokens = token_re
        .find_iter(left)
        .map(|m| m.as_str().to_lowercase())
        .collect::<BTreeSet<_>>();
    let right_tokens = token_re
        .find_iter(right)
        .map(|m| m.as_str().to_lowercase())
        .collect::<BTreeSet<_>>();
    if left_tokens.is_empty() || right_tokens.is_empty() {
        return 0.0;
    }
    let union = left_tokens.union(&right_tokens).count();
    if union == 0 {
        return 0.0;
    }
    left_tokens.intersection(&right_tokens).count() as f64 / union as f64
}

pub(super) fn normalized_text(text: &str) -> String {
    text.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

pub(crate) fn review_ticket_note_with_source_skill(
    root: &Path,
    ticket_key: &str,
    body: &str,
    top_k: usize,
) -> Result<TicketSourceSkillNoteReviewView> {
    let ticket = load_ticket(root, ticket_key)?.context("ticket not found")?;
    let query = build_ticket_source_skill_query_text(&ticket);
    let payload = query_ticket_source_skill(root, &ticket.source_system, &query, top_k)?;
    let binding_result = payload.get("result").cloned().unwrap_or_else(|| json!({}));
    let top_family = binding_result
        .get("families")
        .and_then(Value::as_array)
        .and_then(|items| items.first())
        .cloned()
        .unwrap_or_else(|| json!({}));
    let matched_family = top_family
        .get("family_key")
        .and_then(Value::as_str)
        .map(ToOwned::to_owned);
    let matched_family_score = top_family.get("score").and_then(Value::as_f64);
    let decision = top_family
        .get("decision_support")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    let operator_summary = decision
        .get("operator_summary")
        .and_then(Value::as_str)
        .map(ToOwned::to_owned);
    let note_guidance = decision
        .get("note_guidance")
        .and_then(Value::as_str)
        .map(ToOwned::to_owned);

    let mut findings = Vec::new();
    let mut language_clean = true;
    let mut copy_safe = true;
    let note = body.trim();

    if note.len() < 24 {
        findings.push(TicketSourceSkillNoteReviewFinding {
            kind: "too_short".to_string(),
            excerpt: shorten_review_excerpt(note, 80),
            details: "The internal note is too short to explain concrete ticket progress."
                .to_string(),
        });
    }
    let concise = note.len() <= 420;
    if !concise {
        findings.push(TicketSourceSkillNoteReviewFinding {
            kind: "too_long".to_string(),
            excerpt: shorten_review_excerpt(note, 120),
            details: "The internal note is too long for a concise desk update.".to_string(),
        });
    }

    let leak_patterns = [
        (
            "internal_field_names",
            Regex::new(
                r"`(?:triage_focus|handling_steps|decision_support|operator_summary|family_key|historical_examples|close_when|note_guidance|caution_signals)`",
            )
            .expect("static leak regex"),
            "Avoid quoting internal skill field names in ticket communication.",
        ),
        (
            "code_style_identifiers",
            Regex::new(r"`[a-z0-9]+(?:_[a-z0-9]+){1,}`").expect("static code regex"),
            "Avoid code-like identifiers or schema names in the ticket note.",
        ),
        (
            "tooling_terms",
            Regex::new(r"\b(?:sqlite|json dump|parser|yaml|tooling internals|reference commands|ctox ticket)\b")
                .expect("static tooling regex"),
            "Avoid tooling or storage jargon in the ticket note.",
        ),
    ];
    for (kind, pattern, details) in leak_patterns {
        if let Some(hit) = pattern.find(note) {
            language_clean = false;
            findings.push(TicketSourceSkillNoteReviewFinding {
                kind: kind.to_string(),
                excerpt: shorten_review_excerpt(hit.as_str(), 80),
                details: details.to_string(),
            });
        }
    }

    let normalized_note = normalized_text(note);
    for source in operator_summary.iter().chain(note_guidance.iter()) {
        let normalized_source = normalized_text(source);
        let copied_by_overlap = lexical_overlap_ratio(note, source) >= 0.72;
        let copied_by_substring =
            !normalized_source.is_empty() && normalized_note.contains(&normalized_source);
        if copied_by_overlap || copied_by_substring {
            copy_safe = false;
            findings.push(TicketSourceSkillNoteReviewFinding {
                kind: "copied_skill_language".to_string(),
                excerpt: shorten_review_excerpt(source, 100),
                details: "The note is too close to the desk-skill guidance; write it freshly in desk language.".to_string(),
            });
        }
    }

    let grounded_in_title = lexical_overlap_ratio(note, &ticket.title) >= 0.08;
    let grounded_in_body = lexical_overlap_ratio(note, &ticket.body_text) >= 0.08;
    let grounded_in_ticket = grounded_in_title || grounded_in_body;
    if !grounded_in_ticket {
        findings.push(TicketSourceSkillNoteReviewFinding {
            kind: "not_ticket_grounded".to_string(),
            excerpt: shorten_review_excerpt(note, 100),
            details: "The note does not mention ticket-specific terms strongly enough.".to_string(),
        });
    }

    Ok(TicketSourceSkillNoteReviewView {
        source_system: ticket.source_system,
        ticket_key: ticket.ticket_key,
        query,
        matched_family,
        matched_family_score,
        desk_ready: language_clean
            && copy_safe
            && concise
            && grounded_in_ticket
            && note.len() >= 24,
        language_clean,
        copy_safe,
        concise,
        grounded_in_ticket,
        findings,
        note_guidance,
        operator_summary,
    })
}

pub(crate) fn suggested_skill_for_live_ticket_source(
    root: &Path,
    event: &RoutedTicketEvent,
) -> Result<Option<String>> {
    let explicit_self_work = suggested_skill_for_routed_event(root, event)?;
    if explicit_self_work.is_some() {
        return Ok(explicit_self_work);
    }
    let conn = open_ticket_db(root)?;
    Ok(
        load_active_ticket_source_skill_binding_from_conn(&conn, &event.source_system)?
            .map(|binding| binding.skill_name),
    )
}

pub(super) fn default_skill_for_self_work_kind(kind: &str) -> Option<String> {
    let kind = kind.trim();
    if kind.is_empty() {
        return None;
    }
    match kind {
        "access-request" => Some("ticket-access-and-secrets".to_string()),
        "system-onboarding" => Some("system-onboarding".to_string()),
        "secret-hygiene" => Some("secret-hygiene".to_string()),
        "mission-follow-up" | "timeout-continuation" | "review-rework" => {
            Some("follow-up-orchestrator".to_string())
        }
        WORKFLOW_CASE_KIND | WORKFLOW_STEP_KIND => Some(WORKFLOW_ORCHESTRATOR_SKILL.to_string()),
        _ => None,
    }
}
