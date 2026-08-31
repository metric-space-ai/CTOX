// Origin: CTOX
// License: AGPL-3.0-only

// App-Testing der Business-OS-Dienstebene: Bench, Referenzsuche, Validator,
// Smoke und E2E. Reiner Umzug aus business_os.rs — kein Koerper wurde
// umformuliert. Anlass: die Datei stand mit 7187 Produktionszeilen ueber
// ihrem Budget von 7106, und der Waechter, der das gemeldet haette, war seit
// 94251e6be still deaktiviert.

use super::business_os::{
    args_have_help, existing_dir_path, flag_value, now_ms, BUSINESS_OS_APP_BENCH_EVIDENCE_DIR,
    BUSINESS_OS_APP_BENCH_SKILL, BUSINESS_OS_APP_BENCH_SOURCE, BUSINESS_OS_APP_BENCH_USAGE,
    BUSINESS_OS_APP_CANDIDATES, BUSINESS_OS_APP_REFERENCE_DEFAULT_LIMIT,
    BUSINESS_OS_APP_REFERENCE_MAX_LIMIT,
};
use crate::mission::channels;
use crate::persistence;
use crate::skill_store;
use anyhow::Context;
use base64::Engine;
use serde::Deserialize;
use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::env;
use std::fs;
use std::path::Path;
use std::path::PathBuf;
use std::process::Command;
use std::time::SystemTime;
use std::time::UNIX_EPOCH;
use url::Url;
use uuid::Uuid;

#[derive(Clone, Copy)]
struct BusinessOsAppBenchCase {
    key: &'static str,
    title: &'static str,
    description: &'static str,
    minimum_scope: &'static str,
    automation: &'static str,
}

const BUSINESS_OS_APP_BENCH_CORE_FIVE: &[BusinessOsAppBenchCase] = &[
    BusinessOsAppBenchCase {
        key: "subscriptions",
        title: "Subscriptions",
        description: "Abo-Vertraege, MRR, renewal date, and churn risk.",
        minimum_scope: "subscription contracts, MRR, renewal date, churn risk",
        automation: "Create a CTOX follow-up for renewal or churn-risk review.",
    },
    BusinessOsAppBenchCase {
        key: "inventory",
        title: "Inventory",
        description: "Items, stock locations, minimum stock, and stock movement.",
        minimum_scope: "items, stock locations, minimum stock, stock movement",
        automation: "Create a CTOX follow-up for low-stock review.",
    },
    BusinessOsAppBenchCase {
        key: "projects",
        title: "Projects",
        description: "Time/material vs fixed-price, milestones, and budget vs actual.",
        minimum_scope: "time/material vs fixed-price, milestones, budget vs actual",
        automation: "Create a CTOX follow-up for over-budget or overdue milestone review.",
    },
    BusinessOsAppBenchCase {
        key: "contracts",
        title: "Contracts",
        description: "Customer contracts, SLA, renewal, and termination window.",
        minimum_scope: "customer contracts, SLA, renewal, termination window",
        automation: "Create a CTOX follow-up for renewal or cancellation deadline review.",
    },
    BusinessOsAppBenchCase {
        key: "quality",
        title: "Quality",
        description: "Complaints, corrective actions, audits, owner, and due date.",
        minimum_scope: "complaints, corrective actions, audits, owner, due date",
        automation: "Create a CTOX follow-up or local ticket for compliance action.",
    },
];

pub(super) fn handle_business_os_app_bench(
    root: &Path,
    args: &[String],
) -> anyhow::Result<serde_json::Value> {
    match args.first().map(String::as_str) {
        Some("run") => run_business_os_app_bench(root, &args[1..]),
        Some("status") => collect_business_os_app_bench_status(root, &args[1..]),
        Some("--help") | Some("-h") | None => Ok(serde_json::json!({
            "ok": true,
            "usage": BUSINESS_OS_APP_BENCH_USAGE
        })),
        Some(other) => anyhow::bail!("unknown business-os app bench command `{other}`"),
    }
}

pub(super) fn run_business_os_app_bench(
    root: &Path,
    args: &[String],
) -> anyhow::Result<serde_json::Value> {
    if args_have_help(args) {
        return Ok(serde_json::json!({
            "ok": true,
            "usage": BUSINESS_OS_APP_BENCH_USAGE,
            "runner_contract": {
                "creates_app_files": false,
                "repairs_app_files": false,
                "submits_real_business_commands": false,
                "install_target": "runtime-installed-module"
            }
        }));
    }
    let suite = flag_value(args, "--suite").unwrap_or("core-five");
    anyhow::ensure!(
        suite == "core-five",
        "unsupported Business OS app bench suite `{suite}`"
    );
    let model = flag_value(args, "--model").unwrap_or("minimax-m3");
    let context = flag_value(args, "--context").unwrap_or("256k");
    anyhow::ensure!(
        context == "256k" || context == "262144",
        "Business OS app bench must use the 256k context default"
    );
    let run_id = flag_value(args, "--run-id")
        .map(sanitize_bench_run_id)
        .transpose()?
        .unwrap_or_else(|| format!("r{}", now_ms()));
    let actor_id = flag_value(args, "--actor")
        .or_else(|| flag_value(args, "--actor-user"))
        .map(str::to_owned)
        .unwrap_or_else(|| {
            crate::business_os::store::session_with_persisted_user(
                root,
                crate::business_os::store::session(None, None),
            )
            .unwrap_or_else(|_| crate::business_os::store::session(None, None))
            .user
            .map(|user| user.id)
            .unwrap_or_else(|| "local-dev".to_owned())
        });
    let clean = !args.iter().any(|arg| arg == "--no-clean");
    let run_dir = root.join(BUSINESS_OS_APP_BENCH_EVIDENCE_DIR).join(&run_id);
    fs::create_dir_all(&run_dir)
        .with_context(|| format!("failed to create bench evidence dir {}", run_dir.display()))?;
    let events_path = run_dir.join("events.jsonl");
    let mut events = Vec::new();
    append_bench_event(
        &events_path,
        &serde_json::json!({
            "event": "bench_started",
            "run_id": run_id.as_str(),
            "suite": suite,
            "model": model,
            "context": context,
            "source": BUSINESS_OS_APP_BENCH_SOURCE,
            "created_at_ms": now_ms()
        }),
    )?;

    let removed_modules = if clean {
        cleanup_business_os_app_bench_modules(root)?
    } else {
        Vec::new()
    };
    append_bench_event(
        &events_path,
        &serde_json::json!({
            "event": "cleanup_finished",
            "run_id": run_id.as_str(),
            "removed_modules": removed_modules.clone(),
            "created_at_ms": now_ms()
        }),
    )?;

    for case in BUSINESS_OS_APP_BENCH_CORE_FIVE {
        let module_id = format!("bench_{}_{}", case.key, run_id);
        let command_id = format!("cmd_app_bench_{}_{}", case.key, run_id);
        let document = business_os_app_bench_command_document(
            &command_id,
            case,
            &module_id,
            suite,
            model,
            context,
            &run_id,
            actor_id.as_str(),
        );
        let accepted = crate::business_os::store::accept_rxdb_business_command(root, document)?;
        let event = serde_json::json!({
            "event": "task_submitted",
            "run_id": run_id.as_str(),
            "case": case.key,
            "module_id": module_id,
            "command_id": command_id,
            "accepted": accepted,
            "created_at_ms": now_ms()
        });
        append_bench_event(&events_path, &event)?;
        events.push(event);
    }

    let submitted = events
        .iter()
        .filter_map(|event| event.get("accepted"))
        .collect::<Vec<_>>();
    let accepted_count = submitted
        .iter()
        .filter(|accepted| {
            accepted.get("status").and_then(serde_json::Value::as_str) == Some("accepted")
        })
        .count();
    let ok = accepted_count == BUSINESS_OS_APP_BENCH_CORE_FIVE.len();
    let summary = serde_json::json!({
        "ok": ok,
        "run_id": run_id.as_str(),
        "suite": suite,
        "model": model,
        "context": context,
        "source": BUSINESS_OS_APP_BENCH_SOURCE,
        "evidence_dir": run_dir.display().to_string(),
        "events_path": events_path.display().to_string(),
        "removed_modules": removed_modules,
        "submitted_tasks": events,
        "accepted_count": accepted_count,
        "expected_count": BUSINESS_OS_APP_BENCH_CORE_FIVE.len(),
        "runner_contract": {
            "creates_app_files": false,
            "repairs_app_files": false,
            "submits_real_business_commands": true,
            "install_target": "runtime-installed-module"
        }
    });
    fs::write(
        run_dir.join("summary.json"),
        serde_json::to_vec_pretty(&summary)?,
    )
    .with_context(|| format!("failed to write bench summary in {}", run_dir.display()))?;
    append_bench_event(
        &events_path,
        &serde_json::json!({
            "event": "bench_finished",
            "run_id": run_id.as_str(),
            "ok": ok,
            "accepted_count": accepted_count,
            "expected_count": BUSINESS_OS_APP_BENCH_CORE_FIVE.len(),
            "created_at_ms": now_ms()
        }),
    )?;
    Ok(summary)
}

fn business_os_app_bench_command_document(
    command_id: &str,
    case: &BusinessOsAppBenchCase,
    module_id: &str,
    suite: &str,
    model: &str,
    context: &str,
    run_id: &str,
    actor_id: &str,
) -> serde_json::Value {
    serde_json::json!({
        "id": command_id,
        "command_id": command_id,
        "module": "creator",
        "command_type": "ctox.business_os.app.create",
        "type": "ctox.business_os.app.create",
        "record_id": module_id,
        "status": "pending_sync",
        "payload": {
            "title": format!("Build {}", case.title),
            "instruction": format!(
                "Build a small Business OS {} app for {}. Include one normal CTOX follow-up automation: {}",
                case.title,
                case.minimum_scope,
                case.automation
            ),
            "module_id": module_id,
            "app_id": module_id,
            "app_title": case.title,
            "description": case.description,
            "category": "operations",
            "install_target": "runtime-installed-module",
            "target": "app",
            "mode": "app",
            "desired_version": "0.1.0",
            "required_skills": [BUSINESS_OS_APP_BENCH_SKILL],
            "bench": {
                "suite": suite,
                "run_id": run_id,
                "case": case.key,
                "minimum_scope": case.minimum_scope,
                "required_automation": case.automation
            }
        },
        "client_context": {
            "source": BUSINESS_OS_APP_BENCH_SOURCE,
            "target": "app",
            "mode": "app",
            "module_id": module_id,
            "install_target": "runtime-installed-module",
            "required_skills": [BUSINESS_OS_APP_BENCH_SKILL],
            "bench": {
                "suite": suite,
                "run_id": run_id,
                "case": case.key,
                "model": model,
                "context": context
            },
            "actor": {
                "id": actor_id,
                "display_name": "CTOX App Bench",
                "role": "admin",
                "is_admin": true
            }
        },
        "created_at_ms": now_ms(),
        "updated_at_ms": now_ms()
    })
}

fn cleanup_business_os_app_bench_modules(root: &Path) -> anyhow::Result<Vec<String>> {
    let installed_root = root.join("runtime/business-os/installed-modules");
    if !installed_root.is_dir() {
        return Ok(Vec::new());
    }
    let mut removed = Vec::new();
    for entry in fs::read_dir(&installed_root)
        .with_context(|| format!("failed to read {}", installed_root.display()))?
    {
        let entry = entry?;
        if !entry.file_type()?.is_dir() {
            continue;
        }
        let name = entry.file_name().to_string_lossy().to_string();
        if !(name.starts_with("bench_") || name.starts_with("bench-")) {
            continue;
        }
        fs::remove_dir_all(entry.path())
            .with_context(|| format!("failed to remove bench app {}", entry.path().display()))?;
        removed.push(name);
    }
    removed.sort();
    Ok(removed)
}

fn append_bench_event(path: &Path, event: &serde_json::Value) -> anyhow::Result<()> {
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .with_context(|| format!("failed to open bench evidence {}", path.display()))?;
    let line = serde_json::to_string(event)?;
    std::io::Write::write_all(&mut file, line.as_bytes())
        .and_then(|_| std::io::Write::write_all(&mut file, b"\n"))
        .with_context(|| format!("failed to write bench evidence {}", path.display()))
}

fn sanitize_bench_run_id(raw: &str) -> anyhow::Result<String> {
    let value = raw.trim();
    anyhow::ensure!(!value.is_empty(), "bench run id must not be empty");
    anyhow::ensure!(
        value
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || ch == '_' || ch == '-'),
        "bench run id may only contain ASCII letters, digits, '_' and '-'"
    );
    Ok(value.to_string())
}

pub(super) fn collect_business_os_app_bench_status(
    root: &Path,
    args: &[String],
) -> anyhow::Result<serde_json::Value> {
    if args_have_help(args) {
        return Ok(serde_json::json!({
            "ok": true,
            "usage": BUSINESS_OS_APP_BENCH_USAGE
        }));
    }
    let run_id = flag_value(args, "--run-id")
        .or_else(|| {
            args.iter()
                .find(|arg| !arg.starts_with("--"))
                .map(String::as_str)
        })
        .context("usage: ctox business-os app bench status --run-id <id> [--validate]")?;
    let run_id = sanitize_bench_run_id(run_id)?;
    let run_dir = root.join(BUSINESS_OS_APP_BENCH_EVIDENCE_DIR).join(&run_id);
    let summary_path = run_dir.join("summary.json");
    let events_path = run_dir.join("events.jsonl");
    let summary_raw = fs::read_to_string(&summary_path)
        .with_context(|| format!("failed to read bench summary {}", summary_path.display()))?;
    let summary: serde_json::Value =
        serde_json::from_str(&summary_raw).context("bench summary is not valid JSON")?;
    let validate = args.iter().any(|arg| arg == "--validate");
    let mut apps = Vec::new();
    let mut counts = BenchStatusCounts::default();
    let submitted = summary
        .get("submitted_tasks")
        .and_then(serde_json::Value::as_array)
        .cloned()
        .unwrap_or_default();
    let expected_count = submitted.len();
    for item in submitted {
        let case = item
            .get("case")
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default();
        let module_id = item
            .get("module_id")
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default();
        let command_id = item
            .get("command_id")
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default();
        let task_id = item
            .pointer("/accepted/task_id")
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default();
        let task = if task_id.is_empty() {
            None
        } else {
            channels::load_queue_task(root, task_id)?
        };
        let route_status = task
            .as_ref()
            .map(|task| task.route_status.as_str())
            .unwrap_or("missing");
        counts.observe_route_status(route_status);
        let module_dir = root
            .join("runtime/business-os/installed-modules")
            .join(module_id);
        let artifacts = bench_app_artifact_report(&module_dir)?;
        if artifacts
            .get("exists")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false)
        {
            counts.artifact_dirs_present = counts.artifact_dirs_present.saturating_add(1);
        } else {
            counts.artifact_dirs_missing = counts.artifact_dirs_missing.saturating_add(1);
        }
        if artifacts
            .get("required_missing")
            .and_then(serde_json::Value::as_array)
            .is_some_and(|items| !items.is_empty())
        {
            counts.apps_with_missing_required_files =
                counts.apps_with_missing_required_files.saturating_add(1);
        }
        let validation = if validate && module_dir.is_dir() {
            let validator_args = vec!["--installed".to_string(), "--json".to_string()];
            match run_business_os_app_validator(root, module_id, &validator_args) {
                Ok(output) => {
                    let success = output.status.success();
                    counts.observe_validation(success);
                    serde_json::json!({
                        "ran": true,
                        "ok": success,
                        "status": output.status.code(),
                        "stdout": truncate_bench_text(&String::from_utf8_lossy(&output.stdout), 12000),
                        "stderr": truncate_bench_text(&String::from_utf8_lossy(&output.stderr), 4000)
                    })
                }
                Err(error) => {
                    counts.observe_validation(false);
                    serde_json::json!({
                        "ran": true,
                        "ok": false,
                        "error": error.to_string()
                    })
                }
            }
        } else {
            counts.validation_skipped = counts.validation_skipped.saturating_add(1);
            serde_json::json!({
                "ran": false,
                "reason": if validate { "module_dir_missing" } else { "not_requested" }
            })
        };
        apps.push(serde_json::json!({
            "case": case,
            "module_id": module_id,
            "command_id": command_id,
            "task_id": task_id,
            "queue": task.as_ref().map(|task| serde_json::json!({
                "route_status": task.route_status.as_str(),
                "status_note": task.status_note.as_deref(),
                "lease_owner": task.lease_owner.as_deref(),
                "leased_at": task.leased_at.as_deref(),
                "acked_at": task.acked_at.as_deref(),
                "created_at": task.created_at.as_str(),
                "updated_at": task.updated_at.as_str(),
                "workspace_root": task.workspace_root.as_deref(),
                "suggested_skill": task.suggested_skill.as_deref()
            })).unwrap_or_else(|| serde_json::json!({
                "route_status": "missing"
            })),
            "module_dir": module_dir.display().to_string(),
            "artifacts": artifacts,
            "validation": validation
        }));
    }
    let finished_at_ms = now_ms();
    let status_path = run_dir.join(format!("status-{finished_at_ms}.json"));
    let bench_green = expected_count > 0
        && counts.handled == expected_count
        && counts.artifact_dirs_present == expected_count
        && counts.apps_with_missing_required_files == 0
        && counts.validation_passed == expected_count;
    let needs_attention = counts.failed > 0
        || counts.blocked > 0
        || counts.cancelled > 0
        || counts.missing > 0
        || counts.other > 0
        || counts.artifact_dirs_missing > 0
        || counts.apps_with_missing_required_files > 0
        || counts.validation_failed > 0;
    let report = serde_json::json!({
        "ok": true,
        "bench_green": bench_green,
        "needs_attention": needs_attention,
        "run_id": run_id,
        "suite": summary.get("suite").cloned().unwrap_or(serde_json::Value::Null),
        "model": summary.get("model").cloned().unwrap_or(serde_json::Value::Null),
        "context": summary.get("context").cloned().unwrap_or(serde_json::Value::Null),
        "expected_count": expected_count,
        "status_collected_at_ms": finished_at_ms,
        "validate": validate,
        "counts": counts.to_json(),
        "apps": apps,
        "evidence_dir": run_dir.display().to_string(),
        "status_path": status_path.display().to_string()
    });
    fs::write(&status_path, serde_json::to_vec_pretty(&report)?)
        .with_context(|| format!("failed to write bench status {}", status_path.display()))?;
    append_bench_event(
        &events_path,
        &serde_json::json!({
            "event": "status_collected",
            "run_id": report.get("run_id").and_then(serde_json::Value::as_str).unwrap_or_default(),
            "status_path": status_path.display().to_string(),
            "validate": validate,
            "bench_green": bench_green,
            "needs_attention": needs_attention,
            "counts": report.get("counts").cloned().unwrap_or(serde_json::Value::Null),
            "created_at_ms": finished_at_ms
        }),
    )?;
    Ok(report)
}

#[derive(Default)]
struct BenchStatusCounts {
    pending: usize,
    leased: usize,
    handled: usize,
    failed: usize,
    blocked: usize,
    cancelled: usize,
    missing: usize,
    other: usize,
    validation_passed: usize,
    validation_failed: usize,
    validation_skipped: usize,
    artifact_dirs_present: usize,
    artifact_dirs_missing: usize,
    apps_with_missing_required_files: usize,
}

impl BenchStatusCounts {
    fn observe_route_status(&mut self, route_status: &str) {
        match route_status {
            "pending" => self.pending = self.pending.saturating_add(1),
            "leased" => self.leased = self.leased.saturating_add(1),
            "handled" => self.handled = self.handled.saturating_add(1),
            "failed" => self.failed = self.failed.saturating_add(1),
            "blocked" => self.blocked = self.blocked.saturating_add(1),
            "cancelled" => self.cancelled = self.cancelled.saturating_add(1),
            "missing" => self.missing = self.missing.saturating_add(1),
            _ => self.other = self.other.saturating_add(1),
        }
    }

    fn observe_validation(&mut self, success: bool) {
        if success {
            self.validation_passed = self.validation_passed.saturating_add(1);
        } else {
            self.validation_failed = self.validation_failed.saturating_add(1);
        }
    }

    fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "pending": self.pending,
            "leased": self.leased,
            "handled": self.handled,
            "failed": self.failed,
            "blocked": self.blocked,
            "cancelled": self.cancelled,
            "missing": self.missing,
            "other": self.other,
            "validation_passed": self.validation_passed,
            "validation_failed": self.validation_failed,
            "validation_skipped": self.validation_skipped,
            "artifact_dirs_present": self.artifact_dirs_present,
            "artifact_dirs_missing": self.artifact_dirs_missing,
            "apps_with_missing_required_files": self.apps_with_missing_required_files
        })
    }
}

fn bench_app_artifact_report(module_dir: &Path) -> anyhow::Result<serde_json::Value> {
    const REQUIRED: &[&str] = &[
        "module.json",
        "collections.schema.json",
        "schema.js",
        "index.html",
        "index.css",
        "index.js",
        "icon.svg",
        "locales/en.json",
        "locales/de.json",
    ];
    let mut files = Vec::new();
    if module_dir.is_dir() {
        collect_relative_files(module_dir, module_dir, &mut files)?;
    }
    files.sort();
    let file_set = files.iter().cloned().collect::<BTreeSet<_>>();
    let mut required_missing = REQUIRED
        .iter()
        .filter(|path| !file_set.iter().any(|file| file == *path))
        .map(|path| serde_json::Value::String((*path).to_string()))
        .collect::<Vec<_>>();
    let tests_present = files
        .iter()
        .any(|file| file.starts_with("tests/") && file.ends_with(".test.mjs"));
    if !tests_present {
        required_missing.push(serde_json::Value::String("tests/*.test.mjs".to_string()));
    }
    Ok(serde_json::json!({
        "exists": module_dir.is_dir(),
        "file_count": files.len(),
        "files": files,
        "tests_present": tests_present,
        "required_missing": required_missing
    }))
}

fn collect_relative_files(root: &Path, dir: &Path, output: &mut Vec<String>) -> anyhow::Result<()> {
    if output.len() >= 512 {
        return Ok(());
    }
    for entry in fs::read_dir(dir).with_context(|| format!("failed to read {}", dir.display()))? {
        let entry = entry?;
        let path = entry.path();
        let file_type = entry.file_type()?;
        if file_type.is_dir() {
            collect_relative_files(root, &path, output)?;
            continue;
        }
        if !file_type.is_file() {
            continue;
        }
        if let Ok(relative) = path.strip_prefix(root) {
            output.push(relative.to_string_lossy().replace('\\', "/"));
        }
    }
    Ok(())
}

fn truncate_bench_text(raw: &str, max_chars: usize) -> String {
    if raw.chars().count() <= max_chars {
        return raw.to_string();
    }
    let kept = raw.chars().take(max_chars).collect::<String>();
    format!("{kept}\n... truncated ...")
}

fn business_os_app_reference_limit(args: &[String]) -> anyhow::Result<Option<usize>> {
    if args.iter().any(|arg| arg == "--all") {
        return Ok(None);
    }
    let Some(raw_limit) = flag_value(args, "--limit") else {
        return Ok(Some(BUSINESS_OS_APP_REFERENCE_DEFAULT_LIMIT));
    };
    let parsed = raw_limit
        .parse::<usize>()
        .with_context(|| format!("invalid --limit value `{raw_limit}`"))?;
    anyhow::ensure!(parsed > 0, "--limit must be greater than zero");
    Ok(Some(parsed.min(BUSINESS_OS_APP_REFERENCE_MAX_LIMIT)))
}

fn business_os_app_reference_query_tokens(query: &str) -> Vec<String> {
    query
        .split(|ch: char| !ch.is_ascii_alphanumeric())
        .map(str::trim)
        .filter(|token| token.len() >= 3)
        .map(str::to_ascii_lowercase)
        .collect()
}

fn business_os_app_reference_match_score(
    query_tokens: &[String],
    id: &str,
    title: &str,
    description: &str,
    manifest_text: &str,
) -> i64 {
    if query_tokens.is_empty() {
        return 0;
    }
    let id = id.to_ascii_lowercase();
    let title = title.to_ascii_lowercase();
    let description = description.to_ascii_lowercase();
    let manifest_text = manifest_text.to_ascii_lowercase();
    query_tokens
        .iter()
        .map(|token| {
            let mut score = 0;
            if id.contains(token) {
                score += 16;
            }
            if title.contains(token) {
                score += 12;
            }
            if description.contains(token) {
                score += 6;
            }
            if manifest_text.contains(token) {
                score += 2;
            }
            score
        })
        .sum()
}

fn truncate_reference_text(raw: &str, max_chars: usize) -> String {
    if raw.chars().count() <= max_chars {
        return raw.to_string();
    }
    let kept = raw.chars().take(max_chars).collect::<String>();
    format!("{kept}...")
}

pub(super) fn business_os_app_reference_candidates(
    root: &Path,
    args: &[String],
) -> anyhow::Result<serde_json::Value> {
    let query = flag_value(args, "--query")
        .or_else(|| {
            args.iter()
                .find(|arg| !arg.starts_with("--"))
                .map(String::as_str)
        })
        .unwrap_or("")
        .trim()
        .to_owned();
    let query_tokens = business_os_app_reference_query_tokens(&query);
    let limit = business_os_app_reference_limit(args)?;
    let source_app_root = existing_dir_path(root, BUSINESS_OS_APP_CANDIDATES);
    let mut roots = vec![("source", source_app_root.join("modules"))];
    let installed_app_root =
        if root.join("runtime").exists() || root.join("runtime/business-os").exists() {
            root.join("runtime/business-os")
        } else {
            root.join("business-os")
        };
    roots.push(("installed", installed_app_root.join("installed-modules")));

    let mut modules = Vec::new();
    for (source, modules_root) in roots {
        if !modules_root.is_dir() {
            continue;
        }
        for entry in fs::read_dir(&modules_root)
            .with_context(|| format!("failed to read {}", modules_root.display()))?
        {
            let entry = entry?;
            if !entry.file_type()?.is_dir() {
                continue;
            }
            let module_dir = entry.path();
            let manifest_path = module_dir.join("module.json");
            if !manifest_path.is_file() {
                continue;
            }
            let manifest_text = fs::read_to_string(&manifest_path)
                .with_context(|| format!("failed to read {}", manifest_path.display()))?;
            let manifest: serde_json::Value = serde_json::from_str(&manifest_text)
                .with_context(|| format!("failed to parse {}", manifest_path.display()))?;
            let fallback_id = entry.file_name().to_string_lossy().to_string();
            let id = manifest
                .get("id")
                .and_then(serde_json::Value::as_str)
                .unwrap_or(fallback_id.as_str())
                .to_owned();
            if id.trim().is_empty() {
                continue;
            }
            let title = manifest
                .get("title")
                .and_then(serde_json::Value::as_str)
                .unwrap_or(id.as_str())
                .to_owned();
            let description = manifest
                .get("description")
                .or_else(|| manifest.get("store").and_then(|store| store.get("summary")))
                .and_then(serde_json::Value::as_str)
                .unwrap_or("")
                .to_owned();
            let match_score = business_os_app_reference_match_score(
                &query_tokens,
                &id,
                &title,
                &description,
                &manifest_text,
            );
            if !query_tokens.is_empty() && match_score <= 0 {
                continue;
            }
            let category = manifest
                .get("category")
                .cloned()
                .unwrap_or(serde_json::Value::Null);
            let reference_kind = business_os_app_reference_kind(&id, &category);
            let layout = business_os_app_reference_layout(manifest.get("layout"));
            let warnings = business_os_app_reference_warnings(&manifest, &reference_kind);
            modules.push(serde_json::json!({
                "id": id,
                "title": title,
                "description": truncate_reference_text(&description, 240),
                "source": source,
                "reference_kind": reference_kind,
                "recommended_for_generated_business_app": reference_kind == "business-workflow-reference",
                "match_score": match_score,
                "path": module_dir.display().to_string(),
                "manifest_path": manifest_path.display().to_string(),
                "entry": manifest.get("entry").cloned().unwrap_or(serde_json::Value::Null),
                "collections": manifest.get("collections").cloned().unwrap_or_else(|| serde_json::json!([])),
                "layout": layout,
                "category": category,
                "warnings": warnings,
            }));
        }
    }
    let rank_by_query = !query_tokens.is_empty();
    modules.sort_by(|a, b| {
        if rank_by_query {
            let a_score = a
                .get("match_score")
                .and_then(serde_json::Value::as_i64)
                .unwrap_or(0);
            let b_score = b
                .get("match_score")
                .and_then(serde_json::Value::as_i64)
                .unwrap_or(0);
            let score_cmp = b_score.cmp(&a_score);
            if score_cmp != std::cmp::Ordering::Equal {
                return score_cmp;
            }
        }
        let a_recommended = a
            .get("recommended_for_generated_business_app")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false);
        let b_recommended = b
            .get("recommended_for_generated_business_app")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false);
        if a_recommended != b_recommended {
            return b_recommended.cmp(&a_recommended);
        }
        let a_title = a
            .get("title")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("");
        let b_title = b
            .get("title")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("");
        a_title.cmp(b_title).then_with(|| {
            let a_id = a
                .get("id")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("");
            let b_id = b
                .get("id")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("");
            a_id.cmp(b_id)
        })
    });
    let total_matches = modules.len();
    if let Some(limit) = limit {
        modules.truncate(limit);
    }
    Ok(serde_json::json!({
        "ok": true,
        "query": query,
        "query_tokens": query_tokens,
        "total_matches": total_matches,
        "returned": modules.len(),
        "limit": limit,
        "truncated": limit.is_some_and(|limit| total_matches > limit),
        "instruction": "Choose the three most relevant business-workflow references yourself by matching workflow, data shape, and UI shape. Internal shell/developer modules are poor defaults unless the requested app is itself a shell/developer tool.",
        "usage": "Use --query with workflow/data keywords and inspect the returned candidates. Use --all only for manual debugging, not inside normal app-creation sessions.",
        "runtime_rules": [
            "Do not copy source manifest entry paths. Runtime apps use entry installed-modules/<module-id>/index.html.",
            "Do not copy layout.icon_svg or any inline SVG from source manifests. Runtime apps keep SVG markup in icon.svg.",
            "Do not copy store.installable into runtime-installed module.json.",
            "Do not copy layout.right unless the app truly needs a third pane and module.json includes layout.third_pane_justification.",
            "The skill contract and validator override any source reference field that conflicts with runtime-installed app rules."
        ],
        "runtime_manifest_contract": {
            "entry": "installed-modules/<module-id>/index.html",
            "install_scope": "installed",
            "icon": "Use icon.svg. Do not copy layout.icon_svg or inline SVG into module.json.",
            "store": "Do not set store.installable for runtime-installed modules.",
            "layout": "Prefer left + center or a modal/drawer. Use layout.right only with layout.third_pane_justification."
        },
        "modules": modules,
    }))
}

fn business_os_app_reference_kind(id: &str, category: &serde_json::Value) -> &'static str {
    let category = category.as_str().unwrap_or("").trim().to_ascii_lowercase();
    if matches!(
        id,
        "app-store" | "browser" | "coding-agents" | "creator" | "credentials" | "ctox"
    ) || matches!(
        category.as_str(),
        "development" | "security" | "system" | "workspace"
    ) {
        "internal-shell-reference"
    } else {
        "business-workflow-reference"
    }
}

fn business_os_app_reference_layout(layout: Option<&serde_json::Value>) -> serde_json::Value {
    let Some(layout) = layout.and_then(serde_json::Value::as_object) else {
        return serde_json::Value::Null;
    };
    let mut output = serde_json::Map::new();
    for key in [
        "shell",
        "launch_kind",
        "left",
        "center",
        "top",
        "bottom",
        "drawers",
        "third_pane_justification",
    ] {
        if let Some(value) = layout.get(key) {
            output.insert(key.to_string(), value.clone());
        }
    }
    if let Some(value) = layout.get("right") {
        output.insert("right".to_string(), value.clone());
        output.insert(
            "right_pane_is_exception".to_string(),
            serde_json::Value::Bool(true),
        );
    }
    output.insert(
        "icon_source".to_string(),
        serde_json::Value::String("icon.svg for generated runtime apps".to_string()),
    );
    serde_json::Value::Object(output)
}

fn business_os_app_reference_warnings(
    manifest: &serde_json::Value,
    reference_kind: &str,
) -> Vec<&'static str> {
    let mut warnings = Vec::new();
    if reference_kind != "business-workflow-reference" {
        warnings.push(
            "Internal shell/developer module: inspect sparingly; do not use as a default business-app UI template.",
        );
    }
    if manifest.pointer("/layout/icon_svg").is_some() {
        warnings.push(
            "Source manifest contains layout.icon_svg; generated runtime apps must use icon.svg instead.",
        );
    }
    if manifest.pointer("/store/installable").is_some() {
        warnings.push(
            "Source manifest contains store.installable; runtime-installed module.json must not copy it.",
        );
    }
    if manifest.pointer("/layout/right").is_some()
        && manifest
            .pointer("/layout/third_pane_justification")
            .is_none()
    {
        warnings.push(
            "Source manifest uses layout.right without third_pane_justification; generated apps should prefer two panes or modal/detail workflows.",
        );
    }
    warnings
}

pub(super) fn run_business_os_app_validator(
    root: &Path,
    module_id: &str,
    args: &[String],
) -> anyhow::Result<std::process::Output> {
    if module_id.is_empty()
        || module_id == "."
        || module_id == ".."
        || module_id.contains('/')
        || module_id.contains('\\')
    {
        anyhow::bail!("invalid Business OS app module id `{module_id}`");
    }
    let script = root.join("src/apps/business-os/scripts/validate-app-module.mjs");
    anyhow::ensure!(
        script.is_file(),
        "Business OS app validator is not available at {}",
        script.display()
    );
    let mut command = Command::new(resolve_business_os_validator_node(root));
    command.current_dir(root).arg(script).arg(module_id);
    let mut workspace_root = root.to_path_buf();
    let mut has_mode = false;
    let mut idx = 0;
    while idx < args.len() {
        match args[idx].as_str() {
            "--installed" | "--source" | "--json" | "--skip-tests" | "--skip-node-check" => {
                if args[idx] == "--installed" || args[idx] == "--source" {
                    has_mode = true;
                }
                command.arg(&args[idx]);
                idx += 1;
            }
            "--workspace" => {
                let value = args
                    .get(idx + 1)
                    .with_context(|| format!("{} requires a value", args[idx]))?;
                workspace_root = PathBuf::from(value);
                idx += 2;
            }
            "--task-id" | "--reason" => {
                idx += 2;
            }
            value if value.starts_with("--") => {
                anyhow::bail!("unsupported business-os app validator option `{value}`")
            }
            _ => {
                idx += 1;
            }
        }
    }
    if !has_mode
        && workspace_root
            .join("runtime/business-os/installed-modules")
            .is_dir()
    {
        command.arg("--installed");
    }
    command.arg("--workspace").arg(&workspace_root);
    command
        .output()
        .context("failed to run Business OS app validator")
}

pub(super) fn run_business_os_app_smoke(
    root: &Path,
    module_id: &str,
    args: &[String],
) -> anyhow::Result<std::process::Output> {
    if module_id.is_empty()
        || module_id == "."
        || module_id == ".."
        || module_id.contains('/')
        || module_id.contains('\\')
    {
        anyhow::bail!("invalid Business OS app module id `{module_id}`");
    }
    let script = root.join("src/apps/business-os/scripts/smoke-app-module.mjs");
    anyhow::ensure!(
        script.is_file(),
        "Business OS app browser smoke is not available at {}",
        script.display()
    );
    let mut command = Command::new(resolve_business_os_validator_node(root));
    command.current_dir(root).arg(script).arg(module_id);
    let caller_cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let mut idx = 0;
    while idx < args.len() {
        match args[idx].as_str() {
            "--installed" | "--source" | "--json" => {
                command.arg(&args[idx]);
                idx += 1;
            }
            "--url" | "--create-action" | "--timeout-ms" | "--output" | "--screenshot" => {
                let value = args
                    .get(idx + 1)
                    .with_context(|| format!("{} requires a value", args[idx]))?;
                command.arg(&args[idx]).arg(app_browser_evidence_arg(
                    &args[idx],
                    value,
                    &caller_cwd,
                ));
                idx += 2;
            }
            value if value.starts_with("--") => {
                anyhow::bail!("unsupported business-os app smoke option `{value}`")
            }
            _ => {
                idx += 1;
            }
        }
    }
    command
        .output()
        .context("failed to run Business OS app browser smoke")
}

pub(super) fn run_business_os_app_e2e(
    root: &Path,
    module_id: &str,
    args: &[String],
) -> anyhow::Result<std::process::Output> {
    if module_id.is_empty()
        || module_id == "."
        || module_id == ".."
        || module_id.contains('/')
        || module_id.contains('\\')
    {
        anyhow::bail!("invalid Business OS app module id `{module_id}`");
    }
    let script = root.join("src/apps/business-os/scripts/e2e-app-module.mjs");
    anyhow::ensure!(
        script.is_file(),
        "Business OS app browser E2E is not available at {}",
        script.display()
    );
    let mut command = Command::new(resolve_business_os_validator_node(root));
    command.current_dir(root).arg(script).arg(module_id);
    let caller_cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let mut idx = 0;
    while idx < args.len() {
        match args[idx].as_str() {
            "--installed" | "--source" | "--json" | "--require-scenario" => {
                command.arg(&args[idx]);
                idx += 1;
            }
            "--url" | "--timeout-ms" | "--output" | "--screenshot" | "--marker" | "--profile"
            | "--scenario" => {
                let value = args
                    .get(idx + 1)
                    .with_context(|| format!("{} requires a value", args[idx]))?;
                command.arg(&args[idx]).arg(app_browser_evidence_arg(
                    &args[idx],
                    value,
                    &caller_cwd,
                ));
                idx += 2;
            }
            value if value.starts_with("--") => {
                anyhow::bail!("unsupported business-os app e2e option `{value}`")
            }
            _ => {
                idx += 1;
            }
        }
    }
    command
        .output()
        .context("failed to run Business OS app browser E2E")
}

#[derive(Debug, Clone, Serialize)]
struct BusinessOsAppAuditTarget {
    version: &'static str,
    target_kind: &'static str,
    module_id: String,
    mode: String,
    source_root: String,
    manifest_path: String,
    source_sha256: String,
    shell_url: String,
    route_fragment: String,
    scanner_boundary: &'static str,
}

pub(crate) fn run_business_os_app_audit(
    root: &Path,
    module_id: &str,
    args: &[String],
) -> anyhow::Result<Value> {
    validate_business_os_app_module_id(module_id)?;
    validate_app_audit_args(args)?;
    let profile = app_audit_flag_value(args, "--profile").unwrap_or("quick");
    anyhow::ensure!(
        matches!(profile, "quick" | "release" | "full"),
        "business-os app audit --profile must be quick, release, or full"
    );
    let mode = if args.iter().any(|arg| arg == "--source") {
        "source"
    } else {
        "installed"
    };
    anyhow::ensure!(
        !(mode == "source" && args.iter().any(|arg| arg == "--installed")),
        "business-os app audit accepts only one of --source or --installed"
    );
    let shell_url = app_audit_flag_value(args, "--url")
        .unwrap_or("http://127.0.0.1:8765")
        .to_string();
    validate_app_audit_url(&shell_url, "--url")?;
    let deployed_url = app_audit_flag_value(args, "--deployed-url").map(str::to_owned);
    if let Some(url) = deployed_url.as_deref() {
        validate_app_audit_url(url, "--deployed-url")?;
        anyhow::ensure!(
            Url::parse(url)? != Url::parse(&shell_url)?,
            "--deployed-url must identify an isolated deployment, not the Business OS shell URL"
        );
    }
    let active = args.iter().any(|arg| arg == "--active");
    let approval_id = app_audit_flag_value(args, "--approval-id");
    if active {
        anyhow::ensure!(
            deployed_url.is_some(),
            "active app auditing requires --deployed-url; the Business OS #module route is not an isolated HTTP target"
        );
        anyhow::ensure!(
            approval_id.is_some(),
            "active app auditing requires --approval-id"
        );
    }

    let source_root = resolve_app_audit_source_root(root, module_id, mode)?;
    let manifest_path = source_root.join("module.json");
    let source_sha256 = hash_app_audit_source(&source_root)?;
    let target = BusinessOsAppAuditTarget {
        version: "ctox.appsec.target.v1",
        target_kind: "ctox-business-os-app",
        module_id: module_id.to_string(),
        mode: mode.to_string(),
        source_root: source_root.display().to_string(),
        manifest_path: manifest_path.display().to_string(),
        source_sha256: source_sha256.clone(),
        shell_url: shell_url.clone(),
        route_fragment: format!("#{module_id}"),
        scanner_boundary: "The Business OS hash route is browser context, not an isolated HTTP scanner target. Generic HTTP scanners may target only an explicitly supplied --deployed-url.",
    };
    let state_dir = app_audit_state_dir(root, module_id, &source_sha256);
    fs::create_dir_all(&state_dir)
        .with_context(|| format!("failed to create app audit state {}", state_dir.display()))?;
    write_app_audit_json(
        &state_dir.join("target.json"),
        &serde_json::to_value(&target)?,
    )?;

    let mode_flag = format!("--{mode}");
    let validator =
        run_business_os_app_validator(root, module_id, &[mode_flag.clone(), "--json".to_string()])?;
    let validation = app_audit_process_result("validate", &validator);

    let smoke = run_business_os_app_smoke(
        root,
        module_id,
        &[
            mode_flag.clone(),
            "--json".to_string(),
            "--url".to_string(),
            shell_url.clone(),
        ],
    )?;
    let smoke_result = app_audit_process_result("smoke", &smoke);

    let e2e_result = if profile == "quick" {
        serde_json::json!({
            "ok": true,
            "status": "not-required",
            "reason": "quick profile proves validation and generic mount health only"
        })
    } else {
        let e2e = run_business_os_app_e2e(
            root,
            module_id,
            &[
                mode_flag,
                "--json".to_string(),
                "--url".to_string(),
                shell_url.clone(),
                "--profile".to_string(),
                profile.to_string(),
                "--require-scenario".to_string(),
            ],
        )?;
        app_audit_process_result("e2e", &e2e)
    };

    let local_ok = app_audit_step_ok(&validation)
        && app_audit_step_ok(&smoke_result)
        && app_audit_step_ok(&e2e_result);
    let security = if profile == "quick" {
        serde_json::json!({
            "ok": true,
            "status": "not-run",
            "coverage_gap": "Source security and deployment review are outside the quick profile."
        })
    } else {
        run_app_audit_security(
            root,
            &state_dir,
            module_id,
            &source_root,
            profile,
            deployed_url.as_deref(),
            active,
            approval_id,
        )?
    };
    let security_review = security
        .pointer("/review/completion_review")
        .cloned()
        .unwrap_or(Value::Null);
    let security_closable = security_review.get("closable").and_then(Value::as_bool) == Some(true)
        && security_review.get("blocker_count").and_then(Value::as_u64) == Some(0);
    let workflow_closable = local_ok && (profile == "quick" || security_closable);
    let release_decision = if !local_ok {
        "blocked"
    } else if profile == "quick" {
        "not-assessed"
    } else if !security_closable {
        "incomplete"
    } else {
        security
            .pointer("/report/release_decision/decision")
            .and_then(Value::as_str)
            .or_else(|| {
                security
                    .pointer("/report/release_decision")
                    .and_then(Value::as_str)
            })
            .or_else(|| security.pointer("/report/decision").and_then(Value::as_str))
            .unwrap_or("review-required")
    };
    let blockers = app_audit_blockers(&validation, &smoke_result, &e2e_result, &security);
    let aggregate = serde_json::json!({
        "ok": workflow_closable,
        "command": "business-os app audit",
        "version": "ctox.business_os.app_audit.v1",
        "profile": profile,
        "state_dir": state_dir,
        "target": target,
        "stages": {
            "validation": validation,
            "smoke": smoke_result,
            "e2e": e2e_result,
            "security": security,
        },
        "completion_review": {
            "closable": workflow_closable,
            "blocker_count": blockers.len(),
            "blockers": blockers,
        },
        "release_decision": release_decision,
        "go_live_approved": security.pointer("/report/release_decision/go_live_approved").and_then(Value::as_bool).unwrap_or(false)
            && matches!(release_decision, "ready" | "approved"),
    });
    write_app_audit_json(&state_dir.join("app-audit.json"), &aggregate)?;
    Ok(aggregate)
}

fn validate_business_os_app_module_id(module_id: &str) -> anyhow::Result<()> {
    anyhow::ensure!(
        !module_id.is_empty()
            && module_id != "."
            && module_id != ".."
            && module_id.len() <= 160
            && module_id
                .chars()
                .all(|ch| ch.is_ascii_alphanumeric() || ch == '-' || ch == '_'),
        "invalid Business OS app module id `{module_id}`"
    );
    Ok(())
}

fn resolve_app_audit_source_root(
    root: &Path,
    module_id: &str,
    mode: &str,
) -> anyhow::Result<PathBuf> {
    let candidate = if mode == "source" {
        crate::business_os::store::resolve_business_os_app_root(root)?
            .join("modules")
            .join(module_id)
    } else {
        crate::business_os::store::resolve_business_os_installed_app_root(root)
            .join("installed-modules")
            .join(module_id)
    };
    let candidate_metadata = fs::symlink_metadata(&candidate).with_context(|| {
        format!(
            "module `{module_id}` was not found in {mode} mode at {}",
            candidate.display()
        )
    })?;
    anyhow::ensure!(
        candidate_metadata.is_dir() && !candidate_metadata.file_type().is_symlink(),
        "app audit source must be a real module directory, not a symlink"
    );
    anyhow::ensure!(
        candidate.join("module.json").is_file(),
        "module `{module_id}` was not found in {mode} mode"
    );
    let expected_parent = candidate
        .parent()
        .context("app audit source has no parent")?
        .canonicalize()
        .with_context(|| {
            format!(
                "failed to canonicalize app audit parent {}",
                candidate.display()
            )
        })?;
    let canonical = candidate.canonicalize().with_context(|| {
        format!(
            "failed to canonicalize app audit source {}",
            candidate.display()
        )
    })?;
    anyhow::ensure!(
        canonical.parent() == Some(expected_parent.as_path()),
        "app audit source must be a direct module directory and may not escape through symlinks"
    );
    Ok(canonical)
}

fn hash_app_audit_source(source_root: &Path) -> anyhow::Result<String> {
    let mut files = Vec::new();
    collect_app_audit_files(source_root, source_root, &mut files)?;
    anyhow::ensure!(files.len() <= 4096, "app audit source exceeds 4096 files");
    files.sort();
    let mut hasher = Sha256::new();
    let mut total = 0u64;
    for path in files {
        let metadata = fs::symlink_metadata(&path)?;
        anyhow::ensure!(
            !metadata.file_type().is_symlink(),
            "app audit source may not contain symlinks: {}",
            path.display()
        );
        total = total.saturating_add(metadata.len());
        anyhow::ensure!(
            total <= 128 * 1024 * 1024,
            "app audit source exceeds 128 MiB"
        );
        let relative = path.strip_prefix(source_root)?;
        hasher.update(relative.to_string_lossy().as_bytes());
        hasher.update([0]);
        hasher.update(metadata.len().to_le_bytes());
        hasher.update(fs::read(&path)?);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn collect_app_audit_files(
    root: &Path,
    current: &Path,
    out: &mut Vec<PathBuf>,
) -> anyhow::Result<()> {
    for entry in fs::read_dir(current)? {
        let entry = entry?;
        let path = entry.path();
        let metadata = fs::symlink_metadata(&path)?;
        anyhow::ensure!(
            !metadata.file_type().is_symlink(),
            "app audit source may not contain symlinks: {}",
            path.display()
        );
        if metadata.is_dir() {
            collect_app_audit_files(root, &path, out)?;
        } else if metadata.is_file() {
            anyhow::ensure!(path.starts_with(root), "app audit source escaped its root");
            out.push(path);
        }
    }
    Ok(())
}

fn app_audit_state_dir(root: &Path, module_id: &str, sha256: &str) -> PathBuf {
    let runtime = if root.file_name().and_then(|name| name.to_str()) == Some("runtime") {
        root.to_path_buf()
    } else {
        root.join("runtime")
    };
    runtime
        .join("appsec")
        .join("apps")
        .join(module_id)
        .join(&sha256[..16])
}

fn run_app_audit_security(
    root: &Path,
    state_dir: &Path,
    module_id: &str,
    source_root: &Path,
    profile: &str,
    deployed_url: Option<&str>,
    active: bool,
    approval_id: Option<&str>,
) -> anyhow::Result<Value> {
    let security_state = state_dir.join("security");
    let tools_root = if root.file_name().and_then(|name| name.to_str()) == Some("runtime") {
        root.join("tools/appsec")
    } else {
        root.join("runtime/tools/appsec")
    };
    let scanner_profile = if profile == "full" {
        "full"
    } else {
        "standard"
    };
    let mut init_args = vec![
        "init".to_string(),
        "--name".to_string(),
        format!("CTOX app {module_id}"),
        "--profile".to_string(),
        scanner_profile.to_string(),
        "--target".to_string(),
        source_root.display().to_string(),
        "--json".to_string(),
    ];
    if let Some(url) = deployed_url {
        init_args.extend(["--url".to_string(), url.to_string()]);
    }
    let init = run_raw_appsec(root, &security_state, &tools_root, init_args)?;
    let assessment = if let Some(url) = deployed_url {
        let mut args = vec![
            "audit".to_string(),
            "run".to_string(),
            "--url".to_string(),
            url.to_string(),
            "--source".to_string(),
            source_root.display().to_string(),
            "--profile".to_string(),
            scanner_profile.to_string(),
            "--json".to_string(),
        ];
        if active {
            args.extend([
                "--active".to_string(),
                "--approval-id".to_string(),
                approval_id.unwrap_or_default().to_string(),
            ]);
        }
        run_raw_appsec(root, &security_state, &tools_root, args)?
    } else {
        run_raw_appsec(
            root,
            &security_state,
            &tools_root,
            vec![
                "assess".to_string(),
                "--profile".to_string(),
                scanner_profile.to_string(),
                "--target".to_string(),
                source_root.display().to_string(),
                "--no-authz".to_string(),
                "--json".to_string(),
            ],
        )?
    };
    let artifact_audit = run_raw_appsec(
        root,
        &security_state,
        &tools_root,
        vec![
            "artifact".to_string(),
            "audit".to_string(),
            "--json".to_string(),
        ],
    )?;
    let initial_review = run_raw_appsec(
        root,
        &security_state,
        &tools_root,
        vec!["review".to_string(), "--json".to_string()],
    )?;
    if initial_review
        .pointer("/completion_review/closable")
        .and_then(Value::as_bool)
        != Some(true)
    {
        return Ok(serde_json::json!({
            "ok": true,
            "status": "incomplete",
            "state_dir": security_state,
            "init": init,
            "assessment": assessment,
            "artifact_audit": artifact_audit,
            "review": initial_review,
            "report": Value::Null,
        }));
    }

    let finish = run_raw_appsec(
        root,
        &security_state,
        &tools_root,
        vec![
            "finish".to_string(),
            "--executive-summary".to_string(),
            format!("Evidence-gated CTOX Business OS app audit for module {module_id}."),
            "--methodology".to_string(),
            format!("Server-resolved source snapshot, browser smoke, declared end-to-end scenarios, and the CTOX AppSec {scanner_profile} profile."),
            "--technical-analysis".to_string(),
            "Only evidence-backed findings and retained proof artifacts contribute to the release decision.".to_string(),
            "--recommendations".to_string(),
            "Remediate every validated release blocker and rerun the complete audit before deployment.".to_string(),
            "--json".to_string(),
        ],
    )?;
    let final_artifact_audit = run_raw_appsec(
        root,
        &security_state,
        &tools_root,
        vec![
            "artifact".to_string(),
            "audit".to_string(),
            "--json".to_string(),
        ],
    )?;
    let review = run_raw_appsec(
        root,
        &security_state,
        &tools_root,
        vec!["review".to_string(), "--json".to_string()],
    )?;
    let final_closable = review
        .pointer("/completion_review/closable")
        .and_then(Value::as_bool)
        == Some(true)
        && review
            .pointer("/completion_review/blocker_count")
            .and_then(Value::as_u64)
            == Some(0);
    let (report, report_markdown) = if final_closable {
        let reports_dir = security_state.join("reports");
        (
            run_raw_appsec(
                root,
                &security_state,
                &tools_root,
                vec![
                    "report".to_string(),
                    "--format".to_string(),
                    "json".to_string(),
                    "--out".to_string(),
                    reports_dir
                        .join("deployment-audit-report.json")
                        .display()
                        .to_string(),
                    "--gate".to_string(),
                    "go-live".to_string(),
                    "--json".to_string(),
                ],
            )?,
            run_raw_appsec(
                root,
                &security_state,
                &tools_root,
                vec![
                    "report".to_string(),
                    "--format".to_string(),
                    "markdown".to_string(),
                    "--out".to_string(),
                    reports_dir
                        .join("deployment-audit-report.md")
                        .display()
                        .to_string(),
                    "--json".to_string(),
                ],
            )?,
        )
    } else {
        (Value::Null, Value::Null)
    };
    Ok(serde_json::json!({
        "ok": true,
        "status": if final_closable { "complete" } else { "incomplete" },
        "state_dir": security_state,
        "init": init,
        "assessment": assessment,
        "artifact_audit": final_artifact_audit,
        "initial_artifact_audit": artifact_audit,
        "finish": finish,
        "review": review,
        "report": report,
        "report_markdown": report_markdown,
    }))
}

fn run_raw_appsec(
    root: &Path,
    state_dir: &Path,
    tools_root: &Path,
    command: Vec<String>,
) -> anyhow::Result<Value> {
    let mut forwarded = vec![
        "ctox-app-audit".to_string(),
        "--state-dir".to_string(),
        state_dir.display().to_string(),
        "--tools-root".to_string(),
        tools_root.display().to_string(),
    ];
    forwarded.extend(command);
    let output = ctox_appsec_pentest::run_cli_json(forwarded.clone(), Some(root.to_path_buf()))?;
    let _ = crate::appsec_state::project_cli_result(root, &forwarded, &output)?;
    Ok(output)
}

fn app_audit_process_result(stage: &str, output: &std::process::Output) -> Value {
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
    let parsed = serde_json::from_str::<Value>(&stdout).unwrap_or(Value::Null);
    serde_json::json!({
        "ok": output.status.success() && parsed.get("ok").and_then(Value::as_bool) != Some(false),
        "stage": stage,
        "exit_code": output.status.code(),
        "result": parsed,
        "stderr": stderr.chars().take(4000).collect::<String>(),
    })
}

fn app_audit_step_ok(value: &Value) -> bool {
    value.get("ok").and_then(Value::as_bool) == Some(true)
}

fn app_audit_blockers(
    validation: &Value,
    smoke: &Value,
    e2e: &Value,
    security: &Value,
) -> Vec<Value> {
    let mut blockers = Vec::new();
    for (name, value) in [("validation", validation), ("smoke", smoke), ("e2e", e2e)] {
        if !app_audit_step_ok(value) {
            blockers.push(serde_json::json!({ "kind": "app-stage-failed", "stage": name }));
        }
    }
    if security.get("status").and_then(Value::as_str) == Some("incomplete") {
        let nested = security
            .pointer("/review/completion_review/blockers")
            .cloned()
            .unwrap_or_else(|| serde_json::json!([]));
        blockers.push(serde_json::json!({
            "kind": "security-audit-incomplete",
            "blockers": nested,
        }));
    }
    blockers
}

fn write_app_audit_json(path: &Path, value: &Value) -> anyhow::Result<()> {
    fs::write(path, serde_json::to_vec_pretty(value)?)
        .with_context(|| format!("failed to write app audit artifact {}", path.display()))
}

fn app_audit_flag_value<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    args.iter()
        .position(|arg| arg == flag)
        .and_then(|index| args.get(index + 1))
        .map(String::as_str)
}

fn validate_app_audit_url(value: &str, flag: &str) -> anyhow::Result<()> {
    let url = Url::parse(value).with_context(|| format!("{flag} must be a valid URL"))?;
    anyhow::ensure!(
        matches!(url.scheme(), "http" | "https"),
        "{flag} must use http or https"
    );
    anyhow::ensure!(url.host_str().is_some(), "{flag} must include a host");
    anyhow::ensure!(
        url.username().is_empty() && url.password().is_none(),
        "{flag} must not embed credentials"
    );
    anyhow::ensure!(
        url.query().is_none() && url.fragment().is_none(),
        "{flag} must not contain a query or fragment"
    );
    Ok(())
}

fn validate_app_audit_args(args: &[String]) -> anyhow::Result<()> {
    let mut seen = std::collections::HashSet::new();
    let mut index = 0usize;
    while index < args.len() {
        let flag = args[index].as_str();
        anyhow::ensure!(
            matches!(
                flag,
                "--source"
                    | "--installed"
                    | "--active"
                    | "--profile"
                    | "--url"
                    | "--deployed-url"
                    | "--approval-id"
            ),
            "unsupported business-os app audit option `{flag}`"
        );
        anyhow::ensure!(
            seen.insert(flag.to_string()),
            "duplicate app audit option `{flag}`"
        );
        if matches!(
            flag,
            "--profile" | "--url" | "--deployed-url" | "--approval-id"
        ) {
            let value = args
                .get(index + 1)
                .with_context(|| format!("{flag} requires a value"))?;
            anyhow::ensure!(
                !value.trim().is_empty() && !value.starts_with("--"),
                "{flag} requires a value"
            );
            index += 2;
        } else {
            index += 1;
        }
    }
    Ok(())
}

pub(super) fn app_browser_evidence_arg(flag: &str, value: &str, caller_cwd: &Path) -> String {
    if flag != "--output" && flag != "--screenshot" {
        return value.to_string();
    }
    let path = PathBuf::from(value);
    if path.is_absolute() {
        value.to_string()
    } else {
        caller_cwd.join(path).to_string_lossy().into_owned()
    }
}

pub(crate) fn resolve_business_os_validator_node(_root: &Path) -> PathBuf {
    let mut candidates = Vec::new();
    if let Ok(path) = env::var("PATH") {
        candidates.extend(env::split_paths(&path).map(|dir| dir.join("node")));
    }
    candidates.extend([
        PathBuf::from("/opt/homebrew/bin/node"),
        PathBuf::from("/usr/local/bin/node"),
        PathBuf::from("/usr/bin/node"),
    ]);
    candidates
        .into_iter()
        .find(|path| path.is_file())
        .unwrap_or_else(|| PathBuf::from("node"))
}

pub(super) fn app_validator_args_from_finalize_args(args: &[String]) -> Vec<String> {
    let mut out = Vec::new();
    let mut skip_next = false;
    for arg in args.iter().skip(2) {
        if skip_next {
            skip_next = false;
            continue;
        }
        match arg.as_str() {
            "--task-id" | "--reason" => skip_next = true,
            "--installed" | "--source" | "--json" | "--skip-tests" | "--skip-node-check" => {
                out.push(arg.clone())
            }
            _ => {}
        }
    }
    if !out
        .iter()
        .any(|arg| arg == "--installed" || arg == "--source")
    {
        out.push("--installed".to_string());
    }
    out
}

#[cfg(test)]
mod app_audit_tests {
    use super::{
        app_audit_state_dir, hash_app_audit_source, resolve_app_audit_source_root,
        run_business_os_app_audit, validate_app_audit_args, validate_app_audit_url,
        validate_business_os_app_module_id,
    };
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn target_contract_rejects_path_like_ids_and_non_http_urls() {
        for invalid in ["", ".", "..", "../app", "nested/app", "app name"] {
            assert!(
                validate_business_os_app_module_id(invalid).is_err(),
                "{invalid}"
            );
        }
        assert!(validate_business_os_app_module_id("customer-portal_v2").is_ok());
        assert!(validate_app_audit_url("https://example.test/app", "--url").is_ok());
        assert!(validate_app_audit_url("file:///etc/passwd", "--url").is_err());
        assert!(validate_app_audit_url("javascript:alert(1)", "--url").is_err());
        assert!(validate_app_audit_url("https://user:secret@example.test", "--url").is_err());
        assert!(validate_app_audit_url("https://example.test/?token=secret", "--url").is_err());
        assert!(validate_app_audit_args(&["--unknown".into()]).is_err());
        assert!(validate_app_audit_args(&["--profile".into(), "--active".into()]).is_err());

        let temp = tempdir().expect("temporary app audit root");
        let error = run_business_os_app_audit(
            temp.path(),
            "sample-app",
            &[
                "--url".into(),
                "https://example.test/app".into(),
                "--deployed-url".into(),
                "https://example.test/app".into(),
            ],
        )
        .expect_err("Business OS shell must not be accepted as scanner target");
        assert!(
            error.to_string().contains("isolated deployment"),
            "{error:#}"
        );
    }

    #[test]
    fn source_hash_is_content_bound_and_namespaces_state() -> anyhow::Result<()> {
        let temp = tempdir()?;
        let source = temp.path().join("sample-app");
        fs::create_dir_all(&source)?;
        fs::write(source.join("module.json"), r#"{"id":"sample-app"}"#)?;
        fs::write(source.join("index.js"), "export const version = 1;\n")?;
        let first = hash_app_audit_source(&source)?;
        let first_state = app_audit_state_dir(temp.path(), "sample-app", &first);

        fs::write(source.join("index.js"), "export const version = 2;\n")?;
        let second = hash_app_audit_source(&source)?;
        assert_ne!(first, second);
        assert_ne!(
            first_state,
            app_audit_state_dir(temp.path(), "sample-app", &second)
        );
        assert!(first_state.ends_with(&first[..16]));
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn source_hash_fails_closed_on_symlinks() -> anyhow::Result<()> {
        use std::os::unix::fs::symlink;

        let temp = tempdir()?;
        let source = temp.path().join("sample-app");
        fs::create_dir_all(&source)?;
        fs::write(source.join("module.json"), r#"{"id":"sample-app"}"#)?;
        fs::write(temp.path().join("outside.js"), "secret")?;
        symlink(temp.path().join("outside.js"), source.join("index.js"))?;
        let error = hash_app_audit_source(&source).expect_err("symlink must fail closed");
        assert!(error.to_string().contains("symlink"), "{error:#}");

        let installed = temp.path().join("business-os/installed-modules");
        let real_module = installed.join("real-module");
        fs::create_dir_all(&real_module)?;
        fs::write(real_module.join("module.json"), r#"{"id":"real-module"}"#)?;
        symlink(&real_module, installed.join("linked-module"))?;
        let error = resolve_app_audit_source_root(temp.path(), "linked-module", "installed")
            .expect_err("module root symlink must fail closed");
        assert!(error.to_string().contains("not a symlink"), "{error:#}");
        Ok(())
    }
}
