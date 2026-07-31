// CLI dispatch for `ctox ticket ...`: flag parsing, usage texts and the
// subcommand fan-out into the ticket store.

use super::{
    ack_leased_ticket_events, append_ticket_self_work_note, apply_ticket_workflow_delta,
    assign_ticket_self_work_item, close_case, compose_ticket_source_skill_reply, create_dry_run,
    create_learning_candidate, create_ticket_clarification_request, create_ticket_knowledge_load,
    decide_case_approval, decide_learning_candidate, default_execution_actions,
    export_ticket_history_dataset, find_flag_value, flag_present,
    import_ticket_source_skill_bundle, lease_pending_ticket_events, list_audit_records,
    list_autonomy_grants, list_cases, list_control_bundles, list_learning_candidates,
    list_ticket_history, list_ticket_knowledge_entries, list_ticket_self_work_assignments,
    list_ticket_self_work_items, list_ticket_self_work_notes, list_ticket_source_controls,
    list_ticket_source_skill_bindings, list_tickets, load_case, load_latest_dry_run_for_case,
    load_ticket, load_ticket_knowledge_entry, load_ticket_label_assignment,
    load_ticket_self_work_item, load_ticket_workflow,
    materialize_ready_workflow_steps_for_workflow, open_ticket_db, parse_domain_csv,
    parse_json_string_array, parse_json_value, parse_limit, positional_after_flags, print_json,
    publish_ticket_clarification_request, publish_ticket_self_work_item, put_autonomy_grant,
    put_control_bundle, put_ticket_knowledge_entry, put_ticket_self_work_item,
    put_ticket_source_skill_binding, put_ticket_workflow_step, query_ticket_source_skill,
    record_execution_action, record_verification, refresh_observed_ticket_knowledge,
    required_flag_value, resolve_ticket_clarification_request,
    resolve_ticket_source_skill_for_target, review_ticket_note_with_source_skill, set_ticket_label,
    show_ticket_source_skill, start_ticket_workflow, summarize_monitoring_snapshot,
    sync_ticket_system, test_ticket_system, ticket_system_capabilities,
    transition_ticket_self_work_item, workflow_mark_step_queue_ready, writeback_comment,
    writeback_transition, AutonomyGrantInput, ControlBundleInput, TicketClarificationRequestInput,
    TicketKnowledgeUpsertInput, TicketSelfWorkUpsertInput, TicketWorkflowStartInput,
    TicketWorkflowStepInput, WorkItemStatus, DEFAULT_APPROVAL_MODE, DEFAULT_AUDIT_LIMIT,
    DEFAULT_AUTONOMY_LEVEL, DEFAULT_LIST_LIMIT, DEFAULT_RISK_LEVEL, DEFAULT_SUPPORT_MODE,
    WORKFLOW_MATERIALIZE_DEFAULT_LIMIT, WORKFLOW_ROLE_LEAF,
};
use super::{resolve_db_path, schema_state};
use anyhow::{bail, Context, Result};
use serde_json::{json, Value};
use std::path::Path;

pub fn handle_ticket_command(root: &Path, args: &[String]) -> Result<()> {
    let command = args.first().map(String::as_str).unwrap_or("");
    match command {
        "init" => {
            let conn = open_ticket_db(root)?;
            print_json(&json!({
                "ok": true,
                "db_path": resolve_db_path(root),
                "initialized": schema_state(&conn)?,
            }))
        }
        "sync" => {
            let system = required_flag_value(args, "--system")
                .context("usage: ctox ticket sync --system <local>")?;
            let result = sync_ticket_system(root, system)?;
            print_json(&result)
        }
        "test" => {
            let system = required_flag_value(args, "--system")
                .context("usage: ctox ticket test --system <local>")?;
            let result = test_ticket_system(root, system)?;
            print_json(&result)
        }
        "capabilities" => {
            let system = required_flag_value(args, "--system")
                .context("usage: ctox ticket capabilities --system <name>")?;
            let result = ticket_system_capabilities(system)?;
            print_json(&result)
        }
        "sources" => {
            let controls = list_ticket_source_controls(root)?;
            print_json(&json!({"ok": true, "count": controls.len(), "sources": controls}))
        }
        "source-skills" => {
            let system = find_flag_value(args, "--system");
            let bindings = list_ticket_source_skill_bindings(root, system)?;
            print_json(&json!({"ok": true, "count": bindings.len(), "source_skills": bindings}))
        }
        "source-skill-set" => {
            let system = required_flag_value(args, "--system")
                .context("usage: ctox ticket source-skill-set --system <name> --skill <name> [--archetype <value>] [--status <active|inactive>] [--origin <value>] [--artifact-path <path>] [--notes <text>]")?;
            let skill = required_flag_value(args, "--skill")
                .context("usage: ctox ticket source-skill-set --system <name> --skill <name> [--archetype <value>] [--status <active|inactive>] [--origin <value>] [--artifact-path <path>] [--notes <text>]")?;
            let archetype = find_flag_value(args, "--archetype").unwrap_or("operating-model");
            let status = find_flag_value(args, "--status").unwrap_or("active");
            let origin = find_flag_value(args, "--origin").unwrap_or("ticket-onboarding");
            let artifact_path = find_flag_value(args, "--artifact-path");
            let notes = find_flag_value(args, "--notes");
            let binding = put_ticket_source_skill_binding(
                root,
                system,
                skill,
                archetype,
                status,
                origin,
                artifact_path,
                notes,
            )?;
            print_json(&json!({"ok": true, "source_skill": binding}))
        }
        "source-skill-show" => {
            let system = required_flag_value(args, "--system")
                .context("usage: ctox ticket source-skill-show --system <name>")?;
            let view = show_ticket_source_skill(root, system)?;
            print_json(&json!({"ok": true, "source_skill": view}))
        }
        "source-skill-query" => {
            let system = required_flag_value(args, "--system").context(
                "usage: ctox ticket source-skill-query --system <name> --query <text> [--top-k <n>]",
            )?;
            let query = required_flag_value(args, "--query").context(
                "usage: ctox ticket source-skill-query --system <name> --query <text> [--top-k <n>]",
            )?;
            let top_k = find_flag_value(args, "--top-k")
                .and_then(|raw| raw.parse::<usize>().ok())
                .unwrap_or(3);
            let result = query_ticket_source_skill(root, system, query, top_k)?;
            print_json(&result)
        }
        "source-skill-import-bundle" => {
            let system = required_flag_value(args, "--system").context(
                "usage: ctox ticket source-skill-import-bundle --system <name> --bundle-dir <path> [--embedding-model <model>] [--skip-embeddings]",
            )?;
            let bundle_dir = required_flag_value(args, "--bundle-dir").context(
                "usage: ctox ticket source-skill-import-bundle --system <name> --bundle-dir <path> [--embedding-model <model>] [--skip-embeddings]",
            )?;
            let result = import_ticket_source_skill_bundle(
                root,
                system,
                bundle_dir,
                find_flag_value(args, "--embedding-model"),
                flag_present(args, "--skip-embeddings"),
            )?;
            print_json(&result)
        }
        "source-skill-resolve" => {
            let top_k = find_flag_value(args, "--top-k")
                .and_then(|raw| raw.parse::<usize>().ok())
                .unwrap_or(3);
            let result = resolve_ticket_source_skill_for_target(
                root,
                find_flag_value(args, "--ticket-key"),
                find_flag_value(args, "--case-id"),
                top_k,
            )?;
            print_json(&result)
        }
        "source-skill-compose-reply" => {
            let result = compose_ticket_source_skill_reply(
                root,
                find_flag_value(args, "--ticket-key"),
                find_flag_value(args, "--case-id"),
                find_flag_value(args, "--send-policy").unwrap_or("suggestion"),
                find_flag_value(args, "--subject"),
                flag_present(args, "--body-only"),
            )?;
            match result {
                Value::String(body) => {
                    println!("{body}");
                    Ok(())
                }
                other => print_json(&other),
            }
        }
        "source-skill-review-note" => {
            let body = required_flag_value(args, "--body").context(
                "usage: ctox ticket source-skill-review-note (--ticket-key <key> | --case-id <id>) --body <text> [--top-k <n>]",
            )?;
            let top_k = find_flag_value(args, "--top-k")
                .and_then(|raw| raw.parse::<usize>().ok())
                .unwrap_or(1);
            if let Some(ticket_key) = find_flag_value(args, "--ticket-key") {
                let review = review_ticket_note_with_source_skill(root, ticket_key, body, top_k)?;
                print_json(&json!({"ok": true, "review": review}))
            } else if let Some(case_id) = find_flag_value(args, "--case-id") {
                let case = load_case(root, case_id)?.context("ticket case not found")?;
                let review =
                    review_ticket_note_with_source_skill(root, &case.ticket_key, body, top_k)?;
                print_json(&json!({"ok": true, "review": review}))
            } else {
                anyhow::bail!(
                    "usage: ctox ticket source-skill-review-note (--ticket-key <key> | --case-id <id>) --body <text> [--top-k <n>]"
                );
            }
        }
        "history-export" => {
            let system = required_flag_value(args, "--system")
                .context("usage: ctox ticket history-export --system <name> --output <path>")?;
            let output = required_flag_value(args, "--output")
                .context("usage: ctox ticket history-export --system <name> --output <path>")?;
            let result = export_ticket_history_dataset(root, system, Path::new(output))?;
            print_json(&result)
        }
        "knowledge-bootstrap" => {
            let system = required_flag_value(args, "--system")
                .context("usage: ctox ticket knowledge-bootstrap --system <name>")?;
            let entries = refresh_observed_ticket_knowledge(root, system)?;
            print_json(
                &json!({"ok": true, "system": system, "count": entries.len(), "entries": entries}),
            )
        }
        "knowledge-list" => {
            let system = find_flag_value(args, "--system");
            let domain = find_flag_value(args, "--domain");
            let status = find_flag_value(args, "--status");
            let limit = parse_limit(args, DEFAULT_LIST_LIMIT);
            let entries = list_ticket_knowledge_entries(root, system, domain, status, limit)?;
            print_json(&json!({"ok": true, "count": entries.len(), "entries": entries}))
        }
        "knowledge-show" => {
            let system = required_flag_value(args, "--system").context(
                "usage: ctox ticket knowledge-show --system <name> --domain <name> --key <value>",
            )?;
            let domain = required_flag_value(args, "--domain").context(
                "usage: ctox ticket knowledge-show --system <name> --domain <name> --key <value>",
            )?;
            let key = required_flag_value(args, "--key").context(
                "usage: ctox ticket knowledge-show --system <name> --domain <name> --key <value>",
            )?;
            let entry = load_ticket_knowledge_entry(root, system, domain, key)?
                .context("ticket knowledge entry not found")?;
            print_json(&json!({"ok": true, "entry": entry}))
        }
        "knowledge-load" => {
            let ticket_key = required_flag_value(args, "--ticket-key").context(
                "usage: ctox ticket knowledge-load --ticket-key <key> [--domains <csv>]",
            )?;
            let domains = find_flag_value(args, "--domains").map(parse_domain_csv);
            let load = create_ticket_knowledge_load(root, ticket_key, domains.as_deref())?;
            print_json(&json!({"ok": true, "knowledge_load": load}))
        }
        "monitoring-ingest" => {
            let system = required_flag_value(args, "--system").context(
                "usage: ctox ticket monitoring-ingest --system <name> --snapshot-json <json> [--key <value>] [--title <text>] [--summary <text>] [--status <value>]",
            )?;
            let snapshot_raw = required_flag_value(args, "--snapshot-json").context(
                "usage: ctox ticket monitoring-ingest --system <name> --snapshot-json <json> [--key <value>] [--title <text>] [--summary <text>] [--status <value>]",
            )?;
            let snapshot = parse_json_value(snapshot_raw)?;
            let knowledge_key = find_flag_value(args, "--key").unwrap_or("observed");
            let status = find_flag_value(args, "--status").unwrap_or("observed");
            let title = find_flag_value(args, "--title")
                .map(str::to_string)
                .unwrap_or_else(|| format!("{system} monitoring landscape"));
            let summary = find_flag_value(args, "--summary")
                .map(str::to_string)
                .unwrap_or_else(|| summarize_monitoring_snapshot(&snapshot));
            let entry = put_ticket_knowledge_entry(
                root,
                TicketKnowledgeUpsertInput {
                    source_system: system.to_string(),
                    domain: "monitoring_landscape".to_string(),
                    knowledge_key: knowledge_key.to_string(),
                    title,
                    summary,
                    status: status.to_string(),
                    content: snapshot,
                },
            )?;
            print_json(&json!({"ok": true, "entry": entry}))
        }
        "access-request-put" => {
            let system = required_flag_value(args, "--system").context(
                "usage: ctox ticket access-request-put --system <name> --title <title> --body <text> [--required-scopes <csv>] [--secret-refs <csv>] [--channels <csv>] [--skill <name>] [--metadata-json <json>] [--publish]",
            )?;
            let title = required_flag_value(args, "--title").context(
                "usage: ctox ticket access-request-put --system <name> --title <title> --body <text> [--required-scopes <csv>] [--secret-refs <csv>] [--channels <csv>] [--skill <name>] [--metadata-json <json>] [--publish]",
            )?;
            let body = required_flag_value(args, "--body").context(
                "usage: ctox ticket access-request-put --system <name> --title <title> --body <text> [--required-scopes <csv>] [--secret-refs <csv>] [--channels <csv>] [--skill <name>] [--metadata-json <json>] [--publish]",
            )?;
            let required_scopes = find_flag_value(args, "--required-scopes")
                .map(parse_domain_csv)
                .unwrap_or_default();
            let secret_refs = find_flag_value(args, "--secret-refs")
                .map(parse_domain_csv)
                .unwrap_or_default();
            let channels = find_flag_value(args, "--channels")
                .map(parse_domain_csv)
                .unwrap_or_else(|| vec!["mail".to_string()]);
            let explicit_skill = find_flag_value(args, "--skill")
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(ToOwned::to_owned);
            let mut metadata = find_flag_value(args, "--metadata-json")
                .map(parse_json_value)
                .transpose()?
                .unwrap_or_else(|| json!({}));
            if let Some(object) = metadata.as_object_mut() {
                object.insert("required_scopes".to_string(), json!(required_scopes));
                object.insert("secret_refs".to_string(), json!(secret_refs));
                object.insert("channels".to_string(), json!(channels));
                if !object.contains_key("skill") {
                    object.insert(
                        "skill".to_string(),
                        json!(
                            explicit_skill
                                .clone()
                                .unwrap_or_else(|| "ticket-access-and-secrets".to_string())
                        ),
                    );
                }
            }
            let item = put_ticket_self_work_item(
                root,
                TicketSelfWorkUpsertInput {
                    source_system: system.to_string(),
                    kind: "access-request".to_string(),
                    title: title.to_string(),
                    body_text: body.to_string(),
                    state: WorkItemStatus::Open.as_str().to_string(),
                    metadata,
                },
                flag_present(args, "--publish"),
            )?;
            print_json(&json!({"ok": true, "item": item}))
        }
        "self-work-list" | "internal-work-list" => {
            let system = find_flag_value(args, "--system");
            let state = find_flag_value(args, "--state");
            let limit = parse_limit(args, DEFAULT_LIST_LIMIT);
            let items = list_ticket_self_work_items(root, system, state, limit)?;
            print_json(&json!({"ok": true, "count": items.len(), "items": items}))
        }
        "self-work-show" | "internal-work-show" => {
            let work_id = required_flag_value(args, "--work-id")
                .context("usage: ctox ticket internal-work-show --work-id <id>")?;
            let item = load_ticket_self_work_item(root, work_id)?
                .context("ticket internal work item not found")?;
            let assignments = list_ticket_self_work_assignments(root, work_id, DEFAULT_LIST_LIMIT)?;
            let notes = list_ticket_self_work_notes(root, work_id, DEFAULT_LIST_LIMIT)?;
            print_json(
                &json!({"ok": true, "item": item, "assignments": assignments, "notes": notes}),
            )
        }
        "self-work-put" | "internal-work-put" => {
            let system = required_flag_value(args, "--system").context(
                "usage: ctox ticket internal-work-put --system <name> --kind <kind> --title <title> --body <text> [--skill <name>] [--metadata-json <json>] [--publish]",
            )?;
            let kind = required_flag_value(args, "--kind").context(
                "usage: ctox ticket internal-work-put --system <name> --kind <kind> --title <title> --body <text> [--skill <name>] [--metadata-json <json>] [--publish]",
            )?;
            let title = required_flag_value(args, "--title").context(
                "usage: ctox ticket internal-work-put --system <name> --kind <kind> --title <title> --body <text> [--skill <name>] [--metadata-json <json>] [--publish]",
            )?;
            let body = required_flag_value(args, "--body").context(
                "usage: ctox ticket internal-work-put --system <name> --kind <kind> --title <title> --body <text> [--skill <name>] [--metadata-json <json>] [--publish]",
            )?;
            let explicit_skill = find_flag_value(args, "--skill")
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(ToOwned::to_owned);
            let mut metadata = find_flag_value(args, "--metadata-json")
                .map(parse_json_value)
                .transpose()?
                .unwrap_or_else(|| json!({}));
            if let Some(skill) = explicit_skill {
                if let Some(object) = metadata.as_object_mut() {
                    object.insert("skill".to_string(), json!(skill));
                }
            }
            let item = put_ticket_self_work_item(
                root,
                TicketSelfWorkUpsertInput {
                    source_system: system.to_string(),
                    kind: kind.to_string(),
                    title: title.to_string(),
                    body_text: body.to_string(),
                    state: WorkItemStatus::Open.as_str().to_string(),
                    metadata,
                },
                flag_present(args, "--publish"),
            )?;
            print_json(&json!({"ok": true, "item": item}))
        }
        "self-work-publish" | "internal-work-publish" => {
            let work_id = required_flag_value(args, "--work-id")
                .context("usage: ctox ticket internal-work-publish --work-id <id>")?;
            let item = publish_ticket_self_work_item(root, work_id)?;
            print_json(&json!({"ok": true, "item": item}))
        }
        "self-work-assign" | "internal-work-assign" => {
            let work_id = required_flag_value(args, "--work-id").context(
                "usage: ctox ticket internal-work-assign --work-id <id> --assignee <name> [--assigned-by <actor>] [--rationale <text>]",
            )?;
            let assignee = required_flag_value(args, "--assignee").context(
                "usage: ctox ticket internal-work-assign --work-id <id> --assignee <name> [--assigned-by <actor>] [--rationale <text>]",
            )?;
            let item = assign_ticket_self_work_item(
                root,
                work_id,
                assignee,
                find_flag_value(args, "--assigned-by").unwrap_or("ctox"),
                find_flag_value(args, "--rationale"),
            )?;
            print_json(&json!({"ok": true, "item": item}))
        }
        "self-work-note" | "internal-work-note" => {
            let work_id = required_flag_value(args, "--work-id").context(
                "usage: ctox ticket internal-work-note --work-id <id> --body <text> [--authored-by <actor>] [--visibility <internal|public>]",
            )?;
            let body = required_flag_value(args, "--body").context(
                "usage: ctox ticket internal-work-note --work-id <id> --body <text> [--authored-by <actor>] [--visibility <internal|public>]",
            )?;
            let note = append_ticket_self_work_note(
                root,
                work_id,
                body,
                find_flag_value(args, "--authored-by").unwrap_or("ctox"),
                find_flag_value(args, "--visibility").unwrap_or("internal"),
            )?;
            print_json(&json!({"ok": true, "note": note}))
        }
        "self-work-transition" | "internal-work-transition" => {
            let work_id = required_flag_value(args, "--work-id").context(
                "usage: ctox ticket internal-work-transition --work-id <id> --state <value> [--transitioned-by <actor>] [--note <text>] [--visibility <internal|public>]",
            )?;
            let state = required_flag_value(args, "--state").context(
                "usage: ctox ticket internal-work-transition --work-id <id> --state <value> [--transitioned-by <actor>] [--note <text>] [--visibility <internal|public>]",
            )?;
            let item = transition_ticket_self_work_item(
                root,
                work_id,
                state,
                find_flag_value(args, "--transitioned-by").unwrap_or("ctox"),
                find_flag_value(args, "--note"),
                find_flag_value(args, "--visibility").unwrap_or("internal"),
            )?;
            print_json(&json!({"ok": true, "item": item}))
        }
        "workflow-start" => {
            let title = required_flag_value(args, "--title").context(
                "usage: ctox ticket workflow-start --title <title> --goal <text> [--system <name>] [--thread-key <key>] [--workspace-root <path>] [--skill <name>] [--priority <urgent|high|normal|low>] [--phase <name>] [--phase-goal <text>] [--exit-gate <text>] [--first-step-title <title>] [--first-step-prompt <text>] [--queue-now]",
            )?;
            let goal = required_flag_value(args, "--goal").context(
                "usage: ctox ticket workflow-start --title <title> --goal <text> [--system <name>] [--thread-key <key>] [--workspace-root <path>] [--skill <name>] [--priority <urgent|high|normal|low>] [--phase <name>] [--phase-goal <text>] [--exit-gate <text>] [--first-step-title <title>] [--first-step-prompt <text>] [--queue-now]",
            )?;
            let workflow = start_ticket_workflow(
                root,
                TicketWorkflowStartInput {
                    source_system: find_flag_value(args, "--system")
                        .unwrap_or("internal")
                        .to_string(),
                    title: title.to_string(),
                    goal: goal.to_string(),
                    thread_key: find_flag_value(args, "--thread-key").map(ToOwned::to_owned),
                    workspace_root: find_flag_value(args, "--workspace-root")
                        .map(ToOwned::to_owned),
                    skill: find_flag_value(args, "--skill").map(ToOwned::to_owned),
                    priority: find_flag_value(args, "--priority").map(ToOwned::to_owned),
                    first_phase: find_flag_value(args, "--phase")
                        .unwrap_or("plan")
                        .to_string(),
                    first_phase_goal: find_flag_value(args, "--phase-goal").map(ToOwned::to_owned),
                    first_exit_gate: find_flag_value(args, "--exit-gate").map(ToOwned::to_owned),
                    first_step_title: find_flag_value(args, "--first-step-title")
                        .map(ToOwned::to_owned),
                    first_step_prompt: find_flag_value(args, "--first-step-prompt")
                        .map(ToOwned::to_owned),
                    queue_now: flag_present(args, "--queue-now"),
                },
            )?;
            print_json(&json!({"ok": true, "workflow": workflow}))
        }
        "workflow-step-put" => {
            let workflow_id = required_flag_value(args, "--workflow-id").context(
                "usage: ctox ticket workflow-step-put --workflow-id <id> --phase <phase> --title <title> --body <text> [--step-id <id>] [--role <leaf|reducer>] [--predecessors <csv>] [--predecessor-steps <csv>] [--phase-goal <text>] [--exit-gate <text>] [--skill <name>] [--priority <urgent|high|normal|low>] [--metadata-json <json>] [--queue-now]",
            )?;
            let phase = required_flag_value(args, "--phase").context(
                "usage: ctox ticket workflow-step-put --workflow-id <id> --phase <phase> --title <title> --body <text> [--step-id <id>] [--role <leaf|reducer>] [--predecessors <csv>] [--predecessor-steps <csv>] [--phase-goal <text>] [--exit-gate <text>] [--skill <name>] [--priority <urgent|high|normal|low>] [--metadata-json <json>] [--queue-now]",
            )?;
            let title = required_flag_value(args, "--title").context(
                "usage: ctox ticket workflow-step-put --workflow-id <id> --phase <phase> --title <title> --body <text> [--step-id <id>] [--role <leaf|reducer>] [--predecessors <csv>] [--predecessor-steps <csv>] [--phase-goal <text>] [--exit-gate <text>] [--skill <name>] [--priority <urgent|high|normal|low>] [--metadata-json <json>] [--queue-now]",
            )?;
            let body = required_flag_value(args, "--body").context(
                "usage: ctox ticket workflow-step-put --workflow-id <id> --phase <phase> --title <title> --body <text> [--step-id <id>] [--role <leaf|reducer>] [--predecessors <csv>] [--predecessor-steps <csv>] [--phase-goal <text>] [--exit-gate <text>] [--skill <name>] [--priority <urgent|high|normal|low>] [--metadata-json <json>] [--queue-now]",
            )?;
            let metadata = find_flag_value(args, "--metadata-json")
                .map(parse_json_value)
                .transpose()?
                .unwrap_or_else(|| json!({}));
            let item = put_ticket_workflow_step(
                root,
                TicketWorkflowStepInput {
                    workflow_id: workflow_id.to_string(),
                    role: find_flag_value(args, "--role")
                        .unwrap_or(WORKFLOW_ROLE_LEAF)
                        .to_string(),
                    phase: phase.to_string(),
                    step_id: find_flag_value(args, "--step-id").map(ToOwned::to_owned),
                    title: title.to_string(),
                    body_text: body.to_string(),
                    phase_goal: find_flag_value(args, "--phase-goal").map(ToOwned::to_owned),
                    exit_gate: find_flag_value(args, "--exit-gate").map(ToOwned::to_owned),
                    predecessor_work_ids: find_flag_value(args, "--predecessors")
                        .map(parse_domain_csv)
                        .unwrap_or_default(),
                    predecessor_step_ids: find_flag_value(args, "--predecessor-steps")
                        .map(parse_domain_csv)
                        .unwrap_or_default(),
                    skill: find_flag_value(args, "--skill").map(ToOwned::to_owned),
                    priority: find_flag_value(args, "--priority").map(ToOwned::to_owned),
                    metadata,
                },
            )?;
            let queued = if flag_present(args, "--queue-now") {
                Some(workflow_mark_step_queue_ready(root, &item.work_id)?)
            } else {
                None
            };
            print_json(&json!({"ok": true, "item": item, "queued": queued}))
        }
        "workflow-apply-delta" => {
            let workflow_id = required_flag_value(args, "--workflow-id").context(
                "usage: ctox ticket workflow-apply-delta --workflow-id <id> --delta-json <json> [--queue-now]",
            )?;
            let delta_json = required_flag_value(args, "--delta-json").context(
                "usage: ctox ticket workflow-apply-delta --workflow-id <id> --delta-json <json> [--queue-now]",
            )?;
            let delta_value = parse_json_value(delta_json)?;
            let result = apply_ticket_workflow_delta(
                root,
                workflow_id,
                delta_value,
                flag_present(args, "--queue-now"),
            )?;
            print_json(&json!({"ok": true, "result": result}))
        }
        "workflow-materialize" => {
            let workflow_id = find_flag_value(args, "--workflow-id");
            let limit = parse_limit(args, WORKFLOW_MATERIALIZE_DEFAULT_LIMIT);
            let result = materialize_ready_workflow_steps_for_workflow(root, workflow_id, limit)?;
            print_json(&json!({"ok": true, "result": result}))
        }
        "workflow-show" => {
            let workflow_id = required_flag_value(args, "--workflow-id")
                .context("usage: ctox ticket workflow-show --workflow-id <id>")?;
            let workflow = load_ticket_workflow(root, workflow_id)?
                .context("ticket workflow not found")?;
            print_json(&json!({"ok": true, "workflow": workflow}))
        }
        "take" => {
            let limit = parse_limit(args, DEFAULT_LIST_LIMIT);
            let lease_owner = find_flag_value(args, "--lease-owner").unwrap_or("codex");
            let events = lease_pending_ticket_events(root, limit, lease_owner)?;
            print_json(&json!({"ok": true, "count": events.len(), "events": events}))
        }
        "ack" => {
            let status = required_flag_value(args, "--status").context(
                "usage: ctox ticket ack --status <handled|failed|duplicate|blocked> <event-key>...",
            )?;
            let event_keys = positional_after_flags(&args[1..]);
            if event_keys.is_empty() {
                anyhow::bail!(
                    "usage: ctox ticket ack --status <handled|failed|duplicate|blocked> <event-key>..."
                );
            }
            let updated = ack_leased_ticket_events(root, &event_keys, status)?;
            print_json(
                &json!({"ok": true, "updated": updated, "status": status, "event_keys": event_keys}),
            )
        }
        "list" => {
            let limit = parse_limit(args, DEFAULT_LIST_LIMIT);
            let system = find_flag_value(args, "--system");
            let tickets = list_tickets(root, system, limit)?;
            print_json(&json!({"ok": true, "count": tickets.len(), "tickets": tickets}))
        }
        "show" => {
            let ticket_key = required_flag_value(args, "--ticket-key")
                .context("usage: ctox ticket show --ticket-key <key>")?;
            let ticket = load_ticket(root, ticket_key)?.context("ticket not found")?;
            let label_assignment = load_ticket_label_assignment(root, ticket_key)?;
            print_json(&json!({
                "ok": true,
                "ticket": ticket,
                "label_assignment": label_assignment,
            }))
        }
        "history" => {
            let ticket_key = required_flag_value(args, "--ticket-key")
                .context("usage: ctox ticket history --ticket-key <key> [--limit <n>]")?;
            let limit = parse_limit(args, DEFAULT_LIST_LIMIT);
            let events = list_ticket_history(root, ticket_key, limit)?;
            print_json(&json!({"ok": true, "count": events.len(), "events": events}))
        }
        "label-set" => {
            let ticket_key = required_flag_value(args, "--ticket-key")
                .context("usage: ctox ticket label-set --ticket-key <key> --label <label>")?;
            let label = required_flag_value(args, "--label")
                .context("usage: ctox ticket label-set --ticket-key <key> --label <label>")?;
            let assigned_by = find_flag_value(args, "--assigned-by").unwrap_or("manual");
            let rationale = find_flag_value(args, "--rationale");
            let evidence = find_flag_value(args, "--evidence-json")
                .map(parse_json_value)
                .transpose()?
                .unwrap_or_else(|| json!({}));
            let assignment =
                set_ticket_label(root, ticket_key, label, assigned_by, rationale, evidence)?;
            print_json(&json!({"ok": true, "assignment": assignment}))
        }
        "label-show" => {
            let ticket_key = required_flag_value(args, "--ticket-key")
                .context("usage: ctox ticket label-show --ticket-key <key>")?;
            let assignment = load_ticket_label_assignment(root, ticket_key)?
                .context("ticket label assignment not found")?;
            print_json(&json!({"ok": true, "assignment": assignment}))
        }
        "bundle-put" => {
            let label = required_flag_value(args, "--label").context(
                "usage: ctox ticket bundle-put --label <label> --runbook-id <id> --policy-id <id>",
            )?;
            let runbook_id = required_flag_value(args, "--runbook-id").context(
                "usage: ctox ticket bundle-put --label <label> --runbook-id <id> --policy-id <id>",
            )?;
            let policy_id = required_flag_value(args, "--policy-id").context(
                "usage: ctox ticket bundle-put --label <label> --runbook-id <id> --policy-id <id>",
            )?;
            let actions = find_flag_value(args, "--actions")
                .map(parse_json_string_array)
                .transpose()?
                .unwrap_or_else(default_execution_actions);
            let bundle = put_control_bundle(
                root,
                ControlBundleInput {
                    label: label.to_string(),
                    runbook_id: runbook_id.to_string(),
                    runbook_version: find_flag_value(args, "--runbook-version")
                        .unwrap_or("v1")
                        .to_string(),
                    policy_id: policy_id.to_string(),
                    policy_version: find_flag_value(args, "--policy-version")
                        .unwrap_or("v1")
                        .to_string(),
                    approval_mode: find_flag_value(args, "--approval-mode")
                        .unwrap_or(DEFAULT_APPROVAL_MODE)
                        .to_string(),
                    autonomy_level: find_flag_value(args, "--autonomy-level")
                        .unwrap_or(DEFAULT_AUTONOMY_LEVEL)
                        .to_string(),
                    verification_profile_id: find_flag_value(args, "--verification-profile-id")
                        .unwrap_or("default-verification")
                        .to_string(),
                    writeback_profile_id: find_flag_value(args, "--writeback-profile-id")
                        .unwrap_or("default-writeback")
                        .to_string(),
                    support_mode: find_flag_value(args, "--support-mode")
                        .unwrap_or(DEFAULT_SUPPORT_MODE)
                        .to_string(),
                    default_risk_level: find_flag_value(args, "--risk-level")
                        .unwrap_or(DEFAULT_RISK_LEVEL)
                        .to_string(),
                    execution_actions: actions,
                    notes: find_flag_value(args, "--notes").map(ToOwned::to_owned),
                },
            )?;
            print_json(&json!({"ok": true, "bundle": bundle}))
        }
        "bundle-list" => {
            let bundles = list_control_bundles(root)?;
            print_json(&json!({"ok": true, "count": bundles.len(), "bundles": bundles}))
        }
        "autonomy-grant-set" => {
            let label = required_flag_value(args, "--label").context(
                "usage: ctox ticket autonomy-grant-set --label <label> --approval-mode <mode> --autonomy-level <level>",
            )?;
            let approval_mode = required_flag_value(args, "--approval-mode").context(
                "usage: ctox ticket autonomy-grant-set --label <label> --approval-mode <mode> --autonomy-level <level>",
            )?;
            let autonomy_level = required_flag_value(args, "--autonomy-level").context(
                "usage: ctox ticket autonomy-grant-set --label <label> --approval-mode <mode> --autonomy-level <level>",
            )?;
            let bundle_version = find_flag_value(args, "--bundle-version")
                .and_then(|value| value.parse::<i64>().ok());
            let grant = put_autonomy_grant(
                root,
                AutonomyGrantInput {
                    label: label.to_string(),
                    bundle_version,
                    approval_mode: approval_mode.to_string(),
                    autonomy_level: autonomy_level.to_string(),
                    approved_by: find_flag_value(args, "--approved-by")
                        .unwrap_or("owner")
                        .to_string(),
                    source_candidate_id: find_flag_value(args, "--candidate-id")
                        .map(ToOwned::to_owned),
                    rationale: find_flag_value(args, "--rationale").map(ToOwned::to_owned),
                },
            )?;
            print_json(&json!({"ok": true, "grant": grant}))
        }
        "autonomy-grant-list" => {
            let grants = list_autonomy_grants(root)?;
            print_json(&json!({"ok": true, "count": grants.len(), "grants": grants}))
        }
        "dry-run" => {
            let ticket_key = required_flag_value(args, "--ticket-key").context(
                "usage: ctox ticket dry-run --ticket-key <key> [--understanding <text>]",
            )?;
            let record = create_dry_run(
                root,
                ticket_key,
                find_flag_value(args, "--understanding"),
                find_flag_value(args, "--risk-level"),
            )?;
            print_json(&json!({"ok": true, "dry_run": record}))
        }
        "cases" => {
            let limit = parse_limit(args, DEFAULT_LIST_LIMIT);
            let ticket_key = find_flag_value(args, "--ticket-key");
            let cases = list_cases(root, ticket_key, limit)?;
            print_json(&json!({"ok": true, "count": cases.len(), "cases": cases}))
        }
        "case-show" => {
            let case_id = required_flag_value(args, "--case-id")
                .context("usage: ctox ticket case-show --case-id <id>")?;
            let case = load_case(root, case_id)?.context("ticket case not found")?;
            let dry_run = load_latest_dry_run_for_case(root, case_id)?;
            print_json(&json!({"ok": true, "case": case, "dry_run": dry_run}))
        }
        "approve" => {
            let case_id = required_flag_value(args, "--case-id").context(
                "usage: ctox ticket approve --case-id <id> --status <approved|rejected>",
            )?;
            let status = required_flag_value(args, "--status").context(
                "usage: ctox ticket approve --case-id <id> --status <approved|rejected>",
            )?;
            let case = decide_case_approval(
                root,
                case_id,
                status,
                find_flag_value(args, "--decided-by").unwrap_or("owner"),
                find_flag_value(args, "--rationale"),
            )?;
            print_json(&json!({"ok": true, "case": case}))
        }
        "execute" => {
            let case_id = required_flag_value(args, "--case-id")
                .context("usage: ctox ticket execute --case-id <id> --summary <text>")?;
            let summary = required_flag_value(args, "--summary")
                .context("usage: ctox ticket execute --case-id <id> --summary <text>")?;
            let case = record_execution_action(root, case_id, summary)?;
            print_json(&json!({"ok": true, "case": case}))
        }
        "verify" => {
            let case_id = required_flag_value(args, "--case-id")
                .context("usage: ctox ticket verify --case-id <id> --status <passed|failed> [--summary <text>]")?;
            let status = required_flag_value(args, "--status")
                .context("usage: ctox ticket verify --case-id <id> --status <passed|failed> [--summary <text>]")?;
            let case =
                record_verification(root, case_id, status, find_flag_value(args, "--summary"))?;
            print_json(&json!({"ok": true, "case": case}))
        }
        "clarification-request" => {
            let case_id = required_flag_value(args, "--case-id").context(
                "usage: ctox ticket clarification-request --case-id <id> --question <text> [--target-type requester|owner|internal] [--target-channel ticket|email|jami|tui] [--missing-inputs <csv>] [--publish-reviewed]",
            )?;
            let question = required_flag_value(args, "--question").context(
                "usage: ctox ticket clarification-request --case-id <id> --question <text> [--target-type requester|owner|internal] [--target-channel ticket|email|jami|tui] [--missing-inputs <csv>] [--publish-reviewed]",
            )?;
            let request = create_ticket_clarification_request(
                root,
                TicketClarificationRequestInput {
                    case_id: Some(case_id.to_string()),
                    ticket_key: None,
                    work_id: find_flag_value(args, "--work-id").map(ToOwned::to_owned),
                    target_type: find_flag_value(args, "--target-type")
                        .unwrap_or("requester")
                        .to_string(),
                    target_channel: find_flag_value(args, "--target-channel")
                        .unwrap_or("ticket")
                        .to_string(),
                    question: question.to_string(),
                    missing_inputs: find_flag_value(args, "--missing-inputs")
                        .map(parse_domain_csv)
                        .unwrap_or_default(),
                    unblock_criteria: find_flag_value(args, "--unblock-criteria")
                        .map(ToOwned::to_owned),
                    resume_state: find_flag_value(args, "--resume-state")
                        .unwrap_or("executable")
                        .to_string(),
                    created_by: find_flag_value(args, "--created-by")
                        .unwrap_or("ctox")
                        .to_string(),
                    metadata: find_flag_value(args, "--metadata-json")
                        .map(parse_json_value)
                        .transpose()?
                        .unwrap_or_else(|| json!({})),
                },
            )?;
            let clarification = if flag_present(args, "--publish-reviewed") {
                publish_ticket_clarification_request(
                    root,
                    &request.clarification_id,
                    find_flag_value(args, "--reviewed-by").unwrap_or("ctox-review"),
                    find_flag_value(args, "--review-summary")
                        .unwrap_or("Clarification question reviewed for this ticket."),
                )?
            } else {
                request
            };
            print_json(&json!({"ok": true, "clarification": clarification}))
        }
        "clarification-resolve" => {
            let clarification_id = required_flag_value(args, "--clarification-id").context(
                "usage: ctox ticket clarification-resolve --clarification-id <id> --response-key <key> [--body <text>]",
            )?;
            let response_key = required_flag_value(args, "--response-key").context(
                "usage: ctox ticket clarification-resolve --clarification-id <id> --response-key <key> [--body <text>]",
            )?;
            let clarification = resolve_ticket_clarification_request(
                root,
                clarification_id,
                response_key,
                find_flag_value(args, "--body"),
                find_flag_value(args, "--resolved-by").unwrap_or("ctox"),
            )?;
            print_json(&json!({"ok": true, "clarification": clarification}))
        }
        "learn-candidate-create" => {
            let case_id = required_flag_value(args, "--case-id").context(
                "usage: ctox ticket learn-candidate-create --case-id <id> --summary <text> [--actions <json-array>] [--evidence-json <json>]",
            )?;
            let summary = required_flag_value(args, "--summary").context(
                "usage: ctox ticket learn-candidate-create --case-id <id> --summary <text> [--actions <json-array>] [--evidence-json <json>]",
            )?;
            let actions = find_flag_value(args, "--actions")
                .map(parse_json_string_array)
                .transpose()?;
            let evidence = find_flag_value(args, "--evidence-json")
                .map(parse_json_value)
                .transpose()?;
            let candidate =
                create_learning_candidate(root, case_id, summary, actions.as_deref(), evidence)?;
            print_json(&json!({"ok": true, "candidate": candidate}))
        }
        "learn-candidate-list" => {
            let limit = parse_limit(args, DEFAULT_LIST_LIMIT);
            let candidates = list_learning_candidates(
                root,
                find_flag_value(args, "--label"),
                find_flag_value(args, "--status"),
                limit,
            )?;
            print_json(&json!({"ok": true, "count": candidates.len(), "candidates": candidates}))
        }
        "learn-candidate-decide" => {
            let candidate_id = required_flag_value(args, "--candidate-id").context(
                "usage: ctox ticket learn-candidate-decide --candidate-id <id> --status <approved|rejected>",
            )?;
            let status = required_flag_value(args, "--status").context(
                "usage: ctox ticket learn-candidate-decide --candidate-id <id> --status <approved|rejected>",
            )?;
            let candidate = decide_learning_candidate(
                root,
                candidate_id,
                status,
                find_flag_value(args, "--decided-by").unwrap_or("owner"),
                find_flag_value(args, "--notes"),
                find_flag_value(args, "--promote-autonomy-level"),
            )?;
            print_json(&json!({"ok": true, "candidate": candidate}))
        }
        "writeback-comment" => {
            let case_id = required_flag_value(args, "--case-id")
                .context("usage: ctox ticket writeback-comment --case-id <id> --body <text>")?;
            let body = required_flag_value(args, "--body")
                .context("usage: ctox ticket writeback-comment --case-id <id> --body <text>")?;
            let case = writeback_comment(root, case_id, body, flag_present(args, "--internal"))?;
            print_json(&json!({"ok": true, "case": case}))
        }
        "writeback-transition" => {
            let case_id = required_flag_value(args, "--case-id").context(
                "usage: ctox ticket writeback-transition --case-id <id> --state <value> [--body <text>] [--internal]",
            )?;
            let state = required_flag_value(args, "--state").context(
                "usage: ctox ticket writeback-transition --case-id <id> --state <value> [--body <text>] [--internal]",
            )?;
            let case = writeback_transition(
                root,
                case_id,
                state,
                find_flag_value(args, "--body"),
                flag_present(args, "--internal"),
            )?;
            print_json(&json!({"ok": true, "case": case}))
        }
        "close" => {
            let case_id = required_flag_value(args, "--case-id")
                .context("usage: ctox ticket close --case-id <id> [--summary <text>]")?;
            let case = close_case(root, case_id, find_flag_value(args, "--summary"))?;
            print_json(&json!({"ok": true, "case": case}))
        }
        "audit" => {
            let limit = parse_limit(args, DEFAULT_AUDIT_LIMIT);
            let ticket_key = find_flag_value(args, "--ticket-key");
            let records = list_audit_records(root, ticket_key, limit)?;
            print_json(&json!({"ok": true, "count": records.len(), "records": records}))
        }
        "local" => crate::mission::ticket_local_native::handle_local_command(root, &args[1..]),
        _ => anyhow::bail!(
            "usage:\n  ctox ticket init\n  ctox ticket sync --system <local|zammad>\n  ctox ticket test --system <local|zammad>\n  ctox ticket capabilities --system <name>\n  ctox ticket sources\n  ctox ticket source-skills [--system <name>]\n  ctox ticket source-skill-set --system <name> --skill <name> [--archetype <value>] [--status <active|inactive>] [--origin <value>] [--artifact-path <path>] [--notes <text>]\n  ctox ticket source-skill-show --system <name>\n  ctox ticket source-skill-query --system <name> --query <text> [--top-k <n>]\n  ctox ticket source-skill-import-bundle --system <name> --bundle-dir <path> [--embedding-model <model>] [--skip-embeddings]\n  ctox ticket source-skill-resolve (--ticket-key <key> | --case-id <id>) [--top-k <n>]\n  ctox ticket source-skill-compose-reply (--ticket-key <key> | --case-id <id>) [--send-policy <suggestion|draft|send>] [--subject <text>] [--body-only]\n  ctox ticket source-skill-review-note (--ticket-key <key> | --case-id <id>) --body <text> [--top-k <n>]\n  ctox ticket history-export --system <name> --output <path>\n  ctox ticket knowledge-bootstrap --system <name>\n  ctox ticket knowledge-list [--system <name>] [--domain <name>] [--status <value>] [--limit <n>]\n  ctox ticket knowledge-show --system <name> --domain <name> --key <value>\n  ctox ticket knowledge-load --ticket-key <key> [--domains <csv>]\n  ctox ticket monitoring-ingest --system <name> --snapshot-json <json> [--key <value>] [--title <text>] [--summary <text>] [--status <value>]\n  ctox ticket access-request-put --system <name> --title <title> --body <text> [--required-scopes <csv>] [--secret-refs <csv>] [--channels <csv>] [--skill <name>] [--metadata-json <json>] [--publish]\n  ctox ticket internal-work-put --system <name> --kind <kind> --title <title> --body <text> [--skill <name>] [--metadata-json <json>] [--publish]\n  ctox ticket internal-work-show --work-id <id>\n  ctox ticket internal-work-publish --work-id <id>\n  ctox ticket internal-work-assign --work-id <id> --assignee <name> [--assigned-by <actor>] [--rationale <text>]\n  ctox ticket internal-work-note --work-id <id> --body <text> [--authored-by <actor>] [--visibility <internal|public>]\n  ctox ticket internal-work-transition --work-id <id> --state <value> [--transitioned-by <actor>] [--note <text>] [--visibility <internal|public>]\n  ctox ticket internal-work-list [--system <name>] [--state <value>] [--limit <n>]\n  ctox ticket take [--lease-owner <owner>] [--limit <n>]\n  ctox ticket ack --status <handled|failed|duplicate|blocked> <event-key>...\n  ctox ticket list [--system <name>] [--limit <n>]\n  ctox ticket show --ticket-key <key>\n  ctox ticket history --ticket-key <key> [--limit <n>]\n  ctox ticket label-set --ticket-key <key> --label <label> [--assigned-by <actor>] [--rationale <text>] [--evidence-json <json>]\n  ctox ticket label-show --ticket-key <key>\n  ctox ticket bundle-put --label <label> --runbook-id <id> --policy-id <id> [--runbook-version <v>] [--policy-version <v>] [--approval-mode <mode>] [--autonomy-level <level>] [--verification-profile-id <id>] [--writeback-profile-id <id>] [--support-mode <mode>] [--risk-level <level>] [--actions <json-array>] [--notes <text>]\n  ctox ticket bundle-list\n  ctox ticket autonomy-grant-set --label <label> --approval-mode <mode> --autonomy-level <level> [--bundle-version <n>] [--approved-by <actor>] [--candidate-id <id>] [--rationale <text>]\n  ctox ticket autonomy-grant-list\n  ctox ticket dry-run --ticket-key <key> [--understanding <text>] [--risk-level <level>]\n  ctox ticket cases [--ticket-key <key>] [--limit <n>]\n  ctox ticket case-show --case-id <id>\n  ctox ticket approve --case-id <id> --status <approved|rejected> [--decided-by <actor>] [--rationale <text>]\n  ctox ticket execute --case-id <id> --summary <text>\n  ctox ticket verify --case-id <id> --status <passed|failed> [--summary <text>]\n  ctox ticket clarification-request --case-id <id> --question <text> [--target-type requester|owner|internal] [--target-channel ticket|email|jami|tui] [--missing-inputs <csv>] [--publish-reviewed]\n  ctox ticket clarification-resolve --clarification-id <id> --response-key <key> [--body <text>]\n  ctox ticket learn-candidate-create --case-id <id> --summary <text> [--actions <json-array>] [--evidence-json <json>]\n  ctox ticket learn-candidate-list [--label <label>] [--status <value>] [--limit <n>]\n  ctox ticket learn-candidate-decide --candidate-id <id> --status <approved|rejected> [--decided-by <actor>] [--notes <text>] [--promote-autonomy-level <level>]\n  ctox ticket writeback-comment --case-id <id> --body <text> [--internal]\n  ctox ticket writeback-transition --case-id <id> --state <value> [--body <text>] [--internal]\n  ctox ticket close --case-id <id> [--summary <text>]\n  ctox ticket audit [--ticket-key <key>] [--limit <n>]\n  ctox ticket local <subcommand> ..."
        ),
    }
}
