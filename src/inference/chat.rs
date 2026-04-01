use anyhow::Context;
use anyhow::Result;
use serde_json::Value;
use std::collections::BTreeMap;
use std::io::Read;
use std::path::Path;
use std::process::Command;
use std::process::Output;
use std::process::Stdio;
use std::thread;
use std::time::Duration;
use std::time::Instant;

use crate::browser;
use crate::channels;
use crate::context_health;
use crate::governance;
use crate::inference::engine;
use crate::inference::runtime_env;
use crate::inference::turn_engine;
use crate::lcm;

pub const CHAT_CONVERSATION_ID: i64 = 1;
const DEFAULT_ACTIVE_MODEL: &str = "openai/gpt-oss-20b";
const CTOX_CHAT_SYSTEM_PROMPT: &str = include_str!("../prompts/ctox_chat_system_prompt.md");
const CONTINUITY_REFRESH_TIMEOUT_SECS: u64 = 20;
const DEFAULT_CHAT_TURN_TIMEOUT_SECS: u64 = 180;
const MAX_RENDERED_SUMMARY_ITEMS: usize = 8;
const MAX_RENDERED_MESSAGE_ITEMS: usize = 24;
const MAX_RENDERED_CONTEXT_CHARS: usize = 24_000;

pub fn run_chat_turn_with_events<F>(
    root: &Path,
    db_path: &Path,
    prompt: &str,
    mut emit: F,
) -> Result<String>
where
    F: FnMut(&str),
{
    let settings = runtime_env::effective_runtime_env_map(root).unwrap_or_default();
    let config = turn_engine::ChatTurnConfig {
        max_context_tokens: read_usize_setting(&settings, "CTOX_CHAT_MODEL_MAX_CONTEXT", 131_072)
            as i64,
        turn_timeout_secs: read_usize_setting(
            &settings,
            "CTOX_CHAT_TURN_TIMEOUT_SECS",
            DEFAULT_CHAT_TURN_TIMEOUT_SECS as usize,
        ) as u64,
    };
    emit("lcm-open");
    let engine = lcm::LcmEngine::open(db_path, lcm::LcmConfig::default())?;
    let _ = engine.continuity_init_documents(CHAT_CONVERSATION_ID)?;
    emit("turn-plan");
    let plan = turn_engine::build_turn_plan(&engine, CHAT_CONVERSATION_ID, config.clone())?;
    emit(&format!(
        "turn-plan context={} timeout={}s stage={}",
        plan.max_context_tokens,
        plan.turn_timeout_secs,
        plan.stage.as_str()
    ));
    emit("compaction-check");
    let decision = plan.compaction.clone();
    emit(&format!(
        "compaction-window {} / {} ({})",
        decision.current_tokens, decision.threshold, decision.reason
    ));
    let mut compaction_result = None;
    if decision.should_compact {
        emit("compaction-run");
        let result = engine.compact(
            CHAT_CONVERSATION_ID,
            config.max_context_tokens,
            &lcm::HeuristicSummarizer,
            false,
        )?;
        emit(&format!(
            "compaction-result before={} after={} rounds={} created={}",
            result.tokens_before,
            result.tokens_after,
            result.rounds,
            result.created_summary_ids.len()
        ));
        compaction_result = Some(result);
        emit("compaction-complete");
    }
    let compaction_guard =
        turn_engine::assess_compaction_guard(&decision, compaction_result.as_ref());
    emit(&format!("compaction-guard {}", compaction_guard.summary));
    emit("persist-user-turn");
    lcm::run_add_message(db_path, CHAT_CONVERSATION_ID, "user", prompt)
        .context("failed to persist user message into LCM")?;
    emit("snapshot-context");
    let snapshot = engine.snapshot(CHAT_CONVERSATION_ID)?;
    let continuity = engine.continuity_show_all(CHAT_CONVERSATION_ID)?;
    let forgotten_entries = engine.continuity_forgotten(CHAT_CONVERSATION_ID, None, None)?;
    let health = context_health::assess_with_forgotten(
        &snapshot,
        &continuity,
        &forgotten_entries,
        prompt,
        config.max_context_tokens,
    );
    let governance_snapshot =
        governance::prompt_snapshot(root, CHAT_CONVERSATION_ID).unwrap_or_default();
    emit(&format!(
        "context-health {} {}",
        health.status.as_str(),
        health.overall_score
    ));
    emit("render-prompt");
    let rendered_context = select_rendered_context(&snapshot);
    emit(&format!(
        "context-selection rendered={} omitted={}",
        rendered_context.entries.len(),
        rendered_context.omitted_items
    ));
    let rendered_prompt = render_chat_prompt(
        &continuity,
        &governance_snapshot,
        &health,
        prompt,
        &rendered_context,
    );
    emit("invoke-model");
    let reply = invoke_codex_exec_with_timeout(
        root,
        &settings,
        &rendered_prompt,
        Some(Duration::from_secs(config.turn_timeout_secs)),
    )?;
    emit("persist-assistant-turn");
    lcm::run_add_message(db_path, CHAT_CONVERSATION_ID, "assistant", &reply)?;
    let engine = lcm::LcmEngine::open(db_path, lcm::LcmConfig::default())?;
    emit("continuity-refresh");
    let continuity_stats = refresh_continuity_documents(root, &settings, &engine, &mut emit)?;
    let outcome = turn_engine::ChatTurnOutcome {
        stage: turn_engine::TurnStage::Complete,
        health_status: health.status,
        health_score: health.overall_score,
        context_items_rendered: rendered_context.entries.len(),
        context_items_omitted: rendered_context.omitted_items,
        reply_chars: reply.chars().count(),
        compaction: compaction_result,
        continuity: continuity_stats,
        compaction_guard,
    };
    emit(&format!(
        "turn-outcome stage={} health={} score={} reply_chars={} continuity_updates={} continuity_skips={} omitted={}",
        outcome.stage.as_str(),
        outcome.health_status.as_str(),
        outcome.health_score,
        outcome.reply_chars,
        outcome.continuity.updated,
        outcome.continuity.skipped_prompt_build
            + outcome.continuity.skipped_invoke
            + outcome.continuity.skipped_apply,
        outcome.context_items_omitted
    ));
    emit("turn-complete");
    Ok(reply)
}

fn render_chat_prompt(
    continuity: &lcm::ContinuityShowAll,
    governance_snapshot: &governance::GovernancePromptSnapshot,
    health: &context_health::ContextHealthSnapshot,
    latest_user_prompt: &str,
    rendered_context: &RenderedContextSelection,
) -> String {
    let mut lines = vec![
        "Reply to the latest user turn using the structured context below.".to_string(),
        "Treat older blocked conclusions, stale follow-up prompts, and previous recovery notes as advisory only. If the latest user turn asks to continue or retry work, re-check the current runtime and host state before repeating an old blocker.".to_string(),
        "Do not assume a previous `blocked:` assistant reply is still true unless current evidence confirms it.".to_string(),
        String::new(),
        "Continuity documents:".to_string(),
        continuity_block("Narrative", &continuity.narrative.content),
        continuity_block("Anchors", &continuity.anchors.content),
        continuity_block("Focus", &continuity.focus.content),
        String::new(),
        governance::render_prompt_block(governance_snapshot),
        String::new(),
        context_health::render_prompt_block(health),
        String::new(),
        "Conversation context:".to_string(),
    ];
    for entry in &rendered_context.entries {
        lines.push(entry.clone());
    }
    if rendered_context.omitted_items > 0 {
        lines.push(format!(
            "context_notice: {} older conversation item(s) omitted from the live prompt; rely on continuity and newer concrete evidence unless those items become relevant again.",
            rendered_context.omitted_items
        ));
    }
    lines.push(String::new());
    lines.push(format!(
        "Latest user turn: {}",
        sanitize_context_message(latest_user_prompt)
    ));
    lines.join("\n")
}

fn continuity_block(label: &str, content: &str) -> String {
    format!("## {label}\n{}", content.trim_end())
}

struct RenderedContextSelection {
    entries: Vec<String>,
    omitted_items: usize,
}

fn select_rendered_context(snapshot: &lcm::LcmSnapshot) -> RenderedContextSelection {
    let mut summary_lines = Vec::new();
    let mut message_lines = Vec::new();
    for item in &snapshot.context_items {
        match item.item_type {
            lcm::ContextItemType::Message => {
                if let Some(message_id) = item.message_id {
                    if let Some(message) = snapshot
                        .messages
                        .iter()
                        .find(|entry| entry.message_id == message_id)
                    {
                        message_lines.push(render_context_message(&message.role, &message.content));
                    }
                }
            }
            lcm::ContextItemType::Summary => {
                if let Some(summary_id) = item.summary_id.as_deref() {
                    if let Some(summary) = snapshot
                        .summaries
                        .iter()
                        .find(|entry| entry.summary_id == summary_id)
                    {
                        summary_lines.push(format!("summary: {}", summary.content));
                    }
                }
            }
        }
    }

    let summary_start = summary_lines
        .len()
        .saturating_sub(MAX_RENDERED_SUMMARY_ITEMS);
    let selected_summaries = summary_lines[summary_start..].to_vec();
    let mut entries = selected_summaries.clone();
    let mut seen = std::collections::BTreeSet::new();
    let mut selected_messages = Vec::new();
    let mut total_chars = entries.iter().map(|line| line.len()).sum::<usize>();
    let mut omitted_messages = 0usize;

    for line in message_lines.iter().rev() {
        if selected_messages.len() >= MAX_RENDERED_MESSAGE_ITEMS {
            omitted_messages += 1;
            continue;
        }
        if !seen.insert(line.clone()) {
            omitted_messages += 1;
            continue;
        }
        let projected = total_chars + line.len();
        if !selected_messages.is_empty() && projected > MAX_RENDERED_CONTEXT_CHARS {
            omitted_messages += 1;
            continue;
        }
        total_chars = projected;
        selected_messages.push(line.clone());
    }
    selected_messages.reverse();
    entries.extend(selected_messages);

    let omitted_summaries = summary_start;
    RenderedContextSelection {
        entries,
        omitted_items: omitted_summaries + omitted_messages,
    }
}

fn refresh_continuity_documents(
    root: &Path,
    settings: &BTreeMap<String, String>,
    engine: &lcm::LcmEngine,
    emit: &mut impl FnMut(&str),
) -> Result<turn_engine::ContinuityRefreshStats> {
    let mut stats = turn_engine::ContinuityRefreshStats::default();
    for kind in [
        lcm::ContinuityKind::Narrative,
        lcm::ContinuityKind::Anchors,
        lcm::ContinuityKind::Focus,
    ] {
        let kind_label = match kind {
            lcm::ContinuityKind::Narrative => "narrative",
            lcm::ContinuityKind::Anchors => "anchors",
            lcm::ContinuityKind::Focus => "focus",
        };
        stats.attempted += 1;
        emit(&format!("continuity-{kind_label}-build"));
        let payload = match engine.continuity_build_prompt(CHAT_CONVERSATION_ID, kind) {
            Ok(payload) => payload,
            Err(err) => {
                stats.skipped_prompt_build += 1;
                eprintln!("ctox continuity refresh skipped {kind_label} prompt build: {err}");
                continue;
            }
        };
        emit(&format!("continuity-{kind_label}-invoke"));
        let diff = match invoke_codex_exec_with_timeout(
            root,
            settings,
            &payload.prompt,
            Some(Duration::from_secs(CONTINUITY_REFRESH_TIMEOUT_SECS)),
        ) {
            Ok(diff) => diff,
            Err(err) => {
                stats.skipped_invoke += 1;
                eprintln!("ctox continuity refresh skipped {kind_label} invocation: {err}");
                continue;
            }
        };
        if !diff.trim().is_empty() {
            emit(&format!("continuity-{kind_label}-apply"));
            if let Err(err) = engine.continuity_apply_diff(CHAT_CONVERSATION_ID, kind, diff.trim())
            {
                stats.skipped_apply += 1;
                eprintln!("ctox continuity refresh skipped invalid {kind_label} diff: {err}");
            } else {
                stats.updated += 1;
            }
        }
    }
    Ok(stats)
}

pub(crate) fn invoke_codex_exec_with_timeout(
    root: &Path,
    settings: &BTreeMap<String, String>,
    prompt: &str,
    timeout: Option<Duration>,
) -> Result<String> {
    invoke_codex_exec_with_timeout_and_instructions(root, settings, prompt, timeout, None, None)
}

pub(crate) fn invoke_codex_exec_with_timeout_and_instructions(
    root: &Path,
    settings: &BTreeMap<String, String>,
    prompt: &str,
    timeout: Option<Duration>,
    base_instructions_override: Option<&str>,
    include_apply_patch_tool_override: Option<bool>,
) -> Result<String> {
    let dependencies = engine::discover_vendored_dependency_paths(root);
    let model = runtime_env::effective_chat_model_from_map(settings)
        .unwrap_or_else(|| DEFAULT_ACTIVE_MODEL.to_string());
    channels::sync_prompt_identity(root, settings)?;
    let use_compact_local_prompt = is_compact_local_chat_model(&model);
    let base_instructions_text = if use_compact_local_prompt {
        if let Some(override_text) = base_instructions_override
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            format!(
                "{}\n\n# Task-Specific Instructions\n{}",
                compact_local_model_instructions(&model),
                override_text
            )
        } else {
            compact_local_model_instructions(&model).to_string()
        }
    } else if let Some(override_text) = base_instructions_override {
        override_text.to_string()
    } else {
        render_chat_system_prompt(root, settings)?
    };
    let include_apply_patch_tool =
        include_apply_patch_tool_override.unwrap_or(!use_compact_local_prompt);
    let base_instructions_override =
        toml_multiline_override("base_instructions", &base_instructions_text);
    let mut args = vec![
        "-m".to_string(),
        model.clone(),
        "--skip-git-repo-check".to_string(),
        "--dangerously-bypass-approvals-and-sandbox".to_string(),
        "--json".to_string(),
    ];

    if engine::uses_ctox_proxy_model(&model) {
        let proxy_host = settings
            .get("CTOX_PROXY_HOST")
            .cloned()
            .unwrap_or_else(|| "127.0.0.1".to_string());
        let proxy_port = settings
            .get("CTOX_PROXY_PORT")
            .cloned()
            .unwrap_or_else(|| "12434".to_string());
        let provider_config = format!(
            "model_providers.cto_local={{name=\"cto-local\",base_url=\"http://{proxy_host}:{proxy_port}/v1\",wire_api=\"responses\",requires_openai_auth=false}}"
        );
        args.extend([
            "-c".to_string(),
            "model_provider=\"cto_local\"".to_string(),
            "-c".to_string(),
            provider_config,
        ]);
    }
    if engine::is_openai_api_chat_model(&model) {
        args.extend([
            "-c".to_string(),
            "features.plugins=false".to_string(),
            "-c".to_string(),
            "features.apps=false".to_string(),
            "-c".to_string(),
            "features.tool_suggest=false".to_string(),
        ]);
    }
    for override_entry in browser::codex_config_overrides(root) {
        args.extend(["-c".to_string(), override_entry]);
    }
    args.extend([
        "-c".to_string(),
        base_instructions_override,
        "-c".to_string(),
        if include_apply_patch_tool {
            "include_apply_patch_tool=true".to_string()
        } else {
            "include_apply_patch_tool=false".to_string()
        },
        "-c".to_string(),
        "web_search=\"live\"".to_string(),
        "--".to_string(),
        prompt.to_string(),
    ]);

    if use_compact_local_prompt {
        let realized_context = read_usize_setting(
            settings,
            "CTOX_CHAT_MODEL_REALIZED_CONTEXT",
            read_usize_setting(settings, "CTOX_CHAT_MODEL_MAX_CONTEXT", 4096),
        )
        .max(2048);
        let compact_limit = local_chat_compact_limit(&model, realized_context);
        args.splice(
            4..4,
            [
                "-c".to_string(),
                format!("model_context_window={realized_context}"),
                "-c".to_string(),
                format!("model_auto_compact_token_limit={compact_limit}"),
            ],
        );
    }

    let mut command = if dependencies.codex_exec_binary.exists() {
        let mut cmd = Command::new(&dependencies.codex_exec_binary);
        cmd.current_dir(root);
        cmd
    } else {
        let mut cmd = Command::new("cargo");
        cmd.current_dir(&dependencies.codex_rs_root);
        cmd.args([
            "run",
            "--quiet",
            "--release",
            "-p",
            "codex-exec",
            "--bin",
            "codex-exec",
            "--",
        ]);
        cmd
    };
    command.args(&args);
    if engine::is_openai_api_chat_model(&model) {
        ensure_codex_api_auth(settings)?;
    }
    for (key, value) in settings {
        command.env(key, value);
    }
    if engine::is_openai_api_chat_model(&model) {
        if let Some(api_key) = settings
            .get("OPENAI_API_KEY")
            .map(|value| value.trim())
            .filter(|value| !value.is_empty())
        {
            command.env("CODEX_API_KEY", api_key);
        }
    }
    if let Some(codex_home) = resolve_codex_home(settings) {
        std::fs::create_dir_all(&codex_home).with_context(|| {
            format!(
                "failed to create CODEX_HOME for CTOX chat runtime at {}",
                codex_home.display()
            )
        })?;
        command.env("CODEX_HOME", codex_home);
    }
    command.env("CTOX_ROOT", root);
    command.env("CTOX_CONTEXT_DB", root.join("runtime/ctox_lcm.db"));
    let output = if let Some(timeout) = timeout {
        command.stdout(Stdio::piped()).stderr(Stdio::piped());
        let child = command
            .spawn()
            .map_err(|err| anyhow::anyhow!("failed to launch codex-exec for CTOX: {err:#}"))?;
        collect_child_output_with_timeout(child, timeout)?
    } else {
        command
            .output()
            .map_err(|err| anyhow::anyhow!("failed to launch codex-exec for CTOX: {err:#}"))?
    };
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
    if !output.status.success() {
        let stdout_response = if !stdout.is_empty() {
            extract_codex_text_response(&stdout)
        } else {
            None
        };
        if let Some(response) = stdout_response.filter(|value| !value.trim().is_empty()) {
            return Ok(response);
        }
        let stdout_error = if !stdout.is_empty() {
            extract_codex_error_response(&stdout)
        } else {
            None
        };
        let message = if let Some(summary) = stdout_error {
            summary
        } else if !stderr.is_empty() {
            stderr
        } else {
            stdout
        };
        anyhow::bail!("{message}");
    }
    let response = if !stdout.is_empty() {
        if let Some(text) = extract_codex_text_response(&stdout) {
            text
        } else if let Some(summary) = extract_codex_error_response(&stdout) {
            anyhow::bail!("{summary}");
        } else {
            stdout
        }
    } else {
        stderr
    };
    if response.is_empty() {
        anyhow::bail!("codex-exec returned empty output");
    }
    Ok(response)
}

fn collect_child_output_with_timeout(
    mut child: std::process::Child,
    timeout: Duration,
) -> Result<Output> {
    let mut stdout = child
        .stdout
        .take()
        .context("codex-exec missing stdout pipe")?;
    let mut stderr = child
        .stderr
        .take()
        .context("codex-exec missing stderr pipe")?;

    let stdout_handle = thread::spawn(move || -> Vec<u8> {
        let mut buffer = Vec::new();
        let _ = stdout.read_to_end(&mut buffer);
        buffer
    });
    let stderr_handle = thread::spawn(move || -> Vec<u8> {
        let mut buffer = Vec::new();
        let _ = stderr.read_to_end(&mut buffer);
        buffer
    });

    let started = Instant::now();
    let mut timed_out = false;
    let status = loop {
        if let Some(status) = child
            .try_wait()
            .map_err(|err| anyhow::anyhow!("failed to poll codex-exec for CTOX chat: {err:#}"))?
        {
            break status;
        }
        if started.elapsed() >= timeout {
            timed_out = true;
            let _ = child.kill();
            break child.wait().map_err(|err| {
                anyhow::anyhow!("failed to wait for timed out codex-exec: {err:#}")
            })?;
        }
        thread::sleep(Duration::from_millis(100));
    };

    let stdout = stdout_handle
        .join()
        .map_err(|_| anyhow::anyhow!("failed to join stdout reader for codex-exec"))?;
    let stderr = stderr_handle
        .join()
        .map_err(|_| anyhow::anyhow!("failed to join stderr reader for codex-exec"))?;

    if timed_out {
        anyhow::bail!("codex-exec timed out after {}s", timeout.as_secs());
    }

    Ok(Output {
        status,
        stdout,
        stderr,
    })
}

fn extract_codex_text_response(stdout: &str) -> Option<String> {
    let mut last_agent_message: Option<String> = None;
    for line in stdout.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let Ok(value) = serde_json::from_str::<Value>(trimmed) else {
            return None;
        };
        let item = match value.get("item") {
            Some(item) => item,
            None => continue,
        };
        let agent_message = if item.get("type").and_then(Value::as_str) == Some("agent_message") {
            Some(item)
        } else {
            item.get("item").filter(|nested| {
                nested.get("type").and_then(Value::as_str) == Some("agent_message")
            })
        };
        if let Some(agent_message) = agent_message {
            if let Some(text) = agent_message.get("text").and_then(Value::as_str) {
                let trimmed = text.trim();
                if !trimmed.is_empty() {
                    last_agent_message = Some(trimmed.to_string());
                }
            }
        }
    }
    last_agent_message
}

fn extract_codex_error_response(stdout: &str) -> Option<String> {
    let mut last_error: Option<String> = None;
    for line in stdout.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let Ok(value) = serde_json::from_str::<Value>(trimmed) else {
            return None;
        };
        if value.get("type").and_then(Value::as_str) == Some("error") {
            if let Some(message) = value.get("message").and_then(Value::as_str) {
                let message = message.trim();
                if !message.is_empty() {
                    last_error = Some(message.to_string());
                }
            }
            continue;
        }
        if value.get("type").and_then(Value::as_str) == Some("turn.failed") {
            if let Some(message) = value
                .get("error")
                .and_then(|error| error.get("message"))
                .and_then(Value::as_str)
            {
                let message = message.trim();
                if !message.is_empty() {
                    last_error = Some(message.to_string());
                }
            }
            continue;
        }
    }
    last_error
}

fn sanitize_context_message(content: &str) -> String {
    let trimmed = content.trim();
    if trimmed.is_empty() {
        return String::new();
    }
    if let Some(summary) = summarize_inbound_email_wrapper(trimmed) {
        return clip_prompt_text(&summary, 8_000);
    }
    let normalized = if looks_like_codex_event_stream(trimmed) {
        extract_codex_text_response(trimmed).unwrap_or_else(|| {
            "Previous turn stored raw Codex event-stream output instead of a final reply. Use only the stable user-visible outcome or re-check the current runtime state before continuing.".to_string()
        })
    } else {
        trimmed.to_string()
    };
    clip_prompt_text(&normalized, 8_000)
}

pub fn summarize_runtime_error(content: &str) -> String {
    let trimmed = content.trim();
    if trimmed.is_empty() {
        return "CTOX execution failed without a stable error payload.".to_string();
    }
    if let Some(summary) = summarize_known_infra_error(trimmed) {
        return summary;
    }
    if looks_like_codex_event_stream(trimmed) {
        if let Some(summary) = extract_codex_text_response(trimmed) {
            return clip_prompt_text(&summary, 700);
        }
        return "The turn failed after emitting raw Codex event-stream output instead of a stable final reply. Re-check the current runtime state and recover from the real blocker instead of replaying raw event data.".to_string();
    }
    clip_prompt_text(trimmed, 700)
}

pub fn synthesize_failure_reply(content: &str) -> String {
    let summary = summarize_runtime_error(content);
    format!("Status: `blocked`\n\nBlocker: {summary}")
}

pub fn hard_runtime_blocker_retry_cooldown_secs(content: &str) -> Option<u64> {
    let lower = content.to_ascii_lowercase();
    if lower.contains("quota exceeded")
        || lower.contains("billing details")
        || lower.contains("openai api quota is exhausted")
        || lower.contains("billing is unavailable for the selected model")
    {
        return Some(1_800);
    }
    if summarize_known_infra_error(content).is_some()
        || lower.contains("chat backend could not start on this host")
    {
        return Some(900);
    }
    None
}

fn summarize_known_infra_error(content: &str) -> Option<String> {
    let lower = content.to_ascii_lowercase();
    if lower.contains("quota exceeded") || lower.contains("billing details") {
        return Some(
            "CTOX chat could not continue because the configured OpenAI API quota is exhausted or billing is unavailable for the selected model.".to_string(),
        );
    }
    if lower.contains("feature `edition2024` is required")
        || (lower.contains("edition2024") && lower.contains("cargo"))
    {
        return Some(
            "CTOX chat backend could not start on this host because the vendored Codex runtime requires a newer Rust/Cargo toolchain with Edition 2024 support.".to_string(),
        );
    }
    if lower.contains("error[e0583]")
        && lower.contains("file not found for module")
        && lower.contains("state/src/runtime.rs")
        && (lower.contains("`agent_jobs`") || lower.contains("`backfill`"))
    {
        return Some(
            "CTOX chat backend could not start on this host because the vendored Codex runtime checkout is incomplete: `state/src/runtime/` is missing required module files such as `agent_jobs.rs` and `backfill.rs`.".to_string(),
        );
    }
    if lower.contains("failed to load manifest for workspace member")
        && lower.contains("cargo.toml")
    {
        return Some(
            "CTOX chat backend could not start on this host because the vendored Codex workspace manifest is not buildable in its current remote environment.".to_string(),
        );
    }
    None
}

fn render_context_message(role: &str, content: &str) -> String {
    let sanitized = sanitize_context_message(content);
    let label = render_message_role_label(role, content);
    if role == "assistant" && is_historical_status_note(&sanitized) {
        format!(
            "assistant_status_history: Historical assistant status note only; re-check the current runtime and host state before relying on it. {}",
            sanitized
        )
    } else {
        format!("{label}: {sanitized}")
    }
}

fn render_message_role_label<'a>(role: &'a str, content: &str) -> &'a str {
    if role == "user" && is_internal_queue_prompt(content) {
        "internal_queue"
    } else {
        role
    }
}

fn is_internal_queue_prompt(content: &str) -> bool {
    let trimmed = content.trim_start();
    [
        "Continue the broader goal using the latest completed turn as the starting point.",
        "Review the blocked owner-visible task without losing continuity.",
        "Recover or finish the owner-visible task without losing continuity.",
        "Use the queue-cleanup skill first.",
    ]
    .iter()
    .any(|prefix| trimmed.starts_with(prefix))
}

fn is_historical_status_note(content: &str) -> bool {
    let trimmed = content.trim_start();
    let lower = trimmed.to_ascii_lowercase();
    lower.starts_with("blocked:")
        || lower.starts_with("completed:")
        || lower.starts_with("failed:")
        || lower.starts_with("prepared:")
        || lower.starts_with("still blocked")
        || lower.starts_with("nextcloud_")
        || lower.starts_with("zammad_")
        || lower.starts_with("redis_")
}

fn looks_like_codex_event_stream(content: &str) -> bool {
    let lines = content.lines().take(8).collect::<Vec<_>>();
    if lines.len() < 2 {
        return false;
    }
    lines
        .iter()
        .filter(|line| {
            let trimmed = line.trim();
            trimmed.starts_with('{') && trimmed.contains("\"type\"")
        })
        .count()
        >= 2
}

fn clip_prompt_text(content: &str, max_chars: usize) -> String {
    if content.chars().count() <= max_chars {
        return content.to_string();
    }
    let mut clipped = content
        .chars()
        .take(max_chars.saturating_sub(1))
        .collect::<String>();
    clipped.push('…');
    clipped
}

fn summarize_inbound_email_wrapper(content: &str) -> Option<String> {
    if !content.starts_with("[E-Mail eingegangen]") {
        return None;
    }
    let sender =
        extract_labeled_line(content, "Sender:").unwrap_or_else(|| "unknown sender".to_string());
    let subject =
        extract_labeled_line(content, "Betreff:").unwrap_or_else(|| "(ohne Betreff)".to_string());
    let thread =
        extract_labeled_line(content, "Thread:").unwrap_or_else(|| "unknown thread".to_string());
    Some(format!(
        "Inbound email wrapper from {sender} with subject {subject} on thread {thread}. The original wrapper also contained reply instructions and historical communication context; treat this as prior mail context only and rely on the newest concrete task evidence before repeating old conclusions."
    ))
}

fn extract_labeled_line(content: &str, prefix: &str) -> Option<String> {
    content
        .lines()
        .find_map(|line| {
            line.strip_prefix(prefix)
                .map(|value| value.trim().to_string())
        })
        .filter(|value| !value.is_empty())
}

fn ensure_codex_api_auth(settings: &BTreeMap<String, String>) -> Result<()> {
    let Some(api_key) = settings
        .get("OPENAI_API_KEY")
        .map(|value| value.trim())
        .filter(|value| !value.is_empty())
    else {
        return Ok(());
    };
    let Some(codex_home) = resolve_codex_home(settings) else {
        return Ok(());
    };
    std::fs::create_dir_all(&codex_home)?;
    let auth_path = codex_home.join("auth.json");
    let payload = serde_json::json!({
        "OPENAI_API_KEY": api_key,
    });
    std::fs::write(&auth_path, serde_json::to_vec_pretty(&payload)?)?;
    Ok(())
}

fn resolve_codex_home(settings: &BTreeMap<String, String>) -> Option<std::path::PathBuf> {
    settings
        .get("CODEX_HOME")
        .map(|value| value.trim())
        .filter(|value| !value.is_empty())
        .map(std::path::PathBuf::from)
        .or_else(|| std::env::var_os("CODEX_HOME").map(std::path::PathBuf::from))
        .or_else(|| {
            std::env::var_os("HOME").map(|home| std::path::PathBuf::from(home).join(".codex"))
        })
}

fn toml_multiline_override(key: &str, value: &str) -> String {
    let escaped = value.replace("\"\"\"", "\\\"\\\"\\\"");
    format!("{key} = \"\"\"\n{escaped}\n\"\"\"")
}

fn render_chat_system_prompt(root: &Path, settings: &BTreeMap<String, String>) -> Result<String> {
    let owner = channels::load_prompt_identity(root, settings)?;
    let channels_block = owner.channels.join("\n");
    let preferred_channel = owner
        .preferred_channel
        .unwrap_or_else(|| "not set".to_string());
    let owner_email = owner
        .owner_email_address
        .unwrap_or_else(|| "not configured".to_string());
    let allowed_email_domain = owner
        .allowed_email_domain
        .unwrap_or_else(|| "not configured".to_string());
    let admin_email_policies = owner.admin_email_policies.join("\n");
    Ok(CTOX_CHAT_SYSTEM_PROMPT
        .replace("{{OWNER_NAME}}", &owner.owner_name)
        .replace("{{OWNER_CHANNELS}}", &channels_block)
        .replace("{{OWNER_EMAIL_ADDRESS}}", &owner_email)
        .replace("{{OWNER_EMAIL_DOMAIN}}", &allowed_email_domain)
        .replace("{{OWNER_EMAIL_ADMINS}}", &admin_email_policies)
        .replace("{{OWNER_PREFERRED_CHANNEL}}", &preferred_channel))
}

fn read_usize_setting(settings: &BTreeMap<String, String>, key: &str, default: usize) -> usize {
    settings
        .get(key)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(default)
}

fn is_local_qwen_chat_model(model: &str) -> bool {
    matches!(
        model.trim(),
        "Qwen/Qwen3.5-4B"
            | "Qwen/Qwen3.5-9B"
            | "Qwen/Qwen3.5-27B"
            | "Qwen/Qwen3.5-35B-A3B"
            | "nvidia/Nemotron-Cascade-2-30B-A3B"
    )
}

#[cfg(test)]
mod tests {
    use crate::lcm;

    use super::collect_child_output_with_timeout;
    use super::extract_codex_error_response;
    use super::hard_runtime_blocker_retry_cooldown_secs;
    use super::is_historical_status_note;
    use super::is_internal_queue_prompt;
    use super::render_context_message;
    use super::render_message_role_label;
    use super::sanitize_context_message;
    use super::select_rendered_context;
    use super::summarize_inbound_email_wrapper;
    use super::summarize_runtime_error;
    use super::synthesize_failure_reply;

    #[test]
    fn sanitize_context_message_reduces_raw_event_stream_to_agent_text() {
        let raw = concat!(
            "{\"type\":\"item.completed\",\"item\":{\"type\":\"agent_message\",\"text\":\"Ich prüfe Redis.\"}}\n",
            "{\"type\":\"item.completed\",\"item\":{\"type\":\"command_execution\",\"command\":\"echo hi\",\"aggregated_output\":\"hi\",\"exit_code\":0,\"status\":\"completed\"}}\n",
            "{\"type\":\"item.completed\",\"item\":{\"type\":\"agent_message\",\"text\":\"Redis ist noch nicht bereit.\"}}\n"
        );
        let sanitized = sanitize_context_message(raw);
        assert!(sanitized.contains("Redis ist noch nicht bereit."));
        assert!(!sanitized.contains("\"type\":\"item.completed\""));
    }

    #[test]
    fn sanitize_context_message_clips_oversized_plain_text() {
        let huge = "a".repeat(10_000);
        let sanitized = sanitize_context_message(&huge);
        assert!(sanitized.len() < huge.len());
        assert!(sanitized.ends_with('…'));
    }

    #[test]
    fn internal_follow_up_prompt_is_not_rendered_as_owner_user_message() {
        let prompt = "Review the blocked owner-visible task without losing continuity.\n\nGoal:\nInstall Redis";
        assert!(is_internal_queue_prompt(prompt));
        assert_eq!(render_message_role_label("user", prompt), "internal_queue");
        assert_eq!(
            render_message_role_label("user", "Install Redis now"),
            "user"
        );
    }

    #[test]
    fn inbound_email_wrapper_is_reduced_to_compact_summary() {
        let wrapped = concat!(
            "[E-Mail eingegangen]\n",
            "Sender: Max Mustermann <max@example.com>\n",
            "Betreff: Re: Helpdesk kaputt\n",
            "Thread: <abc@example.com>\n\n",
            "[Bisheriger Thread-Kontext]\n- ...\n\n",
            "[Letzte owner-relevante Kommunikation ueber alle Kanaele]\n- ...\n"
        );
        let summary = summarize_inbound_email_wrapper(wrapped).expect("expected wrapper summary");
        assert!(summary.contains("Max Mustermann <max@example.com>"));
        assert!(summary.contains("Re: Helpdesk kaputt"));
        assert!(summary.contains("<abc@example.com>"));
        assert!(!summary.contains("[Bisheriger Thread-Kontext]"));
    }

    #[test]
    fn assistant_blocked_reply_is_marked_as_history() {
        assert!(is_historical_status_note(
            "blocked: redis install still missing sudo access"
        ));
        let rendered = render_context_message(
            "assistant",
            "blocked: redis install still missing sudo access",
        );
        assert!(rendered.starts_with("assistant_status_history:"));
        assert!(rendered.contains("Historical assistant status note only"));
    }

    #[test]
    fn codex_error_stream_extracts_turn_failure_message() {
        let raw = concat!(
            "{\"type\":\"thread.started\",\"thread_id\":\"abc\"}\n",
            "{\"type\":\"turn.started\"}\n",
            "{\"type\":\"error\",\"message\":\"Quota exceeded. Check your plan and billing details.\"}\n",
            "{\"type\":\"turn.failed\",\"error\":{\"message\":\"Quota exceeded. Check your plan and billing details.\"}}\n"
        );
        assert_eq!(
            extract_codex_error_response(raw).as_deref(),
            Some("Quota exceeded. Check your plan and billing details.")
        );
    }

    #[test]
    fn runtime_error_summary_maps_quota_exceeded() {
        let summary =
            summarize_runtime_error("Quota exceeded. Check your plan and billing details.");
        assert!(summary.contains("configured OpenAI API quota is exhausted"));
    }

    #[test]
    fn quota_blocker_gets_long_retry_cooldown() {
        assert_eq!(
            hard_runtime_blocker_retry_cooldown_secs(
                "Quota exceeded. Check your plan and billing details."
            ),
            Some(1_800)
        );
        assert_eq!(
            hard_runtime_blocker_retry_cooldown_secs(
                "CTOX chat could not continue because the configured OpenAI API quota is exhausted or billing is unavailable for the selected model."
            ),
            Some(1_800)
        );
    }

    #[test]
    fn rendered_context_omits_older_items_when_history_is_large() {
        let messages = (1..=40)
            .map(|index| lcm::MessageRecord {
                message_id: index,
                conversation_id: 1,
                seq: index,
                role: if index % 2 == 0 {
                    "assistant".to_string()
                } else {
                    "user".to_string()
                },
                content: format!("message {index}: {}", "x".repeat(250)),
                token_count: 40,
                created_at: "2026-03-26T10:00:00Z".to_string(),
            })
            .collect::<Vec<_>>();
        let context_items = (1..=40)
            .map(|index| lcm::ContextItemSnapshot {
                ordinal: index,
                item_type: lcm::ContextItemType::Message,
                message_id: Some(index),
                summary_id: None,
                seq: index,
                depth: 0,
                token_count: 40,
            })
            .collect::<Vec<_>>();
        let snapshot = lcm::LcmSnapshot {
            conversation_id: 1,
            messages,
            summaries: Vec::new(),
            context_items,
            summary_edges: Vec::new(),
            summary_messages: Vec::new(),
        };

        let selected = select_rendered_context(&snapshot);
        assert!(!selected.entries.is_empty());
        assert!(selected.omitted_items > 0);
        assert!(selected.entries.len() <= super::MAX_RENDERED_MESSAGE_ITEMS);
        let joined = selected.entries.join("\n");
        assert!(!joined.contains("message 1:"));
        assert!(joined.contains("message 40:"));
    }

    #[test]
    fn runtime_error_summary_reduces_raw_event_stream() {
        let raw = concat!(
            "{\"type\":\"item.completed\",\"item\":{\"type\":\"agent_message\",\"text\":\"Ich prüfe noch den API-Pfad.\"}}\n",
            "{\"type\":\"item.started\",\"item\":{\"type\":\"command_execution\",\"command\":\"curl ...\"}}\n"
        );
        let summary = summarize_runtime_error(raw);
        assert!(summary.contains("Ich prüfe noch den API-Pfad."));
        assert!(!summary.contains("\"type\":\"item.completed\""));
    }

    #[test]
    fn runtime_error_summary_compacts_missing_module_build_failures() {
        let raw = concat!(
            "error[E0583]: file not found for module `agent_jobs`\n",
            "--> state/src/runtime.rs:52:1\n",
            "= help: to create the module `agent_jobs`, create file \"state/src/runtime/agent_jobs.rs\"\n"
        );
        let summary = summarize_runtime_error(raw);
        assert!(summary.contains("vendored Codex runtime checkout is incomplete"));
        assert!(summary.contains("state/src/runtime/"));
        assert!(!summary.contains("error[E0583]"));
    }

    #[test]
    fn runtime_error_summary_compacts_old_cargo_failures() {
        let raw = concat!(
            "failed to parse manifest at `/home/metricspace/ctox/references/openai-codex/codex-rs/backend-client/Cargo.toml`\n\n",
            "Caused by:\n",
            "  feature `edition2024` is required\n"
        );
        let summary = summarize_runtime_error(raw);
        assert!(summary.contains("newer Rust/Cargo toolchain"));
        assert!(summary.contains("Edition 2024"));
        assert!(!summary.contains("failed to parse manifest"));
    }

    #[test]
    fn synthesized_failure_reply_has_operator_shape() {
        let reply = synthesize_failure_reply("codex-exec timed out after 180s");
        assert!(reply.starts_with("Status: `blocked`"));
        assert!(reply.contains("180s"));
    }

    #[test]
    fn child_output_is_drained_while_waiting_for_exit() {
        let mut command = std::process::Command::new("python3");
        command.args([
            "-c",
            "import sys; sys.stdout.write('x' * 200000); sys.stdout.flush()",
        ]);
        command.stdout(std::process::Stdio::piped());
        command.stderr(std::process::Stdio::piped());
        let child = command.spawn().expect("failed to spawn python3");
        let output = collect_child_output_with_timeout(child, std::time::Duration::from_secs(10))
            .expect("failed to collect child output");
        assert!(output.status.success());
        assert_eq!(output.stdout.len(), 200000);
    }
}

fn is_local_glm_chat_model(model: &str) -> bool {
    matches!(model.trim(), "zai-org/GLM-4.7-Flash")
}

fn is_compact_local_chat_model(model: &str) -> bool {
    is_local_qwen_chat_model(model) || is_local_glm_chat_model(model)
}

fn compact_local_model_instructions(model: &str) -> &'static str {
    if is_local_glm_chat_model(model) {
        "You are Codex running through CTOX on a local GLM model. Be extremely concise and tool-accurate. Prefer exec_command for shell work and minimal edits. Do not restate instructions. When the user asks for an exact marker or short final answer, return only that required text after any needed tool calls."
    } else {
        "You are Codex running through CTOX on a local ChatML model. Be concise and tool-accurate. Prefer exec_command for shell work and simple file edits. When the user asks for an exact marker or short final answer, return only that required text after any needed tool calls."
    }
}

fn local_chat_compact_limit(model: &str, realized_context: usize) -> usize {
    match model {
        "Qwen/Qwen3.5-27B" => 2_560.min(realized_context.saturating_sub(512)).max(1_536),
        "Qwen/Qwen3.5-35B-A3B" => 1_536.min(realized_context.saturating_sub(256)).max(1_024),
        "nvidia/Nemotron-Cascade-2-30B-A3B" => {
            1_536.min(realized_context.saturating_sub(384)).max(1_024)
        }
        "zai-org/GLM-4.7-Flash" => 1_280.min(realized_context.saturating_sub(384)).max(896),
        _ => ((realized_context as f64) * 3.0 / 4.0).round() as usize,
    }
}
