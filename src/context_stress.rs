use anyhow::Result;
use serde::Serialize;
use std::path::Path;

use crate::context_health;
use crate::lcm;

const DEFAULT_STRESS_CONVERSATION_ID: i64 = 42;
const DEFAULT_STRESS_ITERATIONS: usize = 24;
const DEFAULT_STRESS_TOKEN_BUDGET: i64 = 160;

#[derive(Debug, Clone, Serialize)]
pub struct ContextStressRoundReport {
    pub round: usize,
    pub compaction_action_taken: bool,
    pub compaction_rounds: usize,
    pub created_summary_ids: usize,
    pub tokens_before: i64,
    pub tokens_after: i64,
    pub overall_score: u8,
    pub status: context_health::ContextHealthStatus,
    pub repair_recommended: bool,
    pub warning_codes: Vec<String>,
    pub forgotten_lines: usize,
    pub degradation_markers: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ContextStressReport {
    pub conversation_id: i64,
    pub iterations_requested: usize,
    pub compactions_completed: usize,
    pub stable: bool,
    pub min_score: u8,
    pub final_score: u8,
    pub final_status: context_health::ContextHealthStatus,
    pub failure_reason: Option<String>,
    pub rounds: Vec<ContextStressRoundReport>,
}

#[derive(Debug, Clone)]
struct DynamicContinuityState {
    focus_status: String,
    focus_next: String,
    narrative_status: String,
    artifact_line: String,
    cause_line: String,
    retry_boundary: String,
}

pub fn run_context_stress(
    db_path: &Path,
    conversation_id: Option<i64>,
    iterations: Option<usize>,
    token_budget: Option<i64>,
) -> Result<ContextStressReport> {
    let conversation_id = conversation_id.unwrap_or(DEFAULT_STRESS_CONVERSATION_ID);
    let iterations = iterations.unwrap_or(DEFAULT_STRESS_ITERATIONS).max(1);
    let token_budget = token_budget.unwrap_or(DEFAULT_STRESS_TOKEN_BUDGET).max(64);
    let engine = lcm::LcmEngine::open(db_path, stress_config())?;
    let _ = engine.continuity_init_documents(conversation_id)?;

    let mut continuity_state = seed_stress_continuity(&engine, conversation_id, iterations)?;
    let mut rounds = Vec::with_capacity(iterations);
    let mut min_score = u8::MAX;
    let mut failure_reason = None;
    let mut compactions_completed = 0usize;
    let mut final_status = context_health::ContextHealthStatus::Healthy;
    let mut final_score = 100u8;

    for round in 1..=iterations {
        apply_round_continuity(
            &engine,
            conversation_id,
            iterations,
            round,
            &mut continuity_state,
        )?;
        add_round_messages(&engine, conversation_id, iterations, round)?;

        let compaction = engine.compact(
            conversation_id,
            token_budget,
            &lcm::HeuristicSummarizer,
            true,
        )?;
        if compaction.action_taken {
            compactions_completed += 1;
        }

        let snapshot = engine.snapshot(conversation_id)?;
        let continuity = engine.continuity_show_all(conversation_id)?;
        let forgotten = engine.continuity_forgotten(conversation_id, None, None)?;
        let latest_prompt = format!(
            "Continue stress round {round}/{iterations} with the preserved mission contract."
        );
        let health = context_health::assess_with_forgotten(
            &snapshot,
            &continuity,
            &forgotten,
            &latest_prompt,
            token_budget,
        );
        let warning_codes = health
            .warnings
            .iter()
            .map(|warning| warning.code.clone())
            .collect::<Vec<_>>();
        let mut degradation_markers = Vec::new();
        if !compaction.action_taken {
            degradation_markers.push("compaction_skipped".to_string());
        }
        if health.repair_recommended {
            degradation_markers.push("repair_recommended".to_string());
        }
        if health.status == context_health::ContextHealthStatus::Critical {
            degradation_markers.push("critical_status".to_string());
        }
        degradation_markers.extend(warning_codes.iter().cloned());

        min_score = std::cmp::min(min_score, health.overall_score);
        final_status = health.status;
        final_score = health.overall_score;

        rounds.push(ContextStressRoundReport {
            round,
            compaction_action_taken: compaction.action_taken,
            compaction_rounds: compaction.rounds,
            created_summary_ids: compaction.created_summary_ids.len(),
            tokens_before: compaction.tokens_before,
            tokens_after: compaction.tokens_after,
            overall_score: health.overall_score,
            status: health.status,
            repair_recommended: health.repair_recommended,
            warning_codes,
            forgotten_lines: forgotten.len(),
            degradation_markers: degradation_markers.clone(),
        });

        if !degradation_markers.is_empty() {
            failure_reason = Some(format!(
                "round {round} produced degradation markers: {}",
                degradation_markers.join(", ")
            ));
            break;
        }
    }

    let stable = failure_reason.is_none() && compactions_completed >= iterations;
    Ok(ContextStressReport {
        conversation_id,
        iterations_requested: iterations,
        compactions_completed,
        stable,
        min_score,
        final_score,
        final_status,
        failure_reason,
        rounds,
    })
}

fn stress_config() -> lcm::LcmConfig {
    lcm::LcmConfig {
        context_threshold: 0.25,
        min_compaction_tokens: 0,
        fresh_tail_count: 0,
        leaf_chunk_tokens: 120,
        leaf_target_tokens: 24,
        condensed_target_tokens: 24,
        leaf_min_fanout: 1,
        condensed_min_fanout: 2,
        max_rounds: 6,
    }
}

fn seed_stress_continuity(
    engine: &lcm::LcmEngine,
    conversation_id: i64,
    iterations: usize,
) -> Result<DynamicContinuityState> {
    let focus_status = "running deterministic context stress harness".to_string();
    let focus_next = "process round 1".to_string();
    let narrative_status = format!("stress harness initialized for {iterations} rounds");
    let artifact_line = "stress-report-round-0.json".to_string();
    let cause_line = "Earlier retries without preserved evidence degraded continuity.".to_string();
    let retry_boundary =
        "Do not retry a tactic unless the latest round adds fresh validation evidence.".to_string();

    engine.continuity_apply_diff(
        conversation_id,
        lcm::ContinuityKind::Narrative,
        &format!(
            "## Ausgangslage\n+ Deterministic stress harness for repeated compaction and continuity checks.\n## Problem\n+ Verify that repeated compaction does not degrade the loop state.\n## Ursache\n+ {cause_line}\n## Wendepunkte\n+ The harness will compact context repeatedly and refuse silent degradation.\n## Aktueller Stand\n+ {narrative_status}\n## Offene Spannung\n+ Stability must survive at least {iterations} forced compactions.\n"
        ),
    )?;
    engine.continuity_apply_diff(
        conversation_id,
        lcm::ContinuityKind::Anchors,
        &format!(
            "## Artefakte\n+ {artifact_line}\n## Hosts / Ports\n+ local stress harness against SQLite LCM state.\n## Skripte / Commands\n+ ctox context-stress <db-path> {conversation_id} {iterations} {DEFAULT_STRESS_TOKEN_BUDGET}\n## Invarianten / Verbote\n+ {retry_boundary}\n## Gates / Pruefpfade\n+ No warnings, no repair recommendation, and one successful compaction per round.\n"
        ),
    )?;
    engine.continuity_apply_diff(
        conversation_id,
        lcm::ContinuityKind::Focus,
        &format!(
            "## Status\n+ {focus_status}\n## Blocker\n+ none\n## Next\n+ {focus_next}\n## Done / Gate\n+ complete all rounds with no degradation markers.\n"
        ),
    )?;

    Ok(DynamicContinuityState {
        focus_status,
        focus_next,
        narrative_status,
        artifact_line,
        cause_line,
        retry_boundary,
    })
}

fn apply_round_continuity(
    engine: &lcm::LcmEngine,
    conversation_id: i64,
    iterations: usize,
    round: usize,
    state: &mut DynamicContinuityState,
) -> Result<()> {
    let next_focus_status = format!("round {round}/{iterations} ready for compaction");
    let next_focus_next = if round < iterations {
        format!("process round {}", round + 1)
    } else {
        "finalize stress report".to_string()
    };
    let next_narrative_status = format!("round {round}/{iterations} completed without drift");
    let next_artifact_line = format!("stress-report-round-{round}.json");
    let next_cause_line =
        format!("Round {round} preserved retry evidence before any future replay attempt.");
    let next_retry_boundary =
        format!("Retry only after fresh validation evidence from round {round} is present.");

    apply_line_swap(
        engine,
        conversation_id,
        lcm::ContinuityKind::Focus,
        "Status",
        Some(&state.focus_status),
        Some(&next_focus_status),
    )?;
    apply_line_swap(
        engine,
        conversation_id,
        lcm::ContinuityKind::Focus,
        "Next",
        Some(&state.focus_next),
        Some(&next_focus_next),
    )?;
    apply_line_swap(
        engine,
        conversation_id,
        lcm::ContinuityKind::Narrative,
        "Aktueller Stand",
        Some(&state.narrative_status),
        Some(&next_narrative_status),
    )?;
    apply_line_swap(
        engine,
        conversation_id,
        lcm::ContinuityKind::Narrative,
        "Ursache",
        Some(&state.cause_line),
        Some(&next_cause_line),
    )?;
    apply_line_swap(
        engine,
        conversation_id,
        lcm::ContinuityKind::Anchors,
        "Artefakte",
        Some(&state.artifact_line),
        Some(&next_artifact_line),
    )?;
    apply_line_swap(
        engine,
        conversation_id,
        lcm::ContinuityKind::Anchors,
        "Invarianten / Verbote",
        Some(&state.retry_boundary),
        Some(&next_retry_boundary),
    )?;

    state.focus_status = next_focus_status;
    state.focus_next = next_focus_next;
    state.narrative_status = next_narrative_status;
    state.artifact_line = next_artifact_line;
    state.cause_line = next_cause_line;
    state.retry_boundary = next_retry_boundary;
    Ok(())
}

fn apply_line_swap(
    engine: &lcm::LcmEngine,
    conversation_id: i64,
    kind: lcm::ContinuityKind,
    section: &str,
    old_line: Option<&str>,
    new_line: Option<&str>,
) -> Result<()> {
    let mut lines = vec![format!("## {section}")];
    if let Some(old_line) = old_line.filter(|line| !line.trim().is_empty()) {
        lines.push(format!("- {old_line}"));
    }
    if let Some(new_line) = new_line.filter(|line| !line.trim().is_empty()) {
        lines.push(format!("+ {new_line}"));
    }
    engine.continuity_apply_diff(conversation_id, kind, &format!("{}\n", lines.join("\n")))?;
    Ok(())
}

fn add_round_messages(
    engine: &lcm::LcmEngine,
    conversation_id: i64,
    iterations: usize,
    round: usize,
) -> Result<()> {
    let lines = [
        (
            "user",
            format!(
                "Round {round}/{iterations} request: continue the delivery program with the new validation bundle, release notes, rollback checkpoints, and owner acceptance evidence. Preserve the mission contract and keep the done gate explicit."
            ),
        ),
        (
            "assistant",
            format!(
                "Round {round} assessment: the latest evidence bundle is fresh, the previous retry boundary is satisfied, and the loop should preserve the exact blocker-free state before any new compaction pass."
            ),
        ),
        (
            "user",
            format!(
                "Round {round} detail: artifact group {round} contains deployment transcript fragments, telemetry snapshots, and a compact acceptance checklist. Keep the constraints sticky and avoid inventing stale blockers."
            ),
        ),
        (
            "assistant",
            format!(
                "Round {round} continuity note: update the durable record with the current status, next action, and retry condition so the next slice can resume without replaying discarded assumptions."
            ),
        ),
        (
            "user",
            format!(
                "Round {round} verification task: compress earlier details if needed, but retain enough evidence to prove that the loop still knows what success, failure, and the next safe action look like."
            ),
        ),
        (
            "assistant",
            format!(
                "Round {round} verification result: the context remains aligned to the main objective, the constraints are explicit, and the next compaction should preserve retrievability rather than flattening the mission into generic summaries."
            ),
        ),
    ];
    for (role, content) in lines {
        engine.add_message(conversation_id, role, &content)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::run_context_stress;
    use anyhow::Result;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_db() -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time")
            .as_nanos();
        std::env::temp_dir().join(format!("ctox-context-stress-{unique}.db"))
    }

    #[test]
    fn stress_harness_survives_twenty_forced_compactions() -> Result<()> {
        let db_path = temp_db();
        let report = run_context_stress(&db_path, Some(77), Some(20), Some(160))?;
        assert!(report.stable, "{report:#?}");
        assert_eq!(report.compactions_completed, 20);
        assert!(report.min_score >= 70, "{report:#?}");
        let _ = std::fs::remove_file(db_path);
        Ok(())
    }
}
