use anyhow::Result;
use serde::Serialize;
use std::path::Path;

use crate::lcm;

const NARRATIVE_TEMPLATE: &str =
    "# CONTINUITY NARRATIVE\n\n## Ausgangslage\n\n## Problem\n\n## Ursache\n\n## Wendepunkte\n\n## Dauerhafte Entscheidungen\n\n## Aktueller Stand\n\n## Offene Spannung\n";
const ANCHORS_TEMPLATE: &str =
    "# CONTINUITY ANCHORS\n\n## Artefakte\n\n## Hosts / Ports\n\n## Skripte / Commands\n\n## Invarianten / Verbote\n\n## Gates / Pruefpfade\n";
const FOCUS_TEMPLATE: &str =
    "# ACTIVE FOCUS\n\n## Status\nMission:\nMission state:\nContinuation mode:\nTrigger intensity:\n\n## Blocker\nCurrent blocker:\n\n## Next\nNext slice:\n\n## Done / Gate\nDone gate:\nClosure confidence:\n";

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ContextHealthStatus {
    Healthy,
    Watch,
    Degraded,
    Critical,
}

impl ContextHealthStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Healthy => "healthy",
            Self::Watch => "watch",
            Self::Degraded => "degraded",
            Self::Critical => "critical",
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum WarningSeverity {
    Info,
    Warning,
    Critical,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ContextHealthDimension {
    pub name: String,
    pub score: u8,
    pub summary: String,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ContextHealthWarning {
    pub code: String,
    pub severity: WarningSeverity,
    pub summary: String,
    pub evidence: String,
    pub recommended_action: String,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ContextHealthSnapshot {
    pub conversation_id: i64,
    pub overall_score: u8,
    pub status: ContextHealthStatus,
    pub summary: String,
    pub repair_recommended: bool,
    pub dimensions: Vec<ContextHealthDimension>,
    pub warnings: Vec<ContextHealthWarning>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ContextRepairGovernorDecision {
    pub should_enqueue_repair: bool,
    pub reason: String,
}

pub fn assess_for_conversation(
    db_path: &Path,
    conversation_id: i64,
    token_budget: i64,
    latest_user_prompt: Option<&str>,
) -> Result<ContextHealthSnapshot> {
    let engine = lcm::LcmEngine::open(db_path, lcm::LcmConfig::default())?;
    let snapshot = engine.snapshot(conversation_id)?;
    let continuity = engine.continuity_show_all(conversation_id)?;
    let forgotten_entries = engine.continuity_forgotten(conversation_id, None, None)?;
    Ok(assess_with_forgotten(
        &snapshot,
        &continuity,
        &forgotten_entries,
        latest_user_prompt.unwrap_or(""),
        token_budget,
    ))
}

#[cfg_attr(not(test), allow(dead_code))]
pub fn assess(
    snapshot: &lcm::LcmSnapshot,
    continuity: &lcm::ContinuityShowAll,
    latest_user_prompt: &str,
    token_budget: i64,
) -> ContextHealthSnapshot {
    assess_with_forgotten(snapshot, continuity, &[], latest_user_prompt, token_budget)
}

pub fn assess_with_forgotten(
    snapshot: &lcm::LcmSnapshot,
    continuity: &lcm::ContinuityShowAll,
    forgotten_entries: &[lcm::ContinuityForgottenEntry],
    latest_user_prompt: &str,
    token_budget: i64,
) -> ContextHealthSnapshot {
    let context_tokens = snapshot
        .context_items
        .iter()
        .map(|item| item.token_count.max(0))
        .sum::<i64>();
    let pressure_score = score_context_pressure(context_tokens, token_budget);
    let continuity_score = score_continuity_coverage(continuity);
    let repetition = repeated_recent_user_turns(snapshot, latest_user_prompt);
    let repetition_score = score_repetition_risk(repetition);
    let blocked_count = recent_blocked_status_count(snapshot);
    let blocked_score = score_blocked_loop(blocked_count);
    let repair_prompt_count = recent_internal_repair_prompt_count(snapshot);
    let repair_score = score_repair_churn(repair_prompt_count);
    let mission_contract = inspect_mission_contract(continuity);
    let mission_contract_score = score_mission_contract(&mission_contract);
    let negative_memory = inspect_negative_memory(continuity, forgotten_entries);
    let negative_memory_score = score_negative_memory(
        &negative_memory,
        repetition,
        blocked_count,
        repair_prompt_count,
    );

    let dimensions = vec![
        ContextHealthDimension {
            name: "context_pressure".to_string(),
            score: pressure_score,
            summary: format!(
                "Live context uses about {} tokens against a {} token budget.",
                context_tokens,
                token_budget.max(1)
            ),
        },
        ContextHealthDimension {
            name: "continuity_coverage".to_string(),
            score: continuity_score,
            summary: continuity_coverage_summary(continuity),
        },
        ContextHealthDimension {
            name: "mission_contract".to_string(),
            score: mission_contract_score,
            summary: mission_contract.summary(),
        },
        ContextHealthDimension {
            name: "negative_memory".to_string(),
            score: negative_memory_score,
            summary: negative_memory.summary(),
        },
        ContextHealthDimension {
            name: "repetition_risk".to_string(),
            score: repetition_score,
            summary: if repetition == 0 {
                "The latest user turn does not look like a recent duplicate.".to_string()
            } else {
                format!(
                    "The latest user turn overlaps with {} recent user turn(s).",
                    repetition
                )
            },
        },
        ContextHealthDimension {
            name: "blocked_loop_risk".to_string(),
            score: blocked_score,
            summary: if blocked_count == 0 {
                "Recent assistant history does not show repeated blocked-status notes.".to_string()
            } else {
                format!("{blocked_count} recent assistant status note(s) look blocked or stalled.")
            },
        },
        ContextHealthDimension {
            name: "repair_churn".to_string(),
            score: repair_score,
            summary: if repair_prompt_count == 0 {
                "Recent context is not dominated by internal repair or continuation prompts."
                    .to_string()
            } else {
                format!("{repair_prompt_count} recent internal prompt(s) look like repair or continuation churn.")
            },
        },
    ];

    let weighted_score = weighted_average(&[
        (pressure_score, 15_u32),
        (continuity_score, 15_u32),
        (mission_contract_score, 20_u32),
        (negative_memory_score, 15_u32),
        (repetition_score, 15_u32),
        (blocked_score, 10_u32),
        (repair_score, 10_u32),
    ]);
    let warnings = build_warnings(
        snapshot.conversation_id,
        continuity,
        forgotten_entries,
        token_budget,
        context_tokens,
        &mission_contract,
        &negative_memory,
        repetition,
        blocked_count,
        repair_prompt_count,
    );
    let status = merge_status_with_warnings(status_for_score(weighted_score), &warnings);
    let repair_recommended = weighted_score < 60
        || warnings
            .iter()
            .any(|warning| warning.severity == WarningSeverity::Critical);
    let summary = summarize_dimensions(&dimensions, &warnings, status, weighted_score);

    ContextHealthSnapshot {
        conversation_id: snapshot.conversation_id,
        overall_score: weighted_score,
        status,
        summary,
        repair_recommended,
        dimensions,
        warnings,
    }
}

pub fn render_prompt_block(health: &ContextHealthSnapshot) -> String {
    let mut lines = vec![
        "Context health:".to_string(),
        "This block is advisory. Use it to steer the next slice, but it does not authorize hidden background repair work on its own.".to_string(),
        format!(
            "overall_score: {} ({})",
            health.overall_score,
            health.status.as_str()
        ),
        health.summary.clone(),
    ];
    let mut weakest = health.dimensions.iter().collect::<Vec<_>>();
    weakest.sort_by_key(|dimension| dimension.score);
    lines.push("weak_dimensions:".to_string());
    for dimension in weakest.into_iter().take(3) {
        lines.push(format!("- [{}] {}", dimension.name, dimension.summary));
    }
    if health.warnings.is_empty() {
        lines.push("warnings: none".to_string());
    } else {
        lines.push("warnings:".to_string());
        for warning in health.warnings.iter().take(4) {
            lines.push(format!(
                "- [{}] {} | evidence: {} | action: {}",
                warning.code, warning.summary, warning.evidence, warning.recommended_action
            ));
        }
    }
    if health.repair_recommended {
        lines.push("repair_guidance: If you can safely improve context quality while still serving the real goal, prioritize that repair now. Do not let context repair become the goal if direct progress is still possible.".to_string());
    }
    lines.join("\n")
}

pub fn evaluate_repair_governor(
    health: &ContextHealthSnapshot,
    source_label: &str,
    current_goal: &str,
    existing_open_repair_task: bool,
    open_repair_task_count: usize,
) -> ContextRepairGovernorDecision {
    if !health.repair_recommended {
        return ContextRepairGovernorDecision {
            should_enqueue_repair: false,
            reason: "context health is still within the no-repair band".to_string(),
        };
    }
    if existing_open_repair_task {
        return ContextRepairGovernorDecision {
            should_enqueue_repair: false,
            reason: "an open context-health repair task already exists".to_string(),
        };
    }
    if open_repair_task_count >= 2 {
        return ContextRepairGovernorDecision {
            should_enqueue_repair: false,
            reason: "context-health repair is already consuming multiple open work slots"
                .to_string(),
        };
    }
    if is_context_repair_source(source_label) || looks_like_context_repair_goal(current_goal) {
        return ContextRepairGovernorDecision {
            should_enqueue_repair: false,
            reason: "the current work already is a context-health repair slice".to_string(),
        };
    }
    ContextRepairGovernorDecision {
        should_enqueue_repair: true,
        reason:
            "context-health warnings crossed the repair threshold without an active repair slice"
                .to_string(),
    }
}

pub fn render_repair_task_prompt(health: &ContextHealthSnapshot) -> String {
    let mut lines = vec![
        "Review and repair CTOX context health without letting repair become the main mission."
            .to_string(),
        String::new(),
        format!(
            "Current context health score: {} ({})",
            health.overall_score,
            health.status.as_str()
        ),
        health.summary.clone(),
        String::new(),
        "Warnings:".to_string(),
    ];
    for warning in &health.warnings {
        lines.push(format!(
            "- {}: {} | evidence: {} | recommended action: {}",
            warning.code, warning.summary, warning.evidence, warning.recommended_action
        ));
    }
    lines.push(String::new());
    lines.push("Repair contract:".to_string());
    lines.push("- Reconstruct the minimum mission contract for the next safe step: status, blocker, next action, done gate, and durable constraints.".to_string());
    lines.push("- Refresh negative memory: record which tactic failed, why it failed, and what new evidence would justify a retry.".to_string());
    lines.push("- Do not create another context-health repair task.".to_string());
    lines.push("- If one bounded repair pass does not materially improve clarity or reduce repetition risk, stop repairing and either replan, escalate, or ask for the exact missing input.".to_string());
    lines.push(
        "- Separate actual mission progress from context housekeeping in the final reply."
            .to_string(),
    );
    lines.join("\n")
}

fn weighted_average(values: &[(u8, u32)]) -> u8 {
    let total_weight = values.iter().map(|(_, weight)| *weight).sum::<u32>().max(1);
    let total = values
        .iter()
        .map(|(score, weight)| u32::from(*score) * *weight)
        .sum::<u32>();
    ((total + (total_weight / 2)) / total_weight) as u8
}

fn status_for_score(score: u8) -> ContextHealthStatus {
    match score {
        80..=100 => ContextHealthStatus::Healthy,
        60..=79 => ContextHealthStatus::Watch,
        40..=59 => ContextHealthStatus::Degraded,
        _ => ContextHealthStatus::Critical,
    }
}

fn merge_status_with_warnings(
    base_status: ContextHealthStatus,
    warnings: &[ContextHealthWarning],
) -> ContextHealthStatus {
    let strongest_warning_status = warnings
        .iter()
        .filter_map(|warning| match warning.severity {
            WarningSeverity::Info => None,
            WarningSeverity::Warning => Some(ContextHealthStatus::Watch),
            WarningSeverity::Critical => Some(ContextHealthStatus::Critical),
        })
        .max_by_key(|status| status_rank(*status));
    strongest_warning_status
        .filter(|warning_status| status_rank(*warning_status) > status_rank(base_status))
        .unwrap_or(base_status)
}

fn status_rank(status: ContextHealthStatus) -> u8 {
    match status {
        ContextHealthStatus::Healthy => 0,
        ContextHealthStatus::Watch => 1,
        ContextHealthStatus::Degraded => 2,
        ContextHealthStatus::Critical => 3,
    }
}

fn score_context_pressure(tokens: i64, budget: i64) -> u8 {
    let budget = budget.max(1) as f64;
    let ratio = (tokens.max(0) as f64) / budget;
    if ratio <= 0.45 {
        100
    } else if ratio <= 0.65 {
        85
    } else if ratio <= 0.80 {
        65
    } else if ratio <= 0.92 {
        40
    } else {
        20
    }
}

fn score_continuity_coverage(continuity: &lcm::ContinuityShowAll) -> u8 {
    let scores = [
        score_continuity_document(&continuity.narrative.content, NARRATIVE_TEMPLATE),
        score_continuity_document(&continuity.anchors.content, ANCHORS_TEMPLATE),
        score_continuity_document(&continuity.focus.content, FOCUS_TEMPLATE),
    ];
    ((u16::from(scores[0]) + u16::from(scores[1]) + u16::from(scores[2]) + 1) / 3) as u8
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct MissionContractCoverage {
    status_lines: usize,
    blocker_lines: usize,
    next_lines: usize,
    done_gate_lines: usize,
    invariant_lines: usize,
}

impl MissionContractCoverage {
    fn control_points(&self) -> usize {
        [
            self.status_lines > 0,
            self.blocker_lines > 0,
            self.next_lines > 0,
            self.done_gate_lines > 0,
            self.invariant_lines > 0,
        ]
        .into_iter()
        .filter(|value| *value)
        .count()
    }

    fn summary(&self) -> String {
        format!(
            "Mission contract currently captures {}/5 control points: status={}, blocker={}, next={}, done_gate={}, constraints={}.",
            self.control_points(),
            yes_no(self.status_lines > 0),
            yes_no(self.blocker_lines > 0),
            yes_no(self.next_lines > 0),
            yes_no(self.done_gate_lines > 0),
            yes_no(self.invariant_lines > 0),
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct NegativeMemoryCoverage {
    cause_lines: usize,
    turning_point_lines: usize,
    invariant_lines: usize,
    forgotten_lines: usize,
}

impl NegativeMemoryCoverage {
    fn total_signals(&self) -> usize {
        self.cause_lines + self.turning_point_lines + self.invariant_lines + self.forgotten_lines
    }

    fn summary(&self) -> String {
        format!(
            "Negative memory exposes {} signal(s): cause={}, turning_points={}, constraints={}, forgotten={}.",
            self.total_signals(),
            self.cause_lines,
            self.turning_point_lines,
            self.invariant_lines,
            self.forgotten_lines,
        )
    }
}

fn score_continuity_document(content: &str, template: &str) -> u8 {
    if normalize_text(content) == normalize_text(template) {
        return 20;
    }
    let meaningful_lines = content
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .count();
    match meaningful_lines {
        0 => 20,
        1 => 45,
        2 => 65,
        3 => 80,
        _ => 100,
    }
}

fn score_mission_contract(contract: &MissionContractCoverage) -> u8 {
    match contract.control_points() {
        0 => 20,
        1 => 35,
        2 => 55,
        3 => 75,
        4 => 90,
        _ => 100,
    }
}

fn score_negative_memory(
    memory: &NegativeMemoryCoverage,
    repetition_count: usize,
    blocked_count: usize,
    repair_prompt_count: usize,
) -> u8 {
    let loop_pressure = repetition_count + blocked_count + repair_prompt_count;
    if loop_pressure == 0 {
        return match memory.total_signals() {
            0 => 85,
            1 => 92,
            _ => 100,
        };
    }
    match memory.total_signals() {
        0 => 20,
        1 => 55,
        2 => 75,
        _ => 100,
    }
}

fn score_repetition_risk(repetition_count: usize) -> u8 {
    match repetition_count {
        0 => 100,
        1 => 55,
        _ => 20,
    }
}

fn score_blocked_loop(blocked_count: usize) -> u8 {
    match blocked_count {
        0 => 100,
        1 => 75,
        2 => 45,
        _ => 15,
    }
}

fn score_repair_churn(repair_prompt_count: usize) -> u8 {
    match repair_prompt_count {
        0 => 100,
        1 => 80,
        2 => 50,
        _ => 20,
    }
}

fn build_warnings(
    conversation_id: i64,
    continuity: &lcm::ContinuityShowAll,
    forgotten_entries: &[lcm::ContinuityForgottenEntry],
    token_budget: i64,
    context_tokens: i64,
    mission_contract: &MissionContractCoverage,
    negative_memory: &NegativeMemoryCoverage,
    repetition: usize,
    blocked_count: usize,
    repair_prompt_count: usize,
) -> Vec<ContextHealthWarning> {
    let mut warnings = Vec::new();
    let ratio = (context_tokens.max(0) as f64) / (token_budget.max(1) as f64);
    if ratio > 0.80 {
        warnings.push(ContextHealthWarning {
            code: "context_window_pressure".to_string(),
            severity: if ratio > 0.92 {
                WarningSeverity::Critical
            } else {
                WarningSeverity::Warning
            },
            summary: "The live context window is under heavy pressure.".to_string(),
            evidence: format!(
                "conversation {} is using about {} / {} tokens of active context",
                conversation_id, context_tokens, token_budget
            ),
            recommended_action:
                "Compact or refresh continuity before repeating verbose recovery history."
                    .to_string(),
        });
    }
    if normalize_text(&continuity.focus.content) == normalize_text(FOCUS_TEMPLATE) {
        warnings.push(ContextHealthWarning {
            code: "focus_document_thin".to_string(),
            severity: WarningSeverity::Critical,
            summary: "The active focus document is still effectively empty.".to_string(),
            evidence: "the focus document still matches the bootstrap template".to_string(),
            recommended_action: "Rebuild focus with current status, blocker, next action, and done gate before more continuation work.".to_string(),
        });
    }
    if mission_contract.control_points() <= 2 {
        warnings.push(ContextHealthWarning {
            code: "mission_contract_thin".to_string(),
            severity: if mission_contract.control_points() <= 1 {
                WarningSeverity::Critical
            } else {
                WarningSeverity::Warning
            },
            summary: "The durable mission contract is underspecified.".to_string(),
            evidence: mission_contract.summary(),
            recommended_action: "Refresh focus and anchors so the loop has a current status, blocker, next step, done gate, and durable constraints.".to_string(),
        });
    }
    if repetition > 0 {
        warnings.push(ContextHealthWarning {
            code: "recent_user_turn_repeated".to_string(),
            severity: if repetition > 1 {
                WarningSeverity::Critical
            } else {
                WarningSeverity::Warning
            },
            summary: "The latest user turn overlaps with a recent user turn.".to_string(),
            evidence: format!(
                "detected {} recent duplicate-like user prompt(s)",
                repetition
            ),
            recommended_action:
                "Check whether the loop is retrying a failed tactic without new evidence."
                    .to_string(),
        });
    }
    if blocked_count >= 2 {
        warnings.push(ContextHealthWarning {
            code: "blocked_status_loop".to_string(),
            severity: if blocked_count >= 3 {
                WarningSeverity::Critical
            } else {
                WarningSeverity::Warning
            },
            summary: "Recent assistant history shows repeated blocked-status notes.".to_string(),
            evidence: format!("{blocked_count} recent assistant messages look blocked or stalled"),
            recommended_action: "Revalidate the blocker against current evidence and ban the failed tactic unless new inputs appeared.".to_string(),
        });
    }
    if repair_prompt_count >= 2 {
        warnings.push(ContextHealthWarning {
            code: "repair_prompt_churn".to_string(),
            severity: if repair_prompt_count >= 3 {
                WarningSeverity::Critical
            } else {
                WarningSeverity::Warning
            },
            summary: "Internal continuation or repair prompts are crowding out normal work."
                .to_string(),
            evidence: format!(
                "{repair_prompt_count} recent internal prompts look like recovery or cleanup work"
            ),
            recommended_action:
                "Do one bounded repair pass only, then replan or resume the real goal.".to_string(),
        });
    }
    let loop_pressure = repetition + blocked_count + repair_prompt_count;
    if loop_pressure > 0 && negative_memory.total_signals() == 0 {
        warnings.push(ContextHealthWarning {
            code: "failure_memory_missing".to_string(),
            severity: WarningSeverity::Critical,
            summary: "The loop is under pressure, but the context does not preserve any explicit failure memory.".to_string(),
            evidence: format!(
                "loop pressure={} while cause={}, turning_points={}, constraints={}, forgotten={}",
                loop_pressure,
                negative_memory.cause_lines,
                negative_memory.turning_point_lines,
                negative_memory.invariant_lines,
                forgotten_entries.len()
            ),
            recommended_action: "Record the failed tactic, the real blocker, and the retry condition before trying again.".to_string(),
        });
    }
    warnings
}

fn continuity_coverage_summary(continuity: &lcm::ContinuityShowAll) -> String {
    let narrative = meaningful_continuity_lines(&continuity.narrative.content);
    let anchors = meaningful_continuity_lines(&continuity.anchors.content);
    let focus = meaningful_continuity_lines(&continuity.focus.content);
    format!(
        "Continuity currently exposes {} narrative, {} anchor, and {} focus line(s) beyond section headers.",
        narrative, anchors, focus
    )
}

fn inspect_mission_contract(continuity: &lcm::ContinuityShowAll) -> MissionContractCoverage {
    MissionContractCoverage {
        status_lines: section_lines(&continuity.focus.content, "Status").len(),
        blocker_lines: section_lines(&continuity.focus.content, "Blocker").len(),
        next_lines: section_lines(&continuity.focus.content, "Next").len(),
        done_gate_lines: section_lines(&continuity.focus.content, "Done / Gate").len(),
        invariant_lines: section_lines(&continuity.anchors.content, "Invarianten / Verbote").len(),
    }
}

fn inspect_negative_memory(
    continuity: &lcm::ContinuityShowAll,
    forgotten_entries: &[lcm::ContinuityForgottenEntry],
) -> NegativeMemoryCoverage {
    let invariant_lines = section_lines(&continuity.anchors.content, "Invarianten / Verbote")
        .into_iter()
        .filter(|line| looks_like_negative_constraint(line))
        .count();
    NegativeMemoryCoverage {
        cause_lines: section_lines(&continuity.narrative.content, "Ursache").len(),
        turning_point_lines: section_lines(&continuity.narrative.content, "Wendepunkte").len(),
        invariant_lines,
        forgotten_lines: forgotten_entries.len(),
    }
}

fn section_lines(content: &str, section_name: &str) -> Vec<String> {
    let mut active = false;
    let mut lines = Vec::new();
    for line in content.lines() {
        let trimmed = line.trim();
        if let Some(header) = trimmed.strip_prefix("## ") {
            active = header == section_name;
            continue;
        }
        if active && !trimmed.is_empty() && !trimmed.starts_with('#') {
            lines.push(trimmed.to_string());
        }
    }
    lines
}

fn looks_like_negative_constraint(line: &str) -> bool {
    let normalized = normalize_text(line);
    [
        "nicht",
        "never",
        "avoid",
        "verbot",
        "ban",
        "retry only",
        "do not",
        "kein",
        "ohne",
    ]
    .iter()
    .any(|needle| normalized.contains(needle))
}

fn summarize_dimensions(
    dimensions: &[ContextHealthDimension],
    warnings: &[ContextHealthWarning],
    status: ContextHealthStatus,
    overall_score: u8,
) -> String {
    let mut weakest = dimensions.to_vec();
    weakest.sort_by_key(|dimension| dimension.score);
    let drivers = weakest
        .into_iter()
        .take(2)
        .map(|dimension| format!("{}={}", dimension.name, dimension.score))
        .collect::<Vec<_>>()
        .join(", ");
    let warning_summary = if warnings.is_empty() {
        "No active warnings.".to_string()
    } else {
        let strongest = warnings
            .iter()
            .max_by_key(|warning| match warning.severity {
                WarningSeverity::Info => 0,
                WarningSeverity::Warning => 1,
                WarningSeverity::Critical => 2,
            })
            .map(|warning| match warning.severity {
                WarningSeverity::Info => "info",
                WarningSeverity::Warning => "warning",
                WarningSeverity::Critical => "critical",
            })
            .unwrap_or("warning");
        format!(
            "{} active warning(s), strongest severity {}.",
            warnings.len(),
            strongest
        )
    };
    format!(
        "Context health is {} at score {}. Weakest dimensions: {}. {}",
        status.as_str(),
        overall_score,
        drivers,
        warning_summary
    )
}

fn yes_no(value: bool) -> &'static str {
    if value {
        "yes"
    } else {
        "no"
    }
}

fn repeated_recent_user_turns(snapshot: &lcm::LcmSnapshot, latest_user_prompt: &str) -> usize {
    let target = normalize_text(latest_user_prompt);
    if target.is_empty() {
        return 0;
    }
    snapshot
        .messages
        .iter()
        .rev()
        .filter(|message| message.role == "user")
        .skip(1)
        .take(8)
        .filter(|message| normalize_text(&message.content) == target)
        .count()
}

fn recent_blocked_status_count(snapshot: &lcm::LcmSnapshot) -> usize {
    snapshot
        .messages
        .iter()
        .rev()
        .filter(|message| message.role == "assistant")
        .take(8)
        .filter(|message| looks_like_blocked_status(&message.content))
        .count()
}

fn looks_like_blocked_status(content: &str) -> bool {
    let normalized = normalize_text(content);
    normalized.starts_with("status blocked")
        || normalized.starts_with("blocked")
        || normalized.contains("still blocked")
        || normalized.contains("remains blocked")
        || normalized.contains("bleibt blockiert")
}

fn recent_internal_repair_prompt_count(snapshot: &lcm::LcmSnapshot) -> usize {
    snapshot
        .messages
        .iter()
        .rev()
        .filter(|message| message.role == "user")
        .take(8)
        .filter(|message| is_internal_repair_prompt(&message.content))
        .count()
}

fn is_internal_repair_prompt(content: &str) -> bool {
    let trimmed = content.trim_start();
    [
        "Continue the broader goal using the latest completed turn as the starting point.",
        "Review the blocked owner-visible task without losing continuity.",
        "Recover or finish the owner-visible task without losing continuity.",
        "Use the queue-cleanup skill first.",
        "Review and repair CTOX context health without letting repair become the main mission.",
    ]
    .iter()
    .any(|prefix| trimmed.starts_with(prefix))
}

fn is_context_repair_source(source_label: &str) -> bool {
    let normalized = normalize_text(source_label);
    normalized.contains("context-health") || normalized.contains("queue-guard")
}

fn looks_like_context_repair_goal(goal: &str) -> bool {
    let normalized = normalize_text(goal);
    normalized.contains("context health")
        || normalized.contains("repair ctox context")
        || normalized.contains("queue cleanup")
}

fn meaningful_continuity_lines(content: &str) -> usize {
    content
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .count()
}

fn normalize_text(content: &str) -> String {
    content
        .to_ascii_lowercase()
        .chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { ' ' })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::assess;
    use super::assess_with_forgotten;
    use super::evaluate_repair_governor;
    use super::render_prompt_block;
    use super::ContextHealthStatus;
    use super::WarningSeverity;
    use crate::lcm;

    fn sample_snapshot(messages: Vec<(&str, &str)>, tokens: i64) -> lcm::LcmSnapshot {
        let message_records = messages
            .into_iter()
            .enumerate()
            .map(|(index, (role, content))| lcm::MessageRecord {
                message_id: index as i64 + 1,
                conversation_id: 1,
                seq: index as i64 + 1,
                role: role.to_string(),
                content: content.to_string(),
                token_count: 100,
                created_at: "2026-03-31T00:00:00Z".to_string(),
            })
            .collect::<Vec<_>>();
        lcm::LcmSnapshot {
            conversation_id: 1,
            messages: message_records,
            summaries: Vec::new(),
            context_items: vec![lcm::ContextItemSnapshot {
                ordinal: 1,
                item_type: lcm::ContextItemType::Message,
                message_id: Some(1),
                summary_id: None,
                seq: 1,
                depth: 0,
                token_count: tokens,
            }],
            summary_edges: Vec::new(),
            summary_messages: Vec::new(),
        }
    }

    fn healthy_continuity() -> lcm::ContinuityShowAll {
        lcm::ContinuityShowAll {
            conversation_id: 1,
            narrative: lcm::ContinuityDocumentState {
                conversation_id: 1,
                kind: lcm::ContinuityKind::Narrative,
                head_commit_id: "n1".to_string(),
                content: "# CONTINUITY NARRATIVE\n\n## Ausgangslage\nRunning on host A\n\n## Problem\nQueue drift\n\n## Ursache\nStale blocker notes kept getting reused.\n\n## Wendepunkte\nA failed retry was already observed.\n\n## Aktueller Stand\nRepair pending\n".to_string(),
                created_at: "2026-03-31T00:00:00Z".to_string(),
                updated_at: "2026-03-31T00:00:00Z".to_string(),
            },
            anchors: lcm::ContinuityDocumentState {
                conversation_id: 1,
                kind: lcm::ContinuityKind::Anchors,
                head_commit_id: "a1".to_string(),
                content: "# CONTINUITY ANCHORS\n\n## Artefakte\nruntime/ctox_lcm.db\n\n## Hosts / Ports\n127.0.0.1:12435\n\n## Skripte / Commands\ncargo test context_health\n\n## Invarianten / Verbote\nDo not retry the same repair without new evidence.\n\n## Gates / Pruefpfade\ncargo test\n".to_string(),
                created_at: "2026-03-31T00:00:00Z".to_string(),
                updated_at: "2026-03-31T00:00:00Z".to_string(),
            },
            focus: lcm::ContinuityDocumentState {
                conversation_id: 1,
                kind: lcm::ContinuityKind::Focus,
                head_commit_id: "f1".to_string(),
                content: "# ACTIVE FOCUS\n\n## Status\nrepairing queue drift\n\n## Blocker\nnone\n\n## Next\nadd scoring\n\n## Done / Gate\ntests green\n".to_string(),
                created_at: "2026-03-31T00:00:00Z".to_string(),
                updated_at: "2026-03-31T00:00:00Z".to_string(),
            },
        }
    }

    #[test]
    fn assess_marks_repeated_blocked_context_as_critical() {
        let snapshot = sample_snapshot(
            vec![
                ("user", "retry redis"),
                (
                    "assistant",
                    "Status: `blocked`\n\nBlocker: redis still offline",
                ),
                ("user", "retry redis"),
                (
                    "assistant",
                    "Status: `blocked`\n\nBlocker: redis still offline",
                ),
                ("user", "retry redis"),
            ],
            118_000,
        );
        let mut continuity = healthy_continuity();
        continuity.focus.content =
            "# ACTIVE FOCUS\n\n## Status\n\n## Blocker\n\n## Next\n\n## Done / Gate\n".to_string();
        let health = assess(&snapshot, &continuity, "retry redis", 131_072);
        assert_eq!(health.status, ContextHealthStatus::Critical);
        assert!(health.repair_recommended);
        assert!(health
            .warnings
            .iter()
            .any(|warning| warning.code == "blocked_status_loop"));
        assert!(health
            .warnings
            .iter()
            .any(|warning| warning.severity == WarningSeverity::Critical));
    }

    #[test]
    fn warns_when_mission_contract_and_failure_memory_are_missing() {
        let snapshot = sample_snapshot(
            vec![
                ("user", "retry deploy"),
                (
                    "assistant",
                    "Status: `blocked`\n\nBlocker: deploy still failing",
                ),
                ("user", "retry deploy"),
            ],
            10_000,
        );
        let continuity = lcm::ContinuityShowAll {
            conversation_id: 1,
            narrative: lcm::ContinuityDocumentState {
                conversation_id: 1,
                kind: lcm::ContinuityKind::Narrative,
                head_commit_id: "n1".to_string(),
                content: "# CONTINUITY NARRATIVE\n\n## Ausgangslage\nRollout in progress\n\n## Problem\nDeployment drift\n\n## Ursache\n\n## Wendepunkte\n\n## Dauerhafte Entscheidungen\n\n## Aktueller Stand\nstalled\n\n## Offene Spannung\n".to_string(),
                created_at: "2026-03-31T00:00:00Z".to_string(),
                updated_at: "2026-03-31T00:00:00Z".to_string(),
            },
            anchors: lcm::ContinuityDocumentState {
                conversation_id: 1,
                kind: lcm::ContinuityKind::Anchors,
                head_commit_id: "a1".to_string(),
                content: "# CONTINUITY ANCHORS\n\n## Artefakte\nrelease.tar\n\n## Hosts / Ports\nprod-1\n\n## Skripte / Commands\ndeploy.sh\n\n## Invarianten / Verbote\n\n## Gates / Pruefpfade\nsmoke test\n".to_string(),
                created_at: "2026-03-31T00:00:00Z".to_string(),
                updated_at: "2026-03-31T00:00:00Z".to_string(),
            },
            focus: lcm::ContinuityDocumentState {
                conversation_id: 1,
                kind: lcm::ContinuityKind::Focus,
                head_commit_id: "f1".to_string(),
                content: "# ACTIVE FOCUS\n\n## Status\ninvestigating\n\n## Blocker\n\n## Next\n\n## Done / Gate\n\n".to_string(),
                created_at: "2026-03-31T00:00:00Z".to_string(),
                updated_at: "2026-03-31T00:00:00Z".to_string(),
            },
        };
        let health = assess(&snapshot, &continuity, "retry deploy", 131_072);
        assert_eq!(health.status, ContextHealthStatus::Critical);
        assert!(health
            .warnings
            .iter()
            .any(|warning| warning.code == "mission_contract_thin"));
        assert!(health
            .warnings
            .iter()
            .any(|warning| warning.code == "failure_memory_missing"));
    }

    #[test]
    fn render_prompt_block_surfaces_repair_guidance() {
        let snapshot = sample_snapshot(vec![("user", "continue task")], 120_000);
        let mut continuity = healthy_continuity();
        continuity.focus.content =
            "# ACTIVE FOCUS\n\n## Status\n\n## Blocker\n\n## Next\n\n## Done / Gate\n".to_string();
        let health = assess(&snapshot, &continuity, "continue task", 131_072);
        let block = render_prompt_block(&health);
        assert!(block.contains("Context health:"));
        assert!(block.contains("repair_guidance:"));
    }

    #[test]
    fn forgotten_lines_count_as_negative_memory() {
        let snapshot = sample_snapshot(
            vec![
                ("user", "retry deploy"),
                (
                    "assistant",
                    "Status: `blocked`\n\nBlocker: deploy still failing",
                ),
                ("user", "retry deploy"),
            ],
            10_000,
        );
        let mut continuity = healthy_continuity();
        continuity.narrative.content =
            "# CONTINUITY NARRATIVE\n\n## Ausgangslage\nRunning on host A\n\n## Problem\nQueue drift\n\n## Ursache\n\n## Wendepunkte\n\n## Dauerhafte Entscheidungen\n\n## Aktueller Stand\nRepair pending\n\n## Offene Spannung\n".to_string();
        let forgotten = vec![lcm::ContinuityForgottenEntry {
            commit_id: "c1".to_string(),
            conversation_id: 1,
            kind: lcm::ContinuityKind::Narrative,
            line: "Do not retry deploy.sh until the missing secret is restored.".to_string(),
            created_at: "2026-03-31T00:00:00Z".to_string(),
        }];
        let health =
            assess_with_forgotten(&snapshot, &continuity, &forgotten, "retry deploy", 131_072);
        assert!(!health
            .warnings
            .iter()
            .any(|warning| warning.code == "failure_memory_missing"));
    }

    #[test]
    fn critical_warning_overrides_healthy_score_band() {
        let snapshot = sample_snapshot(vec![("user", "continue task")], 0);
        let mut continuity = healthy_continuity();
        continuity.focus.content =
            "# ACTIVE FOCUS\n\n## Status\n\n## Blocker\n\n## Next\n\n## Done / Gate\n".to_string();
        let health = assess(&snapshot, &continuity, "continue task", 131_072);
        assert_eq!(health.status, ContextHealthStatus::Critical);
        assert!(health.summary.contains("strongest severity critical"));
    }

    #[test]
    fn repair_governor_blocks_recursive_repair() {
        let snapshot = sample_snapshot(vec![("user", "continue task")], 120_000);
        let mut continuity = healthy_continuity();
        continuity.focus.content =
            "# ACTIVE FOCUS\n\n## Status\n\n## Blocker\n\n## Next\n\n## Done / Gate\n".to_string();
        let health = assess(&snapshot, &continuity, "continue task", 131_072);
        let decision = evaluate_repair_governor(
            &health,
            "context-health",
            "Review and repair CTOX context health",
            false,
            0,
        );
        assert!(!decision.should_enqueue_repair);
        assert!(decision.reason.contains("already is"));
    }
}
