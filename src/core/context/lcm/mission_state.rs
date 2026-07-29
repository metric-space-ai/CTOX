// Mission state: load/persist with the clobber guard, derivation from
// the free-text continuity document (the P8b review's core finding),
// focus repair, and the status/mode canonicalizers.

use super::{
    continuity_commit_id, continuity_document_id, fetch_continuity_document_with, iso_now,
    ContinuityKind, ContinuityShowAll, MissionClaimRecord, MissionStateRecord,
    StrategicDirectiveRecord, VerificationRunRecord,
};
use anyhow::{anyhow, bail, Context, Result};
use rusqlite::{params, Connection, OptionalExtension};
use std::cell::RefCell;
use std::path::Path;

pub(super) fn load_mission_state_with(
    conn: &Connection,
    conversation_id: i64,
) -> Result<Option<MissionStateRecord>> {
    conn.query_row(
        "SELECT mission, mission_status, continuation_mode, trigger_intensity, blocker, next_slice, done_gate, closure_confidence, is_open, allow_idle, focus_head_commit_id, last_synced_at, watcher_last_triggered_at, watcher_trigger_count, agent_failure_count, deferred_reason, rewrite_failure_count FROM mission_states WHERE conversation_id = ?1",
        [conversation_id],
        |row| {
            Ok(MissionStateRecord {
                conversation_id,
                mission: row.get(0)?,
                mission_status: row.get(1)?,
                continuation_mode: row.get(2)?,
                trigger_intensity: row.get(3)?,
                blocker: row.get(4)?,
                next_slice: row.get(5)?,
                done_gate: row.get(6)?,
                closure_confidence: row.get(7)?,
                is_open: row.get::<_, i64>(8)? != 0,
                allow_idle: row.get::<_, i64>(9)? != 0,
                focus_head_commit_id: row.get(10)?,
                last_synced_at: row.get(11)?,
                watcher_last_triggered_at: row.get(12)?,
                watcher_trigger_count: row.get(13)?,
                agent_failure_count: row.get(14)?,
                deferred_reason: row.get(15)?,
                rewrite_failure_count: row.get(16)?,
            })
        },
    )
    .optional()
    .context("failed to load mission state")
}

pub(super) fn load_mission_states_with(
    conn: &Connection,
    open_only: bool,
) -> Result<Vec<MissionStateRecord>> {
    let mut stmt = conn
        .prepare(
            "SELECT conversation_id, mission, mission_status, continuation_mode, trigger_intensity, blocker, next_slice, done_gate, closure_confidence, is_open, allow_idle, focus_head_commit_id, last_synced_at, watcher_last_triggered_at, watcher_trigger_count, agent_failure_count, deferred_reason, rewrite_failure_count
             FROM mission_states
             WHERE (?1 = 0 OR is_open = 1)
             ORDER BY is_open DESC, last_synced_at DESC, conversation_id ASC",
        )
        .context("failed to prepare mission state listing query")?;
    let rows = stmt.query_map(params![if open_only { 1 } else { 0 }], |row| {
        Ok(MissionStateRecord {
            conversation_id: row.get(0)?,
            mission: row.get(1)?,
            mission_status: row.get(2)?,
            continuation_mode: row.get(3)?,
            trigger_intensity: row.get(4)?,
            blocker: row.get(5)?,
            next_slice: row.get(6)?,
            done_gate: row.get(7)?,
            closure_confidence: row.get(8)?,
            is_open: row.get::<_, i64>(9)? != 0,
            allow_idle: row.get::<_, i64>(10)? != 0,
            focus_head_commit_id: row.get(11)?,
            last_synced_at: row.get(12)?,
            watcher_last_triggered_at: row.get(13)?,
            watcher_trigger_count: row.get(14)?,
            agent_failure_count: row.get(15)?,
            deferred_reason: row.get(16)?,
            rewrite_failure_count: row.get(17)?,
        })
    })?;
    rows.collect::<rusqlite::Result<Vec<_>>>()
        .map_err(anyhow::Error::from)
        .context("failed to load mission states")
}

/// One recorded attempt to clobber a protected `mission_states` field. The
/// guard preserved the prior non-empty value; this entry is staged on the
/// thread-local buffer and flushed to `governance_events` after the
/// surrounding transaction commits (governance writes open a separate
/// connection and would deadlock against an open lcm write transaction on
/// the same DB if emitted inline).
#[derive(Debug, Clone)]
pub(crate) struct PendingMissionStateClobberAttempt {
    pub conversation_id: i64,
    pub field: &'static str,
    pub previous_value: String,
    pub attempted_value: String,
    pub previous_value_chars: usize,
}

thread_local! {
    /// Per-thread buffer of suppressed clobber attempts. Drained by
    /// `LcmEngine::drain_pending_mission_state_clobber_events_to_governance`
    /// once the surrounding lcm transaction has committed and a governance
    /// connection can safely be opened.
    static PENDING_MISSION_STATE_CLOBBERS: RefCell<Vec<PendingMissionStateClobberAttempt>> =
        const { RefCell::new(Vec::new()) };
}

pub(super) fn push_pending_mission_state_clobber(attempt: PendingMissionStateClobberAttempt) {
    PENDING_MISSION_STATE_CLOBBERS.with(|cell| cell.borrow_mut().push(attempt));
}

pub(crate) fn drain_pending_mission_state_clobbers() -> Vec<PendingMissionStateClobberAttempt> {
    PENDING_MISSION_STATE_CLOBBERS.with(|cell| std::mem::take(&mut *cell.borrow_mut()))
}

/// Drain any clobber attempts that the P2 guard suppressed during this
/// thread's recent persist calls and publish them as
/// `mission_state_field_clobbered_blocked` governance events. Safe to call
/// from any post-turn / post-boot maintenance pass: a no-op when the buffer
/// is empty. Failures are swallowed so the audit channel never breaks a
/// successful state transition (mirrors the `let _ =
/// governance::record_event(...)` pattern in service.rs).
pub fn drain_pending_mission_state_clobber_events_to_governance(root: &Path) {
    let pending = drain_pending_mission_state_clobbers();
    for attempt in pending {
        crate::governance::record_event_or_count(
            root,
            crate::governance::GovernanceEventRequest {
                mechanism_id: "mission_state_field_clobbered_blocked",
                conversation_id: Some(attempt.conversation_id),
                severity: "warning",
                reason: "mission_state_field_clobber_blocked",
                action_taken: "preserved_prior_non_empty_field",
                details: serde_json::json!({
                    "field": attempt.field,
                    "previous_value_chars": attempt.previous_value_chars,
                    "previous_value": attempt.previous_value,
                    "attempted_value": attempt.attempted_value,
                }),
                idempotence_key: None,
            },
        );
    }
}

#[cfg(test)]
pub(crate) fn drain_pending_mission_state_clobbers_for_test(
) -> Vec<PendingMissionStateClobberAttempt> {
    drain_pending_mission_state_clobbers()
}

/// True when `value` is empty after trimming whitespace (`""`, `"   "`,
/// `"\n"`, etc. all collapse to "this writer cleared the field"). Structural
/// — no parsing, no string-matching against any sentinel.
pub(super) fn is_blank_field(value: &str) -> bool {
    value.trim().is_empty()
}

/// Bypass key the dedicated owner-intent clearer flips before issuing a
/// legitimate clear. Wired through a thread-local so we don't have to
/// thread an extra parameter through every persist path.
thread_local! {
    static OWNER_INTENT_CLEAR_BYPASS_DEPTH: RefCell<u32> = const { RefCell::new(0) };
}

pub(super) struct OwnerIntentClearGuard;
impl OwnerIntentClearGuard {
    pub(super) fn enter() -> Self {
        OWNER_INTENT_CLEAR_BYPASS_DEPTH.with(|cell| *cell.borrow_mut() += 1);
        Self
    }
}
impl Drop for OwnerIntentClearGuard {
    fn drop(&mut self) {
        OWNER_INTENT_CLEAR_BYPASS_DEPTH.with(|cell| {
            let mut depth = cell.borrow_mut();
            if *depth > 0 {
                *depth -= 1;
            }
        });
    }
}

pub(super) fn owner_intent_clear_active() -> bool {
    OWNER_INTENT_CLEAR_BYPASS_DEPTH.with(|cell| *cell.borrow() > 0)
}

pub(super) fn persist_mission_state_with(
    conn: &Connection,
    record: &MissionStateRecord,
) -> Result<()> {
    // P2 — Mission-state field clobber guard.
    //
    // Production smoke-test (Befund C) saw `next_slice` (81 chars) and
    // `done_gate` (289 chars) silently collapse to length 0 within ~25
    // minutes while `mission` (217 chars) was preserved. The suspected
    // writer is `derive_mission_state_from_continuity`, which produces
    // empty `next_slice` / `done_gate` strings whenever the focus
    // continuity document does not currently carry an explicit
    // `next_slice:` / `done_gate:` line. That overwrite path is a
    // mission-continuity-normalize pass triggered by every
    // `continuity_apply_diff` / full-replace / string-replace / sync.
    //
    // We install a one-way ratchet on `next_slice` and `done_gate`: once
    // they hold non-empty content, automation may only replace them with
    // new non-empty content. A blank-incoming write while the prior row
    // is non-empty preserves the prior value field-locally, and the
    // attempted clobber is staged on a thread-local buffer that the
    // engine flushes to `governance_events` once the surrounding
    // transaction has committed (we cannot open a second connection
    // against the same WAL DB while a write transaction is still open
    // on this thread without risking a busy_timeout deadlock).
    //
    // Operator/skill paths that legitimately *want* to clear these
    // fields call `clear_mission_state_done_fields_with_owner_intent`,
    // which sets a thread-local bypass for the duration of the clear.
    let mut effective_next_slice = record.next_slice.clone();
    let mut effective_done_gate = record.done_gate.clone();
    if !owner_intent_clear_active() {
        let existing = load_mission_state_with(conn, record.conversation_id)?;
        if let Some(existing) = existing {
            if !is_blank_field(&existing.next_slice) && is_blank_field(&effective_next_slice) {
                push_pending_mission_state_clobber(PendingMissionStateClobberAttempt {
                    conversation_id: record.conversation_id,
                    field: "next_slice",
                    previous_value: existing.next_slice.clone(),
                    attempted_value: effective_next_slice.clone(),
                    previous_value_chars: existing.next_slice.chars().count(),
                });
                effective_next_slice = existing.next_slice;
            }
            if !is_blank_field(&existing.done_gate) && is_blank_field(&effective_done_gate) {
                push_pending_mission_state_clobber(PendingMissionStateClobberAttempt {
                    conversation_id: record.conversation_id,
                    field: "done_gate",
                    previous_value: existing.done_gate.clone(),
                    attempted_value: effective_done_gate.clone(),
                    previous_value_chars: existing.done_gate.chars().count(),
                });
                effective_done_gate = existing.done_gate;
            }
        }
    }

    conn.execute(
        "INSERT INTO mission_states (
            conversation_id,
            mission,
            mission_status,
            continuation_mode,
            trigger_intensity,
            blocker,
            next_slice,
            done_gate,
            closure_confidence,
            is_open,
            allow_idle,
            focus_head_commit_id,
            last_synced_at,
            watcher_last_triggered_at,
            watcher_trigger_count,
            agent_failure_count,
            deferred_reason,
            rewrite_failure_count
        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18)
        ON CONFLICT(conversation_id) DO UPDATE SET
            mission = excluded.mission,
            mission_status = excluded.mission_status,
            continuation_mode = excluded.continuation_mode,
            trigger_intensity = excluded.trigger_intensity,
            blocker = excluded.blocker,
            next_slice = excluded.next_slice,
            done_gate = excluded.done_gate,
            closure_confidence = excluded.closure_confidence,
            is_open = excluded.is_open,
            allow_idle = excluded.allow_idle,
            focus_head_commit_id = excluded.focus_head_commit_id,
            last_synced_at = excluded.last_synced_at,
            watcher_last_triggered_at = excluded.watcher_last_triggered_at,
            watcher_trigger_count = excluded.watcher_trigger_count,
            agent_failure_count = excluded.agent_failure_count,
            deferred_reason = excluded.deferred_reason,
            rewrite_failure_count = excluded.rewrite_failure_count",
        params![
            record.conversation_id,
            record.mission,
            record.mission_status,
            record.continuation_mode,
            record.trigger_intensity,
            record.blocker,
            effective_next_slice,
            effective_done_gate,
            record.closure_confidence,
            if record.is_open { 1 } else { 0 },
            if record.allow_idle { 1 } else { 0 },
            record.focus_head_commit_id,
            record.last_synced_at,
            record.watcher_last_triggered_at,
            record.watcher_trigger_count,
            record.agent_failure_count,
            record.deferred_reason,
            record.rewrite_failure_count,
        ],
    )?;
    Ok(())
}

pub(super) fn derive_mission_state_from_continuity(
    continuity: &ContinuityShowAll,
    previous: Option<&MissionStateRecord>,
) -> MissionStateRecord {
    let contract_lines = continuity_section_lines(&continuity.focus.content, "Contract");
    let state_lines = continuity_section_lines(&continuity.focus.content, "State");
    let legacy_status_lines = continuity_section_lines(&continuity.focus.content, "Status");
    let legacy_blocker_lines = continuity_section_lines(&continuity.focus.content, "Blocker");
    let legacy_next_lines = continuity_section_lines(&continuity.focus.content, "Next");
    let legacy_gate_lines = continuity_section_lines(&continuity.focus.content, "Done / Gate");

    let mission = last_named_value(&contract_lines, &["mission", "goal"])
        .or_else(|| last_named_value(&legacy_status_lines, &["Mission"]))
        .or_else(|| first_non_meta_line(&contract_lines))
        .or_else(|| first_non_meta_line(&legacy_status_lines))
        .filter(|value| !value.trim().is_empty())
        .or_else(|| previous.map(|record| record.mission.clone()))
        .unwrap_or_default();
    let mission_status = canonicalize_mission_status(
        last_named_value(&contract_lines, &["mission_state", "mission state"])
            .or_else(|| last_named_value(&legacy_status_lines, &["Mission state"]))
            .as_deref(),
    )
    .or_else(|| previous.map(|record| record.mission_status.clone()))
    .unwrap_or_else(|| "active".to_string());
    let continuation_mode = canonicalize_continuation_mode(
        last_named_value(&contract_lines, &["continuation_mode", "continuation mode"])
            .or_else(|| last_named_value(&legacy_status_lines, &["Continuation mode"]))
            .as_deref(),
    )
    .or_else(|| previous.map(|record| record.continuation_mode.clone()))
    .unwrap_or_else(|| "continuous".to_string());
    let trigger_intensity = canonicalize_trigger_intensity(
        last_named_value(&contract_lines, &["trigger_intensity", "trigger intensity"])
            .or_else(|| last_named_value(&legacy_status_lines, &["Trigger intensity"]))
            .as_deref(),
    )
    .or_else(|| previous.map(|record| record.trigger_intensity.clone()))
    .unwrap_or_else(|| "hot".to_string());
    let blocker = last_named_value_allow_empty(&state_lines, &["blocker", "current blocker"])
        .or_else(|| last_named_value_allow_empty(&legacy_blocker_lines, &["Current blocker"]))
        .or_else(|| first_meaningful_line(&state_lines))
        .or_else(|| first_meaningful_line(&legacy_blocker_lines))
        .unwrap_or_default();
    let next_slice = last_named_value_allow_empty(&state_lines, &["next_slice", "next slice"])
        .or_else(|| last_named_value_allow_empty(&legacy_next_lines, &["Next slice"]))
        .or_else(|| first_meaningful_line(&state_lines))
        .or_else(|| first_meaningful_line(&legacy_next_lines))
        .unwrap_or_default();
    let done_gate = last_named_value_allow_empty(&state_lines, &["done_gate", "done gate"])
        .or_else(|| last_named_value_allow_empty(&legacy_gate_lines, &["Done gate"]))
        .or_else(|| first_non_meta_line(&state_lines))
        .or_else(|| first_non_meta_line(&legacy_gate_lines))
        .unwrap_or_default();
    let closure_confidence = canonicalize_closure_confidence(
        last_named_value(&state_lines, &["closure_confidence", "closure confidence"])
            .or_else(|| last_named_value(&legacy_gate_lines, &["Closure confidence"]))
            .as_deref(),
    )
    .or_else(|| previous.map(|record| record.closure_confidence.clone()))
    .unwrap_or_else(|| "low".to_string());
    let is_open = mission_is_open(
        &mission,
        &mission_status,
        &continuation_mode,
        &next_slice,
        &done_gate,
        &closure_confidence,
    );
    let allow_idle = mission_allows_idle(&mission_status, &continuation_mode, &trigger_intensity);

    MissionStateRecord {
        conversation_id: continuity.conversation_id,
        mission,
        mission_status,
        continuation_mode,
        trigger_intensity,
        blocker,
        next_slice,
        done_gate,
        closure_confidence,
        is_open,
        allow_idle,
        focus_head_commit_id: continuity.focus.head_commit_id.clone(),
        last_synced_at: iso_now(),
        watcher_last_triggered_at: previous
            .and_then(|record| record.watcher_last_triggered_at.clone()),
        watcher_trigger_count: previous
            .map(|record| record.watcher_trigger_count)
            .unwrap_or(0),
        agent_failure_count: previous
            .map(|record| record.agent_failure_count)
            .unwrap_or(0),
        deferred_reason: previous.and_then(|record| record.deferred_reason.clone()),
        rewrite_failure_count: previous
            .map(|record| record.rewrite_failure_count)
            .unwrap_or(0),
    }
}

pub(super) fn maybe_repair_focus_continuity_with(
    conn: &Connection,
    continuity: &mut ContinuityShowAll,
    previous: Option<&MissionStateRecord>,
) -> Result<bool> {
    if focus_semantic_conflicts_local(&continuity.focus.content).is_empty() {
        return Ok(false);
    }

    let repaired_content = render_canonical_focus_continuity(continuity, previous);
    if repaired_content.trim() == continuity.focus.content.trim() {
        return Ok(false);
    }

    let created_at = iso_now();
    let commit_id = continuity_commit_id(
        continuity.conversation_id,
        ContinuityKind::Focus,
        "## Status\n+ Canonicalized conflicting focus fields during mission-state resync.\n",
        &repaired_content,
        &created_at,
    );
    let document_id = continuity_document_id(continuity.conversation_id, ContinuityKind::Focus);
    conn.execute(
        "INSERT INTO continuity_commits (commit_id, document_id, parent_commit_id, diff_text, rendered_text, created_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
        params![
            commit_id,
            document_id,
            continuity.focus.head_commit_id,
            "## Status\n+ Canonicalized conflicting focus fields during mission-state resync.\n",
            repaired_content,
            created_at
        ],
    )?;
    conn.execute(
        "UPDATE continuity_documents SET head_commit_id = ?1, updated_at = ?2 WHERE document_id = ?3",
        params![commit_id, created_at, document_id],
    )?;
    continuity.focus =
        fetch_continuity_document_with(conn, continuity.conversation_id, ContinuityKind::Focus)?
            .context("focus continuity missing after repair")?;
    Ok(true)
}

pub(super) fn render_canonical_focus_continuity(
    continuity: &ContinuityShowAll,
    previous: Option<&MissionStateRecord>,
) -> String {
    let record = derive_mission_state_from_continuity(continuity, previous);
    render_focus_continuity_from_record(continuity, &record)
}

pub(super) fn render_focus_continuity_from_record(
    continuity: &ContinuityShowAll,
    record: &MissionStateRecord,
) -> String {
    let contract_lines = continuity_section_lines(&continuity.focus.content, "Contract");
    let state_lines = continuity_section_lines(&continuity.focus.content, "State");
    let legacy_gate_lines = continuity_section_lines(&continuity.focus.content, "Done / Gate");
    let source_lines = continuity_section_lines(&continuity.focus.content, "Sources");
    let retry_condition = last_named_value(&state_lines, &["retry_condition", "retry condition"])
        .or_else(|| last_named_value(&legacy_gate_lines, &["Retry condition"]))
        .unwrap_or_default();
    let missing_dependency =
        last_named_value(&state_lines, &["missing_dependency", "missing dependency"])
            .unwrap_or_default();
    let slice = last_named_value(&contract_lines, &["slice"])
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| record.next_slice.clone());
    let slice_state = last_named_value(&contract_lines, &["slice_state", "slice state"])
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| {
            if record.is_open {
                "active".to_string()
            } else {
                "closed".to_string()
            }
        });
    let canonical_source_lines = canonical_focus_source_lines(&source_lines);
    let mut lines = vec![
        "# ACTIVE FOCUS".to_string(),
        String::new(),
        "## Status".to_string(),
        format!("- Mission: {}", record.mission),
        format!("- Mission state: {}", record.mission_status),
        format!("- Continuation mode: {}", record.continuation_mode),
        format!("- Trigger intensity: {}", record.trigger_intensity),
        String::new(),
        "## Blocker".to_string(),
        format!("- Current blocker: {}", record.blocker),
        String::new(),
        "## Next".to_string(),
        format!("- Next slice: {}", record.next_slice),
        String::new(),
        "## Done / Gate".to_string(),
        format!("- Done gate: {}", record.done_gate),
        format!("- Retry condition: {}", retry_condition),
        format!("- Closure confidence: {}", record.closure_confidence),
        String::new(),
        "## Contract".to_string(),
        format!("- mission: {}", record.mission),
        format!("- mission_state: {}", record.mission_status),
        format!("- continuation_mode: {}", record.continuation_mode),
        format!("- trigger_intensity: {}", record.trigger_intensity),
        format!("- slice: {}", slice),
        format!("- slice_state: {}", slice_state),
        String::new(),
        "## State".to_string(),
        format!("- goal: {}", record.mission),
        format!("- blocker: {}", record.blocker),
        format!("- missing_dependency: {}", missing_dependency),
        format!("- next_slice: {}", record.next_slice),
        format!("- done_gate: {}", record.done_gate),
        format!("- retry_condition: {}", retry_condition),
        format!("- closure_confidence: {}", record.closure_confidence),
        String::new(),
        "## Sources".to_string(),
    ];
    lines.extend(
        canonical_source_lines
            .into_iter()
            .map(|line| format!("- {line}")),
    );
    lines.push(String::new());
    lines.join("\n")
}

pub(super) fn canonical_focus_source_lines(lines: &[String]) -> Vec<String> {
    let mut out = Vec::new();
    for line in lines {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if !out.iter().any(|existing: &String| {
            normalize_mission_text(existing) == normalize_mission_text(trimmed)
        }) {
            out.push(trimmed.to_string());
        }
    }
    if out.is_empty() {
        out.push("source_refs:".to_string());
        out.push("none".to_string());
        out.push("updated_at:".to_string());
    }
    out
}

pub(super) fn focus_semantic_conflicts_local(content: &str) -> Vec<String> {
    let tracked_fields = [
        "Mission",
        "Mission state",
        "Continuation mode",
        "Trigger intensity",
        "Current blocker",
        "Next slice",
        "Done gate",
        "Closure confidence",
    ];
    let mut seen: std::collections::BTreeMap<&'static str, Vec<String>> =
        std::collections::BTreeMap::new();

    for raw_line in content.lines() {
        let line = raw_line.trim_start_matches(['-', '+', '*', ' ']).trim();
        if line.is_empty() {
            continue;
        }
        let Some((name, value)) = line.split_once(':') else {
            continue;
        };
        for field in tracked_fields {
            if normalize_mission_text(name) == normalize_mission_text(field) {
                let value = value.trim();
                if !value.is_empty() {
                    seen.entry(field).or_default().push(value.to_string());
                }
            }
        }
    }

    let mut conflicts = Vec::new();
    for (field, values) in seen {
        let mut distinct = Vec::new();
        for value in values {
            if !distinct.iter().any(|existing: &String| {
                normalize_mission_text(existing) == normalize_mission_text(&value)
            }) {
                distinct.push(value);
            }
        }
        if distinct.len() > 1 {
            conflicts.push(format!("{field} has conflicting values {:?}", distinct));
        }
    }
    conflicts
}

pub(super) fn map_verification_run_row(
    row: &rusqlite::Row<'_>,
    conversation_id: i64,
) -> rusqlite::Result<VerificationRunRecord> {
    let review_reasons: Vec<String> =
        serde_json::from_str(&row.get::<_, String>(10)?).unwrap_or_default();
    let failed_gates: Vec<String> =
        serde_json::from_str(&row.get::<_, String>(14)?).unwrap_or_default();
    let semantic_findings: Vec<String> =
        serde_json::from_str(&row.get::<_, String>(15)?).unwrap_or_default();
    let open_items: Vec<String> =
        serde_json::from_str(&row.get::<_, String>(16)?).unwrap_or_default();
    let evidence: Vec<String> =
        serde_json::from_str(&row.get::<_, String>(17)?).unwrap_or_default();
    let handoff = row.get::<_, String>(18)?;
    Ok(VerificationRunRecord {
        run_id: row.get(0)?,
        conversation_id,
        source_label: row.get(1)?,
        goal: row.get(2)?,
        preview: row.get(3)?,
        result_excerpt: row.get(4)?,
        blocker: row.get(5)?,
        review_required: row.get::<_, i64>(6)? != 0,
        review_verdict: row.get(7)?,
        review_summary: row.get(8)?,
        review_score: row.get(9)?,
        review_reasons,
        report_excerpt: row.get(11)?,
        raw_report: row.get(12)?,
        mission_state: row.get(13)?,
        failed_gates,
        semantic_findings,
        open_items,
        evidence,
        handoff: if handoff.trim().is_empty() {
            None
        } else {
            Some(handoff)
        },
        claim_count: row.get(19)?,
        open_claim_count: row.get(20)?,
        closure_blocking_claim_count: row.get(21)?,
        created_at: row.get(22)?,
    })
}

pub(super) fn map_mission_claim_row(
    row: &rusqlite::Row<'_>,
    conversation_id: i64,
) -> rusqlite::Result<MissionClaimRecord> {
    Ok(MissionClaimRecord {
        claim_key: row.get(0)?,
        conversation_id,
        last_run_id: row.get(1)?,
        claim_kind: row.get(2)?,
        claim_status: row.get(3)?,
        blocks_closure: row.get::<_, i64>(4)? != 0,
        subject: row.get(5)?,
        summary: row.get(6)?,
        evidence_summary: row.get(7)?,
        recheck_policy: row.get(8)?,
        expires_at: row.get(9)?,
        created_at: row.get(10)?,
        updated_at: row.get(11)?,
    })
}

/// Count open, closure-blocking mission claims for a conversation.
///
/// Single source of truth for the closure-assurance gate
/// (`core_transition_guard::validate_closure_assurance_claims`). The
/// open-claim predicate mirrors `LcmEngine::list_mission_claims` with
/// `include_verified = false` (a claim is open while it is not yet `verified`,
/// or while its verification has expired), restricted here to
/// `blocks_closure = 1`. This is the same set `mission_assurance_snapshot`
/// exposes as `closure_blocking_claims`, but counted authoritatively (no
/// display `LIMIT`) so the gate sees every open blocker.
///
/// Tolerates a database whose LCM schema has not been initialized yet
/// (`mission_claims` absent) by reporting zero open blockers, so a transition
/// guard running against a bare connection never errors on the lookup.
pub fn count_open_closure_blocking_claims(conn: &Connection, conversation_id: i64) -> Result<i64> {
    let now_millis: i64 = iso_now().parse().unwrap_or(i64::MAX);
    let result = conn.query_row(
        "SELECT COUNT(*) FROM mission_claims
         WHERE conversation_id = ?1
           AND blocks_closure = 1
           AND (claim_status != 'verified'
                OR (expires_at IS NOT NULL AND CAST(expires_at AS INTEGER) <= ?2))",
        params![conversation_id, now_millis],
        |row| row.get::<_, i64>(0),
    );
    match result {
        Ok(count) => Ok(count),
        Err(rusqlite::Error::SqliteFailure(_, Some(message)))
            if message.contains("no such table") =>
        {
            // A bare connection (LCM schema never initialized) legitimately
            // has zero blockers. A database that HAS the LCM base tables but
            // lost mission_claims is a schema regression — the closure gate
            // must fail loudly instead of silently swinging open.
            let lcm_initialized: i64 = conn.query_row(
                "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='messages')",
                [],
                |row| row.get(0),
            )?;
            if lcm_initialized != 0 {
                bail!(
                    "mission_claims table is missing from an initialized LCM database — \
                     refusing to report zero closure-blocking claims"
                );
            }
            Ok(0)
        }
        Err(err) => Err(err.into()),
    }
}

pub(super) fn map_strategic_directive_row(
    row: &rusqlite::Row<'_>,
) -> rusqlite::Result<StrategicDirectiveRecord> {
    Ok(StrategicDirectiveRecord {
        directive_id: row.get(0)?,
        conversation_id: row.get(1)?,
        thread_key: row.get(2)?,
        directive_kind: row.get(3)?,
        title: row.get(4)?,
        body_text: row.get(5)?,
        status: row.get(6)?,
        revision: row.get(7)?,
        previous_directive_id: row.get(8)?,
        author: row.get(9)?,
        decided_by: row.get(10)?,
        decision_reason: row.get(11)?,
        created_at: row.get(12)?,
        updated_at: row.get(13)?,
    })
}

pub(super) fn continuity_section_lines(content: &str, section_name: &str) -> Vec<String> {
    let mut active = false;
    let mut lines = Vec::new();
    for line in content.lines() {
        let trimmed = line.trim();
        if let Some(header) = trimmed.strip_prefix("## ") {
            active = header == section_name;
            continue;
        }
        if active && !trimmed.is_empty() && !trimmed.starts_with('#') {
            lines.push(trimmed.trim_start_matches("- ").trim().to_string());
        }
    }
    lines
}

pub(super) fn last_named_value(lines: &[String], names: &[&str]) -> Option<String> {
    let mut out = None;
    for line in lines {
        if let Some((prefix, value)) = line.split_once(':') {
            if names
                .iter()
                .any(|name| prefix.trim().eq_ignore_ascii_case(name))
            {
                let value = value.trim();
                if !value.is_empty() {
                    out = Some(value.to_string());
                }
            }
        }
    }
    out
}

pub(super) fn last_named_value_allow_empty(lines: &[String], names: &[&str]) -> Option<String> {
    let mut out = None;
    for line in lines {
        if let Some((prefix, value)) = line.split_once(':') {
            if names
                .iter()
                .any(|name| prefix.trim().eq_ignore_ascii_case(name))
            {
                out = Some(value.trim().to_string());
            }
        }
    }
    out
}

pub(super) fn first_non_meta_line(lines: &[String]) -> Option<String> {
    lines
        .iter()
        .find(|line| !line.contains(':'))
        .map(|line| line.trim().to_string())
        .filter(|line| !line.is_empty())
}

pub(super) fn first_meaningful_line(lines: &[String]) -> Option<String> {
    lines
        .iter()
        .map(|line| line.trim())
        .find(|line| {
            !line.is_empty()
                && !line.ends_with(':')
                && !line.eq_ignore_ascii_case("none")
                && !line.eq_ignore_ascii_case("kein")
                && !line.eq_ignore_ascii_case("keiner")
                && !line.eq_ignore_ascii_case("no blocker")
                && !line.eq_ignore_ascii_case("n/a")
                && !line.eq_ignore_ascii_case("na")
        })
        .map(ToOwned::to_owned)
}

pub(super) fn canonicalize_mission_status(raw: Option<&str>) -> Option<String> {
    let raw = raw?;
    let normalized = normalize_mission_text(raw);
    match normalized.as_str() {
        "done" | "complete" | "completed" | "closed" | "abgeschlossen" => Some("done".to_string()),
        "maintenance" => Some("maintenance".to_string()),
        "scheduled" => Some("scheduled".to_string()),
        "dormant" => Some("dormant".to_string()),
        "open" | "active" | "ongoing" | "in progress" => Some("active".to_string()),
        _ => None,
    }
}

pub(super) fn canonicalize_continuation_mode(raw: Option<&str>) -> Option<String> {
    let raw = raw?;
    let normalized = normalize_mission_text(raw);
    match normalized.as_str() {
        "maintenance" => Some("maintenance".to_string()),
        "scheduled" | "cron" => Some("scheduled".to_string()),
        "dormant" | "archive" => Some("dormant".to_string()),
        "closed" => Some("closed".to_string()),
        "continuous" | "continue" | "open" | "reopen" | "reopened" | "resume" | "active"
        | "ongoing" => Some("continuous".to_string()),
        _ => None,
    }
}

pub(super) fn canonicalize_trigger_intensity(raw: Option<&str>) -> Option<String> {
    let raw = raw?;
    let normalized = normalize_mission_text(raw);
    match normalized.as_str() {
        "archive" => Some("archive".to_string()),
        "cold" | "low" => Some("cold".to_string()),
        "warm" | "medium" | "moderate" => Some("warm".to_string()),
        "hot" | "high" | "urgent" => Some("hot".to_string()),
        _ => None,
    }
}

pub(super) fn canonicalize_closure_confidence(raw: Option<&str>) -> Option<String> {
    let raw = raw?;
    let normalized = normalize_mission_text(raw);
    match normalized.as_str() {
        "complete" | "completed" | "certain" => Some("complete".to_string()),
        "high" => Some("high".to_string()),
        "medium" | "moderate" => Some("medium".to_string()),
        "low" | "partial" | "provisional" | "tentative" | "pending" | "unverified" | "unclear"
        | "unknown" => Some("low".to_string()),
        _ => None,
    }
}

pub(super) fn mission_is_open(
    mission: &str,
    mission_status: &str,
    continuation_mode: &str,
    next_slice: &str,
    done_gate: &str,
    closure_confidence: &str,
) -> bool {
    let status = normalize_mission_text(mission_status);
    let mode = normalize_mission_text(continuation_mode);
    let _ = closure_confidence;
    if status == "done" || mode == "closed" || mode == "dormant" {
        return false;
    }
    !mission.trim().is_empty() || !next_slice.trim().is_empty() || !done_gate.trim().is_empty()
}

pub(super) fn mission_allows_idle(
    mission_status: &str,
    continuation_mode: &str,
    trigger_intensity: &str,
) -> bool {
    let status = normalize_mission_text(mission_status);
    let mode = normalize_mission_text(continuation_mode);
    let intensity = normalize_mission_text(trigger_intensity);
    status == "done"
        || mode == "closed"
        || mode == "dormant"
        || (mode == "scheduled" && intensity != "hot")
}

pub(super) fn normalize_mission_text(value: &str) -> String {
    value
        .to_ascii_lowercase()
        .chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { ' ' })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}
