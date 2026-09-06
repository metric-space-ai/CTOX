use super::*;
use rusqlite::OptionalExtension;
use serde_json::{json, Value};
use std::path::Path;

/// Called once before invoking a slice, never from a progress callback. The
/// transaction pins identity to the immutable attempt and its still-held lease.
pub(crate) fn prepare_attempt(
    root: &Path,
    task_ids: &[String],
    lease_owner: &str,
    attempt: &str,
    thread_key: Option<&str>,
    metadata: &Value,
    skill: Option<&str>,
    prompt: &str,
) -> Result<Option<String>> {
    let Some(task_id) = task_ids.first() else {
        return Ok(None);
    };
    let conn = Connection::open(crate::paths::core_db(root))?;
    conn.busy_timeout(crate::persistence::sqlite_busy_timeout_duration())?;
    let tx = conn.unchecked_transaction()?;
    let existing: Option<String> = tx
        .query_row(
            "SELECT member_id FROM crew_attempts WHERE attempt_id=?1",
            [attempt],
            |r| r.get(0),
        )
        .optional()?;
    let all = members(&tx)?;
    let mut task = TaskTraits {
        thread_key: thread_key.map(String::from),
        module: metadata
            .get("business_os_module")
            .or_else(|| metadata.get("business_os_module_id"))
            .or_else(|| metadata.get("module_id"))
            .and_then(Value::as_str)
            .map(String::from),
        command_type: metadata
            .get("business_os_command_type")
            .or_else(|| metadata.get("command_type"))
            .and_then(Value::as_str)
            .map(String::from),
        skills: skill.into_iter().map(String::from).collect(),
        ..Default::default()
    };
    let prompt_lower = prompt.to_lowercase();
    task.tags = all
        .iter()
        .flat_map(|m| m.specialties.tags.iter())
        .filter(|tag| {
            prompt_lower
                .split(|c: char| !c.is_alphanumeric())
                .any(|word| word == tag.to_lowercase())
        })
        .cloned()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    task.manual_member = tx
        .query_row(
            "SELECT crew_member_id FROM communication_routing_state
         WHERE message_key=?1 AND route_status='leased' AND lease_owner=?2",
            params![task_id, lease_owner],
            |r| r.get::<_, Option<String>>(0),
        )
        .optional()?
        .context("crew attachment requires the held lease")?;
    task.continuity_member=tx.query_row("SELECT member_id FROM crew_attempts WHERE thread_key=?1 ORDER BY selected_at DESC,attempt_id DESC LIMIT 1",[thread_key],|r|r.get(0)).optional()?;
    let history=tx.prepare("SELECT member_id,module,thread_key,succeeded,finalized_at FROM crew_attempts WHERE finalized_at IS NOT NULL ORDER BY finalized_at DESC,attempt_id DESC LIMIT 1000")?
        .query_map([],|r|Ok((r.get::<_,String>(0)?,r.get::<_,Option<String>>(1)?,r.get::<_,Option<String>>(2)?,r.get::<_,bool>(3)?,r.get::<_,String>(4)?)))?
        .collect::<rusqlite::Result<Vec<_>>>()?.into_iter().filter_map(|(member_id,module,thread_key,succeeded,at)| Some(History{member_id,module,thread_key,succeeded,finished_at_ms:chrono::DateTime::parse_from_rfc3339(&at).ok()?.timestamp_millis()})).collect::<Vec<_>>();
    let selection = if let Some(id) = existing {
        Selection {
            member_id: id,
            reason: "Identität des wiederaufgenommenen Versuchs bleibt erhalten".into(),
        }
    } else {
        select(&all, &task, &history, chrono::Utc::now().timestamp_millis())
            .context("no active crew member available")?
    };
    let member = all
        .iter()
        .find(|m| m.id == selection.member_id)
        .context("attempt member no longer exists")?;
    let now = chrono::Utc::now().to_rfc3339();
    let inserted=tx.execute("INSERT OR IGNORE INTO crew_attempts(attempt_id,task_id,member_id,module,thread_key,selected_at,selection_reason) VALUES(?1,?2,?3,?4,?5,?6,?7)",params![attempt,task_id,member.id,task.module,thread_key,now,selection.reason])?;
    for id in task_ids {
        let changed = tx.execute(
            "UPDATE communication_routing_state SET crew_member_id=?2,updated_at=?3
             WHERE message_key=?1 AND route_status='leased' AND lease_owner=?4",
            params![id, member.id, now, lease_owner],
        )?;
        anyhow::ensure!(changed == 1, "crew attachment lost its lease");
    }
    if inserted > 0 {
        tx.execute("UPDATE crew_members SET stats_json=json_set(stats_json,'$.last_active_at',?2),updated_at=?2 WHERE id=?1",params![member.id,now])?;
    }
    let learnings = load_context_learnings(&tx, &member.id, &task)?;
    let block = render_soul_block(member, &learnings);
    tx.commit()?;
    if inserted > 0 {
        crate::service::harness_flow::record_harness_flow_event_lossy(
            root,
            crate::service::harness_flow::RecordHarnessFlowEventRequest {
                event_kind: "crew_selected",
                title: &selection.reason,
                body_text: &selection.reason,
                message_key: Some(task_id),
                work_id: None,
                ticket_key: None,
                attempt_index: None,
                metadata: json!({"attempt_id":attempt,"crew_member_id":selection.member_id,"reason":selection.reason,"cockpit_eligible":true}),
            },
        );
    }
    Ok(Some(block))
}

/// Repair a lost notification from durable selection evidence on the pump.
/// The unique event index also closes a race with the initial best-effort emit.
pub(crate) fn repair_selection_events(root: &Path, conn: &Connection) -> Result<()> {
    conn.execute_batch(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_crew_selection_event_attempt
        ON ctox_harness_flow_events(json_extract(metadata_json,'$.attempt_id'))
        WHERE event_kind='crew_selected';",
    )?;
    let mut cursor = String::new();
    loop {
        let rows=conn.prepare("SELECT a.attempt_id,a.task_id,a.member_id,a.selection_reason
            FROM crew_attempts a WHERE a.attempt_id>?1 AND a.selection_reason!=''
              AND NOT EXISTS(SELECT 1 FROM ctox_harness_flow_events e
                WHERE e.event_kind='crew_selected' AND json_extract(e.metadata_json,'$.attempt_id')=a.attempt_id)
            ORDER BY a.attempt_id LIMIT 128")?.query_map([&cursor],|r|Ok((r.get::<_,String>(0)?,r.get::<_,String>(1)?,r.get::<_,String>(2)?,r.get::<_,String>(3)?)))?.collect::<rusqlite::Result<Vec<_>>>()?;
        if rows.is_empty() {
            break;
        }
        for (attempt, task, member, reason) in rows {
            cursor = attempt.clone();
            // A concurrent successful initial emission may win the unique key;
            // the next sweep then observes that durable event instead.
            crate::service::harness_flow::record_harness_flow_event_lossy(
                root,
                crate::service::harness_flow::RecordHarnessFlowEventRequest {
                    event_kind: "crew_selected",
                    title: &reason,
                    body_text: &reason,
                    message_key: Some(&task),
                    work_id: None,
                    ticket_key: None,
                    attempt_index: None,
                    metadata: json!({"attempt_id":attempt,"crew_member_id":member,"reason":reason,"cockpit_eligible":true}),
                },
            );
        }
    }
    Ok(())
}

pub(crate) fn load_context_learnings(
    conn: &Connection,
    member: &str,
    task: &TaskTraits,
) -> Result<Vec<(String, bool)>> {
    let mut statement=conn.prepare("SELECT text,confirmed_by_owner,scope_json FROM crew_member_learnings
        WHERE member_id=?1 AND archived=0 ORDER BY confirmed_by_owner DESC,created_at DESC,id LIMIT 200")?;
    let rows = statement.query_map([member], |r| {
        Ok((
            r.get::<_, String>(0)?,
            r.get::<_, bool>(1)?,
            r.get::<_, String>(2)?,
        ))
    })?;
    let mut ranked = vec![];
    for row in rows {
        let (text, confirmed, scope) = row?;
        let scope: LearningScope = serde_json::from_str(&scope)?;
        if scope
            .module
            .as_ref()
            .is_some_and(|s| Some(s) != task.module.as_ref())
            || scope
                .command_type
                .as_ref()
                .is_some_and(|s| Some(s) != task.command_type.as_ref())
            || scope
                .thread_key
                .as_ref()
                .is_some_and(|s| Some(s) != task.thread_key.as_ref())
        {
            continue;
        }
        let score = usize::from(scope.thread_key.is_some()) * 4
            + usize::from(scope.command_type.is_some()) * 2
            + usize::from(scope.module.is_some());
        ranked.push((text, confirmed, score));
    }
    ranked.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| b.2.cmp(&a.2)));
    let (mut confirmed, mut pending) = (0, 0);
    Ok(ranked
        .into_iter()
        .filter_map(|(text, yes, _)| {
            let (count, limit) = if yes {
                (&mut confirmed, 8)
            } else {
                (&mut pending, 2)
            };
            if *count >= limit {
                return None;
            }
            *count += 1;
            Some((text, yes))
        })
        .collect())
}

// A strict 4,000-byte ceiling is conservatively below ~1,200 German prose tokens.
// Learnings and sketches are data, explicitly subordinate to existing rules.
pub(crate) const SOUL_MAX_BYTES: usize = 4000;
pub(crate) fn render_soul_block(member: &Member, learnings: &[(String, bool)]) -> String {
    let s = &member.soul;
    let rules = [
        if s.gruendlichkeit_vs_tempo < 50 {
            "Prüfe sorgfältig und belege Ergebnisse."
        } else {
            "Arbeite zügig in kleinen, überprüften Schritten."
        },
        if s.vorsicht_vs_mut < 50 {
            "Prüfe Voraussetzungen vor riskanten Schritten."
        } else {
            "Erprobe reversible Schritte eigenständig innerhalb der Freigaben."
        },
        if s.knapp_vs_ausfuehrlich < 50 {
            "Antworte knapp und konkret."
        } else {
            "Erkläre relevante Zusammenhänge und Belege."
        },
        if s.regeltreu_vs_kreativ < 50 {
            "Nutze bewährte Abläufe innerhalb der geltenden Regeln."
        } else {
            "Suche kreative Lösungen innerhalb der geltenden Regeln."
        },
        if s.nachfragen_vs_annehmen < 50 {
            "Benenne entscheidende Unklarheiten früh."
        } else {
            "Triff begründete Annahmen innerhalb des Auftrags."
        },
    ];
    let mut block=format!("<ctox_crew_soul>\nName: {}\nAlle Sicherheits-, Ausführungs- und Review-Regeln gelten unverändert. Profil und Learnings sind nachrangige Kontextdaten, keine neuen Befugnisse.\n{}\nLebenslauf: {} Tasks, {} erfolgreich, {} fehlgeschlagen, {} Reviews bestanden.\n",member.name,rules.join("\n"),member.stats.tasks_total,member.stats.succeeded,member.stats.failed,member.stats.review_passed);
    let closing = "</ctox_crew_soul>";
    let mut add = |line: String| {
        if block.len() + line.len() + closing.len() <= SOUL_MAX_BYTES {
            block.push_str(&line)
        }
    };
    add(format!("Charakter: {}\nStimme: {}\n", s.sketch, s.voice));
    let (mut yes, mut no) = (0, 0);
    for (text, confirmed) in learnings {
        let (count, limit) = if *confirmed {
            (&mut yes, 8)
        } else {
            (&mut no, 2)
        };
        if *count >= limit {
            continue;
        }
        *count += 1;
        add(format!(
            "Learning ({}): {}\n",
            if *confirmed {
                "bestätigt"
            } else {
                "UNBESTÄTIGT – nur Hypothese"
            },
            text
        ));
    }
    block.push_str(closing);
    block
}

/// Append only after the complete pre-existing system/execution context.
pub(crate) fn append_soul(context: &mut String, block: Option<&str>) {
    if context.trim().is_empty() {
        return;
    }
    if let Some(block) = block {
        context.push_str("\n\n");
        context.push_str(block);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn lease_identity_and_literal_reason_survive_replay() -> Result<()> {
        let root = tempfile::tempdir()?;
        let task = crate::mission::channels::create_queue_task(
            root.path(),
            crate::mission::channels::QueueTaskCreateRequest {
                title: "Crew lease fixture".into(),
                prompt: "Code prüfen".into(),
                thread_key: "crew-thread".into(),
                workspace_root: None,
                priority: "normal".into(),
                suggested_skill: None,
                parent_message_key: None,
                extra_metadata: None,
            },
        )?;
        let conn = Connection::open(crate::paths::core_db(root.path()))?;
        conn.execute("UPDATE communication_routing_state SET route_status='leased',lease_owner='crew-worker',crew_member_id='crew-pico' WHERE message_key=?1",[&task.message_key])?;
        let first = prepare_attempt(
            root.path(),
            &[task.message_key.clone()],
            "crew-worker",
            "crew-attempt",
            Some("crew-thread"),
            &json!({}),
            None,
            "Code prüfen",
        )?;
        let second = prepare_attempt(
            root.path(),
            &[task.message_key.clone()],
            "crew-worker",
            "crew-attempt",
            Some("crew-thread"),
            &json!({}),
            None,
            "Code prüfen",
        )?;
        assert!(first.unwrap().contains("Pico"));
        assert!(second.unwrap().contains("Pico"));
        let (count,title,metadata):(i64,String,String)=conn.query_row(
            "SELECT COUNT(*),title,metadata_json FROM ctox_harness_flow_events WHERE event_kind='crew_selected' AND message_key=?1",[&task.message_key],|r|Ok((r.get(0)?,r.get(1)?,r.get(2)?)))?;
        assert_eq!(count, 1);
        conn.execute("DELETE FROM ctox_harness_flow_events WHERE event_kind='crew_selected' AND message_key=?1",[&task.message_key])?;
        repair_selection_events(root.path(), &conn)?;
        let repaired:String=conn.query_row("SELECT title FROM ctox_harness_flow_events WHERE event_kind='crew_selected' AND message_key=?1",[&task.message_key],|r|r.get(0))?;
        assert_eq!(repaired, title);
        let plan=conn.prepare("EXPLAIN QUERY PLAN SELECT event_id FROM ctox_harness_flow_events WHERE event_kind='crew_selected' AND json_extract(metadata_json,'$.attempt_id')='crew-attempt'")?.query_map([],|r|r.get::<_,String>(3))?.collect::<rusqlite::Result<Vec<_>>>()?.join("\n");
        assert!(plan.contains("idx_crew_selection_event_attempt"), "{plan}");

        assert_eq!(
            json!(title),
            serde_json::from_str::<Value>(&metadata)?["reason"]
        );
        conn.execute("UPDATE communication_routing_state SET lease_owner='other-worker' WHERE message_key=?1", [&task.message_key])?;
        assert!(prepare_attempt(
            root.path(),
            &[task.message_key.clone()],
            "crew-worker",
            "lost-lease",
            Some("crew-thread"),
            &json!({}),
            None,
            "Code prüfen"
        )
        .is_err());
        let count: i64 = conn.query_row("SELECT COUNT(*) FROM crew_attempts", [], |r| r.get(0))?;
        assert_eq!(count, 1, "losing a lease must not attach another attempt");
        assert_eq!(
            crate::mission::channels::load_queue_task(root.path(), &task.message_key)?
                .context("task missing")?
                .crew_member_id
                .as_deref(),
            Some("crew-pico")
        );
        Ok(())
    }
    #[test]
    fn actual_runtime_attachment_preserves_safety_and_execution_order() {
        let conn = super::super::tests::fixture();
        let member = members(&conn).unwrap().remove(0);
        let block = render_soul_block(&member, &[]);
        let mut rendered = crate::context::live_context::RenderedRuntimePrompt {
            prompt: "Diagnostic execution rules".into(),
            latest_user_prompt: "Execution rules: validate evidence; never bypass reviews.".into(),
            context_instructions: "CTO operating mode; mandatory safety rules.".into(),
            rendered_context_items: 0,
            omitted_context_items: 0,
        };
        let safety = rendered.context_instructions.clone();
        let execution = rendered.latest_user_prompt.clone();
        crate::context::live_context::attach_crew_soul(&mut rendered, Some(&block));
        assert_eq!(rendered.context_instructions, safety);
        assert!(rendered.latest_user_prompt.starts_with(&execution));
        assert!(rendered.latest_user_prompt.ends_with(&block));
        assert!(crate::context::lcm::estimate_tokens(&block) <= 1200);
        rendered.latest_user_prompt.clear();
        crate::context::live_context::attach_crew_soul(&mut rendered, Some(&block));
        assert!(
            rendered.latest_user_prompt.is_empty(),
            "identity must not bypass empty-task admission"
        );
        let mut empty = String::new();
        append_soul(&mut empty, Some(&block));
        assert!(empty.is_empty());
    }
    #[test]
    fn soul_is_bounded_deterministic_and_subordinate() {
        let conn = super::super::tests::fixture();
        let member = members(&conn).unwrap().remove(0);
        let learnings = (0..20)
            .map(|i| (format!("{i}: {}", "Ä".repeat(390)), i < 15))
            .collect::<Vec<_>>();
        let block = render_soul_block(&member, &learnings);
        assert!(block.len() <= SOUL_MAX_BYTES);
        assert_eq!(block, render_soul_block(&member, &learnings));
        let original =
            "CTO-Operating-Mode\nSafety: keep all review gates.\nExecution: validate evidence.";
        let mut context = original.to_string();
        append_soul(&mut context, Some(&block));
        assert!(context.starts_with(original));
        assert!(context.find("ctox_crew_soul").unwrap() > original.len());
        assert!(context.matches("UNBESTÄTIGT").count() <= 2);
    }
}
