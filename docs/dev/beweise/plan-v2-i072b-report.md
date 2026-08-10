## Ergebnis

I-072b ist umgesetzt. Eigene Änderungen liegen ausschließlich in `src/core/service/service.rs`; `src/core/mission/channels/mod.rs` wurde nicht verändert.

### Wahlbegründungen

1. **Founder-Telemetrie**
   - Gewählt wurde die Erweiterung des Idempotenz-Kontexts um `repair_outcome`, nicht eine Aufspaltung der `repair_kind`s.
   - Begründung: `repair_stalled_founder_communications` bezeichnet weiterhin denselben Reparaturmechanismus; das Outcome ist dessen semantischer Diskriminator.
   - Der Digest wird jetzt über das strukturierte Tupel `[context_key, repair_outcome]` gebildet.
   - Ergebnis: Verschiedene Outcomes derselben Message erhalten verschiedene dauerhafte Events; dieselbe Message mit identischem Outcome dedupliziert weiterhin.

2. **App-Validation-Rework**
   - Der Helper akzeptiert optional eine `attempt_id`, weil er zusätzlich in Recovery-Pfaden ohne Worker-Attempt verwendet wird.
   - Im Worker-Fehlerpfad wird `Some(attempt_id)` übergeben:
     - Bereits persistiertes `review_rework` wird beim Resume in den Attempt-Marker adoptiert.
     - Der erste Ack läuft über `ack_leased_messages_for_attempt`.
     - Ein Resume überspringt Feedback-, Proof-, Queue- und Projektions-Rewrites.
   - Recovery ohne Attempt verwendet weiterhin bewusst `None`.

3. **App-Validation-Success**
   - Beide Worker-Pfade übergeben `Some(attempt_id)`.
   - Vor der Ausführung wird ein bereits dauerhaftes `handled` adoptiert.
   - Nach erfolgreichem `complete_*` wird `queue_effects_applied_at` mit `mark_worker_attempt_queue_effects_applied_if_status(..., "handled")` gesetzt.
   - Nicht-attempt-basierte Recovery-Aufrufer bleiben unverändert über `None`.

### Vorher/Nachher

| Lauf | Ergebnis |
|---|---:|
| Baseline vorher | 379 passed / 5 failed / 384 |
| Finaler Wiederholungslauf | 379 passed / 5 failed / 384 |
| Warning-Zeilen vorher/nachher | 295 / 295 |
| Neue rote Tests | 0 |
| `cargo fmt --check` | grün |
| `git diff --check` | grün |

Die fünf finalen Fehler sind exakt die fünf Baseline-Fehler. Ein Zwischenlauf zeigte zusätzlich den unabhängigen Flake `idle_empty_gate_reopens_when_durable_queue_source_changes`; der Test lief isoliert grün und der anschließende vollständige Wiederholungslauf war wieder bei 379/5.

### Isolierte Beweistests

- `service::service_loop::tests::stalled_founder_email_superseded_by_later_reviewed_thread_send_is_cancelled`
  - grün
  - Beweist zwei verschiedene Outcomes derselben Message als zwei Datensätze und identische Wiederholung als Dedupe.

- `service::service_loop::tests::resumed_attempt_does_not_repeat_business_os_app_validation_rework_after_ack`
  - grün
  - Beweist unveränderten Prompt, unveränderte Queue-Zeitstempel und unveränderte Proof-Anzahl beim Resume.

- `service::service_loop::tests::business_os_app_validation_worker_error_after_green_marks_same_task_handled`
  - grün
  - Beweist den gesetzten `queue_effects_applied_at`-Marker nach App-Validation-Success.

### Weitere Prüfungen

- Neue API-Referenzen sind in `HEAD` definiert:
  - `ack_leased_messages_for_attempt`
  - `mark_worker_attempt_queue_effects_applied_if_status`
- Ressourcen-Gate beim Start: 322 GiB frei auf `/Volumes/tmp`, 21 GiB auf dem Repo-Volume, 5-Minuten-Load 6,15.
- Kein `git add`, kein Commit, keine Subagenten.
- Vertagte Panic-/Cleanup-/Working-Hours-Acks und Batch-Writeback-Replay wurden nicht angefasst.

### Offene Bedenken

- Die fünf bereits in der Baseline roten Tests bleiben offen.
- Recovery-Pfade ohne Worker-Attempt verwenden weiterhin plain Acks; dort existiert kein Attempt-Marker, an den gebunden werden könnte.
- Der Arbeitsbaum enthält umfangreiche vorbestehende Änderungen. Diese wurden erhalten; eigene Source-Hunks betreffen nur `service.rs`.

```yaml
workjet_completion_receipt: v1
role: implementation_agent
work_id: I-072b
verdict: completed_no_new_red_tests
implementation:
  founder_telemetry:
    strategy: include_repair_outcome_in_idempotence_context
    repair_kind_split: false
    distinct_outcomes_same_message: durable_separate_records
    identical_outcome_replay: deduplicated
  app_validation_rework:
    worker_failure_ack: attempt_bound
    resume_adoption_status: review_rework
    duplicate_feedback_write_on_tested_resume: prevented
    duplicate_projection_refresh_on_tested_resume: prevented
    non_attempt_recovery: retained
  app_validation_success:
    normal_worker_success_marker: bound
    after_worker_error_marker: bound
    resume_adoption_status: handled
acceptance:
  service_before: "379 passed / 5 failed / 384 total"
  service_after: "379 passed / 5 failed / 384 total"
  warning_lines_before: 295
  warning_lines_after: 295
  new_red_tests: 0
  cargo_fmt_check: passed
  git_diff_check: passed
proof_tests:
  stalled_founder_email_superseded_by_later_reviewed_thread_send_is_cancelled: passed
  resumed_attempt_does_not_repeat_business_os_app_validation_rework_after_ack: passed
  business_os_app_validation_worker_error_after_green_marks_same_task_handled: passed
symbol_closure:
  ack_leased_messages_for_attempt_in_HEAD: true
  mark_worker_attempt_queue_effects_applied_if_status_in_HEAD: true
scope:
  source_whitelist_respected: true
  service_rs_touched: true
  mission_channels_touched_by_this_work: false
  deferred_non_blocking_paths_touched: false
git:
  git_add_run: false
  commit_created: false
workspace:
  preexisting_dirty_changes_preserved: true
resource_gate:
  start_tmp_free_gib: 322
  start_repo_volume_free_gib: 21
  start_five_minute_load: 6.15
elapsed_bound: under_60m
```
