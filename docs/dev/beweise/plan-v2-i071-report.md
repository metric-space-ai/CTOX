## I-071 Abschlussbericht — Atomarer Attempt-Abschluss

### Ergebnis

I-071 ist implementiert. Worker-Abschlüsse besitzen jetzt einen dauerhaften Attempt-Datensatz, der vor Abschlusswirkungen als `finalizing` geschrieben und erst nach einem frischen Artifact-/Outcome-Check terminalisiert wird.

Erreicht wurden insbesondere:

- Autoritative erfolgreiche Assistant-Nachrichten mit `messages.agent_outcome='Success'`.
- Genau ein recoverable Attempt pro logischer Arbeit.
- Idempotente Wiederaufnahme ohne erneuten Modellaufruf.
- Keine zweite Assistant-Reply und kein zweiter normaler Queue-Ack nach Wiederaufnahme.
- Review und Outcome Witness vor Failure-Counter-Reset.
- Kein Counter-Reset bei transientem `ExecutionError`.
- Timeout-Originalattempt wird vor dem Kindtask als `timed_out`, `resumable` persistiert.
- Symmetrische Wiederherstellung von `mission_status`, `is_open`, `allow_idle`, `agent_failure_count` und `deferred_reason`.
- Alle drei bestehenden Sicherheitsnetze bleiben erhalten:
  - Outcome-Witness-Recovery.
  - Timeout-Kindtask.
  - Agent-Failure-Recovery.
- Queue-Vokabular und `work.outcome`-Projektionsformat blieben unverändert.
- I-070-Seed-Logik wurde nicht verändert.
- `src/core/core_state/guard.rs` blieb unverändert.

---

## Architekturentscheidung

### Storage Surface

Der Attempt-Datensatz liegt in der bestehenden Core-/LCM-SQLite-Datenbank in:

```sql
worker_attempt_finalizations
```

Diese Oberfläche wurde gewählt, weil LCM bereits die unmittelbar zusammengehörenden autoritativen Daten besitzt:

- `messages`
- `messages.agent_outcome`
- `mission_states`
- `lcm_data_migrations`

Damit können Attempt-Identität, typed Assistant-Reply und Mission-Recovery in derselben Datenbank koordiniert werden. Die Communication Queue verwendet im Service ebenfalls diese Core-Datenbank, wodurch der normale Queue-Ack und sein Attempt-Effektmarker in einer gemeinsamen SQLite-Transaktion geschrieben werden können.

Eine separate Datenbank hätte eine zusätzliche Cross-Database-Recovery-Grenze erzeugt und die Bindung zwischen Attempt und Assistant-Nachricht geschwächt.

### Schema und Lebenszyklus

Der Attempt besitzt die Zustände:

- `finalizing`
- `succeeded`
- `failed`
- `timed_out`

Zusätzlich trennt `effects_completed` den terminalen Prozessausgang vom vollständigen Abschluss aller nachgelagerten Wirkungen.

Wichtige Bindungen:

- `attempt_id`: Identität des konkreten Worker-Attempts.
- `work_key`: stabile Identität der logischen Arbeit.
- `reply_message_id`: bindet genau eine Assistant-Nachricht.
- `artifact_checked_at`: Voraussetzung jeder Terminalisierung.
- `queue_effects_applied_at`: macht den normalen Communication-Queue-Ack idempotent.
- `resumable`: kennzeichnet insbesondere den ursprünglichen Timeout-Attempt.
- `effects_completed`: entscheidet, ob der Attempt wieder aufgenommen werden muss.

Ein partieller Unique-Index erlaubt pro `work_key` nur einen Attempt mit `effects_completed=0`.

Der Check auf bestehende Attempt-ID, bestehende recoverable Arbeit und das Einfügen des neuen `finalizing`-Datensatzes laufen in einer `IMMEDIATE`-Transaktion. Dadurch wird auch ein Cross-Process-Rennen sauber auf den bereits autoritativen Attempt aufgelöst.

### Logische Atomarität

Externe Artifact-Dateien und spezialisierte Business-OS-Zustandsmaschinen können nicht Teil derselben SQLite-Transaktion sein. Deshalb wird logische Atomarität durch einen dauerhaften Recovery-Ablauf hergestellt:

1. Attempt als `finalizing` persistieren.
2. Assistant-Nachricht einmalig persistieren und über `reply_message_id` binden.
3. Completion Review und Outcome Witness ausführen.
4. Frische Artifact-Beobachtung im Attempt persistieren.
5. Queue- und weitere Abschlusswirkungen anwenden.
6. Attempt terminalisieren beziehungsweise `effects_completed` setzen.
7. Bei Absturz denselben Attempt laden und den Modellaufruf überspringen.

Bei einem internen Finalisierungsfehler wird nicht mehr der vorhandene Panic-Pfad verwendet. Insbesondere wird kein erfundener terminaler Queue-Fehler geschrieben; der Attempt bleibt recoverable.

---

## Geänderte Dateien

- `src/core/context/lcm/mod.rs`
  - Attempt-Typen, Migration und Persistenzoperationen.
  - Idempotente Assistant-Nachricht.
  - Artifact-Check, Terminalisierung und Effects-Abschluss.
  - Vollständige Mission-Recovery.

- `src/core/context/lcm/tests.rs`
  - Success-/Typed-Reply-Beweis.
  - Crash-/Resume-Beweis ohne doppelte Reply.

- `src/core/execution/agent/turn_loop.rs`
  - `WorkerAttemptContext`.
  - Erfolgreiche Assistant-Turns werden typed als `Success` geschrieben.
  - Service-Attempts werden vor der Assistant-Nachricht als `finalizing` persistiert.

- `src/core/service/service.rs`
  - Stabile Work-Key-Bildung und Attempt-Wiederaufnahme.
  - Review-/Witness-Reihenfolge.
  - Timeout-Terminalisierung vor Kindtask.
  - Attempt-gebundene Queue-Acks.
  - Vollständige Defer-/Recover-Symmetrie.
  - Kein Panic-Cleanup für normale Finalisierungsfehler.
  - Beweistests.

- `src/core/mission/channels/mod.rs`
  - `ack_leased_messages_for_attempt`.
  - Queue-Transition, Projektion und `queue_effects_applied_at` in einer Transaktion.

`src/core/core_state/guard.rs` wurde nicht geändert: Der bestehende Outcome-Witness-/Core-Transition-Pfad konnte direkt aus dem Service weiterverwendet werden. Eine Guard-Änderung hätte keinen zusätzlichen Beweiswert geliefert.

---

## Consumer-Liste

### Attempt-Erstellung und Reply-Bindung

- `execution::agent::turn_loop::persist_successful_assistant_with_retry`
- `service::service_loop::start_prompt_worker`

### Attempt-Wiederaufnahme

- `service::service_loop::start_prompt_worker`
  - lädt `run_recoverable_worker_attempt`
  - rekonstruiert das Ergebnis aus dem Attempt
  - überspringt den Modellaufruf

### Artifact-Check und Terminalisierung

- Erfolgs-/Review-Pfad in `start_prompt_worker`
- Failure-Pfad in `start_prompt_worker`
- `finalize_timeout_attempt_then_enqueue`

### Attempt-gebundener Queue-Ack

- Normaler reviewed-handled Queue-Abschluss.
- Cancelled-/No-Send-Abschluss.
- Terminale Worker-Failure-Routen.
- Generische nicht-pending Failure-Routen.
- `terminalize_reviewed_queue_messages`.

### Unveränderte Projektion

- `record_work_outcome_flow_event` bleibt Consumer des lossy `work.outcome`.
- Das Event ist weiterhin Projektion, nicht autoritativer Attempt-Datensatz.

---

## Geforderte Beweise

Alle I-071-Beweise sind grün:

1. **Erfolgreicher Turn**
   - Genau ein Attempt-Datensatz.
   - Attempt trägt `Success`.
   - Assistant-Nachricht trägt `agent_outcome='Success'`.
   - Wiederholte Persistenz liefert dieselbe Message-ID.

2. **Crash zwischen `finalizing` und terminal**
   - Datenbank wird geschlossen und erneut geöffnet.
   - Derselbe recoverable Attempt wird geladen.
   - Dieselbe Assistant-Message-ID wird verwendet.
   - Queue-Ack-Replay liefert `0`.
   - `acked_at` bleibt unverändert.

3. **Abgelehnter Witness nach Prozess-Success**
   - Failure-Counter bleibt unverändert und wird nicht zurückgesetzt.

4. **Timeout**
   - Originalattempt wird zuerst `timed_out`.
   - `resumable=true`.
   - Erst danach existiert der Kindtask.
   - Test vergleicht `terminal_at <= child.observed_at`.

5. **Defer → Recover**
   - Wiederhergestellt werden:
     - `mission_status="active"`
     - `is_open=true`
     - `allow_idle=false`
     - `agent_failure_count=0`
     - `deferred_reason=None`

Relevante fokussierte Tests:

```text
worker_attempt_success_persists_one_marker_and_typed_reply ... ok
worker_attempt_finalizing_crash_resumes_without_duplicate_reply ... ok
successful_turn_persists_one_attempt_and_typed_success_reply ... ok
queue_ack_refreshes_business_os_command_projection ... ok
rejected_outcome_witness_does_not_reset_failure_counter ... ok
mission_failure_counter_increments_resets_and_defers ... ok
timeout_blocker_queues_durable_artifact_recovery ... ok
```

---

## Suite-Vergleich

| Suite | Vorher | Nachher | Bewertung |
|---|---:|---:|---|
| `service::service_loop` | 375 passed, 5 known failed, 2434 filtered | 376 passed, 5 known failed, 2437 filtered | Keine neuen Roten |
| `service::state_invariants` | 5 passed, 0 failed | 5 passed, 0 failed | Grün |
| `mission::queue` | 15 passed, 0 failed | 15 passed, 0 failed | Grün |
| `context::` | 79 passed, 2 known failed, 2733 filtered | 81 passed, 2 known failed, 2735 filtered | Keine neuen Roten |

Die fünf weiterhin bekannten `service_loop`-Fehler sind unverändert:

- `business_os_app_queue_jobs_use_lean_mcp_free_session`
- `business_os_app_recovery_idle_gate_ignores_unrelated_churn_and_reopens_on_app_sources`
- `reviewed_founder_reply_closes_stale_rework_item`
- `stalled_founder_email_requeues_blocked_rework`
- `stop_guard_blocks_leased_business_os_rxdb_app_queue_task`

Die zwei weiterhin bekannten Context-Fehler sind unverändert:

- `context::context_stress::tests::stress_harness_survives_adversarial_summarizer`
- `context::lcm::tests::continuity_prompt_contains_document_and_diff_rules`

Ein zusätzlicher `idle_durable_queue_empty_gate_ignores_sync_run_metadata_churn`-Fehler trat einmal im parallelen Lauf auf. Der Test bestand unmittelbar danach fokussiert; der bestätigende vollständige `service_loop`-Lauf enthielt wieder exakt nur die fünf bekannten Roten.

### Weitere Prüfungen

- `cargo check --bin ctox`: grün.
- `cargo fmt --check`: grün.
- `git diff --check` für alle Whitelist-Dateien: grün.

Logs liegen unter `/Volumes/tmp/ctox-pipeline/`, insbesondere:

- `i071-final-service-loop-rerun.log`
- `i071-final-state-invariants.log`
- `i071-final-mission-queue.log`
- `i071-final-context.log`
- `i071-final-turn-loop-success-proof.log`
- `i071-final-integrated-crash-proof.log`
- `i071-check-final.log`
- `i071-fmt-check-final-2.log`

---

## Offene Punkte als Akzeptanzkriterien

1. **Spezialisierte Terminal-Owner**
   - Business-OS-App-Validation, Hold/Rework und Ticket-Event-Pfade behalten ihre bestehenden, eigenen Zustandsmaschinen.
   - Akzeptanzkriterium: Diese Pfade müssen weiterhin idempotent beziehungsweise fail-closed bleiben und dürfen nicht durch einen zweiten konkurrierenden Terminal-Owner ersetzt werden.
   - Die bestehenden Abnahmesuiten für diese Pfade zeigen keine neue Regression.

2. **Normale Communication-Queue**
   - Akzeptanzkriterium: Jeder normale terminale Queue-Ack eines Worker-Attempts muss über `ack_leased_messages_for_attempt` laufen.
   - Die aktuellen Consumer erfüllen dies; der integrierte Crash-Test beweist unverändertes `acked_at` bei Replay.

3. **Gemeinsame Datenbank**
   - Akzeptanzkriterium: LCM und Communication Queue müssen weiterhin auf denselben Core-DB-Pfad auflösen.
   - Der Service verwendet dafür durchgehend `crate::paths::core_db(root)` beziehungsweise die darauf auflösende Channel-DB.

4. **Process-Success versus Completion-Success**
   - Ein Attempt kann `agent_outcome=Success`, aber terminal `status=failed` tragen, wenn Review oder Witness den Abschluss ablehnt.
   - Akzeptanzkriterium: `agent_outcome` beschreibt den Agent-Prozessausgang; der Attempt-Status beschreibt den autoritativen Abschluss. Diese Ebenen dürfen nicht zusammengelegt werden.

5. **Externe Artifacts**
   - Dateisystem-Artefakte sind nicht Bestandteil der SQLite-Transaktion.
   - Akzeptanzkriterium: Jede Terminalisierung muss weiterhin einen frischen, persistierten Artifact-Check voraussetzen. `terminalize_worker_attempt` verweigert die Transition andernfalls.

Eine Erweiterung der atomaren Marker auf zusätzliche Ticket- oder spezialisierte Terminal-Owner-APIs würde weitere Aufrufer außerhalb der aktuellen Änderung benötigen und müsste als eigene Arbeit mit entsprechend erweiterter Whitelist erfolgen.

---

## Repository-Hygiene

- Kein `git add`.
- Kein Commit.
- Keine Subagenten.
- Keine Änderung an `core_state/guard.rs`.
- Keine absichtliche Änderung außerhalb der erlaubten Dateien.
- Bereits vorhandene fremde Dirty-Dateien wurden nicht bearbeitet.

## Workjet Completion Receipt v1

```yaml
work_id: I-071
title: "Der atomare Attempt-Abschluss — Worker-Finalisierung als eine Transition"
status: completed
authoritative_surface: "core SQLite / worker_attempt_finalizations"
terminal_states:
  - succeeded
  - failed
  - timed_out
recovery_marker: "status=finalizing OR effects_completed=0"
typed_success_messages: true
idempotent_reply_binding: true
idempotent_normal_queue_ack: true
timeout_terminal_before_child: true
review_and_witness_before_counter_reset: true
symmetric_defer_recovery: true
safety_nets_preserved:
  - outcome_witness_recovery
  - timeout_child_task
  - agent_failure_recovery
acceptance:
  cargo_check: passed
  cargo_fmt_check: passed
  service_loop: "376 passed; same 5 known failures"
  state_invariants: "5 passed; 0 failed"
  mission_queue: "15 passed; 0 failed"
  context: "81 passed; same 2 known failures"
git_add: false
git_commit: false
```
