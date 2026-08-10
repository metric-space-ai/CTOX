## I-072 Abschlussbericht

### Ergebnis

I-072 ist innerhalb der vorgegebenen Source-Whitelist umgesetzt. Es gab kein `git add` und keinen Commit. `turn_loop.rs`, `guard.rs`, der I-070-Seed und die fremden Metadata-Hunks blieben unangetastet.

Bearbeitete Repository-Dateien:

- `src/core/context/lcm/mod.rs`
- `src/core/mission/channels/mod.rs`
- `src/core/mission/channels/tests.rs`
- `src/core/service/service.rs`

Der vorgeschriebene Fortschrittsbericht wurde unter `/Volumes/tmp/ctox-pipeline/i072-fortschritt.md` aktualisiert.

### Telemetrie-Entscheidung

Alle drei Repair-Pfade verwenden jetzt die bestehende persistente Oberfläche `governance_events` über:

- `governance::record_event_if_new`
- bestehende Mechanik `state_invariant_guard`
- deterministischen Idempotenzschlüssel  
  `repair:<repair_kind>:<stable_digest(context_key)>`

Jeder Eintrag enthält:

- Ursache in `reason`
- tatsächliche Reparatur in `action_taken`
- strukturierten Kontext in `details_json`
- `repair_kind`
- `context_key`
- gegebenenfalls `message_key`

Diese Oberfläche wurde gewählt, weil:

1. sie bereits dauerhaft und über einen Unique-Key dedupliziert ist;
2. keine neue Event-Mechanik oder neues Governance-Vokabular eingeführt werden musste;
3. eine neue Governance-Registrierung Änderungen außerhalb der Whitelist erfordert hätte;
4. alle drei Reparaturen einen unterbrochenen oder inkonsistenten dauerhaften Zustand wiederherstellen und damit zur vorhandenen State-Invariant-Mechanik passen.

Instrumentiert wurden:

- CV-Parser-Kompakt-Recovery
- Preserve-Lease für spezialisierte Business-OS-App-Recovery
- Founder-Kommunikations-Reparaturen:
  - unreviewed `handled` zurück in Rework
  - stalled/failed Kommunikation zurück in den Routing-Loop
  - durch späteren geprüften Thread-Versand überholte Kommunikation schließen

Die Tests belegen jeweils einen dauerhaften Datensatz mit Ursache und Kontext. Der zweite identische Aufruf liefert `false` beziehungsweise erzeugt keinen zweiten Datensatz.

### Attempt-Bindung: vorher / nachher

| Pfad | Vorher | Nachher |
|---|---|---|
| Normales handled/cancelled Ack | bereits attempt-gebunden | unverändert attempt-gebunden |
| Typed Hold | plain `hold_leased_messages` | `hold_leased_messages_for_attempt`; Hold, Retry-Zähler, Zeitstempel, Projektion und Marker in einer Transaktion |
| Runtime/API-Retry-Hold | plain Hold über `release_retryable_worker_messages` | ebenfalls an dieselbe Attempt-ID gebunden |
| Pending / review_rework | plain Ack | `ack_leased_messages_for_attempt` |
| Failed Ack mit Fehlergrund | plain Ack/Failure-Reason | `ack_leased_messages_for_attempt(..., Some(reason))` |
| CV-Recovery Ack | plain handled Ack | attempt-gebundenes handled Ack |
| App-Validation-Terminalfehler | plain failed Ack | attempt-gebundenes failed Ack |
| Business-Command-Writeback | bei Resume erneut ausführbar | Gate über `queue_effects_applied_at`; ein bereits dauerhaft `handled` gesetzter Queue-Effekt wird ohne Queue-Rewrite in den Attempt-Marker übernommen |
| Outcome-Witness-Recovery-Enqueue | bei Resume erneut ausführbar | separater dauerhafter `recovery_effects_applied_at`-Marker |

Für Business-Command-Writebacks deckt der neue Beweistest das reale Fenster ab:

1. `complete_business_command_from_queue_reply` schreibt den Effekt und setzt die Queue auf `handled`.
2. Der Attempt-Marker ist noch nicht gesetzt – simulierter Crash.
3. Beim Resume erkennt der Gate den bereits dauerhaften Queue-Endzustand.
4. Die Writeback-Closure wird nicht erneut ausgeführt.
5. Der bestehende Queue-Zeitstempel wird nicht verändert.
6. Der vorhandene Queue-Effekt wird in `queue_effects_applied_at` übernommen.

### Neue beziehungsweise erweiterte Beweistests

- `attempt_bound_hold_and_failed_ack_do_not_rewrite_on_resume`
  - zweiter Hold erhöht weder Retry-Zähler noch Zeitstempel;
  - zweites failed Ack verändert weder `updated_at`, `acked_at` noch `last_error`.

- `resumed_attempt_does_not_repeat_business_command_writeback_after_ack`
  - tatsächlicher Business-Command-Writeback genau einmal;
  - Resume im Fenster nach Queue-Ack und vor Attempt-Marker;
  - keine zweite Closure-Ausführung;
  - genau eine terminale Business-Command-Transition;
  - keine zweite Queue-Timestamp-Schreibung.

- `resumed_attempt_does_not_repeat_outcome_recovery_enqueue`
  - erster Enqueue-Aufruf genau einmal;
  - Marker über DB-Neuöffnung sichtbar;
  - Resume führt keinen zweiten Enqueue aus.

- `cv_print_repair_telemetry_is_durable_contextual_and_deduplicated`
  - persistente Ursache, Attempt-ID, Message-Key und Update-Anzahl;
  - identischer zweiter Repair dedupliziert.

- `boot_release_preserves_business_os_app_leases_for_specialized_recovery`
  - zwei identische Preserve-Sweeps;
  - genau ein dauerhafter Governance-Event mit Ursache und Message-Kontext.

- `stalled_founder_email_superseded_by_later_reviewed_thread_send_is_cancelled`
  - tatsächliche Founder-Reparatur;
  - dauerhafter Cause/Action/Thread/Status-Kontext;
  - identischer zweiter Telemetrie-Aufruf dedupliziert.

### Testzahlen

Vorgeschriebene Umgebung:

```text
CARGO_TARGET_DIR=/Volumes/tmp/ctox-pipeline/i070-target
CARGO_INCREMENTAL=0
```

#### `service::service_loop`

Vorher:

```text
381 Tests
375 passed
6 failed
```

Nachher:

```text
384 Tests
379 passed
5 failed
```

Die fünf nachher roten Tests sind eine Teilmenge der sechs Baseline-Fehler:

- `business_os_app_queue_jobs_use_lean_mcp_free_session`
- `business_os_app_recovery_idle_gate_ignores_unrelated_churn_and_reopens_on_app_sources`
- `reviewed_founder_reply_closes_stale_rework_item`
- `stalled_founder_email_requeues_blocked_rework`
- `stop_guard_blocks_leased_business_os_rxdb_app_queue_task`

Der vorher rote Test `idle_durable_queue_empty_gate_ignores_sync_run_metadata_churn` war im Abschlusslauf grün. Es wurde keine I-072-spezifische Behebung dieses fremden Tests beansprucht.

**Keine neue rote Regression.** Alle neuen I-072-Service-Tests liefen im vollständigen Abschlusslauf grün.

#### `mission::queue`

Vorher:

```text
15 passed / 0 failed
```

Nachher:

```text
15 passed / 0 failed
```

Zusätzlich:

- Attempt-bound Hold/Failed-Ack: `1 passed`
- Worker-Attempt-LCM-Regressions: `2 passed`
- Runtime-Retry-Hold: `1 passed`

### Format, Scope und Symbolschluss

- `cargo fmt --check`: grün
- `git diff --check`: grün
- Alle neu referenzierten externen Symbole wurden gegen `HEAD` geprüft.
- Die drei neuen lokalen APIs sind innerhalb derselben Whitelist-Änderung definiert.
- Keine I-072-Hunks referenzieren das nur im dirty Baum vorhandene Metadata-Drive-by.
- Keine Änderungen an `turn_loop.rs`, `guard.rs` oder I-070-Seed.
- Alle vier I-072-Dateien sind ausschließlich unstaged.
- Kein `git add`, kein Commit.

### Offene Punkte

1. `state_invariant_guard` ist die passendste bereits registrierte Governance-Mechanik, ihr bisheriger Beschreibungstext ist jedoch enger als die nun erfassten CV-, Lease- und Founder-Reparaturen. Eine präzisere neue Mechanik hätte eine nicht freigegebene Änderung an `governance.rs` erfordert.

2. Outcome-Recovery liegt weiterhin in einer prozesslokalen Queue. Deshalb wird `recovery_effects_applied_at` nach erfolgreichem Enqueue geschrieben: Bei einem echten Prozessabbruch ist der alte In-Memory-Eintrag verloren und darf erneut erzeugt werden. Der Marker verhindert die geforderte Wiederholung bei einer späteren Finalisierungsaufnahme im bestehenden Prozess beziehungsweise nach bereits persistiertem Marker, ist aber keine gemeinsame Transaktion mit der In-Memory-Queue.

3. Bei einem Batch mit mehreren Business-Command-Nachrichten könnte ein Fehler nach einem bereits erfolgreichen Teil-Writeback, aber vor Abschluss der übrigen Nachrichten, weiterhin eine teilweise Wiederholung verlangen. Das geforderte Fenster nach dauerhaftem Queue-Acknowledgment ist geschlossen; eine atomare Batch-Writeback-Transaktion über den Business-OS-Store lag außerhalb der Whitelist.

## Workjet-Completion-Receipt v1

```yaml
workjet_completion_receipt: v1
role: implementation_agent
work_id: I-072
verdict: completed_no_new_red_tests
telemetry:
  surface: governance_events
  api: governance::record_event_if_new
  mechanism_id: state_invariant_guard
  dedupe: "repair:<repair_kind>:<stable_digest(context_key)>"
  repair_paths:
    cv_parser_recovery: durable_tested
    preserve_lease: durable_tested
    founder_communication_repair: durable_tested
attempt_binding:
  typed_hold: bound
  runtime_retry_hold: bound
  pending_ack: bound
  review_rework_ack: bound
  failed_ack_with_reason: bound
  cv_recovery_ack: bound
  business_command_writeback_resume: gated_and_tested
  outcome_witness_recovery_enqueue: marked_and_tested
acceptance:
  service_before: "375 passed / 6 failed / 381 total"
  service_after: "379 passed / 5 failed / 384 total"
  mission_queue_before: "15 passed / 0 failed"
  mission_queue_after: "15 passed / 0 failed"
  new_red_tests: 0
  cargo_fmt_check: passed
  git_diff_check: passed
scope:
  source_whitelist_respected: true
  turn_loop_touched: false
  guard_touched: false
  i070_seed_touched: false
  dirty_metadata_drive_by_touched: false
git:
  staged: false
  committed: false
elapsed:
  started: "2026-08-10 17:06 CEST"
  finished: "2026-08-10 18:24 CEST"
  within_100_minute_limit: true
```
