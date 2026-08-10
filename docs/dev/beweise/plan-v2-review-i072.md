Ich habe genug Evidenz am Commit-Tree. Hier das adversariale Review.

---

# Adversariales Review — I-072 (`4b5aca6ee`)

Read-only, commit-basiert (`git show 4b5aca6ee:<…>`), nicht der dirty Arbeitsbaum.  
Messgrundlage: `docs/dev/beweise/plan-v2-review-i071.md` (vertagte Q3/Q7), Worker-Report `docs/dev/beweise/plan-v2-i072-report.md`, Plan `docs/ctox-sync-plan-2026-08-10.md` §S2a.

Commit-Umfang (5 Dateien):  
`lcm/mod.rs`, `channels/mod.rs`, `channels/tests.rs`, `service.rs`, Report — **23 service-Hunks + 3 channels-Hunks**, exakt wie behauptet.

---

## 1) Montage: dirty-only-Symbole / zerstörte Nachbarschafts-Nähte

### ENTKRÄFTET — kein I-070/I-071-Montagebruch „Call ohne Definition“

| Symbol | Definiert in `4b5aca6ee`? | Aufgerufen? |
|---|---|---|
| `hold_leased_messages_for_attempt` | ja (`channels/mod.rs:2598`) | ja (Success-Hold, Runtime-Retry `service.rs:25795`) |
| `mark_worker_attempt_queue_effects_applied_if_status` | ja (`channels/mod.rs:2461+`) | ja (`terminalize_reviewed_queue_messages`, Writeback-Gate) |
| `mark_worker_attempt_recovery_effects_applied` / `run_mark_*` | ja (`lcm/mod.rs`) | ja (`apply_recovery_enqueue_for_attempt`) |
| `record_repair_telemetry` / `record_cv_print_parser_repair_telemetry` | ja (`service.rs`) | ja |
| `business_os_app_module_target_from_metadata` | **kein Call** | alle Call-Sites `…_from_prompt`; Definition bei `service.rs:11107` |
| `fail_business_command_from_queue_error` nach terminal review fail | ja (`service.rs:7729–7766`) | konsumiert `terminal_review_failure_reason` |

**Kleinster Check**

```bash
git grep -n 'business_os_app_module_target_from_metadata' 4b5aca6ee -- '*.rs'   # 0
git show 4b5aca6ee --unified=0 -- src/core/service/service.rs | rg -c '^@@'     # 23
git show 4b5aca6ee --unified=0 -- src/core/mission/channels/mod.rs | rg -c '^@@' # 3
```

### ENTKRÄFTET — Hunk-Auswahl zerstört keine bewusste Repair-Naht

- Parent (`c8314b308` / `4b5aca6ee^` für `service.rs`, Blob-identisch) hatte die I-071b-Naht `fail_business_command_from_queue_error` nach terminal review fail; sie bleibt im Commit.
- `hold_leased_messages`-Körper wird nach `hold_leased_messages_impl` gezogen und um Attempt-Gate + Marker in **derselben** TX ergänzt — Projektion, Zähler, `retry_not_before` bleiben im gleichen Block (`channels/mod.rs:2608–2778`).
- Success-Disposition: plain `hold`/`ack`/`ack_with_failure_reason` → `*_for_attempt` an denselben Stellen; kein Löschen des Failure-Projektionsblocks.

**Urteil Q1:** Montage **geschlossen**. Die Fehlerklasse I-070/I-071 (dirty-Symbol + geopferte Repair-Naht) greift hier **nicht**.

---

## 2) Drive-by

### ENTKRÄFTET für metadata / fremde Themenflächen

- Geänderte Sources: nur Whitelist-thematisch (Attempt-Bindung, Recovery-Marker-Spalte, Repair-Telemetrie, Tests, Report).
- Kein `turn_loop.rs` / `guard.rs` / Metadata-Migration.
- `recovery_effects_applied_at` per `ensure_column` nach I-071-Migration — notwendige Extension der Attempt-Fläche, kein Drive-by.

### REST (kein Drive-by, aber Scope-Ehrlichkeit)

Plain `ack`/`hold` außerhalb der Worker-Finalisierung bleiben (Boot-Repairs, Working-Hours, Panic-Drop, Lease-Cleanup). Das ist thematisch kein Finalisierungs-Attempt und war auch nicht I-072-Soll.

**Urteil Q2:** Drive-by **sauber vermieden**.

---

## 3) Telemetrie ehrlich? `repair:<kind>:<digest(context_key)>`

### ENTKRÄFTET für CV-Parser (Granularität passend)

```text
service.rs:10080–10092
repair_kind = complete_cv_print_parser_recovery_to_leased_queue
context_key = attempt_id
```

Zweiter **identischer** Call auf demselben Attempt → Dedup (getestet). Echter zweiter Repair auf **neuem** Attempt → neuer Key. Weder Spam pro Retry-Schleife desselben Attempts noch Verschlucken über Attempts hinweg.

### TEILWEISE KONFIRMIERT — Preserve-Lease (grob, aber handlungsneutral)

```text
service.rs:2122–2128
context_key = message_key  (Lebensdauer der Message)
```

Zweiter Boot mit derselben noch-stale Lease: Telemetrie `false`, Aktion weiter „preserve“ (`continue`). Kein Silent-Release. Spam-Schutz gewollt; ein **zweiter** semantisch neuer Preserve nach Re-Lease derselben Key-Identität würde telemetrisch verschluckt — real selten.

### KONFIRMIERT — Founder-Repairs: Digest **zu grob** (echter zweiter Repair verschluckt)

Drei **verschiedene** Outcomes teilen denselben Idempotenzschlüssel:

| Outcome | `repair_kind` | `context_key` |
|---|---|---|
| `restored_unreviewed_handled` | `repair_stalled_founder_communications` | `message.message_key` |
| `superseded_by_later_reviewed_thread_send` | **gleich** | **gleich** |
| `restored_stalled_founder_communication` | **gleich** | **gleich** |

`repair_outcome` steckt nur in `details_json`, **nicht** im Key (`record_repair_telemetry` `service.rs:1134–1138`).

**Szenario (kleinster Beweis am Code):** Message X wird erst `restored_unreviewed_handled` (Event geschrieben). Später legitimer Stall → `restored_stalled_*` oder `superseded_*`: **Repair-Aktion läuft**, `record_event_if_new` liefert `false` → **kein** zweiter dauerhafter Datensatz. Das widerspricht dem Report-Claim „je Repair ein dauerhafter Datensatz“.

Test `stalled_founder_email_superseded_…` prüft nur denselben Outcome erneut (`assert!(!record_repair_telemetry(…superseded…))`), nicht die Sequenz zweier Outcomes.

**Urteil Q3:** Surface + Persistenz ja; Founder-Idempotenz **unehrlich grob** → echte Zweit-Repairs telemetrisch unsichtbar. CV ok. Spam-Risiko gering (eher Under- als Over-Reporting).

---

## 4) Attempt-Bindung vollständig? (I-071-Reste + neue Lücken)

### ENTKRÄFTET für die in I-071 namentlich genannten Finalisierungs-Pfade (Kern)

| Pfad | Commit-Status |
|---|---|
| Typed Hold (Approved-Hold + Disposition-Hold) | `hold_leased_messages_for_attempt` (`service.rs:7655/7670`) |
| Runtime/API-Retry-Hold | `release_retryable_worker_messages` → `hold_*_for_attempt` (`:25795`) |
| pending / review_rework / failed (Success-Disposition) | `ack_*_for_attempt` (`:7693/7704`) |
| Failed mit Grund (Worker-Error-Route) | `ack_*_for_attempt(..., Some(reason))` (`:8136`) |
| CV-Recovery Ack | `ack_*_for_attempt(..., "handled")` (`:8212`) |
| App-Validation-Terminalfehler | `ack_*_for_attempt` failed (`:8256`, Success-Zweig analog) |
| Business-Command-Writeback Resume | Gate `queue_effects_applied_at` / `mark_if_status("handled")` (`:1057–1080`, Test resume-after-ack) |
| Witness-Recovery-Enqueue | `recovery_effects_applied_at` vor Prozess-lokalem Enqueue (`:1082–1097`, `:8700+`) |

I-071b-Repair-Naht bleibt.

### KONFIRMIERT — Restlücken (Claim „vollstaendig“ überzogen)

**A. Failure-Pfad App-Validation-Rework — plain `review_rework`**

```text
service.rs:8012 apply_business_os_app_validation_rework_to_leased_queue(...)
service.rs:11391 channels::ack_leased_messages(..., "review_rework")  // KEIN attempt_id
```

Danach wird der attempt-gebundene Failure-Ack-Block wegen `app_validation_rework` **übersprungen** (`:8101–8103`).  
Crash zwischen plain-Ack und `effects_completed=1` → Resume schreibt `review_rework`/Timestamps erneut.  
Success-Pfad setzt `app_validation_rework` **ohne** Helper und ackt später `for_attempt` — asymmetrisch.

**B. App-Validation-Terminal-Success (Success + after-worker-error)**

`complete_business_os_app_validation_success_to_leased_queue` terminalisiert Queue/Command **ohne** `queue_effects_applied_at`; Success-Zweig skippt generic Ack (`:7585+`). Resume im Fenster vor `effects_completed` kann `complete_*` erneut fahren (kein Writeback-Gate wie bei Business-Command).

**C. Bewusst out-of-scope (kein Attempt vorhanden)**

Panic-Drop / Lease-Cleanup / Working-Hours: plain failed/pending — korrekt kein Attempt-Marker.

**D. Report-Offenheit (ehrlich, nicht neu entdeckt)**

- Recovery-Marker ≠ TX mit In-Memory-Queue (Prozesscrash darf re-enqueuen).
- Batch-Writeback partial nicht atomar über Business-OS-Store.

**Urteil Q4:** Die **vertagten I-071-Finalisierungspfade** sind weitgehend geschlossen; „Attempt-Bindung **vollständig**“ ist **nicht** gedeckt (Failure-App-Validation-Rework + App-Validation-Success-Terminal ohne Marker).

---

## 5) Transaktionsgrenzen Hold+Zähler+Zeitstempel+Projektion+Marker

### KONFIRMIERT — eine SQLite-TX auf der channels-Seite

```text
channels/mod.rs:2616–2778 hold_leased_messages_impl
  tx = conn.transaction()
  if attempt: early-exit on queue_effects_applied_at
  … HoldReason-Zweige: route/command, failure_attempt_count++, retry_not_before, hold_reason, last_error, updated_at …
  if attempt: UPDATE worker_attempt_finalizations SET queue_effects_applied_at
  load_queue_projection_tasks + refresh_queue_projection_tasks
  tx.commit()
```

Analog `ack_leased_messages_for_attempt`: Ack + Marker + Projektion in einer TX (`:2407–2458`).

Writeback selbst ist **bewusst** 2-phasig (eigene Store-TX, dann Adoption) — getestet, keine versteckte Ein-TX-Lüge.

**Urteil Q5:** Hold-Zusicherung **hält**.

---

## 6) Tests: Zusicherung vs. Beschreibung

| Test | Beweist | Lässt unentdeckt |
|---|---|---|
| `attempt_bound_hold_and_failed_ack_do_not_rewrite_on_resume` | 2. Hold: Zähler/Timestamps gleich; 2. failed: `updated_at`/`acked_at`/`last_error` gleich | pending/review_rework Replay; Failure-App-Validation-Helper |
| `resumed_attempt_does_not_repeat_business_command_writeback_after_ack` | echtes Fenster nach Queue-handled, vor Marker; Closure 1×; 1 Transition; keine Timestamp-Rewrite | Batch partial; App-Validation-Success ohne Marker |
| `resumed_attempt_does_not_repeat_outcome_recovery_enqueue` | Marker über DB-Reopen; 2. apply false | Enqueue-dann-Crash-vor-Marker (bewusst prozesslokal) |
| CV / Preserve / Founder telemetry | 1 Row + identical second call dedupe | Founder **Outcome-Wechsel** auf gleichem `message_key`; Preserve nach echtem Re-Lease |
| Compile/Symbol-Schluss | indirekt geschlossen (Defs im Commit) | — (kein dirty-only Call) |

**Urteil Q6:** Die Kern-Zusicherungen Hold/failed/Writeback/Recovery sind **beweisend**, nicht nur beschreibend. Telemetrie- und App-Validation-Restlücken sind **test-blind**.

---

## Gesamturteil

# **nacharbeit_noetig**

Nicht `zurueckrollen`:  
Montage (23+3) ist symbolgeschlossen, I-071-Fehlerklassen (metadata-Drive-by, geopferte `fail_business_command`-Naht) greifen **nicht**, Hold-TX ist ehrlich, die namentlich vertagten Attempt-Pfade und Resume-Gates für Writeback/Witness sind die richtige Architektur und größtenteils belegt.

Nicht `landung_haelt`, weil:

1. **Founder-Telemetrie-Key zu grob** — `repair:repair_stalled_founder_communications:<digest(message_key)>` verschluckt echte Zweit-Repairs mit anderem `repair_outcome` (Under-Reporting, Claim „je Repair ein Datensatz“ falsch).
2. **Attempt-Bindung nicht „vollständig“** — Failure-Pfad `apply_business_os_app_validation_rework_to_leased_queue` bleibt plain `ack_leased_messages(..., "review_rework")` ohne Marker; App-Validation-Success terminalisiert ohne `queue_effects_applied_at`.
3. Tests decken genau die gelandeten Happy-Paths, nicht die beiden Defekte oben.

### Minimal-Nacharbeit

1. Founder-Idempotenz: `context_key` um `repair_outcome` (oder getrennte `repair_kind`s) erweitern; Beweistest mit Sequenz zweier Outcomes auf derselben Message.
2. `apply_business_os_app_validation_rework_to_leased_queue` attempt-binden **oder** Failure-Pfad wie Success nur Feedback setzen und Finalisierungs-`ack_*_for_attempt` nutzen.
3. App-Validation-Success: nach `complete_*` `mark_worker_attempt_queue_effects_applied_if_status(..., "handled")` (analog Writeback-Adoption).
4. Optional: Clean-tree `cargo check` auf `4b5aca6ee` allein (Symbole sehen geschlossen aus; nicht in diesem Review ausgeführt — Build-Verbot).

---

## Workjet-Completion-Receipt v1

```yaml
workjet_completion_receipt: v1
role: independent_adversarial_reviewer
work_id: I-072
commit: 4b5aca6ee0616700269d5f5e5319e77bdd2d810a
mode: read_only_commit_based
verdict: nacharbeit_noetig
questions:
  1_montage: ENTKRAEFTET  # no dirty-only symbol; 23+3 hunks; fail_business_command seam kept; hold body preserved in impl
  2_drive_by: ENTKRAEFTET  # metadata absent; whitelist thematic only
  3_telemetry: KONFIRMIERT_partial  # CV ok; founder key too coarse swallows distinct repair_outcome; preserve coarse but action-safe
  4_attempt_binding: KONFIRMIERT_partial  # I-071 named finalization paths mostly bound; app_validation failure rework + success terminal unbound
  5_transactions: ENTKRAEFTET  # hold+counters+timestamps+projection+marker one TX; ack_for_attempt one TX
  6_tests: KONFIRMIERT_gaps  # prove hold/failed/writeback/recovery; miss founder multi-outcome and app_validation rework resume
blocking_findings:
  - path: src/core/service/service.rs
    line: 1104-1138
    class: telemetry_key_too_coarse
    detail: founder repairs share repair:repair_stalled_founder_communications:<digest(message_key)>; distinct repair_outcome values dedupe incorrectly
  - path: src/core/service/service.rs
    line: 11391
    class: incomplete_attempt_binding
    detail: apply_business_os_app_validation_rework_to_leased_queue plain ack review_rework; failure path skips for_attempt block
non_blocking_findings:
  - app_validation terminal success completes queue without queue_effects_applied_at adoption
  - recovery_effects_applied_at not co-transactional with in-memory enqueue (report-disclosed)
  - batch writeback partial replay (report-disclosed)
  - panic/cleanup/working-hours plain acks remain (no attempt context)
acceptance_claims_rejected:
  - "Attempt-Bindung vollstaendig"
  - "je Repair ein dauerhafter Datensatz" for founder multi-outcome same message_key
acceptance_claims_accepted:
  - montage service 23 + channels 3 hunks
  - metadata-Drive-by draussen / Symbol-Schluss der montierten Fassung (for introduced symbols)
  - Hold+Zaehler+Zeitstempel+Projektion+Marker in einer Transaktion
  - writeback/witness resume gates for the designed windows
  - I-071b fail_business_command repair seam retained
i071_deferred_findings:
  hold_pending_failed_cv: largely_fixed
  business_writeback_resume: fixed_and_tested
  witness_enqueue_resume: fixed_and_tested_process_local
  metadata_drive_by: fixed_absent
  repair_seam_fail_business_command: retained
next_required: founder telemetry key include outcome; bind or re-route failure app_validation rework; optional mark_if_status after app_validation success
git_write: false
build_test_run: false
elapsed_bound: under_45m
```
