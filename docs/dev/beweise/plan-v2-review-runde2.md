Evidenz reicht: I-072b/c schließt die Blocking-Findings; I-073 hat dirty-only Test-Locks; I-074 kollabiert die drei Klassen am Protokollrand.

---

# Adversariales Sammelreview — SYNC-F Runde 2 Abschluss  
(I-072b/c, I-073, I-074)

Read-only, commit-basiert (`git show <hash>:<datei>`). Ziel: WIDERLEGEN.  
Bezug: `docs/dev/beweise/plan-v2-review-i072.md` (blocking_findings).

---

## 1) I-072b `6a2f44a17` + I-072c `b045d7994`

### Blocking-Findings aus dem I-072-Review

| Finding | Status | Nachweis |
|---|---|---|
| Founder-Idempotenz zu grob (`context_key` ohne Outcome) | **ERFÜLLT** | `record_repair_telemetry` digestet jetzt `[context_key, repair_outcome]` (`service.rs` ~1134–1142 auf `b045d7994`). Alle drei Founder-Call-Sites setzen `repair_outcome` in `details`. |
| App-Validation-Rework plain-Ack | **ERFÜLLT** | `apply_business_os_app_validation_rework_to_leased_queue(..., attempt_id: Option<&str>)`: Adoption via `mark_…_if_status(..., "review_rework")`, Worker-Pfad `ack_*_for_attempt`, Recovery `None` → plain. Worker-Call-Sites mit `Some(&attempt_id)`. |
| (non-blocking) Success ohne Marker | **ERFÜLLT** | `complete_*_success` adoptiert `handled`, setzt nach Complete `mark_…_if_status(..., "handled")`; Worker Success + after-error übergeben `Some(&attempt_id)`. |

**Kleinster Verifikation (Founder-Key):**

```text
b045d7994:service.rs ~1134–1142
idempotence_context_key = json!([context_key, repair_outcome]) | fallback context_key
```

Test `stalled_founder_email_superseded_…` beweist Sequenz: realer `superseded_*`-Repair + manuell zweites Outcome `restored_stalled_*` → `COUNT=2`, `DISTINCT outcomes=2`, identisches Replay deduped.

**App-Validation-Rework-Resume:** Test renames/erweitert zu `resumed_attempt_does_not_repeat_…`; zweiter Call `replayed == 0`, Prompt/Timestamps/Proofs unverändert, Marker gesetzt.

### I-072c Montage

- `business_os_app_module_target_from_metadata` im Commit-Tree **ohne Definition** (`git grep` auf `6a2f44a17`/`b045d7994`/`bcc72ca6d` → 0 Defs).
- I-072c revertiert den Success-Pfad zurück auf `_from_prompt` (2 Zeilen).  
- **ENTKRÄFTET als offener Defect im Finalbaum** — aber bestätigt die wiederkehrende Montage-Fehlerklasse (Audit und Commit nicht sequenziell; Commit-Message ehrlich).

### Rest (nicht blocking)

- Crash zwischen Feedback-Schreiben und Ack im Rework-Helper: Feedback/Proofs werden erneut geschrieben; Adoption greift erst nach `review_rework`. Entspricht dem bekannten 2-Phasen-Muster (Writeback), kein neuer Klassenbruch.
- Preserve-Lease-Telemetrie bleibt outcome-los grob — action-safe, wie zuvor non-blocking.

### Urteil I-072b+c: **`landung_haelt`**

Beide blocking_findings aus `plan-v2-review-i072.md` sind im finalen Tree geschlossen; Tests beweisen Outcome-Diskriminierung und Resume-Adoption; I-072c entfernt den dirty-only Metadata-Call.

---

## 2) I-073 `0d05240c3`

### Boot-Pfad

**ENTKRÄFTET — Boot bleibt vollständig und unverändert.**

`release_stale_service_communication_leases_on_boot` (Parent vs. Commit, byte-gleich im relevanten Block):

1. `recover_abandoned_business_os_app_queue_tasks(..., 16)`
2. `release_stale_queue_task_leases(..., &HashSet::new())` — **leere** active_keys → kein Schutz vor Release
3. `release_stale_service_communication_leases` + Audits `boot_queue_lease_reclaim` / `boot_lease_reclaim`

Boot-Call-Site ~1447 bleibt.

### Dedupe / active_keys-Semantik

**ENTKRÄFTET — kein verlorener Release-Fall, den nur Reconcile abdeckte.**

`active_keys` in `reconcile_ticket_runtime_state` und `run_orphaned_queue_lease_sweep` sind **identisch konstruiert**:

```text
pending_prompts.{leased_message_keys, leased_ticket_event_keys}
+ if busy: leased_message_keys_inflight
```

`release_stale_queue_task_leases` filtert beides gleich (`if active_message_keys.contains → continue`).  
Reconcile-Release war eine **doppelte Owner-Fläche** mit denselben Kandidaten, nicht eine strengere Semantik. Nach Dedupe:

| Owner | Queue-Lease-Release |
|---|---|
| Boot | ja (`HashSet::new()`) |
| 60s-Sweep | ja (active_keys-Schutz) |
| Reconcile | **nein** (entfernt) |

Reconcile behält: App-Recovery, `stuck_lease_escalation` (active_keys-geschützt), Ticket-Event-Lease-Release, Blocked-Ready-Release. Beschreibung `ticket_reconciliation` und Mechanism-Eintrag `orphaned_queue_lease_sweep` stimmen damit überein.

**Latenz-Hinweis (non-blocking):** Ohne Reconcile-Release wartet ein Orphan außerhalb des Boot-Fensters bis zu ~60s auf den Sweep. Bewusste Ownership-Verschiebung, kein Funktionsverlust.

### Sweep-Audit

- Mechanism registriert in `DEFAULT_MECHANISMS` + Inventory-Test.
- Audit nur bei `released_count > 0`; pure Failures → Feed-Event, kein Governance-Spam; Null-Sweep still. Entspricht Claim.

### Gate-Test ehrlich?

**TEILWEISE KONFIRMIERT — umbenannt, aber weiterhin nur Gate-Semantik.**

`ticket_reconcile_gate_skips_idle_state_and_reopens_for_remaining_recovery_work` prüft:

- idle skip nach `mark_ticket_reconcile_ran`
- Reopen nach `create_queue_task` (Core-DB-Änderung)

Es beweist **nicht**, dass App-Recovery/Stuck/Ticket-Event tatsächlich laufen — nur, dass das Idle-Gate wieder öffnet. Die Assertions-Message verkauft „remaining recovery work“; der Test bleibt beschreibend auf Gate-Ebene.  
Eigentliche Ownership-Beweise liegen in den neuen Tests:

- `orphaned_queue_lease_sweep_audits_one_real_release_and_noop_stays_silent`
- `ticket_reconcile_leaves_stale_queue_lease_for_orphaned_sweep_owner`

### KONFIRMIERT — dirty-only Test-Lock-Statics (Montagebruch)

Im Commit-Tree `0d05240c3` (und noch `bcc72ca6d`):

```text
// Verwendung (Helper + Tests):
ticket_reconcile_gate_test_lock() → TICKET_RECONCILE_GATE_TEST_LOCK
orphaned_queue_lease_sweep_test_lock() → ORPHANED_QUEUE_LEASE_SWEEP_TEST_LOCK
// + stuck_protected_queue_lease_is_escalated_not_released nimmt ticket_reconcile_gate_test_lock
```

**Keine** `static …TEST_LOCK`-Definition im Commit-Blob:

```bash
git show 0d05240c3:src/core/service/service.rs | rg 'static (TICKET_RECONCILE|ORPHANED_QUEUE).*TEST_LOCK'
# → leer
```

Die Definitionen existieren **nur im dirty Working Tree** (`src/core/service/service.rs` ~26802–26803).  
Das ist dieselbe Fehlerklasse wie I-071/I-072c (Call ohne Definition im Commit). Clean-Checkout von `0d05240c3` → **Testmodul kompiliert nicht**.

Report-Claim „Symbol-Audit fing den fehlenden Testhelfer-Hunk VOR dem Commit“ ist damit **widerlegt** für die Static-Definitionen (Helper-Hunks landeten, die `static`-Defs nicht).

### Urteil I-073: **`nacharbeit_noetig`**

Nicht zurückrollen: Ownership-Dedupe und Boot sind inhaltlich korrekt; Beweistests zur Ownership sind ehrlich.  
Nacharbeit: die beiden `static …TEST_LOCK` in denselben Commit-Tree bringen (I-073b), analog I-072c.

---

## 3) I-074 `bcc72ca6d`

### Montage

| Frage | Befund |
|---|---|
| dirty-only-Symbole? | Harness-Varianten + `TurnRuntimeError*` im Commit definiert und verdrahtet. **ENTKRÄFTET** für Produktions-Symbole. |
| rustls-`Cargo.toml`-Hunk draußen? | Commit enthält kein `harness/Cargo.toml` — korrekt; ResponseIncomplete braucht rustls nicht. **ENTKRÄFTET** „fehlt etwas“. |
| FORK.md | 7 Harness-Deltas dokumentiert, Dateiliste deckungsgleich. **ENTKRÄFTET**. |
| Drive-by | **KONFIRMIERT (klein):** `cv_print_filename_from_prompt` ist byte-äquivalent zu `prompt_line_value(prompt, "- filename:")` — unnötige Refactor-Naht im CV-Hunk, thematisch nicht typisiertes Gate. |

I-073s fehlende TEST_LOCK-Statics bleiben im Tree von I-074 unbehoben (Querschnitt-Montage, nicht I-074-eigen).

### Display-Byte-Gleichheit

**ENTKRÄFTET als Bruch — hält für den Incomplete-Pfad.**

| Schicht | Display |
|---|---|
| `ApiError::ResponseIncomplete` | `stream error: {message}` (wie `Stream`) |
| `CodexErr::ResponseIncomplete` / `Stream` | `stream disconnected before completion: …` |
| Tests | `incomplete_*_preserves_typed_reason_and_display`, bridge + core error_tests |

### Typfluss bis Service — **KONFIRMIERT: Klassenkollaps + tote Varianten**

Produktionspfad:

```text
response.incomplete
  → ResponseIncompleteReason::{MaxOutputTokens|Other}
  → ApiError::ResponseIncomplete
  → CodexErr::ResponseIncomplete
  → CodexErrorInfo::ResponseStreamDisconnected   // reason verworfen
  → TurnRuntimeErrorClass::from_codex_error_info
       matches nur ResponseStreamDisconnected
       → ALWAYS StreamDisconnected
  → Gate matches OutputTokenLimit|IncompleteResponse|StreamDisconnected
```

`from_codex_error_info` (`direct_session.rs` ~554–559):

```rust
matches!(error_info, CodexErrorInfo::ResponseStreamDisconnected { .. })
    .then_some(Self::StreamDisconnected)
```

**`OutputTokenLimit` und `IncompleteResponse` werden im Produktionscode nirgends konstruiert.**  
Die Service-Tests `cv_print_typed_output_token_limit_*` / `*_incomplete_response_*` basteln die Klassen manuell — sie beweisen Match-Arme, **nicht** den End-to-End-Fluss.

**Funktionale Erholung** für echte Incomplete/Stream-Fehler bleibt möglich (alles landet in `StreamDisconnected` → Gate offen). Die **Textfalle** ist zu:

```text
cv_print_user_content_substring_does_not_allow_compact_recovery  // ehrlich
```

Der Claim „drei typisierte Recovery-Klassen bis zum Service-Gate“ ist **überzogen**; Report-Text gibt den Kollaps am Protokollrand teilweise zu, die Enum-/Test-Oberfläche suggeriert das Gegenteil.

### NICHT-CV-Konsument: `Other` → `ResponseStreamDisconnected`

**KONFIRMIERT — Mapping ändert sich; Schaden unbewiesen, aber real am Protokollrand.**

Parent: `CodexErr::Stream` fiel in `_ => Other`.  
I-074: `Stream | ResponseIncomplete → ResponseStreamDisconnected`.

| Konsument | Wirkung |
|---|---|
| `CodexErrorInfo::affects_turn_status` | Other **und** ResponseStreamDisconnected → `true` (keine Status-Regression) |
| `direct_session` CV/typed path | neu: mapped ResponseStreamDisconnected → recovery class |
| app-server | leitet `codex_error_info` durch; Clients sehen anderen Enum-Wert |
| `runtime_error_is_transient_api_failure` | **unverändert textbasiert** (bewusst vertagt) |

Kein Service-Branch, der `Other` für Stream-Disconnects **positiv** voraussetzte und jetzt bricht — aber jedes externe UI/Telemetry, das `Other` vs. `ResponseStreamDisconnected` unterscheidet, sieht eine Verhaltensänderung für **alle** Stream-Disconnects, nicht nur CV.

### Gate vs. Timeout (Querschnitt)

- Timeout: `AgentOutcome::TurnTimeout` + `timeout_auto_retry_enabled` — parallel, text-/outcome-basiert.
- CV-Gate: `turn_runtime_error_class(&err)` — typisiert, nur CV-Jobs.
- Kein Widerspruch zu I-071-Attempt / I-072-Bindung / I-073-Sweep-Ownership.  
  CV-Recovery-Ack bleibt attempt-gebunden (I-072-Pfad unangetastet).

### Urteil I-074: **`nacharbeit_noetig`**

Nicht zurückrollen: Textfalle geschlossen, Display hält, Fork dokumentiert, Recovery für reale Stream/Incomplete-Ereignisse greift über `StreamDisconnected`.  
Nacharbeit:

1. Entweder `ResponseIncompleteReason` bis `TurnRuntimeErrorClass` durchreichen (`MaxOutputTokens→OutputTokenLimit`, sonst `IncompleteResponse`) **oder** die toten Varianten + Schein-Tests entfernen und den Kollaps ehrlich als eine Klasse führen.
2. Drive-by `cv_print_filename_from_prompt` zurück auf `prompt_line_value` (optional, sauber).
3. Optional: Protokoll-Consumer-Notiz, dass Stream nicht mehr als `Other` ankommt.

---

## 4) Querschnitt fünf Wellen (I-071…I-074)

| Naht | Widerspruch? |
|---|---|
| I-071 Attempt-Transition | intakt; I-072b bindet App-Validation daran |
| I-072 Founder-Telemetry / Binding | nach b/c ehrlich genug |
| I-073 Sweep-Ownership | orthogonal; Boot ≠ Sweep ≠ Reconcile klar |
| I-074 CV-Gate | orthogonal zu Timeout-Retry; öffnet nicht fälschlich per Content-Substring |
| Montage-Disziplin | **wiederholt gebrochen**: I-072b metadata-Call, I-073 Test-Lock-Statics — Pattern, nicht Einzelunfall |

Kein semantischer Konflikt Timeout-Klassifikation vs. CV-Gate.  
Das wiederkehrende Dirty-Tree-Montageproblem ist der querliegende Systembefund.

---

## Gesamturteile

| Commit | Urteil |
|---|---|
| `6a2f44a17` + `b045d7994` (I-072b/c) | **`landung_haelt`** |
| `0d05240c3` (I-073) | **`nacharbeit_noetig`** |
| `bcc72ca6d` (I-074) | **`nacharbeit_noetig`** |

Runde-2-Abschluss als Ganzes: **nicht freigeben**, bis I-073-Statics und I-074-Klassenwahrheit nachgezogen sind. Kein Zurückrollen einzelner Wellen.

---

## Workjet-Completion-Receipt v1

```yaml
workjet_completion_receipt: v1
role: independent_adversarial_reviewer
work_id: SYNC-F-R2-close-I072b-I073-I074
mode: read_only_commit_based
git_write: false
build_test_run: false
elapsed_bound: under_60m
subjects:
  - commit: 6a2f44a17fa4f65d1f0ee262ce780ab42d677ca5
    work_id: I-072b
    verdict: landung_haelt
  - commit: b045d7994488b49ea4e5db45f536c87fa1e8e267
    work_id: I-072c
    verdict: landung_haelt
  - commit: 0d05240c36d1d84a7ceae200c9dbe6652cf61ff7
    work_id: I-073
    verdict: nacharbeit_noetig
  - commit: bcc72ca6d0d02b7900de8a4a18d3b4b1389b52b4
    work_id: I-074
    verdict: nacharbeit_noetig
i072_review_blocking_findings:
  founder_telemetry_key_too_coarse:
    status: ERFUELLT
    evidence: "record_repair_telemetry digests [context_key, repair_outcome]; founder call sites set repair_outcome; multi-outcome test COUNT=2"
  app_validation_rework_unbound:
    status: ERFUELLT
    evidence: "apply_*_rework takes attempt_id; worker uses ack_for_attempt + mark_if_status review_rework; resume test"
  app_validation_success_marker_non_blocking:
    status: ERFUELLT
    evidence: "complete_* binds handled via mark_if_status after complete; both worker paths pass Some(attempt_id)"
  metadata_dirty_call:
    status: ERFUELLT_by_I072c
    evidence: "b045d7994 reverts to business_os_app_module_target_from_prompt; from_metadata undefined in commit trees"
findings:
  - object: I-072b/c
    status: ENTKRAEFTET
    class: prior_blocking_open
    detail: both plan-v2-review-i072 blocking_findings closed on final tree
  - object: I-073
    status: ENTKRAEFTET
    class: boot_path_regression
    detail: release_stale_service_communication_leases_on_boot unchanged; empty active_keys full reclaim
  - object: I-073
    status: ENTKRAEFTET
    class: dedupe_loses_reconcile_only_case
    detail: reconcile and sweep build identical active_keys; same release filter; ownership move not semantic hole
  - object: I-073
    status: KONFIRMIERT
    class: dirty_only_symbol
    path: src/core/service/service.rs
    line: "26907-26915 (use); static defs absent in 0d05240c3 and bcc72ca6d blobs; only dirty WT ~26802-26803"
    detail: TICKET_RECONCILE_GATE_TEST_LOCK and ORPHANED_QUEUE_LEASE_SWEEP_TEST_LOCK referenced without definition in commit tree; clean checkout test compile break
    smallest_check: "git show 0d05240c3:src/core/service/service.rs | rg 'static (TICKET_RECONCILE|ORPHANED_QUEUE).*TEST_LOCK' → empty"
  - object: I-073
    status: KONFIRMIERT_partial
    class: gate_test_descriptive
    detail: renamed gate test only asserts idle skip/reopen via DB mtime; ownership proved by separate tests
  - object: I-074
    status: KONFIRMIERT
    class: typed_class_collapse_dead_variants
    path: src/core/execution/agent/direct_session.rs
    line: "554-559"
    detail: from_codex_error_info only yields StreamDisconnected; OutputTokenLimit and IncompleteResponse never produced in production; ResponseIncompleteReason discarded at CodexErrorInfo projection
    smallest_check: "git grep OutputTokenLimit bcc72ca6d -- '*.rs' → enum def + synthetic tests only"
  - object: I-074
    status: ENTKRAEFTET
    class: display_byte_break
    detail: Api/Core Display strings preserved for incomplete/stream paths; unit tests assert equality
  - object: I-074
    status: KONFIRMIERT_partial
    class: other_consumer_shift
    detail: CodexErr::Stream was Other, now ResponseStreamDisconnected; affects_turn_status both true; external clients see new code for all stream disconnects
  - object: I-074
    status: KONFIRMIERT
    class: drive_by
    path: src/core/service/service.rs
    line: "10209-10218"
    detail: cv_print_filename_from_prompt duplicates prompt_line_value(\"- filename:\")
  - object: I-074
    status: ENTKRAEFTET
    class: missing_rustls_cargo_hunk
    detail: Cargo.toml rustls not required for ResponseIncomplete compile path; correctly left out
  - object: cross_wave
    status: ENTKRAEFTET
    class: timeout_vs_cv_gate_contradiction
    detail: TurnTimeout path separate from turn_runtime_error_class CV gate; attempt binding and sweep ownership untouched
acceptance_claims:
  accepted:
    - I-072b founder outcome-discriminated telemetry + app_validation attempt binding
    - I-072c metadata dirty-call revert
    - I-073 boot path complete; queue-lease sole runtime owner = orphan sweep outside boot
    - I-074 substring CV trap closed; Display preserved; FORK.md lists harness deltas
  rejected:
    - I-073 symbol-audit-before-commit fully closed (TEST_LOCK statics missing in commit)
    - I-074 three distinct typed recovery classes reach the service gate in production
next_required:
  - I-073b: add static TICKET_RECONCILE_GATE_TEST_LOCK and ORPHANED_QUEUE_LEASE_SWEEP_TEST_LOCK to committed service.rs tests module
  - I-074b: either thread ResponseIncompleteReason into TurnRuntimeErrorClass or delete dead variants/tests and document single StreamDisconnected class
  - optional: drop cv_print_filename_from_prompt drive-by
aggregate_round_verdict: nacharbeit_noetig
```
