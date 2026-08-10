# Adversariales Review — I-071 (`5a1dc3d35`)

Read-only, commit-basiert (`git show 5a1dc3d35:<…>`), nicht der dirty Arbeitsbaum.  
Messgrundlage: `docs/dev/beweise/i-066-frisch.md`, Plan `docs/ctox-sync-plan-2026-08-10.md` §S2a, Worker-Report `docs/dev/beweise/plan-v2-i071-report.md`.

Commit-Umfang (5 Dateien):  
`lcm/mod.rs`, `lcm/tests.rs`, `turn_loop.rs`, `channels/mod.rs`, `service.rs` — **27 service-Hunks + 1 channels-Hunk**, wie behauptet.

---

## 1) Drive-by-Check

### KONFIRMIERT — metadata-Drive-by ist im Commit gelandet

Commit-Message behauptet:

> „die fremden 245 Hunks und der **metadata-Drive-by bleiben unversioniert**“  
> „**Symbol-Schluss** der montierten Fassung geprueft“

Beides ist falsch.

Im Commit:

```text
service.rs:10840
if business_os_app_module_target_from_metadata(&job.queue_task_metadata).is_some() {
```

Parent hatte hier `business_os_app_module_target_from_prompt(&job.prompt)`.

**Kleinster entscheidender Verifikationsschritt**

```bash
git grep -n 'business_os_app_module_target_from_metadata' 5a1dc3d35 -- '*.rs'
# → genau 1 Treffer: service.rs:10840 (Aufruf)

git grep -n 'fn business_os_app_module_target_from_metadata' 5a1dc3d35 -- '*.rs'
# → 0 Treffer

git grep -n 'fn business_os_app_module_target_from_metadata' 5a1dc3d35^ -- '*.rs'
# → 0 Treffer
```

Das Symbol existiert im dirty Baum (`service.rs` ~10937), aber **nicht** im commit-Tree.  
Thematisch gehört der Prompt→Metadata-Umstieg **nicht** zu Attempt-Finalisierung; er ist der genau in der Message benannte, angeblich ausgeschlossene Drive-by — nur halb eingebaut.

### KONFIRMIERT — Repair-Kollateralschaden im selben Hunk-Block

Beim Umbau des Success-Queue-Zweigs wurde der Parent-Block gelöscht:

```rust
// Parent, nach terminal failed-ack:
if let Some(reason) = terminal_review_failure_reason.as_deref() {
    for message_key in &job.leased_message_keys {
        crate::business_os::store::fail_business_command_from_queue_error(...)
    }
}
```

Im Commit (`service.rs:7523–7587`) bleibt `terminal_review_failure_reason` gesetzt (`:7563`), wird aber **nie** konsumiert.  
Das ist kein „fremder“ Hunk, aber dieselbe Fehlerklasse wie I-070: Hunk-Extraktion zerstört eine bewusste Repair-Naht. Kommentar im Parent nannte den Defekt explizit („Business OS command stayed `accepted` forever“).

### ENTKRÄFTET für die restliche Hunk-Masse

- `channels/mod.rs` (1 Hunk): nur `ack_leased_messages_for_attempt` — thematisch I-071.
- `lcm/*`, `turn_loop.rs`: Attempt-Typen, Migration, typed Success, Beweistests — thematisch I-071.
- Die übrigen ~25 service-Hunks sind Attempt-Lifecycle, Reihenfolge Review/Reset, Timeout-vor-Kind, Tests — thematisch I-071.

**Urteil Q1:** Drive-by **nicht** sauber vermieden (1 Call-Site metadata + Löschung des `fail_business_command`-Repair).

---

## 2) Konsistenz der Montage

### KONFIRMIERT — I-070-Fehlermuster (fremdes Symbol nur im dirty Baum)

| Symbol | Definiert in Commit? | Aufgerufen in Commit? |
|---|---|---|
| `WorkerAttemptContext` / `worker_attempt` | ja (`turn_loop.rs:37–55`) | ja |
| `run_begin/ensure/recoverable/terminalize/...` | ja (`lcm/mod.rs`) | ja |
| `ack_leased_messages_for_attempt` | ja (`channels/mod.rs:2407`) | ja |
| `finalize_timeout_attempt_then_enqueue` | ja (`service.rs:25097`) | ja |
| `recover_mission_after_accepted_attempt` | ja (`service.rs:25642`) | ja |
| **`business_os_app_module_target_from_metadata`** | **nein** | **ja (`service.rs:10840`)** |

**Kleinster entscheidender Check:** Clean checkout von `5a1dc3d35` allein → `cargo check --bin ctox` muss an unresolved name `business_os_app_module_target_from_metadata` scheitern.  
Der Report-Claim „cargo check grün“ gilt nur für den **dirty** Baum, in dem die Metadata-Migration unvollständig mit-existiert.

Weitere Call-Sites in denselben 5 Dateien sind symbol-geschlossen. Externe Aufrufer außerhalb des Commits rufen die neuen LCM-`run_*`-APIs nicht; der einzige gebrochene externe Schluss ist der unvollständige Drive-by *innerhalb* von `service.rs`.

**Urteil Q2:** Montage **gebrochen** — gleiches Muster wie I-070.

---

## 3) Atomarität ehrlich? finalizing→terminal

### TEILWEISE KONFIRMIERT — gute Kernmarker, unvollständige Wirkungs-Idempotenz

**Was hält:**

1. `begin_worker_attempt_finalization` in Immediate-TX mit Attempt-ID- und work_key-Probe + partieller Unique-Index (`lcm/mod.rs:1379–1450`, Schema `:1238–1240`).
2. Reply einmalig via `reply_message_id` in derselben TX-Bindung (`ensure_worker_attempt_assistant_message`, `lcm/mod.rs:1484+`).
3. Normale handled/cancelled Success-Acks attempt-gebunden + `queue_effects_applied_at` in **einer** TX (`channels/mod.rs:2407–2458`; Consumer `service.rs:7471`, `:9760`).
4. Timeout: Artifact-Check → `timed_out` + `resumable` + `effects_completed=false` **vor** Kindtask (`finalize_timeout_attempt_then_enqueue`, `service.rs:25097–25114`); Kindtask-Dedupe via `existing_timeout_continuation` (`:25184+`).
5. Finalisierungsfehler im Worker-Thread → kein Panic-Cleanup / kein erfundener terminaler Queue-Fail (`service.rs:8608+`).

**Wo Resume nicht ehrlich idempotent ist (Code-Reihenfolge, nicht Report):**

| Crash-Fenster | Zustand | Wiederaufnahme-Effekt |
|---|---|---|
| nach `finalizing` + Reply, vor terminal | `effects_completed=0` | Modellaufruf entfällt ✓; **gesamter** Finalisierungsarm läuft erneut |
| nach `ack_leased_messages_for_attempt` (Marker gesetzt), vor `mark/terminalize effects` | Ack idempotent ✓ | aber `complete_business_command_from_queue_reply` (`service.rs` ~7400+) **ohne** Attempt-Marker erneut |
| nach Witness-Reject + `outcome_recovery_prompt` enqueued, vor `effects_completed=1` | Attempt noch recoverable | erneutes `enqueue_prompt` für Witness-Recovery möglich (keine Attempt-gebundene Dedupe) |
| Success-Zweig Hold/pending/failed | `hold_leased_messages` / `ack_leased_messages` / `ack_…_with_failure_reason` **ohne** Attempt (`:7523–7579`) | zweiter Hold/Ack kann Status/Timestamps erneut schreiben |
| Failure-Zweig CV-Parser-Recovery | plain `ack_leased_messages(..., "handled")` (`:8044`) | kein `queue_effects_applied_at` |
| Timeout-Kind | größtenteils idempotent | Doppel-Kindtask durch `existing_timeout_continuation` entkräftet ✓ |

Doppelte **Reply**: entkräftet (Message-ID-Bindung).  
Doppelter **Ack** auf dem normalen handled-Pfad: entkräftet (Attempt-Marker).  
Doppelter **Kindtask**: entkräftet (existierende Open-Task-Suche).  
Doppelte **Business-OS-Writebacks / Recovery-Prompts / Hold-Acks**: **nicht** entkräftet.

**Urteil Q3:** Logische Atomarität für Reply+normal-Ack ja; für den vollen Wirkungsraum **nein** — Resume re-exekutiert nicht-attempt-gebundene Seiteneffekte.

---

## 4) Reihenfolge: Failure-Zähler vor Review/Zeuge?

### ENTKRÄFTET für den Success-Reset-Pfad (Kernziel von I-071)

- Früherer Success-Reset im Pre-Review-Zweig ist entfernt (`service.rs:6529–6533`: „Success recovery is intentionally not performed here“).
- Reset nur noch über `recover_mission_after_accepted_attempt` **nach** Artifact-/Witness-/Review-Auswertung (`:7333–7365`, Helper `:25642–25654`).
- Transient/`ExecutionError`: `retryable_runtime_failure` verhindert Increment; ohne Success-Fallthrough kein Reset (`:6473–6533`).

### ENTKRÄFTET für cancelled / no-send / mail als „Reset-vor-Review“

Kein Pfad setzt den Failure-Zähler vor Review zurück. No-Send/cancelled laufen im Success-Arm **nach** Review-Disposition; Reset nur bei `accepted_success` (Approved|None|NoSend **und** kein Witness-/Send-/App-Fail).

### REST — nicht „Reset vor Review“, aber verwandt

Bei Agent-Failure wird der Counter **weiter vor** dem späteren Lease-Handling inkrementiert (`:6484+`). Das ist beabsichtigtes Failure-Bump, nicht der i-066-Success-Reset-Bug.

**Urteil Q4:** Kern-Defekt „Reset vor Review/Zeuge“ ist **behoben**.

---

## 5) Netze intakt?

### TEILWEISE KONFIRMIERT — nicht abgeschafft, aber einer geschwächt

| Netz | Status |
|---|---|
| Outcome-Witness-Recovery | bleibt; Enqueue weiter im Success-Arm vor Attempt-Terminalisierung. Nicht abgeschwächt, aber bei Resume ggf. mehrfach (s. Q3). |
| Timeout-Kindtask | **nachgeordnet**: erst `timed_out`+artifact, dann `maybe_enqueue_*` (`:25097–25114`, Call-Site nach `begin`+Reply). Nicht abgeschwächt; Dedupe erhalten. |
| Agent-Failure-Recovery | Defer→Recover symmetrischer (`lcm/mod.rs:2542–2551`: status/is_open/allow_idle). Reset-Zeitpunkt nach Witness. **Geschwächt** ist der benachbarte Repair: `fail_business_command_from_queue_error` nach terminalem Review-Fail im Success-Zweig **entfernt** (s. Q1). Failure-Pfad behält die Projektion noch (`:7996`, `:8101`). |

**Urteil Q5:** Die drei i-066-Netze sind nicht entfernt; Timeout ist korrekt nachgeordnet. Eine konkrete, schon einmal bewusst eingebaute Repair-Naht (Business-OS-Command bei terminal review failure) wurde im Success-Zweig geopfert.

---

## 6) Migration `worker_attempt_finalizations`

### ENTKRÄFTET — einmal pro DB

```text
lcm/mod.rs:1200–1255
MIGRATION_ID = "i-071-worker-attempt-finalization-v1"
IMMEDIATE TX → EXISTS(lcm_data_migrations) → CREATE TABLE/INDEX → INSERT marker → commit
```

Zweiter Open: Early-Exit, kein zweites Schema-Schreiben.

### ENTKRÄFTET — partieller Unique-Index vs. parallele Jobs

```sql
CREATE UNIQUE INDEX ... ON worker_attempt_finalizations(work_key)
WHERE effects_completed = 0;
```

`work_key` = Digest über **konkrete** Lease-Identität (`service.rs:1015–1044`: sorted message keys / ticket events / internal work / outbound / ephemeral).  
Nicht „ein Attempt je Conversation“. Zwei parallele Jobs derselben Conversation mit **verschiedenen** Message-Keys → verschiedene `work_key` → erlaubt.  
Zwei Writer für **dieselbe** logische Arbeit → einer gewinnt, `begin_*` gibt die autoritative Zeile zurück (`lcm/mod.rs:1410–1425`).

**Urteil Q6:** Migration und Index-Semantik sind stimmig.

---

## 7) Tests: Welcher Defekt bliebe von den fünf Beweistests unentdeckt?

Belegt im Commit:

1. `worker_attempt_success_persists_one_marker_and_typed_reply` (`lcm/tests.rs`)
2. `worker_attempt_finalizing_crash_resumes_without_duplicate_reply` (`lcm/tests.rs`)
3. `successful_turn_persists_one_attempt_and_typed_success_reply` (`turn_loop.rs`)
4. integrierter Ack-Replay ohne zweites `acked_at` (erweitert `queue_ack…`, `service.rs:36652+`)
5. `timeout … terminal_at <= child.observed_at` (`service.rs:33204+`)
6. Defer→Recover-Felder (`mission_failure…` erweitert)
7. `rejected_outcome_witness_does_not_reset_failure_counter` (`service.rs:45500+`)

**Unentdeckt / nicht abgedeckt:**

| Defekt | Warum die 5/7 Beweise schweigen |
|---|---|
| **Fehlendes Symbol `…_from_metadata`** | Kein Compile des reinen Commit-Trees; Tests liefen auf dirty Tree |
| **Gelöschtes `fail_business_command_from_queue_error` im Success-terminal-fail-Zweig** | Kein Test „terminal review fail → command not stuck in accepted“ |
| Resume nach Ack+Writeback, vor `effects_completed` | Crash-Test deckt Reply+Ack, **nicht** `complete_business_command_*` / Witness-Enqueue |
| Hold/pending/failed ohne Attempt-Ack | Nur handled-for-attempt getestet |
| CV-Parser plain `ack_leased_messages` | ungetestet |
| Parallele distinct `work_key` / Collision same key | ungetestet |
| Report „cargo check grün auf Commit“ | Faktisch Arbeitsbaum, nicht `5a1dc3d35` alone |

---

## Gesamturteil

# **nacharbeit_noetig**

Nicht `zurueckrollen`: der Attempt-Datensatz, typed Success, Review-vor-Reset, Timeout-vor-Kind und attempt-gebundener Normal-Ack sind die richtige Architektur und größtenteils ehrlich implementiert.

Nicht `landung_haelt`, weil **dieselben Fehlerklassen wie I-070** erneut greifen:

1. **Montagebruch** — Call auf `business_os_app_module_target_from_metadata` ohne Definition im Commit (`service.rs:10840`).
2. **Drive-by** — genau der angeblich ausgeschlossene metadata-Hunk-Anteil.
3. **Repair-Lücke** — `fail_business_command_from_queue_error` nach terminalem Review-Fail im Success-Zweig entfernt.
4. **Unvollständige Attempt-Bindung** auf Hold/pending/failed/CV-Recovery und nicht-idempotente Resume-Seiteneffekte (Business-OS-Writeback, Witness-Recovery-Prompt).

### Minimal-Nacharbeit (ohne Scope-Creep)

1. `service.rs:10840` zurück auf `business_os_app_module_target_from_prompt` **oder** die Definition aus dem dirty Baum mit-commiten — beides ist I-071-fremd; Prefer Revert.
2. Parent-Block `fail_business_command_from_queue_error` nach terminal review fail wieder einsetzen.
3. Clean-tree `cargo check` auf dem montierten Commit, nicht dem dirty Tree.
4. Optional aber ehrlich: Hold/failed-Acks attempt-binden; Resume-Gate für Writeback/Witness-Enqueue.

---

## Workjet-Completion-Receipt v1

```yaml
workjet_completion_receipt: v1
role: independent_adversarial_reviewer
work_id: I-071
commit: 5a1dc3d35f54ff19de0df343f23eadd44ba4ec92
mode: read_only_commit_based
verdict: nacharbeit_noetig
questions:
  1_drive_by: KONFIRMIERT  # metadata call @ service.rs:10840; fail_business_command repair deleted
  2_montage: KONFIRMIERT   # business_os_app_module_target_from_metadata undefined in commit tree
  3_atomicity: TEILWEISE_KONFIRMIERT  # reply+normal-ack yes; writeback/hold/witness-resume no
  4_counter_order: ENTKRAEFTET  # reset only after accepted review/witness
  5_safety_nets: TEILWEISE_KONFIRMIERT  # nets kept; success-path command fail projection removed
  6_migration_index: ENTKRAEFTET  # once-per-DB; unique per open work_key not conversation
  7_test_gaps: KONFIRMIERT  # compile break, command-fail deletion, incomplete resume effects
blocking_findings:
  - path: src/core/service/service.rs
    line: 10840
    class: montage_drive_by
    detail: calls business_os_app_module_target_from_metadata; fn absent in 5a1dc3d35
  - path: src/core/service/service.rs
    line: 7523-7587
    class: repair_regression
    detail: terminal_review_failure_reason set but fail_business_command_from_queue_error block removed vs parent
non_blocking_findings:
  - incomplete attempt-binding on hold/pending/failed and CV recovery acks
  - resume re-runs complete_business_command and outcome recovery enqueue
acceptance_claims_rejected:
  - "metadata-Drive-by bleiben unversioniert"
  - "Symbol-Schluss der montierten Fassung geprueft"
  - "cargo check gruen" as proof for commit-only tree
safety_nets_preserved:
  outcome_witness_recovery: yes_with_resume_dup_risk
  timeout_child_task: yes_ordered_after_timed_out
  agent_failure_recovery: yes_symmetric_reset_after_accept
next_required: Grok-Re-Review after montage fix + repair restore; clean checkout compile
git_write: false
build_test_run: false
elapsed_bound: under_45m
```
