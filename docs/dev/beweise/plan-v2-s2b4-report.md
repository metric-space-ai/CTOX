## Ergebnis

Das 10.000er-Tor ist erreicht:

- `rxdb_peer.rs`: **10.165 → 9.941 Produktionszeilen**
- neues `rxdb_peer_commands.rs`: **251 Produktionszeilen**
- **8 Funktionen** als reiner Move ausgelagert
- keine Dateien außerhalb der Whitelist verändert
- nichts gestaged oder committed

## Schnittkarte

### Bewegt nach `rxdb_peer_commands.rs`

**Outbox-Verarbeitung**

1. `deliver_business_command_outbox_background_loop`

**Command-Enqueue und mechanischer Schreibpfad**

2. `enqueue_business_command_document_with_database`
3. `incremental_upsert_document_with_envelope`
4. `command_id_from_document`
5. `typed_app_action_error_code`

**Command-Result-Projektoren**

6. `project_support_command_result`
7. `project_threads_command_result`
8. `project_appsec_command_result`

Dafür wurden nur die nötigen Peer-Grenzen auf `pub(super)` erweitert:

- drei Command-Poll-Konstanten
- `projection_sleep_secs`
- `upsert_business_record_projection`

### Bewusst im Peer verblieben

| Bereich | Beispiele | Grund |
|---|---|---|
| Consumer-Loop | `consume_business_commands_loop` | Scheduling und Loop-Zustand ausdrücklich ausgeschlossen |
| Idle/Wake | `business_command_poll_sleep_secs`, `wait_for_business_command_wake` | Poll-Intervalle, Table-Wake und Idle-Backoff bleiben unangetastet |
| Candidate-/Source-Semantik | `BUSINESS_COMMAND_RETRY_CANDIDATE_SQL`, `business_commands_source_*`, `pending_business_command_documents*` | Statusauswahl und Wake-Stamp sind mit Retry-/Attempt-Zuständen gekoppelt |
| I-071/I-072-Attempts | `consume_pending_business_commands`, `BUSINESS_COMMAND_ACCEPT_RETRY_BUDGET`, `accept_pending_business_command` | Failure-Budget, persistierte Attempts und Exhaustion bleiben vollständig im Peer |
| Retry/Lifecycle | `is_transient_business_command_store_error`, `transient_business_command_retry_document`, `enrich_native_command_lifecycle` | Enthält Retry-Attempts beziehungsweise v2-Lifecycle-/Attempt-Semantik |
| Breite Projektion | übrige `sync_*`- und Record-Projektoren | Kein notwendiger Bestandteil dieser kleinen, sicheren Naht |

Damit wurde bewusst nur der klar bewegliche Teil der erwarteten ungefähr 31 Funktionen gezogen.

## Move-Nachweise

### Namens-Schnittmenge

- `rxdb_peer.rs`: **476** eindeutige Funktionsnamen
- `rxdb_peer_commands.rs`: **8**
- Schnittmenge: **0**

### Normalisierte SHA-256

Whitespace und notwendiges `pub(super)` wurden normalisiert:

```text
Vorher: 6e9aa187fab042a43928b498e5a7b38bcf501c691636c28707ed5442114deb14
Nachher: 6e9aa187fab042a43928b498e5a7b38bcf501c691636c28707ed5442114deb14
```

Ergebnis: **identisch**.

### Dirty-only-Symbole

Für alle `fn`-, `static`- und `const`-Namen der neuen Peer/Commands-Vereinigung wurde je Symbol gegen `HEAD` geprüft:

- geprüft: **559**
- dirty-only: **0**

## Zeilenbilanz

### Physische Zeilen

- vorher Peer: **20.110**
- nachher Peer: **19.887**
- Commands-Modul: **253**
- neue Vereinigung: **20.140**
- Import-/Modul-Overhead: **+30**

### Produktionszeilen

- vorher Peer: **10.165**
- nachher Peer: **9.941**
- Commands-Modul: **251**
- neue Vereinigung: **10.192**
- Overhead: **+27**

## Budgets

```text
src/core/business_os/rxdb_peer.rs = 9941
src/core/business_os/rxdb_peer_commands.rs = 251
```

Beide Einträge stimmen exakt mit der Vertragszählung überein.

## Pflicht-Checks

### `cargo check --bin ctox`

- Exit **0**
- **grün**
- **449 bestehende Warnungen**

### Modulgrößen-Gate

Aufgabenbezogene Budgets sind exakt. Der Test scheitert ausschließlich an denselben sechs fremden OVERs:

- `office_engine.rs`: 14.598 / 13.953
- `store.rs`: 28.413 / 27.516
- `store_outbound_commands.rs`: 5.354 / 5.270
- `context/lcm/mod.rs`: 6.365 / 5.627
- `mission/channels/mod.rs`: 7.388 / 7.221
- `service/service.rs`: 26.789 / 26.237

Testzahlen:

- **0 bestanden**
- **1 fehlgeschlagen**
- **2.825 ausgefiltert**

### Command-Consumer-Filter

```text
cargo test --bin ctox rxdb_peer::tests::native_peer_consumes -- --test-threads=2
```

Ergebnis:

- **8 bestanden**
- **1 fehlgeschlagen**
- **2.817 ausgefiltert**
- Laufzeit: **563,89 s**

Verbliebener bekannter Roter:

- `native_peer_consumes_pending_business_command_written_directly_to_sqlite`
- Fehlerbild: `DOC_CACHE_REV` beim Lesen des erwarteten accepted-Dokuments

**Abweichung von der übernommenen Baseline:**  
`native_peer_consumes_pending_module_governance_commands`, historisch der zweite klassifizierte Rote, war auf diesem Arbeitsbaum grün. Damit wurde das erwartete Zwei-Rote-Set nicht vollständig reproduziert. Der Test und sämtliche Attempt-/Consumer-Semantik blieben unangetastet; außerdem war `store.rs` bereits vor Beginn stark fremd-dirty (**+1.649/-14 Zeilen**). Das spricht für Arbeitsbaum-Baseline-Drift, ist aber kein kausaler Beweis.

### `include_str!`-Guard

Der Idle-Guard liest nun zusätzlich `rxdb_peer_commands.rs`.

- **1 bestanden**
- **0 fehlgeschlagen**
- **2.825 ausgefiltert**

### Weitere Checks

- Rustfmt-Check der berührten Rust-Dateien: **grün**
- `git diff --check`: **grün**
- Peer-Vollsuite nicht neu gemessen; Baseline **143/3** übernommen
- Ressourcen: initialer 5-Minuten-Load **7,92**, final **22,30** und damit unter der `>30`-Warteauslösung
- `/Volumes/tmp` final frei: **338.918.392 KiB**
- Fortschrittsdatei: `/Volumes/tmp/ctox-pipeline/s2b4-fortschritt.md`

## Offene Bedenken

1. Der zweite historisch rote Command-Test war wegen Arbeitsbaum-Drift grün; damit ist die geforderte identische Zwei-Rote-Baseline formal nicht vollständig bestätigt.
2. Das globale Größen-Gate bleibt wegen sechs außerhalb der Whitelist liegender OVERs rot.
3. Das neue Modul greift bewusst über `pub(super)` auf einige Peer-Helfer zurück; dies ist eine Extraktionsnaht, keine Architektur-Neuordnung.
4. `rxdb_peer_commands.rs` bleibt wie verlangt untracked. Keine Task-Datei ist gestaged.

```text
WORKJET-COMPLETION-RECEIPT v1
task: S2b-4
outcome: completed_under_10000_with_dirty_worktree_command_baseline_mismatch
moved_functions: 8
name_intersection: 0
dirty_only_fn_static_const: 0_of_559
normalized_move_sha256: 6e9aa187fab042a43928b498e5a7b38bcf501c691636c28707ed5442114deb14
line_balance_physical: +30
line_balance_production: +27
peer_budget_before: 10165
peer_budget_after: 9941
commands_module_budget: 251
cargo_check: passed_449_warnings
module_size: target_entries_exact_6_known_external_overs
command_consumer_tests: 8_passed_1_known_DOC_CACHE_REV_failed_2817_filtered
command_expected_red_set: mismatch_governance_test_passed
include_str_guard: 1_passed_0_failed_2825_filtered
fmt_check: passed
diff_check: passed
full_suite_baseline: reused_143_passed_3_known_red
task_files_staged: false
git_committed: false
```
