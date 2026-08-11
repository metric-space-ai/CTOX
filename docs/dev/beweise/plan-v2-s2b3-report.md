## Ergebnis

Das mechanische Projektions-Gerüst wurde aus `rxdb_peer.rs` nach dem neuen Modul

`src/core/business_os/rxdb_peer_projections.rs`

verschoben. Die Schwelle wurde mit **21 sauber beweglichen Funktionen** überschritten; ein Karten-only-Stopp war daher nicht erforderlich.

Geändert wurden ausschließlich die erlaubten Dateien:

- `src/core/business_os/rxdb_peer.rs`
- `src/core/business_os/rxdb_peer_projections.rs` — neu, untracked
- `src/core/business_os/mod.rs`
- `contracts/module_size_budget.txt`

Nichts wurde gestaged oder committed.

## Schnittkarte

### Bewegt: 21 Funktionen

**Metrik- und Testinstrumentierungs-Mechanik**

1. `chat_tracking_batch_document_lookups`
2. `record_chat_tracking_batch_document_lookup`
3. `reset_chat_tracking_batch_document_lookups`
4. `chat_tracking_batch_document_lookup_count`
5. `record_native_peer_loop_result`
6. `record_desktop_file_index_loop_result`
7. `record_native_peer_bool_loop_result`

**Generisches Loop-Runner-Gerüst und dünne Bindings**

8. `run_background_projection_loop`
9. `sync_channel_state_background_loop`
10. `sync_business_users_background_loop`
11. `sync_runtime_settings_background_loop`
12. `sync_workspace_branding_background_loop`
13. `sync_module_catalog_background_loop`
14. `sync_ticket_state_background_loop`
15. `sync_knowledge_tables_background_loop`

**Collection-Filter-Tabellen**

16. `support_projection_collection`
17. `appsec_projection_collection`
18. `threads_projection_collection`

**Lookup-/Upsert-Batching-Mechanik**

19. `find_projection_documents_by_id`
20. `upsert_business_record_projection_document`
21. `bulk_upsert_business_record_projection_documents`

Zusätzlich wurden **12 Statics** verschoben:

- `CHAT_TRACKING_BATCH_DOCUMENT_LOOKUPS`
- die elf `*_LOOP_METRICS`-Instanzen für Notes, Desktop-Index, Channel State, Business Users, Runtime Settings, Workspace Branding, Module Catalog, Ticket State, Knowledge Tables, Business Records und Business Commands.

### Bewusst in `rxdb_peer.rs` verblieben

| Gruppe | Beispiele | Grund |
|---|---|---|
| Stamp-Disziplin | `ProjectionStampStrategy`, `sync_projection_if_changed_with_strategy`, alle `*_projection_stamp`-/`*_source_stamp`-Funktionen | Semantische Entscheidung, wann ein beobachteter Stamp als verarbeitet gilt. |
| Loop-Konfiguration | `BackgroundProjectionLoopConfig`, `*_PROJECTION_LOOP` | Intervalle, Fehlertexte und Stamp-Strategie bleiben beim semantischen Kern. |
| Scheduling und Idle-Backoff | `update_projection_idle_rounds`, `business_os_projection_sleep_secs`, `business_record_projection_sleep_secs`, `projection_sleep_secs` | Explizit aus S2b ausgeschlossen; der verschobene Runner konsumiert diese Helfer nur. |
| Cursor-/Fortschrittslogik | `sync_business_record_projections_background_loop`, `load_business_record_projection_progress`, `persist_business_record_projection_progress`, Slice-/Cursor-Funktionen | Persistierte Cursor, Seitenfortschritt und Partial-Slice-Verhalten gehören zu S3. |
| Fachliche Projektoren | `sync_*_with_database`, `project_support_command_result`, `project_threads_command_result`, `project_appsec_command_result` | Enthalten Collection- und Business-Semantik, nicht nur Mechanik. |
| Idempotenzentscheidungen | `incremental_upsert_projection_if_changed`, `canonical_projection_document_for_compare`, `projection_document_has_valid_revision` | Entscheiden fachlich, wann ein Dokument als unverändert gilt. |
| Dokumentnormalisierung | `normalize_business_record_projection_document`, `normalize_projection_*`, `fill_projection_document_envelope` | Schema- und Default-Semantik bleibt im Kern. |
| Tombstone-Verhalten | `is_projection_tombstone`, `upsert_business_record_projection_tombstone`, `prepare_projection_tombstone_document` | Lösch- und Revisionssemantik wurde nicht verlagert. |
| Einzelrecord-Auswahl | `upsert_business_record_projection` | Enthält Pull-Logik und Collection-Sonderfälle. |
| Collection-Ermittlung | `business_record_projection_collections`, `business_record_projection_collections_for_root` | Runtime-Schema- und Collection-Policy-Semantik. |
| Spezial-Loops | Notes-, Business-Record- und Command-Loops | Watcher, Cursor, Command-Wake und eigene Backoff-Disziplin sind nicht generisch genug. |
| Metrik-Typ/Snapshot | `NativePeerLoopMetrics`, `native_peer_performance_snapshot`, `native_peer_loop_metrics` | Wird auch vom Browser-Loop und zentralen Peer-Status konsumiert. |

Die verbliebenen Symbole erhielten nur die minimal nötigen `pub(super)`-Grenzen.

## Pflicht-Checks

### Funktionsnamens-Schnittmenge

Eindeutige erkannte Funktionsnamen:

- `rxdb_peer.rs`: **484**
- `rxdb_peer_projections.rs`: **21**
- Schnittmenge: **0**

Damit liegt kein Copy-statt-Move und keine doppelte Implementierung vor.

### Dirty-only-Symbole

Für alle `fn`-, `static`- und `const`-Namen der neuen Vereinigung wurde je Symbol `git grep <symbol> HEAD` ausgeführt:

- geprüft: **591**
- dirty-only: **0**

### Normalisierte Move-SHA

Normalisiert wurden Whitespace, notwendige `pub(super)`-Zusätze und Rustfmt-Parameterkommas.

```text
Vorher:  dd4ecb78ceaba9f85f928b598fa1ccb71776f9c591884aa16ceff699431919bf
Nachher: dd4ecb78ceaba9f85f928b598fa1ccb71776f9c591884aa16ceff699431919bf
```

Ergebnis: **identisch** für alle 21 verschobenen Funktionsspans.

### Zeilenbilanz

Physische Zeilen:

- vorher `rxdb_peer.rs`: **20.601**
- nachher `rxdb_peer.rs`: **20.110**
- neues Modul: **553**
- neue Vereinigung: **20.663**
- Overhead: **+62**

Produktionszeilen nach der Budget-Vertragsregel:

- vorher: **10.661**
- nachher Peer: **10.165**
- neues Modul: **551**
- neue Vereinigung: **10.716**
- Overhead: **+55**

Der Overhead besteht aus expliziten Imports, Modulkopf, Sichtbarkeitsgrenzen, Guard-Anpassung und Modulgrößen-Terminierung.

## Budgets

Vorher:

```text
src/core/business_os/rxdb_peer.rs = 10661
```

Nachher:

```text
src/core/business_os/rxdb_peer.rs = 10165
src/core/business_os/rxdb_peer_projections.rs = 551
```

Beide Zielwerte stimmen exakt mit der Vertragszählung überein. Ein abschließender leerer `#[cfg(test)] mod tests {}` hält den letzten Testmarker nach dem gesamten Produktionscode; die vorhandene test-only Batch-Instrumentierung liegt weiter oben im Modul.

## `include_str!`-Guard

Der Idle-Strategie-Guard liest nun beide Quellen:

- `rxdb_peer.rs`
- `rxdb_peer_projections.rs`

Für den Source-Scan wird `pub(super) async fn` testintern auf `async fn` normalisiert. Damit bleiben die verschobenen Loop-Bindings von derselben Ratsche erfasst.

Guard-Test:

- **1 bestanden**
- **0 fehlgeschlagen**

## Verifikation

### Ressourcen-Gate

- 5-Minuten-Load vor den finalen Checks: **16,77**
- frei auf `/Volumes/tmp`: **338.920.948 KiB**
- 5-GiB-Hard-Stop nicht annähernd erreicht.

### `cargo check --bin ctox`

- Exit **0**
- **grün**
- 449 bestehende Workspace-Warnungen

### Projektionsfilter

```text
cargo test --bin ctox projection -- --test-threads=2
```

- **103 bestanden**
- **1 fehlgeschlagen**
- **2.722 ausgefiltert**
- Laufzeit: **642,39 s**

Einzige Rote, bereits Teil der übernommenen Baseline:

- `sync_business_record_projections_materializes_procedural_knowledge`

Keine neue Rote.

### Modulgrößen-Gate

Aufgabenbezogene Budgets sind exakt. Der Test bleibt ausschließlich wegen der sechs bekannten fremden OVERs rot:

- `office_engine.rs`: 14.598 / 13.953
- `store.rs`: 28.413 / 27.516
- `store_outbound_commands.rs`: 5.354 / 5.270
- `context/lcm/mod.rs`: 6.365 / 5.627
- `mission/channels/mod.rs`: 7.388 / 7.221
- `service/service.rs`: 26.789 / 26.237

Testausgabe:

- **0 bestanden**
- **1 fehlgeschlagen**
- **2.825 ausgefiltert**

### Weitere Checks

- Rustfmt-Check der berührten Rust-Dateien: **grün**
- `git diff --check`: **grün**
- Peer-Vollsuite nicht neu gemessen; Baseline **143/3** übernommen.
- Fortschritt fortlaufend nach `/Volumes/tmp/ctox-pipeline/s2b3-fortschritt.md` geschrieben.

## Offene Bedenken

1. Die verschobene Mechanik konsumiert weiterhin zahlreiche semantische Peer-Helfer über `pub(super)`. Das ist absichtlich die Vorarbeit für S3 und keine Runner-Vereinheitlichung.
2. Das globale Modulgrößen-Gate bleibt wegen sechs außerhalb der Whitelist liegender Überschreitungen rot.
3. Das neue Modul ist entsprechend der Vorgabe untracked; es wurde weder `git add` noch ein Commit ausgeführt.
4. Der geteilte Checkout enthält weiterhin zahlreiche fremde Änderungen; die vier S2b-3-Zieldateien waren zu Beginn sauber und sind nicht gestaged.

```text
WORKJET-COMPLETION-RECEIPT v1
task: S2b-3
outcome: completed_with_known_projection_red_and_external_module_size_overs
moved_functions: 21
moved_statics: 12
name_intersection: 0
dirty_only_fn_static_const: 0_of_591
normalized_move_sha256: dd4ecb78ceaba9f85f928b598fa1ccb71776f9c591884aa16ceff699431919bf
line_balance_physical: +62
line_balance_production: +55
peer_budget_before: 10661
peer_budget_after: 10165
projection_module_budget: 551
cargo_check: passed_449_warnings
projection_tests: 103_passed_1_known_failed_2722_filtered
module_size: target_entries_exact_6_known_external_overs
include_str_guard: passed_1_0
fmt_check: passed
diff_check: passed
full_suite_baseline: reused_143_passed_3_known_red
git_staged: false
git_committed: false
```
