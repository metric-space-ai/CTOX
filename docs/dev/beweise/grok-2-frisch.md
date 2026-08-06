# GROK-2 — Reine Messung: globale Test-Zaehler in `src/`

**Arbeitsbaum:** `/Users/michaelwelsch/Documents/ctox` (nur gelesen, keine Aenderungen).
**Datum der Messung:** 2026-08-05
**Methode:** `rg` ueber `src/**/*.rs` nach
`static … Atomic* / Mutex / OnceLock`, dann Filter auf `#[cfg(test)]` und
auf Zaehler, die ausschliesslich (oder primaer) von Testcode gelesen/geschrieben
werden; anschliessend jede Fundstelle bis zur umschliessenden `#[test]`-Funktion
zurueckverfolgt und Assertion-Stil (Absolut vs. Diff/Existenz) klassifiziert.

**Kriterium Parallelitaet:** zwei Tests koennen parallel laufen, wenn sie denselben
Namenspraefix teilen **oder** im selben Modul / unter demselben cargo-Filter
(z. B. `reconcile_business_chat_tracking_projections`, `rescan_…` /
`materialized_…`, `queue_task_*`, `durable_status_*`, `file_backed_*`) liegen.

---

## Kurzantwort

**Ja, es ist ein Muster — kein Einzelfall.**
Neben dem bekannten `CHAT_TRACKING_BATCH_DOCUMENT_LOOKUPS` gibt es **mehrere
prozessglobale Zaehler**, die mit `store(0)` + absolute `assert_eq!(…, N)`
arbeiten und damit unter breitem Filter flaky-verdaechtig sind. Daneben gibt es
eine groessere Klasse **pfad-/schluessel-partitionierter** Zaehler (`OnceLock<Mutex<Map>>`),
die meist unkritisch sind, sowie **Delta-basierte** Atomarzähler.

---

## A) Prozessglobale Atomare (Absolute Assertions → flaky-verdaechtig)

### 1. `CHAT_TRACKING_BATCH_DOCUMENT_LOOKUPS`
| | |
|---|---|
| **Datei:Zeile** | `src/core/business_os/rxdb_peer.rs:466` |
| **Typ** | `#[cfg(test)] static AtomicUsize` |
| **Inkrement** | `find_projection_documents_by_id` (L8993) — wird von `reconcile_business_chat_tracking_projections` und damit auch von allgemeinerer Projection-Reconciliation aufgerufen |
| **Tests R/W** | nur **ein** Test liest/schreibt: `reconcile_business_chat_tracking_projections_batches_active_document_lookups` (store 0 → assert `== 2`) |
| **Parallelitaet** | **ja** — Schwester-Tests `reconcile_business_chat_tracking_projections_*` und jeder andere Test, der `reconcile_business_chat_tracking_projections` / `find_projection_documents_by_id` anstoesst, erhoeht denselben Zaehler; unter Filter `reconcile_business_chat_tracking_projections` laufen sie parallel |
| **Assertion** | **ABSOLUT** (`store(0)` + `assert_eq!(load(), 2)`) |
| **flaky-verdaechtig** | **JA** — das ist der bereits beobachtete Fall: Reset und Messung teilen sich mit allen parallel laufenden Lookups |

### 2. `DESKTOP_FILE_CHUNK_COMPLETENESS_CHECKS`
| | |
|---|---|
| **Datei:Zeile** | `src/core/business_os/desktop_files.rs:28` |
| **Typ** | `#[cfg(test)] static AtomicUsize` |
| **Inkrement** | `desktop_file_chunk_generation_is_complete` (L465) — Produktionspfad der Desktop-File-Scans |
| **Tests R/W** | `rescan_of_unchanged_workspace_is_a_no_op` (rxdb_peer.rs ~22597); `materialized_large_file_survives_lazy_rescan` (~22720). Beide: `store(0)` + `assert_eq!(load(), 0)` |
| **Parallelitaet** | **ja** — beide Tests teilen Praefix-Umfeld (Desktop-File-Scans in `rxdb_peer` tests) und inkrementieren denselben globalen Zaehler; jeder andere Test, der `desktop_file_chunk_generation_is_complete` anstoesst, stoert |
| **Assertion** | **ABSOLUT** (`== 0` nach Reset) |
| **flaky-verdaechtig** | **JA** — zwei Tests resetten denselben Zaehler; ein Reset des einen kann die Null-Assertion des anderen zerstoeren, und fremde Scans erhoehen den Wert |

### 3. `FIND_DOCUMENTS_BY_ID_WRITER_FALLBACKS`
| | |
|---|---|
| **Datei:Zeile** | `src/core/rxdb/src/storage/sqlite/instance.rs:74` |
| **Typ** | `#[cfg(test)] static AtomicUsize` |
| **Inkrement** | `with_read_connection` Fallback-Pfad (L468) — prozessglobal, bei *jedem* Writer-Fallback fuer find-by-id |
| **Tests R/W** | `find_documents_by_id_file_backed_uses_read_only_connection` (store 0 → assert `== 0`) |
| **Parallelitaet** | **ja** — alle sqlite-instance-Tests im selben Binary koennen den Fallback triggern; Name-Praefix `find_documents_by_id_*` / breiter `sqlite` Filter |
| **Assertion** | **ABSOLUT** (`== 0`) |
| **flaky-verdaechtig** | **JA** — `== 0` auf globalem Zaehler, obwohl nur *dieser* Test resettet; parallele Fallbacks anderer Tests machen ihn gruen→rot |

### 4. `QUERY_WRITER_FALLBACKS`
| | |
|---|---|
| **Datei:Zeile** | `src/core/rxdb/src/storage/sqlite/instance.rs:78` |
| **Typ** | `#[cfg(test)] static AtomicUsize` |
| **Inkrement** | `with_read_connection` (L471) |
| **Tests R/W** | `query_fallback_does_not_wait_for_writer_mutex` (store 0 → assert `== 0`) |
| **Parallelitaet** | **ja** (sqlite instance tests) |
| **Assertion** | **ABSOLUT** (`== 0`) |
| **flaky-verdaechtig** | **JA** — gleiches Muster wie #3 |

### 5. `CHANGED_DOCUMENTS_SINCE_WRITER_FALLBACKS`
| | |
|---|---|
| **Datei:Zeile** | `src/core/rxdb/src/storage/sqlite/instance.rs:76` |
| **Typ** | `#[cfg(test)] static AtomicUsize` |
| **Inkrement** | `with_read_connection` (L474) |
| **Tests R/W** | `changed_documents_since_file_backed_uses_read_only_connection` (store 0 → assert `== 0`) |
| **Parallelitaet** | **ja** (sqlite instance tests) |
| **Assertion** | **ABSOLUT** (`== 0`) |
| **flaky-verdaechtig** | **JA** — gleiches Muster wie #3/#4 |

### 6. `SQLITE_DOCUMENT_BY_ID_CALL_COUNT` / `SQLITE_DOCUMENTS_BY_IDS_CALL_COUNT`
| | |
|---|---|
| **Datei:Zeile** | `src/core/rxdb/src/storage/sqlite/sql.rs:27` / `:29` |
| **Typ** | `#[cfg(test)] static AtomicUsize` |
| **API** | `reset_sqlite_document_lookup_counts()`, `sqlite_document_by_id_call_count()`, `sqlite_documents_by_ids_call_count()` |
| **Inkrement** | `document_by_id` / `documents_by_ids` (L659/L679) — Produktionspfad |
| **Tests R/W** | **keine aktuellen Test-Leser/Schreiber gefunden** (API vorhanden, aber unbenutzt) |
| **Parallelitaet** | n/a |
| **Assertion** | — |
| **flaky-verdaechtig** | **nein (derzeit latent)** — sobald ein Test `reset`+absolute Assert nutzt, wird es flaky |

---

## B) Prozessglobale Atomare mit Diff-/Existenz-Assertions (unkritisch)

### 7. `DIRECT_SESSION_EVENT_DESERIALIZE_CALLS`
| | |
|---|---|
| **Datei:Zeile** | `src/core/execution/agent/direct_session.rs:92` |
| **Typ** | `#[cfg(test)] static AtomicUsize` |
| **Tests** | `direct_session_ignores_stream_delta_events_before_deserialize` (`before` → `== before`); `direct_session_extracts_agent_message_events` (`before` → `== before + 1`) |
| **Parallelitaet** | ja (gemeinsamer Praefix `direct_session_*`) |
| **Assertion** | **DIFF** |
| **flaky-verdaechtig** | **nein** — Delta-basiert; theoretische Race nur bei gleichzeitigem Inkrement waehrend der Messung desselben Counters, aber die gemessenen Pfade sind rein lokal/synchron und rufen keine anderen Tests |

### 8. `TICKET_SELF_WORK_ASSIGNMENT_BATCH_HYDRATION_CALLS`
| | |
|---|---|
| **Datei:Zeile** | `src/core/mission/tickets/mod.rs:250` |
| **Typ** | `#[cfg(test)] static OnceLock<Mutex<usize>>` — **ein** globaler Skalar (nicht path-keyed!) |
| **Inkrement** | `hydrate_ticket_self_work_items_with_latest_assignments` via `record_ticket_self_work_assignment_batch_hydration_for_tests` |
| **Tests** | `ticket_self_work_list_batches_latest_assignment_hydration` (`before` → delta `== 1`) |
| **Parallelitaet** | ja (`ticket_self_work_*`); andere Tests, die self-work listen, inkrementieren mit |
| **Assertion** | **DIFF** |
| **flaky-verdaechtig** | **nein (schwach)** — Diff schuetzt vor parallelen Geschwistern; nur eine sehr enge Interleaving-Race (fremdes Inkrement exakt zwischen before und after) waere moeglich. Praktisch harmloser als absolute Zaehler |

### 9. `DROPPED_AUDIT_WRITES` / `DROPPED_HARNESS_FLOW_EVENTS`
| | |
|---|---|
| **Datei:Zeile** | `src/core/service/governance.rs:499`; `src/core/service/harness_flow.rs:312` |
| **Typ** | Produktions-`AtomicU64` (nicht `cfg(test)`), aber Tests lesen sie |
| **Tests** | governance: before/after delta; harness: `after > before` |
| **Assertion** | **DIFF / Existenz** |
| **flaky-verdaechtig** | **nein** |

---

## C) Path-/Key-partitionierte Map-Zaehler (`OnceLock<Mutex<Map>>`)

Diese teilen zwar den globalen Container, partitionieren aber nach `PathBuf` /
`(path, …)` / Tabellennamen. Tests resetten/asserten typischerweise **nur ihren
eigenen Schluessel**. Absolute Assertions auf den eigenen Key sind deshalb
**normalerweise nicht flaky**, solange kein Test den gesamten Map cleart.

### 10. Store-Projektion: `RXDB_TABLE_COLUMN_LOADS`, `RXDB_COLLECTION_WRITER_OPENS`, `PUSH_COLLECTION_RECORDS_STORE_TRANSACTIONS`
- **Datei:Zeile:** `src/core/business_os/store.rs:217 / 219 / 221` (`#[cfg(test)]`)
- **Key:** `(root, table/collection)`-String
- **Tests:**
  - `rxdb_projection_writer_cache_reuses_table_metadata_for_batch` (absolute `== 1` auf eigenem root/table)
  - `rxdb_projection_writer_cache_reuses_writers_across_command_fanout` (absolut auf eigenem root)
  - `push_collection_records_batches_non_command_store_writes_in_one_transaction`
  - `push_collection_records_refreshes_stale_queue_projection_in_same_batch_transaction`
- **Parallelitaet:** ja (gemeinsamer Praefix `rxdb_projection_writer_cache_*` / `push_collection_records_*`)
- **Assertion:** absolut, aber **pro root**
- **flaky-verdaechtig:** **nein** (Key-Isolation)

### 11. Channels: `CHANNEL_SCHEMA_ENSURE_COUNTS`, `CHANNEL_OPEN_ROUTING_ENSURE_COUNTS`, `QUEUE_TASK_LIST_CACHE_MISS_COUNTS`, `QUEUE_TASK_COUNT_CACHE_MISS_COUNTS`, `CHANNEL_DB_OPEN_CALL_COUNTS`
- **Datei:Zeile:** `src/core/mission/channels/mod.rs:252 / 261 / 265 / 269 / 273` (`#[cfg(test)]`)
- **Keys:** Schema/Routing: `ChannelSchemaCacheKey` (unix: `(PathBuf, dev, ino)`); Queue-Misses: cache keys inkl. db_path; DB-opens: `PathBuf`
- **Tests:**
  - `channel_schema_is_ensured_once_per_open_database_file` (absolut `== 1` auf eigener db)
  - `queue_task_list_cache_reuses_idle_reads_until_store_changes` (Miss-Counts steigen/bleiben — pro Key)
  - `queue_task_caches_ignore_sync_run_metadata_churn`
  - `queue_task_count_cache_reuses_idle_reads_until_store_changes`
  - zusaetzlich Leser in `store.rs` / `queue.rs` Tests via `channel_db_open_count_for_tests` (reset pro path, absolut)
- **Parallelitaet:** ja (`queue_task_*`, `channel_*`)
- **flaky-verdaechtig:** **nein** (Key-Isolation; Schema-Counts sind monoton pro Datei und werden nicht global genullt)

### 12. Tickets: `TICKET_SELF_WORK_LIST_CACHE_MISS_COUNTS`, `TICKET_WORKFLOW_MATERIALIZE_CACHE_MISS_COUNTS`, `TICKET_DB_OPEN_CALL_COUNTS`
- **Datei:Zeile:** `src/core/mission/tickets/mod.rs:242 / 246 / 252` (`#[cfg(test)]`)
- **Tests:**
  - `ticket_self_work_list_cache_reuses_idle_reads_until_store_changes` (Miss-Counts pro db_path)
  - `ticket_workflow_materialize_cache_reuses_idle_noop_until_store_changes`
  - `business_os_ticket_projection_reuses_one_ticket_db_connection` (**delta** auf db_path)
  - `queue_ticket_bridge_list_batch_hydrates_tasks_and_tickets` (reset/assert ticket db opens pro path)
- **flaky-verdaechtig:** **nein** (Key-Isolation / Diff)

### 13. Plan / Queue-Bridge / Queue-Metadata / API-Costs / Mailserver
| Zaehler | Datei:Zeile | Tests | Assertion | flaky? |
|---|---|---|---|---|
| `PLAN_DB_OPEN_COUNTS` | plan.rs:45 | `emit_due_steps_reuses_plan_db_connection_across_due_goals` | reset+`==1` pro root | nein |
| `QUEUE_BRIDGE_DB_OPEN_COUNTS` | queue.rs:54 | `queue_ticket_bridge_list_batch_hydrates_tasks_and_tickets`; `spill_candidates_batch_bridge_and_failure_signature_evidence` | reset+`==1` pro root | nein |
| `QUEUE_METADATA_DB_OPEN_COUNTS` | queue.rs:57 | `cleanup_scope_batches_metadata_reads_for_scanned_tasks` | reset+`==1` pro root | nein |
| `API_COST_DB_OPEN_COUNTS` | api_costs.rs:14 | `batch_recording_uses_one_db_open_for_multiple_events` | reset+`==0`/`==1` pro root | nein |
| `SQLITE_STORE_OPEN_COUNTS` | mailserver/…/sqlite.rs:21 | `imap_select_reuses_cached_connection_for_mailbox_and_count`; `imap_fetch_and_store_hot_path_reuses_cached_connection`; `smtp_calendar_contact_and_greylist_hot_paths_reuse_cached_connection` | reset+`==1` pro db_path | nein |

### 14. Service: `CHANNEL_ROUTER_SOURCE_STAMP_LOAD_COUNTS`
- **Datei:Zeile:** `service.rs:235` (`#[cfg(test)]`)
- **Test:** `channel_router_source_stamp_cache_reuses_unchanged_db_and_reopens_on_queue_work` (reset+absolut pro root)
- **flaky-verdaechtig:** **nein**

### 15. Service: `DURABLE_STATUS_LOAD_COUNTS`, `DURABLE_STATUS_LCM_OUTCOME_OPEN_COUNTS`
- **Datei:Zeile:** `service.rs:3794 / 3796` (`#[cfg(test)]`)
- **Keys:** root bzw. core_db path
- **Tests:**
  - `durable_status_snapshot_reuses_unchanged_store_after_ttl` (absolut `== 1`, `== 2` auf eigenem root)
  - `durable_status_snapshot_ignores_sync_run_metadata_churn` (absolut auf eigenem root)
- **Besonderheit:** `clear_durable_status_snapshot_cache_for_tests` (L4598–4607) macht **`.clear()` auf der gesamten Map**, nicht nur den eigenen Key. Beide Tests rufen das zu Beginn. Parallel unter Filter `durable_status_snapshot_*` kann Test A die Counts von Test B loeschen → absolute Asserts von B fallen.
- **Parallelitaet:** ja (gemeinsamer Praefix `durable_status_snapshot_*`)
- **flaky-verdaechtig:** **JA** — nicht wegen des Zaehlers selbst, sondern wegen des **globalen Clear** in der Reset-Hilfe

### 16. RxDB sqlite: `CHANGED_DOCUMENTS_SINCE_TABLE_CALLS`
- **Datei:Zeile:** `instance.rs:71` (`#[cfg(test)]`)
- **Key:** Tabellenname
- **Tests:** `change_stream_emits_other_connection_sqlite_writes`; `file_backed_external_poll_has_no_per_collection_idle_safety_drains` (reset pro table, assert `== 0` nach Idle)
- **Hinweis:** Tabellennamen sind pro Test-Instanz typischerweise unique (`idle_docs_{idx}`, temp schemas). Kollision nur bei identischen collection names parallel.
- **flaky-verdaechtig:** **nein (praktisch)**; theoretisch bei geteilten Tabellennamen

---

## D) Globale Test-Konfiguration / Sequencer (keine Zaehler-Assertions)

Diese teilen Zustand, sind aber **keine Lookup-Zaehler** im Sinne der Frage.
Der Vollstaendigkeit halber:

| Datei:Zeile | Name | Rolle | flaky-Zaehler? |
|---|---|---|---|
| instance.rs:80 | `TEST_EXTERNAL_POLL_SAFETY_INTERVAL_MS` | globaler Override; Drop-Guard setzt 0 | nein (Config; Race moeglich, aber kein Count-Assert) |
| instance.rs:82 | `TEST_QUERY_STREAM_ROW_DELAY_MS_BY_TABLE` | Delay pro table | nein (table-keyed) |
| service.rs:26234+ | `*_GATE_TEST_LOCK` | Mutex-Serialisierung fuer process-globale Gates | bewusst gegen Flakes |
| lcm/mod.rs:52 | `TEMP_DB_COUNTER` | unique temp paths | Sequencer, nicht Assert |
| state_invariants.rs:271, service.rs:26403, render.rs:4789, rxdb_peer.rs:14788 | `TEMP_ROOT_SEQUENCE` / `TEST_APP_SEQUENCE` / `TEST_RXDB_DATABASE_COUNTER` | Sequencer | nein |
| sql.rs:24 | `SQLITE_JSON_DOCUMENT_DECODE_COUNT` | **thread_local!** Cell | nein (thread-lokal; absolute Asserts ok) |
| unlock_report.rs:1461 | `COUNTER` | test-local unique ids | nein |
| metrics.rs:* | `SQLITE_*` runtime Atomaru64s | Produktionsmetriken; Tests nutzen `runtime_counter` **before/after** | nein |

---

## E) Zusammenfassungstabelle

| datei:zeile | Zaehlername | Tests | flaky-verdaechtig | Begruendung |
|---|---|---|---|---|
| `business_os/rxdb_peer.rs:466` | `CHAT_TRACKING_BATCH_DOCUMENT_LOOKUPS` | `reconcile_business_chat_tracking_projections_batches_active_document_lookups` (+ Inkrement durch Schwester-Reconciliation) | **ja** | `store(0)` + absolut `== 2`; parallele Lookups anderer Tests zaehlen mit |
| `business_os/desktop_files.rs:28` | `DESKTOP_FILE_CHUNK_COMPLETENESS_CHECKS` | `rescan_of_unchanged_workspace_is_a_no_op`, `materialized_large_file_survives_lazy_rescan` | **ja** | zwei Tests resetten denselben globalen Atomic und asserten absolut `== 0`; fremde Scans stoeren |
| `rxdb/…/instance.rs:74` | `FIND_DOCUMENTS_BY_ID_WRITER_FALLBACKS` | `find_documents_by_id_file_backed_uses_read_only_connection` | **ja** | global `store(0)` + absolut `== 0`; jeder parallele Writer-Fallback bricht |
| `rxdb/…/instance.rs:78` | `QUERY_WRITER_FALLBACKS` | `query_fallback_does_not_wait_for_writer_mutex` | **ja** | wie oben |
| `rxdb/…/instance.rs:76` | `CHANGED_DOCUMENTS_SINCE_WRITER_FALLBACKS` | `changed_documents_since_file_backed_uses_read_only_connection` | **ja** | wie oben |
| `service.rs:3794` | `DURABLE_STATUS_LOAD_COUNTS` | `durable_status_snapshot_reuses_unchanged_store_after_ttl`, `durable_status_snapshot_ignores_sync_run_metadata_churn` | **ja** | Map ist path-keyed, aber `clear_durable_status_snapshot_cache_for_tests` loescht **alle** Keys; Parallelitaet unter `durable_status_snapshot_*` |
| `service.rs:3796` | `DURABLE_STATUS_LCM_OUTCOME_OPEN_COUNTS` | dieselben zwei | **ja** | gleiches globales Clear |
| `rxdb/…/sql.rs:27,29` | `SQLITE_DOCUMENT_BY_ID_CALL_COUNT` / `_BY_IDS_` | (keine aktuellen Tests) | nein (latent) | Reset-API global; sobald absolut genutzt → flaky |
| `execution/…/direct_session.rs:92` | `DIRECT_SESSION_EVENT_DESERIALIZE_CALLS` | `direct_session_ignores_…`, `direct_session_extracts_…` | nein | Diff (`before` / `before+1`) |
| `tickets/mod.rs:250` | `TICKET_SELF_WORK_ASSIGNMENT_BATCH_HYDRATION_CALLS` | `ticket_self_work_list_batches_latest_assignment_hydration` | nein (schwach) | Diff; skalar global, aber delta-geschuetzt |
| `governance.rs:499` | `DROPPED_AUDIT_WRITES` | dropped-audit-write Test | nein | Diff |
| `harness_flow.rs:312` | `DROPPED_HARNESS_FLOW_EVENTS` | lossy-event Test | nein | `after > before` |
| `store.rs:217` | `RXDB_TABLE_COLUMN_LOADS` | `rxdb_projection_writer_cache_reuses_*` | nein | path/table-keyed |
| `store.rs:219` | `RXDB_COLLECTION_WRITER_OPENS` | `rxdb_projection_writer_cache_reuses_*` | nein | path/collection-keyed |
| `store.rs:221` | `PUSH_COLLECTION_RECORDS_STORE_TRANSACTIONS` | `push_collection_records_*` | nein | path/collection-keyed |
| `channels/mod.rs:252` | `CHANNEL_SCHEMA_ENSURE_COUNTS` | `channel_schema_is_ensured_once_per_open_database_file` | nein | path/(dev,ino)-keyed |
| `channels/mod.rs:261` | `CHANNEL_OPEN_ROUTING_ENSURE_COUNTS` | derselbe | nein | path-keyed |
| `channels/mod.rs:265` | `QUEUE_TASK_LIST_CACHE_MISS_COUNTS` | `queue_task_list_cache_*`, `queue_task_caches_*` | nein | cache-key-partitioniert |
| `channels/mod.rs:269` | `QUEUE_TASK_COUNT_CACHE_MISS_COUNTS` | `queue_task_count_cache_*`, `queue_task_caches_*` | nein | cache-key-partitioniert |
| `channels/mod.rs:273` | `CHANNEL_DB_OPEN_CALL_COUNTS` | store/queue Tests via `channel_db_open_count_for_tests` | nein | path-keyed reset |
| `tickets/mod.rs:242` | `TICKET_SELF_WORK_LIST_CACHE_MISS_COUNTS` | `ticket_self_work_list_cache_reuses_*` | nein | key-partitioniert |
| `tickets/mod.rs:246` | `TICKET_WORKFLOW_MATERIALIZE_CACHE_MISS_COUNTS` | `ticket_workflow_materialize_cache_reuses_*` | nein | key-partitioniert |
| `tickets/mod.rs:252` | `TICKET_DB_OPEN_CALL_COUNTS` | `business_os_ticket_projection_reuses_*`, queue-bridge Test | nein | path-keyed / Diff |
| `plan.rs:45` | `PLAN_DB_OPEN_COUNTS` | `emit_due_steps_reuses_plan_db_connection_across_due_goals` | nein | path-keyed |
| `queue.rs:54` | `QUEUE_BRIDGE_DB_OPEN_COUNTS` | `queue_ticket_bridge_list_*`, `spill_candidates_batch_*` | nein | path-keyed |
| `queue.rs:57` | `QUEUE_METADATA_DB_OPEN_COUNTS` | `cleanup_scope_batches_metadata_reads_for_scanned_tasks` | nein | path-keyed |
| `api_costs.rs:14` | `API_COST_DB_OPEN_COUNTS` | `batch_recording_uses_one_db_open_for_multiple_events` | nein | path-keyed |
| `mailserver/…/sqlite.rs:21` | `SQLITE_STORE_OPEN_COUNTS` | `imap_select_*`, `imap_fetch_*`, `smtp_calendar_*` | nein | db_path-keyed |
| `service.rs:235` | `CHANNEL_ROUTER_SOURCE_STAMP_LOAD_COUNTS` | `channel_router_source_stamp_cache_reuses_*` | nein | path-keyed |
| `rxdb/…/instance.rs:71` | `CHANGED_DOCUMENTS_SINCE_TABLE_CALLS` | `change_stream_emits_*`, `file_backed_external_poll_*` | nein | table-keyed |

---

## F) Muster-Befund

1. **Gefaehrliches Muster (absolut + global):**
   `static AtomicUsize` → Produktionscode `fetch_add` → Test `store(0)` → `assert_eq!(load(), N)`.
   Betroffen: Chat-Tracking-Lookups, Desktop-Chunk-Checks, drei Writer-Fallback-Zaehler.

2. **Sicheres Muster (partitioniert):**
   `OnceLock<Mutex<Map<Path|Key, usize>>>` + reset/assert nur des eigenen Keys.
   Dominantes Muster bei DB-Open-/Cache-Miss-Zaehlern.

3. **Sicheres Muster (Diff):**
   `let before = load(); …; assert_eq!(load(), before + N)` bzw. `> before`.
   z. B. direct_session, dropped_* counters, ticket assignment hydration.

4. **Falle trotz Partitionierung:**
   Reset-Hilfe, die `.clear()` auf der **gesamten** Map macht (`clear_durable_status_snapshot_cache_for_tests`)
   entwertet die Key-Isolation unter Parallelitaet.

5. **Latent:**
   `SQLITE_DOCUMENT_BY_ID_CALL_COUNT` / `_BY_IDS_` haben die gefaehrliche Reset-API,
   aber derzeit keinen Test-Verbraucher.

---

## G) Suchmethode (Reproduzierbarkeit)

```text
rg -n --type rust 'static.*(Atomic|Mutex|OnceLock)' src/ -g '*.rs'
rg -n --type rust -U '#\[cfg\(test\)\]\s*\nstatic' src/
rg -n --type rust 'AtomicUsize|AtomicU64' src/
# dann pro Fund:
# - Inkrementstellen (fetch_add / insert/counts)
# - Leser (load / *_for_tests)
# - umschliessende #[test] fn
# - assert_eq! absolut vs. before/after
```

**Kein cargo-Lauf. Keine Datei geaendert. Kein Commit.**
