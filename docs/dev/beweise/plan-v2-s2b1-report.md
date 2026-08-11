## Ergebnis

Der Desktop-Datei-Index wurde als reiner Umzug aus

- `src/core/business_os/rxdb_peer.rs`

nach

- `src/core/business_os/rxdb_peer_desktop_files.rs`

ausgelagert. Das neue Modul ist in `src/core/business_os/mod.rs` angemeldet; alle Modulimporte sind explizit, ohne `use super::*`.

Kein `git add` und kein Commit wurden ausgeführt.

## Bewegte Funktionen: 54

1. `desktop_file_chunk_index_window`
2. `active_desktop_file_chunk_rows_from_sqlite`
3. `dedupe_desktop_file_chunks_by_idx`
4. `desktop_file_chunk_stream_score`
5. `equivalent_desktop_file_chunk_rows_from_sqlite`
6. `active_desktop_file_metadata_from_sqlite`
7. `desktop_file_chunk_rows_by_id_from_sqlite`
8. `desktop_file_chunk_rows_by_row_id_from_sqlite`
9. `desktop_file_chunk_rows_for_file_id`
10. `prune_desktop_file_chunk_generations`
11. `desktop_file_chunk_generation_key`
12. `sync_desktop_file_index_with_database`
13. `sync_desktop_file_index_with_database_if_changed`
14. `sync_desktop_file_scan_roots_with_database`
15. `sync_desktop_file_scan_roots_with_database_unbounded`
16. `sync_desktop_file_scan_roots_with_database_if_changed`
17. `sync_desktop_file_scan_with_database`
18. `desktop_file_scan_may_mark_missing`
19. `log_desktop_file_index_maintenance_stats`
20. `compact_desktop_file_index_store`
21. `compact_desktop_file_index_store_sync`
22. `compact_desktop_file_index_store_sync_with_config`
23. `apply_desktop_file_chunk_cache_policy`
24. `desktop_file_chunk_cache_live_bytes`
25. `desktop_file_chunk_cache_candidates`
26. `desktop_file_chunk_cache_eviction_metadata`
27. `prepare_desktop_file_cache_eviction`
28. `ensure_desktop_file_chunk_cache_state_table`
29. `desktop_file_chunk_cache_state`
30. `save_desktop_file_chunk_cache_state`
31. `unsafe_desktop_file_index_candidates_sql`
32. `desktop_file_chunk_id_bounds`
33. `desktop_file_index_document_is_unsafe`
34. `is_unsafe_desktop_file_index_path`
35. `ensure_desktop_file_index_query_indexes`
36. `prepare_unsafe_desktop_file_tombstone`
37. `collect_desktop_file_index_candidates`
38. `collect_desktop_file_index_scan`
39. `collect_desktop_file_index_scan_unbounded`
40. `collect_desktop_file_index_scan_sync`
41. `collect_desktop_file_index_scan_sync_unbounded`
42. `normalize_desktop_file_scan_roots`
43. `desktop_file_scan_roots`
44. `desktop_file_scan_root_label`
45. `is_safe_desktop_file_scan_root`
46. `ensure_safe_desktop_file_index_path`
47. `is_broad_desktop_file_scan_root`
48. `desktop_file_virtual_location`
49. `desktop_file_id`
50. `mark_missing_scanned_desktop_files`
51. `load_live_ctox_desktop_file_documents`
52. `ensure_desktop_file_index_query_indexes_for_root`
53. `ensure_desktop_file_index_query_indexes_for_root_sync`
54. `load_live_ctox_desktop_file_documents_sync`

Alle 54 Funktionsnamen waren bereits in `HEAD` vorhanden. Es wurden keine dirty-only `fn`-, `static`- oder `const`-Symbole eingeführt.

## Bewusst in `rxdb_peer.rs` verblieben

### Vier Stamp-/Scheduling-Funktionen

Diese bleiben entsprechend der verbindlichen Kartierung beim Native-Peer-Loop:

- `desktop_file_index_projection_stamp`
- `desktop_file_scan_roots_stamp`
- `desktop_file_index_should_collect_scan`
- `desktop_file_index_sleep_interval`

### Öffentliche Fassade und Peer-Methoden

Diese bleiben in der Hauptdatei, weil sie Teil der bestehenden Native-Peer-/Datenbank-Fassade sind und nicht die verschobene Indeximplementierung darstellen:

- `sync_desktop_file_from_path`
- `materialize_desktop_file_from_path`
- `sync_desktop_file_from_path_with_policy`
- `sync_desktop_files_from_workspace_root`
- `sync_desktop_file_index`
- `sync_desktop_file_index_if_changed`
- `NativePeer::upsert_desktop_file_from_path`
- `NativePeer::sync_desktop_files_from_scan_roots`
- `NativePeer::sync_desktop_files_from_scan_roots_unbounded`
- `record_desktop_file_index_loop_result`

### Geteilte Grenzhelfer

Generische Demand-Chunk-Helfer bleiben in der Hauptdatei, weil sie außer Desktop-Dateien auch Dokument-, Spreadsheet-, Modulquell- und dynamische Runtime-Chunk-Collections bedienen:

- `demand_file_source_configs`
- `register_demand_file_sources`
- `stream_demand_file_chunks`
- `stream_demand_file_chunks_inner`
- `demand_file_chunk_rows_for_key_from_sqlite`
- `demand_file_source_error`
- `chunk_id_prefix_bounds`
- Runtime-Modul-Demand-Chunk-Helfer

Ebenfalls verblieben sind gemeinsam verwendete SQLite-, Pfad-, Revisions-, Scan- und Folder-Helfer wie `sqlite_table_exists`, `sqlite_table_has_column`, `sqlite_pragma_u64`, `maintenance_revision`, `collect_files_bounded`, `collect_files_unbounded`, `should_eager_sync_file` und die CTOX-Desktop-Folder-Helfer. Die notwendigen Querzugriffe erhielten minimale `pub(super)`-Sichtbarkeit.

## Pflicht-Checks

### 1. Funktionsnamens-Schnittmenge

Über alle definierten Funktionsnamen einschließlich Tests:

- `rxdb_peer.rs`: 509 eindeutige Namen
- `rxdb_peer_desktop_files.rs`: 54 eindeutige Namen
- Schnittmenge: **0**

Damit liegt weder Copy-statt-Move noch ein Rust-Shadowing-Split-Brain vor.

### 2. Zeilenbilanz

Physische Zeilen:

- Original `rxdb_peer.rs`: 22.658
- Neue `rxdb_peer.rs`: 20.815
- Neue Desktop-Datei: 1.908
- Vereinigung: 22.723
- Abweichung: **+65 Zeilen**

Produktionszeilen nach der Modulgrößen-Vertragsregel:

- Original: 12.718
- Neue Hauptdatei: 10.875
- Neues Modul: 1.908
- Vereinigung: 12.783
- Abweichung: **+65 Zeilen**

Zusätzlich kam eine Modul-Anmeldungszeile in `mod.rs` hinzu. Die Abweichung liegt deutlich unter der Stoppschwelle von 200 Zeilen und besteht aus expliziten Imports, Modulkopf, Sichtbarkeitsgrenzen und Rustfmt-Zeilenumbrüchen.

### 3. Normalisierte Vereinigung

Normalisiert wurden:

- Whitespace,
- die für die Modulgrenze nötigen `pub(super)`-Zusätze,
- ausschließlich durch Rustfmt entstandene abschließende Parameter-Kommas.

Danach wurden die 54 nach Funktionsnamen versehenen Spans sortiert und per SHA-256 geprüft:

- Original: `4189efcb5200eb954a68fbe86306431900800f3e5e17d953d5e5ebaef2ac73d5`
- Verschoben: `4189efcb5200eb954a68fbe86306431900800f3e5e17d953d5e5ebaef2ac73d5`
- Ergebnis: **identisch**

## Modulgrößen-Budgets

Vorher:

```text
src/core/business_os/rxdb_peer.rs = 12718
```

Nachher:

```text
src/core/business_os/rxdb_peer.rs = 10875
src/core/business_os/rxdb_peer_desktop_files.rs = 1908
```

## `include_str!`-Wächter

Gefunden wurde genau ein Wächter:

```text
src/core/business_os/rxdb_peer.rs:
include_str!("rxdb_peer.rs")
```

Er prüft ausschließlich `async fn *_loop` auf erlaubte Idle-Strategien. Keine der 54 verschobenen Funktionen ist ein solcher Loop. Daher musste seine Dateiliste nicht erweitert werden.

## Verifikation

### Vorher-Baseline aus Nachtrag v2

Die übernommene Baseline bleibt:

- 146 Tests
- 143 bestanden
- 3 bekannte rote Tests:
  - `native_peer_consumes_pending_business_command_written_directly_to_sqlite`
  - `native_peer_consumes_pending_module_governance_commands`
  - `sync_business_record_projections_materializes_procedural_knowledge`

### Nachher

- `cargo check --bin ctox`: **grün** nach dem Umzug und den Import-/Sichtbarkeitsanpassungen.
- `cargo test --bin ctox desktop_file -- --test-threads=2`:
  - **36 bestanden**
  - **0 fehlgeschlagen**
  - 2.790 ausgefiltert
  - Laufzeit: 622,83 Sekunden
  - Keine neuen roten Tests.
- `cargo fmt --check` für die berührten Rust-Dateien: **grün**.
- `cargo test --bin ctox module_size -- --test-threads=4`:
  - Der neue `rxdb_peer`-Eintrag und der neue Modul-Eintrag sind exakt.
  - Der Test bleibt wegen sechs bereits dirty und außerhalb der Whitelist liegenden Fremdmodule rot:
    - `office_engine.rs`: 14.598 / 13.953
    - `store.rs`: 28.413 / 27.516
    - `store_outbound_commands.rs`: 5.354 / 5.270
    - `context/lcm/mod.rs`: 6.365 / 5.627
    - `mission/channels/mod.rs`: 7.388 / 7.221
    - `service/service.rs`: 26.789 / 26.237

## Offene Bedenken

- Der abschließende zweite `cargo check` wurde bei erneutem Fremdlastanstieg nach zehn Minuten mit Exit 143 beendet. Zu diesem Zeitpunkt lag der 5-Minuten-Load bei 75,07. Der vorangegangene Check mit demselben Quellstand vor reinen Rustfmt-/Budgetanpassungen war grün.
- Das Modulgrößen-Gate kann auf der vorgegebenen Arbeitsbaum-Basis wegen der sechs fremden Budgetüberschreitungen nicht insgesamt grün werden. Die beiden von dieser Aufgabe verantworteten Budgets stimmen exakt.
- Die 90-Minuten-Zeitbox wurde wegen Ressourcenwartezeit und des abschließenden Lastanstiegs erreicht; deshalb wurden keine weiteren Vollsuite-Wiederholungen gestartet.

WORKJET-COMPLETION-RECEIPT v1
task: S2b-1
outcome: completed_with_external_baseline_blockers
moved_functions: 54
name_intersection: 0
normalized_move_sha256: 4189efcb5200eb954a68fbe86306431900800f3e5e17d953d5e5ebaef2ac73d5
cargo_check: passed_then_final_retry_timed_out_under_load
desktop_file_tests: 36_passed_0_failed
module_size: target_entries_exact_external_dirty_modules_failed
fmt_check: passed
git_staged: false
git_committed: false
