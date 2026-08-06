# GROK-3 Report — Vier weitere globale Test-Zaehler werden partitioniert

**Arbeitsbaum:** `/Volumes/tmp/ctox-grok3` (Worktree auf `origin/main` Snapshot)
**HEAD:** `8efb91448` (clean vor Aenderung; Aenderungen uncommittet)
**Datum:** 2026-08-05
**Whitelist-Dateien:**
- `src/core/business_os/desktop_files.rs` — **unverändert**
- `src/core/rxdb/src/storage/sqlite/instance.rs` — **geändert**

---

## was_geaendert

Nur `src/core/rxdb/src/storage/sqlite/instance.rs` (git diff --stat: `+81 / -13`).

Drei prozessglobale Writer-Fallback-Zaehler wurden vom Muster
`AtomicUsize` + `store(0)`/`load()` auf GROK-1-kompatible
`OnceLock<StdMutex<HashMap<String, usize>>>` umgestellt:

1. `FIND_DOCUMENTS_BY_ID_WRITER_FALLBACKS`
2. `CHANGED_DOCUMENTS_SINCE_WRITER_FALLBACKS`
3. `QUERY_WRITER_FALLBACKS`

**Schluessel:** `database_key_for_path(&self.database_path)` (= `path.to_string_lossy()`),
am Inkrementpunkt in `with_read_connection` vorhanden und pro Test-Tempdir
eindeutig.

**API (cfg(test)):**
- `record_writer_fallback(cell, database_path)`
- `reset_writer_fallback_count(cell, database_path)`
- `writer_fallback_count(cell, database_path) -> usize`

Lock-Zugriff ist vergiftungstolerant:
`.lock().unwrap_or_else(|poisoned| poisoned.into_inner())`
(wie GROK-1 / CHAT_TRACKING_BATCH_DOCUMENT_LOOKUPS).

Assertions in den drei betroffenen Tests bleiben **absolut `== 0`**, aber je
Datenbankpfad:
- `find_documents_by_id_file_backed_uses_read_only_connection`
- `query_fallback_does_not_wait_for_writer_mutex`
- `changed_documents_since_file_backed_uses_read_only_connection`

Keine Assertion geschwaecht, kein `--test-threads=1`.

---

## ursache_belegt

Gemessen in GROK-2 (`/Volumes/tmp/ctox-pipeline/reports/grok-2-frisch.md`):

| Zaehler | Pattern | Parallelitaet |
|---|---|---|
| FIND/QUERY/CHANGED `*_WRITER_FALLBACKS` | global `store(0)` + absolut `== 0` | ja — 38 Tests im Modul `storage::sqlite::instance::tests`, breiter Filter laesst sie parallel laufen |
| `DESKTOP_FILE_CHUNK_COMPLETENESS_CHECKS` | global `store(0)` + absolut `== 0` | ja — zwei Tests resetten denselben Atomic |

Writer-Fallbacks werden bei *jedem* fehlgeschlagenen Read-Only-Open inkrementiert
(typisch `:memory:`-DBs). Ein parallel laufender In-Memory-Test kann den globalen
Zaehler zwischen Reset und Assertion des file-backed-Opfers erhoehen → flaky rot.

---

## tests

Umgebungsvariablen:
```
CARGO_TARGET_DIR=/Volumes/tmp/ctox-pipeline-targets/GROK-3
TMPDIR=/Volumes/tmp/grok3-rust-tmp
```

| Befehl | Ergebnis |
|---|---|
| `cargo fmt --check` | gruen |
| `cargo fmt --check --manifest-path src/core/rxdb/Cargo.toml` | gruen |
| `cargo check --bin ctox --tests` | gruen (Finished, ~12 min nach Cache; kalter Teil laenger) |
| `cargo test --bin ctox desktop_file_chunk` | **4 Treffer**, 3× gruen (71s / 69s / 62s) |
| `cargo test --manifest-path src/core/rxdb/Cargo.toml writer_fallback` | **0 Treffer** (Name steckt nicht in den Testnamen) |
| Ersatzfilter `file_backed_uses_read_only` | 2 Treffer (FIND + CHANGED) — gruen |
| Ersatzfilter `query_fallback_does_not_wait` | 1 Treffer (QUERY) — gruen |
| `cargo test --manifest-path src/core/rxdb/Cargo.toml` (voller Crate) | **338 + 31 + 1 + 1 = 371** Tests gruen, 0 failed |

**desktop_file_chunk Treffer (4):**
- `desktop_file_chunk_cache_quota_evicts_active_rematerializable_file`
- `desktop_file_chunk_cache_quota_keeps_eager_scan_root_file_pinned`
- `desktop_file_chunk_cleanup_uses_primary_key_range_plan`
- `desktop_file_chunk_completion_uses_primary_chunk_ids`

**Rotmenge gegen origin/main (beide Richtungen):** voller rxdb-Crate-Lauf
nach dem Patch ist 0-failed. Es gibt keine neuen roten Tests und keine
entfernten. (Worktree startete auf dem gleichen Commit-Inhalt wie der
Task-Snapshot; Aenderung ist rein cfg(test)-Partitionierung.)

---

## gegenprobe

**Gewaehlt:** `FIND_DOCUMENTS_BY_ID_WRITER_FALLBACKS` (haeufigster
Read-Pfad; 38 parallele Nachbartests im Modul).

**Vorgehen (temporaer, exakt zurueckgebaut):**
1. FIND-Zaehler auf globales `AtomicUsize` zurueckgestellt (QUERY/CHANGED blieben partitioniert).
2. Temporaerer Produzenten-Test `file_backed_gegenprobe_inflates_global_find_fallback`
   (teilt Filter `file_backed`, oeffnet `:memory:`, ruft 3s lang
   `find_documents_by_id` → Writer-Fallback → global +1).
3. Opfer-Test mit 800 ms Sleep zwischen `store(0)` und Assertion, um das Race-Fenster zu oeffnen.
4. 5 Laeufe unter Filter `file_backed`:

| Lauf | Opfer FIND |
|---|---|
| 1 | **FAILED** (global inflated) |
| 2 | **FAILED** |
| 3 | ok (Race verfehlt) |
| 4 | ok |
| 5 | **FAILED** |

→ Flake unter breitem Filter **belegt** (3/5 rot). Ursache: paralleler
In-Memory-Fallback erhoeht den globalen Atomic zwischen Reset und absolutem
`assert_eq!(…, 0)`.

5. Exakt zurueckgebaut aus Snapshot `/tmp/grok3-instance-partitioned.rs`.
   `git diff --stat` danach wieder nur die Partitionierung:
   `src/core/rxdb/src/storage/sqlite/instance.rs | 94 ++++++++++++++++++++++++----`
   (keine TEMP-Reste, kein Produzententest).

**Nebenbefund waehrend Gegenprobe:** der unpartitionierte globale Runtime-Zaehler
`SQLITE_READ_ONLY_OPEN_CALLS` (Produktionsmetrik, nicht Teil dieses Auftrags)
zeigt unter dem kuenstlichen Produzenten ebenfalls Races in
`file_backed_reads_reuse_cached_read_only_connection` (left 15 / right 14).
Das bestaetigt das Parallelitaetsmodell, gehoert aber **nicht** zu den vier
Zielzaehlern.

---

## verblieben

### `DESKTOP_FILE_CHUNK_COMPLETENESS_CHECKS` — **unverändert**

**Begruendung (kein natuerlicher Schluessel am Inkrementpunkt innerhalb der Whitelist):**

- Inkrement in `desktop_file_chunk_generation_is_complete(database, …)`
  (`desktop_files.rs:465`).
- Am Inkrementpunkt verfuegbar: `database: &Arc<RxDatabase>`.
- `database.name` ist fuer den Produktionspfad der betroffenen Tests
  (`sync_desktop_files_from_workspace_root` / `sync_desktop_file_from_path`
  → `open_database`) **immer** der feste String `"ctox-business-os"`.
  Damit sind parallele Tests mit eigenen Tempdir-Roots **nicht** trennbar.
- `database.token` ist pro Open unique, aber den Tests nicht bekannt
  (sie halten nur `root: tempfile::TempDir`, oeffnen/schliessen die DB intern).
- Der DB-Pfad steckt in `database.storage` als `Arc<dyn RxStorage>` —
  ohne Downcast/Trait-Erweiterung (ausserhalb der Whitelist und ausserhalb
  des RxDB-Trait-Contracts) nicht erreichbar.
- Assertions (`store(0)`/`load() == 0`) liegen in
  `src/core/business_os/rxdb_peer.rs` (Tests), was **nicht** auf der Whitelist
  steht. Partitionierung nur am Inkrement ohne Test-Update waere nutzlos;
  Test-Update waere Whitelist-Bruch.
- Signature-Aenderung (`root`/`path` zusaetzlich durchreichen) wuerde
  Call-Sites in `desktop_files.rs` *und* `rxdb_peer.rs` beruehren → Whitelist-Bruch.

Gemäss Auftrag: **GENAU DIESE Stelle unveraendert**, gemeldet unter `verblieben`.

---

## offene_bedenken

1. **Desktop-Zaehler bleibt flaky** unter breitem Filter, solange nicht
   Whitelist erweitert wird (mindestens `rxdb_peer.rs` fuer Test-Resets, und/oder
   ein Pfad-Export am `RxDatabase`/`RxStorage`). Empfohlener Folgeschritt:
   Schluessel = `store::rxdb_store_path(root)` auf der Testseite und denselben
   Pfad am Inkrement aus `database.storage` (nach kontrollierter Trait-Erweiterung
   oder einem internen Test-Hook).
2. Filtername `writer_fallback` trifft 0 Tests — die echten Namen enthalten
   `file_backed_uses_read_only_connection` bzw. `query_fallback_does_not_wait`.
3. In-Memory-DBs teilen den Schluessel `":memory:"`. Mehrere parallele
   In-Memory-Tests wuerden denselben Bucket teilen. Die *assertierenden*
   Tests sind alle file-backed mit unique Tempdir-Pfaden, daher unkritisch
   fuer die aktuellen Assertions.
4. `SQLITE_READ_ONLY_OPEN_CALLS` und aehnliche globalen Runtime-Metriken mit
   absoluten Diff-Assertions in Tests bleiben ein separates Parallelitaetsrisiko
   (nicht Teil dieses Auftrags).
5. Nicht committed, wie gefordert.

