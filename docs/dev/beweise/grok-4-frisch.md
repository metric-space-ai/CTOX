# GROK-4 Report — Desktop-Chunk-Zaehler wird partitioniert

**Arbeitsbaum:** `/Volumes/tmp/ctox-grok4` (Worktree auf `origin/main`)
**HEAD Basis:** `22cf6406c` (clean vor Aenderung; Aenderungen uncommittet)
**Datum:** 2026-08-05
**Whitelist-Dateien:**
- `src/core/business_os/desktop_files.rs` — **geaendert**
- `src/core/business_os/rxdb_peer.rs` — **geaendert**

---

## was_geaendert

`git diff --stat`:
```
 src/core/business_os/desktop_files.rs | 53 ++++++++++++++++++++++++++++++++---
 src/core/business_os/rxdb_peer.rs     | 20 +++++++++----
 2 files changed, 63 insertions(+), 10 deletions(-)
```

### `DESKTOP_FILE_CHUNK_COMPLETENESS_CHECKS`

Vom Muster `AtomicUsize` + `store(0)`/`load()` auf GROK-1/2/3-kompatible
`OnceLock<Mutex<HashMap<String, usize>>>` umgestellt.

**Schluessel:** `store::rxdb_store_path(root).to_string_lossy()` —
pro Test-Tempdir-Root eindeutig und an Inkrement- sowie Reset-/Assert-Stellen
verfuegbar.

**API (cfg(test)):**
- `record_desktop_file_chunk_completeness_check(root)`
- `reset_desktop_file_chunk_completeness_checks(root)`
- `desktop_file_chunk_completeness_check_count(root) -> usize`

Lock-Zugriff ist vergiftungstolerant:
`.lock().unwrap_or_else(|poisoned| poisoned.into_inner())`
(wie CHAT_TRACKING_BATCH_DOCUMENT_LOOKUPS / Writer-Fallback-Zaehler).

### Signatur-Durchreichung

`desktop_file_chunk_generation_is_complete` bekommt zusaetzlich `root: &Path`.
Am Inkrementpunkt war vorher nur `database: &Arc<RxDatabase>` — und
`database.name` ist im Produktionspfad immer der feste String
`"ctox-business-os"`. Der Root ist in den beiden Call-Sites innerhalb
`upsert_desktop_file_with_parent` (Whitelist) bereits vorhanden.

**Call-Site-Zaehlung vor der Aenderung (alle in Whitelist):**
1. `desktop_files.rs` — zwei Aufrufe in `upsert_desktop_file_with_parent`
2. `rxdb_peer.rs` — zwei Aufrufe im Test
   `desktop_file_chunk_completion_uses_primary_chunk_ids`
3. Keine Call-Sites ausserhalb der Whitelist → kein STOPP.

In non-test Builds: `let _ = root;` (kein Dead-Code-Warn).

### Test-Resets/Asserts

`rescan_of_unchanged_workspace_is_a_no_op` und
`materialized_large_file_survives_lazy_rescan` nutzen jetzt
`reset_…(root.path())` / `…_count(root.path())`.

Assertions bleiben **absolut `== 0`**, aber je Store-Pfad. Kein Test
geschwaecht, kein `--test-threads=1`.

Keine Aenderung an `src/core/rxdb/` (kein Trait-/Storage-Export noetig).

---

## ursache_belegt

Gemessen in GROK-2 (`/Volumes/tmp/ctox-pipeline/reports/grok-2-frisch.md`) und
in GROK-3 als `verblieben` dokumentiert:

| Zaehler | Pattern | Parallelitaet |
|---|---|---|
| `DESKTOP_FILE_CHUNK_COMPLETENESS_CHECKS` | global `store(0)` + absolut `== 0` | ja — zwei Tests resetten denselben Atomic; jeder andere Aufruf von `desktop_file_chunk_generation_is_complete` stoert |

Unter breitem Filter koennen parallele Schwester-Tests denselben globalen
Zaehler zwischen Reset und Assertion erhoehen → flaky rot.

---

## tests

Umgebungsvariablen:
```
CARGO_TARGET_DIR=/Volumes/tmp/ctox-pipeline-targets/GROK-4
TMPDIR=/Volumes/tmp/grok4-rust-tmp
```

| Befehl | Ergebnis |
|---|---|
| `cargo fmt --check` | gruen |
| `cargo check --bin ctox --tests` | gruen (Finished, kalter Bau ~12m 33s) |
| `cargo test --bin ctox desktop_file_chunk` Lauf 1 | **4 Treffer**, gruen (89.80s) |
| `cargo test --bin ctox desktop_file_chunk` Lauf 2 | **4 Treffer**, gruen (82.62s) |
| `cargo test --bin ctox desktop_file_chunk` Lauf 3 | **4 Treffer**, gruen (86.15s) |
| Nach Gegenprobe-Restore erneut | **4 Treffer**, gruen (353.19s, inkl. Rebuild) |

**desktop_file_chunk Treffer (4):**
- `desktop_file_chunk_cache_quota_evicts_active_rematerializable_file`
- `desktop_file_chunk_cache_quota_keeps_eager_scan_root_file_pinned`
- `desktop_file_chunk_cleanup_uses_primary_key_range_plan`
- `desktop_file_chunk_completion_uses_primary_chunk_ids`

(Die beiden absoluten Assertions leben in
`rescan_of_unchanged_workspace_is_a_no_op` und
`materialized_large_file_survives_lazy_rescan` — nicht im Filter
`desktop_file_chunk`, aber vom Patch betroffen und in der Gegenprobe
mitgeprueft.)

---

## gegenprobe

**Vorgehen (temporaer, exakt zurueckgebaut aus Snapshot):**

1. Zaehler auf globales `AtomicUsize` zurueckgestellt (kompilierbar, gleiche
   Reset/Count-API mit ignoriertem Root).
2. Temporaerer Produzenten-Test
   `desktop_file_gegenprobe_inflates_global_completeness_checks`
   (120s lang `fetch_add(1)` alle 5 ms).
3. Opfer-Tests mit Race-Fenster am Anfang (Reset → Sleep 2s → absolut
   `assert_eq!(count, 0)`), bevor der langsame First-Scan laeuft.
4. Gemeinsamer Filter `desktop_file_gegenprobe` (3 Tests, `--test-threads=3`).

**Ergebnis Lauf 1 (empirisch rot):**
```
running 3 tests
desktop_file_gegenprobe_victim_rescan_of_unchanged_workspace ... FAILED
desktop_file_gegenprobe_victim_materialized_large_file ... FAILED
desktop_file_gegenprobe_inflates_global_completeness_checks ... ok

assertion `left == right` failed: TEMP GEGENPROBE: global counter must stay zero while producer runs
  left: 298
 right: 0
(and same for the second victim)

test result: FAILED. 1 passed; 2 failed; finished in 120.00s
```

→ Flake unter parallelem Filter **belegt**: der globale Atomic wird zwischen
Reset und absoluter Null-Assertion vom Produzenten aufgeblasen
(298 statt 0).

5. **Exakt zurueckgebaut** aus Snapshot:
   - `/tmp/grok4-desktop_files-partitioned.rs`
   - `/tmp/grok4-rxdb_peer-partitioned.rs`
   `git diff --stat` danach wieder nur die Partitionierung:
   ```
   src/core/business_os/desktop_files.rs | 53 ++++++++++++++++++++++++++++++++---
   src/core/business_os/rxdb_peer.rs     | 20 +++++++++----
   2 files changed, 63 insertions(+), 10 deletions(-)
   ```
   Keine TEMP-Reste, kein Produzententest, Original-Testnamen wiederhergestellt.
   Partitionierte Tests nach Restore erneut gruen.

**Nebenbefund zur Filterwahl:** der Auftragstext nennt den breiten Filter
`desktop_file`. Die beiden Opfer-Testnamen
(`rescan_of_unchanged_workspace_is_a_no_op`,
`materialized_large_file_survives_lazy_rescan`) enthalten das Substring
`desktop_file` **nicht**. Empirische Belegung lief deshalb unter dem
kuenstlichen gemeinsamen Praefix `desktop_file_gegenprobe` (nach temporaerer
Umbenennung). Das Parallelitaetsmodell ist identisch: absolute Assertion auf
einem prozessglobalen Zaehler, den parallele Threads inflatieren.

---

## verblieben

Nichts aus dem Auftragsumfang. `DESKTOP_FILE_CHUNK_COMPLETENESS_CHECKS` ist
partitioniert. Die in GROK-2/GROK-3 als flaky-verdaechtig markierten fuenf
Zaehler sind damit alle behandelt:
1. Chat-Tracking-Lookups — 22cf6406c
2–4. Writer-Fallbacks — 163fb78e4
5. Desktop-Chunk-Completeness — dieser Patch

---

## offene_bedenken

1. Die Opfer-Testnamen enthalten `desktop_file` nicht; unter dem breiten
   Filter `desktop_file` laufen sie nativ nicht mit. Das aendert nichts an der
   Parallelitaetsgefahr (andere Tests im Binary koennen
   `desktop_file_chunk_generation_is_complete` anstossen), ist aber fuer die
   Testauswahl relevant.
2. `root` wird in non-test Builds nur fuer die Signatur mitgeschleppt
   (`let _ = root;`). Alternative waere gewesen, den Store-Pfad aus
   `database.storage` zu ziehen — dafuer haette es Trait-Erweiterung in
   `src/core/rxdb/` gebraucht (Whitelist-Bruch → laut Auftrag STOPP).
3. In-Memory-/Sonderpfade: der Schluessel ist immer
   `root/runtime/business-os-rxdb.sqlite3`. Parallel laufende Tests mit
   verschiedenen Tempdir-Roots bleiben getrennt.
4. Nicht committed, wie gefordert. Keine Subagenten.
