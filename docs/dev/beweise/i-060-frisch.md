# I-060 Report

## was_geaendert

- `RuntimeModuleCollectionEntry` traegt jetzt die aus der kanonischen `collections.schema.json` gelesenen `migration_strategies` als nach Zielversion indizierte Map. `schema.js` wird nicht ausgewertet.
- Der native Bring-up laeuft jetzt in der Reihenfolge: Datenbank oeffnen -> Collections registrieren/Target-Tabellen materialisieren -> Runtime-Altversionen migrieren und verifizieren -> bestehenden Stale-Version-Sweep ausfuehren -> Replikation hochfahren.
- `migrate_additive_native_rxdb_collection_versions` migriert jede nichtleere Runtime-Quelltabelle ueber alle Zwischenschritte bis zur registrierten Zielversion. `operations: []` ist Identity; `set_from_first_truthy` und `set_boolean` verwenden den vorhandenen nativen Executor.
- Jede Quellzeile wird transaktional in die Zielversion geschrieben. Eine strikt neuere Zielzeile gewinnt; gleich alte oder aeltere Zielzeilen werden deterministisch ersetzt. Ein Wiederanlauf ist dadurch idempotent.
- Nach jedem Tabellen-Schritt wird per ID/`lastWriteTime` verifiziert, dass kein Quelldokument im Ziel fehlt oder dort aelter ist. Erst nach erfolgreicher Verifikation committet der Schritt.
- Nichtleere Alt-Tabellen ohne vollstaendige Strategiekette brechen den Bring-up mit Collection, Quell-/Zielversion und dem fehlenden `migration_strategies.<collection>.<step>` ab.
- Leere Alt-Tabellen brauchen keine Strategiekette und werden vom nachgelagerten Sweep toleriert entfernt.
- Der bestehende Sweep kennt jetzt neben statischen auch Runtime-Collections und deren deklarierte Zielversion.
- Neue Waechter:
  - `runtime_installed_declarative_migration_is_discovered_and_copied`
  - `runtime_migration_without_strategy_retains_old_table_and_fails_closed`
  - `additive_thread_schema_migration_copies_and_verifies_before_cleanup`
  - zusaetzlich `runtime_empty_legacy_table_without_strategy_is_tolerated_and_cleaned`

## ursache_belegt

- Vorher lag der loeschende Aufruf von `repair_stale_rxdb_collection_schema_versions` vor `add_collections_tolerant`; jetzt steht die Migration nach der Registrierung bei `src/core/business_os/rxdb_peer.rs:2419`, der Sweep folgt bei `:2421`.
- Der Parser traegt Strategien jetzt im Runtime-Eintrag (`src/core/business_os/rxdb_peer.rs:13417`) und liest sie ueber `runtime_module_migration_strategies_for_collection` (`:13497`).
- Der zuvor nur im Test benutzte deklarative Executor wird nun aus dem produktiven Migrationspfad `migrate_additive_native_rxdb_collection_versions` (`:13852`) erreicht.
- Die Rot-Gegenproben zeigen separat, dass fehlender Copy/Verify und fehlendes Fail-closed von den benannten Waechtern erkannt werden.

## kompensationen_geloescht

- Keine. Der Auftrag verlangte, `repair_stale_rxdb_collection_schema_versions` zu behalten und erst nach erfolgreicher Migration laufen zu lassen.

## verblieben

- `repair_stale_rxdb_collection_schema_versions` bleibt als eigentliche Cleanup-Phase fuer bereits migrierte bzw. leere Alt-Tabellen und alte Trigger bestehen. Es kompensiert nicht mehr den fehlenden Migrator, sondern wird erst nach Copy+Verify freigegeben.
- Das bestehende Verhalten, einen Fehler des nachgelagerten Sweeps zu protokollieren und den Peer weiterzubringen, bleibt unveraendert. Ein solcher Fehler laesst Alt-Tabellen stehen, loescht aber keine unverifizierten Quelldaten.

## tests

Alle Cargo-Aufrufe verwendeten `CARGO_TARGET_DIR=/Volumes/tmp/ctox-pipeline-targets/I-060`.

- `cargo fmt --check` — gruen; Format-Check erzeugt keine `test result`-Zeile.
- `cargo check --bin ctox` — gruen: `Finished dev profile ... in 29.80s`; Compile-only, daher keine `test result`-Zeile.
- `cargo check --bin ctox --tests` — gruen: `Finished dev profile ... in 39.22s`; Compile-only, daher keine `test result`-Zeile.
- `cargo test --bin ctox runtime_installed_declarative_migration`
  - `test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 2735 filtered out; finished in 0.03s`
- `cargo test --bin ctox runtime_migration_without_strategy`
  - `test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 2735 filtered out; finished in 0.03s`
- `cargo test --bin ctox native_declarative_migration_matches_browser_operations`
  - `test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 2735 filtered out; finished in 0.00s`
- `cargo test --bin ctox additive_thread_schema_migration_copies_and_verifies_before_cleanup`
  - `test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 2735 filtered out; finished in 0.06s`
- `cargo test --bin ctox runtime_empty_legacy_table_without_strategy`
  - `test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 2735 filtered out; finished in 0.05s`

Es wurden keine verbotenen breiten Filter und kein ungefilterter Testlauf verwendet.

## gegenprobe

### Copy/Verify ausgelassen

Im kompilierbaren Mutanten wurde der Source-to-Target-Write ausgelassen; die Verifikation blieb aktiv.

- `runtime_installed_declarative_migration_is_discovered_and_copied`:
  - `test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 2735 filtered out; finished in 0.04s`
  - Fehler: `1 source id(s) are absent or older`.
- `additive_thread_schema_migration_copies_and_verifies_before_cleanup`:
  - `test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 2735 filtered out; finished in 0.03s`
  - Fehler: `1 source id(s) are absent or older`.

### Fail-closed durch Weiterlaufen ersetzt

Im kompilierbaren Mutanten wurde ein fehlender Schritt absichtlich als Identity behandelt.

- `runtime_migration_without_strategy_retains_old_table_and_fails_closed`:
  - `test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 2735 filtered out; finished in 0.15s`
  - Der Waechter sah unerwarteten Erfolg mit `migrated_tables=1`, `migrated_rows=1`, `verified_rows=1`.

### Exakter Rueckbau

- Vor den Mutanten wurde die finale Datei unter `/tmp/i-060-rxdb-peer-final.rs` gesichert.
- Nach jeder Gegenprobe wurde exakt zurueckgebaut.
- SHA-256 danach auf beiden Dateien: `7f5ad7faad19b827c901975c4fea96927ea625d19402196f7304c0d0bc315690`.
- Finales `git diff --stat`: `src/core/business_os/rxdb_peer.rs | 725 ...`, insgesamt `698 insertions(+), 27 deletions(-)`; nur die Whitelist-Datei ist geaendert.

### Rot-Mengen gegen `origin/main`, beide Richtungen

- `origin/main` enthaelt von den vier benannten Migrationswaechtern nur `native_declarative_migration_matches_browser_operations` (Quellzeile 16320) und keine der drei neuen Lifecycle-Waechter. Der bestehende Paritaetswaechter ist gruen; relevante origin/main-Rot-Menge: `{}`.
- Copy-Mutant-Rot-Menge: `{runtime_installed_declarative_migration_is_discovered_and_copied, additive_thread_schema_migration_copies_and_verifies_before_cleanup}`.
  - Mutant minus origin/main: beide genannten Tests.
  - origin/main minus Mutant: `{}`.
- Fail-open-Mutant-Rot-Menge: `{runtime_migration_without_strategy_retains_old_table_and_fails_closed}`.
  - Mutant minus origin/main: der genannte Test.
  - origin/main minus Mutant: `{}`.
- Die finalen Gruenlaeufe nach dem exakten Rueckbau belegen, dass keine Mutantenfassung im Arbeitsbaum verblieben ist.

## offene_bedenken

- Keine Aenderung in `src/core/rxdb/src/` war erforderlich; der Versionsuebergang konnte mit den vorhandenen SQLite-Tabellen- und Registrierungsvertraegen in `rxdb_peer.rs` umgesetzt werden.
- Der erste Testprofil-Aufbau war sehr lang, endete aber erfolgreich. Es bestehen viele vorbestehende Compiler-Warnungen ausserhalb der Whitelist; keine davon wurde fuer I-060 angefasst.
- Die Doku-Markierungen `NOT IMPLEMENTED` wurden auftragsgemaess nicht entfernt.

## pfade

- Geaendert: `src/core/business_os/rxdb_peer.rs`.
- Keine weiteren Dateien erforderlich.
- Insbesondere keine Aenderung unter `src/core/rxdb/src/` und keine Browser-/Dist-Datei erforderlich.
