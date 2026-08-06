# I-054 — Runde 1 (reine Messung)

## was_geaendert

- Im Repository wurde absichtlich keine fachliche Datei geändert.
- Angelegt wurden nur Mess-/Build-Artefakte außerhalb des Repositories unter `/tmp` und `/Volumes/tmp`.
- `cargo check` hat im geteilten Checkout während der Messung `Cargo.lock` aufgelöst; die von diesem Lauf identifizierte einzelne Ergänzung wurde unmittelbar zurückgebaut. Ein parallel laufender fremder Cargo-Prozess hat dieselbe Zeile danach erneut eingetragen; der finale Baum ist deshalb gegenüber dem Gesprächsbeginn bei `Cargo.lock` um eine Diff-Zeile weitergelaufen. Index und fremde Änderungen wurden nicht angefasst.

## ursache_belegt

### 1. Wo der Bump wirksam wird und was mit der alten Tabelle geschieht

1. Es gibt keinen zentralen „Version erhöhen“-Vorgang. Die Versionsnummer ist ein Literal im jeweiligen JSON-Schema. Runtime-Module werden aus `collections.schema.json` gelesen; der Parser übernimmt `version` unverändert und ergänzt nur dann `0`, wenn das Feld fehlt (`src/core/business_os/rxdb_peer.rs:13926-14049`).
2. `RxDatabase::add_single_collection` schreibt für die neue Version ein neues internes Metadokument und öffnet danach genau eine Storage-Instanz für dieses Schema (`src/core/rxdb/src/rx_database.rs:338-369`). Der Metaschlüssel ist absichtlich versionsabhängig (`src/core/rxdb/src/rx_database.rs:669-684`; `src/core/rxdb/src/rx_database_internal_store.rs:473-478`).
3. SQLite leitet den Tabellennamen direkt aus `params.schema.version` ab (`src/core/rxdb/src/storage/sqlite/index_mod.rs:21-40`; `src/core/rxdb/src/storage/sqlite/sql.rs:57-67`). Dadurch erzeugt ein Literalwechsel von 0 auf 1 eine neue Tabelle `...__v1`.
4. Beim Anlegen dieser neuen Storage-Instanz passiert mit `...__v0` nichts: kein Copy, kein Verify, kein Drop. Der vollständige native Migration-Plugin ist ausdrücklich „out of scope“/nur Stub (`src/core/rxdb/src/plugins/migration_schema/mod.rs:1-5`); `migration_needed`, `start_migration` und `migrate_promise` liefern nur `PLUGIN_MISSING` (`src/core/rxdb/src/rx_collection.rs:1013-1028`).
5. Das ist keine dokumentierte Rückwärtsmigrations-/Rollback-Strategie. Der einzige einschlägige Kommentar begründet nur getrennte Metadokumente pro Version (`src/core/rxdb/src/rx_database_internal_store.rs:473-478`), während der Migration-Kommentar explizit eine nicht implementierte Laufzeit beschreibt. Das explizite Collection-Remove entfernt alle bekannten Versionen (`src/core/rxdb/src/rx_collection_helper.rs:110-178`); eine Aufbewahrung für Rollback wird nirgends genannt oder getestet.
6. Browserseitig werden `migrationStrategies` zwar an die Collection-Definition angehängt (`src/apps/business-os/app.js:5201-5253`), aber die CTOX-RxDB-Laufzeit hat nur einen Platzhalter (`src/apps/business-os/rxdb/src/rx-database.mjs:62-64`) und `addCollections` liest `migrationStrategies` überhaupt nicht (`src/apps/business-os/rxdb/src/rx-database.mjs:124-145`). Der Guard beweist nur, dass JSON-Strategien kompilierbar sind (`src/apps/business-os/scripts/assert-declarative-migrations.mjs:19-31,87-99`), nicht dass eine Datenmigration ausgeführt wird.
7. Der App-Starter verschärft die Lücke: `collections.schema.json.tpl` erzeugt Schema v1 plus Identity-Strategie (`src/apps/business-os/app-starter/v2/collections.schema.json.tpl:1-27`), `schema.js.tpl` exportiert für dasselbe v1-Schema aber leere `migrationStrategies` (`src/apps/business-os/app-starter/v2/schema.js.tpl:1-19`).

**Ursache:** Ein Schema-Bump ist derzeit nur „neues versionsabhängiges Metadokument + neue leere Storage-Tabelle“. Die Migrationsdeklarationen werden validiert/weitergereicht, aber weder der native noch der Browser-Runtime führt den allgemeinen Migrationsvertrag aus. Die alte Tabelle bleibt nicht wegen einer Rückwärtsmigrationsentscheidung stehen, sondern weil der ausführende Migrationsschritt fehlt.

### 2. Warum genau die vier sellify-Sammlungen zwei Versionen tragen

Die gemessene Historie ist keine Hypothese:

- `runtime/ctox_service.log:14162` protokolliert: **238.913 Rows über genau 4 additive Schema-Upgrades migriert**.
- Direkt danach wurden 4 alte Tabellen und 12 Trigger entfernt (`runtime/ctox_service.log:14163`). Das war eine kurzzeitig laufende, ausgelöste Migration und als solche legitim.
- Unmittelbar danach erkannte der Peer erneut geänderte Runtime-Schemata und startete kontrolliert neu (`runtime/ctox_service.log:14169-14170`; der heutige Watcher liegt in `src/core/business_os/rxdb_peer.rs:2836-2847`). Also: **kein Bump ohne Neustart**.
- Die SQLite-Schema-Reihenfolge zeigt, dass v1 zuerst existierte und v0 später neu angelegt wurde: `sqlite_schema.rowid` v1 = 711/713/715/717, v0 = 719/721/723/725.
- Die heute installierten Sellify-Dateien deklarieren diese vier Sammlungen wieder als v0 (`runtime/business-os/installed-modules/sellify/collections.schema.json:98,213,350,485`; Browser-Zwilling `schema.js:97,212,349,484`), enthalten aber Identity-Strategien zum Ziel v1 (`collections.schema.json:763-775`). `sellify_records` bleibt dagegen v1 (`collections.schema.json:741-760`). Das erklärt exakt, warum nur die vier großen Datenkollektionen doppelt sind.

Live-Zeilenzahlen:

| Sammlung | v0 | v1 |
|---|---:|---:|
| `sellify_activities` | 0 | 74.209 |
| `sellify_campaigns` | 0 | 86.549 |
| `sellify_companies` | 0 | 17.516 |
| `sellify_people` | 0 | 60.639 |
| **Summe** | **0** | **238.913** |

Die vier v0-Tabellen sind damit leere, später erzeugte Trümmer; die v1-Tabellen enthalten sämtliche Nutzdaten. Die v0-Trümmer plus 48 leere Indizes belegen 212.992 Byte (52 SQLite-Seiten à 4096 Byte); zusätzlich existieren 12 v0-Change-Trigger.

Der heutige Sweep scheitert an diesen vier nicht; er betrachtet sie gar nicht:

- Er iteriert ausschließlich `business_os_collections()` (`src/core/business_os/rxdb_peer.rs:14370-14377`).
- Diese Liste stammt ausschließlich aus dem statischen `business_os_schema_contract.json` (`src/core/business_os/rxdb_peer.rs:14224-14245`); alle vier Sellify-Namen fehlen dort (DB/JSON-Messung: 4× `false`). Runtime-Collections werden separat entdeckt (`src/core/business_os/rxdb_peer.rs:13740-13782`).
- Auch der öffentliche Einzelsammlungs-Repair lehnt nicht-statische Namen als unbekannt ab (`src/core/business_os/rxdb_peer.rs:14330-14331`).
- Selbst bei einem internen Aufruf würde `expected_rxdb_collection_version` für unbekannte Sammlungen auf 0 fallen (`src/core/business_os/rxdb_peer.rs:14558-14564`), während `active_rxdb_collection_version` die höchste Metaversion 1 liefert (`src/core/business_os/rxdb_peer.rs:14566-14589`). Der Guard `active_version != expected_version` liefert dann absichtlich keine stale tables (`src/core/business_os/rxdb_peer.rs:14592-14601`). Das verhindert hier zurecht, dass die 238.913 v1-Zeilen als vermeintlicher Müll gelöscht werden.

### 3. Idempotenz und heutige Kosten

- Physisch ist der Sweep idempotent: Er ermittelt zuerst die noch vorhandenen Fremdversionen und verwendet danach `DROP ... IF EXISTS` (`src/core/business_os/rxdb_peer.rs:14471-14491`). Nach erfolgreichem Drop ist die nächste stale-Liste leer. Er entfernt allerdings die alten internen Metadokumente nicht; deshalb ist die Idempotenz keine vollständige Versions-Lifecycle-Semantik.
- Der aktuelle Live-Store enthält 3.291 `sqlite_master`-Objekte: 372 Tabellen, 681 Trigger, 2.238 Indizes. Der statische Sweep prüft 178 Collections.
- Read-only Benchmark der heutigen Statement-Folge gegen dieselbe Live-DB, 20 Wiederholungen nach Warm-up: min 282,013 ms; Median 349,554 ms; Mittel 367,673 ms; p95 445,440 ms; max 469,563 ms. Ergebnis jedes Laufs: 0 Tabellen/0 Trigger zu reparieren.
- Emulation des früheren „eine Connection pro Collection“-Musters, 3 Wiederholungen: 1.515,563 / 1.479,331 / 1.488,207 ms; Median 1.488,207 ms. Die heutige eine Connection ist auf dieser Maschine im Warm-Cache rund 4,3× günstiger.
- Die Behauptung „many minutes“ in `src/core/business_os/rxdb_peer.rs:14355-14359` ließ sich auf dem heutigen lokalen Warm-Cache **nicht** reproduzieren. Die Richtung ist belegt, die historische Größenordnung nicht. Der heutige Sweep kostet trotzdem messbare ca. 0,35 s pro Peer-Start/Respawn und ist kein Nullkostenpfad.

**Bewertung:** Der Startup-Sweep ist nicht als Migrationswerkzeug gerechtfertigt. Eine an einen konkreten Versionsübergang gekoppelte Copy→Transform→Verify→Cleanup-Migration wäre legitim. Der heutige Sweep ist statisch, läuft vor Collection-Registrierung (`src/core/business_os/rxdb_peer.rs:2390-2413`), kennt Runtime-Module nicht und hat keinen Migrationsauslöser. Er ist eine Kompensation.

## kompensationen_geloescht

- Keine; reine Messung.

## verblieben

- `repair_stale_rxdb_collection_schema_versions` (`src/core/business_os/rxdb_peer.rs:14342-14403`) bleibt unverändert.
- `repair_rxdb_collection_schema_version_drift` und `_with_connection` (`src/core/business_os/rxdb_peer.rs:14405-14541`) bleiben unverändert.
- Für die konkrete Live-DB darf v1 **nicht** gelöscht werden: Dort liegen 238.913 Zeilen. Es braucht zuerst eine bewusste Entscheidung „Sellify-Deklaration zurück auf v1“ oder eine echte v1→v0-Rückmigration. Die aktuellen leeren v0-Tabellen dürfen erst danach triggergebunden entfernt werden.
- Alte interne Collection-Metadokumente bleiben selbst nach dem Tabellen-Sweep bestehen.

## tests

- `CARGO_TARGET_DIR=/Volumes/tmp/ctox-pipeline-targets/I-054 cargo fmt --check` — Exit 0. Keine `test result`-Zeile, da Formatcheck; Trefferzahl nicht anwendbar.
- Geteilter Arbeitsbaum: `CARGO_TARGET_DIR=/Volumes/tmp/ctox-pipeline-targets/I-054 cargo check --bin ctox` — Exit 101; Rot-Menge `{E0432, E0053}` in den fremden/untracked CLIProxyAPI-Arbeiten (`src/core/execution/cliproxyapi_host.rs`), 71 Warnungen. Keine `test result`-Zeile, da Compile-Check.
- Sauberer HEAD-Snapshot außerhalb des Repos, mit dem für den Build erforderlichen ignorierten `pi-sidecar/dist/ctox-pi-sidecar.mjs`: derselbe `cargo check --bin ctox` — Exit 0, `Finished dev ... in 1m 35s`. Rot-Menge `{}`. Mengenvergleich: Arbeitsbaum−HEAD = `{E0432,E0053}`; HEAD−Arbeitsbaum = `{}`.
- Erster sauberer-HEAD-Testversuch mit Filter `rxdb_schema_drift_repair`: Build vor Testausführung durch SIGKILL beim `mime_guess`-Build beendet; keine `test result`-Zeile/keine Trefferzahl.
- Zweiter Versuch (`-j 2`): Build vor Testausführung durch `No space left on device` in Rust-TMP beendet; keine `test result`-Zeile/keine Trefferzahl.
- Dritter Versuch (`-j 1`, `TMPDIR=/Volumes/tmp/i054-rust-tmp`) war bei Berichtserstellung noch im erstmaligen Testprofil-Build; noch keine `test result`-Zeile. Filter ist spezifisch und zulässig; erwartet wird genau der Test `rxdb_schema_drift_repair_drops_stale_version_table_after_active_meta_upgrade` (`src/core/business_os/rxdb_peer.rs:16242-16365`).

## gegenprobe

- Entfällt laut Auftrag (reine Messung).
- Datenbank-Gegenbelege wurden read-only erhoben; keine Tabelle/Zeile wurde verändert.

## offene_bedenken

- Die exakten transienten Sellify-Schema-Dateiinhalte während des v1-Laufs sind nicht mehr auf dem Dateisystem vorhanden. Gesichert sind aber (a) das Log „238913 / 4 upgrades“, (b) der unmittelbar folgende Cleanup, (c) die spätere Schemaänderungs-Rekonfiguration, (d) die SQLite-Erstellreihenfolge und (e) Schemagleichheit v0↔v1 bis auf das Feld `version`.
- Der Kostenbenchmark bildet die SQL-Folge read-only in Python nach; er ruft die private Rust-Funktion nicht auf, damit der Live-Store garantiert nicht mutiert. Warm-Cache-Werte sind keine Cold-Boot-Garantie.
- Der geteilte Baum verändert sich parallel. Zu Beginn zeigte `git diff --stat` 33 Dateien, 1.800 Einfügungen/1.547 Löschungen; am Ende waren es 33 Dateien, 1.801/1.547. Die zusätzliche Lockfile-Zeile stammt aus parallel laufender Cargo-Arbeit, nicht aus einer fachlichen Änderung dieses Auftrags.

## pfade

Für eine Reparaturwelle werden mindestens diese Pfade benötigt:

1. `src/core/rxdb/src/plugins/migration_schema/mod.rs:1-5` und `src/core/rxdb/src/rx_collection.rs:1013-1028` — echten nativen Versionsmigrations-Lifecycle statt Stub bereitstellen **oder** die Business-OS-spezifische Migration vollständig vor der Storage-Umschaltung ausführen.
2. `src/core/rxdb/src/rx_database.rs:338-369,663-741` — Versionsübergang als explizites Ereignis erkennen; neue Meta/Tabelle nicht als vollständige Migration behandeln.
3. `src/core/business_os/rxdb_peer.rs:13740-14049,14342-14601` — Runtime-Module und `migration_strategies` in einen triggergebundenen Copy→Verify→Cleanup-Pfad integrieren; statischen Startup-Sweep danach entfernen. Keine neue Env-Konfiguration, kein HTTP-Pfad.
4. `src/apps/business-os/rxdb/src/rx-database.mjs:62-64,124-145` — `migrationStrategies` tatsächlich ausführen oder die irreführende Oberfläche entfernen. Wegen Bereichsregel anschließend `dist/ctox-rxdb-js.mjs` neu bauen und beide identischen Cache-Buster in `src/apps/business-os/shared/db.js` und `src/apps/business-os/shared/sync.js` bumpen.
5. `src/apps/business-os/app.js:5201-5253` und `src/apps/business-os/shared/declarative-migrations.js:1-80` — Strategieübergabe und echte Runtime-Ausführung zu einem Vertrag schließen.
6. `src/apps/business-os/app-starter/v2/collections.schema.json.tpl:1-27` und `src/apps/business-os/app-starter/v2/schema.js.tpl:1-19` — v1/Identity-Strategie auf beiden Deklarationsseiten konsistent erzeugen.
7. Konkrete Instanzkorrektur: `runtime/business-os/installed-modules/sellify/collections.schema.json:98,213,350,485,763-775` und `runtime/business-os/installed-modules/sellify/schema.js:97,212,349,484,763` — v0/v1-Absicht vereinheitlichen, ohne die 238.913 v1-Zeilen zu verlieren. Diese Runtime-Dateien sind Instanzdaten, kein allgemeiner Source-Fix.
