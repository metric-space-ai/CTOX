# I-056 Report

## was_geaendert

Nichts im Arbeitsbaum geaendert. Ich habe nach der vorgeschriebenen Whitelist-Regel gestoppt, bevor ich das Sicherheitsnetz oder eine halbe Migration angefasst habe.

Entscheidung: **Variante (a), `migrationStrategies` wirklich ausfuehren.**

Begruendung: Das Repository hat bereits zwei deklarative Strategieformen (Browser-Funktionen und JSON-DSL), versionierte persistente Daten sowie eine dokumentierte Fail-closed-Migrationsabsicht. Ein Entfernen nach Variante (b) waere nicht nur ein kleiner API-Cleanup: Es muesste die Strategieoberflaeche aus Shell, App-Validator, Starter, allen bestehenden Modulen und der nativen JSON-Seite entfernen und liesse die vorhandenen Versionswechsel ohne datenerhaltenden Lifecycle. Die richtige Reparatur ist deshalb ein vollstaendiger Lifecycle: Vorversion feststellen, alle Quellzeilen transformieren, Zielzeilen gegen das Zielschema verifizieren, den Versionsmarker atomar umschalten und erst danach Altbestand entfernen. Ich habe (a) nicht halb implementiert.

`git status --short`: leer.

`git diff --stat`: leer.

## ursache_belegt

- `src/apps/business-os/rxdb/src/rx-database.mjs:62-64` exportiert nur den Migration-Plugin-Platzhalter.
- `src/apps/business-os/rxdb/src/rx-database.mjs:124-145` liest bei `addCollections` `schema`, `conflictStrategy`, `deleteStrategy` und `syncProfile`, aber nicht `migrationStrategies`.
- `src/apps/business-os/app.js:1345-1352` und `:5218-5243` reichen `migrationStrategies` dennoch an `addCollections` weiter; `:5256-5272` baut gerade die vom Runtime-Code ignorierte Collection-Definition.
- Ein vollstaendiger Browser-Lifecycle ist nicht allein in der gewhitelisteten `rx-database.mjs` sauber implementierbar: `src/apps/business-os/rxdb/src/storage-indexeddb.mjs:15-18,49-59,414-466,1164-1184` hat weder einen persistenten Collection-Schemaversionsmarker noch eine Storage-Operation, die migrierte Zeilen und Versionsumschaltung atomar in derselben IndexedDB-Transaktion schreibt. Eine Strategie nach einem Crash einfach erneut auszufuehren waere bei beliebigen Funktionsstrategien nicht sicher und waere die verbotene halbe Variante (a).
- Die native Seite muss mitziehen: `src/core/business_os/rxdb_peer.rs:2390-2413` fuehrt den loeschenden Stale-Version-Repair **vor** Datenbankoeffnung und Collection-Registrierung aus; `:2420-2423` registriert erst danach die Zielsammlungen. `:13805-14014` nimmt beim Parsen von `collections.schema.json` keine `migration_strategies` in den Runtime-Collection-Eintrag auf. `:14093-14169` implementiert zwar das Anwenden einzelner JSON-Operationen, diese Helfer sind aber nicht in einen Zeilen-/Versions-Lifecycle verdrahtet. `:14342-14403` loescht anschliessend alte Tabellen/Trigger statt sie vorher zu migrieren und zu verifizieren.
- Der Guard belegt nur Uebersetzbarkeit/Beispielfunktionsausgabe: `src/apps/business-os/scripts/assert-declarative-migrations.mjs:19-31,87-99`. Er oeffnet keine Datenbank, registriert keine versionierte Collection und beobachtet keine durch das Bundle ausgefuehrte Migration.
- Der Starter widerspricht sich: `src/apps/business-os/app-starter/v2/collections.schema.json.tpl:21-27` deklariert die Identity-Migration nach v1, `src/apps/business-os/app-starter/v2/schema.js.tpl:18-19` exportiert fuer dasselbe v1-Schema leere Strategien.
- Zusaetzliche gemessene Deklarationsluecken, die ein JSON-kanonischer Lifecycle sichtbar macht:
  - `src/apps/business-os/modules/browser/schema.js:186-193` hat vier v1-Identity-Strategien, waehrend `src/apps/business-os/modules/browser/collections.schema.json:3-455,458-460` fuer dieselben vier v1-Schemas kein `migration_strategies` enthaelt.
  - `src/apps/business-os/modules/credentials/schema.js:31-38` hat `business_commands.1`, waehrend `src/apps/business-os/modules/credentials/collections.schema.json:3-50,53-55` die v1-Strategie nicht enthaelt.
  - Umgekehrt enthaelt `src/apps/business-os/modules/creator/collections.schema.json:53-68` `business_commands.1`, waehrend `src/apps/business-os/modules/creator/schema.js:1-23` gar keinen `migrationStrategies`-Export hat.

## kompensationen_geloescht

Keine. Insbesondere wurden weder `assert-declarative-migrations.mjs` noch `repair_stale_rxdb_collection_schema_versions` entfernt oder abgeschwaecht.

## verblieben

- Der Browser nimmt `migrationStrategies` weiterhin stillschweigend an und ignoriert sie.
- Der Guard prueft weiterhin Kompilierbarkeit statt Runtime-Ausfuehrung.
- Der native Startup-Repair bleibt erforderlich und kann alte Versionstabellen entfernen, ohne dass in diesem Pfad vorher ein vollstaendiger Migrations-/Verifikationslauf stattgefunden hat.
- Starter und mehrere reale Module haben weiterhin widerspruechliche Browser-/JSON-Deklarationen.
- Die Behauptung in `docs/ctox-rxdb.md:289-295,706-707`, Runtime-App-Migrationen wuerden nativ ausgefuehrt und auf beiden Peers erzwungen, ist gegen den gemessenen Call-Graph derzeit nicht erfuellt. Nach einer vollstaendigen Variante (a) sollte die Implementierung an diese bereits dokumentierte Absicht angeglichen werden; die Dokumentation sollte nicht als Ersatz fuer den fehlenden Lifecycle geaendert werden.

## tests

Die Akzeptanzserie wurde nach dem verpflichtenden Whitelist-STOPP nicht gestartet. Es gibt keinen Fix, dessen Gruenstatus diese Befehle belegen koennte, und keine Repository-Aenderung, gegen die Trefferzahlen sinnvoll waeren.

Nur lesende Diagnose ausgefuehrt:

- Vergleich aller vorhandenen `schema.js`-/`collections.schema.json`-Strategieversionen: drei Modulgruppen mit Abweichungen gefunden (`browser`, `creator`, `credentials`).
- `git status --short`: 0 geaenderte Pfade.
- `git diff --stat`: 0 Dateien.

Nicht ausgefuehrt:

- `CARGO_TARGET_DIR=/Volumes/tmp/ctox-pipeline-targets/I-056 cargo fmt --check`
- `CARGO_TARGET_DIR=/Volumes/tmp/ctox-pipeline-targets/I-056 cargo check --bin ctox`
- `node src/apps/business-os/rxdb/tests/run-all.mjs`

## gegenprobe

Nicht ausgefuehrt. Die Pflicht-Gegenprobe setzt zuerst einen implementierten Fix und ein daraus gebautes Bundle voraus. Wegen der ausserhalb der Whitelist zwingend benoetigten Lifecycle-Dateien durfte ich weder `src/` noch das Bundle aendern. Entsprechend wurde auch nichts im Bundle zurueckgebaut.

Nachweis des unveraenderten Zustands: `git diff --stat` ist leer.

## offene_bedenken

- Eine nur in `rx-database.mjs` eingebaute Schleife ueber `allDocuments()` plus `bulkWrite()` waere nicht crash-atomar: Zeilen und Schemaversionsmarker koennten auseinanderlaufen. Das waere explizit die verbotene halbe Variante (a).
- Eine nur browserseitige Reparatur liesse den nativen loeschenden Repair unveraendert und machte die genannte Kompensation nicht ueberfluessig.
- Fuer statische/native Collections braucht die Rust-Seite einen kanonischen, JSON-serialisierbaren Strategievertrag. `src/core/rxdb/tools/build_business_os_schema_contract.mjs:4-18,37-41,201-216` liest bereits Browser-Schemas und sogar `migrationStrategies`, verwirft die Strategien aber beim erzeugten Vertrag. Funktionswerte koennen nicht direkt in JSON geschrieben werden; fuer die native Seite muss deshalb die vorhandene deklarative JSON-DSL kanonisch eingesammelt und als separater generierter Vertrag bereitgestellt werden.
- Erst nach erfolgreicher Migration und Zielverifikation darf `repair_stale_rxdb_collection_schema_versions` entfernt oder zu einer reinen, nachweislich leeren Nachkontrolle reduziert werden.

## pfade

Zwingend benoetigte Pfade ausserhalb der Hard Whitelist; deshalb STOPP ohne Aenderung:

1. `src/apps/business-os/rxdb/src/storage-indexeddb.mjs:15-18,49-59,414-466,1164-1184`
   - Persistenten Schemaversionszustand und eine atomare Migrationstransaktion bereitstellen; ohne diesen Pfad ist Variante (a) im Browser nicht crash-sicher.
2. `src/apps/business-os/app.js:1345-1352,5218-5272`
   - Die kanonische Strategiequelle eindeutig machen und deklarative JSON-Strategien tatsaechlich in die Collection-Definition einspeisen; derzeit benutzt die Shell die widerspruechliche `schema.js`-Seite.
3. `src/core/business_os/rxdb_peer.rs:2390-2423,13683-14014,14093-14169,14342-14403`
   - Native Strategien einlesen, Vorversionen vor Cleanup migrieren, jede Quellzeile im Ziel verifizieren, erst danach umschalten/aufräumen; den aktuellen Cleanup-vor-Registrierung-Pfad beseitigen.
4. `src/core/rxdb/tools/build_business_os_schema_contract.mjs:4-18,37-41,201-216`
   - Neben dem Schemavertrag einen kanonischen deklarativen Migrationsvertrag aus `collections.schema.json` generieren und widerspruechliche Duplikate fail-closed ablehnen.
5. `src/core/business_os/business_os_migration_strategies.json:new`
   - Neuer generierter, JSON-only Vertrag fuer statische native Collection-Migrationen; nicht von Hand pflegen.
6. `src/apps/business-os/scripts/assert-declarative-migrations.mjs:19-31,87-99`
   - Von isolierter Kompilier-/Beispielpruefung auf echte Ausfuehrung gegen `dist/ctox-rxdb-js.mjs` und persistierte Vorversionszeilen umstellen.
7. `src/apps/business-os/app-starter/v2/schema.js.tpl:18-19`
   - Die leere v1-Strategieseite an die kanonische JSON-Strategie angleichen beziehungsweise als zweite Migrationsquelle entfernen.
8. `src/apps/business-os/modules/browser/collections.schema.json:3-455,458-460`
   - Die vier vorhandenen Browser-v1-Identity-Strategien in die kanonische JSON-DSL aufnehmen.
9. `src/apps/business-os/modules/credentials/collections.schema.json:3-50,53-55`
   - Die vorhandene `business_commands.1`-Strategie in die kanonische JSON-DSL aufnehmen.
10. `src/apps/business-os/modules/creator/schema.js:1-23`
    - Die zweite Deklarationsseite mit der bereits vorhandenen JSON-Strategie konsistent machen oder nach Umstellung auf JSON als einzige Quelle entfernen.

Danach duerfen die gewhitelisteten Zielpfade (`rx-database.mjs`, Bundle, drei Cache-Buster-Vorkommen, neuer Bundle-Smoke und `run-all.mjs`) in derselben Welle umgesetzt und mit der geforderten Bundle-Gegenprobe belegt werden.
