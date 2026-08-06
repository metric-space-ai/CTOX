# I-062 Report

## was_geaendert

- `browser`: Die vier bereits in `schema.js` vorhandenen v1-Identity-Migrationen stehen jetzt auch auf der kanonischen JSON-Seite als `operations: []` (`src/apps/business-os/modules/browser/collections.schema.json:458-479`).
- `credentials`: Die vorhandene `business_commands.1`-Migration wurde ohne inhaltliche Aenderung nach JSON uebertragen (`src/apps/business-os/modules/credentials/collections.schema.json:53-69`).
- `creator`: Der in JSON vorhandene `business_commands.1`-Schritt wird jetzt von `schema.js` als dieselbe `inbound_channel || module || ''`-Funktion gespiegelt (`src/apps/business-os/modules/creator/schema.js:25-32`).
- App-Starter: `schema.js.tpl` erzeugt jetzt wie das bereits korrekte `collections.schema.json.tpl:21-27` eine v1-Identity-Migration (`src/apps/business-os/app-starter/v2/schema.js.tpl:18-23`). Die JSON-Vorlage brauchte keine Textaenderung.
- Der Waechter importiert nun fuer alle 34 Module auch `schema.js`, vergleicht nichtleere Strategie-Collections, Zielversionen und die geordnete Operationsfolge und behandelt Identity nur als beidseitig leere Operationsfolge (`src/apps/business-os/scripts/assert-declarative-migrations.mjs:18-46,101-358`). Fehlende Spiegel nennen Modul und Collection (`:113-130`). Auch die beiden Starter-Templates werden instanziiert und gegeneinander geprueft (`:360-379`).

## ursache_belegt

- Vor jeder Aenderung wurde auf sauberem `origin/main`/`43845d55f6ec37b8d1360a6c5cc095121852ccb1` jede vorhandene `collections.schema.json` dynamisch gegen den `migrationStrategies`-Export ihres `schema.js` vermessen.
- Gemessene Collection/Versions-Asymmetrien vor dem Fix, und nur diese:
  - `browser`: JSON `[]`; JS `browser_frames.1`, `browser_input_events.1`, `browser_sessions.1`, `browser_tabs.1`.
  - `creator`: JSON `business_commands.1`; JS `[]`.
  - `credentials`: JSON `[]`; JS `business_commands.1`.
- Es wurden keine weiteren Module mit fehlenden Zielversionen gefunden. Bei den bereits beidseitig vorhandenen Migrationen zeigte die Inhaltspruefung keinen Widerspruch; daher war kein STOPP wegen inhaltlicher Divergenz erforderlich.
- Die Starter-Ursache war ebenfalls direkt sichtbar: JSON deklarierte `__COLLECTION__.1` mit `operations: []`, waehrend `schema.js.tpl` `{}` exportierte.

## kompensationen_geloescht

- Keine Datei oder Pruefung wurde geloescht.
- Der bisher nur kompilierende `assert-declarative-migrations.mjs` wurde gemaess Auftrag zum Spiegelgleichheits-Waechter erweitert. Nach Reparatur der Deklarationen ist er nicht mehr noetig, um den aktuellen Bestand funktionsfaehig zu halten, bleibt aber als verlangtes Regressionsnetz bestehen.

## verblieben

- Keine bekannte Deklarationsasymmetrie verbleibt: der neue Waechter ist fuer 34 Module plus die Starter-Vorlagen gruen.
- Keine Aenderung unter `src/apps/business-os/rxdb/src/` oder `rxdb/src/`; kein dist-Neubau und kein Cache-Buster-Bump erforderlich.
- Die JS-Gesamtsuite bleibt auf derselben vorbestehenden Rot-Menge wie `origin/main` (14 Fehler, 2 Skips); Details unter `tests`/`offene_bedenken`.

## tests

Alle Cargo-Aufrufe verwendeten `CARGO_TARGET_DIR=/Volumes/tmp/ctox-pipeline-targets/I-062`.

1. `cargo fmt --check`
   - Exit 0, keine Ausgabe.
   - Das Kommando startet keine Tests und erzeugt daher keine `test result`-Zeile bzw. Trefferzahl.
2. `cargo check --bin ctox`
   - Exit 0.
   - Abschluss: `Finished dev profile [unoptimized + debuginfo] target(s) in 56m 00s`.
   - 403 Warnungen; keine Fehler.
   - Das Kommando startet keine Tests und erzeugt daher keine `test result`-Zeile bzw. Trefferzahl.
3. `node src/apps/business-os/scripts/assert-declarative-migrations.mjs`
   - Exit 0.
   - Ergebniszeile: `Business OS declarative migrations OK (34 modules checked)`.
   - Trefferzahl: 34 Module; zusaetzlich wird `app-starter/v2` geprueft.
4. `node src/apps/business-os/rxdb/tests/run-all.mjs`
   - Finaler Lauf: Exit 1 wegen vorbestehender Rot-Menge.
   - Ergebniszeile: `ctox-rxdb suite: 79 passed, 14 failed, 2 skipped (95 total)`.
   - Trefferzahl: 95 Tests insgesamt.
   - `origin/main`-Baseline auf sauberem HEAD: ebenfalls `79 passed, 14 failed, 2 skipped (95 total)`.
   - Mengenvergleich in beide Richtungen:
     - `origin/main \\ final = []`
     - `final \\ origin/main = []`
   - Identische 14er-Rot-Menge:
     - `chunk-query-demand-disabled-smoke.mjs`
     - `command-consumer-inventory-smoke.mjs`
     - `module-demand-only-collections-smoke.mjs`
     - `multi-tab-browser-smoke.mjs`
     - `non-destructive-reconnect-repair-smoke.mjs`
     - `query-demand-authoritative-read-smoke.mjs`
     - `recovery-journal-browser-smoke.mjs`
     - `recovery-primary-reset-browser-smoke.mjs`
     - `recovery-registration-nonblocking-smoke.mjs`
     - `replication-recovery-smoke.mjs`
     - `stale-while-revalidate-smoke.mjs`
     - `structured-conflict-quarantine-browser-smoke.mjs`
     - `symmetric-capability-handshake-smoke.mjs`
     - `task-id-inventory-smoke.mjs`
   - Ein erster Nachlauf unter konkurrierender Cargo-Last hatte zusaetzlich `command-type-inventory-smoke.mjs` per `ETIMEDOUT`. Der direkte Recheck war gruen (`Business OS command type inventory OK`, 52 exakte Control-Typen, 139 Predicate-Typen, 11 Browser-Runtime-Typen, 13 Control-Predicates, 123 Browser-Literale); der anschliessende volle Lauf hatte wieder exakt die Baseline-Menge.

Es wurde kein Cargo-Testfilter verwendet; die Filterverbote waren fuer die verlangten `fmt`/`check`-Kommandos nicht einschlaegig.

## gegenprobe

- In `credentials` wurde nach dem gruenen Fix temporaer die gesamte JSON-Strategie entfernt.
- Der erweiterte Waechter wurde rot mit Exit 1 und nannte Modul sowie Collection:

  `credentials/business_commands: collections.schema.json missing migration strategies mirrored by schema.js`

- Danach wurde die Datei bytegenau aus der unmittelbar vorher angelegten Kopie wiederhergestellt (`cmp -s` erfolgreich).
- `git diff --stat` nach dem Rueckbau:

  ```text
   src/apps/business-os/app-starter/v2/schema.js.tpl  |   6 +-
   .../modules/browser/collections.schema.json        |  22 ++
   src/apps/business-os/modules/creator/schema.js     |   9 +
   .../modules/credentials/collections.schema.json    |  17 ++
   .../scripts/assert-declarative-migrations.mjs      | 308 ++++++++++++++++++++-
   5 files changed, 354 insertions(+), 8 deletions(-)
  ```

- Die verpflichtende JS-Rotmengen-Gegenprobe gegen den sauberen `origin/main`-Stand ist in beide Richtungen leer, siehe `tests`.

## offene_bedenken

- Die zwei Cross-Process-Tests wurden von der Suite uebersprungen, weil der Wire-Daemon nicht gebaut war: `cross-process-file-fetch-smoke.mjs` und `cross-process-wire-smoke.mjs`. Das ist fehlende Coverage, aber identisch zur Baseline.
- Fuenf der vorbestehenden Fehler beruhen auf fehlendem `src/apps/business-os/node_modules/playwright`; die uebrigen vorbestehenden Fehler sind ebenfalls unveraendert zur Baseline.
- `cargo check` ist gruen, meldet jedoch 403 vorbestehende Warnungen.
- Keine offenen Bedenken an der Migrations-Spiegelgleichheit selbst.

## pfade

Geaendert, alle innerhalb der Hard Whitelist:

- `src/apps/business-os/modules/browser/collections.schema.json:458-479`
- `src/apps/business-os/modules/credentials/collections.schema.json:53-69`
- `src/apps/business-os/modules/creator/schema.js:25-32`
- `src/apps/business-os/app-starter/v2/schema.js.tpl:18-23`
- `src/apps/business-os/scripts/assert-declarative-migrations.mjs:1-379`

Geprueft, aber inhaltlich bereits korrekt und deshalb unveraendert:

- `src/apps/business-os/app-starter/v2/collections.schema.json.tpl:21-27`

Keine zusaetzlichen Pfade ausserhalb der Whitelist erforderlich.
