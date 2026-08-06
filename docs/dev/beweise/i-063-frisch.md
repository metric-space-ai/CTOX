# I-063 Report

## was_geaendert

- Persistenter Collection-Schema-Marker im Browser-Primary eingefuehrt (`collectionSchemaMarkers`, IndexedDB-Layout v4). Der Marker speichert je Collection die **deklarierte Version** und den von `schemaHash(schema, collection)` gelieferten **effektiven Hash**; Registry-Overrides sind damit bereits eingerechnet. Markerzustand ist `invalidating` oder `ready`.
- `addCollections()` berechnet Version+effektiven Hash vor Freigabe der Collection und laesst den Storage den persistierten Marker pruefen. Ein fehlender Marker gilt ebenfalls als Invalidation (ein leerer Primary kann nach Reset/Eviction trotzdem alte Checkpoints/Sidecars haben).
- Bei Abweichung laeuft unter einem collection-/datenbankgenauen Web Lock und einer realm-lokalen Promise-Serialisierung:
  1. live WAL-Status aus `${databaseName}__recovery_v2` per `stateCollection`-Index lesen,
  2. `pushable=1` im Primary erneut im destruktiven IndexedDB-Readwrite-Vorgang zaehlen,
  3. bei `pushable>0` oder pending WAL fail-closed mit `collection_version_invalidation_blocked` und „Nothing was discarded“,
  4. sonst Collection-Zeilen loeschen und Marker atomar auf `invalidating` setzen,
  5. collection-spezifischen Query-/Demand-Sidecar leeren sowie alle retained Pull-/Push-/Readiness-Records der Collection entfernen,
  6. Marker erst danach auf `ready` finalisieren.
- Alle regulaeren Primary-Mutationspfade (`bulkUpsert`, `bulkWrite`, Recovery-Replay, Hard-Delete/Eviction und Schema-Index-Rebuild) benutzen dieselbe Serialisierung und pruefen den aktuellen `ready`-Marker. Damit kann zwischen Dirty-Guard und Clear kein neuer Runtime-Write einsickern; alte Collection-Handles schreiben nach einem Markerwechsel fail-closed nicht weiter.
- Replication-State verwirft nach einer Versionsinvalidierung retained Checkpoints explizit; der erste Pull startet damit bei `null` und drainiert ueber den vorhandenen Pullpfad bis zur leeren Antwort.
- `migrationStrategies` ist in `addCollections()` jetzt explizit als **native-only** dokumentiert und wird im Browser bewusst nicht ausgefuehrt. Ignorieren statt Ablehnen erhaelt die gemeinsame Deklarationsform, ohne eine zweite Migrationsmaschine zu bauen.
- `resetBusinessDb()` blieb unveraendert und ist kein Versionspfad.
- `dist/ctox-rxdb-js.mjs` mit dem gepinnten esbuild-0.28.0-Befehl neu gebaut; alle drei identischen Cache-Buster auf `20260805-collection-version-v90` angehoben.
- Neuer automatisch entdeckter Bundle-Browser-Smoke `collection-version-invalidation-smoke.mjs` mit 13 Assertions.

## ursache_belegt

- Ausgangsbefund I-061: kein persistenter Marker, `addCollections()` verglich nichts, 2/2 v0-Zeilen blieben bei v1 kommentarlos liegen.
- Der neue Smoke reproduziert den realen Bundle-/IndexedDB-Pfad und belegt nun:
  - v0-Marker wird persistent geschrieben;
  - eine saubere Native-Origin-v0-Zeile wird bei v1 entfernt (`clearedRows=1`);
  - der fertige v1-Marker enthaelt `declaredVersion=1`, `state=ready` und exakt den effektiven `schemaHash()`;
  - retained Pull/Push plus `firstPullCompletedAtMs` verschwinden;
  - eine isolierte `pushable=1`-Zeile bei `pendingBatches=0` blockiert mit dem typisierten Fehler; Zeile, v0-Marker und Unsynced-Count bleiben unveraendert;
  - eine absichtlich werfende Browser-`migrationStrategies`-Funktion wird nicht ausgefuehrt.
- Die beiden Pflicht-Gegenproben belegen kausal, dass der Smoke nicht nur zufaellig gruen ist: ohne Dirty-Guard wird exakt der fail-closed-Teil rot; ohne Collection-Clear wird exakt der Invalidations-Teil rot.

## kompensationen_geloescht

- Keine Datei/Logik der Kompensationen geloescht.
- `resetBusinessDb()` bleibt wie verlangt als datenbankweiter Notausgang bestehen, wird aber fuer Browser-Versionswechsel nicht mehr benoetigt oder aufgerufen.
- Der native Startup-Sweep bleibt fuer die native Seite; die Browser-Kopie hat jetzt ihren eigenen Marker-/Invalidationspfad und strandet bei einem Versionswechsel nicht mehr still.

## verblieben

- Keine Kompensation bleibt fuer den normalen Browser-Versionspfad erforderlich.
- Die fuenf weiteren, in I-061 genannten Browser-Smokes (WAL-only, Multi-Tab-TOCTOU, Sidecar-Fenster, Full-Pull-Netzpfad, Reset/WAL-Interaktion) bleiben gemaess Auftrag ein Folge-Los. Der Produktionscode fuer live WAL, Web-Lock-Serialisierung, Sidecar- und Checkpoint-Invalidierung ist bereits verdrahtet; dieses Los fuegt absichtlich nur den einen geforderten Kern-Smoke hinzu.
- Browser ohne Web Locks API invalidieren nicht unsicher, sondern scheitern typisiert mit `collection_version_lock_unavailable`; es wird nichts verworfen.

## tests

- `CARGO_TARGET_DIR=/Volumes/tmp/ctox-pipeline-targets/I-063 cargo fmt --check` — **gruen**, keine Ausgabe. Keine `test result`-Zeile, weil der Befehl 0 Tests ausfuehrt; kein Filter verwendet.
- `CARGO_TARGET_DIR=/Volumes/tmp/ctox-pipeline-targets/I-063 cargo check --bin ctox` — **gruen**, `Finished dev profile ... in 8m 10s`, 400 vorbestehende Warnungen, 0 Fehler. Keine `test result`-Zeile, weil `cargo check` 0 Tests ausfuehrt; kein Filter verwendet.
- `PLAYWRIGHT_MODULE_PATH=/Users/michaelwelsch/Documents/ctox-dev/node_modules/playwright node src/apps/business-os/rxdb/tests/collection-version-invalidation-smoke.mjs` — **gruen**, `assertions=13`, 1 OK-Marker.
- `PLAYWRIGHT_MODULE_PATH=... node src/apps/business-os/rxdb/tests/run-all.mjs` — neue Probe **PASS**; Gesamtergebnis: `ctox-rxdb suite: 85 passed, 9 failed, 2 skipped (96 total)`.
  - Zwei Cross-Process-Smokes wurden mangels gebautem Wire-Daemon laut Suite sichtbar uebersprungen.
  - Die neun roten Tests waren auf sauberem HEAD/origin/main (beide `22cf6406c2d1505fdf0b2a85507f1ec3ca2c6946`) bereits identisch rot: `chunk-query-demand-disabled-smoke`, `command-consumer-inventory-smoke`, `module-demand-only-collections-smoke`, `non-destructive-reconnect-repair-smoke`, `query-demand-authoritative-read-smoke`, `replication-recovery-smoke`, `stale-while-revalidate-smoke`, `symmetric-capability-handshake-smoke`, `task-id-inventory-smoke`.
  - Mengenvergleich in beide Richtungen: **final minus HEAD = ∅; HEAD minus final = ∅**. Die Testsuite wuchs nur um den neuen gruenen Smoke von 95 auf 96 Tests.
- `node src/apps/business-os/rxdb/tests/bundle-reproducible-smoke.mjs` — **gruen**, 1 OK-Marker (`ctox-rxdb bundle reproducibility guard OK`).
- `node src/apps/business-os/rxdb/tests/data-plane-guard-smoke.mjs` — **gruen**, 1 OK-Marker; Guard-Inventar `{ jsFiles: 29, rustFiles: 13 }`.
- `git diff --check` — **gruen**, 0 Whitespace-Fehler.
- Verbotene Cargo-Filter wurden nicht benutzt; es wurde ueberhaupt kein Cargo-Testfilter verwendet.

## gegenprobe

1. **Unsynced-Guard ausgebaut, Bundle neu gebaut:** Bedingung fuer `pushable>0 || pendingBatches>0` temporaer deaktiviert. Der Bundle-Smoke wurde erwartungsgemaess rot (Exit 1):
   - `Error: pushable guard did not fail closed:`
2. **Clear-Schritt ausgebaut, Bundle neu gebaut:** `cursor.delete()` im collection-genauen Clear temporaer entfernt. Der Bundle-Smoke wurde erwartungsgemaess rot (Exit 1):
   - `Error: clean v0 cache rows were not cleared on v1 bring-up`
3. Nach jeder Gegenprobe Source exakt aus der Sicherung zurueckkopiert und das Bundle mit dem gepinnten Befehl neu gebaut. `cmp` bestaetigte Source und Bundle bytegleich zum Stand vor der jeweiligen Probe.
4. `git diff --stat` nach jedem Rueckbau war identisch: tracked `9 files changed, 6829 insertions(+), 5974 deletions(-)` (der erwartete neue untracked Smoke wird von `git diff --stat` nicht mitgezaehlt). Danach war der Ziel-Smoke wieder gruen mit `assertions=13`, ebenso Bundle-Reproducibility und Data-Plane-Guard.

## offene_bedenken

- `indexedDB.databases()` wird nur als Optimierung benutzt, um beim ersten Marker-Setup keinen noch nicht existierenden Sidecar anzulegen. Wenn Enumeration nicht verfuegbar/fehlerhaft ist, wird der Sidecar normal geoeffnet und fail-closed geleert.
- Die gleiche-Realm-Replikationsinstanz erhaelt zusaetzlich eine In-Memory-Checkpoint-Invalidierung; der normative Bring-up invalidiert jedoch vor Replication-Start. Die explizite Multi-Tab-Race-Gegenprobe bleibt wie beauftragt im Folge-Los.
- Keine HTTP-, npm-/`node:`- oder Env-Fallbacks hinzugefuegt.

## pfade

Alle Aenderungen liegen innerhalb der Hard Whitelist; keine weitere Datei ist erforderlich.

- `src/apps/business-os/rxdb/src/storage-indexeddb.mjs:15-18,73-125,142-258,363-373,515-525,879-897,1230-1238,1262-1456,1495-1497` — Marker-Store, Bring-up-Vergleich, live Dirty-Guard, atomarer collection-genauer Clear/Marker, Write-Serialisierung und fail-closed Markerpruefung.
- `src/apps/business-os/rxdb/src/recovery-journal.mjs:365-374` — live pending-by-collection WAL-Summary ueber IndexedDB.
- `src/apps/business-os/rxdb/src/rx-database.mjs:17-23,126-173,220-227` — `addCollections`-Orchestrierung, effektiver Hash, dokumentiertes Ignorieren von `migrationStrategies`, Invalidation-Ergebnis an Collection.
- `src/apps/business-os/rxdb/src/replication-webrtc.mjs:74-102,951-959,1968-1984` — retained Pull/Push/Readiness verwerfen und Null-Cursor erzwingen.
- `src/apps/business-os/rxdb/src/query-meta-storage.mjs:7-19` und `src/apps/business-os/rxdb/src/query-meta-backend-indexeddb.mjs:230-260` — collection-spezifischen Query-/Demand-Sidecar ohne Memory-Fallback leeren.
- `src/apps/business-os/rxdb/tests/collection-version-invalidation-smoke.mjs:1-166` — 13 Assertions gegen das gebaute Bundle.
- `src/apps/business-os/rxdb/dist/ctox-rxdb-js.mjs` — reproduzierbar neu gebaut, nicht direkt editiert.
- `src/apps/business-os/shared/db.js:19`, `src/apps/business-os/shared/sync.js:343,1080` — drei identische Cache-Buster.
