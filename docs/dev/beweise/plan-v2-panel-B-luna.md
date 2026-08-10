## 1. approach

1. Pflichtlektüre von `AGENTS.md`, Sync-/Refactoring-/Service-Dokumenten und Größenvertrag.
2. Prüfung der Wächter-Registrierung, physischen Zeilen und des tatsächlichen Produktionszeilen-Algorithmus.
3. Navigation zu Browser-Boot, Collection-Registrierung, Peer-Lifecycle, Batches, Checkpoints und Projektionsschleifen.
4. Trennung von Move-only-Schnitten und semantischen Wellen anhand von Symbolen, Invarianten und vorhandenen Planbefunden.
5. Keine Builds, Tests, Node-Läufe oder Schreibzugriffe; Abschlussprüfung ausschließlich per Git-Leseoperationen.

## 2. prototype paths or no-code evidence

### Prämissen und Größen

- `mod module_size_tests;` ist auf dem geprüften HEAD registriert: `src/core/main.rs:105-108`, konkret `src/core/main.rs:107`.
- Der Guard zählt bis zum letzten alleinstehenden `#[cfg(test)]`, einschließlich Leerzeilen und Kommentaren: `src/core/module_size_tests.rs:39-62`; der Vertrag bestätigt dieselbe Regel: `contracts/module_size_budget.txt:3-16`.
- Die genannten physischen Größen stimmen exakt mit `wc -l` überein:
  - `src/core/business_os/store.rs`: 43.781
  - `src/core/service/service.rs`: 46.039
  - `src/core/business_os/rxdb_peer.rs`: 22.658
  - `src/core/business_os/office_engine.rs`: 16.888
  - `src/core/business_os/mcp_channel.rs`: 11.673
  - `src/core/business_os/store_outbound_commands.rs`: 10.797
  - `src/apps/business-os/app.js`: 12.321
  - `src/apps/business-os/shared/sync.js`: 3.026
  - `src/apps/business-os/shared/business-chat.js`: 6.559
- Der physische Umfang ist nicht die Budgetzahl. Nach dem Guard-Algorithmus sind aktuell über Budget:
  - `src/core/business_os/office_engine.rs`: 14.598 Produktion vs. 13.953 Budget.
  - `src/core/business_os/store.rs`: 28.413 vs. 27.516.
  - `src/core/business_os/store_outbound_commands.rs`: 5.354 vs. 5.270.
  - `src/core/mission/channels/mod.rs`: 7.226 vs. 7.221.
- `service.rs` ist nach dem tatsächlichen HEAD nicht über Budget: 26.224 vs. 26.237. Die frühere Board-Behauptung zu `service/business_os.rs` ist durch den aktuellen Stand überholt; `src/core/service/business_os.rs` liegt bei 6.060 Produktionszeilen gegenüber einem Budget von 7.106.
- `src/core/context/lcm/mod.rs` liegt mit 5.600 unter dem vertraglichen Wert 5.627. Das ist ein Budget-Ratschen-Drift in umgekehrter Richtung, den der Guard ebenfalls rot machen würde.

### Performance-Hebel

#### Hebel 1 — Serialer Browser-Start aller Collections

- Evidenz:
  - `src/apps/business-os/shared/sync.js:56-58`: `COLLECTION_START_GAP_MS = 500`, Queue-Step-Timeout 3 s.
  - `src/apps/business-os/shared/sync.js:98`: globale `collectionStartQueue`.
  - `src/apps/business-os/rxdb/src/replication-webrtc.mjs:316-350`: room-weite Catch-up-Queue mit zeitlich begrenzten Slices.
  - `src/core/business_os/rxdb_peer.rs:2518-2612`: native Seite registriert alle Collections und startet eine multiplexierte Session.
  - `docs/ctox-rxdb.md:177-204`: Collection-Bridges starten serialisiert mit 500 ms Abstand; initiales Catch-up ist room-weit serialisiert.
- Wirkung: Bei vielen Collections addiert sich die Startlatenz direkt. Multiplexing reduziert bereits die Zahl der WebRTC-Verbindungen, verhindert aber nicht die serielle Registrierung und Catch-up-Reihenfolge.
- Messung: Browser-Trace mit Zeitstempeln für `startCollection`, `peer-open`, erstes `masterChangesSince`, `initialReplication=complete`; zusätzlich p50/p95 bis zur letzten Core-Collection. Kontrollgröße: Anzahl Collections und Datenmenge.
- Aufwand: **M**, weil Queue-Semantik, Reconnect und die Invariante „kein Catch-up-Verlust“ zusammen betrachtet werden müssen. Kein pauschales Parallelisieren ohne Backpressure.

#### Hebel 2 — Batch-/Checkpoint-Chattiness

- Evidenz:
  - `src/apps/business-os/shared/sync-contract.js:48-68`: reguläre Collections Batch 20, Chunk-Collections 8 bzw. `desktop_file_chunks` 6, `knowledge_tables` 1.
  - `src/core/business_os/rxdb_peer.rs:2579-2602`: native Multiplex-Session wird mit `20, 20, 5_000` für Pull, Push und Retry gestartet.
  - `docs/ctox-rxdb.md:468-510`: 16-KiB-Framegrenze, 8-MiB-Transfergrenze, ACK-Fenster 4, Query-Chunks bis 200 Dokumente/256 KiB.
  - `src/apps/business-os/rxdb/src/replication-webrtc.mjs:442-468`: `masterChangesSince` wird über die gemeinsame Raum-Session geroutet.
- Wirkung: Zu kleine Batches erzeugen mehr RPCs, Checkpoint-Schreibvorgänge und JSON-/Frame-Overhead; zu große Batches erhöhen Fragmentierung, Speicherbedarf und Retries.
- Messung: RPCs pro initial synchronisiertem Dokument/MiB, Bytes pro Dokument, Checkpoint-Commits, Retries, DataChannel-Backpressure und Zeit bis zum ersten/letten Checkpoint; je Collection-Klasse separat.
- Aufwand: **S**, wenn nur Messinstrumentierung und Konfigurationsmatrix; **M**, wenn adaptive Größen eingeführt werden. `knowledge_tables` und Chunk-Pfade brauchen eigene Sicherheitsgrenzen.

#### Hebel 3 — Projektionsschleifen und unnötige Reconciliation

- Evidenz:
  - `src/core/business_os/rxdb_peer.rs:3831-3895`: generische Projektion über Source-Stamp, Projektion und Sleep/Idle-Backoff.
  - `src/core/business_os/rxdb_peer.rs:4078-4147`: Business-Record-Projektion mit persistentem Fortschritt, Collection-Cursor und Sleep.
  - `src/core/business_os/rxdb_peer.rs:6211-6317`: Slice-Verarbeitung über Collections und Cursor.
  - `src/core/business_os/rxdb_peer.rs:4365-4410`: Outbox-/Command-Schleife mit aktiven und idle Sleep-Intervallen.
  - `docs/ctox-sync-fundament-plan-2026-07-28.md:85-91,129-131,165-167`: sieben divergierende Projektions-Loops; Ziel ist ein Runner mit einheitlicher Stamp-Disziplin und idempotenten Upserts.
- Wirkung: Jede unnötige Projektion erzeugt SQLite-Lese-/Schreiblast, RxDB-Write-Last und anschließend Replikationsverkehr. Vorzeitiges Stempeln kann Änderungen überspringen und erzeugt dann nachträgliche Reconcile-Pässe.
- Messung: Projektionen pro Minute bei unverändertem Core-Store, SQLite-Zeit, RxDB-Writes, replizierte Dokumente/Minute, Reconcile-Anzahl, „source changed but projected=0“-Fälle. Besonders wichtig: idle vs. Änderungs-Workload.
- Aufwand: **L** für die semantische Vereinheitlichung; **S-M** für reine Telemetrie und Baseline.

#### Hebel 4 — Desktop-Datei-Index und Chunk-Last

- Evidenz:
  - `src/core/business_os/rxdb_peer.rs:3543ff`: Desktop-Datei-Upsert/Indexierung.
  - `src/core/business_os/rxdb_peer.rs:8470-8605`: Demand-Chunk-Registry und Chunk-Streaming.
  - `docs/ctox-rxdb.md:565-589`: `desktop_file_chunks` wird nicht eager synchronisiert; File-Viewer verwendet `rxdb.file.fetch`.
  - `docs/ctox-rxdb.md:706-709`: Rescan muss unveränderte Dateien als No-op erkennen und darf verifizierte Chunk-Generationen nicht alle 15 Sekunden erneut prüfen.
- Wirkung: Ein Rescan, der neue Generationen oder Chunk-Prüfungen unnötig erzeugt, vervielfacht Tombstones, Chunk-Reads und Replikationsvolumen. Demand-Fetch ist bereits der richtige Architekturpfad; die verbleibende Optimierung liegt in Scan- und Verifikationskosten.
- Messung: CPU-/SQLite-Zeit pro Scan, erzeugte Dokumente/Tombstones, Chunk-IDs gelesen, replizierte Bytes, File-Open-Latenz und Scan-Kosten bei unverändertem Workspace.
- Aufwand: **M**, weil die vorhandenen Invarianten gegen Datenverlust und Dematerialisierung ausdrücklich erhalten bleiben müssen.

#### Hebel 5 — Shell-Boot und Kontrollplane-Requests

- Evidenz:
  - `src/apps/business-os/app.js:84-105`: Maintenance-Poll, 2-s aktiver Poll, 60-s idle Poll und Single-flight für `modules/registry.json`.
  - `src/apps/business-os/app.js:962-1040`: Boot-Reihenfolge: Launch Context, Session, Maintenance, Sync Config, DB, WebRTC/RxDB, Module Catalog.
  - `src/apps/business-os/app.js:8336-8408`: Icon-Fetches werden durch In-flight-Promise-Deduplication zusammengelegt.
  - `src/apps/business-os/app.js:8603-8645`: Maintenance-Poll pausiert bei verborgenem Tab und backt im Idle-Zustand auf 60 s.
  - Board, `docs/dev/ctox-refactoring-board.html:234-237,274-275`: gemessene Verbesserung 208 → 129 HTTP-Requests und Poll ca. 30/min → 1,6/min; weitere Arbeit: Keep-alive/Worker-Pool und Status-Rückfall.
  - `docs/ctox-rxdb.md:83-99`: HTTP bleibt Control-/Bootstrap-Plane; kein HTTP-Fallback für Sync-Daten.
- Wirkung: Der große Shell-Gewinn ist bereits realisiert. Verbleibende Requests sind Boot-/Status-/statische Assets; sie dürfen nicht durch Sync-Daten-Fallback „optimiert“ werden.
- Messung: HAR/PerformanceObserver pro Boot, Requests bis `dataPlaneReady`, Time-to-first-module, Maintenance-Requests pro Tab-Stunde, Cache-Hit-Rate, SSH-Aufbauzahl und Verbindungsaufbauzeit.
- Aufwand: **S-M** für Messung/Keep-alive; **M** für Worker-Pool oder Status-Fallback. Der Datenpfad ist ein harter Non-Goal.

### Refactoring-Schnittkarte

#### Reine Umzüge, relativ billig

- `rxdb_peer.rs` → bereits begonnene bzw. geplante Verantwortungsgrenzen:
  - Desktop-Dateien: Quelle/Verantwortung sichtbar ab `src/core/business_os/rxdb_peer.rs:3543`; Plan: `docs/ctox-sync-fundament-plan-2026-07-28.md:133-147`.
  - Browser-Control: historisch bereits aus dem Peer geschnitten; `src/core/business_os/rxdb_peer.rs:21-28` importiert `rxdb_peer_browser`; Board dokumentiert die bereits erfolgte Kernzerlegung in `docs/dev/ctox-refactoring-board.html:352ff`.
  - Demand-File-Streaming: `src/core/business_os/rxdb_peer.rs:8470-8605`.
  - Ein klar abgegrenzter Loop-/Metrics-Block kann move-only verschoben werden, solange keine Stamp- oder Scheduling-Semantik verändert wird.
- `store.rs`:
  - Outbound-Commands haben bereits eine erkennbare fachliche Grenze; im aktuellen Arbeitsstand existiert `src/core/business_os/store_outbound_commands.rs`, außerdem verweisen `store.rs`-Symbole auf Outbound-Pfade, etwa `src/core/business_os/store.rs:14311-14333`.
  - Projection- und Catalog-Projektionen haben erkennbare Gruppen, etwa `src/core/business_os/store.rs:2874-3000`, `:4046-4346`, `:8704ff`.
  - Ein reiner Textumzug ist nur dann billig, wenn Imports, Sichtbarkeiten, Attribute und Testbereich unverändert bleiben.
- `service.rs`:
  - Große Belang-Blöcke sind strukturell identifizierbar: State-Invariant-Repair `src/core/service/service.rs:1659ff`, Outcome-/Recovery-Mechanik `:5839ff` und Agent-/Timeout-Pfade `:6151ff, :6341ff`.
  - Ein Move-only-Schnitt ist hier möglich, aber wegen gemeinsamer Persistenz- und Abschlusszustände weniger sicher als beim Peer.
- `app.js`:
  - `loadModules`: `src/apps/business-os/app.js:9346ff`.
  - `loadModuleLayout`: `src/apps/business-os/app.js:9482ff`.
  - `openBusinessDataPlane`: `src/apps/business-os/app.js:1179ff`.
  - `registerCustomModuleIcons`: `src/apps/business-os/app.js:8336ff`.
  - `startMaintenanceMonitor`: `src/apps/business-os/app.js:8603ff`.
  - Diese Funktionen sind gute Kandidaten für Facade-/Import-Seams, aber der Schnitt von `app.js` ist nicht vollständig move-only, solange globaler `state`, DOM und Bootstrap-Reihenfolge direkt gekoppelt sind. `docs/ctox-sync-fundament-plan-2026-07-28.md:259-274` bestätigt ausdrücklich: null Exports und Regex-Smokes sind ein Testbarkeitsproblem.

#### Semantische Wellen / Sol-Tier

- Projektionen: Stamp-Zeitpunkt, Idempotenz, Cursor und Reconcile-Verhalten. Belege: `src/core/business_os/rxdb_peer.rs:3831-3895,4078-4147,6211-6317`; Planbefund `docs/ctox-sync-fundament-plan-2026-07-28.md:85-91,165-167`.
- Revisionen/Envelopes und Chunk-ID-Konvergenz in `store.rs`/Peer: Planbefund `docs/ctox-sync-fundament-plan-2026-07-28.md:67-83,150-164`.
- Command-Plane: mehrere Zustände, Autorisierung, Submit-/Push-/Projection-Receipt; Plan `docs/ctox-sync-fundament-plan-2026-07-28.md:168-245`.
- `service.rs` SYNC-F Runde 2:
  - I-070 Mission-Seed vor I-071.
  - I-071 atomarer Attempt-Abschluss vor I-072/I-073/I-074.
  - Reihenfolge und Befunde sind in `docs/ctox-service-plan-2026-08-05.md:52-66` festgehalten.
- Browser-RxDB-Performance: Jede Änderung unter `src/apps/business-os/rxdb/src/` verlangt Dist-Rebuild und Cache-Buster-Abgleich: `AGENTS.md:87-98`; `docs/ctox-rxdb.md:861-890`.

#### Kollisionsminimierende Reihenfolge

1. Wächter und Baseline-Messung aktivieren bzw. verifizieren.
2. R-01: rote Tests klassifizieren; unabhängig von den Produktionsschnitten.
3. Nur native, klar move-only Peer-Nähte schneiden: Desktop-Index, Demand-Source, bereits vorbereitete Browser-Control-Grenzen.
4. `rxdb_peer.rs`-Semantik stabilisieren: Projektionen und Command-Plane erst nach den Move-only-Schnitten.
5. `store.rs`-Revision/Outbound/Projection-Schnitte; danach semantische Revision-/ID-Wellen.
6. `service.rs`: I-070 → I-071 → I-072 → I-073 → I-074 strikt seriell innerhalb der Datei.
7. Browser-`app.js`-Seams separat, danach Browser-Runtime-Performance in einer einzigen Dist-Welle.
8. Shell-HTTP-Optimierung zuletzt und nur auf Control-/Asset-Pfaden.

## 3. commands run and results

- `ls AGENTS.md ... contracts/module_size_budget.txt ...`  
  Ergebnis: alle Pflichtdateien vorhanden.
- `rg -n "module_size_tests|mod .*tests" src/core/main.rs`  
  Ergebnis: `src/core/main.rs:107` registriert `mod module_size_tests;`.
- `wc -l ...` für die im Brief genannten Dateien  
  Ergebnis: alle oben genannten physischen Größen stimmen exakt.
- Lesen von `src/core/module_size_tests.rs` und Python-Auswertung des dort beschriebenen Algorithmus  
  Ergebnis: vier aktuelle Budgetüberschreitungen: `office_engine.rs`, `store.rs`, `store_outbound_commands.rs`, `mission/channels/mod.rs`.
- `rg`/`sed` auf Browser-Sync, Peer, Store, Service und App  
  Ergebnis: Queue-/Batch-/Checkpoint-/Projection-/Boot-Anker oben dokumentiert.
- `git log -5 --oneline`  
  Ergebnis: HEAD enthält unter anderem `6a917e13b`, `d328d3a1d`, `afa5b21c0`, `3ac1075d5`, `3f46fb06e`.
- Verbotene Kommandos wurden nicht ausgeführt: kein `cargo build/check/test`, kein Node-Testlauf, kein npm, kein Netzwerkzugriff.
- `git status --porcelain | wc -l`  
  Ergebnis: **136** bereits vorhandene Änderungen im Checkout. Es wurden keine Dateien verändert; `git diff --name-only` zeigt ausschließlich den vorbestehenden Dirty-Stand.

## 4. difficulty 1-5 with reasons

**5/5**

- Mehrere Pläne und Board-Stände widersprechen dem aktuellen Codezustand, insbesondere bei Budgetwerten und bereits gelandeten Schnitten.
- Produktionszeilen und physische Zeilen werden absichtlich unterschiedlich gezählt.
- `rxdb_peer.rs`, `store.rs` und `service.rs` vermischen Lifecycle, Persistenz, Projektion, Reparatur und Tests; Move-only und Semantik sind daher nur über Invarianten unterscheidbar.
- Performanceänderungen berühren dieselben Dateien wie Refactorings und können Checkpoint-, Stamp-, Retry- und Autoritätssemantik verändern.
- Ein sauberer Testlauf war laut Brief verboten; deshalb konnte die 59-Rot-Klassifikation nicht selbst ausgeführt, sondern nur als Planarbeit eingeordnet werden.

## 5. hidden constraints

- Der aktuelle Checkout ist nicht sauber: `git status --porcelain | wc -l` ergibt 136. Das ist ein vorbestehender Zustand; ein sauberer Commit-/Merge-Stand darf als Kampagnentor nicht stillschweigend vorausgesetzt werden.
- Kein HTTP-Datenfallback: `AGENTS.md:77-85`, `docs/ctox-rxdb.md:59-99`.
- Browser-RxDB-`src/`-Änderungen benötigen Dist-Rebuild mit gepinntem esbuild und identischen Cache-Bustern in `shared/db.js` und `shared/sync.js`: `AGENTS.md:87-98`, `docs/ctox-rxdb.md:861-890`.
- Generierte Wire-Verträge dürfen nicht einseitig editiert werden: `AGENTS.md:93-98`, `docs/ctox-rxdb.md:592-614`.
- Die native Seite ist passiver WebRTC-Responder; Initiatorverhalten darf nicht „zur Vereinfachung“ reaktiviert werden: `docs/ctox-rxdb.md:371-391`, `docs/ctox-rxdb.md:1082-1086`.
- Native Bring-up-Fehler sind fatal und müssen durch die Supervisor-Schleife neu gestartet werden: `src/core/business_os/rxdb_peer.rs:2579-2642`, `docs/ctox-rxdb.md:302-313`.
- Mandanten verwenden alte Binäre. Protokoll-, Schema- und Checkpointänderungen brauchen Mixed-Version- und Rollback-Nachweis: `docs/ctox-rxdb.md:627-641,840-849`.
- Die Briefregel „maximal `--test-threads=4`“ ist konservativer als die RxDB-Doku, die für die belasteten Crate-Tests `--test-threads=1` empfiehlt: `docs/ctox-rxdb.md:1019-1043`. Für Messungen sollte deshalb maximal 4, für SQLite-/RxDB-Baselines bevorzugt 1 verwendet werden.
- Große Testlast: Die bestehende Analyse nennt komplette SQLite-Datenbanken, Tokio-Runtimes und echte TCP-Listener als Ursache hoher Last: `docs/ctox-sync-fundament-plan-2026-07-28.md:272-274`.
- Ein dynamischer Collection-Bestand ist möglich: `src/core/business_os/rxdb_peer.rs:11296ff,11531-11583`; die Zahl „~195 Collections“ muss vor einer Messung aus dem tatsächlich registrierten Katalog ermittelt werden, nicht als feste Konstante angenommen werden.
- Der Dirty-Stand enthält parallel berührte Sync-/Store-/Dist-Dateien. Refactoring und Performancearbeit an denselben Dateien dürfen daher nicht über unversionierte Zwischenstände koordiniert werden.

## 6. likely failure modes

1. **Wächter wieder still deaktiviert:** `mod module_size_tests;` oder ein anderer Guard verschwindet bei einem selektiven Commit. Der aktuelle HEAD zeigt zwar die Registrierung (`src/core/main.rs:107`), aber die Board-Historie dokumentiert genau diesen Ausfall: `docs/dev/ctox-refactoring-board.html:288-289,334`.
2. **Budget künstlich erhöht:** Statt Zeilen zu entfernen oder auszulagern, wird der Vertrag angehoben. Das widerspricht `contracts/module_size_budget.txt:18-26` und dem Guard `src/core/module_size_tests.rs:65-95`.
3. **Serialisierung zu aggressiv parallelisiert:** Mehrere Collection-Catch-ups überlasten die gemeinsame Session oder überspringen Aktivierungsereignisse. Die Aktivierungsinvariante ist in `docs/ctox-rxdb.md:710` festgehalten.
4. **Checkpoint wird zu früh gestempelt:** Projektion oder Pull meldet Fortschritt, bevor der Write abgeschlossen ist; dadurch entstehen stille Lücken. Die zugrunde liegende Invariante steht in `docs/ctox-rxdb.md:702-706`.
5. **Reines Move-only behauptet, aber Semantik verändert:** Besonders bei `service.rs` können Outcome-Zeuge, Review, Failure-Counter und Queue-Ack durch Reihenfolgeänderungen auseinanderlaufen: `docs/ctox-service-plan-2026-08-05.md:34-50`.
6. **Browser-Dist driftet:** `src/` wird geändert, `dist/` oder Cache-Buster nicht. Das kann eine zweite Bundle-Instanz mit doppelten Peer-Registern erzeugen: `docs/ctox-rxdb.md:875-890`.
7. **HTTP wird als Performance-Fallback eingeführt:** Das würde die harte Daten-Grenze verletzen, selbst wenn es kurzfristig schneller erscheint: `AGENTS.md:77-85`.
8. **Alte Mandanten brechen durch Contract- oder Schemaänderung:** Einseitige Wire-/Schemaänderungen quiescieren Collections oder verhindern Reconnect: `docs/ctox-rxdb.md:616-625,1065-1068`.
9. **59 rote Tests als „Umgebungsrauschen“ verworfen:** Die Board-Historie nennt ausdrücklich die Gefahr eines unerreichbaren „alles grün“-Tors: `docs/dev/ctox-refactoring-board.html:339-352,464-469`.
10. **Messung durch Maschinenlast verfälscht:** SQLite-/Tokio-Tests und parallele Läufe können unter hoher Last timeouten; die Messung wird dann fälschlich als Sync-Regression interpretiert: `docs/ctox-rxdb.md:1039-1043`.

## 7. decisive tests

1. **Boot-/Initial-Sync-Latenz**
   - Messen: Zeit von `openBusinessDataPlane` bis `dataPlaneReady`, Zeit bis zur ersten und letzten `initialReplication=complete`, RPC-Anzahl und p95 pro Collection.
   - Trägt den Plan, wenn Queue-/Batchänderungen die Latenz senken, ohne neue Catch-up-, Checkpoint- oder Aktivierungsfehler.
   - Kippt den Plan, wenn Parallelisierung zwar Zeit spart, aber `active-collections-catchup` oder Checkpoint-Invarianten verletzt.

2. **Batch-/Frame-Kostenmatrix**
   - Workloads: kleine Business-Dokumente, `desktop_file_chunks`, `knowledge_tables`, große Unicode-Dokumente.
   - Messen: RPCs/MiB, framed bytes, Retries, ACK-Resumes, Peak-RAM und Checkpoint-Commits.
   - Ziel: weniger Chattiness ohne Überschreitung der 16-KiB-Frame- und 8-MiB-Transfergrenzen aus `docs/ctox-rxdb.md:468-510`.

3. **Idle-Projektions- und Reconcile-Test**
   - Unveränderter Store über definierte Zeit, danach kontrollierte Einzeländerung und Burst-Änderung.
   - Messen: Projektionsdurchläufe, SQLite-Writes, RxDB-Writes, replizierte Dokumente, Reconcile-/Repair-Anzahl und Quellstamp-Lücken.
   - Muss zeigen, dass ein unveränderter Store nahezu keine Arbeit erzeugt und eine Änderung genau einmal idempotent projiziert wird.

4. **Reconnect-/Checkpoint-Resume-Messung**
   - Abbruch während Initial-Pull, Push und Projektionsrewrite; anschließend Reconnect mit alter und neuer Binärversion.
   - Messen: Zeit bis Konvergenz, erneut gelesene Dokumente, vollständige Resyncs, verlorene/duplizierte Rows und Checkpoint-Invalidierungen.
   - Der Test ist entscheidend, weil `docs/ctox-rxdb.md:655` retained checkpoints als zentrale Chattiness-Reduktion beschreibt.

5. **Shell-Boot-Request- und Sync-Grenztest**
   - HAR/PerformanceObserver für kalten Boot, warmen Boot, versteckten Tab und aktive Maintenance.
   - Messen: Requests bis zum ersten Modul, Requests pro Stunde, SSH-Verbindungen, Module-Catalog-Deduplication und WebRTC-Datenpfad.
   - Muss bestätigen, dass weitere Reduktion nur Control-/Asset-Anfragen betrifft und keine Business-Records über HTTP laufen.

## 8. recommended additions to the final brief

### Stufe 0 — Evidenz einfrieren und Arbeitsbaum kontrollieren

- Aufgaben:
  - Baseline der tatsächlichen Budgetverletzer festschreiben.
  - `mod module_size_tests;` plus Meta-Wächter gegen eigenes Verschwinden absichern.
  - R-01: 59 rote Tests nach Commit, Ursache und Kategorie klassifizieren.
  - Boot-, Initial-Sync-, Projektion- und Request-Baselines erheben.
- Messbares Tor: **Keine neuen roten Tests gegenüber der dokumentierten Basis; Budgetverletzer und 59 rote Tests sind namentlich klassifiziert; jede Messung hat einen reproduzierbaren Input.**
- Seriell: Baseline vor jeder Performanceänderung.
- Parallel: R-01-Klassifikation, Guard-Audit und Request-Baseline können in verschiedenen Dateien/Artefakten laufen.

### Stufe 1 — Wächter und sichere Schnitte

- Aufgaben:
  - Budget-Guard registriert und selbstbewacht.
  - Aktuelle vier Budgetüberschreitungen nicht durch Budgeterhöhung kaschieren.
  - `rxdb_peer.rs`-Move-only-Nähte: Desktop-Dateien, Demand-Source und verbleibende klar isolierte Helfer.
  - `app.js`-Facades für `loadModules`, `openBusinessDataPlane`, Icon-Loader und Maintenance-Monitor vorbereiten.
- Messbares Tor: **Keine neuen roten Tests, keine neue Budgetüberschreitung, keine Änderung der normalisierten Move-only-Vereinigung, keine neue öffentliche API ohne Begründung.**
- Parallel:
  - R-01 und `app.js`-Seams parallel zu nativen Move-only-Schnitten.
  - Keine Browser-Runtime-Sourceänderung parallel zu einer anderen Dist-Welle.

### Stufe 2 — Native Sync-Struktur

- Reihenfolge:
  1. Peer-Restnähte als Move-only.
  2. Gemeinsamer Projektions-Runner und Stamp-Disziplin.
  3. Revision-/Envelope-Single-Source.
  4. Chunk-ID-Konvergenz mit Migration/Rollback.
  5. Command-Plane als einheitlicher Zustandsautomat.
- Messbares Tor: **Keine neuen roten Tests; jede extrahierte Verantwortung hat einen Owner; keine neuen `repair_*`-/`reconcile_*`-Funktionen ohne Allowlist; Projektionen sind doppelanwendungs-idempotent.**
- Seriell:
  - Projektionssemantik vor Performance-Tuning der Projektionsschleifen.
  - Revision-/ID-Migration vor dem Löschen von Legacy-Lesepfaden.
- Parallel:
  - Nach stabilen Grenzen können Revision/Envelope und Chunk-ID-Analyse parallel vorbereitet werden; ihre produktiven Änderungen bleiben wegen gemeinsamer `store.rs`-Berührung sequenziell.

### Stufe 3 — SYNC-F Runde 2

- Reihenfolge zwingend:
  1. **I-070:** Mission-Seed-Root-Fix.
  2. **I-071:** atomarer Attempt-Abschluss mit typisiertem Success, dauerhaftem Finalisierungsdatensatz und idempotenter Wiederaufnahme.
  3. **I-072:** dauerhafte Repair-Telemetrie.
  4. **I-073:** Sweep-Audit und Entfernung der Queue-Deduplizierung.
  5. **I-074:** typisiertes CV-Gate statt Fehlertext-Substring.
- Quelle: `docs/ctox-service-plan-2026-08-05.md:52-66`.
- Messbares Tor: **Keine neuen roten Tests; I-070 zeigt keine neuen State-Invariant-Vorher-Zustände; I-071 beweist atomare Abschlusskonsistenz; I-072–I-074 liefern dauerhafte, typisierte Telemetrie statt flüchtiger oder textbasierter Indizien.**
- Strikt seriell innerhalb `service.rs`, weil alle Änderungen dieselben Abschluss-, Recovery- und Counter-Pfade berühren.

### Stufe 4 — Performance-Welle

- Reihenfolge:
  1. Messinstrumentierung und Baselines.
  2. Batch-/Checkpoint-Matrix.
  3. vorsichtige Collection-Start-/Catch-up-Optimierung.
  4. Projektions-Idle- und Reconcile-Optimierung.
  5. Desktop-Scan-/Chunk-Verifikation.
  6. Shell-Controlplane-Requests und Keep-alive/Pool.
- Messbares Tor: **Mindestens ein vorher definierter p95-/Chattiness-Gewinn je Hebel, keine neuen roten Tests, keine neuen Full-Resyncs, keine HTTP-Datenpfade.**
- Parallel:
  - Native Batch-/Projection-Telemetrie kann parallel zu `app.js`-Requestmessung laufen.
  - Desktop-Scan und Shell-Controlplane liegen in unterschiedlichen Dateien.
- Seriell:
  - Browser-Runtime-Änderungen wegen Dist-Rebuild und Cache-Buster.
  - Performanceänderungen an Projektionen erst nach I-070/I-071 bzw. stabiler Stamp-Semantik.

### Stufe 5 — Abnahme und Mandantenprobe

- Aufgaben:
  - Alte Binär-/neue Binär-Kompatibilität.
  - Browser-Dist-Reproduzierbarkeit.
  - WebRTC-only/Data-Plane-Guard.
  - Kontrollierte Tests mit höchstens vier Threads, für SQLite-Baselines bevorzugt einem Thread.
- Messbares Tor: **Keine neuen roten Tests gegenüber der klassifizierten Basis; alle fünf entscheidenden Messungen liegen als Vorher/Nachher vor; vier Mandanten bleiben mit ihren jeweiligen Binärständen funktionsfähig; `git status` ist nur dann sauber, wenn der gemeinsame Arbeitsbaum ausdrücklich bereinigt und versioniert wurde.**

## 9. unresolved questions

1. Woher stammt die Briefzahl „~195 Collections“ genau: aus dem statischen Katalog, einem konkreten Mandanten oder einem Laufzeitbestand? Der native Code verwendet einen dynamischen Bestand (`src/core/business_os/rxdb_peer.rs:11296ff`).
2. Welche 59 roten Tests sind in `docs/dev/beweise/rot-basis.txt` aktuell noch vorhanden, und welche davon wurden seit dem Board-Eintrag bereits durch neue Commits verändert?
3. Soll die Kampagne den bestehenden gemeinsamen Dirty-Checkout als externe Voraussetzung akzeptieren oder ist ein separater sauberer Integrationsstand verpflichtend?
4. Welche konkrete Zielmetrik gilt für „Performance verbessert“: p95-Bootzeit, Requests, RPCs/MiB, CPU, SQLite-Zeit oder Peak-RAM?
5. Ist die Collection-Start-Serialisierung eine bewusst konservative Sicherheitsgrenze oder nur ein historischer Backpressure-Wert?
6. Welche Mandanten laufen noch mit einer Binärversion vor dem Multiplex-/Checkpoint-Stand, und wie lange muss die Mixed-Version-Kompatibilität unterstützt werden?
7. Soll `service.rs` nach I-071 überhaupt weiter zerlegt werden, bevor der atomare Abschluss als stabiler Owner-Kontrakt etabliert ist?
8. Welche bestehenden Dirty-Änderungen in `store.rs`, `shared/sync.js` und `rxdb/dist/` sind bereits semantische Performancearbeit und müssen vor neuen Schnitten übernommen oder verworfen werden?

```workjet-completion-receipt-v1
{"schemaVersion":1,"status":"completed","summary":"Read-only discovery completed. Verified module-size registration, current budget violations, sync performance levers, refactoring cut order, constraints, and measurable campaign gates. No files were written. The checkout already contained 136 pre-existing dirty entries.","changedFiles":[],"verification":[{"command":"git status --porcelain | wc -l","result":"136 (pre-existing; no writes made)"}],"concerns":["Acceptance condition of a clean worktree cannot be claimed because the checkout was already dirty at discovery start.","No builds or tests were run because the brief explicitly prohibited them."],"producedPaths":[]}
```
