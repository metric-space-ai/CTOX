# Refactoring-Kampagne: Baseline und erste Nachweise

Stand: 2026-08-16

## Unveränderliche Baseline

- Commit: `4c9cd805259c722985fdb023a7090e3cd2136fcb`
- Performance-Messungen werden aus einem `git archive` dieses Commits und aus einem entsprechenden Snapshot des Vergleichsstands ausgeführt.
- Messdaten verwenden ausschließlich synthetische Datensätze.
- Rohdaten: `raw/refactoring-baseline-4c9cd8052.json`

## Move-only: CV-Print-Recovery

Neun zusammenhängende CV-Print-Helfer wurden aus `service.rs` nach
`service_cv_print_recovery.rs` verschoben. Das Werkzeug
`src/scripts/verify-rust-move.mjs` extrahiert die benannten Rust-Funktionen
lexikalisch, normalisiert Whitespace und vergleicht SHA-256.

- Ergebnis: 9 von 9 Funktionsrümpfen identisch.
- `service.rs`: 26.244 → 26.129 Produktionszeilen.
- Neues Modul: 122 Produktionszeilen.
- Rohbeweis: `raw/service-cv-print-move-4c9cd8052.json`

## OA-6

Die Intake-Schnittstelle liefert typisierte Ergebnisse. Beim Erreichen des
Retry-Budgets wird auch ein vorhandenes nichtterminales Aggregat in derselben
SQLite-Transaktion terminalisiert, die Transition und Outbox anlegt.
Konflikte verändern das kanonische Intent nicht und liefern eine terminale
Fehlerprojektion.

Gezielte Prüfung:

```text
cargo test --bin ctox intake_failure_ -- --nocapture
running 2 tests
2 passed; 0 failed; 2838 filtered out

cargo test --bin ctox customer_intake_pattern_reaches_budget_for_six_existing_aggregates -- --nocapture
running 1 test
1 passed; 0 failed; 2840 filtered out

cargo test --bin ctox exhausted_conflicting_command_projects_one_terminal_conflict_and_leaves_intent_immutable -- --nocapture
running 1 test
1 passed; 0 failed; 2840 filtered out
```

Das zusätzliche Fixture bildet sechs `failed`/nichtterminale Commands mit
bereits vorhandenen Aggregaten und anfänglich einem beziehungsweise zwei
offenen Fehlversuchen ab. Alle sechs erreichen Versuch 5, erhalten je genau
eine terminale Transition und hinterlassen keine offene Failure-Historie.
Der Konflikttest belegt zusätzlich, dass das kanonische Intent unverändert
bleibt, die terminale Projektion genau einmal entsteht und der zweite Sweep
keinen Kandidaten mehr konsumiert.

## Aktueller Integrationszustand

Der Größenwächter bestätigt die neuen exakten Budgets für `service.rs`,
`service_cv_print_recovery.rs`, `rxdb_peer_intake.rs` und
`rxdb_peer_intake_state.rs`. Der Gesamttest bleibt wegen bereits vorhandener,
nicht zu dieser Etappe gehörender Änderungen in sieben anderen Modulen rot:
`mcp_channel.rs`, `office_engine_spreadsheet_layout.rs`, `rxdb_peer.rs`,
`store_security_projections.rs`, `store_outbound_delivery_policy.rs`,
`account_helpers.rs` und `service_runtime_support.rs`. Deren Budgets werden
nicht angehoben und deren
laufende Änderungen werden in dieser Etappe nicht verändert.

`cargo check --bin ctox`, `cargo fmt --check` und
`cargo fmt --check --manifest-path src/core/rxdb/Cargo.toml` sind erfolgreich.
Der vollständige native RxDB-Lauf erreichte 345 von 346 Tests; der einzige
zeitabhängige External-Poll-Test scheiterte einmal beim Startup-Warten und war
in der unmittelbar anschließenden isolierten Wiederholung grün (1 von 1).

## Sellify-Scale: native Query-Baseline

`src/core/rxdb/tools/sellify_scale_benchmark.mjs` erzeugt die festgelegten
synthetischen Populationen in einer isolierten SQLite-Datenbank und misst pro
Lauf vier sortierte, gefilterte Fenster mit höchstens 200 Dokumenten.

- Sellify-Dokumente: 304.515
- Gesamt einschließlich Commands und File-Chunks: 314.169
- Datenbankgröße: 140.996.608 Bytes
- Seed: 9.571,694 ms
- 30 Läufe: p50 27,998 ms, p95 31,343 ms, Maximum 31,809 ms
- Maximal materialisiert: 800 Dokumente
- Query-RPC-Äquivalent: 4

Rohdaten: `raw/sellify-scale-sqlite-window-baseline.json`. Diese Messung ist
die reproduzierbare native Query-Baseline. Sie ersetzt nicht die noch
ausstehende Cold-/Warm-Browsermessung mit WebRTC-, IndexedDB- und
Interaktionsmarken.

## OA-1: bounded Demand-Sync

Demand-Fenster sind auf 200 Dokumente begrenzt, werden nach 30 Sekunden
revalidiert und lesen lokal ausschließlich die autoritativen `documentIds` des
Fensters. Ein leeres Fenster fällt nicht mehr auf eine unbeschränkte lokale
Collection-Abfrage zurück. Sellify-Daten bleiben `demand-only`, während
`sellify_sync_status` eager bleibt. Der Status ergänzt `syncProfile`,
`localCoverage` und `queryReady`; fehlende Query-Fetch-Capability wird als
sichtbarer inkompatibler Zustand ausgewiesen.

Gezielte Node-Smokes für autoritative Fenster, Window-Correctness,
Stale-while-Revalidate, Demand-Loader, Sync-Profile, V1.5-Status und
Query-Fetch-Capability sind grün. Das Bundle wurde aus `rxdb/src` reproduzierbar
neu gebaut; die drei Cache-Buster sind identisch.

Die gezielte Command-Bus-Suite ist mit 25 von 25 Tests grün. Der
Scale-Benchmark-Smoke und der Bundle-Reproduzierbarkeitswächter sind ebenfalls
grün.

Die vollständige Suite meldet 93 grün, 7 rot und 2 übersprungene
Cross-Process-Tests. Alle sieben roten Tests wurden einzeln im isolierten
`4c9cd8052`-Archiv reproduziert und sind damit keine Kampagnenregression;
Rohklassifikation: `raw/rxdb-suite-baseline-red-4c9cd8052.json`.

## OA-4: echter Command-Roundtrip

Der Browser/Rust-Smoke führt einen ungemessenen Warmup und anschließend 30
serielle `ctox.provider_subscription.status`-Commands mit allen sieben Marken
aus. Dabei wurden zwei Harness-Races behoben: die Launch-Konfiguration endet
vor der Design-Template-Zuweisung, und der Browserstart wartet auf den nativen
Peer statt nur auf die Existenz der Config.

Die Baseline hatte einen Gesamt-p50 von 1.790,5 ms und p95 von 2.107,5 ms.
Der dominante rohe Intake-Abschnitt wurde durch den SQLite-Table-Notifier von
p50 639,5 ms auf etwa 200–300 ms reduziert. Terminalzustand und Timingmarken
werden in genau einer RxDB-Projektion veröffentlicht. Eine endliche Folge von
höchstens fünf gezielten Terminal-Revalidierungen ersetzt den einzelnen
1.500-ms-Timer.

Der aktuell wiederholte Debug-Smoke bleibt trotz dieser Verbesserungen rot:
Gesamt-p50 1.255 ms, p95 1.763,9 ms, Maximum 9.449 ms. Der belegte verbleibende
Engpass sind Query-Fetch-Ausreißer auf dem Commit→Browser-Pfad. Das Ziel
`p50 < 300 ms` wird daher ausdrücklich **nicht** als erreicht gewertet.

- Baseline: `raw/command-roundtrip-warm-baseline-marks.json` und
  `raw/command-roundtrip-warm-baseline-report.json`
- Intake-Event-Zwischenstand:
  `raw/command-roundtrip-warm-intake-event-marks.json` und
  `raw/command-roundtrip-warm-intake-event-report.json`
- Aktueller Stand: `raw/command-roundtrip-warm-optimized-marks.json` und
  `raw/command-roundtrip-warm-optimized-report.json`

Die zwei Rust-Regressionstests für Probe-Markierungen sind grün (2 von 2,
2.839 gefiltert): eine explizite Probe schreibt geordnete native Marken, der
normale Pfad bleibt frei von Timing-Metadaten.

## OA-3: move-only Großmodule

`service.rs` liegt nach vier weiteren Extraktionen bei 21.820
Produktionszeilen. Die neuen Dateien enthalten Service-Statusquellen,
Business-OS-App-Authoring, App-Recovery und den Systematic-Research-Worker.
128 von 128 extrahierten Funktionsrümpfen stimmen mit `4c9cd8052` überein.

`store.rs` liegt bei 21.970 Produktionszeilen. Sync/TURN/Auth,
Workspace-Branding, Runtime-Sync-Settings, Why-Diagnostics und
Provider-Subscription/Auth wurden extrahiert. 202 von 202 Funktionsrümpfen
stimmen mit `4c9cd8052` überein. Alle neuen Dateien und beide Restdateien haben
exakt gesenkte Größenbudgets. `cargo check --bin ctox` ist nach beiden
Move-Wellen erfolgreich.

Die zehn Rohbeweise liegen als
`raw/service-*-move-4c9cd8052.json` beziehungsweise
`raw/store-*-move-4c9cd8052.json` vor.

## OA-5: multiplexer Handshake-Nachweis

Der gemeinsame Browser-Room-Peer veröffentlicht jetzt additive Zähler für
Collection-Registrierungen, Peer-Open-Ereignisse, gestartete/erfolgreiche
Protocol-Negotiations sowie aktuelle und maximale offene DataChannels. Damit
kann der Boot-Smoke die Abnahmebedingungen direkt prüfen, statt die Anzahl aus
Logzeilen zu schätzen.

`multiplex-handshake-metrics-smoke.mjs` registriert 25 Collections vor dem
Peer-Open und belegt: ein gemeinsamer DataChannel, null durch die Registrierung
ausgelöste Protocol-Roundtrips. Schema-, Capability- und Auth-Prüfungen bleiben
im unveränderten Room-Handshake. Das Browser-Bundle ist aus `rxdb/src`
reproduzierbar gebaut; alle drei Bundle-Cache-Buster sind identisch.

Mixed-Mode-Handshake und parallele Checkpoint-Initialisierung sind in den
gezielten Smokes ebenfalls grün. Der Nachweis ist ein Registrierungs-/
Protokollwächter; er ersetzt noch keine reale 30-Lauf-Bootmessung.

## Noch offene Abnahmetore

- Die echte Cold-/Warm-Browsermatrix mit WebRTC, IndexedDB und sichtbarer
  Interaktion ist noch nicht durchgeführt. Die native 304.515-Dokumente-
  Baseline darf dafür nicht als Ersatz gelten.
- Das warme Command-Ziel `p50 < 300 ms` ist mit aktuell 1.255 ms nicht erreicht.
  Der verbleibende Commit→Browser-Engpass muss vor einer Performanceabnahme
  weiter eingegrenzt werden.
- Der reale Boot-p95 bis alle kritischen Collections live sind sowie Reconnect,
  Peerwechsel, Multi-Tab, Berechtigungs- und Schemawechsel benötigen noch den
  vollständigen Browser-/Peer-Harness.
- `app.js` bleibt unangetastet, solange die dort vorhandene fremde laufende
  Browserarbeit nicht committed beziehungsweise eindeutig separiert ist.
- Kundenrollout, einstündiger OA-6-Betriebsnachweis, OA-7-Kompaktierung sowie
  der Live-Nachweis für OA-8/OA-9 beginnen erst nach bestandenen lokalen Gates,
  Backup/Rollback-Vorbereitung und exklusivem Wartungsfenster.

Der lokale AP3-Vertrags-Smoke ist bereits grün: acht Sekunden künstlicher
Stillstand liefern `data_plane_no_progress`, genau einen Reparaturversuch und
nach erneutem Fortschritt wieder einen gesunden Zustand. Offen bleibt bewusst
der gleichartige Nachweis mit einem real verbundenen Browser auf der
Kundeninstanz.
