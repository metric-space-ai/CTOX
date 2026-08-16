# Gesamtplan: Refactoring und Sync-Performance

Stand: 16.08.2026

Arbeitsbranch: `main`

Kampagnen-Baseline: `4c9cd805259c722985fdb023a7090e3cd2136fcb`

Dieses Dokument ist der zentrale Ausführungs- und Fortschrittsplan für die
Übernahme der Arbeiten aus `OFFENE-ARBEITEN.md`. Es unterscheidet bewusst
zwischen implementiert, lokal abgenommen und auf der Kundeninstanz
nachgewiesen. Messdetails und Rohdaten liegen unter `docs/dev/beweise/`.

## Zielzustand

- Große Sellify-Collections blockieren den ersten nutzbaren Render nicht mehr
  durch eine Vollreplikation.
- Der warme Command-Roundtrip hat einen p50 unter 300 ms, ohne neue lange
  Tail-Latenz.
- Alle kritischen Collections sind beim Boot im p95 nach weniger als fünf
  Sekunden live.
- Command-Intake-Fehler enden deterministisch und erzeugen keine dauerhafte
  CPU-/Revisionsschleife.
- `service.rs` und `store.rs` liegen bei höchstens 22.000 Produktionszeilen;
  `app.js` liegt nach der Zerlegung bei höchstens 10.000 physischen Zeilen.
- Jeder Performancegewinn besitzt eine reproduzierbare Vorher-/Nachhermessung.
- Der Kundenrollout erfolgt ausschließlich mit Backup, Prüfsummen, Rollback und
  exklusivem Wartungsfenster.

## Verbindliche Arbeitsregeln

1. Performanceänderungen erhalten vorab eine reproduzierbare Baseline.
2. Reine Codeverschiebungen enthalten keine Semantikänderungen.
3. Refactoring und Verhaltensänderung derselben Region landen nicht im selben
   Commit.
4. Fremde uncommittierte Änderungen werden weder verändert noch mitcommittet.
5. WebRTC bleibt der einzige Business-Data-Pfad; es entsteht kein neuer
   HTTP-Datenpfad.
6. Produktionsverhalten erhält keine neuen Process-Environment-Schalter.
7. Größenbudgets werden nur gesenkt, niemals zur Beseitigung eines roten Tests
   angehoben.
8. Synthetische Daten sind der Standard für Performanceprüfungen.
9. Eine lokale Implementierung gilt nicht automatisch als Kundenabnahme.

## Gesamtstatus

| Etappe | Implementiert | Lokal abgenommen | Kundenabnahme | Status |
|---|---:|---:|---:|---|
| Baseline, Beweise und Integrationshygiene | ja | ja | entfällt | erledigt |
| Größenwächter und `service.rs`-Moves | ja | ja | entfällt | erledigt |
| OA-6 endlicher Command-Intake | ja | ja | nein | Betriebsmessung offen |
| OA-2 synthetische 300k-Baseline | teilweise | teilweise | entfällt | Browsermatrix offen |
| OA-1 bounded Demand-Sync | ja | gezielte Smokes ja | nein | Scale-Abnahme offen |
| OA-4 Command-Roundtrip | teilweise | Messung vorhanden | nein | Zielwert verfehlt |
| `store.rs`-Refactoring | ja | ja | entfällt | erledigt |
| `app.js`-Refactoring | nein | nein | entfällt | wartet auf saubere Arbeitsregion |
| OA-5 Handshake-Optimierung | teilweise | Instrumentierung ja | nein | reale Bootmessung offen |
| OA-7 Store-Kompaktierung | nein | nein | nein | Wartungsfenster erforderlich |
| OA-8 ehrliches `replicationUp` | lokal vorhanden | teilweise | nein | Live-Nachweis offen |
| OA-9 AP3-Reparaturnachweis | lokal vorhanden | Smoke grün | nein | Live-Nachweis offen |

## Etappe 1: Baseline und Integrationshygiene

Status: **erledigt**

- `4c9cd8052` ist als Kampagnen-Baseline festgehalten.
- Rohmessdaten werden als JSON unter `docs/dev/beweise/raw/` versioniert.
- Auswertungen liegen unter `docs/dev/beweise/`.
- Das Move-Prüfwerkzeug extrahiert benannte Rust-Funktionen lexikalisch,
  normalisiert Whitespace und vergleicht SHA-256 vor und nach dem Move.
- Die Änderungen wurden in voneinander prüfbare Commits getrennt.

Abnahme:

- Baseline-JSON vorhanden.
- Move-Beweise maschinenlesbar vorhanden.
- Keine fremden Arbeitsbaumänderungen in den Kampagnencommits.

## Etappe 2: OA-6 — endliche Command-Intake-Zustandsmaschine

Status: **lokal erledigt; Kunden-Langzeitprüfung offen**

Implementiert:

- `accept_pending_business_command` liefert ein typisiertes Ergebnis für
  Annahme, kanonische Wiedergabe, retryfähigen Fehler und Terminalisierung.
- Der Versuchszähler wird aus offenen Intake-Failure-Datensätzen bestimmt und
  bei einem tatsächlichen Intake-Fehler erhöht.
- Beim ausgeschöpften Budget wird auch ein vorhandenes nichtterminales
  Aggregat atomar terminalisiert; Transition und Outbox entstehen in derselben
  SQLite-Transaktion.
- Konflikte lassen das kanonische Intent unverändert und schließen die
  replizierte Lifecycle-Projektion mit einem typisierten Konfliktfehler.
- Failure-Historie wird erst nach erfolgreicher Annahme oder nachweislich
  terminaler Projektion aufgelöst.
- Manuelle Wiederholung benötigt eine neue Command-ID.

Lokale Abnahme:

- Kundenmuster mit sechs fehlgeschlagenen, nichtterminalen Dokumenten ist als
  Regressionstest vorhanden.
- Alle sechs Commands erreichen das Budget, erzeugen je genau eine terminale
  Transition und verschwinden aus der Kandidatenabfrage.
- Idempotenz- und Payload-Konflikt sind separat geprüft.

Noch offen:

- Auf der Kundeninstanz mindestens eine Stunde messen: CPU-Dauerlast unter
  5 %, `idle_ticks > 0`, Kandidatenmenge dauerhaft null und kein weiterer
  Revisionszuwachs.

## Etappe 3: OA-2 und OA-1 — Scale-Test und bounded Demand-Sync

Status: **Implementierung, native Baseline und Browser-Messstrecke erledigt;
30×30-Abnahme und Latenzziel offen**

Implementiert:

- Synthetischer Store mit 139.804 Activities, 86.551 Campaigns, 60.640 People,
  17.520 Companies, 5.964 Commands und 3.690 File-Chunks.
- Große Sellify-Collections verwenden `syncProfile: "demand-only"`;
  `sellify_sync_status` bleibt eager.
- Das erste Query-Fenster ist standardmäßig auf höchstens 200 Datensätze
  begrenzt.
- Demand-Fenster werden nach 30 Sekunden revalidiert.
- Die Anzeige wird anhand der autoritativen `documentIds` des Fensters
  aufgebaut. Entfernte oder herausgefallene Dokumente bleiben nicht sichtbar.
- Der Status enthält additiv `syncProfile`, `localCoverage` und `queryReady`.
- Ein Peer ohne Query-Fetch-Capability erzeugt einen sichtbaren inkompatiblen
  Zustand statt still leerer Daten.
- Cache-Migration, Demand-Loader, Window-Correctness, Sync-Profil und
  Bundle-Reproduzierbarkeit besitzen gezielte Smokes.
- `business-os-sellify-scale-ui` provisioniert die sechs synthetischen
  Populationen, rendert eine echte sortierte/filtrierte Activities-Seite und
  erfasst Query-RPCs, Materialisierung, IndexedDB-Nutzung sowie Readiness.
- `sellify_scale_browser_matrix.mjs` führt nach einer nicht gewerteten
  Provisionierung 30 kalte und 30 warme Browserläufe mit reproduzierbarem
  Profilzustand aus und schreibt ein versioniertes JSON-Artefakt.

Gemessene native Baseline:

- 304.515 Sellify-Dokumente, 314.169 Dokumente insgesamt.
- Vier begrenzte Fenster, maximal 800 materialisierte Dokumente.
- Query-RPC-Äquivalent: vier.
- 30 native Läufe: p50 27,998 ms, p95 31,343 ms.

Der reale Einzel-Smoke besteht das strukturelle Gate ohne Vollpull mit vier
Query-RPCs und höchstens 800 materialisierten Sellify-Dokumenten. Die kalte
Latenz liegt noch über fünf Sekunden und ist deshalb noch kein Release-Gate.

Noch offene Browserabnahme, jeweils 30 Läufe kalt und warm:

- Shell-ready, erster sichtbarer Datensatz und erste bedienbare Seite.
- WebRTC-Requests/-Responses und Query-Fetch-Frames.
- Materialisierte Dokumente, IndexedDB-Größe und Collection-Readiness.
- Kein Sellify-Vollpull vor dem ersten Render.
- Höchstens fünf Query-RPCs und höchstens 1.000 materialisierte
  Sellify-Dokumente bis zur Benutzbarkeit.
- Kalter p95 unter fünf Sekunden, warmer p95 unter einer Sekunde.
- Kein Wachstum proportional zu allen 304.515 Serverdokumenten.
- Löschung außerhalb eines geladenen Fensters sowie Pagination prüfen.

## Etappe 4: OA-4 — Command-Roundtrip

Status: **instrumentiert und teilweise optimiert; Zielwert nicht erreicht**

Implementiert:

- Ein echter Smoke führt einen Warmup und anschließend 30 warme
  `ctox.provider_subscription.status`-Commands mit
  `command_timing_probe=true` aus.
- Sieben Zeitmarken werden über `consumeCommandRoundtripTiming` gesammelt und
  vom Stage-Report ausgewertet.
- Der SQLite-Table-Notifier reduziert den dominanten Intake-Abschnitt.
- Terminalzustand und Timingmarken werden in einer Projektion veröffentlicht.
- Eine endliche gezielte Terminal-Revalidierung ersetzt den einzelnen langen
  Polling-Timer.

Messstand:

| Stand | p50 | p95 | Maximum |
|---|---:|---:|---:|
| Baseline | 1.790,5 ms | 2.107,5 ms | siehe Rohdaten |
| aktueller Stand | 1.255 ms | 1.763,9 ms | 9.449 ms |
| Ziel | < 300 ms | sinkend, keine neue Tail-Latenz | keine Ausreißerklasse |

Noch offen:

- Commit→Browser-/Query-Fetch-Ausreißer weiter instrumentieren.
- Den dort belegten Engpass optimieren.
- Erneut 30 warme Läufe ausführen und Zielwert nachweisen.
- Erst nach bestandenem Zielwert als Performancegewinn abnehmen.

## Etappe 5: OA-3 — Großmodule zerlegen

### Rust

Status: **erledigt**

- `service.rs`: 21.820 Produktionszeilen.
- `store.rs`: 21.970 Produktionszeilen.
- 128 von 128 Service-Funktionen und 202 von 202 Store-Funktionen stimmen
  nach Whitespace-Normalisierung mit der Baseline überein.
- Zielbudgets von höchstens 22.000 Produktionszeilen sind erreicht.
- Neue Module und Restdateien besitzen exakt gesenkte Größenbudgets.

### Browser `app.js`

Status: **offen**

Vorgesehene Seams:

1. Data-Plane-Boot
2. Module Loader
3. Maintenance Monitor
4. Icon Registry

Abnahme:

- Reine Moves getrennt von Verhaltensänderungen.
- Node-Import-Smokes für die extrahierten Module.
- Physisches Größenbudget für `app.js` von höchstens 10.000 Zeilen.
- Beginn erst, wenn die vorhandenen parallelen Änderungen in `app.js`
  committed oder eindeutig separiert sind.

## Etappe 6: OA-5 — Bridge-Handshake

Status: **Messbarkeit hergestellt; reale Performanceabnahme offen**

Implementiert:

- Additive Metriken für Collection-Registrierungen, Peer-Open-Ereignisse,
  gestartete/erfolgreiche Protokollverhandlungen sowie aktuelle und maximale
  DataChannels.
- Der gezielte Smoke registriert 25 Collections über einen gemeinsamen Peer
  und weist genau einen offenen DataChannel ohne zusätzliche
  Registrierungs-Roundtrips nach.
- Schema-, Capability- und Auth-Prüfungen bleiben erhalten.

Noch offen:

- Reale Bootmessung mit 30 Läufen.
- p95 bis alle kritischen Collections live unter fünf Sekunden.
- Reconnect, Peerwechsel und Mixed-Version-Verhalten.
- Kein zweiter DataChannel und kein Full-Resync.
- Multi-Tab, Berechtigungs- und Schemawechsel gemeinsam mit der Browsermatrix.

## Etappe 7: Betriebliche Kundenabnahme

Status: **offen; erst nach den lokalen Performancegates**

Reihenfolge:

1. Wartungsfenster und exklusiven Zugriff bestätigen.
2. Alte Binary und Datenbank sichern.
3. Prüfsummen an Build-, Transfer- und Zielort vergleichen.
4. Kontrolliert neu starten; automatischen Rollback bereithalten.
5. OA-6 mindestens eine Stunde beobachten.
6. OA-8 ohne Browser mit `replicationUp=false` und mit verbundenem Browser mit
   `replicationUp=true` nachweisen.
7. OA-9 mit acht Sekunden künstlichem Stillstand prüfen: genau ein
   AP3-Reparaturversuch, anschließend Rückkehr zu `ok=true` bei Fortschritt.
8. OA-7 bei gestopptem Dienst durchführen: Sicherung, `VACUUM INTO`,
   `integrity_check`, Tabellen-/Schemaabgleich und atomarer Austausch.

Diese Schritte sind technische Betriebsfreigaben, keine noch ausstehende
Produktentscheidung des Owners.

## Test- und Abnahmetore

### Rust

- Größenwächter.
- Gezielte Intake-/Lifecycle-Tests mit ausgewiesener Trefferzahl.
- `cargo fmt --check`.
- `cargo check --bin ctox`.
- Relevante gefilterte `cargo test`-Läufe.

### Native RxDB

- `cargo test --manifest-path src/core/rxdb/Cargo.toml`.
- `cargo fmt --check --manifest-path src/core/rxdb/Cargo.toml`.

### Browser

- `node src/apps/business-os/rxdb/tests/run-all.mjs`.
- Scale-UI-Smoke und Cold-/Warm-Matrix.
- Reconnect, Multi-Tab, Berechtigungswechsel und Schemawechsel.
- Löschung außerhalb eines geladenen Fensters.
- Bundle ausschließlich aus `rxdb/src` bauen.
- Alle drei RxDB-Bundle-Cache-Buster identisch aktualisieren.
- Bundle-Reproduzierbarkeitswächter ausführen.

### Commit-Gate

Kein Commit darf:

- neue rote Tests enthalten,
- ein Größenbudget erhöhen,
- einen Performancegewinn ohne Messbeweis behaupten,
- fremde Arbeitsbaumänderungen enthalten.

## Kampagnencommits

| Commit | Inhalt |
|---|---|
| `42af477ad` | Evidence-Baseline und Move-Prüfwerkzeuge |
| `f356bafd8` | `service.rs`-Extraktionen |
| `acebba36e` | OA-6: endlicher Command-Intake |
| `477675304` | `store.rs`-Extraktionen |
| `111e44208` | synthetische Sellify-Scale-Baseline |
| `126df9719` | Command-Roundtrip-Messung und erste Optimierungen |
| `a789ac76b` | bounded Sellify Demand-Sync |
| `c769a5119` | Multiplex-Handshake-Metriken |
| `825ee651d` | konsolidierter Kampagnennachweis |

## Bekannte rote Baseline und nicht übernommene Paralleländerungen

- Die vollständige Browser-Suite meldete 93 grüne, sieben rote und zwei
  übersprungene Cross-Process-Tests. Alle sieben roten Tests wurden im
  isolierten Baseline-Archiv reproduziert und sind keine Regression dieser
  Kampagne.
- Der globale Größenwächter bleibt wegen sieben bereits anderweitig
  veränderter Dateien rot. Die Budgets wurden nicht angehoben.
- Im gemeinsamen Arbeitsbaum liegen weiterhin zahlreiche nicht zu dieser
  Kampagne gehörende Änderungen, darunter `app.js`, Browsermodule und weitere
  Runtime-Dateien. Sie bleiben uncommitted, bis ihre jeweilige Arbeit separat
  abgenommen wird.

## Nächste Ausführungsreihenfolge

1. Commit→Browser-/Query-Fetch-Engpass beheben und Command-p50 erneut messen.
2. Echte 30-Lauf-Cold-/Warm-Browsermatrix auf dem synthetischen Scale-Store.
3. Reale Handshake-/Boot-, Reconnect-, Peerwechsel- und Multi-Tab-Abnahme.
4. Nach Freigabe der Arbeitsregion `app.js` move-only zerlegen.
5. Verbleibende fremde Größenwächter jeweils in ihrer eigenen Arbeitsspur
   bereinigen, ohne Budgets anzuheben.
6. Kundenrollout und OA-6/OA-8/OA-9-Livenachweise.
7. OA-7-Kompaktierung im exklusiven Wartungsfenster.

## Definition of Done

Die Kampagne ist erst vollständig abgeschlossen, wenn:

- alle lokalen Performanceziele mit Rohdaten belegt sind,
- alle Größenbudgets grün sind,
- `app.js` das Zielbudget erreicht,
- die relevante Rust-, RxDB- und Browser-Testmatrix keine neue Regression
  enthält,
- der Kundenrollout inklusive Rollback-Nachweis abgeschlossen ist,
- OA-6, OA-8 und OA-9 auf der Kundeninstanz gemessen sind,
- OA-7 mit erfolgreichem `integrity_check` abgeschlossen ist.

## Zugehörige Dokumente

- `docs/dev/OFFENE-ARBEITEN.md` — ursprüngliche Übergabe und Antworten.
- `docs/dev/beweise/refactoring-kampagne-baseline.md` — aktueller
  Implementierungs- und Messnachweis.
- `docs/dev/beweise/raw/` — maschinenlesbare Rohmessungen und Move-Beweise.
- `docs/ctox-rxdb.md` — kanonische Architektur des RxDB-/WebRTC-Datenpfads.
