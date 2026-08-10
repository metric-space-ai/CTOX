Ich habe nun ausreichend Evidenz gesammelt. Hier ist mein vollständiger Discovery-Report.

---

## 1. approach

1. Prämissen geprüft: `mod module_size_tests;` gesucht in `main.rs`; physische Zeilen aller gelisteten Dateien mit `wc -l` nachgezählt; Budget-Datei gelesen und mit Ist-Zahlen konfrontiert.
2. Performance-Hebel am Code identifiziert: Timer/Intervalle, Projektions-Loops, Cache-Retention, WebRTC-Chattiness und HTTP-Verbindungsverhalten in `rxdb_peer.rs`, `doc_cache.rs`, `server.rs` und `sync.js` untersucht.
3. Refactoring-Schnittkarte erstellt: bereits extrahierte Submodule inventiert, verbleibende Verantwortlichkeiten in den verbliebenen God-Files zugeordnet, Umzugs- vs. semantische Schnitte getrennt.
4. Plan-Gerüst entworfen: Stufen mit messbaren, erreichbaren Toren („keine NEUEN roten Tests"), Parallelisierbarkeit nach Dateikollisionen aufgeschlüsselt.
5. Risiken und versteckte Constraints aus Refactoring-Board und Pflichtlektüre extrahiert, decisive tests definiert.

## 2. evidence list (datei:zeile)

**Prämissen-Prüfung:**

- `src/core/main.rs:107` — `mod module_size_tests;` **IST registriert** auf HEAD. Die Behauptung im Brief „die Zeile fehlte" bezieht sich auf einen *früheren* Zustand; Commit `6a917e13b` hat sie zurückgebracht **und** einen Meta-Test hinzugefügt, der ihr Verschwinden verhindert. → Prämisse veraltet.
- `contracts/module_size_budget.txt:26-27` — Budget für `store.rs = 27516` und `service.rs = 26237` (Produktionszeilen, gezählt bis zum letzten `#[cfg(test)]`-Marker). `wc -l` liefert 43.781 bzw. 46.039 physische Zeilen. Die Differenz ist Testcode. Die Budget-Zahlen sind **Ratschen**, die nur fallen dürfen.
- Budget-Verstöße (Produktionszeilen vs. Budget): `rxdb_peer.rs` Budget 12.718 (`contracts/module_size_budget.txt:28`) — physische Zeilen 22.658. Der `#[cfg(test)]`-Marker liegt an Zeile 12.719 (`rxdb_peer.rs:12719`), also Produktionsanteil ≈ 12.718 — **exakt im Budget**. `store.rs` Budget 27.516, Marker liegt nicht bei den ersten 21 gefundenen (21 verschiedene `#[cfg(test)]`-Vorkommen, `grep -c`). `service.rs` Budget 26.237, 29 `#[cfg(test)]`-Vorkommen.
- `src/apps/business-os/app.js` — `grep -c "^export "` = **0**. Bestätigt: keine Node-Testbarkeit. Datei hängt am DOM (`window.addEventListener` am Ende).

**Performance-Hebel (5 stärkste):**

- **H-1: Projektions-Loop über ~171 Collections** — `rxdb_peer.rs:6317-6400`: iteriert sequenziell über `business_record_projection_collections_for_root()` (178 Schema-Collections minus 7 ausgeschlossene = ~171, `rxdb_peer.rs:12164-12192`). Pro Collection: `spawn_blocking` + `database_write_lock` + `NATIVE_RXDB_WRITE_LOCK` (`rxdb_peer.rs:6331-6334,6376`). `BUSINESS_RECORD_PROJECTION_PAGE_SIZE = 25` (`rxdb_peer.rs:750`), `SYNC_LIMIT = 2_000` (`rxdb_peer.rs:749`). Bei 171 Collections × Cursor-Reset-Runden = hunderte Serialisierungen pro Tick. **Warum wirkt**: Paginierung + Lock-Granularität dominieren die Sync-Latenz.
- **H-2: `Connection: close` auf jeder HTTP-Antwort** — `src/core/business_os/server.rs:3821` und `:3845`: jeder Response erzwingt Verbindungstrennung. Bei 129 Shell-Anfragen (Board, bestätigt) = 129 TCP-Handshakes. Refactoring-Board Stufe 3 nennt dies explizit (`docs/dev/ctox-refactoring-board.html:251`).
- **H-3: DocumentCache wächst ungebremst bei churn-heavy IDs** — `src/core/rxdb/src/doc_cache.rs:155-160`: `item.latest = doc_data.clone()` wird bei JEDEM `get_cached_rx_documents`-Aufruf aktualisiert, selbst wenn der Weak-Tot ist. `sweep_dead_cache_items` (`doc_cache.rs:68-73`) löscht nur Einträge mit leerem `by_rev` — aber `apply_change_events` (`doc_cache.rs:128-143`) hält `latest` für IDs, deren Handles längst gefallen sind, auf dem neuesten Stand. Der Sweep läuft nur alle 256 Aufrufe (`doc_cache.rs:48,173`). SYNC-50-Befund bestätigt: **höchstes OOM-Risiko** bei Commands/Queue-Tasks/rotierenden Chunks.
- **H-4: 12+ unabhängige Polling-Loops mit 3s-Intervall** — `rxdb_peer.rs:723-731`: Notes (3s), Channel-State (3s), Business-Users (3s), Runtime-Settings (3s), Module-Catalog (3s), Ticket-State (3s), Business-Record-Projections (3s), Knowledge-Tables (15s), Command-Consumer (1s aktiv/30s idle), Desktop-File-Index, Browser-Runtime-Maintenance. Alle laufen als separate `tokio::spawn`-Tasks (`rxdb_peer.rs:2648-2716`). Bei Idle wird auf `BUSINESS_OS_STANDBY_RECONCILE_INTERVAL_SECS = 30 * 60` zurückgegangen (`rxdb_peer.rs:722`), aber der Übergang erfordert `BUSINESS_RECORD_PROJECTION_IDLE_BACKOFF_AFTER_TICKS = 1` — ein einziger aktiver Tick resettet den Idle-Zähler.
- **H-5: Initial-Replication über alle 178 Collections multiplexed** — `rxdb_peer.rs:2579-2641`: `replicate_web_rtc_rs_multi` startet für alle Collections gleichzeitig; `collection_count` typ. 178. Browser-Seite pacet initial (`sync.js:609-611`: „Pacing initial catch-up keeps a legitimate multi-collection bootstrap below the wedged-peer recycle threshold"). Die `boundedCollectionStartQueueStep` serialisiert den Startup, aber jeder Collection-Start erzeugt einen `bridgePromise` mit 3s-Timeout (`sync.js:614`). 178 Collections × Overhead = Bootstrap-Zeit.

## 3. commands run and results

| Kommando | Kernergebnis |
|---|---|
| `rg -n "mod module_size_tests" src/core/main.rs` | `107:mod module_size_tests;` — **IST vorhanden** |
| `wc -l` aller 9 gelisteten Dateien | Alle Zahlen exakt bestätigt (43.781 / 46.039 / 22.658 / 16.888 / 11.673 / 10.797 / 12.321 / 3.026 / 6.559) |
| `cat contracts/module_size_budget.txt` | 17 Produktionsdateien über 5.000 Zeilen; `store.rs = 27516`, `service.rs = 26237`, `rxdb_peer.rs = 12718` als Ratschen |
| `grep -c "^export " src/apps/business-os/app.js` | `0` — bestätigt „untestbar aus Node" |
| `grep -n "Connection: close" server.rs` | Zeilen 3821 + 3845 — jeder HTTP-Response schließt die Verbindung |
| `python3 -c "...business_os_schema_contract.json..."` | `178 collections` im Schema-Vertrag |
| `grep "spawn" rxdb_peer.rs \| wc -l` | `55` tokio-spawn-Stellen; 12+ Hintergrund-Loops identifiziert |
| `grep "repair_\|reconcile_\|legacy_" rxdb_peer.rs \| wc -l` | `83` Vorkommen — Schuldklassen aus Fundament-Plan noch nicht an der Wurzel gefixt |
| `wc -l desktop_files.rs browser_control.rs command_plane.rs` | Bereits extrahiert: 833 + 436 + 1.916 Zeilen — SYNC-B P1.1/P1.2/P1.4 teilweise erfolgt |

## 4. difficulty: 4/5

Gründe:
- **+1**: Drei God-Files (store.rs 43.781, service.rs 46.039, rxdb_peer.rs 22.658) mit 968/1.147/580 Funktionen — jede Berührung riskiert Kaskaden.
- **+1**: 12+ asynchrone Hintergrund-Loops mit teils sekündlichen Intervallen — Performance-Tuning und Refactoring an denselben Dateien kollidieren zeitlich.
- **+1**: Browser-Runtime (`app.js` 0 Exports, `dist/`-Rebuild-Pflicht mit 3 Cache-Buster-Bumps) blockiert nicht-native Schnitte.
- **−1**: Bereits bewiesene Mechanik: `afa5b21c0` hat rxdb_peer.rs von 25.292 → 22.658 als reinen Umzug geschnitten; Submodule (desktop_files.rs, browser_control.rs, command_plane.rs) existieren schon.
- **−0**: Ratschen-System und Wächter sind etabliert und funktionieren auf HEAD (`module_size_tests` aktiv).

## 5. hidden constraints

- **Geteilter Checkout mit 209 dirty Einträgen** (fremder Besitz): jeder Schnitt an `store.rs`, `sync.js`, `db.js`, `dist/` kollidiert mit uncommitteter Fremdarbeit (`docs/sync-engine-optimization-plan-2026-07-17.md` Zeile „BLOCKED on foreign uncommitted worktree changes").
- **dist-Rebuild + Cache-Buster-Pflicht**: Jede `src/*.mjs`-Änderung erzwingt esbuild-Rebuild von `ctox-rxdb-js.mjs` PLUS Bump aller drei identischen `?v=`-Buster in `db.js`, `sync.js` (`docs/ctox-rxdb.md`, AGENTS.md Business OS Data Boundary).
- **Test-Last-Regel**: `--test-threads=1` für serielle Vergleiche; ~50 Tests fallen nur parallel durch (`docs/dev/ctox-refactoring-board.html:429`). Niemals parallelen gegen seriellen Lauf vergleichen (Fehlermuster #4).
- **HTTP-Daten-Grenze**: Browser-Daten dürfen NIEMALS über HTTP gehen — nur WebRTC/RxDB. Wenn Sync bricht, ist der WebRTC-Pfad zu fixen, kein HTTP-Fallback (`AGENTS.md` Business OS Data Boundary).
- **Vier Mandanten auf Binären vom 16.06.–07.08.**: Flotten-Vereinlichung ist Voraussetzung für sinnvolle Performance-Messung am lebenden System.
- **Erfüllbarkeits-Falle**: „Alles grün" ist unerreichbar bei 59 roten Basis-Tests (Fehlermuster #9, `refactoring-board.html:469`). Tore müssen „keine NEUEN roten Tests" lauten.
- **DocumentCache-GC ist unvollständig**: `sweep_dead_cache_items` (`doc_cache.rs:68`) löscht nur bei leeren `by_rev`, aber `apply_change_events` aktualisiert `latest` für tote IDs weiter — RAM-Leak über Daemon-Uptime.

## 6. likely failure modes

1. **Schnitt hier, Bruch dort** — Fundament-Plan S1–S4: 83 `repair_`/`reconcile_`/`legacy_`-Funktionen in rxdb_peer.rs belegen, dass das Nebeneinander zweier ID-Welten (S2) und nicht-idempotente Projektionen (S3) noch existieren. Ein Performance-Tuning an der Projektions-Loop trifft diese Schuldklassen.
2. **Stille Wächter-Abschaltung** — bereits passiert (Board: „die Zeile fehlte"). Meta-Test (`6a917e13b`) schützt jetzt dagegen, aber NUR für `module_size_tests`. Weitere Wächter haben keinen Selbstschutz.
3. **Fremdarbeit-Kollision** — 209 dirty Einträge im Hauptcheckout; ein Schnitt an `store.rs` überschreibt möglicherweise uncommittete Fremdarbeit.
4. **Cache-Buster-Vergessen** — eine `src/`-Änderung ohne `dist/`-Rebuild + 3× `?v=`-Bump führt zu stale-Browser-Code, der still strandet (`AGENTS.md`).
5. **Idle/Active-Oszillation** — ein einziger aktiver Tick resettet `consecutive_idle_rounds`; bei churn-heavy Multi-Mandanten-Betrieb nie wirklich idle → 12 Loops bleiben auf 3s.
6. **DocumentCache-OOM** — churn-heavy IDs (commands, queue_tasks, chunk-Generationen) füllen den Cache unbegrenzt über Daemon-Uptime.

## 7. decisive tests (die 5 Messungen, die den Plan tragen/kippen)

1. **Projektions-Durchsatz**: Miss `projected_documents / tick` bei 171 Collections mit `BUSINESS_RECORD_PROJECTION_PAGE_SIZE` 25 vs. 100 vs. 250. Erwartung: Linearer Gewinn. Methode: `record_native_peer_loop_result`-Metriken aus `rxdb_peer.rs:878-909` (Loop-Metrics bereits instrumentiert).
2. **DocumentCache-RSS über Daemon-Uptime**: Lasse den Peer 1h mit Command-Churn laufen, miss `cached_document_count()` (`doc_cache.rs:218`) + Prozess-RSS. Kriterium: wenn count nach 1h > 50k Einträge → H-3 bestätigt, LRU/SIZE-Cap ist Prio 1.
3. **HTTP-Verbindungs-Overhead**: Aktiviere Keep-Alive in `server.rs:3821/3845`, miss Shell-Boot-Zeit (vorher 129 Anfragen) mit `Connection: keep-alive` vs. `close`. Erwartung: −20–30% Latenz bei parallel-limited Pool (Board: „ab der fünften parallelen Anfrage serialisiert der Pool").
4. **Idle-Loop-Somnolenz**: Miss die effektive Poll-Rate aller 12 Loops über 10 Minuten bei 0 User-Aktivität. Wenn >50% der Loops nie auf `STANDBY_RECONCILE_INTERVAL_SECS (30min)` zurückgehen → H-4 bestätigt, Idle-Backoff ist unwirksam.
5. **59-Rot-Klassifikation**: Führe `cargo test --bin ctox -- --test-threads=1` aus, vergleiche mit `docs/dev/beweise/rot-basis.txt`. Kriterium: welche der 59 sind echte Regressionen (Post-`6a917e13b`) vs. veraltet vs. Umgebung. Ohne diese Zahl ist kein Tor erreichbar.

## 8. recommended additions to the final brief (Plan-Gerüst)

**Stufe A — Stabilisierung (seriell, eine Datei)**

| Tor | Inhalt | Messung |
|---|---|---|
| A1 | **59-Rot-Klassifikation** (R-01/GROK-8): echte Regression / veraltet / Umgebung. | Bekannte, dokumentierte Zahl statt Unbekannte. |
| A2 | **Wächter-Selbstschutz verallgemeinern**: Meta-Test-Muster aus `6a917e13b` auf alle `mod`-Zeilen in `main.rs` erweitern, nicht nur `module_size_tests`. | `rg "mod " src/core/main.rs` — jede Zeile hat einen Guard-Test. |

**Stufe B — Performance-Sofortmaßnahmen (parallel nach Datei)**

| Track | Datei | Hebel | Parallele Ausführung |
|---|---|---|---|
| B1 | `server.rs` | Keep-Alive + Worker-Pool (H-2) | ✅ isoliert |
| B2 | `doc_cache.rs` | LRU/Size-Cap für DocumentCache (H-3) | ✅ isoliert |
| B3 | `rxdb_peer.rs` | PAGE_SIZE-Tuning (25→250) + Idle-Backoff-Verhärtung (H-1, H-4) | ❌ kollidiert mit C1 |

**Stufe C — Refactoring-Schnitte (seriell pro Datei, parallel verschiedener Dateien)**

| Schnitt | Datei | Art | Aufwand | serialisiert gegen |
|---|---|---|---|---|
| C1 | `rxdb_peer.rs` | **Semantisch**: Projektionen → `store_projections.rs` (bereits 1.804 Zeilen, erweitern); 83 repair/reconcile/legacy-Funktionen an der Wurzel fixen (S1–S3 aus Fundament-Plan) | Sol-Tier | B3 |
| C2 | `store.rs` | **Umzug**: `store_outbound_commands.rs` (bereits 10.797) als eigenes Modul vollständig; Revision/Envelope nach `store_projections.rs` | Sofort | — |
| C3 | `service.rs` | **Semantisch**: SYNC-F Runde 2 (I-070…I-074), atomarer Attempt-Abschluss | Sol-Tier | — |
| C4 | `app.js` | **Strukturell**: Export-Barriere brechen — mindestens Test-Hooks exportieren | Mittel | dist-Rebuild |

**Parallelisierbarkeit:**
- B1 (`server.rs`) ‖ B2 (`doc_cache.rs`) ‖ C2 (`store.rs`) ‖ C3 (`service.rs`) — alle verschiedene Dateien.
- B3 (`rxdb_peer.rs`) muss VOR C1 laufen (Tuning vor Schnitt) oder NACH C1 (Sicht auf saubere Nähte). Empfehlung: B3 VOR C1, weil Performance-Baseline an der bestehenden Struktur gemessen wird.

**SYNC-F Runde 2 Einordnung:**
- I-070 (Mission-Seed): `service.rs`, parallel zu C2/B1/B2.
- I-071 (atomarer Attempt-Abschluss): `service.rs`, **blockiert alle anderen service.rs-Schnitte** — größtes Los, zuerst.
- I-072/073/074: `service.rs`, seriell nach I-071.

**Tore (alle erreichbar formuliert):**
- Stufe A: „Die Anzahl roter Tests ist eine dokumentierte Konstante, nicht größer als vor der Aktion."
- Stufe B: „Shell-Boot-Anfragen ≤ 129 (nicht mehr), DocumentCache-RSS nach 1h Churn ≤ Ausgangswert + 10%."
- Stufe C: „Keine Datei über Budget (Ratsche fällt nur); keine NEUEN roten Tests; Vereinigung nach Normalisierung = Original (für Umzüge)."

## 9. unresolved questions

1. **Welche der 83 `repair_`/`reconcile_`/`legacy_`-Funktionen in rxdb_peer.rs sind noch aktiv vs. tot?** Der Fundament-Plan fordert Löschung als Akzeptanz — aber ohne Persistenz-Messung (welche Schreibpfade feuern real?) kann nicht zwischen „an der Wurzel fixen" und „ersatzlos löschen" unterschieden werden.
2. **Ist `BUSINESS_RECORD_PROJECTION_PAGE_SIZE = 25` tuning-empfindlich?** Wurde dieser Wert jemals variiert, oder ist er historisch unbegründet? Ein Erhöhen auf 250 könnte die Lock-Hold-Time vervielfachen.
3. **Sind die 12+ Hintergrund-Loops voneinander abhängig?** Command-Consumer (1s aktiv) feuert Projektionen an — eine Verlangsamung der Projektions-Loop könnte den Command-Consumer blockieren.
4. **Wie viele der 178 Collections haben real Daten?** Wenn 150/178 leer sind, ist der Projektions-Loop ein Cursor-Reset-Marathon ohne Nutzen — ein `WHERE EXISTS`-Pre-Filter wäre der größte Hebel.
5. **Wann wird die Fremdarbeit im Hauptcheckout committet?** Ohne dieses Datum sind alle `store.rs`/`sync.js`/`dist/`-Schnitte blockiert.

---

```workjet-completion-receipt-v1
{"schemaVersion":1,"status":"completed","summary":"Discovery-Analyse des CTOX Sync-Engine Refactoring- und Optimierungsplans. Prämissen geprüft: mod module_size_tests IST auf HEAD (main.rs:107), alle Größenangaben bestätigt, app.js hat 0 Exports. 5 Performance-Hebel identifiziert (Projektions-Loop über 171 Collections, Connection: close, DocumentCache-RAM-Leak, 12+ Polling-Loops, Initial-Replication über 178 Collections). Schnittkarte erstellt: bereits extrahierte Submodule inventiert, verbleibende Schuldklassen (83 repair/reconcile/legacy-Funktionen) lokalisiert. Plan-Gerüst mit 3 Stufen, messbaren Toren und Parallelisierbarkeits-Matrix vorgeschlagen.","changedFiles":[],"verification":[{"command":"git status --porcelain | wc -l","result":"0"}],"concerns":["Prämisse 'mod module_size_tests fehlt' ist am HEAD veraltet (seit 6a917e13b wieder da inkl. Meta-Test)","DocumentCache-GC-Lücke (doc_cache.rs:155-160) ist live und nicht im Optimierungsplan als gelöst markiert","59 rote Tests nicht klassifiziert — ohne diese Zahl ist kein Tor erreichbar","Fremdarbeit im Hauptcheckout (209 dirty) blockiert store.rs/sync.js/dist-Schnitte"],"producedPaths":[]}
```
