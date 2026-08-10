# Discovery-Report: CTOX Sync-Engine — Optimierung + Refactoring v2

## 1. approach
1. Pflichtlektüre (Board, Budgets, SYNC-F, Sync-Pläne, AGENTS/ctox-rxdb) und Prämissen am Code (main.rs, Budget-Zählregel, `wc -l`).
2. Produktionszeilen vs. `contracts/module_size_budget.txt` exakt nach der Guard-Regel (letzter `#[cfg(test)]`).
3. Performance-Hotspots in `rxdb_peer.rs`, `sync.js`/`sync-contract.js`, `app.js`, Schema-Contract (178 Collections) gelesen.
4. Schnittkarte aus Board + vorhandenen `store_*`/`rxdb_peer_browser`/`business_os_app_testing`-Mustern abgeleitet.
5. Plan-Gerüst: Netz → Klassifikation → serielle `service.rs`-Runde-2 → billige Umzüge → Perf-Hebel; parallele vs. serielle Pfade markiert.

## 2. prototype paths or no-code evidence
| Beleg | datei:zeile / Kommando |
|---|---|
| Größen-Wächter **ist** registriert | `src/core/main.rs:107` `mod module_size_tests;` |
| Budget-Ratsche + Zählregel | `contracts/module_size_budget.txt:4-21`, `src/core/module_size_tests.rs:50-66,65-99` |
| Phys-Zeilen = Brief-Angaben | `wc -l`: store 43781, service 46039, rxdb_peer 22658, office 16888, mcp 11673, outbound 10797, app.js 12321, sync.js 3026, business-chat 6559 |
| Prod vs Budget (Working Tree) | OVER: store 28413>27516, office 14598>13953, outbound 5354>5270, channels 7226>7221; UNDER: lcm 5600<5627, service/business_os 6060<7106, service.rs 26224<26237; OK: rxdb_peer 12718=12718 |
| Board: Wächter-Geschichte + 59 Rot + Stufen 0–3 | `docs/dev/ctox-refactoring-board.html:185-242,288,339,429` |
| 59 Rot-Namen | `docs/dev/beweise/rot-basis.txt` (59 Zeilen) |
| SYNC-F Runde 2 I-070…I-074 | `docs/ctox-service-plan-2026-08-05.md:52-63` |
| Schema-Collections **178** (nicht ~195) | `src/core/business_os/business_os_schema_contract.json` (178 keys); Loader `rxdb_peer.rs:12135-12155` |
| Multiplex: 1 Session statt ~88 PC/DC | `rxdb_peer.rs:2495-2507` |
| Projection-Loops + Intervalle | `rxdb_peer.rs:3781-3901`, Command-Poll `756-761`, `4395-4432` |
| Projection Write-Batch 250 | `rxdb_peer.rs:751`, `6891-6924` |
| Browser batchSizeFor 1/6/8/20 | `src/apps/business-os/shared/sync-contract.js:48-65`; Nutzung `sync.js:1214-1267` |
| Pull/Push Default 10 im Runtime | `rxdb/src/replication-webrtc.mjs:874-875,1222-1263` |
| Shell-Maintenance Poll 2s/60s Idle | `app.js:88-95`; Critical Collections `129-133` |
| NO HTTP-Fallback Sync | `sync.js:6-7`; `docs/ctox-rxdb.md:93` |
| dist-Rebuild + 3× Cache-Buster | `AGENTS.md:96-98,195-196`; `sync.js:8-10,25-26,324` |
| app.js: **keine** `export`-Zeilen | `rg ^export app.js` → 0 (Brief: „0 Exports“ bestätigt) |
| S-02c Umzugsmuster | Board `afa5b21c0`, `rxdb_peer_browser.rs` 2681 phys; Peer phys 22658 |
| store bereits teil-zerlegt | `store_projections.rs` 1804, `store_outbound_commands.rs` 10797, `store_catalog_projections.rs` 1619 |
| service bereits teil-zerlegt | `business_os_app_testing.rs` 1231, `business_os.rs` prod 6060 |
| Test-Parallelität unsicher | Board `:429` (`--test-threads=1` für Vergleiche); RxDB `:1026` in `docs/ctox-rxdb.md` |
| Escape: Worktree **nicht** clean | `git status --porcelain \| wc -l` → **136** (Brief: isoliert/leer — falsch) |

## 3. commands run and results
- `rg -n "mod module_size_tests" src/core/main.rs` → Treffer Zeile 107.
- `wc -l` auf die 9 Brief-Dateien → Größen exakt wie Brief (phys).
- Python `production_lines()` analog Guard → 4 OVER / 3 UNDER / rest OK.
- `python json` auf `business_os_schema_contract.json` → **178** Collections.
- `rg` Hotspots: Polls/Batches/Multiplex/Projections in `rxdb_peer.rs`, `sync-contract.js`, `app.js`.
- `rg` Struktur: Peer ~575 `fn`, Cluster sync/desktop/project/command/demand.
- `wc -l docs/dev/beweise/rot-basis.txt` → 59.
- `git status --porcelain | wc -l` → 136 (vor und nach; keine Schreibzugriffe durch diesen Worker).
- **Nicht** gelaufen (verboten): cargo build/check/test, node-Tests, npm, esbuild, Netz.

## 4. difficulty 1-5 with reasons
**Difficulty: 5**

Gründe: (a) zwei God-Files (`store.rs` ~28k prod, `service.rs` ~26k prod) mit laufenden semantischen Fixes (SYNC-F) und gleichzeitigem Umzugsdruck; (b) Browser+Native-Wire-Vertrag mit dist/Cache-Buster-Disziplin und **null** HTTP-Fallback; (c) 178 Collections Initial-Sync + Dutzende Hintergrund-Loops; (d) unklare Rot-Basis (59) macht „grün“ als Tor unbrauchbar; (e) dirty Shared-Checkout-Kultur (Board Note-F) + Mandanten auf alten Binären — jeder Perf-Gewinn muss rollout-fähig sein, nicht nur main-grün.

## 5. hidden constraints
1. **Dirty Worktree-Realität:** Dieser Checkout hat 136 porcelain-Einträge; Brief-Prämisse „isolierter Worktree, clean HEAD“ ist hier falsch. Jede Naht kollidiert mit Fremdschmutz (Board Stufe 0).
2. **Budget-Ratsche bidirektional:** Wachstum **und** zu hohes Budget nach Schnitt sind rot (`module_size_tests.rs:76-91`). Schnitte **müssen** Budget im selben Commit senken.
3. **Board-Claim „HEAD unter Budget“ ist inkonsistent:** Board nennt store 28413 prod und Budget 27516 — das ist bereits OVER, unabhängig vom Schmutz (`board:223` vs `budget:37`).
4. **dist + drei identische `?v=`-Buster** nach jedem Browser-RxDB-`src/`-Edit (`AGENTS.md:96-98`); sonst laufen Mandanten alte Bundles.
5. **Kein HTTP-Datenpfad** für Collection-Daten (`sync.js:6`, `ctox-rxdb.md:93`) — Perf nur über WebRTC/Batch/Checkpoint/Priorisierung.
6. **Test-Last:** Suite parallel unzuverlässig (~50 nur-parallel-rot); Vergleiche seriell; RxDB-Doku: `--test-threads=1` (nicht „max 4“ als harte Projektregel in AGENTS — Board empfiehlt 1 für Vergleiche).
7. **Flotte:** vier Mandanten, Binäre 16.06–07.08 (Board Stufe 3) — Wire-Änderungen brauchen Kompatibilität/Upgrade-Plan.
8. **app.js untestbar aus Node** (keine Exports) — Perf/Refactor an Shell braucht zuerst Test-Seams (Fundament-Plan T5-Idee).
9. **Worker-Platten-/Parallel-Limit:** Board warnt ENOSPC, Sol ≤3 parallel, workjet-Runs marken.

## 6. likely failure modes
1. **Silent-Wächter-Wiederholung:** `mod module_size_tests` ist zurück, aber **Meta-Test fehlt** noch (Board todo) → erneutes Löschen bleibt unsichtbar.
2. **Schnitt in dirty tree:** Commit nimmt Fremdzeilen mit oder nimmt eigene zurück (Note-F: flate2/hashing/module_size).
3. **Perf-Hebel an denselben Dateien wie SYNC-F:** I-070…I-074 serialisieren `service.rs`; parallele Peer-Perf in `rxdb_peer.rs` kollidiert mit Peer-Nähten.
4. **Batch-Erhöhung ohne Frame-Cap-Messung:** `knowledge_tables`/chunks sind bewusst klein (`sync-contract.js:56-60`); blindes 20 überall strandet Frames.
5. **Uniform native multiplex batches** (`rxdb_peer.rs:2504-2507`) vs. browser `batchSizeFor` — einseitige Tuning-Drift.
6. **Projection double-writer** historisch (Kommentare knowledge/desktop_files `12158-12194`) — generische Business-Record-Projection wieder einschalten = Daten-Flip.
7. **dist ohne Buster / Buster-Drift** → „Fix wirkt lokal, Mandant unverändert“.
8. **Tor „alles grün“** bei 59 bekannten Roten → Arbeit blockiert endlos (Board failure #9).
9. **Attribute-Waise bei Zeilenschnitten** (Board:435) und fehlende `pub(crate)`/Re-Exports (Board:436).

## 7. decisive tests — 5 Messungen, die den Plan tragen/kippen
1. **Wächter-Live + Budget-Inventur:** `cargo test module_size_stays_at_its_declared_budget` (wenn freigegeben) + Meta-Test „mod-Zeile existiert“; Tor: Guard läuft, bekannte OVER-Liste dokumentiert, **keine neuen** OVER-Dateien.
2. **Rot-Basis R-01:** dieselben 59 Namen aus `rot-basis.txt` seriell (`--test-threads=1`) vs. aktuell; Klassifikation Regression/veraltet/Umgebung; Tor: Anzahl Rot = bekannte Zahl, **0 unklassifiziert**.
3. **Initial-Sync Catch-up:** leerer Browser-DB → Zeit/Roundtrips bis `collectionReadinessState=live` für CRITICAL + N größten Collections; Metriken: pull round-trips × batchSize, WebRTC bytes, Peer `NativePeerLoopMetrics` Snapshot (`rxdb_peer.rs:845-903,1139+`).
4. **Projection-CPU/Idle:** unter Null-Last: Häufigkeit Command-Poll (1s→30s Backoff `756-761`) und der 7 Background-Projection-Loops; Tor: idle Sleep greift, keine volle Collection-Scan-Schleife ohne Stamp-Change (`3831-3901`).
5. **Shell-Boot HTTP:** DevTools Request-Count + Maintenance-Poll/min (Baseline Board: 208→129, ~30→1,6/min); Tor: **keine Regression** der Request-Zahl und Idle-Poll bleibt 60s (`app.js:95`), verifiziert im Browser nicht per Statuscode.

## 8. recommended additions to the final brief — Plan-Gerüst (Aufgabe 4)

### Prämissen-Korrekturen (verbindlich in den Kampagnenbrief)
- `mod module_size_tests;` **ist** auf diesem HEAD registriert (`main.rs:107`) — historisch aus, heute an; Meta-Wächter fehlt noch.
- Collection-Contract = **178**, nicht ~195.
- Phys-Größen des Briefs stimmen; **Budget-Lage:** mindestens store/office/outbound/channels **über** deklariertem Budget (Working Tree / Board-eigene 28413).
- Worktree-Clean-Annahme hier falsch (136 dirty) → Stufe-0 bleibt realer Blocker außerhalb dieses Workers.

### Refactoring-Schnittkarte (billig vs. semantisch)

| Fläche | Billige Umzüge (S-02c-Muster) | Semantische Wellen (Sol-Tier) |
|---|---|---|
| `rxdb_peer.rs` (22658 phys / 12718 prod) | desktop_file (~110 Namens-Treffer), projection-loops, demand_file, command_plane; Lifecycle/metrics schon gekapselt | Batch/backpressure, checkpoint validity+filter digest, required/optional bring-up |
| `store.rs` (43781 / 28413) | F-01b `store_revision` (Board fertig gerettet); weitere Command-Cluster analog `store_*` | Projection-Semantik, Grants, push_collection transaction counting |
| `service.rs` (46039 / 26224) | Nach SYNC-F: belangsweise Module (repair telemetry, sweep audit, CV-gate) **nur** als Move nach Fix | **I-070…I-074 strikt seriell** in einer Datei |
| `app.js` (12321, 0 exports) | Extract: boot/registry single-flight, maintenance poll, critical-sync warm, data-plane readiness → importierbare Module | Sync-Recovery-Orchestrierung, multi-tab/direct failover |

**Kollisionsminimierung:**  
- **Nie** gleichzeitig Perf-Semantik und Umzug **derselben** Funktionsspanne.  
- Reihenfolge pro Datei: (1) Baseline-Metrik pin, (2) reiner Umzug + Budget-Ratsche runter, (3) Perf-Änderung mit Mess-Tor.  
- `service.rs`: nur SYNC-F Runde 2, **keine** parallelen Schnitte.  
- `rxdb_peer.rs`: Umzüge der Restnähte **parallel zu** service-Arbeit möglich (andere Datei); Perf an Peer **nach** oder **zwischen** Nähten, nicht in derselben PR-Welle.  
- Browser (`sync.js`/`replication-webrtc`/`sync-contract`) parallel zu Rust-Umzügen, **wenn** Wire unverändert; Wire-Änderungen brauchen Fixture-Regen + dist + Buster.

### Stufen v2 (Netz + Umbau + Perf vereint)

**S0 — Arbeitsnetz (Owner/Plattform, blockiert produktive Nähte)**  
Arbeit: dirty decide/commit/discard; Budget-Verletzer klären; Fremd-Auth-Assist.  
**Tor:** saubere Integrationslinie für Schnitte (idealerweise porcelain 0 auf Integrationsbranch); `cargo check` aus frischem Clone grün.  
*Discovery-Worker kann S0 nicht herstellen — nur voraussetzen.*

**S1 — Wächter & Messbarkeit (parallelisierbar, wenig Datei-Kollision)**  
- Meta-Test: `mod module_size_tests` muss existieren.  
- Budget-Inventur: echte OVER/UNDER-Liste committen (Ratsche angleichen **ohne** Wachstums-Freigabe — nur ehrliche Zahlen oder echte Schnitte).  
- **R-01:** 59 Rot klassifizieren (`rot-basis.txt`).  
- Optional: Instanz-Erreichbarkeit Inhalt≠200, Disk-Schwellwert (Board).  
**Tor:** Wächter+Meta grün; Rot-Zahl = dokumentierte Menge; **keine neuen** roten Tests in den angefassten Paketen.

**S2a — SYNC-F Runde 2 (strikt seriell, nur `service.rs` + enge Caller)**  
I-070 Mission-Seed → I-071 atomarer Attempt-Abschluss → I-072 Repair-Telemetrie → I-073 Sweep-Audit → I-074 CV-Gate.  
**Tor je Ticket:** Vorher-Metrik (Ereigniszähler) fällt oder stabilisiert; **keine neuen** roten Tests; keine Budget-Erhöhung.

**S2b — Billige Umzüge (parallel zu S2a, **andere Dateien**)**  
Priorität:  
1. `rxdb_peer` Restnähte (desktop_file / projection / command / demand) — Muster S-02c, Normalisierung=Original.  
2. `store` F-01b revision + nächste pure command/projection-Extrakte.  
3. `office_engine` nur soweit Budget-Druck, **nicht** Sync-kritisch (Rand).  
**Tor:** je Naht phys/prod↓ und Budget-Ratsche↓; Vereinigung nach Normalisierung = Original; Guard grün.

**S2c — Shell-Testbarkeit (parallel, JS)**  
Seams aus `app.js` (boot, maintenance, critical sync) → Node-importierbar; **kein** Verhaltenswechsel.  
**Tor:** mind. 1 Node-Smoke importiert Seam; Request-Baseline nicht schlechter.

**S3 — Performance-Hebel (nach oder zwischen S2b-Nähten; Mess-Tor Pflicht)**  
Die 5 stärksten **belegbaren** Hebel:

| # | Hebel | Evidenz | Warum | Messung | Aufwand |
|---|---|---|---|---|---|
| 1 | **Initial-Sync Fan-out über 178 Collections** (Master-Registrierung + Catch-up) | schema 178; bring-up `2438-2518`; multiplex `2495-2507` | Jede Collection kostet Handler/Checkpoint/Catch-up-Frames; Critical-only warm im Shell (`app.js:129-133`) vs. Peer registriert alle | Catch-up-Zeit, Frames/Collection, Zeit bis CRITICAL live | M–L (Demand/Priority, nicht „weniger Collections löschen“) |
| 2 | **Browser batchSize vs. Frame-Cap** | `sync-contract.js:48-65`; pull loop `replication-webrtc.mjs:1222+` | Roundtrips dominieren große Chunks; knowledge=1/chunks=6 sind bewusste Deckel | Roundtrips pro MB, Truncation-Häufigkeit, Fehler rate | S–M (profil-basiert, mit Cap-Tests) |
| 3 | **Native Hintergrund-Projektions- & Command-Loops** | Loops `3781-3901`; Command 1s/30s `756-761`,`4395-4432` | Idle-CPU/IO und Write-Lock-Kontention auf RxDB (`database_write_lock`) | Loop-Metrics Snapshot, idle wake count, Lock-Wait | S–M (Stamp/Wake statt Poll; schon teilweise Wake `4440+`) |
| 4 | **Business-Record-Projection Scan/Upsert** | batch 250 `751`,`6891-6924`; Collection-Filter `12158-12194` | O(collections×docs) bei Stamp-Miss; falsche Writer = Korruption | rows touched/cycle, duration `NativePeerLoopMetrics` | M (inkrementelle Diffs, Writer-Invarianten-Tests) |
| 5 | **Shell HTTP-Chattiness (Boot/Maintenance)** | Maintenance 2s/60s `app.js:88-95`; Board 208→129 | SSH-gebundene Endpunkte multiplizieren Latenz; Cache-Buster-Vielfalt war Treiber | Request count/boot, polls/min, TTF data-plane ready | S (Cache-Proxy Board Stufe3; Keep-alive/Pool) |

**S3-Tor:** Nachher-Messung bei vergleichbarem Daemon-Alter; Browser-verifiziert; **keine** neuen Roten; Wire-Kompatibilität zu ältestem Flotten-Binär **oder** dokumentierter Upgrade-Zwang.

### Parallel vs. seriell (Kurz)
- **Parallel:** R-01 Klassifikation ∥ Meta-Wächter ∥ Peer-Umzüge ∥ JS-Seams ∥ (nach Freeze) Browser-Batch-Tuning ohne Wire-Break.  
- **Seriell:** S0 vor Integrations-Commits; I-070→I-074; Umzug → dann Perf in **derselben** Region; Fixture-Änderung → Regen beide Seiten → dist → Buster.

## 9. unresolved questions
1. Sind die 136 dirty Einträge dieses Worktrees absichtliche Kampagnenarbeit oder Leck aus dem Shared-Checkout — und welche Integrationsbranch ist die Wahrheit für S0?
2. Soll die Budget-Datei **sofort** auf Ist-Prod geratcht werden (ehrliche Ratsche) oder nur **mit** Schnitten fallen? (Guard verbietet stille Erhöhung; Board will keine Wachstumsfreigabe.)
3. Wer besitzt `rxdb_peer.rs` für Restnähte — Plattform-Strang oder Sync-Kampagne? (Board:355 Auslöser offen.)
4. Exakte Definition „Initial-Sync fertig“ pro Collection in Produktion (diagnostics `initialReplicationState` vs. `collectionReadinessState` vs. advanced-status)?
5. Dürfen Batch/Checkpoint-Wire-Felder in dieser Kampagne noch wandern, oder ist Wire eingefroren bis Flotten-Upgrade?
6. Liegt die „max --test-threads=4“-Regel irgendwo als harte CI-Policy, oder gilt nur Board/RxDB-Empfehlung threads=1 für Vergleiche?
7. I-070 Status „läuft“ (Service-Plan) — Branch/Commit-Stand in diesem Tree nicht verifiziert (Tests/Builds verboten).

---

```workjet-completion-receipt-v1
{"schemaVersion":1,"status":"completed","summary":"Read-only discovery for CTOX sync-engine refactor+perf v2: module_size_tests registered; phys sizes match brief; 4 budget OVERs; 178 schema collections; 5 perf levers + staged plan (S0 net, S1 guards/R-01, S2a SYNC-F serial, S2b pure moves, S3 measured perf).","changedFiles":[],"verification":[{"command":"git status --porcelain | wc -l","result":"136 (pre-existing dirty tree; no writes by this worker)"},{"command":"rg -n 'mod module_size_tests' src/core/main.rs","result":"107:mod module_size_tests;"},{"command":"python production_lines vs contracts/module_size_budget.txt","result":"OVER store/office/outbound/channels; UNDER lcm/business_os.rs/service.rs; OK rxdb_peer 12718"}],"concerns":["Worktree not clean (136 porcelain) — brief clean-isolation premise false here","Board claim HEAD under budget conflicts with measured/board-stated store 28413>27516","Could not run cargo/node tests (forbidden); 59-red classification status unverified beyond rot-basis.txt listing","Collection count is 178 not ~195"],"producedPaths":[]}
```
