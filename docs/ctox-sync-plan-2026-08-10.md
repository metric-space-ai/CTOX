# CTOX Sync-Engine: Optimierung + Refactoring v2 — konsolidierter Plan

Stand 10.08.2026 · Ersetzt die Stufenfolge des Sanierungsplans vom 09.08.
(Missionstafel `docs/dev/ctox-refactoring-board.html`) und führt sie mit der
Performance-Spur zusammen. Nachfolger von SYNC-A…F; SYNC-F Runde 2 wird hier
als eigene Stufe eingeordnet, nicht ersetzt.

Entstehung: Workjet-Discovery-Panel (Grok 4.5 / Luna 5.6 / GLM 5.2, identischer
Brief, 10.08. vormittags), Reports gesichert im Session-Scratchpad
(`report-A-grok.md`, `report-B-luna.md`, `report-C-glm.md`). Jede tragende
Panel-Behauptung wurde vom Orchestrator am HEAD nachgemessen; Abweichungen sind
unten als Korrekturen ausgewiesen. **Unabhängiges adversariales Review (Kimi)
steht aus** — beide Kimi-Worker sind mit 403-Quota bis zum nächsten
Abrechnungszyklus down (Probe 10.08. 08:18 UTC). Das Review ist Stufe-5-Tor,
kein stiller Verzicht.

## 0. Verifizierter Ist-Stand (Prämissen, am HEAD nachgemessen)

| Prämisse | Befund | Beleg |
|---|---|---|
| Größen-Wächter | **Registriert auf HEAD** (`src/core/main.rs:107`), inkl. Meta-Test, der jede `*_tests.rs` ohne `mod`-Anmeldung namentlich meldet | `d328d3a1d` + `6a917e13b` (10.08.) |
| Budget-Ratschen | **HEAD: 0 Verstöße.** store 27.316/27.516, office_engine 13.953/13.953, outbound 5.195/5.270, channels 7.174/7.221, rxdb_peer 12.718/12.718, service 26.177/26.237 | Orchestrator-Messung nach Guard-Regel (letzter `#[cfg(test)]`) |
| Dirty Arbeitsbaum | **~136–209 Einträge** (je nach Zeitpunkt/Checkout). Die uncommittete Arbeit würde die Ratsche reißen (store.rs dirty ≈ 28.413 prod = +1.097 über Budget; ebenso office/outbound/channels) | Panel A+B unabhängig; Orchestrator-Differenzmessung HEAD vs. Worktree |
| Collections | **178** im Schema-Vertrag, nicht ~195; Laufzeitbestand ist dynamisch | `business_os_schema_contract.json` (178 Keys); `rxdb_peer.rs:11296ff` |
| Rote Tests | 59 seriell reproduzierbar, **unklassifiziert**; ~50 weitere nur unter Parallellast | `docs/dev/beweise/rot-basis.txt` (59 Zeilen) |
| Flotte | Vier Mandanten auf Binären 16.06.–07.08.; leadyoda-Vorfall zeigte: Statuscode-Prüfung lügt (Fehlertafel = HTTP 200) | Board |
| origin/main | 3 Commits voraus (u. a. `c5f13fc91`) — Fetch/Merge-Disziplin nötig | `git log main..origin/main` |
| Workjet-Isolation | **Leck:** Worker-Checkouts trugen die 136 fremden dirty Einträge des Haupt-Baums | Receipts Panel A+B |

## 1. Leitprinzipien

1. **Erst das Netz, dann der Umbau** — bestätigt und teilweise schon erfüllt
   (Wächter + Meta-Test sind gelandet).
2. **Tore erreichbar formulieren:** immer „keine NEUEN roten Tests gegen die
   dokumentierte Basis", nie „alles grün" (Fehlermuster #9).
3. **Messung vor jeder Performance-Änderung:** kein Hebel ohne Vorher-Zahl und
   definierte Nachher-Metrik. Baseline bei vergleichbarem Daemon-Alter.
4. **Nie Perf-Semantik und Umzug derselben Funktionsspanne in einer Welle.**
   Reihenfolge pro Datei: Baseline-Metrik pinnen → reiner Umzug (Ratsche im
   selben Commit runter) → Perf-Änderung mit Mess-Tor.
5. **Datengrenze ist hart:** kein HTTP-Fallback für Sync-Daten; Perf-Arbeit am
   Datenpfad nur über WebRTC/Batch/Checkpoint/Priorisierung
   (`AGENTS.md`, `docs/ctox-rxdb.md:83-99`).
6. **Wire-Kompatibilität:** Vertrags-/Checkpoint-Änderungen brauchen
   Mixed-Version-Nachweis gegen das älteste Flotten-Binär oder einen
   dokumentierten Upgrade-Zwang (Owner-Entscheid).

## 2. Stufen

### S0 — Baum & Basis (Blocker für alle Integrations-Commits)

- **Dirty-Baum-Triage (Owner + Orchestrator):** die ~136 geänderten + 34
  unversionierten Dateien je Fläche entscheiden: übernehmen (committen),
  verwerfen, oder als benannte Baustelle isolieren. Sonderfall: die gestagten
  Löschungen von `docs/dev/beweise/*` und der Missionstafel sind
  Index-Kontamination und werden per `git restore --staged` geheilt, nicht
  committet.
- **origin/main einholen** (3 Commits) per Plumbing-Merge, Worktree unberührt.
- **Baselines erheben und versionieren:** Boot-Requests (129), Initial-Sync-
  Dauer, Projektionen/min im Idle, `cached_document_count()`, RSS.
- **Tor:** Integrationslinie definiert; Beweise unter Git; Baseline-Datei
  committet. *Kein* Erzwingen von porcelain 0 — der geteilte Checkout bleibt
  Realität, aber jede Kampagnen-Datei ist entweder committet oder benannt.

### S1 — Netz vervollständigen (parallelisierbar, kleine Flächen)

- **R-01 (Worker):** die 59 roten Tests klassifizieren — je Test: seit welchem
  Commit, echte Regression / veralteter Test / Umgebung. Auftrag liegt in
  `docs/dev/beweise/R-01.md`. Lehre aus SL-RED gilt: umgekehrte Erwartungen
  sind nur mit Verursacher-Commit als Beleg zulässig.
- **Erreichbarkeits-Wächter je Mandant (Worker):** Inhalt prüfen, nicht
  Statuscode (leadyoda-Lektion).
- **Ratchet-Politik festschreiben:** Budgets werden NIE erhöht; die dirty
  Fremdarbeit muss vor ihrem Commit unter Budget geschnitten oder die Ratsche
  im selben Commit ehrlich per Schnitt gesenkt werden.
- **Tor:** Rote-Zahl ist dokumentierte Konstante mit Klassifikation; Wächter
  laufen auf main; keine neuen Roten.

### S2a — SYNC-F Runde 2 (strikt seriell, nur `service.rs` + enge Caller)

I-070 Mission-Seed → **I-071 atomarer Attempt-Abschluss** (das große Los:
dauerhafter Finalisierungs-Datensatz, typisiertes Success-Outcome, Zeuge VOR
Zähler-Reset, idempotente Wiederaufnahme) → I-072 Repair-Telemetrie →
I-073 Sweep-Audit + Queue-Dedupe → I-074 CV-Gate typisiert.
Quelle: `docs/ctox-service-plan-2026-08-05.md:52-66`.
**Tor je Ticket:** Vorher-Ereigniszähler fällt oder stabilisiert; keine neuen
Roten; keine Budget-Erhöhung. Kein weiterer service.rs-Schnitt vor I-071.

### S2b — Move-only-Nähte (parallel zu S2a, andere Dateien)

Muster S-02c (bewiesen: 25.292 → 22.658, Vereinigung nach Normalisierung =
Original, Zeilenbilanz, Namens-Schnittmengen-Check, Ratsche im selben Commit):

1. `rxdb_peer.rs` Restnähte: Desktop-Datei-Index (ab :3543), Demand-Chunk-
   Streaming (:8470–8605), Projektions-Loop-Gerüst — NUR das mechanische
   Gerüst, keine Stamp-/Scheduling-Semantik.
2. `store.rs`: nächste reine Command-/Projektions-Extrakte analog
   `store_projections.rs` / `store_catalog_projections.rs`.
3. `office_engine.rs` nur bei Budget-Druck (nicht sync-kritisch, Rand).

**Tor:** je Naht phys/prod ↓ und Ratsche ↓ im selben Commit; Guard grün;
HEAD-Build aus `git archive` bewiesen (Move-Commit-Regel).

### S2c — Shell-Testbarkeit (parallel, JS, keine dist-Welle)

Seams aus `app.js` (12.321 Zeilen, 0 Exports): `openBusinessDataPlane`
(:1179ff), `loadModules` (:9346ff), Maintenance-Monitor (:8603ff),
Icon-Registry (:8336ff) → Node-importierbare Module ohne Verhaltenswechsel.
**Tor:** mind. 1 Node-Smoke importiert einen Seam; Boot-Request-Baseline
nicht schlechter; die Regex-Pin-Tests aus dem T0-Audit werden ablösbar.

### S3 — Performance-Welle (nach/zwischen S2b-Nähten; Mess-Tor Pflicht)

Sechs belegte Hebel, Reihenfolge nach Aufwand/Nutzen; jede Zeile vom
Orchestrator am Code verifiziert:

| # | Hebel | Evidenz | Messung (vorher/nachher) | Aufwand |
|---|---|---|---|---|
| P1 | **BLOCKIERT (negatives Ergebnis 10.08.):** `Connection: close` ist ABSICHT — `a429b596d` (22.06.) führte die raw-Writer ein, weil Chromiums ES-Modul-Graph auf Keep-Alive-Loopback hängen blieb. Wiedereröffnung nur mit reproduzierter Chromium-Regression + Browser-Beweis über persistente Verbindung | `server.rs:3821,3845`; Historie `a429b596d` | erst Repro, dann Browser-Proof des Modul-Graphen | — |
| P2 | **DocumentCache-Deckel (LRU/Size-Cap)** — `latest` wird bei jedem Aufruf für tote IDs weiter geklont; Sweep greift nur bei leerem `by_rev`, alle 256 Aufrufe | `doc_cache.rs:150-160,68-73` | `cached_document_count()` + RSS nach 1 h Command-Churn (Kriterium: ≤ Ausgang +10 %) | S–M |
| P3 | **WIDERLEGT (Messung 11.08., docs/dev/beweise/plan-v2-p3-messung.md):** 7/12 Loops schlafen im Idle vollständig, Command-Consumer hält 30-s-Intervall, kein 3-s-Polling in Ruhe. Kein Idle-Hebel; Rest-Ticket: Reset-Empfindlichkeit unter Last | Peer-Status performance.loops | erledigt (10-min-Differenzmessung) | — |
| P4 | **WIDERLEGT (Messung 11.08., docs/dev/beweise/plan-v2-p4-messung.md):** Frischer Boot synct eager nur 15 Prioritäts-Collections (alle complete in ~1 min, 19 MB Heap, 125 Requests) — das Demand-Loading priorisiert bereits. Folgefrage klein: Latenz der Demand-Collections beim ersten Modul-Öffnen | Browser-Diagnose ctoxBusinessOsSyncDiagnostics | erledigt (Frisch-Boot-Messung) | — |
| P5 | **EXISTIERT bereits (Befund 11.08., docs/dev/beweise/plan-v2-p5-messung.md):** Batch-Größen pro Collection-Klasse an gemessenen Dokumentgrößen + Frame-/Transfer-Grenzen ausgerichtet (desktop_chunks 6 halbiert Roundtrips, knowledge_tables 1 gegen Frame-Ceiling, Roundtrip-Zahlen im Code belegt). Kein Bauauftrag; Rest = Feintuning unter echter Last | `sync-contract.js:48-66` | erledigt (statisch) | — |
| P6 | **Projektions-Scan inkrementell** — Page 25, O(Collections×Docs) bei Stamp-Miss; offene Kernfrage: wie viele der 178 tragen real Daten? Pre-Filter evtl. größter Einzelhebel | `rxdb_peer.rs:750,6211-6400,12164-12192` | rows touched/cycle + duration aus `NativePeerLoopMetrics` (:845-909, bereits instrumentiert) | M |

**Tor:** je Hebel ein vorab definierter p95-/Chattiness-Gewinn; keine neuen
Roten; keine neuen Full-Resyncs; Browser-verifiziert (Sichtproof, Tab < 2 GB);
P1 erfordert Anpassung des pinnenden Tests mit Begründung, nicht dessen
Löschung. Semantische Projektions-Vereinheitlichung (ein Runner, eine
Stamp-Disziplin — Fundament-Plan S3) ist Sol-Tier und läuft VOR P6-Tuning.

### S4 — Abnahme & Flotte

- Mixed-Version-Probe: ältestes Flotten-Binär gegen neuen Stand (oder
  dokumentierter Upgrade-Zwang; hängt an der Owner-Entscheidung Release-
  Upgrade brand-demo/leadyoda).
- Browser-Sichtproof auf ruhiger Maschine; RSS-Messung während des Proofs.
- Die fünf entscheidenden Messungen (unten) liegen als Vorher/Nachher vor.
- **Kimi-Re-Review der Kampagnen-Commits, sobald Quota zurück** — bis dahin
  gilt jede Selbst-Abnahme als vorläufig.

## 3. Parallelisierungs-Matrix

| gleichzeitig möglich | strikt seriell |
|---|---|
| R-01 ∥ Erreichbarkeits-Wächter ∥ Peer-Umzüge (S2b) ∥ app.js-Seams (S2c) ∥ S2a (andere Datei) | S0 vor allen Integrations-Commits |
| P1 (`server.rs`) ∥ P2 (`doc_cache.rs`) ∥ S2a (`service.rs`) — drei Dateien | I-070 → I-071 → I-072 → I-073 → I-074 |
| P5-Messung (Browser) ∥ P3-Messung (nativ) | Umzug → dann Perf in derselben Region |
| — | Projektions-Semantik (ein Runner) vor P6 |
| — | je EINE dist-Welle: src-Edit → esbuild → 3× `?v=`-Buster → Tests |

## 4. Die fünf entscheidenden Messungen

1. **Initial-Sync:** leere Browser-DB → Zeit bis CRITICAL live / letzte
   Collection live; Roundtrips × batchSize; Peer-Loop-Metrics-Snapshot.
2. **Idle-Kosten:** Poll-Rate aller Loops + Projektionen/min bei unverändertem
   Store über 10 min; danach Einzel-Änderung → genau eine idempotente Projektion.
3. **Cache-Wachstum:** `cached_document_count()` + RSS nach 1 h Churn.
4. **Reconnect/Resume:** Abbruch mitten in Pull/Push/Projektion → Zeit bis
   Konvergenz, doppelt gelesene Dokumente, Checkpoint-Invalidierungen —
   mit altem UND neuem Binär.
5. **Shell-Boot:** Request-Zahl (Baseline 129) und Maintenance-Polls/min
   (Baseline 1,6) — Browser-verifiziert, nicht per Statuscode.

## 5. Risiken (aus drei unabhängigen Reports konsolidiert)

- Schnitt im dirty Baum nimmt Fremdzeilen mit oder committet Ratschen-Riss
  (dirty store.rs liegt +1.097 über Budget).
- Checkpoint zu früh gestempelt → stille Lücken (`ctox-rxdb.md:702-706`).
- Batch-Erhöhung ohne Frame-Cap-Messung strandet Chunks (`knowledge_tables=1`
  ist ein bewusster Deckel).
- „Move-only" mit verstecktem Semantikwechsel — besonders service.rs
  (Outcome-Zeuge/Zähler-Reihenfolge ist genau der I-066-Defekt).
- dist-Drift: src geändert, Buster nicht → Mandant fährt alten Code weiter.
- Workjet-Worktree-Leck: Worker sehen den fremden Schmutz — Briefe müssen
  HEAD-bezogen messen lassen (`git show HEAD:pfad`), nicht Worktree-Dateien.
- Wächter-Grenze bleibt: der Meta-Test kann seine eigene Abwesenheit nicht
  erkennen (im Commit dokumentiert) — Wächter-Commits gehören ins Review.

## 6. Offene Owner-Fragen

1. Dirty-Baum-Triage: Welche der ~136 Änderungen sind aktive Baustellen
   welcher Session, was darf verworfen werden? (blockiert S0)
2. Wire-Freeze: dürfen Batch-/Checkpoint-Felder in dieser Kampagne wandern,
   oder erst nach Flotten-Upgrade? (bestimmt P4/P5-Radius)
3. Release-Upgrade brand-demo/leadyoda (bestehende Board-Frage; blockiert S4).
4. Zielmetrik-Priorität: p95-Boot, RPCs/MiB, Idle-CPU oder RAM? (bestimmt
   P-Reihenfolge; Vorschlag: Boot p95 zuerst — kundensichtbar)
5. Verwaiste Auth-Assist-Arbeit übernehmen oder verwerfen? (Board-Rückstau)

## 7. Worker-Routing (Workjet)

| Paket | Worker | Bemerkung |
|---|---|---|
| R-01-Klassifikation | Sol (Brief liegt: `docs/dev/beweise/R-01.md`) | Baseline-Beweise Pflicht; keine Fixes |
| I-070…I-074 | Sol, seriell, je eigener Brief | Whitelist nach Aufrufergraph, nicht Befund-Datei |
| S2b-Umzüge | Sol oder direkt (Mechanik bewiesen) | Zeilenbilanz + Schnittmengen-Check im Report |
| P1/P2 | Sol (klein, testbar) | P1: pinnenden Test :4149 begründet anpassen |
| P3–P6 | erst Messung (direkt/Grok), dann Sol-Welle | keine Semantik in Mess-Briefs |
| Erreichbarkeits-Wächter | Sol | Inhalt-Prüfung, je Mandant |
| Review | **Grok 4.5** (Owner-Freigabe 10.08.: Kimi-Quota bis 18.08. erschöpft, Grok als Vertretung autorisiert) | Kimi-Re-Review der Gesamtkampagne nach dem 18.08. bleibt S4-Tor |

Bekannte Panel-Schwäche dieser Runde: beide Erst-Läufe von Grok und Luna
starben nach ~7 min mit Gateway-Transportfehlern; Reruns liefen durch.
Lange Briefe weiter in gesicherte Etappen schneiden.
