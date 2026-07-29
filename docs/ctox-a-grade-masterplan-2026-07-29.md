# CTOX A-Grade-Masterplan (2026-07-29)

**Owner-Direktive:** Feature-Freeze. Die gesamte Codebasis wird auf A-Niveau
gebracht, bevor neue Features entstehen. Nichts wird beschönigt. Abarbeitung
in virtuellen Wochen unter Workjet (Sol = Completion, Kimi = Review/Greenfield,
MiniMax = Bulk, Orchestrator = Plan/Integration/Final-Edit).

**Warum:** Die Feature-Geschwindigkeit ist auf null gefallen, weil jede
Änderung Seiteneffekte in wild gewachsenem Code auslöst. Die Seele von CTOX
steht (Harness → Pivot → Business-OS); die Codebasis bildet sie nicht ab.
Drei God-Files tragen den Kern (store.rs 68k, rxdb_peer.rs 25k, app.js 12k),
~94 Text-Dispatch-Stellen machen Fehlermeldungen zu API, und mehrere Kerne
(mission, service, communication) wurden nie reviewt.

## Definition „A" (global, messbar — gilt für jedes Paket)

1. Unabhängige Re-Review (Kimi, frischer Kontext, gleiche Rubrik): **keine
   offenen HIGH-Findings**, Note ≥ A−.
2. **Kein Kontrollfluss über Fehlermeldungs-Texte** (Guard erzwungen).
3. **Keine repair_/reconcile_-Funktionen**, deren Existenz einen defekten
   Schreibpfad kaschiert — Akzeptanz ist die Löschung, nicht die Verbesserung.
4. **Generierte Wahrheitsquellen werden konsumiert** (Contracts, Inventare,
   Schema-Registry) — keine handgepflegten Parallellisten; Drift-Checks in
   der Validierung.
5. **Modulgrenzen laut Architektur-Manifest** (entsteht in W5) — Guard gegen
   neue Querimporte.
6. Tests: Verhaltens-Regression oder benannte Architektur-Invariante — keine
   Implementierungs-Pins; Suite grün; ▲-Änderungen mit Rot-Beweis.
7. clippy/fmt clean; Doku (docs/ctox-rxdb.md u. Geschwister) deckt den
   Ist-Zustand.

## Betriebsregeln (aus den Vorfällen dieser Kampagne, verbindlich)

- **Ein schwerer Job zugleich** (cargo-Läufe nie parallel zu Builds/Fleets).
  Vor jedem Lauf: `uptime` + `df -h /System/Volumes/Data`.
- Commits über **Temp-Index + commit-tree + CAS** (`update-ref old new`) —
  der geteilte Index gehört niemandem. Vor Verifikation eines „toten"
  Workers: `pgrep` (Wrapper-Exit ≠ Worker-Tod).
- `git show "$REV:pfad"` IMMER quoten (zsh-:s-Falle); jeden hash-object-Blob
  vor Commit inhaltlich prüfen.
- Browser-Proofs: Tab < 2 GB, RSS mitmessen, sofort schließen.
- Worker-Reports sind Claims: Scope-Diff, eigene Testläufe, Assertion-Diff
  gegen HEAD — immer.
- Reviews laufen gegen HEAD-Snapshots, nie gegen fremd-dirty Worktrees.
- Ein Ticket = ein Commit mit `Backlog:`-Trailer; Wellen-Serialisierung auf
  geteilten Dateien ist verbindlich.

## Strategie-Prinzip: Schnitt zuerst (Owner-Entscheid 29.07.)

**Die God-Files werden aufgelöst, und zwar FRÜH — als Move-only-Schnitte vor
der Semantik-Arbeit.** Begründung (Owner): Ohne die Schnitte ist paralleles
Arbeiten an Modulen unmöglich — jede Welle und jede Session serialisiert
sich an denselben drei Dateien (belegt: 4 Index-Kollisionen, erzwungene
Serialisierung C1→C4→C8, Fleet-Gates).

Dieser Entscheid **ersetzt die Adjudikation vom 27.07.** („Konsolidieren
statt zersplittern; rxdb_peer.rs bleibt bewusst EINE Datei"). Die neue Regel:

- **Schnitt entlang der Verantwortungs-Nähte, move-only, mit Kontrakt an der
  Grenze** — keine Utility-Konfetti-Files, aber auch kein 25k-Zeilen-Kern.
- Move-only-Schnitte sind risikoarm (Verhalten identisch, Tests laufen
  unverändert) und kommen deshalb VOR die verhaltensändernden Tickets.
- **Danach parallelisiert die Semantik-Arbeit pro Modul:** C1/C4/C8 berühren
  nach dem Schnitt drei VERSCHIEDENE Dateien und laufen gleichzeitig statt
  seriell. Dasselbe Muster für store.rs (W1) und app.js (W2).

Ziel-Schnitte:
- `rxdb_peer.rs` → `peer_lifecycle` · `projections` · `command_consume` ·
  `desktop_files` · `browser_control` · `iot_supervision`
- `store.rs` → `command_plane` · `module_lifecycle` · `store_projections` ·
  `backup_restore` · Policy-Anteile nach `policy.rs` (Schnittkarte final aus
  P7f)
- `app.js` → `shell-core` · `window-manager` · Fassaden (`live-sync`,
  `live-command-bus`) · `module-catalog` · `startup` (je exportierbar =
  testbar)
- `service.rs` → Schnittkarte aus dem W3-Review.

## Wochenplan

### W0 — SYNC-A/B-Abschluss (läuft)
Ziel: Datenebene fertig, Command-Plane-HIGHs zu.
- ✅ bis B1b (36+ Tickets, Welle F 34/34, R-RED, G1/G5).
- **D-Welle ZUERST (Schnitt-zuerst-Prinzip):** rxdb_peer.rs-Schnitt in
  6 Module (Peer-Lifecycle / Projektionen / Command-Consume / Desktop-Files /
  Browser-Control / IoT), move-only, serielle Commits pro Extraktion, nach
  jedem Commit das Root-Crate-Testnetz.
- **DANN parallel** auf den geschnittenen Dateien: C1 (Policy-Extraktion,
  nach P7d-Abgleich), C4 (Zustandsmaschine im Peer-Lifecycle), C8
  (Revision-Minting in den Projektionen).
- G-Welle Rest nach Gate-Fall von business-chat.js/sync.js: G2 (UI-Wahrheit),
  G3 (generischer Re-Drive), G4 (Multi-Tab-Receipt), G11/G12
  (Lifecycle-Milestones, Inventar browserfähig + Drift-Guard), B4b.
- T1–T4 (Konstanten-Pins, redundante Source-Assertions, Invarianten-Header,
  Regex→Verhalten wo Seams existieren) — MiniMax/Sol, klein.
- Installer-Fix (Binär-Bundle-Validator) + Browser-Sichtproof Welle F.
- **Wochenabschluss: Kimi-Re-Review der Pakete P1–P6 → Ziel je ≥ A−.**

### W1 — SYNC-C: store.rs (Plan nach P7-Review, läuft gerade)
Ziel: Der CTOX-Kern wird wartbar.
- P7-Synthese → SYNC-C-Plan (Kimi-adversarial gegengeprüft, wie SYNC-A v2).
- Erwartete Wellen (nach jetzigem Kenntnisstand, P7 kann korrigieren):
  Dedup in place → Schuldklassen an der Quelle (42 Text-Dispatches raus,
  20 repair/reconcile-fns durch korrekte Schreibpfade ersetzen) →
  Schnitt in Module: `command_plane.rs`, `module_lifecycle.rs`,
  `projections.rs`, `backup_restore.rs`, Policy konsolidiert zu policy.rs →
  Guards.
- **Parallelspur T7** (disjunkte Dateien): Rust-Testbarkeit — geteilte
  Fixtures, Storage-Seams, `tokio::time::pause` für Chaos-Tests. Senkt die
  45-min/40-GB-Läufe und macht alle Folgewochen billiger.
- Wochenabschluss: P7-Re-Review ≥ B+ (A folgt nach W5-Feinschliff).

### W2 — Shell-Woche: app.js + Browser-Command-Ende
Ziel: Die Shell wird testbar, die Command-UX ehrlich.
- T5-Schnitt: Fassaden (live-sync, live-command-bus), Window-Manager-Kern,
  Katalog-Loader, Startup-Retry, Reconnect-Repair aus app.js in
  importierbare shared-Module (Sol; Verhalten identisch, Exporte + Tests).
- T4-Rest: die ~10 Regex-Pin-Testdateien durch Verhaltenstests ersetzen.
- T6: die 5 fehlenden Schlüssel-Regressionstests (Reconnect-datenerhaltend,
  Multi-Tab-Handover, Accept-Crash+Retry-Idempotenz, Journal-Doppel-Replay,
  Mid-Session-Schema-Drift).
- G7 (Poller→Events), G9 (Cancel-Semantik), G10, Chat-Workarounds 1–3 raus.
- Wochenabschluss: Kimi-Review Shell-Paket ≥ A−.

### W3 — Betrieb & service.rs
Ziel: Upgrades laufen, Fehler eskalieren, Feld ist messbar.
- service.rs: Review-Paket + Dedup/Schnitt (dieselbe Behandlung wie P7).
- Upgrade-Pfad end-to-end: Bundle-Validator nach Bundle-Typ, Fehler
  eskalieren statt Dauerbanner, Rollback-Weg verifiziert.
- Beobachtbarkeit: command_plane-/Transport-/Readiness-Zähler in eine
  lesbare Statusfläche (CLI + Status-Drawer); Heartbeat-Ausfall eskaliert.
- Plattendruck-/Loadwächter für den Daemon (No-space war 50 Min. unsichtbar).
- Wochenabschluss: Production-Readiness-Kriterien neu benoten — Ziel:
  kein Kriterium < B.

### W4 — Die nie reviewten Kerne
Ziel: Keine dunklen Flächen mehr.
- Review-Pakete (Kimi, HEAD-Snapshots): mission/context/autonomy ·
  communication/mailserver · capabilities (browser/scrape/web/doc) ·
  install (Rest) · office_engine-Grenze (Vendor-Disziplin) ·
  web_stack/appsec-Flächen · execution-Gateway (Kurz-Audit, gilt als sauber).
- Maßnahmen-Wellen daraus, gleiche Disziplin; Umfang erst nach Review ehrlich
  schätzbar — **dieser Punkt kann W4 in zwei Wochen teilen.**
- Wochenabschluss: jedes neue Paket ≥ B+, HIGHs zu.

### W5 — Architektur-Zement & Gesamtabnahme
Ziel: A bleibt A.
- `docs/architecture-modules.md`: das Modul-Manifest (die Soll-Landkarte aus
  der Modul-Analyse) — pro Modul Zweck, Owner-Datei(en), erlaubte Abhängig-
  keiten.
- Import-/Grenz-Guards (Rust: Modul-Sichtbarkeiten + Test; JS: Import-Guard
  im bestehenden data-plane-Guard-Stil).
- Gesamt-Re-Review ALLER Pakete (frische Kimi-Kontexte, eine Rubrik) —
  **Abnahme = jedes Paket ≥ A−, kein HIGH offen, Production-Readiness
  überall ≥ B+.** Offene Funde werden Nacharbeits-Tickets vor Freeze-Ende.
- Erst danach endet der Feature-Freeze.

## W4-Folgewellen (konkretisiert nach den Erstreviews, 29.07. abend)

Grundlage: `docs/ctox-w4-core-reviews-2026-07-29.md`. Reihenfolge je Paket
nach dem Schnitt-zuerst-Prinzip. Bereits gelandet (Sofortfixes, weil die
Dateien frei waren): Mailserver-Backdoor gelöscht (b49e34d0c),
Passwort-Hashing (d3ba4a5ab), SMTP-250-Ok-Mailverlust (50111e9d5),
scrape-Transient-Klassifikation (in Arbeit).

### COMM-Welle (communication, D+ → A) — größte Einzelbaustelle
1. **COMM-CUT** (move-only, 3 Commits): JS aus meeting_native als echte
   `runner/*.js`-Dateien via include_str!; chat_native →
   providers/{slack,discord,telegram,matrix,mattermost,zulip,google_chat} +
   realtime + error + platform-Tabelle; email_native →
   http/imap/smtp/ews/activesync/graph/mime.
2. **COMM-ERR**: getypter `ProviderError { status, retry_after, code }` —
   ersetzt den retry_after-Format-Parse-Roundtrip (chat_native.rs:4266→4110)
   und die 50-Substring-Klassifikation (3878-3980).
3. **COMM-OUTBOX**: echter Zustell-Zustandsautomat
   (queued→sent→confirmed/failed mit Retry/Backoff, Slack-Backoff als
   Vorlage); löscht die `queued-<digest>`-Fabrikation und den
   `let _ =`-Meeting-Sendepfad. Akzeptanz: kein `"ok": true` mehr bei
   fehlgeschlagenem Send; Digest-Kollisionstest.
4. **COMM-SESS**: Meeting-Session-Status als Enum an einer Seite der
   Rust/JS-Grenze; reconcile_stale_running_session gelöscht (Runner-Exit
   schreibt selbst).
5. **COMM-FAKE**: `ctox-fake`-Magic-String raus aus Produktionspfaden
   (typisierte Test-Injektion).

### MAIL-Welle (mailserver, D → A) — Rest nach den Sofortfixes
1. **MAIL-DKIM**: korrekte relaxed/relaxed-Kanonikalisierung + RFC-Header-
   Menge; fail-closed statt MOCK_SIGNATURE_FAIL/SHA-als-Signatur.
2. **MAIL-IMAP**: persistente UIDs + UIDVALIDITY, \Deleted-Flag + echtes
   EXPUNGE.
3. **MAIL-SMTP-REST**: case-insensitive Envelope-Parse, DSN nur bei
   MAIL FROM:<>, System-Resolver statt 8.8.8.8-Handparser, Puffer-Limits +
   Dot-Unstuffing, user_exists-Fehler fail-closed, Delivery-Outcome als Enum,
   AUTH-Rate-Limit, tote Config (throttle/max_connections) implementieren
   oder löschen.

### CAP-Welle (capabilities, C → A)
1. **CAP-CUT** (move-only): scrape.rs → cli/registry/execute/classify/
   reauth/enrichment/semantic/materialize/templates/repair; Tests raus.
2. **CAP-CLASSIFY**: classify als Zustandsautomat; `ok`/Exit-Code-Semantik
   ehrlich (kein ok:true bei blocked).
3. **CAP-PATHS**: Workspace-Pfade relativ schreiben; Read-Path-Repair
   (load_registered_target-UPDATEs) löschen; die 5 Pin-Tests auf
   Schreibpfad-Verhalten umstellen.
4. **CAP-TRUST**: Protected-Config aus Registry statt String-Parsing des
   untrusted Skripts; Lock mit Liveness-Check; skip_probe ohne fabrizierte
   Beobachtung; SSRF-Grenzen dokumentieren + testen.

### MISSION-Welle (mission, C → A)
1. **MISSION-CUT** (move-only): channels.rs → queue_store/command_saga/
   outbound_send/review_approvals/business_os_projection (+ PDF raus);
   tickets.rs → cli/schema/self_work/workflow/knowledge/cases/source_skills.
2. **MISSION-STATUS**: EIN Status-Enum-Satz statt ~7 String-Vokabulare;
   die 4 String→CoreState-Karten löschen.
3. **MISSION-PROOF**: Terminal-Policy-Proofs als typisierte, signierte
   Struktur statt Actor/Reason-Prefix + request_json-LIKE.
4. **MISSION-REPAIR**: repair_stale_step_routing_state /
   reconcile_business_command_invariants löschen, Schreibpfade atomar;
   set_queue_routing_status_tx-Doppelbesitz auflösen.
5. **MISSION-SCHED**: Meeting-Join-Retry budgetieren; Cron dom/dow
   POSIX-konform.

### CTX-Welle (context, B− → A) — kleinste Welle
1. **CTX-CUT**: lcm.rs → engine/compaction/continuity/mission_state/
   assurance/search/cli/fixture.
2. **CTX-STATE**: Mission-State strukturiert (Feldset + einmalige
   Migration) statt Freitext-Parse+Repair.
3. **CTX-REST**: "no such table"→Ok(0)-Gate schließen, Audit-Puffer
   persistent, Timestamps typisieren, autonomy-Testlücken.

Sicherheits-Sofortliste (nicht auf Wellen warten, Dateien frei):
- [x] Backdoor-Token (b49e34d0c)
- [x] Klartext-Passwörter (d3ba4a5ab)
- [x] SMTP-Mailverlust hinter 250 (50111e9d5)
- [ ] CAP-TRUST Teil 1: untrusted Allowlist (mit CAP-Welle oder vorgezogen)
- [ ] MAIL-DKIM fail-open (mit MAIL-Welle oder vorgezogen)

## Ehrliche Risiken

1. **W4 ist eine Wundertüte** — mission/communication können weitere
   God-Files enthalten; der Plan weist das als Teilungs-Risiko aus, statt es
   wegzuschätzen.
2. **Parallel-Sessions am Checkout** bleiben die größte Prozessgefahr
   (4 Index-Vorfälle, 1 Merge-Verschlucken). Gegenmittel sind etabliert
   (CAS-Commits, Claims in der Kampagnen-Memory), eliminieren es aber nicht.
3. **Maschinen-Budget:** Root-Crate-Läufe kosten bis zur T7-Landung 30–60
   Min.; die Ein-Job-Regel drosselt den Durchsatz bewusst — Stabilität vor
   Tempo (zwei Abstürze waren genug).
4. **Fremd-dirty Gates** (business-chat.js, sync.js, store.rs-Worktree)
   bestimmen die Reihenfolge mit; der Plan hält je Woche eine Parallelspur
   bereit, damit Gates nie Leerlauf erzeugen.

## Tracking

Board (stabile URL): https://claude.ai/code/artifact/c2a8e190-bde5-477e-96db-9262b48dbcf9
— wöchentliches Zeugnis (Komponenten-Noten + Production-Readiness) wird je
Wochenabschluss fortgeschrieben. Kampagnen-Memory ist der
Koordinationskanal zwischen Sessions.
