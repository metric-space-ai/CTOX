# Offene Arbeiten — Sync-Engine, Refactoring, Performance

Stand: 14.08.2026. Alle Zahlen sind selbst gemessen; wo eine Zahl fehlt, steht
„ungemessen" statt einer Schätzung.

---

## 0. Der Befund, der alles andere überlagert

**Die App wird nicht langsamer, weil Code langsam ist, sondern weil die
transportierte Datenmenge wächst und niemand sie begrenzt.**

Auf der Kundeninstanz liegen (gemessen am 14.08.):

| Collection | Dokumente |
|---|---:|
| `sellify_activities` | 139.804 |
| `sellify_campaigns` | 86.551 |
| `sellify_people` | 60.640 |
| `sellify_companies` | 17.520 |
| **Summe Sellify** | **304.515** |

Bedarfsgesteuert repliziert werden im ganzen System **genau drei** Collections,
und alle drei sind Dateiblöcke: `desktop_file_chunks`, `document_blob_chunks`,
`spreadsheet_blob_chunks` (`shared/command-bus.js:107`). Für alles andere gilt
**Vollreplikation**. Ein bounded initial window existiert nirgends — Suche nach
`initialPullWindow`, `since_ms`, `maxInitialDocs` im gesamten Sync-Kern: **kein
Treffer**.

Bei Batchgröße 20 (`shared/sync-contract.js:48`) sind das **~15.225 Rundreisen**
allein für Sellify. Jede neue Aktivität verlängert jeden künftigen Erstsync.
Das ist die Erklärung für „immer langsamer statt schneller".

**Wichtig zur Einordnung meiner bisherigen Zahlen:** Die vielzitierten
„15,1 s Boot" wurden auf einem **leeren** Speicher gemessen und beschreiben den
Kundenfall **nicht**. Das Desktop-Modul ist nachweislich korrekt client-first
(liest lokal, rendert sofort). Meine bisherigen Optimierungen zielten auf
Taktraten und Backoffs — also auf Sekunden — während der bestimmende Faktor die
Datenmenge ist.

### OA-1 (P0) — Begrenztes Erstfenster + Bedarfsabruf für große Collections
Die Mechanik existiert bereits (`leaseCollection`, Demand-Query-Loader) und wird
für Geschäfts-Collections nur nicht benutzt. **Fünfter Fall desselben Musters:
die Reparatur ist da, sie wird nicht gerufen.**
- Erstsync auf ein begrenztes Fenster (z. B. neueste N nach `updated_at_ms`),
  Rest über den vorhandenen Demand-Pfad.
- Eigentümer-Konflikt beachten: der Browser-Pull-Pfad (`shared/sync.js`,
  `replication-webrtc.mjs`) gehört der Parallelsitzung; der native Master
  (`src/core/rxdb/**`) gehört dieser Spur. Muss abgestimmt werden.
- **Abnahme:** Rundreisen bis „Modul benutzbar" auf einem Store mit ≥100.000
  Dokumenten, vorher/nachher. Nicht: Tests grün.

### OA-2 (P0) — Messung auf populiertem Store
Es gibt bis heute **keine einzige** Messung auf realistischer Datenmenge. Alle
M1–M5-Zahlen stammen von leeren Instanzen.
- Store der Kundeninstanz kopieren oder synthetisch füllen (≥300.000 Dokumente).
- Dann: Zeit bis erster sichtbarer Inhalt je Modul, Rundreisen, IndexedDB-Größe.

---

## 1. Refactoring

### Erreicht (gemessen)
| Modul | vorher | jetzt |
|---|---:|---:|
| `rxdb_peer.rs` | 25.292 | **8.897** |
| `office_engine.rs` | 13.953 | 13.561 |
| `channels/mod.rs` | 7.221 | 7.035 |
| `service/business_os.rs` | 7.106 | 6.060 |
| `lcm/mod.rs` | 5.627 | 5.310 |
| `store_outbound_commands.rs` | 5.270 | 5.108 |

Sieben Module aus `rxdb_peer.rs` geschnitten. Kein Budget je angehoben, alle
Wächter grün, `main` baut aus der Historie, Arbeitsbaum leer.

### OA-3 (P1) — Die zwei verbliebenen Riesen
| Datei | Produktionszeilen |
|---|---:|
| `store.rs` | **27.342** |
| `service.rs` | **26.244** |
| `app.js` | **12.388** (physisch) |

Hier ist **nichts** Nennenswertes passiert (−174 bzw. −12 Zeilen). Das ist der
berechtigte Teil der Kritik „an der Codequalität sehe ich keine Fortschritte":
Wer ins Repo schaut, sieht zwei Dateien mit 27.000 und 26.000 Zeilen.
- Naht-Kandidaten in `store.rs`: Projektionen, Command-Handler, App-Runtime.
- `service.rs`: Queue-Dispatch, Recovery, Prompt-Aufbau.
- **Abnahme:** reiner Move mit Hash-Beweis je Funktion, Budgets nur gesenkt.

---

## 2. Performance — Bilanz der sechs geplanten Hebel

| Hebel | Ergebnis |
|---|---|
| P1 Keep-Alive | negativ / blockiert |
| P2 doc_cache-Deckel | **eingelöst** (fa1a18ab6) |
| P3 | durch Messung **widerlegt** |
| P4 | durch Messung **widerlegt** |
| P5 Batch-Matrix | existierte bereits |
| P6 | nur unter Churn relevant |
| Parallelstart (Parallelsitzung) | gemessen Faktor **1,25**, nicht 4 |
| Consumer-Backoff-Fix | Latenz **unverändert** |

**Für den Nutzer spürbar besser geworden ist: nichts.**

### Offene, gemessene Werte
| Größe | Wert | Ziel |
|---|---|---|
| Command-Roundtrip p50 | 1,50 s | < 300 ms warm |
| Command-Roundtrip p95 | 6,2 s | — |
| Boot bis alle Collections live (leerer Store) | 15,1 s | < 5 s |
| Bridge-Handshake je Collection | 0,5–2,3 s | Rundreisen senken |

### OA-4 (P1) — Etappenmessung fahren
Die Instrumentierung ist gelandet (AP1, cfc84ad61): sieben Zeitmarken je
Command, opt-in per `command_timing_probe`. **Gefahren wurde sie noch nicht.**
Erst danach dürfen AP5–AP7 (Terminalstatus ereignisnah, Intake-Wake,
Ein-Projektion-Abschluss) angefasst werden — sonst wieder Optimierung auf
Verdacht.

### OA-5 (P1) — Bridge-Handshake-Rundreisen senken
Von der Parallelsitzung als Haupthebel für den Boot identifiziert und gemessen
begründet. Liegt in `replication-webrtc.mjs` / `webrtc-native.mjs` (deren Spur).

---

## 3. Kundeninstanz

### Erledigt und beim Kunden wirksam (14.08.)
- **Schreibschleife 174 → 0 Rev/min**, Kontrolle nach 15 min ebenfalls 0.
- **Grabsteine ~302.000 → 275** (Fix der Parallelsitzung, mit meinem Bau
  ausgeliefert).
- Beides unabhängig gegengeprüft.

### OA-6 (P0) — Die Schleife läuft weiter, nur ohne Schreibvorgänge
Der Daemon verbrennt **58 % eines Kerns** im Dauerzustand (alter Daemon im
Mittel 11 %). Der Sweep läuft **29×/min, kein einziger Leerlauf**, greift jedes
Mal dieselben **6** Dokumente:

| command_type | status | Anzahl |
|---|---|---:|
| `web_stack.person_research` | failed | 4 |
| `outbound.research_source.generate_adapter` | failed | 1 |
| `outbound.research_source.test` | failed | 1 |

Meine Sperre feuert **nie**: sie verlangt `exhausted = 1`, und **keine** der 12
Intake-Failure-Zeilen hat das (Versuche stehen bei 1–2, Budget 5 nie erreicht).
Ich habe das Symptom behandelt, nicht die Ursache.
- **Richtiger Fix:** Dokumente müssen das Kandidatenfenster verlassen können —
  echter Versuchszähler mit Aufgabe, oder `terminal_status` beim endgültigen
  Scheitern. **Kein weiterer Deckel auf das Symptom.**
- **Abnahme:** CPU-Dauerlast < 5 %, `idle_ticks` > 0.

### OA-7 (P2) — Store-Datei kompaktieren
2.287.198.208 Bytes, unverändert. SQLite gibt gelöschte Seiten erst mit `VACUUM`
zurück — blockierender Schritt auf laufender Instanz, eigenes Wartungsfenster.

### OA-8 (P2) — `replicationUp` dauerhaft false
In **allen** Messpunkten beider Sitzungen. Ursache belegt:
`dataChannelOpen: false` bei `poolCreated`/`signalingJoinAccepted` = true — es
war schlicht kein Browser verbunden. Der Wert ist also ehrlich. Offen bleibt:
ist das „niemand schaut hin" oder „niemand *kann* sich verbinden"?

### OA-9 (P2) — Wirksamkeitsnachweis AP3 auf der Instanz
Die ehrliche Gesundheitsmeldung (b8fbf718f) ist ausgeliefert, aber auf der
Kundeninstanz nie unter Last erlebt worden. Braucht eine verbundene Sitzung.

---

## 4. Symptome der Parallelsitzung (deren Spur)

| # | Symptom | Stand |
|---|---|---|
| 1 | Boot 10–15 s | offen, Hebel = Handshake-Rundreisen |
| 2 | Replikation hängt bis zum Neuladen (`masterChangesSince`-Timeout) | offen |
| 3 | Schreibschleife | **erledigt** |
| 4 | Peer meldet Gesundheit, während nichts fließt | im Repo behoben (AP3) |

---

## 5. Methodische Regeln, die diese Kampagne erzwungen hat

1. **Beweiskette in drei Stufen:** im Repo gelandet → beim Kunden ausgeliefert →
   auf der Instanz gemessen wirksam. Keine Stufe ersetzt eine andere.
2. **Verbrauchte Cache-Buster-Namen sind dauerhaft verbrannt** — ein einziger
   Messabruf verbrennt einen Namen.
3. **Drei Fragen vor jedem Befund:** Gibt es das Symbol jetzt im Baum (`grep`)?
   Gab es das je in der Historie (`git log -S`)? Ist der Stand in einer Minute
   noch derselbe? (Bei zwei Sitzungen an einem Arbeitsbaum: nein.)
4. **Testläufe nur mit Trefferzahl zitieren.** `--exact` mit Kurznamen liefert
   „0 passed" und sieht aus wie Erfolg.
5. **`TMPDIR` mit umlenken**, nicht nur `CARGO_TARGET_DIR` — sonst stirbt der
   Bau an der vollen Systemplatte.
6. **Auf leerem Store gemessene Zahlen sagen nichts über den Kundenfall.**

---

## 6. Reihenfolge, die ich vorschlage

1. **OA-6** — CPU-Dauerlast beim Kunden (läuft seit heute, wächst nicht mit).
2. **OA-2** — Messung auf populiertem Store; ohne sie ist jede weitere
   Optimierung geraten.
3. **OA-1** — begrenztes Erstfenster; das ist der Hebel für „client first".
4. **OA-4** — Etappenmessung fahren, dann erst AP5–AP7.
5. **OA-3** — `store.rs` und `service.rs` schneiden.

---

# 7. Übergabeantworten (100 Fragen), Stand 16.08.2026

**Lesehinweis:** Wo ich etwas nicht gemessen habe, steht „**ungemessen**".
Wo es eine Owner-Entscheidung ist, steht „**OWNER**". Ich rate nirgends.

## Allgemeiner Stand und Eigentümerschaft

**1.** `8b8fc7656` war vollständig, ist aber **überholt** — HEAD ist inzwischen
weitergelaufen. Es gibt **0 ungepushte Commits**, aber **25 Stashes** (Herkunft
ungeprüft, vermutlich mehrere Sitzungen) und **37 uncommittete Dateien**.
Messartefakte liegen außerhalb von git unter `/Volumes/tmp/ctox-pipeline/`
(148 Berichte/Briefe) und `/Volumes/tmp/ctox-messbinary/` (Binaries, Logs).
**Warnung: `/Volumes/tmp` war heute zwischenzeitlich komplett ausgehängt.**

**2.** Alle 37 uncommitteten Dateien sind **Browser-Dateien und nicht von mir**.
Meine Spur (`command-bus.js`, `rxdb/src`, `rxdb/dist`, `rxdb_peer*.rs`,
`command_saga.rs`, `command_plane.rs`, `contracts/`) ist **sauber committet**.
Die 37 stammen aus der Parallelsitzung und/oder weiteren Sitzungen —
**Herkunft je Datei habe ich nicht verifiziert**. Nicht anfassen ohne Rückfrage.

**3.** Die Parallelsitzung war **heute 07:12 Uhr aktiv** (Commit `3fb095ce3`),
27 Minuten vor meinem letzten Commit. Ob sie *jetzt* noch läuft: ungeprüft.
Eigentum laut Absprache: **ihr** gehören `shared/sync.js`,
`replication-webrtc.mjs`, `webrtc-native.mjs` und der Bridge-/Handshake-Pfad;
**mir** gehören `command-bus.js`, die übrigen `rxdb/src`-Dateien und der native
Kern. Das **gemeinsame `dist/`** baut, wer zuletzt ändert — **immer aus dem
aktuellen `src`**, nie ein fremdes Artefakt übernehmen (das hat schon einmal
fremde Fixes gelöscht).

**4.** Begonnen, aber nicht committet: **nichts von mir.** Zwei Grok-Läufe
(AP1/AP4) starben am 90-Minuten-Limit; ihre Arbeit ist geborgen, montiert und
committet. Zwischenpatches liegen unter `/Volumes/tmp/ctox-pipeline/ap*.patch`.

**5.** Bekannte Problem-Commits: `1a0fd9fc3` machte `main` historisch
unbaubar (52 Fehler, fehlende Companions) — geheilt durch `03b250f38` +
`0e1f895e0`. Sonst keine zu ignorierenden Commits.

**6.** Weitere verbindliche Dokumente:
- `docs/ctox-sync-plan-2026-08-10.md` — kanonischer Plan
- `docs/dev/ctox-refactoring-board.html` — Missionstafel (Artefakt-URL im Chat)
- `docs/dev/beweise/*` — alle Messprotokolle, u. a.
  `produkt-messung-m1-m5-2026-08-12.md`,
  `schreibschleife-kundenmessung-2026-08-13.md`,
  `r-01-klassifikation-2026-08-10.md`
- `/Volumes/tmp/ctox-pipeline/sol-plan.md` — Sols 8-Pakete-Plan (**nicht in git,
  flüchtig**)

**7.** **OWNER.** Meine Empfehlung: OA-1 bis OA-6 vollständig; OA-7 (VACUUM) und
OA-9 nur mit Wartungsfenster; OA-8 ist zuerst eine Diagnose, keine Arbeit.

## Messumgebung und Beweise

**8.** In `docs/dev/beweise/` (committet) und `/Volumes/tmp/ctox-pipeline/`
(flüchtig). **CPU, Sweep-Frequenz und IndexedDB-Größe existieren nur als Zahlen
im Chat und in diesem Dokument — die Rohdaten sind nicht gesichert.** Das ist
eine Lücke.

**9.** Dokumentiert sind: Schreibschleife (Befehl in
`schreibschleife-kundenmessung-2026-08-13.md`), M1–M3 (Aufbau in
`produkt-messung-m1-m5-2026-08-12.md`). Für CPU/Sweep: `ps -o times=` über 60 s
bzw. `criticalTasks.metrics` aus `business-os-rxdb-peer.status.json` — **nur
hier dokumentiert, kein Skript**.

**10.** MacBook Pro (arm64, macOS), Claude-Browser-Pane als Browser,
Release-Binary aus dem jeweils genannten Commit, lokale Loopback-Instanz auf
`127.0.0.1:8917`. Kundenmessungen: Linux x86_64 VM `ctox-e5ed9648`, glibc 2.39.
**Netzwerkpfad zur VM: Tailscale + SSH.**

**11.** Alle M1–M3-Messungen: **frischer Browser-Tab, leerer Store, nach
Daemon-Neustart**. **Keine einzige Messung auf bestehender IndexedDB.** Das ist
der Kernmangel (siehe OA-2).

**12.** Definitionen, wie tatsächlich verwendet:
- „alle Collections live" = alle 15 Boot-Collections haben
  `initialReplicationState === 'complete'` in `ctoxBusinessOsSyncDiagnostics`
- „Command-Roundtrip" = **zwei verschiedene Größen**, die nicht verwechselt
  werden dürfen: (a) Dokumentzeit `updated_at_ms − created_at_ms`,
  (b) Dispatch-Wanduhr von `dispatch()` bis Rückkehr. Vorher/Nachher **nur
  innerhalb derselben Größe** vergleichen.
- „Zeit bis erster sichtbarer Inhalt" und „Modul benutzbar": **nie definiert,
  nie gemessen.**

**13.** **Keine Aufwärm-, Wiederholungs- oder Ausreißerregeln.** 30 Läufe je
Serie, Median und p95 roh. Zwei Vergleiche wurden dadurch unbrauchbar (Last
unterschiedlich, Debug vs. Release). **Für die Fortsetzung dringend festlegen.**

**14.** **OWNER-Entscheidung nötig.** Vorschlag: `docs/dev/beweise/` für
Ergebnisse (committet) und ein *versioniertes* Verzeichnis für Rohdaten.
`/Volumes/tmp` ist nachweislich unzuverlässig (heute ausgehängt; ein externer
Aufräumprozess löschte dort dreimal Binaries mitten im Lauf).

**15.** Kein sauberer Baseline-Commit vor den Optimierungen definiert.
Brauchbare Anker: `b9ed00757` (SCHNITT-6, vor allen Latenzfixes) oder
`786cb98ca` (vor AP1/AP3/AP4).

**16.** Bekannte Flakes und teure Läufe:
- `cargo test --bin ctox business_os::rxdb_peer::tests::` dauert **~55 Minuten**
- **lastabhängig rot**: Idle-Gate- und Projektions-Zeittests (3 rot bei geringer
  Last, 6 bei hoher — identischer Code)
- **dauerhaft rot, Altbestand**:
  `native_peer_consumes_pending_business_command_written_directly_to_sqlite`
- `service_loop`-Baseline: **383 grün / 5 rot**, klassifiziert in
  `r-01-klassifikation-2026-08-10.md`
- RxDB-JS-Suite: **14 rot**, identisch mit und ohne aktuelle Änderungen
- **Falle:** `cargo test <kurzname> -- --exact` trifft nichts und meldet
  „0 passed" wie Erfolg. **Immer die Trefferzahl zitieren.**

## OA-6: CPU-Dauerlast und Intake-Schleife

**17.** Build vom 14.08. 09:29 UTC aus `main`, SHA-256-Präfix
`2204dcea23f94b438c899dd2dae9bc61`, enthält AP1/AP3/AP4 + Schleifenfix.

**18.** `ps -o times=` auf den Prozess `ctox-real service --foreground`, zwei
Messpunkte 60 s auseinander: 35 CPU-Sekunden je 60 s Wanduhr = 58 %. Zusätzlich
`systemctl --user show -p CPUUsageNSec`. **Prozessweit, nicht je Thread.**

**19.** Aus `~/.local/state/ctox/business-os-rxdb-peer.status.json`, Feld
`criticalTasks[name=business_commands].metrics`: `ticks` 885 → 914 in 60 s,
`idle_ticks` 0, `rows` 5310 → 5484 (= 174/min = 6 je Tick). **Keine Logzeilen —
die Schleife schreibt lautlos.**

**20.** `/home/ctox/.local/state/ctox/ctox.sqlite3`, Tabelle
`business_command_intake_failures`. Abfrage:
`SELECT command_id, attempt, exhausted, resolved_at_ms IS NULL FROM
business_command_intake_failures ORDER BY attempt DESC;` → 12 Zeilen,
**0 mit `exhausted=1`**, 9 unaufgelöst, `attempt` nur 1–2.

**21.** **Ungeklärt — das ist die wichtigste offene Frage von OA-6.** Meine
Hypothese: `resolve_business_command_intake_failures` löst die Zeilen nach einem
erfolgreichen Teilschritt auf, wodurch `MAX(attempt)+1` wieder bei 1 beginnt und
Budget 5 nie erreicht wird. **Nicht verifiziert.** Vor jedem Fix zu klären.

**22.** Erzeugung: `record_business_command_intake_failure`
(`command_saga.rs:2038`) über `store.rs:21007`. Wiederauswahl:
`BUSINESS_COMMAND_RETRY_CANDIDATE_SQL` in `rxdb_peer_intake.rs` (nach AP4 dort,
vorher `rxdb_peer.rs`) — selektiert `status='accepted' OR (status='failed' AND
terminal_status='none')` für sechs Command-Typen.

**23.** **OWNER/fachlich.** Diese Typen (`web_stack.person_research`,
`outbound.research_source.*`) sind Rechercheläufe gegen externe Quellen. Ob ein
endgültig gescheiterter Lauf manuell wiederaufnehmbar sein muss, ist eine
Produktentscheidung, die ich nicht treffen kann.

**24.–28.** **OWNER/Design.** Meine Empfehlung: **beides atomar** — Versuch
fortschreiben *und* bei Ausschöpfung terminalisieren, in *einer* Transaktion,
damit ein Absturz dazwischen keinen Command verliert oder doppelt ausführt. Der
Zustand nach Ausschöpfung sollte `terminal_status='failed'` **und**
`execution_phase='terminal'` **und** eine aufgelöste Intake-Failure-Zeile
umfassen, sonst bleibt das Dokument im Kandidatenfenster. Audit: Transition-Zeile
+ Outbox-Ereignis (Muster existiert bereits in `command_saga.rs:2110-2130`).
Manueller Retry: **neue Command-ID** — sonst kollidiert er mit dem
Idempotenz-Aggregat.

**29.** **Nein.** Der einzige existierende Test
(`exhausted_conflicting_command_is_not_rewritten_or_recorded_again`) bildet den
*erschöpften* Konfliktfall ab — **also genau den Fall, den es auf der
Kundeninstanz nicht gibt.** Ein Fixture für den Live-Fall fehlt.

**30.** Zusätzlich zu CPU < 5 % und `idle_ticks` > 0: `rows`-Zuwachs = 0 über
15 Minuten, Kandidatenmenge der SQL-Abfrage = 0 Dokumente, und die
Revisionsrate bleibt bei 0. **Stabilitätsfenster: mindestens 1 Stunde**, weil
der 30-s-Idle-Pfad sonst nicht sicher erreicht wird.

## OA-2: Populierter Store

**31.–33.** **OWNER.** Es sind echte Kundendaten (Firmen, Personen,
Aktivitäten). Ich habe sie **nur aggregiert gezählt**, nie Inhalte gelesen oder
kopiert.

**34.** Zwingend: die vier Sellify-Collections in den gemessenen Größen
(139.804 / 86.551 / 60.640 / 17.520), plus `business_commands` (5.964) und
`desktop_file_chunks` (3.690) als Kontrastfälle.

**35.** **Ungeprüft.** Ob ein Seeder existiert, habe ich nicht untersucht.

**36.–39.** **OWNER/Design.** Meine Empfehlung: Erstlauf mit leerer IndexedDB
*und* Wiederanlauf mit bestehender messen — der Unterschied **ist** die
Client-First-Frage. Rundreisen als vollständige Request/Response-Paare zählen.
Zielwerte: zunächst nur Vorher/Nachher-Bilanz, Zielgrößen danach festlegen.

## OA-1: Begrenztes Erstfenster

**40.–44.** **OWNER/Design.** Fachlich unbeantwortet: Fenstergröße, Skopus
(Mandant/Benutzer), und ob `updated_at_ms` überall gepflegt und **nativ
indexiert** ist — Letzteres ist eine Codefrage, die ich **nicht geprüft** habe
und die vor jeder Umsetzung geklärt sein muss.

**45.–47.** **Die kritischen Korrektheitsfragen, und sie sind offen.** Ein
begrenzter Erstpull darf seinen Checkpoint **nicht** so setzen, als sei der
ausgelassene Bereich repliziert — sonst gehen Löschungen und Tombstones
außerhalb des Fensters dauerhaft verloren. Ich halte einen dritten
Readiness-Zustand („teilweise geladen") für nötig; `live` wäre gelogen.

**48.–52.** **Ungemessen.** Welche Queryformen der Demand-Pfad heute bedient
(Sortierung, Pagination, Counts, Aggregationen), habe ich nicht erhoben. Das ist
der erste Prüfschritt von OA-1, **vor** jedem Umbau.

**53.** Kein neuer Prozess-Env-Schalter (Projektregel). Entweder typed config /
Runtime Store oder fester Collection-Vertrag.

**54.** Nötig sind vermutlich Änderungen auf **beiden** Seiten (Master liefert
begrenzt, Browser fragt begrenzt) — also **quer über die Eigentumsgrenze**.
**Mit der Parallelsitzung war das nicht abgestimmt.**

**55.** **Nein.** Kein Entwurf, kein Spike, kein Vertrag. Suche nach
`initialPullWindow`, `since_ms`, `maxInitialDocs`: **kein Treffer**.

**56.** Zwingend zu testen: Reconnect, Peerwechsel, Multi-Tab, Schemawechsel,
Berechtigungswechsel, Offline-Wiederanlauf — plus **Löschung außerhalb des
Fensters**, der gefährlichste Fall.

## OA-4: Etappenmessung

**57.** `client_context.command_timing_probe = true` im Dispatch. Nativ wird nur
dann gemessen und geschrieben.

**58.** Sieben Marken (`browser_dispatch_started`, `browser_local_inserted`,
`browser_push_confirmed`, `native_dispatch_entered`, `native_handler_completed`,
`native_rxdb_projection_committed`, `browser_terminal_observed`), gesetzt in
`command-bus.js` und `command_plane.rs`, zusammengeführt von
`src/apps/business-os/rxdb/tests/command-roundtrip-stage-report.mjs`.

**59.–61.** Vorgesehen war `ctox.provider_subscription.status` (No-op) mit
30 Läufen; der Auswertungsbefehl erwartet Browser-JSON + Daemon-Log.
**Die Serie wurde nie gefahren** — das Skript ist unerprobt.

**62.** AP5–AP7 existieren **nur als Beschreibung** in
`/Volumes/tmp/ctox-pipeline/sol-plan.md`. Kein Branch, kein Patch.

**63.** Hypothese (aus Codelektüre, **ungemessen**): der Rückweg
Projektionscommit → Browserbeobachtung dominiert. Widerlegt wäre sie, wenn die
native Verarbeitung oder der Push den Löwenanteil trägt.

**64.** **Ja** — Priorisierung strikt nach gemessenem Engpass, auch wenn er
außerhalb von AP5–AP7 liegt. Genau das ist der Zweck des Pakets.

## OA-3: `store.rs` und `service.rs`

**65.** Ja: `src/core/business_os/store.rs` und `src/core/service/service.rs`.

**66.–67.** Nahtkandidaten sind **meine Einschätzung, keine Analyse**:
`store.rs` → Projektionen, Command-Handler, App-Runtime; `service.rs` →
Queue-Dispatch, Recovery, Prompt-Aufbau. **Keine Abhängigkeitsanalyse, keine
Nahtkarte, keine verworfenen Versuche.**

**68.** Referenzmuster: die sieben Schnitte aus `rxdb_peer.rs`
(`rxdb_peer_desktop_files`, `_demand_files`, `_projections`, `_commands`,
`_browser`, `_intake`, plus Companion-Sichtbarkeit `pub(super)`), zuletzt
`4221b7304`.

**69.–70.** Hash-Beweis: Funktionsrümpfe extrahieren, **Whitespace
normalisieren** (`\s+` → einzelnes Leerzeichen), SHA-256 vergleichen. Nur der
**Rumpf**, nicht Signatur/Sichtbarkeit/Imports — die dürfen sich ändern. Das
Skript existiert **nicht als Datei**, nur als Inline-Python im Chat. **Bitte als
Werkzeug anlegen.**

**71.** **ACHTUNG — der Größenwächter ist JETZT ROT:** `service.rs` misst
**26.244** Zeilen bei Budget **26.225** (19 zu viel), verursacht von Commit
`bd3bbb895` („a research run keeps its evidence across retries") — **nicht von
mir**. `store.rs` = 27.342, Budget 27.342 (exakt). **`app.js` hat gar kein
Budget** (12.388 physische Zeilen, ungeschützt).

**72.–73.** **Ja** — jeder reine Move ein eigener Commit. Import-, Sichtbarkeits-
und `mod`-Anpassungen im selben Commit sind erlaubt und nötig, solange der
Rumpf unverändert bleibt.

**74.** **Bekannte Blocker aus den bisherigen Schnitten:** Test-Statics
(`TEST_LOCK`) wurden einmal übersehen und mussten nachgereicht (`4c22bdbf`);
private Typen brauchen `pub(super)`; ein Symbol, das nur im dirty Baum existierte,
wurde dreimal versehentlich mitcommittet. **Symbolaudit als eigener Schritt vor
dem Commit** — inklusive Statics und Konstanten, nicht nur Funktionen.

**75.** `service.rs` wird **aktiv von anderen Sitzungen verändert** (zuletzt
`bd3bbb895`). Vor jedem Schnitt frisch prüfen, wer dort gerade arbeitet.

**76.** **OWNER.** `app.js` steht in OA-3, ist aber nicht in der Abnahme —
und es ist eine Browser-Datei, also näher an der Parallelsitzung.

**77.–78.** Keine Zielgrößen definiert außer „Budgets nur senken". Doku zu
Modulgrenzen: ja, im selben Zug (`src/core/business_os/AGENTS.md`).

## OA-5: Bridge-Handshake

**79.–81.** Beleg ist die Messung der **Parallelsitzung**: seriell Startspanne
7.977 ms / letzte Verbindung 15.073 ms; mit 4 Spuren 8.688 ms / 12.123 ms.
Je Bridge ~530 ms seriell, unter Parallelität ~2.300 ms — die Bridges teilen
sich Peer und Signalisierung. **Rohdaten liegen bei ihnen**
(`docs/dev/beweise/m2-browser-boot-pacing.txt`). **Wie viele Rundreisen ein
Handshake genau kostet und welche Schritte redundant sind: ungemessen.**

**82.–84.** Kein abgestimmtes Zielprotokoll, kein Entwurf. Was die
Parallelsitzung dort bereits umgesetzt hat, weiß ich nicht.

**85.** **OWNER** — nach Absprache liegt es bei der Parallelsitzung.

## Kundeninstanz und Betrieb

**86.** VM `ctox-e5ed9648`, erreichbar über ein Skript, das den privaten
Schlüssel aus der Fleet-Datenbank entschlüsselt, benutzt und **sofort wieder
löscht**. Meine Kopie: `~/ctox-mess/vm.mjs` (lesend),
`/Volumes/tmp/ctox-pipeline/vmput2.mjs` (schreibend, langes Zeitfenster —
das Original hat 300 s und reicht für 349 MB nicht).

**87.** **OWNER.** Was ich getan habe: gelesen, gemessen, Binary ausgeliefert,
Dienst neu gestartet — nach ausdrücklicher Aufforderung. Ich habe **nie**
Anmeldedaten eingegeben und **nie** Zugänge auf der Produktionsmaschine erzeugt.

**88.** Über `/proc/<pid>/exe` und `sha256sum`, **nicht** über `command -v ctox`
— dieser Pfad zeigt eine andere Datei, die der Dienst gar nicht fährt. Genau
dieser Fehler hat schon einmal zu einer falschen Diagnose geführt.

**89.** Bewährter Ablauf: Vorher-Messung → altes Binary sichern → komprimiert
hochladen → entpacken, `chmod +x` → **Prüfsumme an drei Stellen vergleichen**
(Baumaschine, lokale Kopie, Zielort) → `systemctl --user stop` → tauschen →
`start` → `is-active` prüfen → bei Fehlschlag automatisch Sicherung
zurückspielen → Nachher-Messung + Kontrolle nach 15 Minuten.

**90.** **OWNER** — nicht definiert.

**91.** **Tailscale-Falle:** Der Dienst war auf dem Mac gestoppt; dadurch war
die Linux-Baumaschine unerreichbar. Das sieht aus wie ein Berechtigungs- oder
Ausfallproblem und ist keines. **Vor jeder SSH-Fehlerdeutung `tailscale status`
prüfen.**

**92.** Logs: `journalctl --user -u ctox.service`. Laufzeit-DBs:
`~/.local/state/ctox/ctox.sqlite3` (Kanal/Kern) und
`business-os-rxdb.sqlite3` (RxDB-Store, 2,29 GB). Peer-Status:
`business-os-rxdb-peer.status.json`. Ausgeliefertes Verzeichnis:
`~/.local/lib/ctox/current` — die `business-os-stage.*` unter `state` werden
**nicht** bedient.

**93.** Niemals: Kundendateninhalte, Zugangsdaten, Fähigkeitstoken, private
Schlüssel. Aggregierte Zählungen wie in diesem Dokument sind unbedenklich.

**94.** **OWNER** — Stufe 3 („auf der Instanz wirksam") konnte ich für AP3 nicht
selbst schließen, weil sie eine angemeldete Sitzung verlangt.

## OA-7 bis OA-9

**95.–97.** **OWNER.** Kein Wartungsfenster vereinbart, **keine aktuelle
Sicherung der 2,29-GB-Datei erstellt**. Freier Platz auf der VM: 39 GB bei 77 GB
— reicht für `VACUUM` (braucht etwa die Dateigröße zusätzlich). Akzeptable
Ausfallzeit: OWNER.

**98.–99.** Feststellung „kein Browser verbunden" stammt aus
`replicationSignals`: `poolCreated: true`, `signalingJoinAccepted: true`,
**`dataChannelOpen: false`**, bei frischer Statusdatei (4 s alt). Der Wert ist
also ehrlich. **Ob ein Benutzer sich verbinden *kann*, ist ungeprüft** — das ist
der eigentliche Inhalt von OA-8.

**100.** AP3 löst aus, wenn ein Command nach dem Push 8 Sekunden ohne
Fortschritt bleibt. Zu beobachten: Status `ok=false` mit Code
`data_plane_no_progress` und Collection-Name, **genau ein** Reparaturversuch,
Rückkehr auf `ok=true` bei wiederkehrendem Fortschritt. Braucht eine verbundene
Browser-Sitzung unter Last.

---

## Was den Start wirklich blockiert

1. **Eigentümerschaft der 37 dirty Browser-Dateien** — nicht von mir, Herkunft
   je Datei ungeprüft.
2. **Status der Parallelsitzung** — heute 07:12 zuletzt aktiv.
3. **Zugriff auf die Kundeninstanz** und die erlaubten Aktionen.

**Sofort untersuchbar ohne diese Klärungen:** der OA-6-Codepfad (Frage 21 ist
der Schlüssel) und die Refactoring-Nähte in `store.rs`/`service.rs` — dort
zuerst den **roten Größenwächter** (Frage 71) reparieren.
