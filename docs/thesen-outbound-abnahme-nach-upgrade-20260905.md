# Abnahme der Outbound-App nach dem Queue/Harness-Upgrade (05.09.2026)

Maßstab (Eigentümer, 05.09.): Die Fixes müssen darauf einzahlen, dass die
Outbound-App anschließend korrekt funktioniert. Grüne Tests sind Voraussetzung,
nicht Abnahme. Abgenommen wird auf `thesen.ctox.dev` im Browser und im
Routing-State (`communication_routing_state`), nie in der Projektion allein.

## Was jeder Befund in der App bewirken muss

| Befund | Sichtbares Verhalten in der App nach dem Upgrade | Messung |
|---|---|---|
| 1 Lease-Sweep | Stirbt ein Worker mitten in einer Recherche, fällt der Lead binnen ≤ 20 min von „Läuft" auf „Wartet" zurück und wird erneut gestartet. Kein Lead bleibt für immer auf „Läuft". | Routing-State: `leased` → `pending` nach `lease_expires_at`; Lead-Zeile in der App wechselt. Probe: laufende Recherche, Worker-Prozess beenden, 20 min warten. |
| 2 Reparatur-Entdopplung | Bei Quellenfehlern entsteht je Ziel höchstens EINE offene Reparatur; nach drei Fehlschlägen ist Schluss (`failed` mit Grund). Die Recherchen stehen nicht mehr hinter Dutzenden Reparaturen. | `ctox queue list --status pending` nach Titel gruppiert: kein Titel mehrfach. Über 24 h beobachten. |
| 3 Kapazität | Mit `ctox queue capacity --workers 4` laufen vier Recherchen gleichzeitig: „Auswahl recherchieren (8)" → vier Leads auf „Läuft", nicht einer. Eine 19-Firmen-Kampagne dauert Stunden, nicht einen Tag. | Routing-State: `count(route_status='leased')` ≥ 4 bei ≥ 8 wartenden. App-Zähler „Laufend". |
| 4 Projektion | Stornieren (CLI oder App) nimmt die Aufgabe sofort aus „Läuft"; die sieben Altlasten vom 28.–31.08. verschwinden; CTOX-Modul zeigt nur, was es gibt. | Projektion `ctox_queue_tasks` == Routing-State je `id`; Anzahl `status=running` in Projektion == `leased` im Routing-State. |
| 5 Writeback | Jede abgeschlossene Recherche schreibt zurück (Lead → „Prüfung nötig" mit Feldern) ODER der Task endet `failed` mit Grund. Nie mehr „erfolgreich" mit 0 Feldern. | Kein Lead mit Task `completed`/`handled` und `field_status`-Zählung 0. Chat zeigt den Grund bei Fehlschlag. |

## Was nicht schlechter werden darf (Regression)

- Klick „Auswahl nachrecherchieren" → Hinweis, Chatfenster, Lead auf „Wartet" (B1, 1.0.94–1.0.99)
- Personenwechsel im Lead-Detail zeigt verschiedene Inhalte
- Dialoge als Overlay in der App, nicht im Seitenfluss (1.0.90)
- Chateingabe bleibt beim Tippen stehen (c378aaf83)
- Kampagnenliste: eine Kampagne „Chemie", 19 Leads, 19 mit Ergebnis
- Schreibtisch lädt in < 90 s; keine Neustart-Schleife der Kollektionen

## Ablauf

1. Befund 1–4 auf `origin/main` (Befund 5 darf nachziehen) → ein Upgrade `ctox upgrade --dev`.
2. Nach dem Umschalten: `ctox queue capacity --workers 4` — **Achtung: der Code-Standard ist bereits 4** (`unwrap_or(4)` in `service_queue_capacity.rs`), nicht 1 wie zunächst angenommen. Parallelität greift also sofort nach dem Umschalten; das explizite Setzen macht sie zur dokumentierten Entscheidung. Erste Beobachtung direkt danach: welche vier Aufgaben werden geleast (Recherche, Auth-Assist, Reparatur?).
3. Regressionsliste im Browser durchklicken.
4. Befund 2/3/4 sofort messen (Routing-State + App); Befund 1 per Probe; Befund 5 an der nächsten realen Recherche.
5. Ergebnis mit Zahlen in dieses Dokument, dann Push auf `main`.

## Baseline VOR dem Upgrade (05.09.2026, 06:38 UTC, Release branch-main-20260904T161717Z)

| Messung | Wert vorher |
|---|---|
| Befund 3: Routing-State | 0 `leased`, 30 `pending` — Kapazitätsbefehl existiert nicht |
| Befund 4: Projektion `running` vs Routing-State `leased` | **7 vs 0** — sieben Phantome (alle im Routing-State `cancelled` seit 28.–31.08.) |
| Befund 2: doppelte offene Titel | **24 Dubletten** (19× evi-gv-at, 3× maps-google-com, 2× handelsregister-de) — vierte Welle; von Hand auf je eine reduziert (21 storniert, 9 offen) |
| Befund 5: Leads mit Ergebnis | 19 / 19 (`research_status`), keiner mit 0 Feldern |
| Befund 1: aktive Leases | keine zum Messzeitpunkt |

Erwartung nach dem Upgrade: Befund 4 → 0 Phantome (oder Reconciler räumt sie);
Befund 2 → über 24 h keine Dubletten mehr; Befund 3 → nach `capacity --workers 4`
mehrere `leased` bei Rückstau; Befund 5 → an der nächsten Recherche; Befund 1 → per Probe.

## Ergebnis NACH dem Upgrade (05.09.2026, Release branch-main-20260905T072559Z, umgeschaltet 07:53 UTC)

Upgrade 07:26:06–07:53 UTC (27,5 min), Start automatisch nach Ende der laufenden
BOOMEX-Recherche (die mit 23 Feldern abschloss, vorher 15). Beim Start war nur eine
Scraper-Reparatur geleast, keine Recherche.

| Befund | Messung nach dem Upgrade | Stand |
|---|---|---|
| 3 Kapazität | `ctox queue capacity` → `max_workers 4, workers_per_thread 1, scope independent business_os.chat.task sessions, storage SQLite runtime store` (explizit gesetzt). Eine Minute nach dem Umschalten **2 `leased`** gleichzeitig (07:53:19 / 07:53:20, beide Worker `21b39433…`, 15-min-Ablauf), 11 `pending`. Vorher nie mehr als 1. | **belegt** (2 von max. 4; mehr braucht mehr unabhängige Aufgaben im Rückstau) |
| 4 Projektion (neu) | Aufgabe `ebe4782bea95…` (Auth-Assist xing): Projektion `queued/pending/rev 1-civfwsvgrf` → `ctox queue cancel` im frischen CLI-Prozess → Projektion **`cancelled/cancelled`, lease_owner null, rev `2-1887bf89…`** um 07:56:57; Routing-State `cancelled`. | **belegt** |
| 4 Projektion (Altlast) | 7 Phantome unverändert (Reconciler `9b3f44e09` + Merge-Fix `5a16d0061` sind NICHT in diesem Release; sie belegen keine Kapazität). | offen bis zum zweiten Upgrade |
| 2 Reparatur-Dubletten | 4× `evi-gv-at` offen, alle **ohne** `scrape_repair`-Metadatum → vom alten Release erzeugt (nach meiner Bereinigung um 06:38, vor dem Umschalten). Drei per CLI storniert. Neue Einreihungen tragen das Metadatum; Dubletten-Freiheit über 24 h zu beobachten. | nicht widerlegt, 24-h-Messung offen |
| 1 Lease-Sweep | Zwei live Leases mit `lease_expires_at` (+15 min) und `lease_worker_id`. Probe „Worker stirbt" noch nicht gefahren. | Probe offen |
| 5 Writeback | 19/19 Leads mit Ergebnis, keiner mit 0 Feldern. App 1.0.100 liefert `payload.writeback_contract` mit `mechanism`/`command_type`. End-to-end (persistierter Vertrag → `business_os.execute_writeback` → Felder) braucht eine neue Recherche aus dem Browser. | **blockiert: Browser-Sitzung nach dem Upgrade abgemeldet**, Anmeldung durch den Eigentümer nötig |
| Regression Browser | — | **blockiert** (Anmeldung) |

Nächste Schritte: (a) Eigentümer meldet sich im Browser an → Regressionsliste + Befund-5-Lauf
(DrinkStar, 7 Felder); (b) zweites Upgrade mit `5a16d0061` (Reconciler + Merge-Fix) → Phantome 7 → 0;
(c) 24-h-Beobachtung Dubletten; (d) Probe Befund 1 bei Gelegenheit.

## Nachmessung 06.09.2026, 17:00 UTC (33 h nach dem Umschalten)

| Messung | Wert | Bewertung |
|---|---|---|
| Queue | 0 `leased`, 0 `pending` | leer, kein Rückstau |
| Befund 2 Dubletten | **0 doppelte offene Titel über 33 h** (vorher vier Wellen in 24 h) | belegt über den Beobachtungszeitraum |
| Befund 3 Kapazität | `max_workers 4`, in der Umschaltminute 2 parallel geleast | belegt (Maximum 4 nur mit ≥4 unabhängigen Aufgaben messbar) |
| Befund 4 neu | Cancel projiziert sofort (05.09. 07:56:57) | belegt |
| Befund 4 Altlast | 7 Phantome unverändert | offen — Reconciler `9b3f44e09`/`5a16d0061` nicht ausgeliefert |
| Befund 5 | Leads 19/19, 249 Felder gesamt, Minimum 7, keiner mit 0 Feldern. **Der neue Writeback-Guard hat noch nie gefeuert** (0 Treffer `Business command writeback failed`) — kein Task mit dem 1.0.100-Vertrag ist bisher gelaufen; der jüngste Nachrecherche-Befehl (05.09. 08:24) trägt noch den alten Vertrag (Daemon-Recovery kopiert das ursprüngliche Payload). | End-to-end offen |
| Seit Umschaltung `failed` | 14 Routing-Übergänge auf `failed`, darunter Beiersdorf („research contract is materially unmet: 2 von 8 Personen-Kategorien … Finite review budget exhausted 5/5") und eine Recovery-Aufgabe („10-path requirement structurally unsatisfiable"). **Das ist der Review-Mechanismus, der Endlosschleifen terminal stoppt — nicht der neue Guard.** Die Writebacks landeten trotzdem: Beiersdorf 12 → 18 Felder, BOOMEX 23. | erwartetes Verhalten, aber im CTOX-Modul als Fehler sichtbar |
| Testdaten | Writeback-Versuche mit Ids `…-writeback-test`, `…-test-v6-…` für BOOMEX vom Worker selbst; ein Versuch gegen `lead_test_001` wurde **abgelehnt**, kein solcher Lead existiert (54 Datensätze, 19 aktiv) | keine Verunreinigung |

**Blockiert:** Browser-Regression und Befund-5-End-to-end — die Browser-Sitzung ist seit dem
Upgrade abgemeldet, kein Chrome verbunden; Anmeldung nur durch den Eigentümer.

**Upgrade 2 zurückgestellt:** `origin/main` liegt 93 Commits / 38.865 Zeilen vor dem
ausgelieferten Stand (Sync-Engine 51 Dateien, RxDB 34, Shell 91; darunter „production peer
crash and data-preserving recovery", „rejected cutover and verified production rollback").
Ein `ctox upgrade --dev` würde das komplett und ungeprüft auf die Kundeninstanz bringen — nur
für sieben kosmetische Phantome. Entscheidung des Eigentümers.

## Upgrade 3 — 07.09.2026, 23:04 UTC (Release branch-main-20260906T223502Z: Threads-Fix, Skill §7, Reconciler; sync.js-Buster vor dem Umschalten im Release-Verzeichnis gepatcht, byteidentisch zu `d714d887a`)

| Messung | Wert | Bewertung |
|---|---|---|
| Shell-Boot | `ready`, Nutzer angemeldet, `sync.js` lädt `command-bus.js`/`sync-contract.js` mit `?v=20260906-office-page-exit` (200) | belegt |
| Befund 4 Altlast | `projektion_running=0`, `routing_leased=0` — **7 Phantome → 0** (Projektionstabelle jetzt `ctox_queue_tasks__v3`) | belegt |
| Kapazität | `max_workers 4` | unverändert |
| Leads | 19/19 mit Ergebnis | unverändert |
| Recherchestart (Threads-Fenster maximiert, keine Kollektionen ausgesetzt) | Klick 23:39:33 → Befehl angelegt 23:39:42 → Server `accepted` 23:40:50 → Worker gestartet 23:39:49 (Lease 7 s nach Anlage) | **Latenz 15–20 min → 77 s**, davon 47 s Sellify-Vorabgleich |
| Writeback | Worker recherchierte 32 Felder (18 verifiziert), meldete Erfolg, Task 23:47:20 `failed`: „no successful outbound.lead.research_writeback receipt" — 0× `business_os.execute_writeback`, 3× `propose_action` (alle `success:false`) | **Defekt 2, siehe unten** |

### Defekt 1 (07.09., 23:04–23:36 UTC): Instanz nach dem Upgrade 32 Minuten für ALLE schreibgeschützt

Der erste Recherchestart nach dem Umschalten scheiterte still: `CTOX_MAINTENANCE_READ_ONLY`
(„CTOX wird aktualisiert – Apps bleiben schreibgeschützt"), unbehandelte Promise-Ablehnung im
Lead-Schreibpfad, Status fiel von „Wird gestartet" auf „Prüfung nötig" zurück. Ursache im
Wartungsprotokoll (`src/core/install/mod.rs`, `src/apps/business-os/app.js`):

1. Nach dem Dienst-Neustart steht der Zustand in `ctox-maintenance.sqlite3` auf
   `waiting_collections` und wird **nur** durch den Business-Command
   `ctox.maintenance.client_ready` eines Browsers beendet, der alle Pflichtkollektionen seiner
   offenen Module `complete` sieht (`tryAcknowledgeMaintenanceReadiness`).
2. Der einzige offene Tab war verdeckt; `maintenancePollDelay()` liefert bei
   `visibilityState === 'hidden'` 0 → kein Poll, keine Bestätigung — obwohl alle 17
   Pflichtkollektionen längst `complete` waren.
3. Serverseitig läuft die Wartezeit unbegrenzt: bei lebendem Peer-Heartbeat wird der Lease
   nur verlängert („a live peer heartbeat owns the wait"). `lease_expires_at` 23:09:11 war
   um 23:36 noch `active`.

Sofortmaßnahme: den Retry-Pfad der Shell (`[data-maintenance-retry]`) ausgelöst — derselbe
Code, den ein sichtbarer Tab bei jedem Poll ausführt; Bestätigung 23:36:15 UTC über den
WebRTC-Befehlskanal, Zustand `completed`. Fix auf main: (a) Server gibt `waiting_collections`
nach 10 min ohne Browser-Bestätigung selbst frei (`MAINTENANCE_CLIENT_ACK_GRACE_MS`,
neues Feld `waiting_collections_since_ms`, Tests), (b) verdeckter Tab pollt alle 30 s
weiter, solange ein Upgrade läuft (`CTOX_MAINTENANCE_HIDDEN_POLL_MS`).

### Defekt 2 (07.09., 23:47 UTC): Der Recherche-Worker hat das Writeback-Werkzeug nie gesehen

`business_os.execute_writeback` steht im MCP-Katalog der Instanz (75 Werkzeuge), aber die
Worker-Sitzung bekommt nur die Allowlist `BUSINESS_OS_MCP_SESSION_TOOLS` aus
`src/core/execution/agent/direct_session.rs` (41 Werkzeuge) — und die enthielt das Werkzeug
nicht. Befund 5 (`e9a346e38`) hat Katalog und Guard ergänzt, die Sitzungs-Allowlist nicht.
Folge: **jede** Recherche seit dem Upgrade vom 05.09. endet zwingend im Guard
(DrinkStar 20:54, Cereda 23:47), unabhängig von Skill und Auftragstext. Die Umstellung
von Skill §7 (`3cf70ee05`) und App 1.0.102 war richtig, aber nicht hinreichend.
Fix: Werkzeug in die Allowlist, Test `business_os_mcp_thread_config_is_local_scoped_and_tool_bounded`
verlangt es ausdrücklich. Auslieferung nur per Binary → Upgrade 4.

## Upgrade 4 — 07.09.2026, 05:29 UTC (Release branch-main-20260907T045949Z = main `c69119697`: Wartungsfreigabe, `execute_writeback` in der Worker-Allowlist, dazu 16 fremde main-Commits: Office-Fixes, Tombstone-/Replikations-Fixes der Sync-Engine)

| Messung | Wert | Bewertung |
|---|---|---|
| Umschaltung | 05:29:01 Stop → 05:29:04 Start → 05:29:14 „replication up for 205 collections" | wie Upgrade 3 |
| Wartungsfreigabe | `waiting_collections` → `completed` 05:29:37 durch Browser-Ack (33 s nach Neustart; Tab sichtbar) | Grace-Pfad nicht gebraucht, aber vorhanden |
| Worker-Werkzeuge | `tools_count=42` (vorher 41) in den Responses-Requests → `business_os.execute_writeback` ist in der Sitzung | belegt |
| Shell | `app.js`/`sync.js` mit `?v=20260906-office-page-exit`, Boot `ready`, angemeldet | belegt |
| App | 1.0.103 live (Übersicht trennt „recherchiert" von „belegt (inkl. Import)") | belegt |
| Datenbestand | Alle Leads/Kampagnen per App-Dialog gelöscht (0 lebend, 54 Löschmarken), „Chemie Test 2026" mit 19 Firmen neu importiert (Import 05:40–05:43, IDs der Löschmarken wiederbelebt, keine Dubletten) | belegt |
| Kampagnenstart | „Alle recherchieren (19)" erzwingt Variante Nachrecherche → 18 Abbrüche „nur Neue Recherche möglich"; danach „Auswahl neu recherchieren (19)" 05:51:11 → Anlage sequenziell ~1 Lead/2,5 min | **App-Defekt 3**, Fix 1.0.104 vorbereitet (Variante je Lead automatisch) |

### Defekt 4 (07.09., 05:50 UTC): Harness bricht Recherche nach zwei Minuten ab — „task execution steps must be a completed prefix"

Der Worker für Carbosulf (Thread 01a07a69…) starb nach drei LLM-Aufrufen mit
`durable task progress failed: task execution steps must be a completed prefix, one active step, then pending steps`
(`src/core/context/lcm/mod.rs`, `validate_task_execution_steps`, seit `ab718a53d` 30.08.). Das Modell
meldet Planfortschritt legitim außer der Reihe (späterer Schritt fertig, zwei aktiv, keiner aktiv);
die Prüfung machte daraus einen Abbruch des ganzen Auftrags. Fix: Statusfolge wird normalisiert
(zweiter aktiver Schritt → pending, ohne aktiven Schritt → erster offener wird aktiv), fünf
Unit-Tests; Auslieferung per Upgrade 5.

### Defekt 5 (07.09., seit Upgrade 3): Thread `cockpit-projections` verbraucht dauerhaft einen Kern

`top -H`: 78–100 % CPU auf dem Pump aus `harness_cockpit_projections.rs:153` bei leerer Queue,
Dienst 194 % CPU, RSS 2,7 GB; keine Schreiblast (WAL-Größen konstant, 5 `business_records`-Writes in
5 min). Erstmals mit Upgrade 3 ausgeliefert (Pump seit `46e3b6d3a`, 05.09. 17:43). An den
Crew-Cockpit-Codex-Thread 01a07107 gemeldet (Queue-Nachricht 01a07a6a…). Nicht belegt, aber
verdächtig: die seit Upgrade 3 gehäuften `ctox_webrtc_incoming_transfer_stalled` /
`Timed out waiting for WebRTC response masterChangesSince` im Browser (jede Reload-Sitzung nach
1–3 min, alle Kollektionen der App), die Löschung (13 von 19 Leads beim ersten Versuch) und Import
(erst nach Wipe des lokalen Browser-Speichers) nur mit Verzögerung durchbrachten und im Sammellauf
einzelne Aufträge lokal als `failed` enden lassen (Beiersdorf 05:53: Befehl nie beim Server).

### Owner-Browser: 36 statt 19 Leads

Der Browser des Eigentümers zeigte 17 seit 04.09. gelöschte Leads („Offen") neben den 19 lebenden und
„synchronisiert 0/5": die Löschmarken kamen dort nie an. Der Neuaufbau (Löschen + Neuimport) und die
Tombstone-Fixes aus main (`20199fe28`, `7a54790c0`, `fcce19763`, mit Upgrade 4) adressieren das;
Nachweis steht aus, bis der Eigentümer neu lädt.

### Sammellauf 07.09., 05:51–06:45 UTC (App 1.0.103, Release branch-main-20260907T045949Z)

| Messung | Wert |
|---|---|
| Aufträge angelegt | 13 von 19 in 54 min — die App startet sequenziell, jeder Start hängt am Sellify-Vorabgleich (Bedarfsabfragen stallen, 120-s-Deckel) |
| Worker parallel | 4 (Kapazität greift) |
| **End-to-End-Nachweis** | Dr. Kurt Richter: `execute_writeback` 06:11:51 + 06:17:22 `completed`, Lead `needs_review`, **13 Felder**; Cereda 06:38:39 `completed`, **4 Felder** |
| Writeback-Ablehnungen | 8, davon 6 „invalid payload" (Destilla 4×: verschachtelter `firma_land`-Schlüssel, `deny_unknown_fields`), 1 „non-verified field must not carry a populated value" — der Worker korrigiert nur, wenn die Ablehnung den Grund nennt → Fix `f3a2aa7fd` |
| Terminal gescheitert durch Plan-Prüfungen | Carbosulf 05:50 („completed prefix"), AKEMI 06:16 („plan is incomplete 1/3"), CHEMOFAST 06:41 („exactly one in-progress step") → Fixes `92a84679b` (Normalisierung) + `80561cbc9` (Wiederholung statt Terminal) |
| Technische Wiederholungen | 4× `stream disconnected before completion` (llm.ctox.dev), 1× `thread/start failed` (Aeroxon) — Backoff 60 s, `technical:worker-runtime-api-failure` |
| Lokale Fehlstarts | Beiersdorf, BEWI RAW, BOOMEX u. a. zeitweise `failed` ohne Server-Auftrag — Submit im Browser lief in den Sync-Stall |

Upgrade 5 gestartet 06:46 UTC (main `f3a2aa7fd`: Plan-Normalisierung, Wiederholungsklasse, Writeback-Fehlerdetail).
Danach: App 1.0.104 ausliefern, offene Leads in kleinen Gruppen nachstarten, Feldtabelle messen.
