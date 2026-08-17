# Gesamtplan: Refactoring und Sync-Performance

Stand: 17.08.2026

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
| OA-6 endlicher Command-Intake | ja | inklusive isolierter Stunde ja | nein | Kundenmessung offen |
| Idle-CPU des Dienstes | ja | 4,02 % über eine isolierte Stunde | nein | Kundenmessung offen |
| OA-2 synthetische 300k-Baseline | teilweise | 304.515-Dokumente-Einzel-Smoke strukturell grün | entfällt | 30×30-Matrix und Latenzziel offen |
| OA-1 bounded Demand-Sync | ja | 4 RPCs, 650 materialisierte Dokumente, kein Vollpull | nein | Latenz- und Löschungsabnahme offen |
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

Idle-CPU-Nachtrag vom 16.08.2026:

- Die zunächst beobachteten 104,4 % gehörten zum laufenden synthetischen
  Sellify-Benchmark und waren keine Idle-Messung.
- Ein separater Sample des installierten Dienstes belegte dennoch einen
  periodischen teuren Pfad: Der Channel-Router öffnete alle 30 Sekunden die
  1,1-GB-Kerndatenbank für einen redundanten Queue-Safety-Poll und ließ SQLite
  dabei das Schema neu parsen.
- Der günstige Acht-Sekunden-Quellstempel bleibt unverändert reaktiv. Nur der
  redundante Vollzugriff folgt nun dem einstündigen Safety-Fenster.
- Drei gezielte Routertests sind grün; darunter ein neuer Nachweis, dass nach
  dem früheren 30-Sekunden-Intervall keine SQLite-Verbindung geöffnet wird.
- Quellcode-Fix und Diagnose sind lokal belegt; Ausrollen, Neustart und
  einstündige Nachhermessung bleiben offen.

Rollout-Zwischenstand vom 16.08.2026:

- Der exakte Stand `eccc24334` wurde aus einem isolierten `git archive` auf
  `/Volumes/tmp` gebaut. Das Release-Binary hat SHA-256
  `bedd8e3aafa4af50fa9baa836642913a3b84a869612f0bfa9f2043ce35707ee8`;
  das bisher installierte Binary bleibt mit SHA-256
  `7fb2350ba879eb0d100401e6f6638ffe393d5d3c2c1232bd885f24a5f2e593b0`
  als Rollback-Referenz erhalten.
- Vor dem Umschalten waren drei Statusabfragen konsistent: Dienst laufend,
  keine aktive Arbeit, keine wartenden Tasks und keine Worker-Aktivität.
- Die konsistente Sicherung
  `backups/update-20260816T170608Z` wurde angelegt. Die Retention hat erst
  danach die vorherige Update-Sicherung vom 07.08.2026 entfernt.
- Der verwaltete Release-Slot `refactor-sync-eccc24334` ist aktiv. Update,
  Symlink-Switch und Neustart endeten erfolgreich; `update status` meldet
  `phase=completed` ohne Fehler und den vorherigen Release
  `cliproxy-provider-contract-r6-20260807T1323Z` als Rollback-Slot.
- Build-, Release- und `current`-Binary sind bytegleich und haben SHA-256
  `29ff1da559cd13465cd6122cbfe7bbfecb439e28c8f54aca33ae44ac6edbb6c1`.
  Der Dienst läuft unter PID `31541`, drei Statusabfragen meldeten konsistent
  `busy=false`, keine wartenden Tasks und keine aktiven Worker. Ohne Browser
  meldet der lebende native Peer erwartungsgemäß `replicationUp=false`.
- Der erste 30-Sekunden-Probe bestätigt, dass der frühere Dauerlauf beseitigt
  ist (CPU-p50 0,7 %), verfehlt das Dauertor aber noch: Ein periodischer Peak
  von 76,8 % ergibt 7,66 % Prozess-CPU über das kurze Messfenster. Ein
  Prozess-Sample ordnet ihn `external_sql_sync::run_due_sources` zu: Der
  Idle-Poll lädt lokale Modul-Manifeste und hasht rekursiv deren Assets, obwohl
  keine externe SQL-Quelle fällig ist.
- Commit `0a4ab13b2` trennt für den External-SQL-Poll die schlanke
  Deklarationsauflösung von der Katalog-Anreicherung. Der Katalog berechnet
  seine Asset-Revisionen weiterhin unverändert; nur der Idle-Poll überspringt
  die für ihn irrelevante rekursive Dateihash-Bildung. Der Regressionstest ist
  1/1 grün, und `store.rs` bleibt exakt bei seinem unveränderten Budget von
  21.970 Produktionszeilen.
- Ein historischer offener Intake-Failure-Datensatz wurde gegen Aggregate,
  Transition, beide zugestellten Outbox-Projektionen und das kanonische
  RxDB-Dokument geprüft. Commit `07c2ef0ca` ergänzt eine konservative
  Reconciliation: Nur nicht erschöpfte transiente SQLite-Sperrfehler mit
  vollständig zugestellter kanonischer Projektion werden aufgelöst;
  Idempotenzkonflikte ausdrücklich nicht. Neuer Test 1/1, angrenzende
  Audit-Regressionen 2/2 und Cancel-Migration 1/1 sind grün.
- Der zweite Code-Stand `c53a86588` wurde aus einem isolierten `git archive`
  mit SHA-256
  `0218071d33bc26c4f79f4d358b24e683cfa8303a66d042bd079c8ee2e470fd9f`
  gebaut und als verwalteter Release `idle-cpu-c53a86588` ausgerollt. Die
  Sicherung `backups/update-20260816T201808Z` ist vorhanden; der vorherige
  Release `refactor-sync-eccc24334` bleibt als Rollback-Slot erhalten.
- Build-Ausgabe, Release-Binary und `current`-Binary sind bytegleich und haben
  SHA-256
  `bcc58ec5553bfb83edff93c1bf4c7e2d1cb02995eadb66597d798902ca935110`.
  Der Dienst läuft nach dem kontrollierten Neustart unter PID `5575`, meldet
  `busy=false`, keine wartenden Tasks und keine aktiven Worker. Der native
  Peer ist lebend und meldet ohne Browser erwartungsgemäß
  `replicationUp=false`.
- Direkt nach diesem Neustart zeigte sich kein weiterer Idle-Loop, aber ein
  endlicher Startup-Engpass: Der Knowledge-Katalog öffnete für jedes Skill-,
  Skillbook-, Runbook- und Resource-Dokument die 1,1-GB-CTOX-Datenbank erneut
  und ließ SQLite das große Schema wiederholt parsen. Nach Abschluss dieses
  Durchlaufs schlief der Prozess bei 0,5 %; die Initialisierung dauerte jedoch
  rund fünf Minuten und verletzt damit das Boot-Ziel.
- Commit `43889102a` verwendet für den gesamten Knowledge-Katalog genau eine
  SQLite-Verbindung; Einzelabrufe bleiben unverändert. Der neue
  Verbindungsanzahl-Regressionstest und `cargo fmt --all -- --check` sind
  grün.
- Der committed Stand `60ff9c957` wurde aus dem isolierten Snapshot mit
  SHA-256
  `170a6a80881ed1a9e5b83aa85501820b532bc549d25e8d9e8d3d686b2215ab47`
  als `idle-cpu-60ff9c957` ausgerollt. Die neue Sicherung liegt unter
  `backups/update-20260816T204848Z`; `idle-cpu-c53a86588` ist der aktuelle
  Rollback-Slot. Die Retention entfernte danach den älteren Release
  `refactor-sync-eccc24334` und die vorherige Update-Sicherung.
- Build-Ausgabe, Release-Binary und `current`-Binary sind bytegleich und haben
  SHA-256
  `cf86ea4b96f66c4751cbfc89415e013ab65863fa9ed48f32515ec1a77d05b733`.
  Der neue Dienst läuft unter der von `launchd` überwachten PID `45237`, ist
  nicht beschäftigt und hat weder wartende Tasks noch aktive Worker. Der
  native Peer hat einen frischen Heartbeat, keine Health-Fehler und meldet
  ohne Browser korrekt `replicationUp=false`.
- Die erste formale 30-Sekunden-Messung dieses finalen Releases lief nach
  Abschluss der Startphase gegen PID `45237`. OA-6 ist dabei betrieblich
  sauber: Kandidatenmenge und offene Intake-Fehler blieben null, der
  vollständige Command-Revisionshash blieb unverändert und die RxDB-Idle-Ticks
  stiegen. Das CPU-Tor ist knapp rot: 5,68 % Prozess-CPU über die Messdauer bei
  p50 0,9 %, p95/Maximum 45,7 %. Damit ist kein Dauer-Spin mehr vorhanden,
  aber ein periodischer Rest-Peak muss vor der Ein-Stunden-Abnahme noch
  lokalisiert und behoben werden. Die Rohmessung liegt vorerst unter
  `/tmp/ctox-idle-probes/idle-30s-60ff9c957.json` und wird erst zusammen mit
  dem bestandenen finalen Nachweis versioniert.
- Ein anschließender 30-Sekunden-Stack-Sample hat den periodischen Rest-Peak
  vollständig aufgelöst: Ein unverändertes `detected`-Harness-Finding erzwang
  alle fünf Minuten einen vollständigen Audit-/Conformance-Replay von rund
  12,8 Sekunden. Parallel öffnete der AppSec-Poller alle zehn Sekunden die
  Kerndatenbank neu; allein das erneute SQLite-Schema-Parsing beanspruchte in
  der Stichprobe rund 0,7 bis 0,9 Sekunden.
- Der Projection-Clock der Queue wird nun über eine thread-lokale
  Read-only-Verbindung gelesen. Der Cache ist an kanonischen Pfad,
  Dateisystem-Gerät und Inode gebunden, beobachtet WAL-Commits weiter und wird
  bei Datenbankaustausch oder Abfragefehler neu geöffnet. Unveränderte
  Harness-Findings werden nach dem ersten Lauf anhand ihres exakten
  Quellstempels bis zum einstündigen Safety-Audit übersprungen; neue oder
  geänderte Auditquellen öffnen das Gate weiterhin beim nächsten Tick.
- Commit `b42c55efa` enthält beide Korrekturen. Die gezielten
  Regressionstests sind jeweils 1/1 grün; außerdem ist
  `cargo fmt --all -- --check` grün. Isolierter Rollout und erneute
  CPU-Abnahme standen als nächster Schritt an.
- Der committed Stand `0bb80a8bd` wurde aus dem isolierten Snapshot
  `/Volumes/tmp/ctox-idle-0bb80a8bd.6cQ1h6` gebaut. Das Git-Archiv hat
  SHA-256
  `2fa22ebf68fd2db65152f25595cb86766edd0944abf8dd681562eafd1b69b21d`.
  Der verwaltete Build dauerte 69 Minuten 46 Sekunden, der vollständige
  Rollout 74 Minuten 49 Sekunden und endete mit `phase=completed` ohne
  Fehler.
- Das aktive Release heißt `idle-cpu-b42c55efa`; die neue konsistente
  Sicherung liegt unter `backups/update-20260816T220430Z`, und
  `idle-cpu-60ff9c957` bleibt der unmittelbare Rollback-Slot. Die Retention
  entfernte danach die ältere Sicherung `update-20260816T204848Z` und den
  älteren Release `idle-cpu-c53a86588`.
- Build-Ausgabe, Release-Binary und `current`-Binary sind bytegleich und haben
  SHA-256
  `12cafa772278a43dac11f96dee87466a74ca92b46b9d960034747bde478dd157`.
  Der von `launchd` überwachte Dienst läuft unter PID `98157`, meldet
  `busy=false`, keine wartenden Tasks und keine aktiven Worker. Der RxDB-Peer
  hat einen frischen Heartbeat, keine Health-Fehler und meldet ohne Browser
  erwartungsgemäß `replicationUp=false`. Kurz- und Ein-Stunden-Probe folgen
  erst nach Abschluss der Startup-Arbeit.
- Die bisherigen Probeversuche nach dem Rollout wurden korrekt verworfen: Ein
  paralleler Codex-Task startete nacheinander die Workjet-Debugprozesse PID
  `1215` und `5134` gegen denselben produktiven State und Socket. Dabei wurde
  der verwaltete LaunchAgent nicht nur verdrängt, sondern seine Plist auf
  `/Volumes/tmp/workjet/ctox-cli-rc5-target/debug/ctox` umgeschrieben. Der
  Besitzer wurde über den Task-Kanal koordiniert; beide Fremdprozesse sind
  beendet. Die Plist ist wieder auf den installierten Wrapper
  `~/.local/bin/ctox` gerichtet, mit `plutil` validiert und neu geladen. Der
  Release `idle-cpu-b42c55efa` startet seitdem unter PID `7123`; die formale
  Messung beginnt erst nach nachgewiesenem Startup-Idle. Keine Messung einer
  fremden oder verdrängten PID wird als Release-Beweis verwendet.
- Auch PID `7123` wurde am 17.08.2026 um `01:35:29 +0200` zusammen mit dem
  vollständigen LaunchAgent von außen beendet, nachdem er zwei aufeinander
  folgende Idle-Samples mit `0.0 %` erreicht hatte. Plist und Releasepfad
  blieben dabei korrekt; das Service-Log enthält keinen Panic-/Fatal- oder
  Shutdown-Hinweis. Der konkurrierende Task wurde erneut ausdrücklich auf
  reine Remote-Arbeit und ein Verbot lokaler CTOX-/LaunchAgent-Eingriffe
  festgelegt und hat diese Abgrenzung ausdrücklich bestätigt. Auch dieser
  unvollständige Lauf zählt nicht als Abnahmebeweis.
- Unter exklusiver Prozesshoheit blieb der anschließend automatisch gestartete
  Release-Prozess PID `8418` stabil. Nach drei ruhigen Startup-Samples
  (`0.0 %`, `0.4 %`, `0.3 %`) blieb die erste formale 30-Sekunden-Probe jedoch
  rot: Prozess-CPU-Zeit `7.77 %`, p50 `0.2 %`, p95/Maximum `85.9 %`. Die
  Korrektheitsgates blieben grün (`retry_candidates=0`, unveränderte
  Command-Revisionen, keine offenen Intake-Fehler, steigende Idle-Ticks). Ein
  anschließendes 12-Sekunden-Stack-Sample belegt als weiteren periodischen
  Treiber `sync_configured_channels -> chat_native::service_sync ->
  process_chat_outbox -> ensure_outbox_schema`: Der 60-Sekunden-Kanalpoll
  öffnet die große SQLite-Datenbank erneut und kompiliert das Outbox-Schema,
  obwohl keine Nachricht fällig ist. Vor der Ein-Stunden-Abnahme wird auch
  dieser belegte Idle-Pfad beseitigt und die Kurzprobe wiederholt.
- Der Outbox-Pfad ist nun quellseitig begrenzt: Alle Adapter desselben
  30-Sekunden-Fensters teilen sich einen kanonisch nach Datenbankpfad
  getrennten Poll-Gate. `persist_queued_message` erhöht nach erfolgreichem
  Commit die Pfadgeneration und umgeht das Gate sofort, sodass neue oder
  retryfähige Zustellungen nicht auf den TTL-Ablauf warten. Der erste
  Regressionstest deckte den macOS-Alias `/var` gegenüber `/private/var` auf;
  nach Kanonisierung ist
  `duplicate_service_polls_are_suppressed_until_dirty_or_ttl` mit exakt `1/1`
  Treffern grün. `cargo fmt --all -- --check` ist ebenfalls grün. Rollout und
  erneute Kurzprobe stehen noch aus.
- Der nächste Rollout wird ausschließlich aus dem isolierten Snapshot
  `/Volumes/tmp/ctox-idle-b41cc74ca.vu2vmY` von Commit `b41cc74ca` gebaut.
  Das zugehörige Archiv
  `/Volumes/tmp/ctox-idle-b41cc74ca.vu2vmY.tar` hat SHA-256
  `de7bcbb0390550fb7d87b89efcde9562b9710df0bebaf87b49e3dc115d08e3da`;
  die extrahierte Quelle enthält das geprüfte Outbox-Gate. Fremde
  Arbeitsbaumänderungen sind damit weiterhin vom Release ausgeschlossen.
- Der Rollout `idle-cpu-b41cc74ca` wurde am 17.08.2026 nach einem erfolgreichen
  Release-Build (`90m36s`, gesamter Rollout `95m46s`) abgeschlossen. Build-,
  Release- und Current-Binary haben identisch SHA-256
  `4b8b596a739c0597ea81b610090be2b31b7e1bdaf90a15d56e03075ea41ab10f`.
  Der verwaltete LaunchAgent zeigt auf `~/.local/bin/ctox` und läuft mit dem
  neuen Current-Slot unter PID `12735`; der Status ist `busy=false`, ohne
  Pending Work oder Last Error. Der frische Backup-Slot ist
  `update-20260816T235952Z`, der unmittelbare Release-Rollback bleibt
  `idle-cpu-b42c55efa`. Retention entfernte den älteren Backup-Slot
  `update-20260816T220430Z` und den älteren Release `idle-cpu-60ff9c957`.
  Startup-Idle, erneute Kurzprobe und Ein-Stunden-Probe stehen noch aus.
- Die erste Startup-Abnahme dieses Releases wurde um `03:36 +0200` erneut
  durch denselben Parallel-Task invalidiert: Trotz ausdrücklicher vorheriger
  Abgrenzungsbestätigung schrieb dessen Workjet-Lauf die Plist auf
  `/Volumes/tmp/workjet/ctox-current-durable-revocation-check-target/debug/ctox`
  um, startete PID `13335` gegen den produktiven State und verdrängte
  Release-PID `12735`. Der fremde Lauf wird nicht als Messwert verwendet. Der
  Task wurde zur sofortigen Freigabe von PID und LaunchAgent sowie zu einem
  isolierten Test-Root verpflichtet und bestätigte die Bereinigung. PID
  `13335` ist beendet, das fremde Label entladen. Die Plist ist wieder auf
  `/bin/bash ~/.local/bin/ctox service --foreground` gerichtet, mit `plutil`
  validiert und geladen; Release `idle-cpu-b41cc74ca` startet nun unter PID
  `14641`. Auch dieser dritte fremd invalidierte Lauf zählt nicht als
  Abnahmebeweis.
- Auch PID `14641` verschwand anschließend um `03:38:58 +0200` zusammen mit
  dem geladenen LaunchAgent-Label. Die Plist blieb dabei korrekt und das
  Service-Log enthält keinen Panic-, Fatal- oder geordneten Shutdown-Hinweis.
  Damit ist auch dieser Lauf kein gültiger Idle-Nachweis. Der verantwortliche
  Parallel-Task bestätigte danach den erfolgreichen `bootout` des zuvor
  fremd gestarteten Dienstes, dass PID `13335` und das Label nicht mehr
  existieren, sowie den dauerhaften Verzicht auf weitere lokale CTOX-Start-,
  Stop-, Update-, `launchctl`- oder Foreground-Service-Aktionen. Die Plist
  wurde bei dieser abschließenden Bereinigung nicht verändert.
- Bis zur belastbaren Trennung konkurrierender lokaler Service-Aktionen wird
  der produktive LaunchAgent in dieser Kampagne nicht erneut manipuliert.
  Weitere Implementierung und Performanceprüfungen erfolgen remote oder mit
  isoliertem Test-Root. Die formale 30-Sekunden- und Ein-Stunden-Abnahme des
  Releases `idle-cpu-b41cc74ca` bleibt deshalb ausdrücklich offen.
- Eine vollständig lokale Reproduktion läuft seitdem ausschließlich unter dem
  isolierten Root `/Volumes/tmp/ctox-idle-b41cc74ca-isolated.voTfBX`. Quelle
  ist das bereits geprüfte Archiv von `b41cc74ca`; die drei Datenbanken stammen
  konsistent aus `backups/update-20260816T235952Z` und bestanden jeweils
  `PRAGMA quick_check`. Business-OS-Webserver, MCP-Autostart, Skill-Bootstrap
  und Backend-Prewarm sind über die typisierte Laufzeitkonfiguration dieses
  Test-Roots deaktiviert. Der produktive lokale Dienst und sein LaunchAgent
  wurden für diese Reproduktion weder gestartet, gestoppt noch verändert.
- Ein Zehn-Sekunden-Stack-Sample des isolierten Dienstes PID `29096` löst den
  verbliebenen periodischen Anteil auf den Acht-Sekunden-Statusstempel auf:
  `route_external_messages -> channel_router_source_stamp` wechselte zwischen
  Kern- und Business-OS-Datenbank, schloss dabei jedes Mal die vorherige
  Read-only-Verbindung und ließ SQLite das jeweils große Schema erneut
  freigeben und parsen. 582 von 6.184 Stack-Samples lagen in diesem Pfad.
- Der sechste Fix ersetzt deshalb die jeweiligen Einzelverbindungen durch
  kleine, pfad- und Dateiidentitätsgebundene Multi-DB-Reader-Caches. WAL-Commits
  bleiben sichtbar; Datenbankaustausch und Abfragefehler erzwingen weiterhin
  ein gezieltes Wiederöffnen. Commit `8b14ee057` enthält ausschließlich diese
  vier Quellfiles. Die beiden neuen Cache-Regressionen sowie die bestehenden
  Projection-Clock- und Router-Quellstempeltests sind jeweils exakt `1/1`
  grün; `cargo fmt --all -- --check` ist ebenfalls grün. Der globale
  Größenwächter bleibt ausschließlich wegen elf fremd veränderter Dateien rot;
  `service_status_sources.rs` liegt exakt auf seinem unveränderten Budget von
  987 Produktionszeilen. Der isolierte Snapshot liegt unter
  `/Volumes/tmp/ctox-idle-8b14ee057.9iXtWW`; sein 404-MB-Git-Archiv hat SHA-256
  `4a017993d53dc10953f02b827ecf5e9dcc06fbd400725b53070d9bd29399d84c`.
  Das im Archiv absichtlich nicht versionierte Pi-Sidecar-Bundle wurde dort
  reproduzierbar mit `npm ci && npm run build` erzeugt (SHA-256
  `a487654e3953c898e675b49ae705a7e7a6ff9024f7cd3f5748fddca90278a4ee`).
  Der isolierte Check deckte zusätzlich eine vorbestehende committed
  RxDB-Inkonsistenz auf: `rxdb_peer.rs` lieferte 13 Argumente an eine bereits
  14-argumentige Replikationsfunktion. Die zusammengehörige Browser-/RxDB-/
  Peer-Welle ist inzwischen als `3ada24cc7` separat committed; der unmittelbar
  danach isoliert sichtbar gewordene fehlende Command-Completion-Helfer wurde
  mit `fa100e322` ergänzt. `cargo check --bin ctox` aus einem reinen
  `git archive fa100e322` ist nach 15:09 Minuten mit Exitcode 0 abgeschlossen;
  der frühere Symbol- und Signaturfehler ist nicht mehr aufgetreten. Für die
  Nachhermessung liegt zusätzlich der unveränderte Snapshot
  `/Volumes/tmp/ctox-idle-fa100e322.puL5n3` vor. Sein Git-Archiv hat SHA-256
  `3be22e429b3b56fbc7ae0085f9e293b8d295601949825384339ab9dc9a8c58dd`;
  das daraus reproduzierbar gebaute Pi-Sidecar hat erneut SHA-256
  `a487654e3953c898e675b49ae705a7e7a6ff9024f7cd3f5748fddca90278a4ee`.
  Der Release-Build ist inzwischen nach 65:57 Minuten abgeschlossen. Das
  isoliert erzeugte arm64-Binary hat SHA-256
  `1496ce5f5b31a0b2b35e07cf19e862eb16a5be578c72c589f457ef0eb6bcc1fc`.
  Es wurde ausschließlich als PID `30871` gegen den isolierten Test-Root
  gestartet und nach dem Profiling sauber beendet; produktiver lokaler Dienst
  und `launchctl` blieben unverändert. Die Nachhermessung zeigte nicht mehr den
  Multi-Reader-/Schema-Reparse-Pfad, aber einen neuen dominanten Startup-
  Engpass im `business_commands`-Quellstempel.
- Der siebte Idle-Fix adressiert diesen gemessenen Restpfad. Der bisherige
  disjunktive JSON-Filter prüfte trotz null Kandidaten alle 12.309 lebenden
  Command-Dokumente; Loop-Metriken meldeten zuletzt 37.157 ms und maximal
  106.317 ms, und 3.208 Stack-Samples lagen in
  `business_commands_table_stamp -> sqlite3_step`. Die semantisch identische
  Abfrage ist deshalb in vier disjunkte `UNION ALL`-Zweige zerlegt. Auf
  derselben read-only geöffneten 2,1-GB-RxDB-Kopie verwendet jeder Zweig einen
  vorhandenen Expression-Index und liefert weiterhin null Kandidaten in
  35 ms. Es gibt weder einen neuen Index noch eine Schemamigration. Ein
  Query-Plan-Test verbietet Tabellenscans, und ein Semantiktest vergleicht die
  neue Aggregation über alle sechs recoverbaren Command-Typen mit dem
  kanonischen Altprädikat. Commit `e42e3386f` enthält ausschließlich diese
  Änderung und Tests. Beide neuen Tests sind `2/2` grün; der bestehende
  vollständige Command-Lifecycle-Test ist zusätzlich `1/1` grün. Der
  erste isolierte Snapshot-Neuaufbau wurde während der
  Abhängigkeitskompilierung kontrolliert beendet, um den gemeinsam genutzten
  Cargo-Lock freizugeben.
- Der anschließende reine Archiv-Build von `8e3edc015` ist nach 56:03 Minuten
  erfolgreich abgeschlossen. Das ausschließlich im isolierten Root gestartete
  arm64-Release-Binary hat SHA-256
  `ce6ab0b5873136a57ba8b99fee893d48dafc53f6bc00394958d0798b55870eec`.
  Der neue Command-Stamp blieb mit 32 bis 244 ms und null Kandidaten ruhig;
  Command-Revisionen und offene Intake-Fehler blieben unverändert, während die
  Idle-Ticks stiegen. Damit ist OA-6 in dieser Messung korrekt und endlich.
- Das CPU-Tor blieb dennoch rot: Die 30-Sekunden-Probe benötigte unter der
  laufenden Datenbanklast real 82,814 Sekunden, verbrauchte 22,98 CPU-Sekunden
  und maß p50 77,4 %, p95 95,2 %. Ein Stack-Sample ordnete die Last eindeutig
  der endlichen historischen Business-Records-Aufholung zu. Deren erste
  Slices verarbeiteten 388 Dokumente in 224,482 Sekunden und anschließend
  2.000 Dokumente in 201,537 Sekunden. Der Cursor rückte dabei auf Collection
  21 weiter; es ist kein unendlicher Wiederholungspfad, aber die alten großen
  Slices halten SQLite nahezu kontinuierlich besetzt und erklären auch den
  langsamen Query-Fetch-Start.
- Commit `b1a605dac` begrenzt deshalb jeden unvollständigen Recovery-Tick auf
  genau eine 25-Dokumente-Seite und ließ zunächst 60 Sekunden Ruhe. Der
  normale Command-/Browserpfad und die sofortige Projektion bleiben
  unverändert. Der neue Headroom-Guard ist `1/1` grün und
  `cargo fmt --all -- --check` ist grün. Der reine Archiv-Release-Build
  `/Volumes/tmp/ctox-idle-b1a605dac-full.nuMWDF` endete nach 25:12 Minuten
  erfolgreich; sein Binary hat SHA-256
  `e9acaeaef498b04c1abf845ff38a9e2f4e027531359256ca7e4c19460e5ed764`.
- Die erste 140-Sekunden-Nachherprobe deckte anschließend einen Messfehler im
  vorhandenen Dauer-Probe-Werkzeug auf: Dessen CPU-Zähler lief korrekt vom
  ersten bis letzten Prozesssample, der Nenner enthielt jedoch zusätzlich die
  unter Last 112 Sekunden dauernden SQLite-Snapshots davor und danach. Commit
  `e3db706fc` verwendet nun für Zähler und Nenner exakt dasselbe Samplefenster.
  Dadurch korrigiert sich die alte 30-Sekunden-Baseline von verwässerten
  27,75 % auf 76,52 % und die 60-Sekunden-Nachhermessung von verwässerten
  2,97 % auf 5,36 %. Die Korrektheitsgates blieben vollständig grün, das
  CPU-Tor war aber ehrlich noch knapp rot.
- Commit `39dcbb031` vergrößert deshalb ausschließlich das Ruhefenster zwischen
  den weiter auf 25 Dokumente begrenzten Recovery-Slices auf 120 Sekunden.
  Das lässt weiterhin etwa 18.000 historische Dokumente pro Tag aufholen. Der
  aktualisierte Headroom-Test ist `1/1` grün und Rustfmt ist grün. Der reine
  Archiv-Build `/Volumes/tmp/ctox-idle-39dcbb031.f2rKWt` endete nach 32:22
  Minuten erfolgreich; das Binary hat SHA-256
  `9d4ef4ef628ef79aef30d5a7626a96f9196dfa97d87da875bddc9d2d14389bc2`.
- Die korrigierte 300-Sekunden-Kurzprobe blieb mit 5,63 % knapp rot, p50 lag
  bei 0,4 %. Kandidaten, Revisionen und Intake-Fehler blieben null/stabil und
  die Idle-Ticks stiegen. Das Fenster enthielt neben zwei Recovery-Slices noch
  einmalige Warm-up-/Memory-Trim-Arbeit. Deshalb wird nicht erneut auf Basis
  eines kurzen Fensters gedrosselt: Die verbindliche warme Ein-Stunden-Probe
  wurde auf demselben Binary begonnen.
- Auch der warme Trend blieb anschließend rot: Über 558 Sekunden stieg die
  Prozess-CPU-Zeit um 33,31 Sekunden (5,97 %), während vier weitere
  Business-Records-Slices liefen und der Command-Intake ausschließlich
  Idle-Ticks mit null Rows meldete. Der Lauf wurde deshalb vorzeitig beendet,
  statt weitere 45 Minuten einen bereits belegten roten Trend zu sammeln.
- Commit `500f416c6` lässt nun fünf Minuten zwischen den weiterhin kleinen
  25er-Slices. Damit bleiben etwa 7.200 historische Dokumente pro Tag als
  endlicher Safety-Catch-up möglich und es entsteht rechnerisch wie gemessen
  genügend Reserve für andere periodische Sicherheitsarbeit. Headroom-Test
  `1/1` und Rustfmt sind grün. Der saubere Archiv-Build
  `/Volumes/tmp/ctox-idle-500f416c6.vpuE5M` endete nach 18:28 Minuten; das
  Binary hat SHA-256
  `b3e70d38aefcdb3b61f11a2aee1c0ddc154bbb5fa276a2ee60daf3dcc8b36cc1`.
- Der korrigierte 600-Sekunden-Vorfilter ist klar grün: 2,13 % tatsächliche
  Prozess-CPU, p50 0,1 %, p95 12,7 %. Kandidaten, Command-Revisionen und
  Intake-Fehler blieben null beziehungsweise unverändert, die Idle-Ticks
  stiegen.
- Die verbindliche warme Ein-Stunden-Probe ist ebenfalls grün: 144,87
  CPU-Sekunden in exakt 3.600,004 Sample-Sekunden entsprechen 4,02 %
  Dauerlast; p50 liegt bei 0,1 %, p95 bei 16,2 %. Kurze periodische Peaks bis
  468 % bleiben sichtbar, bilden aber keinen Hotloop. Retry-Kandidaten und
  offene Intake-Fehler blieben null, der Command-Revisionshash und die
  Changed-Table-Revision unverändert, während die Business-Command-Idle-Ticks
  von 23 auf 143 stiegen. Ohne Browser blieb `replicationUp=false`, der
  Heartbeat war vor und nach der Stunde frisch. Der isolierte Dienst wurde
  danach sauber beendet; produktiver lokaler Dienst und `launchctl` blieben
  unangetastet.
- Die davon getrennte Remote-Abnahme auf `thesen.ctox.dev` bestätigte, dass TID
  `1000424` nicht der hier belegte periodische Acht-Sekunden-Pfad war. Exaktes
  Host-Stackprofil war durch `perf_event_paranoid=4`, `ptrace_scope=1` und
  fehlende Stackwerkzeuge gesperrt; `/proc`-TID-Deltas trennten die Phasen aber
  eindeutig. Remote wurden stattdessen drei eigene Ursachen behoben:
  Person-Research-Recovery filtert nun SQL-seitig nur nichtterminale oder
  fehlende Projektionen (`recovery_candidates=0`), Business-Records verwendet
  einen per-Collection-Clock-Cursor (195/196 generische Scans vermeidbar), und
  konkurrierende Startup-Projektionsloops sind gestaffelt. Im identischen
  8–24-Sekunden-Fenster sank die CPU-Zeit von rund 34 auf 8,1 Sekunden
  (`-76 %`), `replicationUp` kam bei 8 statt rund 37 Sekunden; bei 105–120
  Sekunden gab es keinen heißen CTOX-Thread und keine Loop-Fehler. Das aktive
  Remote-Binary hat SHA-256
  `d6fb277953ab72b535bc91726046fe9e89a02472dda8e1a96a29a72e9c67f576`;
  sieben gezielte Guards sind grün. Die vier Dateien dieses lokalen Reader-Fix
  sowie `mission/channels/mod.rs` waren ausdrücklich nicht Teil des
  Remote-Releases.
- Der reproduzierbare Dauer-Probe ist mit Commit `071fda4bc` versioniert. Er
  misst Prozess-CPU-Zeit, CPU-p95/-Maximum, RxDB-Idle-Ticks, Kandidatenmenge,
  offene Intake-Fehler und den vollständigen Command-Revisionshash.

Beweis:
[`beweise/idle-channel-router-2026-08-16.md`](beweise/idle-channel-router-2026-08-16.md)

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

Browser-Nachmessung vom 17.08.2026:

- Der erste reale Matrixversuch provisionierte alle 304.515 Sellify-Dokumente,
  brach aber korrekt mit `QUERY_NOT_SUPPORTED` ab. Ursache war keine fehlende
  Tabelle, sondern die fail-closed native V1.5-Registry: Der synthetische
  Smoke hatte die vier Sellify-Schemas nur im Browser, nicht als lokales
  Runtime-Modul vor dem Native-Peer-Start registriert.
- Der Smoke materialisiert deshalb nun vor Peer-Start ein isoliertes
  `sellify-scale-smoke`-Modul mit denselben vier Schemas und
  `syncProfile: "demand-only"`. Der statische Regressionstest und der
  Benchmark-Smoke sind grün. Der Native-Peer registriert danach 182 statt 178
  Collections; `QUERY_NOT_SUPPORTED` tritt nicht mehr auf.
- Der erste vollständige reale Einzel-Smoke ist strukturell grün: vier
  Query-RPCs/-Responses, 650 materialisierte Dokumente, 50 sichtbare Zeilen,
  `localCoverage="windowed"`, `queryReady=true`, kein Vollpull und rund
  2,83 MB IndexedDB-Nutzung nach den vier Fenstern.
- Das Latenztor bleibt deutlich rot: 19.653 ms bis benutzbar, davon 13.929 ms
  für das erste Activities-Fenster. Dieselbe indexgestützte SQLite-Abfrage
  benötigt außerhalb des laufenden Dienstes nur 0,06 s (nach `ANALYZE`
  0,02 s). Der belegte Restengpass ist damit Startup-/Write-Lock-Konkurrenz im
  laufenden Native-Peer, nicht ein fehlender Sellify-Index. Eine 30×30-Matrix
  wäre vor Beseitigung dieses Engpasses nur eine teure Wiederholung des roten
  Befunds und bleibt deshalb offen.
- Die dauerhaft reproduzierbare Registrierung und die zugehörigen statischen
  Guards sind als Commit `dfccf3ede` getrennt committed. Rohdaten des realen
  Einzel-Smokes liegen unter
  `beweise/raw/sellify-scale-browser-single-2026-08-17.json`.
- Die vollständige RxDB-JavaScript-Suite ist auf diesem Stand mit 101 grünen,
  null roten und zwei mangels gebautem Wire-Daemon ausdrücklich übersprungenen
  Cross-Process-Smokes abgeschlossen.

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

Nachmessung vom 17.08.2026:

- Der erste Lauf in einem vollständig leeren isolierten Root erreichte die
  Command-Messung nicht: Der native Erstaufbau registrierte 180 Tabellen,
  1.263 Indizes und 537 Trigger und blieb während des 60-Sekunden-Fensters in
  `lifecyclePhase="bring_up"`. Dieser Cold-Schema-Befund gehört zum
  Handshake-/Boot-Gate, nicht zur warmen Command-Latenz.
- Derselbe bereits initialisierte Root brachte den multiplexen Peer beim
  zweiten Start nach 3.335 ms hoch. Der Warmup und die ersten fünf gewerteten
  Commands liefen, der sechste Command wurde nativ bereits nach 31 ms
  verarbeitet, blieb im Browser aber bis zum Timeout `pending_sync`.
- Die sieben Zeitmarken und die native SQLite-Zeile belegen ein Race: Der
  native Handler schrieb die terminale Projektion, danach gewann der
  verspätete Browser-Push mit einem neueren `pending_sync`-Dokument. Der
  Consumer übernahm anschließend diesen offenen Zustand in seinen
  Quellstempel und schlief, statt ihn kanonisch erneut zu projizieren.
- Commit `d9d2a040c` hält deshalb den tatsächlich vor Intake gelesenen
  Quellstempel fest und schließt zusätzlich das Lost-Wakeup-Fenster zwischen
  Stempelprüfung und dem Scharfschalten des SQLite-Notifiers. Drei gezielte
  Idle-Gate-/Notifier-Regressionstests sind mit `3/3` Treffern grün. Der
  saubere Archiv-Release-Build und die erneute 30-Command-Messung laufen; ein
  Performancegewinn ist bis zum fertigen Report ausdrücklich noch nicht
  abgenommen.

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

Aktueller Boot-Befund:

- Ein vollständig leerer Test-Root verfehlt das 60-Sekunden-Gate bereits beim
  einmaligen Aufbau der 178 nativen Collection-Schemas. Der Peer hatte noch
  keinen Pool erzeugt; dies ist kein serieller Signaling-Roundtrip, sondern
  lokaler SQLite-Schemaaufbau vor der Room-Verhandlung.
- Auf demselben initialisierten Root lag der native Peer-Start anschließend
  bei 3.335 ms und nutzte genau die bestehende multiplexe Verbindung. Damit
  ist der Warm-Start einzeln unter fünf Sekunden, die geforderte 30-Lauf-p95-
  Abnahme sowie der Cold-Setup-Fix bleiben offen.

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
| `f9017f579` | redundanten teuren Idle-Queue-Poll begrenzen |
| `071fda4bc` | reproduzierbaren Dauer-Idle-Probe ergänzen |
| `0a4ab13b2` | irrelevantes Modul-Asset-Hashing im External-SQL-Idle-Poll vermeiden |
| `b42c55efa` | Projection-Clock-Verbindung wiederverwenden und unveränderte Harness-Findings bis zum Safety-Audit schlafen lassen |
| `07c2ef0ca` | nachweislich transiente Intake-Failures konservativ auflösen |
| `43889102a` | Knowledge-Katalog über eine SQLite-Verbindung aufbauen |
| `8b14ee057` | Status- und Kommunikationsstempel über mehrere SQLite-Reader wiederverwenden |
| `24f81bdf7` | Multi-Reader-Idle-Diagnose im Kampagnenplan festhalten |
| `d27a46d12` | isolierten Stack-/Reader-Befund als Rohbeweis versionieren |
| `3ada24cc7` | Browser-Streaming, RxDB/Peer und Research-Recovery zusammengehörig integrieren |
| `fa100e322` | Command-Completion der Recovery dauerhaft persistieren und projizieren |
| `8f43dcc58` | indexuntauglichen Command-Stamp-Hotpath und Vorhermessung dokumentieren |
| `e42e3386f` | Command-Stamp über vier indexgerechte Lifecycle-Zweige aggregieren |
| `dfccf3ede` | isolierte Sellify-Scale-Schemas vor dem nativen Peer registrieren und den echten Browser-Smoke härten |
| `b1a605dac` | historische Business-Records-Aufholung auf eine kleine Seite pro Ruhefenster begrenzen |
| `e3db706fc` | Idle-Probe auf den exakten CPU-Samplezeitraum korrigieren |
| `39dcbb031` | nach gemessenem 5,36-%-Rest das Recovery-Ruhefenster auf 120 Sekunden vergrößern |
| `500f416c6` | nach weiterem warmen 5,97-%-Trend das Recovery-Ruhefenster auf 300 Sekunden vergrößern |
| `d9d2a040c` | verspätete Browser-Command-Pushes erneut erkennen und kanonisch projizieren |

## Bekannte rote Baseline und nicht übernommene Paralleländerungen

- Die vollständige Browser-Suite meldete 93 grüne, sieben rote und zwei
  übersprungene Cross-Process-Tests. Alle sieben roten Tests wurden im
  isolierten Baseline-Archiv reproduziert und sind keine Regression dieser
  Kampagne.
- Der globale Größenwächter bleibt wegen derzeit elf bereits anderweitig
  veränderter Dateien rot. `service_status_sources.rs` ist mit exakt 987
  Produktionszeilen budgetkonform; die Budgets wurden nicht angehoben.
- Im gemeinsamen Arbeitsbaum liegen weiterhin zahlreiche nicht zu dieser
  Kampagne gehörende Änderungen, darunter `app.js`, Browsermodule und weitere
  Runtime-Dateien. Sie bleiben uncommitted, bis ihre jeweilige Arbeit separat
  abgenommen wird.

## Nächste Ausführungsreihenfolge

1. Die lokale Betriebsgrenze respektieren: keine weitere Manipulation des
   produktiven LaunchAgents; Messungen nur remote oder im isolierten Test-Root.
2. Den sauberen Archiv-Build von `d9d2a040c` abschließen und den warmen
   30-Command-Smoke im bereits initialisierten Root wiederholen; nur den danach
   noch dominierenden Abschnitt weiter optimieren.
3. Danach den Sellify-Einzel-Smoke auf demselben Code-Stand erneut messen.
4. Erst nach bestandenem Einzel-Smoke eine echte 30-Lauf-Cold-/Warm-
   Browsermatrix auf dem synthetischen Scale-Store ausführen.
5. Den Cold-Schemaaufbau getrennt vom warmen Handshake optimieren und danach
   Reconnect-, Peerwechsel- und Multi-Tab-Abnahme ausführen.
6. Nach Freigabe der Arbeitsregion `app.js` move-only zerlegen.
7. Verbleibende fremde Größenwächter jeweils in ihrer eigenen Arbeitsspur
   bereinigen, ohne Budgets anzuheben.
8. Kundenrollout und OA-6/OA-8/OA-9-Livenachweise.
9. OA-7-Kompaktierung im exklusiven Wartungsfenster.

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
- `docs/dev/beweise/idle-channel-router-2026-08-16.md` — Diagnose und lokaler
  Nachweis des periodischen Idle-CPU-Pfads.
- `docs/dev/beweise/raw/` — maschinenlesbare Rohmessungen und Move-Beweise.
- `docs/ctox-rxdb.md` — kanonische Architektur des RxDB-/WebRTC-Datenpfads.
