# Idle-CPU-Befund des Channel-Routers

Stand: 16.08.2026

## Einordnung der ersten 100-%-Beobachtung

Die zuerst beobachteten 104,4 % CPU entstanden während der ausdrücklich
gestarteten Sellify-Cold-/Warm-Matrix. Zu diesem Zeitpunkt bearbeitete ein
isolierter CTOX-Prozess echte WebRTC-Abfragen gegen 304.515 synthetische
Sellify-Dokumente. Nach Abbruch der Matrix waren ihre Prozesse beendet und die
Ports 8877 sowie 18876 geschlossen. Dieser Wert war daher keine Idle-Messung.

## Separater Idle-Befund

Der regulär installierte Dienst zeigte danach dennoch periodische Ausschläge.
Fünf Messpunkte im Abstand von zwei Sekunden ergaben 48,1 %, 0,6 %, 28,1 %,
0,1 % und 0,5 % CPU. Ein fünfsekündiger Stack-Sample ordnete 297 von 3.259
Wall-Samples dem Pfad

`route_external_messages -> pending_queue_task_count_uncached -> channel_projection_tables_exist`

zu. Der redundante Safety-Poll öffnete alle 30 Sekunden die 1,1-GB-Kerndatenbank
erneut. SQLite musste dabei das große Schema erneut laden und parsen. Der
Signaling-Dienst blieb bei 0,0 % CPU.

## Änderung

Der Router prüft seine günstigen Datei- und Projektionsstempel weiterhin alle
acht Sekunden. Echte Änderungen an Queue, Routing, Zeitplan oder
Dokument-Commands öffnen das Idle-Gate dadurch unverändert. Nur der zusätzliche
Vollzugriff bei vollständig unverändertem Zustand folgt nun dem bereits
vorhandenen einstündigen Safety-Fenster statt einem 30-Sekunden-Intervall.

Ein Regressionstest setzt das Idle-Alter gezielt auf 31 Sekunden und weist
nach, dass dabei keine SQLite-Verbindung geöffnet wird. Der bestehende
Fail-safe-Test weist weiterhin nach, dass ein absichtlich inkonsistenter
Stempel nach Ablauf des Safety-Fensters erkannt wird.

## Lokale Prüfung

- `rustfmt --edition 2021 --check src/core/service/service.rs`: grün.
- `cargo test --bin ctox channel_router_preflight_ --no-default-features`:
  drei bestanden, null fehlgeschlagen, 2.841 ausgefiltert.
- `service.rs`: weiterhin exakt 21.820 Produktionszeilen bei Budget 21.820.
- Der globale Größenwächter war ausschließlich wegen acht fremder, bereits
  vorhandener Arbeitsbaumänderungen rot; kein Budget wurde verändert.

Die installierte Binary wurde für diesen lokalen Quellcode-Nachweis nicht
ersetzt und der Dienst nicht neu gestartet. Eine Nachhermessung am ausgerollten
Build bleibt deshalb ausdrücklich offen.

Maschinenlesbare Rohdaten:
[`raw/idle-channel-router-2026-08-16.json`](raw/idle-channel-router-2026-08-16.json)

## Isolierter Multi-Reader-Nachbefund vom 17.08.2026

Nach dem fünften Fix blieb in einem vollständig isolierten Test-Root ein
periodischer Anteil übrig. Ein Zehn-Sekunden-Sample ordnete 582 von 6.184
Stacks dem Statuspfad

`route_external_messages -> channel_router_source_stamp -> sqlite3Close -> sqlite3SchemaClear`

zu. Der Statusstempel wechselte zwischen Kern- und Business-OS-Datenbank. Der
bisherige Einzelcache schloss dabei jeweils den anderen Reader und zwang SQLite
zum erneuten Freigeben und Parsen der großen Schemata.

Commit `8b14ee057` ersetzt die Einzelverbindungen durch kleine, pro Thread auf
acht Einträge begrenzte Multi-DB-Caches. Pfad, Gerät und Inode bleiben Teil der
Identität; WAL-Commits bleiben sichtbar, während Dateiaustausch oder
Abfragefehler gezielt neu öffnen. Vier betroffene Regressionstests sind jeweils
exakt `1/1` grün, und der globale Formatcheck ist grün. Die vorbestehende
RxDB-/Peer-Signaturinkonsistenz und die dabei sichtbar gewordene fehlende
Command-Completion-Kante sind mit `3ada24cc7` und `fa100e322` getrennt
committed. Ein vollständiges `cargo check --bin ctox` aus einem echten
`git archive fa100e322` ist nach 15:09 Minuten mit Exitcode 0 abgeschlossen.
Release-Build sowie 30-Sekunden- und Ein-Stunden-Nachherlauf bleiben offen.

Maschinenlesbare Rohdaten:
[`raw/idle-multi-reader-diagnosis-2026-08-17.json`](raw/idle-multi-reader-diagnosis-2026-08-17.json)
