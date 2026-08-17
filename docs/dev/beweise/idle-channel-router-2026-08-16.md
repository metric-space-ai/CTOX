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

## Isolierter Command-Stamp-Nachbefund vom 17.08.2026

Der integrierte Snapshot `fa100e322` wurde inzwischen als Release gebaut. Das
Binary hat SHA-256
`1496ce5f5b31a0b2b35e07cf19e862eb16a5be578c72c589f457ef0eb6bcc1fc`.
Der ausschließlich gegen den isolierten Test-Root gestartete PID `30871` wurde
nach dem Profiling sauber beendet; der produktive lokale Dienst und
`launchctl` blieben unangetastet.

Die Multi-Reader-Korrektur entfernte den zuvor belegten Schema-Reparse-Pfad.
Der Dienst erreichte dennoch noch nicht den Idle-Zustand: Der
`business_commands`-Quellstempel benötigte trotz null Kandidaten zuletzt
37.157 ms und maximal 106.317 ms. Ein Stack-Sample lag mit 3.208 Samples
vollständig in `business_commands_table_stamp -> sqlite3_step`. Ursache war
ein disjunktiver JSON-Prädikatsausdruck, für den SQLite nur `deleted = 0`
indexieren konnte und deshalb alle 12.309 lebenden Commands prüfte.

Die logisch identische Kandidatenabfrage ist nun in vier disjunkte
`UNION ALL`-Zweige für `pending_sync`, `waiting_dependencies`, `accepted` und
nichtterminales `failed` geteilt. Auf derselben read-only geöffneten 2,1-GB-
RxDB-Kopie verwendet jeder Zweig einen vorhandenen Expression-Index; ein neuer
Index oder eine Schemamigration ist nicht nötig. Die direkte Nachherabfrage
lieferte weiterhin null Kandidaten in 35 ms. Das ist gegenüber dem letzten
Loop mindestens Faktor 1.061 und gegenüber seinem Maximum Faktor 3.037.
Commit `e42e3386f` enthält ausschließlich die Abfrage und ihre beiden neuen
Regressionstests. Query-Plan und Semantik sind `2/2` grün; der bestehende
vollständige Command-Lifecycle-Test ist zusätzlich `1/1` grün. Der
isolierte Snapshot-Neuaufbau wurde während der Abhängigkeitskompilierung
kontrolliert beendet, um den gemeinsam genutzten Cargo-Lock freizugeben. Ein
reiner Archiv-/Release-Build sowie die isolierte Kurz- und Ein-Stunden-Probe
bleiben deshalb offen.

Maschinenlesbare Rohdaten:
[`raw/idle-business-commands-stamp-2026-08-17.json`](raw/idle-business-commands-stamp-2026-08-17.json)
