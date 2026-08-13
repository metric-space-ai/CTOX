# Produktmessung M1–M5 — erste Fakten (12.08.2026, 21:00–22:00)

## WICHTIGE KORREKTUR (22:00): Release-Nachmessung

Die Debug-Zahlen unten für M1 waren ein **Mess-Artefakt des Debug-Builds**.
Mit Release-Binary (identischer Root, identischer Aufbau):

| Messtor | Debug | **Release** | Einordnung |
|---|---|---|---|
| M1 Kaltstart bis HTTP | Median 15,5 s | **Median 1,45 s** (0,29/0,83/1,45/2,48 s; Erstlauf des frischen Binaries einmalig 49 s) | Daemon ist ok |
| M2 Browser-Boot bis alle 15 Collections live | 15,9 s | **15,1 s** | **strukturell**, kein Debug-Artefakt: serielle Start-Queue ~1 s/Collection |
| M3 Command-Roundtrip (30 Commands, completed) | p50 3,56 s / p95 7,47 s | **p50 1,42 s / p95 4,28 s** (min 0,74, max 9,2) | besser, aber weiterhin Sekunden statt Millisekunden |
| M3 waitForTerminal-Defekt | reproduziert | **unverändert vorhanden** | Architektur, kein Build-Effekt |
| M4 Churn-Erholung + Verlustprobe | 2/10 sauber | (nicht wiederholt) | — |

Konsequenz für die Hebel: ① Tracking-Defekt und ② serielle Start-Queue
bleiben die Haupthebel; ③ desktop_icons trat im Release-Lauf nicht als
Ausreißer auf (Cache-Verdacht, weiter beobachten); ④ Roundtrip-Sekunden
bleiben real (p50 1,4 s für einen No-op-Status).

Messaufbau: isolierter Root `/Volumes/tmp/ctox-pipeline/m1-root` (CTOX_ROOT),
Debug-Binary aus `b9ed00757`-Nachfolge, `ctox business-os serve --addr
127.0.0.1:8917`, Browser = Claude-Browser-Pane, Shell mit `?rxdbSmoke`.
Die Produktivinstanz des Owners wurde nicht berührt. Release-Nachmessung
läuft (Debug-Zahlen sind obere Schranken, die Struktur der Befunde ist
davon unabhängig).

## M1 Kaltstart (Daemon-Start → HTTP antwortet)

| Lauf | Sekunden |
|---|---|
| Erststart, leerer Root (einmalig, inkl. Store-Anlage) | ~88 |
| Neustart 1 | 8,3 |
| Neustart 2 | 27,0 |
| Neustart 3 | 23,9 |
| Neustart 4 | 13,5 |
| Neustart 5 | 15,5 |

**Median 15,5 s, Streuung 8–27 s** — bei praktisch leerem Store. Der Server
bindet den Port absichtlich erst nach Store-Parse und Peer-Vorbereitung.
Das „Peer bereit“-Kriterium der Messreihe war unbrauchbar (`replicationUp`
wird erst mit verbundenem Browser wahr); nicht gewertet.

## M2 Browser-Boot (Navigation → Collections live; leerer Store)

- Shell DOM-interaktiv: **1,75 s**
- Alle 15 Boot-Collections initial repliziert: **15,9 s**
- Verursacher: `desktop_icons` allein **11,1 s** Erstreplikation (bei ~null
  Daten); die übrigen Collections starten **seriell im ~1-s-Takt**
  (serialisierte Start-Queue), Fertigstellungszeiten 6,8 / 8,8 / 10,8 /
  11,0 / 11,8 / 12,8 / 15,9 s.

Die vom Owner benannten „vielen Sekunden“ sind damit lokalisierbar:
serielle Collection-Starts plus ein einzelner Ausreißer.

## M3 — Nachtrag (21:30): Verarbeitung funktioniert, Rückmeldung defekt, Latenz in Sekunden

Direktabfrage der `business_commands`-Collection zeigt: **alle 60 abgesetzten
Commands kamen an und wurden nativ `completed`.** Der Defekt liegt allein im
Browser-Tracking (`waitForTerminal`), das trotz Erfolg den
QUERY_STREAM-Fehler wirft — die UI meldet Fehlschlag bei serverseitigem
Erfolg. Echte Roundtrip-Latenz aus den Dokument-Zeitstempeln
(created→updated, 60 Commands, trivialer Status-Command):
**p50 3,56 s · p95 7,47 s · min 0,18 s · max 9,0 s** — Sekunden statt
Millisekunden.

## M3 Command-Bus — ursprünglicher Befund (Tracking-Pfad)

30 Dispatches `ctox.provider_subscription.status` über den offiziellen
Modul-Command-Bus (`createModuleContext({id:'ctox',
collections:['business_commands']})`): **alle scheitern deterministisch** mit

```
QUERY_NOT_SUPPORTED / SQLITE_QUERY_STREAM_UNSUPPORTED
```

Wurzel: `src/core/rxdb/src/storage/sqlite/instance.rs` (~1613) verweigert
auf dem WebRTC-Query-Fetch-Hotpfad jede Mango-Query, die nicht nach SQL
kompilierbar ist („refusing Rust matcher fallback“); der Tracking-Pfad des
Command-Bus stellt genau so eine Query. Die Command-Plane-Diagnostik zählt
`counters: {}` — **in dieser frischen Instanz ist nie ein Command
durchgekommen.** p50/p95 sind erst nach dem Fix messbar.

Offene Einordnung: ob der reguläre Modul-UI-Pfad dieselbe Query stellt
(dann Totalausfall des Command-Plane auf frischen Instanzen) oder nur der
Smoke-Kontext, ist der erste Prüfschritt des Fixes.

## M4 Abbruch-Churn (Zyklus 1 von 10)

Daemon hart getötet, 20 s Ausfall, Neustart (~18 s bis HTTP). Browser:
15 Collections getrennt → kontinuierlicher Wiederaufbau → **~30 s nach
Serverrückkehr alle wieder verbunden**. Kein hängender Zustand, Room-Circuit
blieb geschlossen, `lastError` null. (Vier Collections melden dauerhaft
Status `reused` — das ist Bridge-Wiederverwendung, kein Fehler.)
Ausstehend nach Zyklus 1: 9 weitere Zyklen, Verlustprüfung.

### Zyklus 2 (mit Verlustprobe) — bestanden

Command 252 ms vor dem Daemon-Kill abgesetzt (Dispatch erfolgreich, lokales
IndexedDB-Write). Daemon 15 s tot, Neustart, Resynchronisation vollständig
(0 getrennte Collections ≤105 s nach Kill inkl. Ausfall- und Startzeit).
Die Verlustprobe: **vorhanden und nativ `completed`** — der ungesyncte Write
überlebte den Kill und wurde nach Rückkehr verarbeitet. Kein Datenverlust.
(Anmerkung: `idempotencyKey` aus dem Dispatch-Aufruf wird ignoriert und
systemseitig vergeben — kleiner API-Vertragsbefund am Rande.)

## M5 Multi-User — ausstehend

Braucht zweite echte Nutzersitzung (Invite-Flow); Multi-Tab-Sync wäre nur
ein Ersatzmaß.

## Vorläufige Produkt-Einordnung (nur Fakten)

- Boot bis „alles live“ auf leerem Store: ~16 s → für ein Desktop-OS-Gefühl
  zu langsam; Hebel sind benannt (serielle Starts, desktop_icons).
- Command-Plane auf frischer Instanz: defekt (M3).
- Abbruch-Erholung: funktioniert, ~30 s (Zyklus 1).

Produktnote bleibt bis zum vollständigen M1–M5-Satz offen; nach aktueller
Faktenlage wäre sie schlecht („funktioniert, aber langsam, ein Kernpfad
defekt“). Repo-Note (B) und Produktnote sind getrennte Größen.

---

# NACHMESSUNG M3 nach TEMPO-1 (12.08., ~09:40)

Aufbau identisch zur Vormessung, aber diesmal **gegen einen sauberen
`git archive HEAD`-Auszug** gebaut und ausgeliefert (nicht gegen den geteilten
Arbeitsbaum): Release-Binary aus `2cde146c0`, Shell-Assets aus demselben Auszug
(`command-bus.js` mit 2× `findDocumentsById` verifiziert, Bundle-Buster
`20260812-tempo1-v95`), isolierter Root, 30 Commands
`ctox.provider_subscription.status`.

## Ergebnis

| Größe | Vorher | Nachher | Bewertung |
|---|---|---|---|
| `SQLITE_QUERY_STREAM_UNSUPPORTED` / `QUERY_NOT_SUPPORTED` | bei **jedem** Tracking-Aufruf | **0 von 30** | **behoben** |
| Fehlgemeldete Commands (UI-Fehlschlag bei Server-Erfolg) | 30/30 | **0/30** | **behoben** |
| Command-Status | 30/30 completed | 30/30 completed | unverändert gut |
| Dokument-Roundtrip (created→updated) p50 | 1.420 ms | **1.497 ms** | unverändert |
| Dokument-Roundtrip p95 | 4.282 ms | 6.227 ms | schlechter, siehe Vorbehalt |
| Dispatch-Wanduhr p50 / p95 | (nicht erhoben) | 4.149 / 15.216 ms | neue Basiszahl |

## Zwei Messfehler auf meiner Seite, die ich offenlege

1. **Falsche Kennung.** Mein erster Nachmesslauf übergab `waitForTerminal` das
   ganze Dispatch-Ergebnis statt der Kennung — das Feld heißt `command_id`
   (snake_case), nicht `commandId`. Folge: 20-s-Zeitabläufe, die wie ein
   Produktdefekt aussahen. Es war mein Aufbau. Korrigiert und neu gemessen.
   Nebenbefund: `dispatch()` wartet bereits intern auf den Endzustand und
   liefert `status: "completed"` zurück — ein separates `waitForTerminal` ist
   für den Normalfall gar nicht nötig.
2. **Vergleich zweier verschiedener Größen.** Die Vorher-Zahl (1,42 s) stammt
   aus Dokument-Zeitstempeln, die naheliegende Nachher-Zahl (4,1 s) ist
   Wanduhr-Zeit des ganzen Dispatch-Aufrufs. Nicht vergleichbar. Oben steht
   deshalb beides getrennt; vergleichbar ist nur die Dokument-Zeile.

## Vorbehalt zur p95-Verschlechterung

Der Nachher-Lauf lief unter deutlich höherer Maschinenlast (5-Minuten-Mittel
14–28, parallel ein Sol-Worker-Bau) als der Vorher-Lauf. p50 ist praktisch
unverändert (1,42 → 1,50 s), p95 ist von 4,3 auf 6,2 s gestiegen. Ich werte
das **nicht** als Regression durch TEMPO-1, aber auch nicht als Verbesserung:
Der Latenz-Hebel ist damit **nicht** eingelöst. Die Sekunden bleiben.

## Was TEMPO-1 tatsächlich eingelöst hat

Der Rückmelde-Defekt ist weg — die Oberfläche meldet keinen Fehlschlag mehr,
wenn der Server erfolgreich war. Das war der schwerwiegendere der beiden
Punkte (falsche Fehlanzeige ist schlimmer als langsam). Die Latenz selbst
bleibt offen; der nächste Hebel dafür ist nicht der Consumer-Backoff, sondern
die Kette aus festen Takten (nativer Intake-Poll 1 s, Browser-Revalidate,
Projektions-Refresh) plus der Multi-RTT-Handshake.

---

# KORREKTUR (13.08.): "Beweistest grün" war zunächst unbelegt

Beim Landen von SCHLEIFE-1 (147b2aaf2) habe ich den Beweistest als grün
gemeldet. Der Lauf hatte aber `0 passed; 2828 filtered out` ausgegeben — ich
hatte `cargo test <kurzname> -- --exact` aufgerufen, und `--exact` verlangt den
VOLLEN Pfad. Kein Test hatte gepasst; „ok" bezog sich auf einen leeren Lauf.

Nachgeholt mit korrektem Filter:

```
test business_os::rxdb_peer::tests::exhausted_conflicting_command_is_not_rewritten_or_recorded_again ... ok
test result: ok. 1 passed; 0 failed; ... finished in 56.20s
```

Der Fix ist damit **tatsächlich** belegt: zwei Consumer-Sweeps über dasselbe
Konfliktdokument lassen Revision und Intake-Failure-Zeilenzahl unverändert.
Die Behauptung war richtig, der Beweis fehlte — das ist derselbe Fehlertyp wie
„im Repo gelandet" gleich „beim Kunden wirksam", nur eine Ebene tiefer.

**Regel daraus:** Bei jedem Testlauf die Zahl gelaufener Tests zitieren. Ein
Lauf mit 0 gelaufenen Tests ist kein grüner Lauf. Diese Regel steht jetzt in
jedem Worker-Brief.

## Zweiter Befund: mein Commit hat den Größenwächter rot gemacht

`rxdb_peer.rs` steht nach 147b2aaf2 bei **10.042** Produktionszeilen gegen ein
Budget von **9.941** — 101 zu viel. Gefunden hat das nicht ich, sondern der
Planungslauf. Das Budget wird nicht angehoben; der Schnitt (Command-Intake in
eine eigene Datei) läuft als Arbeitspaket AP4.
