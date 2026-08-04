# SYNC-D: Kompensationen beseitigen, nicht Dateien verkleinern

Nachfolger des `store.rs`-Plans (`ctox-store-refactoring-plan-2026-07-29.md`,
abgeschlossen: 80 Commits, Gottfunktion zerlegt, Grenzen vertraglich). Dieser
Plan korrigiert dessen Grundfehler und hält fest, was die erste Welle der
Pipeline gelernt hat.

## Der Fehler des Vorgängerplans

Der `store.rs`-Plan hat nach **Dateigröße und Nahtzahl** priorisiert. Beides
sagt nichts über Brüchigkeit. Ergebnis: eine Datei um 60 % verkleinert, dabei
16 Reparaturen beseitigt — von 172 im Repo. Schlechtes Verhältnis, weil das
Auswahlkriterium falsch war.

**Die richtige Metrik ist Kompensationsdichte:** wie viele Funktionen einen
Zustand hinterher reparieren, den der Schreibpfad hätte richtig anlegen müssen.
Sie sagt voraus, wo Zusicherungen gebrochen werden. Sie existierte nicht, bis
die Prämissenprüfung sie erzwang.

## Was eine Kompensation ist (und was nicht)

Eine Kompensation gleicht aus, dass **woanders** schlecht geschrieben wird.
Marker im Namen: `repair`, `reconcile`, `recover`, `backfill`, `heal`,
`resync`. ABER: der Name ist ein Verdacht, kein Urteil. Von 986 namentlichen
Treffern der Prämissenprüfung waren nur **73 echte Schuld (7,4 %)**.

Legitim trotz Marker: `read_repair` (Fachbegriff verteilter Systeme),
`rewrite_and_copy` (wörtlich gemeint), eine einmalige versionierte Migration
MIT Auslöser. Schuld: etwas, das beim Lesen schreibt; das auf Fehlertext
entscheidet; ein Timer, der einen unzuverlässigen Pfad kaschiert; eine
Reparatur an persistierten Daten, deren Ursache noch existiert.

## Reihenfolge — nach Dichte, gemessen am 03.08.

| Datei | Prod-Zeilen | Reparaturen | Dichte |
|---|---|---|---|
| `mission/queue.rs` | 2.627 | 11 | **1:238** ← am dichtesten |
| `business_os/rxdb_peer.rs` | 14.760 | 13 | 1:984 |
| `service/service.rs` | 26.173 | 49 | 1:534 |

`service.rs` hat die meisten (49), `queue.rs` die höchste Dichte. Klein und
dicht zuerst: `queue.rs` ist in einer Sitzung durchmessbar, `service.rs` ist
ein eigener Feldzug.

**KORREKTUR 03.08., beim ersten Einsatz der Metrik.** Die Dichte zählt
FUNKTIONEN und überschätzt dadurch. In `queue.rs` gehören zehn der elf
Treffer zu EINER Sache — der agentischen Reparatur (Prompt bauen, Bericht
parsen, Aktionen anwenden, Ereignis schreiben, Zeilen 1204–1620). Real sind
es **zwei Belange**, nicht elf.

Die Metrik bleibt brauchbar, aber sie ist ein Sortierschlüssel, kein Maß.
Vor der Priorisierung gehört je Datei geprüft, wie viele UNABHÄNGIGE Belange
hinter den Treffern stehen. Sonst gewinnt die Datei mit den meisten Helfern,
nicht die mit den meisten Ursachen.

**`src/tools/appsec-pentest/` ist ausdrücklich AUSSERHALB dieses Plans**
(Owner-Entscheid 03.08.). Die Datei ist die größte des Repos, aber sie gehört
nicht in diese Kampagne und wird nicht vermessen, nicht benotet, nicht
angefasst.

## Das Verfahren (aus der ersten Welle gelernt)

**Zwei Runden je Befund, nicht eine.** Die Ursache liegt fast nie in der Datei,
in der die Kompensation steht — eine Kompensation existiert ja, WEIL woanders
schlecht geschrieben wird.

1. **Lokalisieren.** Worker liest, benennt die Ursache mit `datei:zeile`,
   ändert NICHTS. Ein Fix in der zu engen Datei wäre nur eine umbenannte
   Kompensation (Pollen, Hintergrund-Abgleich, zweiter Reparaturpfad). Drei
   Läufe der ersten Welle endeten so — und lieferten den Befund, ohne den
   Runde zwei unmöglich gewesen wäre. Zwei zeigten unabhängig auf DIESELBE
   Ursache (`command-bus.js`), die keiner allein gefunden hätte.
2. **Reparieren.** Whitelist enthält die Ursachen-Dateien UND einen Testort.
   Erst die Ursache, dann das Netz — die Kompensation fällt NACH ihrer Ursache,
   nie davor. Bleibt ein zweiter Auslöser (z. B. harter Crash neben geordnetem
   Stop), bleibt das Netz stehen: begründet unter `verblieben`.

**Whitelist mit Testort.** Zweimal in Welle 1 landete der Beweis in `/tmp`,
weil die Whitelist nur die Quelldatei nannte — dort führt ihn niemand aus. Jede
Reparatur-Whitelist enthält die Testdatei.

**Abnahme im isolierten Worktree.** Der geteilte Baum trägt keine belastbaren
Testaussagen: zehn Worker mit disjunkten Dateien zerstören einander trotzdem
die Bauten, weil eine Rust-Crate als Ganzes kompiliert. Maßgeblich ist:
Worktree gegen `origin/main`, genau ein Patch, eigener `CARGO_TARGET_DIR`, das
gitignorierte pi-sidecar-Bundle hineinverlinkt. Kaltbau ~80 min, danach jeder
Testlauf Sekunden. Für die Abnahme richtig, pro Worker zu teuer.

**Gegenprobe selbst führen, nicht aus dem Bericht übernehmen.** Einmal in
Welle 1 brach mein Gegenprobe-Skript vor der Änderung ab, der Test lief gegen
unveränderten Code und meldete grün — bewies nichts. Die Gegenprobe ist erst
gültig, wenn der Rückbau den Test ROT macht.

## Infrastruktur (liegt auf dauerhaftem Speicher)

- Pipeline: `/Volumes/tmp/ctox-pipeline/` (NICHT `/private/tmp` — das räumt die
  Maschine unter Plattendruck; hat einmal die gesamte Prämissenprüfung
  gekostet).
- Zielbäume je Worker: `/Volumes/tmp/ctox-pipeline-targets/<ID>` — die Wache
  räumt sie nach gesichertem Report ab (je 18–25 GB).
- Reports: die Wache kopiert aus `/private/tmp` sofort nach
  `/Volumes/tmp/ctox-pipeline/reports/`, VOR dem Aufräumen des Zielbaums.

## Bilanz Welle 1

10 Aufträge, 6 committet, 3 Befund-ohne-Diff, 1 zurück (I-008, `dist/` nicht
neu gebaut). Aus den Befunden: I-040 (Command-Bus-Widerspruch), I-041
(Wartebedingungen). Committet:

- I-004 `queue add`-Idempotenz (Retry faltet auf eine Zeile)
- I-016 Mailserver meldet Zustellergebnis (durable Outbox, fail-closed Ack)
- I-010 unversionierte Schemaänderung scheitert statt den Beweis zu überschreiben
- I-009 Desktop-Icons an der Wire-Grenze projiziert
- I-006 Datenebene startet korrekt statt sich zu reparieren
- I-015 Modul-Install schreibt die Verantwortung mit

## Bilanz Welle 2

- I-040 Command-Bus-Quittung sagt nicht mehr „angenommen" und „synchronisiert
  noch" zugleich (17 Zeilen ersetzten 62, darunter eine 12-Sekunden-Schleife)
- I-041 eine erfüllte Wartebedingung löst der Schreiber auf, der sie erfüllt hat
- I-001 Projektionsdokumente bekommen die vollständige Hülle VOR dem Schreiben,
  keine Reparatur danach — samt Substring-Vergleich auf Fehlertexten
- I-008 gleichzeitige Metadaten-Schreiber summieren ihre Bytes (`06a4b251e`)
- I-042 die agentische Queue-Reparatur ist fort (siehe unten)

## Was die zweite Welle über das Verfahren gelehrt hat

**Arbeit laufend als Patch sichern, nicht erst beim Bericht.** Zweimal traf
eine Notbremse oder eine Räumung einen Worker mitten im Lauf; beim ersten Mal
waren rund 160 Zeilen fort. Seit die Wache jede Minute `git diff` gegen
`origin/main` nach `/Volumes/tmp/…/patches/` schreibt, kostet ein Abbruch
nichts: I-008 und I-042 wurden beide ohne Bericht abgeschlossen, allein aus
dem gesicherten Patch.

**Die Systemplatte ist kein Regelwerk, das diese Pipeline durchsetzen kann.**
Der belegende Bau läuft im isolierten Worktree auf `/Volumes/tmp` mit eigenem
`CARGO_TARGET_DIR`; auf der Systemplatte bleiben nur Git-Operationen. Anders
ist neben einer fremden Sitzung mit fünfzehn parallelen Cargo-Prozessen kein
Testurteil zu bekommen.

**Löschung braucht eine Messung an der Persistenz, keine Codelektüre.** Vor
I-042 wurden vier Datenbanken über drei Wochen geprüft: null `queue.repair_*`
unter 6452 Ereignissen. Ein toter Pfad ist erst tot, wenn die Daten es sagen.

## Offen

- **`queue repair` heißt noch so, repariert aber nichts mehr** — der Pfad zählt
  offene Einträge und zeigt zwanzig davon. Ein ehrlicher Name wäre `status`
  oder `report`. Umbenennen bricht Skripte, deshalb Owner-Entscheid, nicht
  einseitig.
## Nachgemessen am 04.08.: die Reihenfolge kippt

Die Dichtetabelle oben zählt Funktionen. Nach dem `queue.rs`-Befund wurden die
beiden verbliebenen Dateien nach **unabhängigen Belangen** nachgemessen:

| Datei | Funktionen | unabhängige Belange | je Belang |
|---|---|---|---|
| `rxdb_peer.rs` | 13 | 5 | 1:2952 |
| `service.rs` | 50 | 9 | 1:2908 |

Nach Funktionen sah `service.rs` viermal so belastet aus; nach Belangen sind
beide gleichauf. Die Metrik überschätzt hier genau wie bei `queue.rs`.

**Der Fund liegt darunter:** In `service.rs` gehören **22 der 50 Funktionen zu
einem einzigen Belang** — der Business-OS-App-Recovery (Stempel, Idle-Gates,
eigene Schleife, Preflight-Marker, fünf `recover_*`-Einstiege, verteilt über
`service.rs:3912–19709`). Das ist die größte einzelne Kompensation, die im Repo
noch steht — größer als die agentische Queue-Reparatur mit 19 Funktionen.

Damit lautet die Begründung für `service.rs` nicht mehr „zu groß, eigener
Feldzug", sondern: dort steht ein Belang mit 22 Helfern, und der ist das
nächste Ziel. Die fünf Belange in `rxdb_peer.rs` — Schema-Drift (4 Funktionen),
Queue-Chat-Stempel (6), zwei Projektionsabgleiche, verwaiste Browser-Sitzungen
— sind kleiner und voneinander unabhängig, also gut parallelisierbar.

**Reihenfolge ab hier:** erst der 22-Funktionen-Belang in `service.rs` (Runde 1
lokalisiert die Ursache: warum bleiben App-Queue-Tasks überhaupt liegen?), dann
die fünf `rxdb_peer.rs`-Belange als Welle.
