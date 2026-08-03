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

## Offen

- **I-040, I-041** — richtig geschnitten aus Welle-1-Befunden, bereit.
- **I-008** — zurück: Fix korrekt, aber `dist/` nicht neu gebaut, Cache-Buster
  nicht gebumpt (`docs/ctox-rxdb.md`). Neu stellen mit Testort.
- **queue.rs / rxdb_peer.rs / service.rs** — nach Dichte, wie oben.
