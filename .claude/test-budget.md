# Testlauf-Budget (Betriebsregel)

CTOX ist ein Hintergrund-Daemon. Im Leerlauf gehört er unter 15 % CPU;
alles darüber ist ein Befund, kein Normalzustand.

Die Testsuite ist NICHT harmlos: jeder Peer-Test startet eine eigene
Tokio-Laufzeit samt Hintergrundschleifen. Ungebremst parallel ergibt das
300+ Threads und 800 % CPU über zwanzig Minuten — das hat diese Maschine
schon einmal zum Absturz gebracht.

Deshalb gilt für JEDEN Lauf:

    cargo test --bin ctox <filter> -- --test-threads=4

**`--test-threads=4` begrenzt die CPU-Last NICHT** — gemessen am 01.08.:
ein Lauf mit genau dieser Option stand über eine Minute konstant bei
950 %. Die Option begrenzt, wie viele Testfunktionen gleichzeitig laufen,
nicht wie viele Threads jede davon startet. Vier Tests, die je eine
Tokio-Laufzeit mit Arbeitsschleifen hochziehen, sättigen die Maschine
genauso wie vierzig. Die Option bleibt richtig, weil sie den Speicher und
die Fixture-Kollisionen bändigt — als CPU-Schutz taugt sie nicht, und wer
sich auf sie verlässt, misst nicht mehr nach.

Der wirksame Hebel ist deshalb der FILTER, nicht die Thread-Zahl.

Und:
- Vor dem Start: `uptime` prüfen. Load > 30 => warten.
- Während langer Läufe regelmäßig `ps` prüfen; > 400 % CPU über mehrere
  Minuten => abbrechen und den Filter enger fassen. Ein kurzer Ausschlag
  beim Kompilieren ist normal; anhaltende Last aus `deps/ctox-<hash>`
  ist der Testlauf selbst und damit der Befund.
- Konsumenten-Baselines auf die TATSÄCHLICH berührten Module schneiden,
  nicht auf ganze Bäume. Ein Modul-Filter kostet Sekunden, `business_os::`
  kostet 22 Minuten.

## Plattenplatz — die zweite Ressource

CPU war nicht die einzige, die ich übersehen habe. Am 01.08. lief die
Datenpartition voll und JEDER Bash-Aufruf scheiterte danach, weil die
Umgebung ihre eigene Ausgabedatei nicht mehr anlegen konnte. Ursache:
neun liegengebliebene Verifikations-Checkouts (`git archive HEAD` in den
Scratch-Bereich), 6,7 GB, einer pro Welle.

Deshalb:
- `df -h /System/Volumes/Data` vor jeder Welle prüfen. Unter 10 GB frei:
  erst aufräumen, dann arbeiten.
- Ein Verifikations-Checkout wird SOFORT nach seiner Prüfung gelöscht,
  im selben Befehl. Nicht „später", nicht am Ende der Welle.
- Der Cargo-Zielbaum unter `runtime/build/cargo-target` wuchs auf 44 GB.
  Er ist wegwerfbar (`runtime/` ist ignorierter Laufzeitzustand), aber ein
  `cargo clean` kostet einen vollständigen Neubau — das ist eine
  Owner-Entscheidung, keine stille Aufräumaktion.
- **Worker-Aufträge müssen `CARGO_TARGET_DIR` explizit setzen.**
  `.cargo/config.toml` (`target-dir = "runtime/build/cargo-target"`) wird
  relativ zum ARBEITSVERZEICHNIS gelesen, nicht zum Manifest. Ein Worker,
  der `cargo` von woanders aufruft — etwa aus einem Baseline-Worktree —
  baut nach `./target` und legt damit einen zweiten vollständigen
  Artefaktbaum an. Am 01.08. waren das 13 GB, stumm, in einer einzigen
  Welle. Aufgefallen ist es erst, weil meine eigenen Läufe aus dem
  Repo-Root gar kein `target/` erzeugen.
- Worker legen für Baseline-Prüfungen Worktrees unter `/private/tmp` an.
  Ein abgebrochener Lauf lässt sie liegen: `git worktree list` prüfen und
  mit `git worktree remove --force` aufräumen.

**Ein voller Datenträger hinterlässt Trümmer, die nach etwas anderem
aussehen.** Nach dem Vorfall scheiterte der Build an `wha-proto`, einem
Typ, den dessen eigenes Build-Skript erzeugt — das sah aus wie ein Schaden
am gerade laufenden Refactoring. Es war ein abgeschnittenes Artefakt;
`cargo clean -p wha-proto` genügte. Wer den Fehler dem Schnitt zuschreibt,
verwirft saubere Arbeit.
