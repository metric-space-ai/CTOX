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
- `runtime/build/cargo-target/debug/incremental` wuchs auf 9,1 GB. Reine
  Rebuild-Beschleunigung, gefahrlos löschbar — und bei `CARGO_INCREMENTAL=0`
  ohnehin ungenutzt. Der billigste Hebel, bevor man an `cargo clean` denkt.
- **`cargo clean` ist billiger als gedacht — gemessen, nicht geschätzt.**
  Ich hatte es als teuren letzten Ausweg behandelt („kostet einen
  vollständigen Neubau"). Tatsächlich: der Zielbaum war auf 65,4 GB
  angewachsen, der Neubau danach dauerte **2m19s und belegte 3 GB**. Fast
  alles davon war Ablagerung aus vielen Läufen, nicht Notwendigkeit.
  Wenn der Platz knapp wird, ist `cargo clean` also der richtige und
  nicht der letzte Griff. Die Scheu davor kostete hier zwei abgebrochene
  Wellen.

## Nie zwei Worker auf derselben Datei

Am 01.08. liefen zwei Sol-Instanzen gleichzeitig auf `store.rs`. Der zweite
stoppte korrekt mit „parallele Änderung ausserhalb der Whitelist" — der
Fehler lag bei mir, nicht bei ihm.

Ursache war eine **Wache, die eine alte Datei für ein neues Ergebnis hielt**:
sie prüfte nur `[ -f /tmp/<name>-report.md ]`, und dort lag noch der Report
des VORIGEN Anlaufs. Sie meldete sofort „fertig", ich las veraltete Zahlen,
schloss daraus auf einen Fehler des Workers und startete den nächsten Auftrag
— während der erste noch schrieb.

Deshalb:
- Report-Dateien VOR dem Start löschen, nicht danach.
- Die Wache muss zusätzlich prüfen, dass der Report NEUER ist als der
  Startzeitpunkt (`find <datei> -newermt ...`), nicht bloss dass er existiert.
- Vor jedem Dispatch: `ps -Ao args | grep -c "[c]laude --bare"` muss 0 sein.
- Ein Worker-Bericht über „parallele Änderungen" ist ein Alarm über die
  Orchestrierung, kein Randbefund des Workers.

**Ein voller Datenträger hinterlässt Trümmer, die nach etwas anderem
aussehen.** Nach dem Vorfall scheiterte der Build an `wha-proto`, einem
Typ, den dessen eigenes Build-Skript erzeugt — das sah aus wie ein Schaden
am gerade laufenden Refactoring. Es war ein abgeschnittenes Artefakt;
`cargo clean -p wha-proto` genügte. Wer den Fehler dem Schnitt zuschreibt,
verwirft saubere Arbeit.
