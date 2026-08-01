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
