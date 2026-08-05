# SYNC-E: der Migrations-Lebenszyklus, den die Doku schon zusichert

Nachfolger von SYNC-D (`ctox-kompensations-plan-2026-08-03.md`, abgeschlossen:
23/24 Aufträge, 28 Commits). Diese Kampagne baut die eine Sache, die SYNC-D
gefunden und ausdrücklich NICHT halb gebaut hat.

## Der Befund (I-054/I-056, vollständig vermessen)

Ein Schema-Bump ist heute nur „neues versionsabhängiges Metadokument plus neue
leere Tabelle". Kein Copy, kein Verify, kein Fail-closed. Das native
Migrations-Plugin ist ein Stub; `apply_native_declarative_migration` existiert,
hat aber genau einen Aufrufer — seinen eigenen Test. Der Browser nimmt
`migrationStrategies` an, ein Wächter prüft, dass sie kompilieren, und
`addCollections` liest sie nie. `docs/ctox-rxdb.md` sichert den Lebenszyklus zu
und nennt fünf Wächter, von denen keiner existiert — die Stellen sind seit
04.08. als NOT IMPLEMENTED gekennzeichnet.

Real passiert ist der Schaden schon einmal in Gutform: die Sellify-Migration
(238.913 Zeilen) lief über einen kurzlebigen, ausgelösten Einmalpfad; danach
deklarierten die installierten Dateien wieder v0 und hinterließen leere
v0-Trümmer.

## Die Richtungsentscheidung (05.08., Owner-delegiert)

**Der kanonische Store migriert; der Browser zieht nach.**

1. **Nativ (autoritativ):** Beim Bring-up, NACH der Collection-Registrierung
   und VOR jedem Aufräumen: alte Versionszeilen durch die JSON-Strategien
   schicken (`apply_native_declarative_migration` — der Executor existiert und
   ist gegen die Browser-Semantik getestet), jede Quellzeile im Ziel
   verifizieren, erst dann aufräumen. Alte Zeilen ohne Strategie: Bring-up
   scheitert fail-closed. Leere Alt-Tabellen bleiben tolerabel und werden
   geräumt.
2. **Browser (Kopie):** KEINE zweite Migrations-Maschine. Nach einem
   Versionswechsel gilt die lokale Kopie als Cache: verwerfen und über die
   Replikation neu ziehen. Die `migrationStrategies`-Annahme im Browser-Runtime
   wird entfernt oder ausdrücklich als nur-deklarativ (fürs native Ausführen)
   markiert. VORHER zu messen: was passiert mit ungesyncten lokalen
   Schreibvorgängen — gibt es sie zum Zeitpunkt eines Versionswechsels, und
   woran erkennt man sie? Ohne diese Antwort wird nichts verworfen.
3. **Deklarationen:** `collections.schema.json` ist kanonisch (der native Peer
   liest nur JSON). `schema.js` muss spiegeln. Der App-Starter erzeugt beide
   Seiten konsistent; die drei widersprüchlichen Module (`browser`,
   `credentials`, `creator`) werden angeglichen.
4. **Abschluss:** Erst wenn die drei genannten Wächter aus der Doku existieren
   und rot-bewiesen sind, fallen die NOT-IMPLEMENTED-Markierungen. Die Doku
   wird an keinem Punkt „an den Code angeglichen" — der Code wird an die Doku
   angeglichen.

## Warum nicht die Browser-Vollmigration

Sols I-056-Analyse: die Browser-Storage hat weder einen persistenten
Collection-Versionsmarker noch eine Transaktion, die migrierte Zeilen und
Versionsumschaltung atomar schreibt. Eine Strategie nach einem Crash erneut
auszuführen wäre bei beliebigen Funktionsstrategien nicht sicher. Der
Cache-Ansatz umgeht das Problem, statt es zweimal zu lösen — vorausgesetzt,
die Messung zu ungesyncten Schreibvorgängen trägt ihn.

## Aufträge

- **I-060** — nativer Lebenszyklus: registrieren → migrieren (copy+verify) →
  aufräumen; fail-closed ohne Strategie. Die Wächter tragen die Namen aus der
  Doku, damit die Doku wahr wird.
- **I-061** — RUNDE 1, nur messen: trägt der Cache-Ansatz im Browser?
  Ungesyncte lokale Schreibvorgänge zum Zeitpunkt eines Versionswechsels.
- **I-062** — Deklarationen: JSON kanonisch, `schema.js` spiegelt, Starter
  konsistent, drei Module angeglichen. Wächter, der Spiegelgleichheit prüft.

Verfahren wie in SYNC-D: eigene Worktrees bei kollisionsgefährdeten Dateien,
Patches laufend sichern, Gegenprobe selbst führen, Rot-Mengen als Menge in
beide Richtungen, `--tests` nach jeder Signaturänderung.

## Bilanz Welle 1 (05.08.)

- **I-060 gelandet** (`c46499358`, +698/−27): der Lebenszyklus existiert. Die
  drei Wächter tragen die Namen aus der Doku; ein vierter deckt leere
  Alt-Tabellen. Gegenprobe je Wächter rot-bewiesen, Fail-closed vom
  Orchestrator unabhängig wiederholt. Die NOT-IMPLEMENTED-Markierungen in
  `ctox-rxdb.md` sind gefallen; der eine in der Doku genannte, nie gebaute
  Funktionsname (`native_rxdb_additive_migrations`) ist durch den realen
  ersetzt.
- **I-062 gelandet** (`172e8059d`): Deklarationen deckungsgleich, Wächter
  prüft Spiegelgleichheit über 34 Module plus Starter. Achtung: `d70d124ae`
  davor ist ein versehentlich leerer Commit mit der vollen Botschaft — der
  Inhalt liegt in `172e8059d`. Ursache: unquotierte zsh-Variable in der
  Stage-Schleife; seitdem ist ein nicht-leerer Index-Diff Pflichtprüfung vor
  jedem `commit-tree`.
- **I-061 Befund**: der Cache-Ansatz trägt. Zwei persistente Signale für
  ungesyncte Writes (`pushable` je Zeile, Recovery-WAL `pending`); es fehlen
  persistenter Versionsmarker, collection-genauer Clear unter Serialisierung
  und der explizite Invalidationspfad. Gemessen: v0-Zeilen bleiben beim
  Versionswechsel kommentarlos liegen (2/2, `openError=null`). Runde 2 ist
  bis auf die Zeile kartiert, inklusive Sidecar-Invalidierung und sechs
  nötiger Browser-Smokes.

**Offen:** I-061 Runde 2 (Browser-Invalidationspfad — Sol, eigener Worktree,
dist-Neubau + Cache-Buster Pflicht) und danach die sechs Browser-Smokes.

## Schlussbilanz (05.08.2026, abends)

SYNC-E ist abgeschlossen. Beide Seiten des Migrationsvertrags existieren und
sind bewacht:

- **I-060** (`c46499358`): der native Lebenszyklus — registrieren, migrieren,
  verifizieren, erst dann aufräumen; fail-closed ohne Strategie. Die Wächter
  tragen die Namen aus der Doku.
- **I-063** (`24e4f9dc8`): der Browser invalidiert als Replica — persistenter
  Zweiphasen-Marker (Version + effektiver Hash), Fail-closed auf beiden
  Unsynced-Signalen, `pushable`-Zählung INNERHALB der destruktiven Transaktion
  (TOCTOU geschlossen), Web-Lock, typisierte Fehler auch ohne Web-Locks-API.
- **I-062** (`172e8059d`): Deklarationen deckungsgleich über 34 Module plus
  Starter; der Wächter prüft Spiegelgleichheit statt Kompilierbarkeit.
- **Doku** (`8efb91448`): die NOT-IMPLEMENTED-Markierungen sind gefallen — die
  Zusicherung ist wieder wahr, mit historischer Notiz zum Zeitraum der Lücke.
- **Abdeckung** (`fd2ccf738`): sechs Browser-Smokes, 67 Assertions gesamt,
  jede Flanke rot-fähig bewiesen (WAL-pending, Multi-Tab-TOCTOU,
  Sidecar-Fenster, Full-Pull, Reset/WAL).

**Die Grok-Spur** (Standard-Coding-Worker `claude-grok`, in dieser Kampagne
etabliert): vier Aufträge, vier verwertbare Ergebnisse — zwei Zähler-Fixes
(`22cf6406c`, `163fb78e4`, `74f5dc5c3` als Abschluss), eine Musteranalyse,
ein korrektes Teil-Nein, das den Folgeauftrag präzise anforderte. Eine
dokumentierte Schwäche: Gateway-Ausfälle töten den Worker stumm nach getaner
Arbeit — die Wache erkennt sie seither im Log, und die Patch-Sicherung macht
den Verlust zu null.

**Bewusst offen, mit Bedingung:** I-053 (App-Recovery-Löschung) wartet auf
`queue.ack_failed`-Messdaten über Zeit; die Pro-Key-Ack-Reihenfolge ist nur
nötig, falls die Messung sie rechtfertigt. Sonst ist nichts offen.
