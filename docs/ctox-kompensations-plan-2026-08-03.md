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

**`cargo check --bin ctox` übersetzt keine Tests.** Bei meinem Endschliff zu
I-050 habe ich eine Signatur geändert, `cargo check` meldete Exit 0, und der
Testbau brach an drei Stellen mit `E0061`. Meine Regex erwartete `&variable` als
erstes Argument; drei Teststellen übergaben `prompt` ohne `&`. Bei jeder
Signaturänderung gehört `--tests` dazu — sonst ist „grün" nur die halbe
Übersetzungseinheit. Und: was ein regulärer Ausdruck nicht sieht, sieht der
Compiler; die Arbeitsteilung sollte man nutzen, statt dem Muster zu vertrauen.

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

## I-054: der Schema-Sweep ist nicht die Kompensation — der fehlende Migrator ist es

Meine Prämisse war zweimal falsch, und die Messung hat beides umgeworfen.

**Erstens:** Ich schrieb, ein Schema-Bump lasse die alte Tabelle stehen und der
Startup-Sweep räume sie weg. Der Sweep iteriert aber ausschließlich die
statische Liste aus `business_os_schema_contract.json`
(`rxdb_peer.rs:14224-14245`). Die vier auffälligen `sellify_*`-Sammlungen sind
Runtime-Module und stehen dort nicht — **er sieht sie gar nicht an.** Und das
ist richtig so: täte er es, würde der Guard die v1-Tabellen mit 238.913 Zeilen
für Müll halten.

**Zweitens:** Die v0-Tabellen sind nicht Reste eines Upgrades, sondern **leere
Trümmer eines Rückschritts**. Belegt durch die SQLite-Anlagereihenfolge
(`rowid` v1 = 711–717, v0 = 719–725, also v0 SPÄTER angelegt) und durch die
Zeilenzahlen: v0 = 0, v1 = 238.913. Das Log zeigt eine echte, ausgelöste
Migration mit anschließendem Cleanup — die war legitim. Danach deklarieren die
installierten Sellify-Dateien dieselben Sammlungen wieder als v0.

**Die eigentliche Ursache ist architektonisch: es gibt keinen Migrator.**
Ein Schema-Bump ist heute nur „neues versionsabhängiges Metadokument plus neue
leere Tabelle". Kein Copy, kein Verify, kein Drop. Das native Migrations-Plugin
ist ein ausdrücklicher Stub (`migration_schema/mod.rs:1-5`), `migration_needed`
und `start_migration` liefern `PLUGIN_MISSING` (`rx_collection.rs:1013-1028`).

**Und die Browserseite behauptet einen Vertrag, den sie nicht erfüllt.**
`app.js:5201-5253` hängt `migrationStrategies` an die Collection-Definition, ein
Guard prüft, dass sie *kompilieren* — aber `addCollections` liest das Feld nie
(`rx-database.mjs:124-145`). Das ist die schlimmste Sorte Kompensation: eine
Zusicherung, die erfüllt aussieht und es nicht ist. Ein Guard, der nur die
Übersetzbarkeit prüft, verstärkt die Täuschung, statt sie aufzudecken.

Die alte Tabelle bleibt also nicht wegen einer Rückwärtsmigrations-Entscheidung
stehen, sondern weil der ausführende Schritt fehlt.

## `rxdb_peer.rs` fertig vermessen: aus fünf Belangen wurden drei, davon einer legitim

Nachzählung am 04.08., je Belang selbst gemessen:

1. **Schema-Sweep** → I-054, erledigt als Befund. Die Ursache ist der fehlende
   Migrator (eigene Kampagne); der Sweep selbst schützt derzeit sogar korrekt
   die 238.913 v1-Zeilen davor, als Müll zu gelten.
2. **Projektionsabgleicher** (Queue-Tasks + Chat-Tracking) → I-057, Runde 1
   gestellt. Der davorsitzende „Queue-Chat-Stempel" (6 Funktionen) ist **keine
   Kompensation, sondern ein Sparschalter** — er bildet einen Fingerabdruck des
   Stores und überspringt den Abgleich, wenn sich nichts geändert hat. Fünf der
   sechs Funktionen existieren nur, damit dieser Schalter billig ist. Die
   Dichtemetrik hatte sie als eigenen Belang gezählt; das war ihr dritter
   Überschätzungsfall.
3. **Verwaiste Browser-Sitzungen** (`recover_stale_browser_sessions`, `:6989`)
   → **legitim, kein Auftrag.** Gemessen: null aktive Sitzungen im Bestand
   (13 `disconnected`, 1 `failed`), der geordnete Teardown schreibt
   `disconnected` selbst (`:10266`). Die Recovery greift nur, wenn eine Sitzung
   `active` behauptet und der Laufzeit-Manager sie nicht kennt — der harte
   Crash. Für den kann kein Schreibpfad den Endzustand garantieren; genau der
   Fall, für den der Plan das Netz ausdrücklich stehen lässt. (Dasselbe Urteil
   fiel schon am 31.07., als eine falsch gefilterte Zählung „116 Geister"
   behauptete — die Funktion arbeitete die ganze Zeit korrekt.)

## I-057: die Projektionsabgleicher — eine Hälfte trocken, eine nass

Die Messung teilt den Belang exakt an der Ursachengrenze:

**Queue-Hälfte: Netz über trockenem Boden.** Die Ursache — Kern-Mutationen
ohne Projektions-Update — fiel mit `d0d2d0ca8`: seither laufen kanonische
Mutation und Projektions-Refresh in DERSELBEN Transaktion, über alle fünf
Mutationsklassen (Lease, Hold, Ack, Command-Transition, terminale Completion).
Heute: null Kandidaten in beiden Stores (0 von 942), null terminale Commands
mit aktiver Route. Und die Null ist belastbar, weil eine Reparatur dauerhafte
Wirkung hätte — sie schreibt in RxDB UND zurück in den Business-Store. Die 157
fehlenden Projektionen sind Altbestand (letzte 18.07.) und ohnehin außerhalb
der Reichweite des Abgleichers: er startet bei RxDB-Dokumenten und kann
Fehlendes prinzipiell nicht anlegen. → I-058 löscht diese Hälfte.

**Chat-Hälfte: Ursache existiert noch, Netz bleibt.** Der Browser schreibt
Chat-Tracking asynchron (`business-chat.js:2668-2781`); bei geschlossenem
Browser oder Failure/Cancel bleibt ein realer nicht-atomarer Pfad. Diese
Kompensation fällt erst, wenn das Tracking transaktional mitgeschrieben wird.

**Notiert, kein Auftrag:** Ein theoretisches Fenster bei der ERSTEN
Projektionsanlage — der Claim committet (`command_saga.rs:180-181`), erst
danach schreibt `record_command` die Projektionszeilen (`store.rs:11057`). Ein
Crash dazwischen erzeugt eine fehlende (nicht eine stale) Projektion. Seit dem
18.07. kein einziger Fall; wie bei I-055 gilt: nichts auf Vorrat bauen.

## I-056: die Dokumentation sichert einen Lebenszyklus zu, den es nicht gibt

Der schwerste Fund dieser Kampagne, gefunden von I-056 und von mir am Code
nachgeprüft.

`docs/ctox-rxdb.md:289-295` schreibt: „Runtime-installed schema migrations are
native too … It executes the supported declarative operations, verifies every
source envelope in the target version, and only then permits stale-table
cleanup." Die Tabelle bei `:706-707` nennt die durchsetzenden Funktionen und
die Wächtertests dazu:

| in der Doku genannt | existiert im Repo |
|---|---|
| `migrate_additive_native_rxdb_collection_versions` | **nein** |
| `native_rxdb_additive_migrations` | **nein** |
| `additive_thread_schema_migration_copies_and_verifies_before_cleanup` | **nein** |
| `runtime_installed_declarative_migration_is_discovered_and_copied` | **nein** |
| `runtime_migration_without_strategy_retains_old_table_and_fails_closed` | **nein** |
| `native_declarative_migration_matches_browser_operations` | ja |

Der eine existierende Wächter prüft `apply_native_declarative_migration`
isoliert — und diese Funktion hat **genau einen Aufrufer: ihren eigenen Test**
(`rxdb_peer.rs:16744`). In Produktion läuft sie nie. Der Wächter erzeugt also
genau die Sicherheit, die er widerlegen sollte.

**Warum das schlimmer ist als fehlender Code:** Wer diese Doku liest, hält
Datenerhalt beim Schema-Wechsel für zugesichert und geprüft. Beides ist falsch.
Eine Zusicherung, die niemand einlöst, ist gefährlicher als eine fehlende — sie
verhindert, dass jemand nachsieht.

**Nicht zu tun: die Doku an den Code angleichen.** Sie beschreibt die richtige
Absicht; sie zu streichen löschte die Anforderung. Richtig ist, die
Nicht-Umsetzung ausdrücklich zu kennzeichnen, damit niemand sich darauf
verlässt, und den Lebenszyklus als eigene Kampagne zu bauen. Sol hat genau das
vorgeschlagen und ausdrücklich abgelehnt, Variante (a) halb zu bauen — die
Browser-Storage hat keinen persistenten Versionsmarker und keine Transaktion,
die migrierte Zeilen und Versionsumschaltung atomar schreibt.

Nebenbei gemessen: drei Module deklarieren Strategien widersprüchlich zwischen
`schema.js` und `collections.schema.json` (`browser`, `credentials`, `creator`),
und der App-Starter erzeugt beide Seiten uneinheitlich.

## Schlussbilanz (05.08.2026)

Die Kompensationsliste dieser Kampagne ist abgearbeitet: **24 Aufträge, davon
23 abgeschlossen, 27 Commits** seit Kampagnenstart. Der letzte Schnitt
(I-058) entfernte 933 Zeilen Queue-Projektionsreparatur, deren Ursache seit
`d0d2d0ca8` nicht mehr existiert.

**Was gefallen ist (Ursache zuerst, dann das Netz):** die agentische
Queue-Reparatur samt Schichtverletzung zur Modellausführung (I-042, 809
Zeilen); die Prosa-Erkennung der Task-Identität über 27 Aufrufstellen (I-050);
der verschluckte Queue-Ack (I-052); der Queue-Projektionsabgleicher (I-058,
933 Zeilen); dazu die Welle-1- und Welle-2-Landungen.

**Was bewusst steht:** die Chat-Tracking-Reparatur (Ursache existiert: der
Browser schreibt asynchron); die Browser-Sitzungs-Recovery (harter Crash);
der TTL-Sweep und die App-Recovery (bis `queue.ack_failed` über Zeit null
zeigt — I-053 bleibt blockiert, das ist Absicht, kein Rest).

**Die Zahl, die die Kampagne rechtfertigt:** Von 986 namentlichen
Kompensations-Verdachtsfällen waren 73 echte Schuld (7,4 %) — und von den
groß vermessenen Belangen fiel etwa die Hälfte erst nach einer Messung, die
die Prämisse kippte. Vier Befunde-ohne-Diff (I-051, I-054, I-055, I-057)
haben mehr Schaden verhindert als die Commits behoben haben: einen Nachbau
eines existierenden TTL, eine Löschung auf Basis einer Null, die nur einen
Ringpuffer bewies, Vorratsfelder ohne einen einzigen realen Fall, und eine
Reparatur an einem Migrator, den es gar nicht gibt.

**Folgekampagnen, beide Owner-Entscheid, beide zu groß für eine Welle:**

1. **Der Migrations-Lebenszyklus** (aus I-054/I-056): Schema-Bumps kopieren,
   verifizieren und räumen heute nichts; die Doku sichert es zu, fünf genannte
   Wächter existieren nicht; `migrationStrategies` werden angenommen und nie
   gelesen. Betroffen: Browser-Storage, nativer Peer, beide Deklarationsseiten,
   App-Starter, drei widersprüchliche Module.
2. **Die Pro-Key-Ack-Reihenfolge** (aus I-052): Ownership fällt heute pauschal
   vor den Acks; erst mit Ack-Erfolg pro Key kann die App-Recovery
   (22 Funktionen) endgültig fallen.

**Offener Owner-Entscheid:** `ctox queue repair` heißt noch so, zählt aber nur
noch offene Einträge — ehrlich wäre `status`; ein Umbenennen bricht Skripte.

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

## Was Runde 1 dazu ergeben hat (I-051, 04.08.)

**Eine Doku-Behauptung ist keine Messung.** `docs/ctox-harness-review-2026-07-10.md`
schreibt, es gebe kein Lease-TTL und keinen owner-agnostischen Reclaim. Für den
heutigen Code ist das falsch, am Code gegengeprüft: TTL 15 Minuten
(`channels/mod.rs:3394`), owner-agnostischer Reclaim (`channels/mod.rs:3494` —
nimmt `_lease_owner` und ignoriert ihn), dazu 60-Sekunden-Sweep und
Boot-Reclaim. Der Auftrag verlangte ausdrücklich Nachmessen statt Übernehmen;
ohne diese Klausel wäre ein Auftrag entstanden, der einen Mechanismus baut, den
es seit Juli gibt.

**Die Ursache ist ein verschluckter Ack.**
`record_queue_ack_and_refresh_business_os_projections_locked`
(`service.rs:24587`) behandelt einen fehlgeschlagenen Ack mit
`push_event_locked(...)` und `return`. Dieser Puffer hält **24 Einträge im
Prozessspeicher** und stirbt mit dem Prozess. Der Code gibt es selbst zu:
„Record *swallowed* lease-ack failures … only makes a *stuck lease*
diagnosable." Die In-Memory-Ownership fällt zu dem Zeitpunkt bereits, die Zeile
bleibt `leased`, und nichts davon ist dauerhaft belegt.

**Damit ist auch die Messmethode dieses Plans zu korrigieren.** „Null Ereignisse
in der Persistenz" beweist nur dann Totsein, wenn der Pfad überhaupt dauerhaft
schreibt. Hier tut er es nicht — null heißt „wird nicht geschrieben", nicht
„passiert nicht". Vor jeder Löschung ist deshalb zusätzlich zu prüfen, ob der
fragliche Pfad seine Wirkung überhaupt persistiert. Ein dauerhafter Nebenbeleg
fand sich trotzdem: genau ein Marker `...during idle recovery...` vom 23.06. —
vor dem TTL (10.07.) und vor dem Orphaned-Lease-Fix (23.07.).

**Folge für die Reihenfolge:** Der 22-Funktionen-Belang fällt nicht als
Nächstes. Zuerst I-052 (der Ack wird dauerhaft festgehalten), danach ist die
Frage überhaupt erst messbar — I-053 steht bis dahin auf `blockiert`. Offen und
ungemessen bleibt, ob die Validierung der App-Recovery echten Schaden verhindert
oder nur wiederholt, was ein neuer Lauf ohnehin tut.
