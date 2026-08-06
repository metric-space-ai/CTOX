# I-057 — Runde 1: Messbericht

## was_geaendert

- Keine Repository-Datei geändert; reine Messung.
- Nur `/tmp/i-057-report.md` sowie Vergleichssnapshots unter `/tmp/i-057-*` angelegt.
- Vorher-/Nachher-Vergleich von `git status --porcelain`, `git diff --stat` und `git diff --cached --stat`: jeweils **0 Differenzen**. Der vorbestehende Arbeitsbaum blieb unverändert. Aktueller vorbestehender Umfang: unstaged `34 files changed, 1824 insertions(+), 1556 deletions(-)`; staged `60 files changed, 3813 insertions(+), 9270 deletions(-)`.

## ursache_belegt

### 1. Wer schreibt die Queue-Projektion, und wo kann sie abweichen?

**Der normale Schreibpfad hat zwei Stufen:**

1. `record_command` öffnet zuerst den Business-OS-Store (`src/core/business_os/store.rs:11030`), legt dann den kanonischen Queue-Task über `create_ctox_queue_task` an (`store.rs:11041-11042`, Helper `store.rs:24735-24788`) und schreibt anschließend die Kompatibilitätszeilen `business_commands` und `ctox_queue_tasks` in `business-os.sqlite3` (`store.rs:11057-11123`).
2. Der native Projektor liest diese `business_records` seitenweise (`src/core/business_os/rxdb_peer.rs:8311-8334`) und schreibt sie per Bulk-Upsert in die RxDB-SQLite-Projektion (`rxdb_peer.rs:8350-8374`).

**Die frühere Ursache ist im Git-Bestand belegt:** Commit `d0d2d0ca8` benennt fünf kanonische Mutationsklassen — Lease, Hold, Ack, Command-Transition und Control-Completion — die den Kern änderten, ohne die Projektion zu aktualisieren. Der heutige Code hat dafür den Schreibpfad umgebaut:

- Hook-Registrierung beim Öffnen des Business-OS-Stores: `src/core/business_os/store.rs:1066-1074`.
- Attach von `business-os.sqlite3` und `business-os-rxdb.sqlite3` an die kanonische SQLite-Verbindung: `store.rs:1102-1112`.
- Aktualisierung von `business_records`, `ctox_queue_tasks`-RxDB und der verknüpften Command-Projektion: `store.rs:1298-1437`; der direkte RxDB-Upsert läuft über dieselbe angehängte Verbindung bei `store.rs:1185-1259`.
- Kanonische Mutation plus Projektionsrefresh liegen heute in derselben Transaktion, unter anderem bei:
  - Command-Admission: `src/core/mission/channels/command_saga.rs:21-30`, Refresh/Commit `:180-181`.
  - Command-/Queue-Transition: `command_saga.rs:1022-1049`.
  - Terminale Control-Completion: `command_saga.rs:820-945`.
  - Lease: `src/core/mission/channels/mod.rs:3398-3443`.
  - Hold/Wakeup: `channels/mod.rs:2485-2623`, `:2630-2676`.
  - Ack: `channels/mod.rs:5397-5423`.

**Ja, es gibt weiterhin einen nicht-atomaren Ort:** Die *erste* Projektionsanlage eines Business-OS-Commands erfolgt nach dem Commit der kanonischen Claim-Transaktion. `claim_business_command_with_queue` committet bei `command_saga.rs:180-181`; erst danach schreibt `record_command` die Business-OS-Zeilen bei `store.rs:11057-11123`. Zudem überspringt der atomare Refresh eine noch nicht vorhandene Projektionszeile ausdrücklich (`store.rs:1314-1326`). Ein Prozessabbruch in diesem Fenster kann daher eine **fehlende** Projektion erzeugen. Er erzeugt nicht den vom Abgleicher selektierten Zustand „vorhanden, aber aktiv und stale“, und `reconcile_ctox_queue_task_projections` kann fehlende Projektionen ohnehin nicht anlegen, weil es bei RxDB-Dokumenten startet (`rxdb_peer.rs:8418-8429`).

Persistenz bestätigt diesen Unterschied:

- Kanonische Queue-Inbound-Zeilen: **1099**.
- Vorhandene RxDB-Queue-Projektionen: **942**.
- Normalisiert verglichen: **938 passend**, **4 terminal/terminal abweichend**, **157 fehlende Projektionen**, **0 Projektionen ohne kanonische Zeile**.
- Von den **157** fehlenden Projektionen sind **0** mit `business_command_task_links` verknüpft; alle sind alter Bestand (letztes Datum 2026-07-18), also kein Beleg gegen den heutigen typisierten Command-Pfad.
- Die **4** Statusabweichungen sind ebenfalls Altbestand vom 2026-07-04/12/13 und liegen außerhalb des aktiven Selektors: 2× kanonisch `failed` vs. projiziert `review_rework`, 1× `cancelled` vs. `review_rework`, 1× `handled` vs. `failed`.

### 2. Hinterlässt der Abgleicher eine dauerhafte Wirkung, und feuert er heute?

**Queue-Reparatur: ja, dauerhaft.** Sie upsertet zuerst das RxDB-Dokument (`src/core/business_os/rxdb_peer.rs:8588-8590`; Storage-Upsert `:9083-9098`) und schreibt dieselben reparierten Dokumente anschließend über `store::push_collection_records` in den dauerhaften Business-OS-Store zurück (`rxdb_peer.rs:8597-8610`). Der Status wird dabei terminal und fällt dauerhaft aus dem Selektor `queued|running|accepted` (`:8527-8548`). Es gibt kein separates Audit-Event, aber eine persistente Zustandsmutation in beiden Stores.

**Chat-Reparatur: persistent in RxDB, aber ohne Business-Store-Writeback.** Sie setzt die Message-/Summary-Felder einschließlich `tracking_active=false` (`rxdb_peer.rs:8808-8828`, `:8854-8868`) und upsertet das Chat-Dokument in RxDB (`:8832-8838`). Das ist ein dauerhafter SQLite-Schreibvorgang und entfernt das Dokument aus dem Selektor `tracking_active=true`; anders als die Queue-Reparatur schreibt dieser Pfad jedoch nicht nach `business-os.sqlite3` zurück.

Heutige Zahlen aus der Persistenz:

- `ctox_queue_tasks`, Selektor `queued|running|accepted`: **0 von 942**.
- `business_records/ctox_queue_tasks`, derselbe Selektor: **0 von 942**.
- `business_chats`, Selektor `tracking_active=true`: **0 von 210**.
- Business-Store und RxDB stimmen bei den Chat-Tracking-Summaries aktuell **210/210** überein; auch im Business-Store sind **0** Chats aktiv.
- Terminaler kanonischer Command mit aktiver Queue-Route `pending|leased|running`: **0**.

Damit ist die **heutige Reparaturwirkung 0**. Der Codepfad ist nicht tot: Er wird nach einem vollständigen Projektionsdurchlauf aufgerufen (`rxdb_peer.rs:8397-8406`) und bei geändertem Fingerabdruck ausgeführt (`:9888-9901`). Eine tatsächliche Ausführung „heute“ ist in diesem Checkout aber nicht belegbar: laufende CTOX-Service-Prozesse **0**; jüngster Queue-Projektions-Write 2026-07-21 13:34:28 UTC, jüngster Chat-Write 2026-07-18 09:36:45 UTC. Die Null ist daher als **persistenter Bedarfssnapshot** aussagekräftig, nicht als Laufzeit-Zähler für den 2026-08-04.

### 3. Deckt `command_saga.rs:1106-1147` die Fälle ab?

**Allein: nein.** Der Block deckt exakt den dokumentierten Defekt ab: Ein bereits terminaler Command trifft erneut auf eine kanonische Queue-Route `leased|running`; dann wird die Route passend zu `completed|cancelled|failed` terminalisiert (`src/core/mission/channels/command_saga.rs:1106-1147`). Er deckt nicht ab:

- eine fehlende kanonische Task-Zeile bei vorhandener Projektion (`rxdb_peer.rs:8503-8518`),
- einen alten `accepted|pending|pending_sync`-Command ohne kanonischen Task (`rxdb_peer.rs:8505-8518`),
- eingebettetes Chat-Tracking (`rxdb_peer.rs:8683-8841`).

**Für reguläre, verknüpfte Queue-Statusänderungen deckt der heutige Gesamtpfad die Abgleicherfälle aber ab:** Der Saga-Block korrigiert den kanonischen terminalen Lease-Defekt; der seit `d0d2d0ca8` vorhandene Attached-Refresh schreibt Queue- und Command-Projektion in derselben Transaktion. Der Persistenzbefund `terminal command + active route = 0` und `aktive Queue-Projektion = 0` passt dazu. Für den regulären typisierten Command-/Queue-Pfad ist `reconcile_ctox_queue_task_projections` daher **ein Netz über trockenem Boden**. Seine verbleibenden Zweige sind Legacy-/Datenverlust-Fälle, für die heute **0 Kandidaten** existieren.

Für `reconcile_business_chat_tracking_projections` gilt das nicht: Der atomare Queue-Hook aktualisiert Queue und Command (`store.rs:1298-1437`), nicht das eingebettete Chat-Dokument. Queue-Chat-Antworten werden separat geschrieben (`store.rs:14186-14252`), und der Browser synchronisiert Tracking asynchron aus Command/Task-Dokumenten (`src/apps/business-os/shared/business-chat.js:2668-2781`) und persistiert später (`:3295-3347`). Bei geschlossenem/offline Browser sowie bei Failure/Cancel/Orphan bleibt damit ein realer nicht-atomarer Pfad. Diese Kompensation ist noch nicht ursächlich ersetzt.

## kompensationen_geloescht

- **0**. Reine Messung; keine Datei geändert.

## verblieben

- `reconcile_ctox_queue_task_projections` (`src/core/business_os/rxdb_peer.rs:8411-8625`): für den regulären typisierten Schreibpfad nicht mehr erforderlich; heutige Kandidaten **0**. Nur Legacy-/Korruptionszweige bleiben semantisch übrig, ohne aktuellen Treffer.
- `reconcile_business_chat_tracking_projections` (`src/core/business_os/rxdb_peer.rs:8683-8841`): **weiter erforderlich**, weil Chat-Tracking nicht Teil derselben kanonischen Transaktion ist. Heutige Kandidaten zwar **0**, aber der Schreibpfad bleibt asynchron/nicht-atomar.
- Sparschalter `rxdb_peer.rs:9888-10087`: Optimierung, kein eigener Belang; solange Chat-Reconciliation bleibt, kann er höchstens auf Chat-Quellen reduziert werden.

## tests

Alle Cargo-Aufrufe verwendeten `CARGO_TARGET_DIR=/Volumes/tmp/ctox-pipeline-targets/I-057`.

- `cargo fmt --check` — **ok**, keine Ausgabe; kein Testlauf, daher keine `test result`-Zeile.
- `cargo check --bin ctox` — **ok**; `Finished dev profile ... in 19m 04s`, 405 Warnungen; kein Testlauf, daher keine `test result`-Zeile.
- Fehlversuch `cargo test --lib canonical_queue_ack_refreshes_queue_and_command_without_repair -- --nocapture` — Exit 101: `no library targets found in package ctox`; keine Tests gestartet, keine `test result`-Zeile. Danach auf den vorhandenen Bin-Testtarget korrigiert.
- Filter `canonical_queue_ack_refreshes_queue_and_command_without_repair` (zulässig, 1 Treffer): `test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 2748 filtered out; finished in 0.22s`.
- Filter `terminal_command_orphaned_queue_lease_settles_to_terminal_route` (zulässig, 1 Treffer): `test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 2748 filtered out; finished in 0.07s`.
- Filter `reconcile_business_chat_tracking_projections_fails_orphaned_messages` (zulässig, 1 Treffer): `test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 2748 filtered out; finished in 86.42s`.
- Filter `reconcile_ctox_queue_task_projections_completes_stale_completed_commands` (zulässig, 1 Treffer): `test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 2748 filtered out; finished in 83.98s`.

## gegenprobe

- Entfällt laut Auftrag (reine Messung).
- Keine temporäre Codeänderung vorgenommen, daher nichts zurückzubauen.
- Vorher-/Nachher-Vergleich der drei Git-Snapshots ergab jeweils **0 Differenzen**.

## offene_bedenken

- Der Produktionsbestand kann den Fix vom 2026-08-02 nicht zeitlich gegenprüfen: jüngste Queue-/Chat-Projektionswrites sind vom 2026-07-21 bzw. 2026-07-18; laufender CTOX-Service **0**. Der Ursachennachweis für den neuen Pfad kommt daher aus Quellstruktur, Git-Historie `d0d2d0ca8` und den gezielten Regressionstests, nicht aus Post-Fix-Live-Traffic.
- Die erste Command-/Task-Projektionsanlage bleibt nicht atomar (`command_saga.rs:180-181` vor `store.rs:11057-11123`). Gemessener Altbestand: **157** kanonische Queue-Zeilen ohne RxDB-Projektion. Der heutige Abgleicher kann diese Form nicht reparieren.
- Es existieren **4** terminale Zeile-für-Zeile-Abweichungen, die der aktive Selektor absichtlich nicht sieht. Sie sind Altbestand und kein Beleg, dass der aktuelle aktive Pfad driftet, zeigen aber: „0 aktive Kandidaten“ bedeutet nicht „vollständige historische Gleichheit“.
- Chat-Reparatur schreibt nur RxDB (`rxdb_peer.rs:8832-8838`), nicht `business-os.sqlite3`. Aktuell stimmen beide Stores **210/210** überein; bei einem späteren Full-Replay aus einer stale Business-Store-Zeile könnte die RxDB-Reparatur jedoch erneut nötig werden.

## pfade

### Nächste Welle: Queue-Netz entfernen/narrowen

- `src/core/business_os/rxdb_peer.rs:8411-8673` — Queue-Reconciler plus nur dafür benötigte Status-/Orphan-Helper entfernen, sofern keine andere Nutzung verbleibt.
- `src/core/business_os/rxdb_peer.rs:9888-10097` — Fingerabdruck/Gate von Queue+Chat auf den verbleibenden Chat-Bedarf reduzieren.
- `src/core/business_os/rxdb_peer.rs:20510-21055` — Queue-Reconciler- und Gate-Tests entfernen/auf den verbleibenden Chat-Belag zuschneiden.

### Ursache des verbliebenen Chat-Netzes beseitigen

- `src/core/business_os/store.rs:1298-1437` — Attached-Refresh um die verknüpfte `business_chats`-Tracking-Zeile erweitern, sodass Queue/Command/Chat in derselben kanonischen Transaktion aktualisiert werden.
- `src/core/business_os/store_projections.rs:133-250,342-476` — vorhandene Chat-Tracking-Payload-/Summary-Logik als serverautoritative Schreiblogik wiederverwenden bzw. transaktionsfähig machen.
- `src/core/business_os/store.rs:13754-13839,14186-14252` — derzeitige getrennte Terminal-Reihenfolge für Queue-Chat konsolidieren; Failure/Cancel/Blocked ebenso abdecken, nicht nur erfolgreiche Antworttexte.
- `src/apps/business-os/shared/business-chat.js:2668-2781,3295-3347` — Browsertracking danach als UX-Mirror/Fallback belassen, nicht als notwendige Quelle der terminalen Persistenz. Keine Änderung unter `src/apps/business-os/rxdb/src/` erforderlich, solange nur dieser Shared-Helper geändert wird.

### Separater, vom aktuellen Reconciler nicht gedeckter Admission-Spalt

- `src/core/mission/channels/command_saga.rs:21-185`
- `src/core/business_os/store.rs:11030-11123,24735-24788`

Diese Stellen wären nötig, falls auch die erste Projektionsanlage atomar werden soll; das ist ein eigener Belang, weil der heutige Reconciler fehlende Dokumente nicht erzeugt.
