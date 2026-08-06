# I-061 — RUNDE 1 (nur messen)

## was_geaendert

- Keine Repository-Datei geaendert; keine Kompensation geloescht; kein Commit.
- Nur Messskripte unter `/tmp/i061-*.mjs`, ein sauberer HEAD-Export unter `/tmp/i061-head/` und dieser Bericht unter `/tmp/i-061-report.md` angelegt.
- Der abschliessende Arbeitsbaum-Diff bleibt der bereits geteilte Fremd-Diff: `git diff --stat` meldete 37 unstaged Dateien mit 2118 Einfuegungen/1578 Loeschungen; `git diff --cached --stat` 60 staged Dateien mit 4741 Einfuegungen/9438 Loeschungen. Keine dieser Aenderungen stammt aus I-061.

## ursache_belegt

### 1. Woran erkennt der Browser unreplizierte lokale Schreibvorgaenge?

**Es gibt zwei persistente, konservative Signale; Checkpoints und `assumedMasterState` sind nicht das Dirty-Signal.**

1. **Je primaerer Dokumentzeile: `pushable`.** Jede gespeicherte Zeile liegt im einen IndexedDB-Object-Store `documents` und erhaelt `replicationOriginRole` sowie `pushable`; eine Master-/CTOX-Origin-Zeile wird `pushable:0`, eine lokale Browser-Zeile `pushable:1` (`src/apps/business-os/rxdb/src/storage-indexeddb.mjs:1289-1303`). Der Upgradepfad rekonstruiert diese Flags aus `_meta.ctoxReplicationOrigin` (`storage-indexeddb.mjs:1314-1338`). Der Push-Scan liest fuer den CTOX-Peer direkt den Index `[collection,pushable,lwt,id]` und damit nur `pushable=1` (`storage-indexeddb.mjs:987-1021`, Indexanlage `storage-indexeddb.mjs:1181-1184`). Intern ist das **je Zeile** ueber `getStoredRecord(id)` abfragbar (`storage-indexeddb.mjs:190-199`); die oeffentliche DB-Fassade liefert derzeit nur exakte Counts insgesamt und je Collection (`storage-indexeddb.mjs:1238-1255`, `rx-database.mjs:168-170`), keine ID-Liste.
2. **Separates Recovery-WAL: Zustand `pending`.** Vor einer lokalen Primaerschreiboperation wird ein Batch mit Collection, Schema-Hash, Dokument-IDs und `state:'pending'` in `${databaseName}__recovery_v2` geschrieben (`recovery-journal.mjs:25-28`, `recovery-journal.mjs:45-70`; Aufruf vor dem Primaerwrite: `storage-indexeddb.mjs:275-306` und `storage-indexeddb.mjs:423-456`). Er wird erst `master_acked`, wenn eine vom Master materialisierte Zeile den lokalen Inhalt per gleicher HLC bzw. in Mixed-Version identischem Inhalt bestaetigt (`recovery-journal.mjs:82-101`, `recovery-journal.mjs:553-563`). Status und Count sind persistent (`recovery-journal.mjs:260-278`, LocalStorage-Spiegel `recovery-journal.mjs:408-416`). Das WAL schliesst das Crashfenster **vor** dem Primaerwrite; `pushable` schliesst vorhandene Primaerzeilen ein. Fuer einen destruktiven Wechsel muessen deshalb beide Signale geprueft werden.

**Nicht geeignet:**

- Pull-/Push-Checkpoints sind nur Cursor. Nach erfolgreichem `masterWrite` wird der Push-Checkpoint vorgerueckt (`replication-webrtc.mjs:1430-1455`), waehrend die Primaerzeile/WAL erst durch den spaeteren Master-Roundtrip origin-gestempelt bzw. bestaetigt wird. Ein Checkpoint beweist daher nicht „keine lokalen Writes“.
- `assumedMasterState` wird beim Push zunaechst `null` gesetzt und nur aus vom Master gelieferten Konfliktzeilen gefuellt (`replication-webrtc.mjs:1384-1418`); es ist Konfliktbasis, kein Dirty-Flag.
- Der Demand-Sidecar-Dirty-Status (`replication-webrtc.mjs:1332-1348`) ist Cache-/Eviction-Schutz, nicht die kanonische Persistenzantwort.

**Store-Messung:** Ein Browserlauf schrieb in `widgets` eine Master-Origin-Zeile und eine lokale Zeile. Vor und nach Wiedereroeffnung mit deklarierter Schema-Version 1 meldete der Primaerstore exakt `docs=2`, `unsynced.total=1`, `unsynced.byCollection.widgets=1`; das Recovery-WAL meldete `pendingBatches=1`, `pendingWrites=1`. Damit ist die lokale, nicht bestaetigte Zeile sowohl je Primaerrecord (`pushable`) als auch im WAL identifizierbar.

### 2. Erkennt der Browser beim Collection-Open eine aeltere persistierte Collection-Version?

**Nein. Es gibt keinen persistenten Collection-Versionsmarker im Primaerstore.**

- `DB_VERSION = 3` und `indexedDB.open(databaseName, DB_VERSION)` versionieren nur das technische IndexedDB-Layout (`storage-indexeddb.mjs:15-18`, `storage-indexeddb.mjs:1151-1185`).
- `addCollections()` uebernimmt die aktuelle Deklaration und baut ein Collection-Objekt; es liest/vergleicht keine persistierte Schema-Version und keine `migrationStrategies` (`rx-database.mjs:124-161`). `schema.version` lebt nur am Laufzeitobjekt (`rx-database.mjs:194-202`).
- Primaerrecords tragen Collection, ID, lwt, Origin-/Pushable-Flags, Indexwerte und Dokument, aber weder Schema-Version noch Schema-Hash (`storage-indexeddb.mjs:1289-1303`). `schemaIndexSignature` dient nur dem Neuaufbau von Performance-Indizes (`storage-indexeddb.mjs:1118-1147`), nicht der Collection-Versionserkennung.
- Das Recovery-WAL speichert zwar den Schema-Hash **pro lokalem Batch** (`recovery-journal.mjs:45-66`) und kann einen noch nicht primaer-committeten Replay-Batch bei Hashabweichung in Konflikt setzen (`recovery-journal.mjs:133-179`). Es ist aber kein Versionsmarker der persistierten Collection. Besonders wichtig: Batches mit `primaryCommittedAtMs>0` werden vor dem Schema-Hash-Vergleich uebersprungen (`recovery-journal.mjs:133-144`).
- Die WebRTC-Aushandlung vergleicht nur die **aktuell deklarierten** Browser-/Native-Versionen und -Hashes (`schema.mjs:494-523`, Multiplex je Collection `schema.mjs:539-573`; Payload aus der aktuellen Collection `replication-webrtc.mjs:1036-1054`). Sie sagt nichts ueber die Version der bereits in IndexedDB liegenden Zeilen.

**Store-Messung:** Derselbe IndexedDB-Name wurde zuerst mit Schema v0 beschrieben, geschlossen und mit Schema v1 (neues Pflichtfeld `added`) wieder geoeffnet. Ergebnis: `openError=null`, Laufzeitdeklaration `declaredVersion=1`, aber beide alten Zeilen blieben erhalten und hatten `added=null`. Also reale Zahl: **2/2 alte Zeilen blieben**, ohne Versionsfehler.

### 3. Gibt es „Collection verwerfen und voll neu pullen“ bereits? Was passiert real beim Versionswechsel?

**Keinen collection-spezifischen, atomaren Verwerfen-und-Neu-Pullen-Pfad. Es existieren nur Bausteine und ein datenbankweiter Notausgang.**

- Catch-up registriert/validiert eine Collection und ruft `onPeerReady`; bei aktuellem Browser-/Native-Schemamismatch wird die Collection quiesziert (`replication-webrtc.mjs:342-365`, `replication-webrtc.mjs:763-776`). Es leert keine lokalen Zeilen.
- Retained Checkpoints werden bei geaenderter Remote-Generation/Collection-Head/Schema-Hash bzw. geaendertem lokalen Collection-Head verworfen (`replication-webrtc.mjs:1079-1117`; Validity-Key enthaelt Remote-Schema-Hash `replication-webrtc.mjs:2258-2300`). Danach zieht `pullFromPeer` ab `checkpoint=null` bis zur ersten leeren Antwort (`replication-webrtc.mjs:1221-1250`). Das ist ein **voller Pull-Cursor**, aber kein Truncate: vorhandene lokale Zeilen werden nur ID-weise ueberschrieben; nicht erneut gelieferte Altzeilen werden nicht entfernt.
- Der Storage hat nur `hardDeleteByIds(ids)` und verlangt, dass der Aufrufer Dirty-Zeilen schuetzt (`storage-indexeddb.mjs:767-781`; der Eviction-Aufrufer verweigert `pushable!=0`, `replication-webrtc.mjs:1831-1839`). Es gibt kein `clearCollection`/`truncateCollection`.
- Datenbankweit gibt es `resetBusinessDb()`: es prueft einen LocalStorage-Spiegel des Recovery-Status, erlaubt bei Pending Writes nach einem Export weiterzumachen und loescht dann nur die primaere IndexedDB (`shared/db.js:51-74`, `shared/db.js:180-191`). `removeRxDatabase()` ist ebenfalls nur `indexedDB.deleteDatabase(name)` (`rx-database.mjs:48-55`). Der Recovery-Store `${name}__recovery_v2` und die collection-spezifischen Query-Meta-Sidecars werden dadurch nicht geloescht.
- Ein frischer Replica-Namespace ist als manueller Build-/Storage-Generation-Baustein vorhanden: `BUSINESS_DB_STORAGE_GENERATION` fliesst in den DB-Namen; der Kommentar verspricht eine frische lokale Replica und Repopulation ueber Replikation (`app.js:82-87`, Namensbildung `app.js:1099-1117`). Er ist nicht an Collection-Schema-Versionen gekoppelt und prueft den alten Namespace nicht auf ungesyncte Writes.
- Nach einem primaeren Reset verhindert der lokale Collection-Head zwar die Wiederverwendung des alten Pull-Checkpoints; genau das pinnt der Smoke (`src/apps/business-os/rxdb/tests/replication-recovery-smoke.mjs:441-455`). Damit kann ein **datenbankweiter** Reset einen Full Pull ausloesen, aber nur wenn der Reset selbst sicher war und Sidecar/WAL korrekt behandelt werden.

**Real heute beim normalen Versionswechsel:**

1. Kein neuer Browser-Object-Store und keine versionsabhaengige Tabelle entstehen; alle Versionen teilen `documents` mit Key `[collection,id]` (`storage-indexeddb.mjs:1164-1184`).
2. `addCollections` akzeptiert die neue Deklaration ohne Persistenzvergleich (`rx-database.mjs:124-161`).
3. Wenn Browser und Native noch unterschiedliche deklarierte Versionen/Hashes haben, stoppt/quiesziert die Replikation fuer diese Collection (`schema.mjs:494-523`, `replication-webrtc.mjs:763-776`).
4. Wenn beide Deklarationen schon uebereinstimmen, kann der geaenderte Hash den alten Checkpoint invalidieren und einen Pull ab null ausloesen (`replication-webrtc.mjs:1079-1117`, `replication-webrtc.mjs:1221-1250`), **aber die alten lokalen Zeilen bleiben vorher und waehrenddessen liegen**. Anders als nativ bleibt keine separate v0-Tabelle; funktional bleibt jedoch derselbe Altbestand in der gemeinsamen Browser-Tabelle liegen.

**Zweite Store-Messung zum vorhandenen Reset:** Eine lokale eindeutige Zeile ergab vor Reset `docs=1`, `unsynced=1`, WAL `pendingWrites=1`. Nach Recovery-Export erlaubte `resetBusinessDb()` den Reset. Nach Wiedereroeffnung: `docs=0`, `unsynced=0`, aber WAL weiterhin `pendingWrites=1`. Ursache: primaer-committete Pending-Batches werden beim Replay uebersprungen (`recovery-journal.mjs:133-144`). Der Write ist noch im WAL/Export konserviert, aber aus Primaerstore und Pushpfad verschwunden. Der bestehende Reset ist deshalb **kein sicherer Cache-Versionswechselpfad bei Pending Writes**, auch nicht nur deshalb, weil vorher exportiert wurde.

### 4. Verdikt

**JA, der Cache-Ansatz ist sicher umsetzbar; NEIN, er ist heute noch nicht sicher implementiert/aktivierbar.**

Er traegt nur unter dieser fail-closed Schutzbedingung:

1. Collection-Schreibannahme und Multi-Tab-Replikationsbesitz zuerst quieszieren, damit zwischen Pruefung und Verwerfen kein neuer lokaler Write landen kann.
2. Fuer die betroffene Collection **beide persistenten Quellen live** pruefen: Primaerrecords `pushable==1` **und** Recovery-WAL `state=='pending'`. Nicht allein LocalStorage-Snapshot, Push-Checkpoint oder Export-Zeitstempel verwenden.
3. Ist eine der Pruefungen nicht moeglich oder ist irgendein Count > 0: **nicht verwerfen, Versionswechsel fail-closed stoppen**. Der aktuelle `resetBusinessDb`-Sonderfall „Export ist neuer als oldestPending, dann loeschen“ reicht fuer automatisches Weiterlaufen nicht; die Messung strandete den Write ausserhalb des Pushpfads.
4. Bei 0/0: persistierten Collection-Schema-Marker aktualisieren und in einem kontrollierten Ablauf primaere Collection-Zeilen, retained Pull-/Push-Checkpoints, `firstPullCompletedAtMs` und Query-Meta-/Demand-Window-Sidecar invalidieren; danach Pull explizit bei `null` starten und bis zur leeren Antwort drainieren.
5. Fehler beim Clear/Markerwechsel/Checkpoint-Reset ebenfalls fail-closed. Ohne atomaren/serialisierten Ablauf waere die Pruefung TOCTOU-anfaellig.

Was fehlt, ist nicht die Erkennbarkeit ungesyncter Zeilen — die ist vorhanden — sondern (a) persistierte Collection-Version/Schema-Hash, (b) collection-spezifischer serialisierter Clear, (c) gemeinsames Invalidieren von Checkpoints/Readiness/Sidecar, und (d) ein Guard, der Primaer- und WAL-Zustand unter Schreibstopp gemeinsam bewertet.

## kompensationen_geloescht

- Keine; reine Messung.

## verblieben

- Kein persistenter Collection-Versions-/Schema-Hash-Marker im Browser-Primary.
- Kein collection-spezifischer Clear-/Truncate-und-Full-Pull-Mechanismus.
- Kein atomarer/quieszierter Guard gegen neue Multi-Tab-Writes zwischen Unsynced-Pruefung und Clear.
- `resetBusinessDb()` prueft nur den LocalStorage-Recovery-Spiegel und erlaubt Pending Writes nach Export; reale Messung: Primaerzeile weg, Pending-WAL bleibt, Replay stellt sie nicht wieder in den Pushpfad.
- Query-Meta-Sidecar ist separat und muss beim Schema-Cachewechsel mit invalidiert werden (`query-meta-storage.mjs:209-210`, Backend-Clear `query-meta-backend-indexeddb.mjs:202-210`; aktuelle Aktivierung nutzt collection-spezifische Sidecar-DB-Namen `replication-webrtc.mjs:1823-1845`).
- Recovery-WAL-Schemahash schuetzt nur Replay-Batches; primaer-committete Pending-Batches umgehen die Hashpruefung (`recovery-journal.mjs:133-144`).

## tests

- `CARGO_TARGET_DIR=/Volumes/tmp/ctox-pipeline-targets/I-061 cargo fmt --check` — **gruen**, keine Ausgabe. Dieser Befehl fuehrt 0 Tests aus und hat daher keine `test result`-Zeile.
- Erster Lauf `CARGO_TARGET_DIR=/Volumes/tmp/ctox-pipeline-targets/I-061 cargo check --bin ctox` — **nicht fachlich rot**, Abbruch 101 durch verschwundenes Target-Unterverzeichnis/SIGKILL waehrend paralleler Builds (`failed to write ... No such file or directory`); keine Tests, daher keine `test result`-Zeile.
- Retry mit isoliertem Ziel (nach Seeding desselben `I-061`-Targets aus einem abgeschlossenen lokalen Cargo-Cache): `CARGO_TARGET_DIR=/Volumes/tmp/ctox-pipeline-targets/I-061 cargo check --bin ctox` — **gruen**, `Finished dev profile ... in 24m 14s`; 405 vorbestehende Warnungen, 0 Fehler. Keine Tests/keine `test result`-Zeile.
- `node src/apps/business-os/rxdb/tests/storage-index-smoke.mjs` — **gruen**, 1 OK-Marker (`ctox-rxdb-js storage index smoke OK`).
- `node src/apps/business-os/rxdb/tests/first-pull-readiness-smoke.mjs` — **gruen**, 1 OK-Marker.
- `PLAYWRIGHT_MODULE_PATH=/Users/michaelwelsch/Documents/ctox-dev/node_modules/playwright node src/apps/business-os/rxdb/tests/recovery-primary-reset-browser-smoke.mjs` — **gruen**, 2 explizite Assertions/1 OK-Marker (`blockedCode=recovery_export_required`, `retryOk=true`). Der erste Versuch ohne `PLAYWRIGHT_MODULE_PATH` fand Playwright nicht; kein Test lief.
- `PLAYWRIGHT_MODULE_PATH=/Users/michaelwelsch/Documents/ctox-dev/node_modules/playwright node src/apps/business-os/rxdb/tests/recovery-journal-browser-smoke.mjs` — **gruen**, 1 OK-Marker.
- `node src/apps/business-os/rxdb/tests/replication-recovery-smoke.mjs` — **rot**, genau 1 sichtbare Assertion: `targeted push should retry the late handler twice, got 1` bei Zeile 178. Gegen sauberem HEAD via `git archive HEAD` unter `/tmp/i061-head` reproduziert: exakt dieselbe eine rote Assertion; Rot-Mengen in beide Richtungen identisch: `{targeted push should retry the late handler twice, got 1}` / `{targeted push should retry the late handler twice, got 1}`. Kein zusaetzlicher I-061-Rotfall.
- Zwei echte Browser-Store-Messungen aus `/tmp` (keine Repo-Aenderung): (1) Versionswechsel v0→v1: 2/2 Altzeilen blieben, 1 `pushable`, 1 Pending-WAL, `openError=null`; (2) erlaubter Primary-Reset nach Export: vorher 1 Primaerzeile/1 Pending, nachher 0 Primaerzeilen/1 Pending.

Die verbotenen Cargo-Testfilter wurden nicht verwendet; es wurde kein Cargo-Testfilter benutzt, weil keine Cargo-Tests aufgerufen wurden. Die vorgeschriebenen Cargo-Befehle sind Format-/Check-Befehle und erzeugen definitionsgemaess keine `test result`-Zeile.

## gegenprobe

- Pflicht-Gegenprobe entfaellt laut Auftrag (reine Messung).
- Zusaetzlicher Messbeweis statt Code-Gegenprobe: reale IndexedDB-Zahlen aus zwei Headless-Chrome-Laeufen, oben dokumentiert.
- Kein Rueckbau noetig, da keine Repository-Datei geaendert wurde. `git diff --stat`/`git diff --cached --stat` zeigen ausschliesslich den vorbestehenden geteilten Arbeitsbaum (Zahlen unter `was_geaendert`).

## offene_bedenken

- Die oeffentliche Unsynced-API liefert Counts je Collection, aber keine IDs und keinen „check-and-clear unter derselben Sperre“-Vorgang. Fuer den Guard reicht der Count semantisch; fuer TOCTOU-Sicherheit fehlt die Serialisierung.
- Der Recovery-Status in LocalStorage ist diagnostischer Spiegel und kann fehlen/veraltet sein; Runde 2 muss den IndexedDB-WAL live lesen.
- Master-Acknowledgement ist konservativ und kann nach erfolgreichem `masterWrite` bis zum Pull-Roundtrip pending bleiben. Das kann einen Versionswechsel laenger blockieren, ist fuer Datenverlustschutz aber richtig.
- Demand-only Collections benoetigen neben dem primaeren Clear zwingend Sidecar-Window-Invalidierung; ein Full Pull allein fuellt sie definitionsgemaess nicht vollstaendig.
- Bei Built-in Collections koennen Registry-Hashes die Hashberechnung uebersteuern; der neue persistierte Marker muss deklarierte Version **und** effektiven Hash speichern/vergleichen, nicht nur einen der beiden Werte.

## pfade

Runde 2 braucht mindestens:

- `src/apps/business-os/rxdb/src/storage-indexeddb.mjs:15-18,49-64,767-781,987-1050,1151-1255,1289-1338` — persistente Collection-Metadaten (Version+effektiver Hash), collection-spezifischer Unsynced-Scan und Clear unter Serialisierung.
- `src/apps/business-os/rxdb/src/recovery-journal.mjs:25-28,45-101,133-179,260-278,357-369,408-416,441-463,553-563` — live pending-by-collection Guard; Reset-/Replay-Semantik fuer `primaryCommittedAtMs`; kein Export-Bypass fuer automatischen Versionswechsel.
- `src/apps/business-os/rxdb/src/rx-database.mjs:124-170,194-202` — beim `addCollections` deklarierte Version/Hash gegen persistierte Metadaten vergleichen und fail-closed Resetbedarf an die Replikation geben; `migrationStrategies` nicht als Browser-Migrationsmaschine verwenden.
- `src/apps/business-os/rxdb/src/replication-webrtc.mjs:904-935,1036-1117,1136-1250,1931-1941,1966-2063,2258-2329` — expliziter Version-Invalidationspfad: retained Pull/Push, Readiness und Persistenzkey loeschen, dann Pull ab null bis leer.
- `src/apps/business-os/rxdb/src/query-meta-storage.mjs:204-219` und `src/apps/business-os/rxdb/src/query-meta-backend-indexeddb.mjs:202-210` sowie Aufrufstelle `src/apps/business-os/rxdb/src/replication-webrtc.mjs:1799-1845` — Query-/Demand-Sidecar der betroffenen Collection invalidieren.
- `src/apps/business-os/shared/db.js:19,51-74,128,180-191,210-238` — Shell-Fassade fuer fail-closed Versionwechsel; bestehenden export-erlaubten Notreset nicht als Migrationspfad wiederverwenden; Cache-Buster bumpen.
- `src/apps/business-os/shared/sync.js:361,1100` — identischen Bundle-Cache-Buster bumpen; falls die Versioninvalidierung ueber die Sync-Lifecycle-Fassade ausgeloest wird, dort verdrahten.
- `src/apps/business-os/rxdb/tests/` — neue Browser-Smokes fuer (a) v0-Altzeilen werden bei 0/0 geloescht und voll neu gezogen, (b) `pushable>0` blockiert, (c) WAL-pending bei noch fehlender Primaerzeile blockiert, (d) Multi-Tab-Write im Wechsel blockiert/serialisiert, (e) Sidecar-Window wird invalidiert, (f) Reset nach Export darf Pending nicht aus dem Pushpfad stranden.
- `src/apps/business-os/rxdb/dist/ctox-rxdb-js.mjs` — **nur** per gepinntem esbuild-Befehl aus `src/index.mjs` neu bauen, nie direkt editieren (`src/apps/business-os/rxdb/AGENTS.md:13-24`; `docs/ctox-rxdb.md:869-898`).
- `docs/ctox-rxdb.md:136-175,653-665,807-857,869-898` — den implementierten Cache-Versionswechsel, Guard und Tests dokumentieren.

Optional nur bei Wahl des globalen Namespace-Rotationsansatzes statt collection-spezifischem Clear:

- `src/apps/business-os/app.js:82-87,1099-1155,1217-1229` — Storage-Generation aus einem kanonischen Schema-Manifest ableiten und **vor** Rotation den alten DB-/WAL-Namespace live fail-closed pruefen. Ohne diese Pruefung strandet ein Pending Write lediglich im alten Namespace.
