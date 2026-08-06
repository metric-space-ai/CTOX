# F-04 — Sellify: 238.913 CRM-Zeilen in v1, Registrierung nimmt v0

Messung only (2026-08-06). Keine Dateien/Stores verändert. Live-Store und
Code-Pfade gelesen. Live-Binary: `~/.local/lib/ctox/current` →
`releases/branch-main-20260724T072316Z` (`ctox-real` mtime 2026-07-24 10:11).

Zahlen im Store (SELECT, bestätigt):

| Collection | `__v0` | `__v1` |
|---|---:|---:|
| sellify_companies | 0 | 17.516 |
| sellify_campaigns | 0 | 86.549 |
| sellify_people | 0 | 60.639 |
| sellify_activities | 0 | 74.209 |
| **Summe CRM** | **0** | **238.913** |
| sellify_records | (keine v0) | 0 |

`business_records` (business-os.sqlite3) spiegelt dieselben 238.913 Zeilen und
dieselben `updated_at_ms`-Fenster wie die v1-Tabellen.

---

## schreibpfad

### 1) Peer-Registrierung → **deklarierte Version**
(`rxdb_collection_version_table_name` / `schema.version`)

- Tabellenname immer `{db}__{collection}__v{schema_version}`:
  - `src/core/rxdb/src/storage/sqlite/sql.rs:57-58`
  - gebindet in `src/core/rxdb/src/storage/sqlite/index_mod.rs:26-30`
    (`params.schema.version`)
- Runtime-Module (Sellify) kommen **nicht** aus
  `business_os_schema_contract.json` (0 Sellify-Keys, Contract hat 178
  Collections). Sie werden aus
  `runtime/business-os/installed-modules/*/collections.schema.json` geladen:
  - `rxdb_peer.rs:13687-13691` `collection_creators_for_root` merged
    `runtime_installed_module_collection_creators`
  - `rxdb_peer.rs:14027-14036` setzt fehlendes `version` default auf **0**
  - Heutige Manifeste (alle gelesen) deklarieren CRM-Collections als
    **version = 0**, nur `sellify_records` = 1:
    - `~/.local/lib/ctox/current/runtime/business-os/installed-modules/sellify/collections.schema.json`
    - `~/.local/state/ctox/business-os/installed-modules/sellify/collections.schema.json`
    - `~/.local/state/ctox/sellify-private-v0428-source/sellify/collections.schema.json`
    - `~/.local/state/ctox/thesen-private/sellify/collections.schema.json` (wrapper-Form, schema.version=0)
  - Browser `schema.js` analog: `sellify_people` … `"version": 0`
    (Export `migrationStrategies` mappt identity `1: (doc) => doc`, obwohl die
    Collection selbst auf 0 steht).

**Peer-Registrierung heute bindet also `__v0`.**

### 2) Store-Projektor / Import-Spiegel → **expected, dann latest**
(`rxdb_collection_table_name` / nicht `latest_rxdb_collection_table` allein)

- `store.rs:1848-1887` `rxdb_collection_table_name`:
  1. baut `expected = ctox_business_os__{collection}__v{rxdb_schema_version}`
  2. `rxdb_schema_version` liest **nur** den Contract
     (`business_os_schema_contract.json`); unbekannt → **0** (`:1882-1887`)
  3. wenn `expected` existiert → **genau diese** Tabelle
  4. sonst Fallback: **max. vorhandene** Version (latest)
- `RxdbCollectionWriter::open` (`store.rs:15560-15562`) nutzt genau diesen
  Pfad. `BusinessProjectionWriter`/`RxdbProjectionWriterCache` schreiben
  darüber in die so gewählte Tabelle (`:15417-15430`, `:15488-15496`).
- `attached_rxdb_collection_table` (`store.rs:1151-1172`) ebenfalls expected
  aus Contract-Version, dann `rxdb_collection_table_name_from_tables`.

**Heute (v0- und v1-Tabelle existieren):** der Projektor würde
`__v0` wählen (expected=0 und Tabelle vorhanden) — **nicht** die vollen
v1-Daten. Die v1-Füllung stammt damit **nicht** vom aktuellen
Projektor-Default.

### 3) `latest_rxdb_collection_table` (`rxdb_peer.rs:10119-10146`)

- Scannt `sqlite_master`, wählt max. `__{collection}__vN`.
- Im Peer-Code für Sellify-Import/CRM **nicht** der Schreibpfad; genutzt u. a.
  für `business_commands` und andere Lookup-Helfer (`:1590`, `:4688`,
  `:9851`, `:9971`). **Nicht** die Peer-Collection-Registrierung.

### 4) Wer hat die **v1-Zeilen** geschrieben?

Belege, dass die v1-Zeilen **über den Storage-Pfad mit `schema.version = 1`**
entstanden sind (Peer/Browser-Collection mit deklarierter v1), nicht über
„latest“-Lookup:

1. **Internal-Meta enthält beide Versionen** mit unterschiedlichen
   `schemaHash` (Hash enthält die Versionsnummer; Schema sonst identisch —
   Property-Set und Indexes equal sans `version`, gemessen):
   - `collection|sellify_people-0` version=0 hash=`3c8432ae…`
   - `collection|sellify_people-1` version=1 hash=`9642436b…`
   - analog activities/campaigns/companies 0+1; records nur 1.
   - Meta-Schreiben: `rx_database.rs:663-683` (`write_collection_meta`) bei
     `addCollections` mit `schema.version()`.
2. **Dokumentierte v0→v1-Migration am 2026-07-11**
   (`docs/business-os-app-platform-refactoring-plan.md:1406-1431`, Revision 19):
   Sellify hatte damals kanonische **additive v0→v1-Identitätsmigrationen**;
   der installierte Peer migrierte und verifizierte **genau 238.913** Envelopes
   (74.209 / 86.549 / 17.516 / 60.639) und entfernte danach vier v0-Tabellen.
   Das deckt sich **exakt** mit den heutigen v1-Zählern.
3. **`__rxdb_changed_tables`** zählt für **beide**
   `…sellify_people__v0` und `…v1` jeweils **60639** — konsistent mit
   „v0 wurde einmal voll geschrieben, dann nach v1 kopiert; v0 später neu
   als leere Hülle angelegt, Counter in `__rxdb_changed_tables` bleibt“.
4. **Zeilen-Timestamps = 2026-07-04** (siehe zeitliche_reihenfolge) —
   Inhalt aus dem Sellify-Sync-Refresh; physische v1-Tabelle und
   Internal-v1-Meta entstanden im Zuge der späteren Schema-v1-Phase
   (Doku: 2026-07-11).

**Fazit Schreibpfad:** Die 238.913 CRM-Zeilen liegen in `__v1`, weil Sellify
**zeitweise mit version=1 registriert** war (Storage + Migrationscopy). Der
heutige Writer/Peer beachtet wieder version=0 und schreibt/liest `__v0`.
Es war **kein** dauerhaftes „Writer ignoriert deklarierte Version und nimmt
latest“, sondern ein **Versions-Downgrade der Deklaration** nach einer
erfolgreichen v0→v1-Migration, ohne Rückmigration der Daten.

---

## zeitliche_reihenfolge

| Wann (UTC) | Ereignis | Beleg |
|---|---|---|
| 2026-07-04 11:43–14:13 | Sellify-Sync schreibt 238.913 CRM-Records | `lastWriteTime`/`updated_at_ms`: companies 1783173098448–1783173209072; people 1783173210900–1783173646307; campaigns 1783173648900–1783174400030; activities max 1783165397470. Statusnote `~/.local/state/ctox/sellify-rework-status-note-20260704T1428Z.txt` (damals: Daten in `business_records`, **noch keine** sellify-Tabellen im RxDB-Store). |
| 2026-07-11 | Doku: Sellify 0.4.18, v0→v1-Identitätsmigration, 238.913 Envelopes, v0-Tabellen entfernt | `docs/business-os-app-platform-refactoring-plan.md:1423-1431` |
| danach → heute | Deklaration wieder **version=0**; leere `__v0`-Tabellen + Indexes + Triggers existieren wieder; Daten bleiben in `__v1` | Alle installierten/source Manifeste version=0; `COUNT(*)` v0=0 / v1=voll; Internal hat v0- und v1-Meta |
| 2026-08-04 13:56 | Installiertes Modul `sellify` mtime (collections.schema.json version=0) | `stat` auf runtime + state installed-modules |
| 2026-08-06 04:46 | Live-Peer-Start pid=82511, „195 collections“, **kein** sellify-SC39 in diesem Boot | `ctox_service.log` ab Zeile 556360; status.json |

**War Sellify je auf v1 deklariert und wurde heruntergestuft?**
**Ja.** Belege: Internal-Meta version=1; Doku Revision 19 (explizit v0→v1 und
Cleanup der v0-Tabellen); heutige Manifeste/schema.js wieder version=0; leere
v0-Tabellen sind neu provisionierte Hüllen (CREATE TABLE + volle Index-Sets
identisch zu v1), nicht der Datenstand von Juli 4.

**Hat ein Writer die deklarierte Version „nie beachtet“?**
Für die **v1-Füllung: nein** — er hat die damalige Deklaration 1 beachtet.
Für den **heutigen Lese-/Registrierungspfad: er beachtet 0** und landet auf
der leeren Tabelle. `latest_*` ist hier nicht der Import-Writer.

Zusatz: Frühere SC39-Fehler im Log (z. B. ab Zeile 6839) zeigen Registrierung
mit `"version": 0` aber **ohne** `id.maxLength` in einem älteren Schema-Pfad
(`rx_schema.rs:64-74` SC39 = primaryKey ohne maxLength). Das ist ein
**separater** Schema-Validierungsfehler, kein Versions-Mismatch. Heutige
Manifeste haben `id.maxLength: 180`; der Boot vom 06.08. loggt kein SC39 für
Sellify.

---

## peer_registrierung_heute

- **Prozess:** pid **82511**, gestartet 2026-08-06 04:46:32,
  `ctox-real service --foreground`, Binary Release
  `branch-main-20260724T072316Z` (mtime 2026-07-24 — **vor** SYNC-E 05.08.).
- **Statusfile:**
  `~/.local/state/ctox/business-os-rxdb-peer.status.json`
  - `running: true`, `replicationUp: true`
  - `database_path`:
    `/Users/michaelwelsch/.local/lib/ctox/current/runtime/business-os-rxdb.sqlite3`
  - gleicher Inode wie `~/.local/state/ctox/business-os-rxdb.sqlite3`
    (inode 138776861)
  - **keine** Collection-Namenliste im Status (0× „sellify“ im JSON)
- **Boot-Log (pid 82511):**  
  `multiplexed WebRTC replication up for 195 collections` — **kein**
  `skipping … sellify_*` und **kein** SC39 in diesem Boot-Abschnitt.
  Sellify-`collections.schema.json` hat gültiges
  `schema_format: ctox-business-os-module-collections-v1` (im Gegensatz zu
  fünf `rem-*`-Modulen, die als unsupported übersprungen werden).
- **Gebundene Tabelle für `sellify_people`:** aus Code + Deklaration
  zwingend  
  **`ctox_business_os__sellify_people__v0`**  
  (`index_mod.rs:26-30` × Manifest version=0 × Internal active max version
  Query würde 1 liefern, aber die **lebende Collection** nutzt die
  registrierte Schema-Version 0, nicht `active_rxdb_collection_version`).
- Browser-Seite registriert dieselbe Version 0 aus `schema.js`
  (`app.js:5201-5225` `registerModuleSchemas` → `db.addCollections`).

**Messlücke:** Der Peer-Status listet keine Collection→Tabelle-Map; die
Bindung v0 ist aus Code+Manifest+Boot-Erfolg abgeleitet, nicht aus einem
expliziten Log-String „bound sellify_people → …v0“. Es gibt **kein**
Gegenbeispiel im heutigen Boot, das auf v1-Bindung hinweist.

---

## migrationsvertrag

### Was SYNC-E (05.08.) vorsieht

Commit `c46499358` (2026-08-05, Message: SYNC-E /
`docs/ctox-migrations-plan-2026-08-05.md`):

Bring-up-Reihenfolge (HEAD / Doku `docs/ctox-rxdb.md:712-713`):

1. Collections registrieren (`add_collections_tolerant`)
2. `migrate_additive_native_rxdb_collection_versions` — für jede
   **nichtleere** Quellversion `<` Zielversion: Strategieschritte aus
   `migration_strategies`, Copy, Verify nach `lastWriteTime`
3. erst dann `repair_stale_rxdb_collection_schema_versions` (Drop stale)

Relevante Stellen (HEAD):

- `migrate_additive…`: nur **aufwärts** `source_version < target_version`;
  sonst fail-closed  
  *„cannot apply a reverse migration“* (git show HEAD, Funktion ab
  `rxdb_peer.rs` ~13883; Bedingung `source_version < target_version`).
- Identity-Migration: `operations: []` in
  `migration_strategies.<collection>.1` (Sellify-Manifest hat das für
  people/companies/… — aber die **Collection-version ist 0**, Ziel ist also 0).
- `repair_stale…` / `stale_rxdb_collection_version_tables`
  (`rxdb_peer.rs:14596-14604`): räumt **nur**, wenn
  `active_version == expected_version` **und** expected table exists.
  Iteriert zudem nur `business_os_collections()` = **Contract**
  (`:14374`) — **Sellify ist nicht im Contract** → Startup-Sweep greift
  Sellify **gar nicht**.
- `expected_rxdb_collection_version` (`:14562-14567`) ebenfalls nur Contract
  → für Sellify-Namen ohnehin 0.

### Würde der Pfad einen v0→v1-Sprung abdecken?

**Ja, wenn** die deklarierte Zielversion **1** ist, `__v0` nichtleer ist und
`migration_strategies.<col>.1` existiert (Sellify hat identity `operations:[]`).
Dann: Copy v0→v1, Verify, danach Sweep v0.

### Warum er hier nicht gefeuert hat / nicht hilft

1. **Live-Binary ist vom 24.07.2026** — **ohne** den 05.08.-Migrations-
   Lebenszyklus. Der laufende Peer kann `migrate_additive…` nicht ausführen
   (Binary-Strings/Release-Datum; SYNC-E-Commit 05.08. liegt nach dem Release).
2. **Richtung ist umgekehrt:** Daten liegen in **v1**, Deklaration ist **v0**.
   Selbst HEAD-`migrate_additive` würde bei `source=1, target=0, rows>0`
   **fail-closed** („cannot apply a reverse migration“) — und **nicht**
   v1→v0 kopieren.
3. **Stale-Repair** würde v1 auch dann nicht droppen, wenn Sellify im Contract
   wäre: `active_version` (Internal DESC) = 1 ≠ expected 0 →
   `stale_rxdb_collection_version_tables` returns `[]` (`:14603-14604`).
   Zusätzlich: Sellify ist **nicht** im Contract-Iterationsraum.
4. Commit-Message von c46499358 selbst: *„Leere Alt-Tabellen (der heutige
   Sellify-Bestand)“* — der Autor sah denselben Zustand (leere v0, Daten
   woanders/Ziel) und behandelte Sellify als **empty-legacy**-Fall, nicht als
   Reverse-Migration der 238.913 v1-Zeilen.

**Kurz:** SYNC-E deckt den **Aufwärts**-Sprung v0→v1 ab. Der Live-Fall ist
**Deklarations-Downgrade 1→0 bei vollen v1-Tabellen**. Weder der alte Jul-24-
Peer noch der Aug-05-Vertrag migrieren das automatisch zurück oder heben die
Deklaration.

---

## reparaturoptionen

### (a) Manifest (und schema.js) auf version = 1 heben
- **Wirkung:** Peer + Browser registrieren `__v1` → 238.913 Zeilen wieder
  sichtbar; Identity-`migration_strategies.1` passt; Internal-v1-Meta existiert.
- **Risiko:** Mittel. Schema muss **byte-kompatibel** zum gespeicherten
  v1-Schema bleiben (sonst DB6 schemaHash mismatch, `rx_database.rs:713-738`).
  Gemessen: v0/v1-Schemas equal sans version — Heben von 0→1 mit gleichem Body
  ist der vorgesehene Weg. Browser und Native müssen **gleichzeitig** 1 führen.
  `schema.js` exportiert derzeit version 0; nur `collections.schema.json` zu
  ändern reicht für den Browser nicht (`app.js` lädt `schema.js`).
- **Empfehlung:** **Ja, primär empfohlen.** Stellt den dokumentierten Zustand
  nach Revision 19 wieder her, ohne 238.913 Zeilen zu bewegen.

### (b) Daten v1 → v0 migrieren (Copy + Verify, dann optional v1 droppen)
- **Wirkung:** Passt Store an die heutige Deklaration 0 an.
- **Risiko:** Hoch. Kein unterstützter Reverse-Pfad im SYNC-E-Code
  (fail-closed). Manuelles SQL-Copy muss Envelopes/`lastWriteTime`/`_rev`
  erhalten; bei parallelem Peer gefährlich. Danach Internal-Meta v1 und
  leere/volle Tabellen konsistent halten, sonst nächster Schema-Bump
  bricht. 238.913 Zeilen × 4 Tabellen, Store ~2,2 GB.
- **Empfehlung:** Nur wenn version=0 aus Produktgründen **hart** bleiben muss;
  dann als offline, backup-geschützte Operation — nicht „mal schnell SQL“.

### (c) Registrierung auf „latest“ umstellen
- **Wirkung:** `latest_rxdb_collection_table` / Fallback in
  `rxdb_collection_table_name_from_tables` als primäre Bindung → würde heute
  v1 sehen.
- **Risiko:** Hoch/systemisch. Bricht die RxDB-Invariante „Tabellenname =
  deklarierte schema.version“ (`sql.rs:57-58`, Checkpoint/Schema-Hash pro
  Version). Zwei Peers/Browser mit unterschiedlicher Deklaration könnten
  still verschiedene Tabellen meinen. Projektor und Peer müssten **alle**
  umgestellt werden; Replikation/Checkpoints sind versionsgebunden.
- **Empfehlung:** **Nicht** als Sellify-Hotfix. Höchstens langfristig als
  bewusster Plattformvertrag — out of scope für diesen CRM-Vorfall.

### Empfehlung (ohne Ausführung)
**(a) Manifest + schema.js auf 1**, Peer neu starten, prüfen dass
`sellify_people` an `__v1` hängt und Zähler 60.639 liefert. Optional danach
leere Alt-Artefakte nur über den **dokumentierten** Sweep, wenn
`active_version == expected == 1`.  
**(b)** nur als bewusste Downgrade-Migration mit Backup.  
**(c)** ablehnen als Lokalpatch.

---

## unsicherheiten

1. **Exakter Zeitpunkt der erneuten v0-Provisionierung** (nach dem 11.07.-v1-
   Cleanup): kein CREATE-TIMESTAMP in SQLite; Internal-Meta hat
   `lastWriteTime=1.0` (Default) — nicht auswertbar. Modul-mtime 04.08. ist
   nur obere/Installations-Spur, nicht zwingend der erste v0-Rebuild.
2. **Live-Peer listet keine Collection→Tabelle-Bindung** im Statusfile; v0-
   Bindung ist code-deduziert. Ein in-process Dump der registrierten
   `RxCollection.schema.version` wurde nicht gemessen (kein Read-API ohne
   Seiteneffekte am laufenden Prozess).
3. **195 vs. ~205 erwartete Collections:** 178 Contract + 27 gültige
   Runtime-Module-Collections; 10 Differenz ungeklärt (fehlgeschlagene
   optionale Contract-Collections? local-modules? Doppel-Skip?). Sellify
   (8) ist unter den gültigen Runtime-Formaten; ob alle 8 im Multiplex
   sind, ist nicht zeilenweise geloggt.
4. **Working-Tree vs. HEAD vs. Live-Binary:** Repo-Working-Tree hat lokale
   Mods an `rxdb_peer.rs` (`git status MM`); HEAD trägt SYNC-E; **Live-
   Binary ist 24.07.** Aussagen über „was der laufende Peer tut“ beziehen
   sich auf das Jul-24-Binary + Log, nicht auf uncommittete Tree-Stände.
5. **Wer genau die erste v1-Collection geöffnet hat** (native Peer vs.
   Browser-Tab): Doku spricht vom „installierten Peer“; Log-Ausschnitte der
   Migration vom 11.07. wurden in `ctox_service.log` nicht als eigene
   Migrations-Zeilen nachgezogen (Log-Rotation/Volumen). Inhaltliche
   Übereinstimmung der Zähler bleibt der stärkste Beleg.
6. **`business_records_projection_clock`** für sellify_campaigns zeigt
   row_count 86550 vs. Tabelle 86549 — 1 Zeile Drift, für F-04 nebensächlich,
   aber Projektor und RxDB sind nicht bitidentisch synchron.

