# F-02 Runde 1 — Collection-Registrierung im Betrieb (nur messen)

**Arbeitsbaum:** `/Users/michaelwelsch/Documents/ctox` (nur gelesen)  
**Log:** `/Users/michaelwelsch/.local/lib/ctox/current/runtime/ctox_service.log`  
**RxDB-Store:** `/Users/michaelwelsch/.local/lib/ctox/current/runtime/business-os-rxdb.sqlite3`  
**Business-Store:** `/Users/michaelwelsch/.local/lib/ctox/current/runtime/business-os.sqlite3`  
**Stand Messung:** 2026-08-06 (Log enthält 668 Registrierungs-Skips; **letzte** `skipping optional … registration failed` bei Logzeile **24819** — danach kein erneuter Registrierungs-Skip mehr)

---

## ursache_belegt

### 1) Wo wird geloggt, und warum wirkt `{err}` „leer“?

**Schreibstelle:** `src/core/business_os/rxdb_peer.rs:2420–2435`

```text
let (collections, failed_collections) = database
    .add_collections_tolerant(collection_creators_for_root(&root))
    .await ...;
for (collection_name, err) in &failed_collections {
    if is_required_native_collection(collection_name) { return Err(...); }
    eprintln!(
        "[business-os] skipping optional Business OS RxDB collection `{collection_name}` \
        (registration failed: {err})"
    );
}
```

- Tolerante Registrierung: `src/core/rxdb/src/rx_database.rs:308–333` (`add_collections_tolerant`) — Fehler landen in `failed`, Peer läuft mit dem Rest weiter.
- Optional vs. required: `src/core/business_os/rxdb_peer.rs:14312–14321` (`is_required_native_collection`).  
  Die hier betroffenen Collections (`sellify_*`, `user_threads`, `user_thread_messages`, `browser_*`, `business_chats`) sind **optional** → Skip statt Peer-Abort.

**Warum der Log „leer“ wirkt:**

`{err}` ist **nicht** leer. `Display` für `RxError` schreibt nur `message` (`src/core/rxdb/src/rx_error.rs:67–73`). Diese Message wird in `new_rx_error` so gebaut (`rx_error.rs:112–114, 130–139`):

1. `tunnel_error_message(code)` → **beginnt mit `\n`** und enthält **keinen** Klartext der Fehlerbedeutung, nur  
   `RxDB Error-Code: <CODE>.` + Dev-Mode-Hinweis  
   (`src/core/rxdb/src/overwritable.rs:51–57`).
2. Plus URL-Hint und pretty-printed `Parameters:`.

Die **erste** Logzeile endet deshalb mit `(registration failed: ` und **sichtbar nichts** hinter dem Doppelpunkt; Code, URL und Hashes stehen in den **Folgezeilen**.

**Bestätigt im Live-Log** (nicht UTL2 als Haupttäter der 668 Skips):

| Code | Bedeutung (dev-mode message) | Anzahl der 668 Skips |
|------|------------------------------|----------------------|
| **DB6** | „another instance created this collection with a different schema“ | **652** |
| **SC39** | Primary key muss `maxLength` haben | **16** (nur frühe `sellify_*`) |

Beispiele aus dem Log:

- `user_threads` → `DB6`, `previousSchemaHash` ≠ `schemaHash` (Zeile ~12535 ff.)
- frühes `sellify_people` → `SC39` + Schema-Dump mit `"version": 0` (Zeile ~6839 ff.)

**UTL2:** kommt im Log vor, aber **nicht** als Ursache dieser 668 `skipping optional … registration failed`-Events (Parser: nur DB6/SC39). UTL2 ist der generische „Error messages not included“-Tunnel-Hinweis-Kontext / andere Fehlerpfade — die Vermutung „UTL2 ohne Meldung“ trifft die **Form** (Code ohne Klartext), nicht den **konkreten Code**.

**Wie man den echten Grund bekommt (ohne Rate zu raten):**

1. **Sofort aus dem bestehenden Log:** die Folgezeilen nach dem Skip lesen → `RxDB Error-Code: …` + `Parameters:` (`previousSchemaHash` / `schemaHash` / `collection` / `schema`).
2. **Besseres Logging (Runde 2):** statt nur `{err}` explizit z. B.  
   `err.code()`, `err.parameters()`, und bei DB6 die beiden Hashes in **einer** Zeile.  
   Optional: `tunnel_error_message` mit Dev-Mode-Map befüllen (upstream `error-messages.ts`: DB6/SC39-Texte).
3. **Variante:** `{:?}` / strukturiertes JSON loggen (`code`, `parameters`, `url`) statt `Display`.

---

### 2) Warum genau diese Collections?

#### A) `user_threads`, `user_thread_messages`, `browser_frames`, `browser_input_events` → **DB6 Schema-Hash-Drift auf derselben Schema-Version**

- Quelle der Schemas: **statischer Contract**  
  `src/core/business_os/business_os_schema_contract.json` via  
  `collection_creators()` → `business_os_schema()`  
  (`rxdb_peer.rs:13695–13708`, `14197–14234`).
- Registrierung schreibt Collection-Meta unter Key `{name}-{version}`  
  (`collection_name_primary`, `rx_database_internal_store.rs:477–478`).
- Bei existierendem Meta mit **anderem** `schemaHash` und **nicht** auto-repair-kompatiblem Schema → **DB6**  
  (`rx_database.rs:713–738`).

**Store-Zahlen (RxDB SQLite):**

| Collection | Meta v0 hash (prefix) | Meta v1 hash | Contract jetzt | Daten-Tabelle | Rows |
|------------|----------------------|--------------|----------------|---------------|------|
| `user_threads` | `5074a07e…` | `97a22660…` = Contract | **version 1** | `__user_threads__v1` | **11 179** |
| `user_thread_messages` | `27f6e6e6…` | `3e9ac54c…` = Contract | **version 1** | `__user_thread_messages__v1` | **10 940** |
| `browser_frames` | `3718321e…` | `9cf5482d…` = Contract | **version 1** | `__browser_frames__v1` | **15** |
| `browser_input_events` | `5733148e…` | `7435ba46…` = Contract | **version 1** | `__browser_input_events__v1` | **0** |

**Historische DB6-Proben (Peer registrierte damals offenbar noch v0-Schemas):**

| Collection | previousSchemaHash (Log) | schemaHash im Fail (Log) = heutiges stored v0 |
|------------|--------------------------|-----------------------------------------------|
| `user_threads` | `dc212b09…` | `5074a07e…` |
| `user_thread_messages` | `b0e34d15…` | `27f6e6e6…` |
| `browser_frames` | `89e1c139…` | `3718321e…` |
| `browser_input_events` | `dc797063…` | `5733148e…` |

**Zweiversions-Verdacht (teilweise bestätigt, aber präzisiert):**

- Ja: betroffene Collections haben **Meta + Tabellen für v0 und v1**.
- Nein: der Registrierungsfehler ist **nicht** „weil zwei Versionstabellen existieren“.  
  RxDB erlaubt mehrere Versionen nebeneinander (`name-0` vs `name-1`).  
  DB6 greift, wenn **dieselbe** Version (`name-N`) mit einem **anderen Hash** erneut registriert wird und die Form **nicht** additiv/kompatibel genug für Meta-Repair ist (`schemas_compatible_for_meta_repair`, `rx_database.rs:744–830`; Index-only-Drift wird repariert, echte Property-Form-Differenzen nicht).

**Referenz, die (heute) erfolgreich registriert:** z. B. `desktop_files`  
- Contract version **0**, Hash `5c8ea6ed…` = stored Meta  
- Tabelle `__desktop_files__v0`: **5 220** Rows  
- Keine DB6-History in den 668 Skips.

**Warum die Threads/Browser-Collections später wieder laufen:**  
Contract ist auf **version 1** gehoben; Peer registriert `user_threads-1` / … mit Hash = stored v1 → kein DB6 mehr.  
**Letzter Registrierungs-Skip insgesamt: Logzeile 24819.** Aktueller Peer meldet **195** Collections (`multiplexed WebRTC replication up for 195 collections`).

#### B) `sellify_people|companies|campaigns|activities` → **zwei Phasen**

**Phase 1 — SC39 (16×):** Schema ohne Primary-Key-`maxLength`  
- Enforcement: `src/core/rxdb/src/rx_schema.rs:62–74`  
- Upstream-Text: *„The primary key must have the maxLength attribute set.“*  
- Log zeigt Schema `"version": 0` ohne brauchbares maxLength-Setup zum Fail-Zeitpunkt.

**Phase 2 — DB6 (492× auf den vier Collections):** Hash-Mismatch auf **version 0**  
Beispiele letzter Fails (~24645–24819):

| Collection | previous (Log) | current (Log) = stored v0 heute |
|------------|----------------|----------------------------------|
| `sellify_people` | `5bc4bcb5…` | `3c8432ae…` |
| `sellify_companies` | `bf0f59d3…` | `d53fcedc…` |
| `sellify_campaigns` | `00223228…` | `02bd844a…` |
| `sellify_activities` | `af41b282…` | `ca1ec69d…` |

**Schema-Deklaration heute**  
`runtime/business-os/installed-modules/sellify/collections.schema.json`:

- `schema_format`: `ctox-business-os-module-collections-v1` ✓  
- `install_scope`: `installed`, `entry`: `installed-modules/sellify/...` ✓  
- Collections **version 0**, `id.maxLength` gesetzt (180/180/200/220)  
- **nicht** im statischen Contract (`sellify_* NOT IN business_os_schema_contract.json`) → kommen nur über `runtime_installed_module_collection_creators` (`rxdb_peer.rs:13687–13726`, `13892–13948`).

**Store-Zahlen Sellify (entscheidend):**

| Collection | v0 Rows | v1 Rows | business_records |
|------------|---------|---------|------------------|
| `sellify_people` | **0** | **60 639** | 60 639 |
| `sellify_companies` | **0** | **17 516** | 17 516 |
| `sellify_campaigns` | **0** | **86 549** | 86 549 |
| `sellify_activities` | **0** | **74 209** | 74 209 |

**Verdacht „zwei Versionstabellen“:**  
bestätigt als **Datenlage** — die echten CRM-Daten liegen in **`__v1`**, die Modul-Datei deklariert aber **`version: 0`**.  
Wenn der Peer v0 erfolgreich registriert, hängt er an **`ctox_business_os__sellify_*__v0` (leer)**, nicht an v1.  
Das ist eine **zweite, schwerere Betriebsfalle** jenseits des reinen DB6-Skips: Registrierung kann „grün“ sein und trotzdem die **leeren** Tabellen replizieren.

Auto-Repair: Index-Normalisierung (`_deleted`/`id`/`_meta.lwt` in Indexes) + `additionalProperties: false` (`fill_with_default_settings`, `rx_schema_helper.rs:195–196`) machen viele Hash-Differenzen **kompatibel** → Meta-Hash-Repair (`repair_collection_meta_schema_hash`) — erklärt, warum Sellify **heute** wieder in den 195 Creators/Collections landet (8 Sellify-Collections im Runtime-Extra-Set), obwohl die Datei optisch von stored Meta abweicht.

#### C) `business_chats` → **DB6**, 3× früh

- Contract + stored Meta jetzt identisch: version **0**, Hash `0e52de33…`  
- business_records: **210**  
- Früher prev `4f7fc2d2…` vs curr `0e52de33…` → später aligned; seither kein Skip mehr.

#### D) Aktuelle Creator-Bilanz (erklärt „195 Collections“)

- Statischer Contract: **178** Collections  
- Runtime installed-modules akzeptiert: **+17** (8× sellify + bench/evidence/office-equipment)  
- **Summe 195** = letzte stabile Peer-Meldung  
- Ein Ausreißer `178` (Log ~556312) während Reconfigure; sofort wieder 195.

---

### 3) Was heißt `unsupported` bei den rem-Modulen (und Sellify historisch)?

**Logtext:**  
`skipping installed module schema …/collections.schema.json: unsupported schema_format`

**Parser / Ablehnung:** `src/core/business_os/rxdb_peer.rs:13940–13947`

```rust
if schema_doc.get("schema_format").and_then(Value::as_str)
    != Some("ctox-business-os-module-collections-v1")
{
    eprintln!(
        "[business-os] skipping installed module schema {}: unsupported schema_format",
        schema_path.display()
    );
    return Vec::new();
}
```

**Feld:** Top-Level-Key **`schema_format`**  
**Erforderlicher Wert:** exakt `ctox-business-os-module-collections-v1`  
**Ablehnende Zeile:** `rxdb_peer.rs:13940–13947` (Vergleich + Log + early return)

**Live-Dateien:**

| Modul | `schema_format` im File | Log-Skips (unsupported) |
|-------|-------------------------|-------------------------|
| rem-foerdervorhaben-agent | **fehlt** (`None`) | 22 |
| rem-dsgvo-document-writer | **fehlt** | 22 |
| rem-vertriebsmanagement | **fehlt** | 22 |
| rem-fozu-checker | **fehlt** | 22 |
| rem-foerder-explorer | **fehlt** | 22 |
| sellify | heute **korrekt** gesetzt | 2 (nur historisch, Zeile ~6809/6824) |

Keys der rem-Files: nur `collections`, `migration_strategies` — **kein** `schema_format`, oft auch kein `module_id`.

**Netto-Collection-Effekt der rem-Skips: ~0**  
Alle fünf rem-Module deklarieren nur `collections: ["business_commands"]`.  
`business_commands` ist bereits im **statischen Contract** und **required** → würde ohnehin nicht als Runtime-Extra gezählt (`static_collections.contains_key` in `runtime_module_collection_entries_for_root`, ~13777).

---

## folge_fuer_den_nutzer

### Nicht „komplett harmlos“, aber **optional by design** und **teilweise historisch**

Optionalität ist absichtlich (`FIX 4`, Kommentar `rxdb_peer.rs:2416–2419`, `14305–14311`): der Peer bleibt für required Collections (u. a. `business_commands`, `desktop_files`, …) oben.  
**Aber** die optionalen Skips haben echte Feature-Lücken erzeugt, solange sie andauerten.

| Collection | Modul | Nutzer-Folge bei fehlender Registrierung |
|------------|-------|------------------------------------------|
| `user_threads` | **Threads** (`modules/threads`) | Native RxDB-Replikation der Thread-Liste fehlt; Projektion Store→RxDB kann nicht schreiben. 1× Log: projection upsert user_threads failed. |
| `user_thread_messages` | **Threads** | **860×** `user_thread_messages collection is not registered` im Projektions-Sync — Thread-**Nachrichten** kamen nicht in den native RxDB-Peer → Browser-Sync/Offline für Messages defekt. Letzte solche Meldung: Logzeile **13653** (danach still). Daten im Store/RxDB-Tabelle existieren weiter (10 940). |
| `browser_frames` | **Browser** | Frame-Stream/Remote-View (GC, Lease, Projectoren in `rxdb_peer.rs` ~6386+) bricht ab, solange Collection fehlt. |
| `browser_input_events` | **Browser** | Remote-Input-Replay/GC nicht möglich. |
| `business_chats` | **Web Research / Support / CTOX / CV Print Builder** | Chat-Tracking-Projektion/Replikation (nur 3 frühe Fails; 210 Records im Store). |
| `sellify_*` (people/companies/campaigns/activities) | **Sellify** (installed) | Keine native WebRTC-Replikation der CRM-Collections während Skip-Phase. **Zusätzlich:** selbst bei v0-Registrierung hängen **239 913** CRM-Rows in **v1**-Tabellen, Modul deklariert **v0** → Peer/Browser können an **leeren v0-Tabellen** hängen (Sync „leer“ trotz voller Store-Projektion). |
| rem-* Module | rem-Förder/DSGVO/Vertrieb/… | `unsupported schema_format` → keine zusätzlichen Collections. Da nur `business_commands` deklariert ist (bereits vorhanden): **kein** eigener Sync-Ausfall der Module über diesen Pfad; eher Schema-Vertragsbruch / zukünftige Custom-Collections würden unsichtbar bleiben. |

### Ist der Zustand „jetzt“ harmlos?

- **Registrierungs-Skips:** seit Logzeile 24819 **keine** mehr; Peer bei **195** Collections → die ehemals skippenden Contract-Collections (Threads/Browser/Chats) und Sellify-Creators sind **wieder im Bring-up**.
- **Nicht harmlos ohne Prüfung:** Sellify **version 0 vs. Daten in v1** (leere v0-Tabellen). Das ist der Haupt-Folge-Risikopunkt **nach** erfolgreicher Registrierung.
- **rem-`schema_format`:** weiterhin broken, aber für aktuelle Collection-Menge praktisch folgenlos.
- Disk-Full-Nebenbefund im Log (`No space left on device` auf Peer-Status) ist **außerhalb** F-02, kann aber Reconfigure/Status stören.

---

## pfade (was Runde 2 bräuchte)

1. **Logging (klein, hoher Nutzen)**  
   - `rxdb_peer.rs:2432–2435`: `code` + kompakte Parameters in **einer** Zeile loggen (`err.code()`, bei DB6 beide Hashes, bei SC39 `primaryKey`/maxLength-Hinweis).  
   - Optional Dev-Mode-Tunnel für SC39/DB6-Texte.

2. **Sellify Version/Daten-Alignment (kritisch)**  
   - Entweder Modul-`collections.schema.json` auf **version 1** + Schema-Hash der stored v1 bringen,  
   - oder Daten/Meta von v1→v0 migrieren (unwahrscheinlich sinnvoll bei 239k Rows),  
   - plus Smoke: native Peer `collection('sellify_people').count()` == business_records / v1-Tabelle.

3. **DB6-Prävention für Contract-Collections**  
   - Bei Version-Bump: sicherstellen, dass Peer **nie** altes v0-Schema gegen mutierte v0-Meta knallt, oder stale v0-Meta/Tabellen gezielt via bestehendem `repair_optional_rxdb_collection_schema_drift` / version-invalidation räumen.  
   - Hash-Parity: Contract-Hash == stored Meta der **aktiven** Version (Threads/Browser sind hier bereits grün auf v1).

4. **rem-Module**  
   - In allen fünf `installed-modules/rem-*/collections.schema.json` Top-Level setzen:  
     `"schema_format": "ctox-business-os-module-collections-v1"`  
     (und ideal `module_id`).  
   - Quelle im Tree: `src/apps/business-os/installed-modules/rem-*/collections.schema.json` (dort ebenfalls fehlend).

5. **Verifikation Runde 2 (messen, nicht raten)**  
   - Nach Fix: Peer-Start-Log ohne `registration failed` / ohne `unsupported schema_format` für Sellify.  
   - Zähler: `business-os-rxdb.sqlite3` Row-Counts v0 vs v1 pro Sellify-Collection.  
   - Projection: keine neuen `user_thread_messages collection is not registered`.  
   - `business-os-rxdb-peer.status.json`: `running`, Collection-Count ≥ 195, Replication-Signale.

6. **Explizit nicht Runde 1:** keine Code-Änderungen, keine cargo-Läufe, kein Commit (Auftrag erfüllt).

---

## Kurzfazit

| Frage | Antwort |
|-------|---------|
| Leerer `{err}`? | Nein — mehrzeilige RxDB-Code-Message ohne Klartext (`Display` + `tunnel_error_message`); Codes **DB6** (652) / **SC39** (16). |
| Warum diese Collections? | Schema-Hash-Drift (DB6) auf gleicher Version; Sellify zusätzlich historisch SC39; Daten oft in **v1**, Registrierungsversuch **v0**. |
| `unsupported`? | Fehlendes Feld **`schema_format`** ≠ `ctox-business-os-module-collections-v1` an `rxdb_peer.rs:13940–13947`. |
| Harmlos? | Optionalität schützt den Peer, aber Threads-Messages-Projektion und Sellify-Sync waren/sind **nicht** harmlos; heute Skips weg, Sellify **v0-vs-v1** bleibt der offene Betriebsbefund. |

