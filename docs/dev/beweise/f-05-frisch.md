# F-05 — Mehrfachversions-Bestandsaufnahme

Store (read-only): `~/.local/state/ctox/business-os-rxdb.sqlite3`
Abfragebasis: `sqlite3 'file:…?mode=ro' -readonly` / Python `mode=ro` + `PRAGMA query_only=ON`.
Gesamt: 367 Versionstabellen `ctox_business_os__*__v*`, 363 Collection-Namen, gesehene Versionen nur `{0,1}`.

## mehrfachversionen

Genau **4** Collections existieren in mehr als einer Version. Alle gehören zum Laufzeitmodul **sellify** (nicht im Contract).

| Collection | Versionen im Store | COUNT(*) je Version | Deklarierte Version | Quelle |
|---|---|---|---|---|
| `sellify_activities` | 0, 1 | v0=0 · v1=74209 | **0** | Laufzeitmodul `sellify` (`installed-modules/sellify/collections.schema.json`; state- und runtime-Pfad identisch) — **nicht** Contract |
| `sellify_campaigns` | 0, 1 | v0=0 · v1=86549 | **0** | Laufzeitmodul `sellify` (beide installierten Pfade) — **nicht** Contract |
| `sellify_companies` | 0, 1 | v0=0 · v1=17516 | **0** | Laufzeitmodul `sellify` (beide installierten Pfade) — **nicht** Contract |
| `sellify_people` | 0, 1 | v0=0 · v1=60639 | **0** | Laufzeitmodul `sellify` (beide installierten Pfade) — **nicht** Contract |

Nachweise:

- Tabellen: `ctox_business_os__sellify_{activities,campaigns,companies,people}__v{0,1}`
- Manifeste gelesen unter:
  - `~/.local/state/ctox/business-os/installed-modules/sellify/collections.schema.json`
  - `~/.local/lib/ctox/current/runtime/business-os/installed-modules/sellify/collections.schema.json`
- Beide Manifeste: `version: 0` für die vier Collections; `migration_strategies` enthält dennoch leere Strategien für Zielversion `"1"` (Rest einer v0→v1-Migration).
- Contract `src/core/business_os/business_os_schema_contract.json`: **0** Keys mit `sellify` (178 Kern-Collections, keine der vier Mehrfachversions-Collections).

Vollscan der 363 Collection-Namen im Store: keine weiteren Collections mit ≥2 Versionstabellen. Keine v2/v3-Tabellen vorhanden.

## unerreichbare_daten

Fälle, in denen die deklarierte Version eine **leere** Tabelle trifft und eine andere Version Zeilen hält:

| Collection | Deklariert | Gebundene Tabelle | COUNT deklariert | Gefüllte andere Version | Unerreichbare Zeilen |
|---|---|---|---|---|---|
| `sellify_activities` | 0 | `__v0` | 0 | `__v1` = 74209 | **74209** |
| `sellify_campaigns` | 0 | `__v0` | 0 | `__v1` = 86549 | **86549** |
| `sellify_companies` | 0 | `__v0` | 0 | `__v1` = 17516 | **17516** |
| `sellify_people` | 0 | `__v0` | 0 | `__v1` = 60639 | **60639** |

**Außerhalb von Sellify: 0 Collections, 0 unerreichbare Zeilen.**

(Der bekannte Sellify-Schaden bestätigt sich exakt: 74209+86549+17516+60639 = **238913**.)

## unauffaellige_faelle

Mehrfachversionen, bei denen die Deklaration die **gefüllte** Tabelle trifft:

**keine.**

Unter den 4 Mehrfachversions-Collections ist kein Fall, in dem die deklarierte Version die datenhaltende Tabelle bindet. Mehrfachversionen sind in diesem Store also *in jedem* beobachteten Fall ein Defekt — nicht nur „historische Reste neben korrekt gebundener Version“.

Zur Einordnung (keine Mehrfachversion, daher nicht in §mehrfachversionen): diverse Collections existieren nur als `__v1` und sind deklariert als 1 (z. B. `business_commands`, `browser_*`, `matching_*`, `outbound_messages`, `sellify_records`) — das ist konsistent und kein Defekt dieses Musters.

## summe

- **Collections mit unerreichbaren Daten:** **4** (alle Sellify; außerhalb Sellify: **0**)
- **Unerreichbare Zeilen gesamt:** **238913**
- **Unerreichbare Zeilen außerhalb Sellify:** **0**

Antwort auf die Leitfrage „wie gross ist der Schaden ausserhalb von Sellify?“: **null** — der Mehrfachversions-/Leerbindung-Schaden ist im aktuellen Store vollständig auf die vier Sellify-CRM-Collections beschränkt.

## unsicherheiten

- Die „heute deklarierte Version“ wurde aus den installierten Modul-Manifesten und dem Contract gelesen, nicht aus dem laufenden Peer-Prozess-Speicher. Ein in-memory Override zur Laufzeit wurde nicht inspiziert (Daemon pid 82511 unangetastet). Die beiden Manifest-Pfade (state + runtime) sind für Sellify inhaltsgleich.
- Ob der Peer die leeren `__v0`-Tabellen tatsächlich öffnet, folgt aus der bereits belegten Bindungslogik (`expected_rxdb_collection_version` / Manifest-`version`); diese Dateien wurden hier nicht erneut gelesen.
- Collections, die **nur** in einer Version existieren, aber deklariert eine *andere* (nicht existierende) Version haben, wären ein anderes Defektmuster und wurden nicht systematisch über alle 363 Collections gegen den Contract abgeglichen — außer dem Scan aller installierten Modul-Manifeste, der außer Sellify keine `empty_declared_with_other_data`-Fälle zeigte.
- `sellify_records` ist deklariert v1, existiert nur als `__v1` mit COUNT=0; das ist kein Mehrfachversions-Fall und zählt nicht als unerreichbare Daten.
