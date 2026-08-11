## Ergebnis

Der Demand-Chunk-Streaming-Belang wurde als reiner Umzug aus

- `src/core/business_os/rxdb_peer.rs`

nach

- `src/core/business_os/rxdb_peer_demand_files.rs`

ausgelagert.

Der Vorbehalt trat nicht ein: Alle benötigten geteilten Helfer waren über minimale `pub(super)`-Grenzen nutzbar. Es war keine Änderung am Verhalten oder am Replikationskern erforderlich.

Geänderte Repository-Dateien, ausschließlich innerhalb der Whitelist:

- `src/core/business_os/rxdb_peer.rs`
- `src/core/business_os/rxdb_peer_demand_files.rs` — neu, derzeit erwartungsgemäß untracked
- `src/core/business_os/mod.rs`
- `contracts/module_size_budget.txt`

Kein `git add` und kein Commit wurden ausgeführt.

## Schnittkarte

### Bewegt

Vier Funktionsdefinitionen:

1. `demand_file_source_configs`
2. `register_demand_file_sources`
3. `stream_demand_file_chunks`
4. `stream_demand_file_chunks_inner`

Drei zugehörige Registry-/Konfigurationssymbole:

- `DemandFileChunkCollection`
- `DEMAND_FILE_CHUNK_COLLECTIONS`
- `DemandFileSourceConfig`

Die Board-Angabe „~8 Funktionen“ war damit eine Näherung. Der tatsächlich zusammenhängende historische Block enthält vier Funktionen sowie die drei Registry-Datensymbole.

Das neue Modul verwendet ausschließlich explizite Imports; kein `use super::*`.

### Bewusst in `rxdb_peer.rs` verblieben

| Symbol/Gruppe | Grund |
|---|---|
| `runtime_module_demand_chunk_sources` | Gehört weiterhin zur Runtime-Modul- und Collection-Schema-Ermittlung; wird vom neuen Registry-Modul nur konsumiert. |
| `demand_file_chunk_rows_for_key_from_sqlite` | Generischer SQLite-Chunk-Helfer, der auch vom Desktop-Datei-Modul verwendet wird. |
| `demand_file_source_error` | Gemeinsame Fehlerkonstruktion für Demand- und Desktop-Datei-Zugriffe. |
| `DemandFileFetchRequestStats` | Wird sowohl vom Streaming als auch vom Desktop-Datei-Modul befüllt. |
| `DemandFileFetchMetrics` und `DEMAND_FILE_FETCH_METRICS` | Der Native-Peer-Performance-Snapshot bleibt Eigentümer der Metrik. |
| `WebRtcPool` | Zentrale Typalias-Grenze des Replikationskerns. |
| `active_desktop_file_chunk_rows_from_sqlite` | Bleibt im bereits extrahierten Desktop-Datei-Modul; enthält die aktive Generation und Range-Auswahl. |
| SQLite-, Tabellen- und Chunk-ID-Helfer | Werden von mehreren Peer-Belangen geteilt und sind nicht exklusiv Demand-Streaming. |
| Demand-Chunk-Schema-Validierung | Bleibt bei der Runtime-Collection-Registrierung, wo ungültige Schemas fail-closed abgewiesen werden. |

Die erforderlichen Querzugriffe erhielten nur minimale `pub(super)`-Sichtbarkeit. Es wurden keine neuen Funktions-, Static- oder Const-Namen eingeführt.

## Pflicht-Checks

### Funktionsnamens-Schnittmenge

Eindeutige definierte Funktionsnamen einschließlich Tests und Methoden:

- `rxdb_peer.rs`: **495**
- `rxdb_peer_demand_files.rs`: **4**
- Schnittmenge: **0**

Damit gibt es weder Copy-statt-Move noch doppelte Implementierungen.

### Dirty-only-Symbole

Verglichen wurden alle `fn`-, `static`- und `const`-Namen der neuen Vereinigung mit `HEAD`:

- Dirty-only-Symbole: **0**

### Zeilenbilanz

Physische Zeilen:

- vorher `rxdb_peer.rs`: **20.815**
- nachher `rxdb_peer.rs`: **20.601**
- neues Demand-Modul: **237**
- neue Vereinigung: **20.838**
- Abweichung: **+23 Zeilen**

Produktionszeilen nach der Modulgrößen-Vertragsregel:

- vorher: **10.875**
- nachher Peer: **10.661**
- neues Demand-Modul: **237**
- neue Vereinigung: **10.898**
- Abweichung: **+23 Zeilen**

Die Abweichung liegt deutlich unter der 200-Zeilen-Schwelle und besteht aus Modulkopf, expliziten Imports, Reexport und Sichtbarkeitsgrenzen.

### Normalisierte Move-SHA

Geprüft wurden die vier nach Namen sortierten Funktionsspans. Normalisiert wurden:

- Whitespace
- notwendige `pub(super)`-Zusätze
- das von Rustfmt ergänzte abschließende Parameterkomma

Ergebnis:

```text
Original: f0cce65044a49d06414a12add897e3bd49ad19ce6827e45d08a6a3dfc9d722c3
Verschoben: f0cce65044a49d06414a12add897e3bd49ad19ce6827e45d08a6a3dfc9d722c3
```

Die normalisierten Inhalte sind **identisch**.

## Modulgrößen-Budgets

Vorher:

```text
src/core/business_os/rxdb_peer.rs = 10875
src/core/business_os/rxdb_peer_desktop_files.rs = 1908
```

Nachher:

```text
src/core/business_os/rxdb_peer.rs = 10661
src/core/business_os/rxdb_peer_demand_files.rs = 237
src/core/business_os/rxdb_peer_desktop_files.rs = 1908
```

Beide neuen Zielwerte stimmen exakt mit den gemessenen Produktionszeilen überein.

## `include_str!`-Wächter

Es existiert genau ein relevanter Wächter:

```rust
include_str!("rxdb_peer.rs")
```

Er untersucht ausschließlich `async fn *_loop` auf erlaubte Idle-Strategien. Keine der verschobenen Funktionen ist ein solcher Loop. Eine Erweiterung auf `rxdb_peer_demand_files.rs` war daher nicht erforderlich; Guard-Testdateien wurden nicht verändert.

## Verifikation

### Ressourcen-Gate

- Anfangs lag der 5-Minuten-Load über dem Grenzwert.
- Gewartet bis: **17,19**
- Freier Platz beim Gate: **339.275.420 KiB**
- 5-GiB-Hard-Stop wurde nicht erreicht.

### Übernommene Vollsuite-Baseline

Wie verlangt nicht neu gemessen:

- **143 bestanden**
- **3 bekannte rote Tests**
  - `native_peer_consumes_pending_business_command_written_directly_to_sqlite`
  - `native_peer_consumes_pending_module_governance_commands`
  - `sync_business_record_projections_materializes_procedural_knowledge`

### `cargo check --bin ctox`

- Exit: **0**
- Ergebnis: **grün**
- 449 bestehende Workspace-Warnungen, keine Compile-Fehler

### Demand-Filter

```text
cargo test --bin ctox demand -- --test-threads=2
```

Ergebnis:

- **9 bestanden**
- **0 fehlgeschlagen**
- **2.817 ausgefiltert**
- Laufzeit: **84,27 Sekunden**

Keine neuen roten Tests.

### Modulgrößen-Gate

```text
cargo test --bin ctox module_size -- --test-threads=4
```

Die beiden aufgabenbezogenen Einträge sind exakt. Der Test bleibt ausschließlich wegen der sechs bereits bekannten fremden OVERs rot:

- `office_engine.rs`: 14.598 / 13.953
- `store.rs`: 28.413 / 27.516
- `store_outbound_commands.rs`: 5.354 / 5.270
- `context/lcm/mod.rs`: 6.365 / 5.627
- `mission/channels/mod.rs`: 7.388 / 7.221
- `service/service.rs`: 26.789 / 26.237

Direkter Testlauf:

- **0 bestanden**
- **1 fehlgeschlagen**
- **2.825 ausgefiltert**

### Format und Diff

- Rustfmt-Check der berührten Dateien: **grün**
- `git diff --check`: **grün**

## Offene Bedenken

1. Das globale Modulgrößen-Gate bleibt wegen der sechs bekannten, außerhalb der Whitelist liegenden Fremdüberschreitungen rot. Die beiden S2b-2-Budgets sind exakt.
2. Das neue Modul konsumiert weiterhin geteilte Peer-Helfer für Runtime-Modul-Ermittlung, SQLite-Zugriff und Metriken. Diese Abhängigkeiten sind `pub(super)`-fähig und sauber kompiliert; eine weitere Extraktion wäre ein eigener, nicht mehr rein mechanischer Schnitt.
3. Die neue Datei ist mangels `git add` erwartungsgemäß untracked. Es wurde weder gestaged noch committed.

```text
WORKJET-COMPLETION-RECEIPT v1
task: S2b-2
outcome: completed_with_known_external_module_size_overs
moved_functions: 4
moved_registry_data_symbols: 3
name_intersection: 0
dirty_only_fn_static_const: 0
normalized_move_sha256: f0cce65044a49d06414a12add897e3bd49ad19ce6827e45d08a6a3dfc9d722c3
line_balance_physical: +23
line_balance_production: +23
peer_budget_before: 10875
peer_budget_after: 10661
demand_module_budget: 237
cargo_check: passed
demand_tests: 9_passed_0_failed_2817_filtered
module_size: target_entries_exact_6_known_external_overs
fmt_check: passed
diff_check: passed
full_suite_baseline: reused_143_passed_3_known_red
include_str_guard_change: not_required
git_staged: false
git_committed: false
```
