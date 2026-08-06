# I-058 Report

## was_geaendert

- Ausschliesslich `src/core/business_os/rxdb_peer.rs` geaendert.
- Die Queue-Reparaturhaelfte wurde aus dem Projektionslauf entfernt. Der bestehende Sparschalter `reconcile_queue_chat_tracking_projections_if_changed` bleibt bestehen und ruft im neuen Stand nur noch `reconcile_business_chat_tracking_projections` auf (`src/core/business_os/rxdb_peer.rs:9598-9609`, Ziel: `:8412`).
- Der Fingerabdruck wurde auf die verbliebene Chat-Tracking-Reparatur zugeschnitten und intern entsprechend umbenannt (`ChatTrackingRepairProjectionStamp`, `:9492`; `rxdb_chat_tracking_repair_stamps`, `:9638`). Entfernt wurden der kanonische Queue-Anteil samt Core-DB-/WAL-/SHM-/Journal-Stempel und Routing-Status-Hash.
- Die RxDB-Anteile `ctox_queue_tasks`, `business_commands` und `business_chats` bleiben absichtlich im Fingerabdruck (`:9665-9667`): Die Chat-Reparatur liest Command- und Queue-Task-Projektionen, um aktive Chat-Nachrichten auf ihren Folgestatus zu setzen. Eine Queue-Task-Aenderung ist daher weiterhin eine relevante **Chat**-Quelle, nicht mehr ein Wecksignal fuer eine globale Queue-Reparatur.
- `queue_chat_repair_idle_gate_skips_unchanged_sources` wurde beibehalten und auf das verbliebene Verhalten umgestellt: unveraenderter Chat-Fingerabdruck wird uebersprungen; nach einer asynchron zurueckgebliebenen Chat-Tracking-Projektion laeuft die Chat-Reparatur; danach wird die unveraenderte Quelle wieder uebersprungen (`:20111`).

Finaler Diff:

```text
src/core/business_os/rxdb_peer.rs | 1039 ++++---------------------------------
1 file changed, 106 insertions(+), 933 deletions(-)
```

## ursache_belegt

- Vor jeder Loeschung selbst in den Live-Stores nachgemessen:
  - `/Users/michaelwelsch/.local/state/ctox/business-os-rxdb.sqlite3`, Tabelle `ctox_business_os__ctox_queue_tasks__v1`: Selektor `status IN ('queued','running','accepted')` = **0 von 942** nicht geloeschten Dokumenten.
  - `/Users/michaelwelsch/.local/state/ctox/business-os.sqlite3`, `business_records` fuer `ctox_queue_tasks`: derselbe Selektor = **0 von 942** nicht geloeschten Datensaetzen.
- Damit war die STOPP-Bedingung `>0` in keinem Store erfuellt.
- `d0d2d0ca87fe8c72262489c2aab6d25b99226f58` (`business_os: the channel paths refresh their own projection`) ist nachgemessen ein Vorfahr des aktuellen HEAD.
- Ausgangs-HEAD und `origin/main` waren identisch: jeweils `6a60631f8cf8ac9e6cdab6f34a38efb68c80cc90`.
- Die Null ist fuer die geloeschte Funktion relevant: Sie startete ausschliesslich bei bereits vorhandenen aktiven RxDB-Queue-Dokumenten. Bei null Selektortreffern hatte sie nichts mehr zu kompensieren.

## kompensationen_geloescht

Geloescht wurden:

- `reconcile_ctox_queue_task_projections` komplett.
- Nur von dieser Reparatur verwendete Status-/Orphan-Helfer:
  - `terminal_queue_status_for_command`
  - `route_status_for_queue_projection`
  - `queue_projection_status_for_route_status`
  - `projection_queue_task_is_orphaned`
  - `projection_document_age_ms`
- Nur vom Queue-Anteil des Fingerabdrucks verwendete Typen/Helfer:
  - `CanonicalQueueRepairStamp`
  - `SqliteProjectionFilesStamp`
  - `canonical_queue_repair_stamp`
  - `canonical_queue_routing_status_hash`
  - `empty_canonical_queue_repair_stamp`
  - `sqlite_projection_files_stamp`
  - `sqlite_sidecar_path`
- Die vier Tests, deren einziger Gegenstand die geloeschte Reparatur war:
  - `reconcile_ctox_queue_task_projections_completes_stale_completed_commands`
  - `reconcile_ctox_queue_task_projections_filters_to_active_queue_statuses`
  - `reconcile_ctox_queue_task_projections_does_not_run_global_queue_repair`
  - `reconcile_ctox_queue_task_projections_fails_orphaned_accepted_commands`

Aufruferzaehlung vor dem Loeschen:

- Jeder der fuenf kleinen Queue-Status-/Orphan-Helfer hatte genau **einen** Aufruf, jeweils innerhalb von `reconcile_ctox_queue_task_projections`.
- Die kanonische Queue-Stempelkette hatte ebenfalls nur den Queue-Reparatur-Fingerabdruck als Wurzelaufrufer.
- Geteilte Helfer wurden nicht geloescht. Insbesondere `find_rxdb_document_by_id` hat im neuen Stand weiterhin drei fremde Aufrufer (`:10265`, `:10271`, `:11414`) und bleibt bestehen (`:10181`).

## verblieben

- `reconcile_business_chat_tracking_projections` bleibt unveraendert als notwendige Kompensation bestehen (`:8412`). Der nicht-atomare Browserpfad fuer Chat-Tracking ist nicht Teil der Whitelist und laut Auftrag weiterhin real.
- Der Sparschalter `reconcile_queue_chat_tracking_projections_if_changed` bleibt bestehen (`:9598`).
- Im Chat-Fingerabdruck bleiben:
  - RxDB `business_chats` (zu reparierendes Dokument),
  - RxDB `business_commands` (Status-/Task-Aufloesung),
  - RxDB `ctox_queue_tasks` (Task-Status-Aufloesung fuer Chat-Nachrichten),
  - der 10-Minuten-Orphan-Epoch (zeitabhaengige Chat-Orphan-Erkennung).
- Keine Queue-Projektionsreparatur und kein kanonischer Queue-/Core-DB-Fingerabdruck verbleiben.

## tests

Alle Cargo-Aufrufe verwendeten ausnahmslos:

```text
CARGO_TARGET_DIR=/Volumes/tmp/ctox-pipeline-targets/I-058
```

### Baseline auf sauberem `origin/main`/HEAD

- `cargo fmt --check`: **PASS**. Kein `test result` vorhanden, weil der Befehl keine Tests ausfuehrt.
- `cargo check --bin ctox`: **PASS**, `Finished dev profile ... in 13m 05s`. Kein `test result` vorhanden, weil der Befehl keine Tests ausfuehrt.
- `cargo check --bin ctox --tests`: **PASS**, `Finished dev profile ... in 2m 17s`. Kein `test result` vorhanden, weil der Befehl Tests kompiliert, aber nicht ausfuehrt.
- `cargo test --bin ctox reconcile_business_chat_tracking`: Trefferzahl **3**.
  - `test result: FAILED. 2 passed; 1 failed; 0 ignored; 0 measured; 2733 filtered out; finished in 175.65s`
  - Rot: `business_os::rxdb_peer::tests::reconcile_business_chat_tracking_projections_batches_active_document_lookups` (`left: 3`, `right: 2`).
- `cargo test --bin ctox queue_chat_repair_idle_gate_skips_unchanged_sources`: Trefferzahl **1**.
  - `test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 2735 filtered out; finished in 77.42s`

### Formatierung waehrend der Aenderung

- Erster `cargo fmt --check` nach dem Edit: **FAIL**, ausschliesslich zwei rustfmt-Diffs im geaenderten Test; keine Tests und daher keine `test result`-Zeile.
- `cargo fmt`: **PASS**, genau eine Datei formatiert; keine Tests und daher keine `test result`-Zeile.

### Finaler Stand

- `cargo fmt --check`: **PASS**. Kein `test result`, da keine Tests ausgefuehrt werden.
- `cargo check --bin ctox`: **PASS**, `Finished dev profile ... in 1m 24s`. Kein `test result`, da keine Tests ausgefuehrt werden.
- `cargo check --bin ctox --tests`: **PASS**, `Finished dev profile ... in 2m 15s`. Kein `test result`, da Tests nur kompiliert werden.
- `cargo test --bin ctox reconcile_business_chat_tracking`: Trefferzahl **3** (passender Filter; nicht null).
  - `test result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 2729 filtered out; finished in 159.76s`
- `cargo test --bin ctox queue_chat_repair_idle_gate_skips_unchanged_sources`: Trefferzahl **1** (exakter verlangter Filter; nicht null).
  - `test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 2731 filtered out; finished in 60.04s`
- Zusaetzliche Eingrenzung des Baseline-Befunds: `cargo test --bin ctox reconcile_business_chat_tracking_projections_batches_active_document_lookups`, Trefferzahl **1**.
  - `test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 2731 filtered out; finished in 74.59s`

Die um vier gesunkene Gesamtzahl (Baseline 2736 Tests, final 2732 Tests, jeweils aus `running + filtered`) entspricht exakt den vier erlaubterweise geloeschten Queue-Reparaturtests.

## gegenprobe

1. **Keine Aufrufer geloeschter Funktionen:** Ein `rg` ueber den neuen Stand fuer die geloeschte Queue-Funktion, alle fuenf Queue-Helfer und die komplette kanonische Queue-Stempelkette lieferte keinen Treffer: `no deleted queue repair symbols or callers remain`.
2. **Chat-Haelfte laeuft weiter:**
   - Aufrufpfad im neuen Stand:
     - `reconcile_queue_chat_tracking_projections_if_changed` bei `src/core/business_os/rxdb_peer.rs:9598`
     - direkter Aufruf `reconcile_business_chat_tracking_projections(database)` bei `:9608`
     - Zieldefinition bei `:8412`
   - Der bestehende Chat-Testfilter traf drei Tests und war final vollstaendig gruen.
   - Der beibehaltene Idle-Gate-Test traf einen Test und war gruen; er prueft jetzt explizit den Chat-only-Pfad.
3. **Rot-Mengenvergleich gegen `origin/main`, beide Richtungen:**
   - Baseline-Rotmenge: `{reconcile_business_chat_tracking_projections_batches_active_document_lookups}`
   - Final-Rotmenge: `{}`
   - Neu rot (`final - baseline`): `{}`
   - Still gruen (`baseline - final`): `{reconcile_business_chat_tracking_projections_batches_active_document_lookups}` — **Befund**.
   - Der Befund ist nicht durch eine Verhaltensaenderung an der Chat-Funktion erklaert: Der Test benutzt einen globalen Atomic-Zaehler (`:466`), der in jedem Lookup inkrementiert wird (`:8722`), waehrend der breite Filter drei Chat-Tests parallel startet. Der Baseline-Lauf sah deshalb 3 statt 2. Der exakte Einzeltest ist final gruen. An diesem vorbestehenden, parallelitaetsabhaengigen Testzaehler wurde im Rahmen von I-058 nichts geaendert.
4. Es wurde keine kuenstliche Rot-Gegenprobe eingebaut; daher war kein temporaerer Sabotage-Edit zurueckzubauen. `git diff --check` ist sauber, und der finale `git diff --stat` ist oben belegt.

## offene_bedenken

- Der Test `reconcile_business_chat_tracking_projections_batches_active_document_lookups` ist unter dem breiten verlangten Filter parallelitaetsabhaengig, weil sein globaler Zaehler auch Lookups der zwei gleichzeitig laufenden Schwester-Tests erfasst. Das war auf sauberem `origin/main` rot und ist im finalen Lauf zufaellig gruen geworden. Der exakte Einzeltest ist stabil gruen; der Testzaehler selbst liegt in der Whitelist, ist aber nicht Gegenstand der Queue-Kompensationsloeschung und wurde nicht nebenbei veraendert.
- Keine weiteren offenen Bedenken fuer die Queue-Loeschung. Die vorgeschriebenen Compile-/Format-/Zieltests sind abgeschlossen.

## pfade

- Geaendert: `src/core/business_os/rxdb_peer.rs`.
- Zusaetzlich notwendige Dateien: **keine**.
- Ausserhalb der Hard Whitelist wurde nichts geaendert.
