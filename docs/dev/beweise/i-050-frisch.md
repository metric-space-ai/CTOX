# I-050 Report

## was_geaendert

- `QueueTaskView.metadata` (bereits im vorbereiteten Worktree vorhanden) wird in `QueuedPrompt.metadata` weitergetragen (`src/core/service/service.rs:996`, `:18969`). Queue-abgeleitete Jobs kopieren jetzt die Zeilenmetadaten; rein synthetische TUI-/Guard-Jobs verwenden bewusst `Value::Null`.
- Alle App-Task-Erkennungen lesen `business_os_command_type` und die Modulidentitaet aus typisierten Metadaten. Die Funktion heisst nun `business_os_app_module_target_from_metadata` (`src/core/service/service.rs:10607`); die bisherigen 26 promptbasierten Aufrufpfade wurden auf `QueueTaskView.metadata`, `QueuedPrompt.metadata` oder das geparste `metadata_json` umgestellt.
- Die Modul-ID wird in der Reihenfolge `business_os_record_id`, `client_context.module_id`, `client_context.app_id`, `business_os_module` gelesen. Das ist notwendig, weil der Produktionspfad bei Creator-Kommandos `business_os_module = "creator"` und die eigentliche Ziel-App in `business_os_record_id` schreibt.
- Der RxDB-Lease-Check sucht nicht mehr mit `LIKE` im Prompt, sondern prueft das typisierte Projektionsfeld `command_type` auf `ctox.business_os.app.create|modify` (`src/core/service/service.rs:18155`).
- `QueuedPrompt`-Testfixtures und queue-abgeleitete Testjobs tragen Metadaten. Die zwei explizit genannten Fixtures behalten ihre Prompt-Prosa unveraendert und schreiben zusaetzlich die drei typisierten Schluessel (`src/core/install/mod.rs:4837`, `src/core/service/business_os.rs:8542`).
- Zusaetzlicher Roundtrip-Test belegt, dass `QueueTaskView` die Metadaten der Queue-Zeile ausliefert (`src/core/mission/channels/mod.rs:7186`).

## ursache_belegt

- Der Produktions-Anlagepfad schreibt bereits `business_os_module`, `business_os_command_type` und `business_os_record_id` (`src/core/business_os/store.rs:24754-24764`).
- Vorher verlor das Lesemodell diese Werte zwischen Queue-Zeile und Service-Job; der Konsument musste deshalb die Prompt-Prosa durchsuchen.
- Jetzt reicht `QueueTaskView` die Zeilenmetadaten durch, `QueuedPrompt` behaelt sie ueber Lease-, Routing-, Ticket- und Recovery-Pfade, und der App-Konsument entscheidet anhand `business_os_command_type` statt anhand von Prompt-Markern.

## kompensationen_geloescht

- `business_os_app_module_target_from_prompt` wurde entfernt und durch `business_os_app_module_target_from_metadata` ersetzt.
- Die App-spezifische Suche nach `Business OS app task metadata:`, `Business OS app resource context:` und `ctox.business_os.app.*` wurde aus der Erkennung entfernt.
- Der generische Helper `prompt_line_value` wurde entfernt. Die zwei unabhaengigen CV-Print-Nutzungen bleiben als fachlich benannter `cv_print_prompt_line_value`; App-Code verwendet ihn nicht.
- Der RxDB-Fallback parst keine Prompt-Prosa mehr.

## verblieben

- `install_target` und `artifact_directory` sind in der kanonischen Queue-Metadatenzeile nicht typisiert. Aus den vorhandenen drei Schluesseln ist nur der aktuelle Standardpfad sicher nachbildbar: App-Create/Modify wird wie im Producer-Default als `runtime-installed-module` unter `runtime/business-os/installed-modules/<module_id>` behandelt.
- Ein explizites Nicht-Default-/Source-Kommando kann aus `business_os_module`, `business_os_command_type` und `business_os_record_id` allein nicht von diesem Standard unterschieden werden. Dafuer muessen kuenftig im Produktionswriter typisierte Felder wie Installationsmodus und Artefaktpfad mitgeschrieben werden; ich habe sie nicht aus der Prompt-Prosa zurueckgeholt und keine Datei ausserhalb der Whitelist geaendert.

## tests

Alle Cargo-Aufrufe verwendeten `CARGO_TARGET_DIR=/Volumes/tmp/ctox-pipeline-targets/I-050`.

- `cargo fmt --check` — erfolgreich; kein Testbefehl, daher keine `test result`-Zeile/Trefferzahl.
- `cargo check --bin ctox` — erfolgreich: `Finished dev profile ...`; kein Testbefehl, daher keine `test result`-Zeile/Trefferzahl. Es blieben 397 bestehende Compiler-Warnungen.
- `cargo test --bin ctox business_os_app_module_target`
  - Treffer: 2 Tests.
  - `test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 2732 filtered out; finished in 0.00s`
- `cargo test --bin ctox queue_task_metadata`
  - Treffer: 1 Test.
  - `test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 2733 filtered out; finished in 0.07s`
- Zusatz, Fixture `install/mod.rs`: `cargo test --bin ctox release_switch_stop_is_guarded_for_business_os_app_tasks`
  - Treffer: 1 Test.
  - `test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 2733 filtered out; finished in 0.03s`
- Zusatz, Fixture `service/business_os.rs`: `cargo test --bin ctox app_validate_success_does_not_finalize_matching_leased_creator_task`
  - Treffer: 1 Test.
  - `test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 2733 filtered out; finished in 0.10s`

Kein verbotener Filter und kein Nulltreffer wurde verwendet.

## gegenprobe

- Der neue Test `business_os_app_module_target_reads_metadata_without_prompt_markers` verwendet den Prompt `Improve the contracts workflow without changing its task identity.`; er enthaelt weder `Business OS app task metadata:` noch `ctox.business_os.app.modify`, waehrend die typisierten Metadaten gesetzt sind.
- Fuer die Gegenprobe wurde die Zielauflosung temporaer auf die alte Prompt-Suche und das alte Zeilenparsen zurueckgestellt. Der Test lief und wurde rot (kein Compile-Fehler):
  - `test service::service_loop::tests::business_os_app_module_target_reads_metadata_without_prompt_markers ... FAILED`
  - `test result: FAILED. 0 passed; 2 failed; 0 ignored; 0 measured; 2732 filtered out; finished in 0.08s`
- Danach wurde `src/core/service/service.rs` bytegenau wiederhergestellt. SHA-256 vor/nach Rueckbau: `bd8a58f57a8af3bbff442445fb3229019eaf54e35e93a670e8ae5ec5ce4ea22c`.
- `git diff --stat` nach Rueckbau:
  - `src/core/business_os/store_projections.rs | 1 +`
  - `src/core/install/mod.rs | 6 +-
  - `src/core/mission/channels/mod.rs | 47 +++`
  - `src/core/service/business_os.rs | 6 +-
  - `src/core/service/service.rs | 461 ++++++++++++++++++++++--------`
  - `5 files changed, 407 insertions(+), 114 deletions(-)`
- Anschliessend wurden die gruenen Akzeptanztests und der finale Check erneut ausgefuehrt.

## offene_bedenken

- Der oben genannte Source-/Nicht-Default-Installationsmodus bleibt ohne zusaetzliche typisierte Writer-Felder mehrdeutig. Das ist kein verbleibender Prompt-Parser, aber eine Datenvertragsluecke.
- Waehrend der Verifikation lief das temporaere Dateisystem einmal voll. Nach Entfernen ausschliesslich des Cargo-Incremental-Caches unter `/Volumes/tmp/ctox-pipeline-targets/I-050/debug/incremental` wurden alle final gemeldeten Befehle erneut erfolgreich ausgefuehrt; der Repository-Arbeitsbaum blieb auf die Whitelist beschraenkt.

## pfade

Geaendert, ausschliesslich Whitelist:

- `src/core/mission/channels/mod.rs`
- `src/core/install/mod.rs`
- `src/core/service/business_os.rs`
- `src/core/service/service.rs`
- `src/core/business_os/store_projections.rs`

Fuer das unter `verblieben` beschriebene vollstaendige Typisieren waere in einer weiteren Welle ausserhalb dieser Whitelist notwendig:

- `src/core/business_os/store.rs:24754-24764` — `install_target` und der daraus bestimmte `artifact_directory` zusaetzlich in `extra_metadata` schreiben.
