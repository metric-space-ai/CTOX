## Formentscheidung

`TurnRuntimeErrorClass` bleibt als **Ein-Varianten-Enum** mit ausschließlich `StreamDisconnected` bestehen.

Begründung:

- Die typisierte Naht zwischen Direct Session, Turn Loop und Service bleibt sichtbar.
- Bestehende Konstruktoren, Re-Exports und Signaturen müssen nicht unnötig auf einen Marker umgebaut werden.
- Die Form bildet die tatsächliche Produktionssemantik ehrlich ab: Am Protokollrand kollabieren Stream-Disconnect und Incomplete-Responses bewusst auf eine Recovery-Klasse.
- Die genauere Herkunft bleibt weiterhin strukturiert im Harness erhalten.

Geändert wurden ausschließlich:

- `src/core/execution/agent/direct_session.rs`
- `src/core/service/service.rs`

`turn_loop.rs` und sämtliche Harness-Dateien blieben unverändert.

Zusätzlich:

- Gate-Match auf `StreamDisconnected` vereinfacht und den bewussten Protokollkollaps kommentiert.
- `cv_print_filename_from_prompt` entfernt.
- Direkter Aufruf von `prompt_line_value(prompt, "- filename:")` wiederhergestellt.
- Da eine vorbestehende dirty Änderung die in `HEAD` vorhandene `prompt_line_value`-Definition entfernt hatte, wurde diese Definition ebenfalls wiederhergestellt. Damit entsteht keine undefinierte oder dirty-only Referenz.

## Testliste vorher/nachher

### Vorher vorhandene Service-Gate-Tests

- `cv_print_typed_output_token_limit_allows_compact_recovery`
- `cv_print_typed_incomplete_response_allows_compact_recovery`
- `cv_print_typed_stream_disconnect_allows_compact_recovery`
- `cv_print_user_content_substring_does_not_allow_compact_recovery`

### Nachher

Entfernt:

- `cv_print_typed_output_token_limit_allows_compact_recovery`
- `cv_print_typed_incomplete_response_allows_compact_recovery`

Erhalten und grün:

- `cv_print_typed_stream_disconnect_allows_compact_recovery`
- `cv_print_user_content_substring_does_not_allow_compact_recovery`

Die vollständige CV-Testgruppe lief nach der Änderung mit:

- **12 bestanden**
- **0 fehlgeschlagen**
- **0 ignoriert**

Alle fünf Harness-Ursprungstests sind weiterhin vorhanden:

- `incomplete_max_output_tokens_preserves_typed_reason_and_display`
- `incomplete_other_reason_preserves_typed_reason_and_display`
- `map_api_error_preserves_incomplete_response_reason_and_display`
- `stream_disconnect_maps_to_typed_protocol_error_without_changing_display`
- `incomplete_response_maps_to_typed_protocol_error_without_changing_display`

Weitere Prüfungen:

- `cargo fmt --check`: **grün**
- `git diff --check` für beide geschriebenen Rust-Dateien: **grün**
- Symbolaudit: keine Referenzen auf tote Varianten oder `cv_print_filename_from_prompt`
- Ressourcen-Gate: 5-Minuten-Load 8,76; 338 GiB frei auf `/Volumes/tmp`; 21 GiB auf dem Repo-Volume

## Offene Bedenken

- Der Lauf `cargo test --bin ctox service::service_loop -- --test-threads=4` wartete zunächst hinter mehreren Cargo-Läufen auf den gemeinsamen Artifact-Lock und blieb anschließend beim erneuten `ctox-core`-Build. Er wurde an der 45-Minuten-Grenze beendet, bevor Tests starteten. Die erwartete bekannte Fehlermenge von fünf konnte daher nicht erneut bestätigt werden.
- Der vor Änderungen gestartete CV-Lauf benötigte 33 Minuten für den Kaltbau und las beim eigentlichen Kompilieren bereits den geänderten Arbeitsbaum. Er ist deshalb ein gültiger Nachher-Beleg, aber keine belastbare Vorher-Messung. Die Vorher-Liste wurde stattdessen aus dem vorhandenen Quellbaum dokumentiert.
- Der Arbeitsbaum enthält umfangreiche vorbestehende Änderungen. Insbesondere kann die wiederhergestellte `prompt_line_value`-Definition bei der späteren Montage mit der parallelen Metadata-Refaktorierung überlappen; semantisch ist sie für den geforderten direkten Aufruf erforderlich.
- Kein `git add`, kein Commit und keine Harness-Änderung.

## Workjet-Completion-Receipt v1

```yaml
workjet_completion_receipt: v1
role: implementation_agent
work_id: I-074b
status: implementation_complete_acceptance_partial
verdict: typed_class_collapse_honest_service_loop_rerun_blocked_by_deadline
elapsed_bound: stopped_at_45_minutes
form_decision:
  representation: single_variant_enum
  remaining_variant: StreamDisconnected
  rationale: "Preserves the typed Direct-Session/Turn-Loop/Service seam with minimal signature changes while honestly representing the intentional protocol-boundary collapse."
implementation:
  dead_variants_removed:
    - OutputTokenLimit
    - IncompleteResponse
  synthetic_tests_removed:
    - cv_print_typed_output_token_limit_allows_compact_recovery
    - cv_print_typed_incomplete_response_allows_compact_recovery
  retained_service_tests:
    - cv_print_typed_stream_disconnect_allows_compact_recovery
    - cv_print_user_content_substring_does_not_allow_compact_recovery
  gate_comment_added: true
  filename_drive_by_reverted: true
  harness_origin_typing_preserved: true
files_written:
  - src/core/execution/agent/direct_session.rs
  - src/core/service/service.rs
files_not_written:
  - src/core/execution/agent/turn_loop.rs
  - src/core/harness
progress_log: /Volumes/tmp/ctox-pipeline/i074b-fortschritt.md
verification:
  resource_gate:
    five_minute_load: 8.76
    volumes_tmp_free_gib: 338
    repo_volume_free_gib: 21
    passed: true
  cv_test_group:
    command: "CARGO_TARGET_DIR=/Volumes/tmp/ctox-pipeline/i070-target CARGO_INCREMENTAL=0 cargo test --bin ctox cv_print -- --test-threads=4"
    passed: 12
    failed: 0
    status: passed
  service_loop:
    command: "CARGO_TARGET_DIR=/Volumes/tmp/ctox-pipeline/i070-target CARGO_INCREMENTAL=0 cargo test --bin ctox service::service_loop -- --test-threads=4"
    status: stopped_before_tests_at_deadline
    expected_known_failures_reconfirmed: false
  cargo_fmt_check: passed
  diff_check: passed
  dead_symbol_audit: passed
  harness_origin_test_audit: passed
git:
  add_performed: false
  commit_created: false
open_concerns:
  - "The service-loop suite did not reach test execution before the 45-minute deadline."
  - "The long-running pre-change CV invocation compiled the changed source and therefore cannot serve as a reliable before measurement."
  - "The restored HEAD-defined prompt_line_value helper overlaps structurally with a pre-existing dirty metadata refactor but avoids an undefined or dirty-only reference."
```
