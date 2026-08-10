## Ergebnis

Das CV-Recovery-Gate ist auf eine typisierte Fehlerklassifikation umgestellt. Die drei bisherigen Substring-Prüfungen auf `err_text` wurden aus `cv_print_parser_error_allows_compact_recovery` entfernt.

### Typ-Quelle und Datenfluss

Der strukturierte Ursprung liegt im Responses-SSE-Stack:

1. `src/core/harness/ctox-api/src/sse/responses.rs`
   - `response.incomplete` liest `incomplete_details.reason`.
   - `max_output_tokens` wird als `ResponseIncompleteReason::MaxOutputTokens` erhalten.
   - Andere Incomplete-Gründe werden als `ResponseIncompleteReason::Other(reason)` erhalten.

2. `src/core/harness/ctox-api/src/error.rs`
   - Neue minimale Klasse `ResponseIncompleteReason`.
   - `ApiError::ResponseIncomplete` transportiert Grund und unveränderten Meldungstext.

3. `src/core/harness/core/src/api_bridge.rs`
   - Klassen-erhaltende Projektion nach `CodexErr::ResponseIncomplete`.

4. `src/core/harness/core/src/error.rs`
   - `CodexErr::Stream` und `CodexErr::ResponseIncomplete` werden korrekt auf den bestehenden typisierten Protokollcode `CodexErrorInfo::ResponseStreamDisconnected` projiziert, statt als `Other` zu enden.
   - Bestehende Display-Texte bleiben unverändert.

5. `src/core/execution/agent/direct_session.rs`
   - `EventMsg::Error` wertet `codex_error_info` typisiert aus.
   - Ein `TurnRuntimeError` mit `TurnRuntimeErrorClass` wird durch `anyhow` weitergereicht; der bisherige Debug-/Display-Text bleibt erhalten.

6. `src/core/execution/agent/turn_loop.rs`
   - Reicht die Typen ohne neue Textanalyse bis zum Service weiter.

7. `src/core/service/service.rs`
   - Das Gate akzeptiert ausschließlich:
     - `OutputTokenLimit`
     - `IncompleteResponse`
     - `StreamDisconnected`
   - Beliebige Fehlermeldungen mit zufällig enthaltenen Schlüsselwörtern öffnen das Gate nicht mehr.

Am Protokollrand fallen die beiden strukturierten Incomplete-Unterklassen auf den bereits vorhandenen, für das Gate ausreichenden Typ `ResponseStreamDisconnected` zusammen. Die genaue `ResponseIncompleteReason` bleibt bis zu dieser Projektion strukturiert erhalten; am Service-Gate ist nur relevant, dass die Fehlerklasse Recovery-fähig ist.

### Unverändertes Verhalten

Die bestehenden Texte wurden durch Tests abgesichert:

- API: `stream error: Incomplete response returned, reason: …`
- Core: `stream disconnected before completion: Incomplete response returned, reason: …`
- Direct Session: bestehendes Präfix und bisherige Debug-Darstellung

Recovery-Semantik, Attempt-Logik und Telemetrie wurden nicht verändert.

`runtime_error_is_transient_api_failure` bleibt unverändert. Der Klassifikator arbeitet weiterhin auf einem größeren, eigenständigen Textregelsatz. Seine vollständige Typisierung wäre kein kostenloser Nebeneffekt dieses lokalen CV-Gates und wird daher vertagt.

## Beweistests

Neu und grün:

- `cv_print_typed_output_token_limit_allows_compact_recovery`
- `cv_print_typed_incomplete_response_allows_compact_recovery`
- `cv_print_typed_stream_disconnect_allows_compact_recovery`
- `cv_print_user_content_substring_does_not_allow_compact_recovery`
  - Negativbeweis: `der Nutzer schrieb 'max_output_tokens' in den CV`
- `incomplete_max_output_tokens_preserves_typed_reason_and_display`
- `incomplete_other_reason_preserves_typed_reason_and_display`
- `map_api_error_preserves_incomplete_response_reason_and_display`
- `stream_disconnect_maps_to_typed_protocol_error_without_changing_display`
- `incomplete_response_maps_to_typed_protocol_error_without_changing_display`

Ergebnisse:

- CV-Service-Testgruppe einschließlich bestehender Recovery-Tests: **14 bestanden, 0 fehlgeschlagen**
- Vollständige `ctox-api`-Suite: **73 bestanden, 0 fehlgeschlagen**
- Neue `ctox-core`-Typflusstests isoliert: **3 bestanden, 0 fehlgeschlagen**
- Vollständige `ctox-core`-Suite: **1843 bestanden, 23 fehlgeschlagen, 5 ignoriert**
  - Die roten Tests betreffen bestehende Config-, Guardian-, Seatbelt-, Skill- und Tool-Spec-Flächen, nicht den I-074-Typfluss.
- `cargo fmt --check`: bestanden
- Harness-Formatcheck: bestanden
- `git diff --check`: bestanden

## Vorher/Nachher

### Vorher

Der verpflichtende Baseline-Lauf entdeckte **386 Tests**, wurde jedoch nach bereits grünen Tests vom Betriebssystem per **SIGKILL** beendet. Es gab vor dem Kill keine Assertion-Fehlermeldung; deshalb liegt keine vollständige Pass/Fail-Zahl vor.

### Nachher

Der genaue Nachher-Lauf konnte nicht vollständig beendet werden:

1. Erster Versuch: Das vorgegebene `CARGO_TARGET_DIR` wurde während des Builds extern entfernt; Cargo scheiterte mit `rmeta: No such file or directory`.
2. Wiederholung: kompletter Neuaufbau des gelöschten Targets; beim Erreichen der Zeitgrenze liefen noch Kompilierung und Linking. Der Prozess wurde beendet, bevor Tests starteten.

Damit ist der vollständige Vorher/Nachher-Regressionsdelta unbekannt. Die betroffene CV-Gruppe sowie sämtliche neuen Typfluss-Beweise sind dagegen vollständig grün.

## Geschriebene Dateien

- `src/core/service/service.rs`
- `src/core/execution/agent/direct_session.rs`
- `src/core/execution/agent/turn_loop.rs`
- `src/core/harness/ctox-api/src/error.rs`
- `src/core/harness/ctox-api/src/sse/responses.rs`
- `src/core/harness/core/src/api_bridge.rs`
- `src/core/harness/core/src/error.rs`
- `src/core/harness/core/src/response_debug_context.rs`
- `src/core/harness/core/src/api_bridge_tests.rs`
- `src/core/harness/core/src/error_tests.rs`
- `src/core/harness/FORK.md`

Alle Harness-Dateien sind als I-074-Fork-Delta in `FORK.md` dokumentiert. Vom Testlauf erzeugte Lockfile- und Snapshot-Artefakte wurden entfernt. Es wurde weder `git add` noch ein Commit ausgeführt.

## Offene Bedenken

- Der vollständige Nachher-Lauf der 386 Service-Tests muss mit einem stabilen, nicht extern gelöschten `CARGO_TARGET_DIR` wiederholt werden.
- Der finale Rebuild überschritt während des Wartens die Zeitgrenze; er wurde bei Erkennung um 00:15 Uhr beendet.
- Die breitere Textklassifikation in `runtime_error_is_transient_api_failure` bleibt als separater zukünftiger Typisierungsgegenstand bestehen.

**Workjet-Completion-Receipt v1**
```json
{
  "task_id": "I-074",
  "status": "implemented_acceptance_incomplete",
  "acceptance": "typed_gate_and_isolated_proofs_passed_full_post_run_blocked_by_target_cleanup_and_deadline",
  "files_written": [
    "src/core/service/service.rs",
    "src/core/execution/agent/direct_session.rs",
    "src/core/execution/agent/turn_loop.rs",
    "src/core/harness/ctox-api/src/error.rs",
    "src/core/harness/ctox-api/src/sse/responses.rs",
    "src/core/harness/core/src/api_bridge.rs",
    "src/core/harness/core/src/error.rs",
    "src/core/harness/core/src/response_debug_context.rs",
    "src/core/harness/core/src/api_bridge_tests.rs",
    "src/core/harness/core/src/error_tests.rs",
    "src/core/harness/FORK.md"
  ],
  "proof_tests_added": 9,
  "proof_tests_passed": 9,
  "cv_regression_tests": {
    "passed": 14,
    "failed": 0
  },
  "ctox_api_suite": {
    "passed": 73,
    "failed": 0
  },
  "ctox_core_suite": {
    "passed": 1843,
    "failed": 23,
    "ignored": 5
  },
  "baseline": "386 tests discovered; process terminated by SIGKILL before final counts",
  "post_change_full_service_suite": "not completed; target directory was externally deleted and rebuild was stopped at deadline",
  "format_check": "passed",
  "diff_check": "passed",
  "scope_creep": "none",
  "commit_created": false
}
```
