# GROK-7 — Runde 1 (messen): CV-Print-Parser-Recovery in service.rs

Stand: 2026-08-05. Arbeitsbaum nur gelesen. Keine Dateien geändert.

---

## ursache_belegt

### Was der Belang repariert

Wenn ein **geleaster** Queue-Job als CV-Print-Parser erkannt wird und der Worker-Turn mit `Err(err)` endet, fängt der Service den Fehler ab und schreibt trotzdem eine **synthetische** `ctox.cv_print_profile.v1`-Antwort als Command-Writeback, statt Queue/Command zu failen oder nur zu retryen.

Call-Site und Flag:

- `src/core/service/service.rs:7641–7685` — im `Err(err)`-Zweig nach dem Agent-Turn
- Gate: `!job.leased_message_keys.is_empty() && is_cv_print_parser_queue_job(&job) && cv_print_parser_error_allows_compact_recovery(&err_text)` (`:7659–7661`)
- Erfolg: `cv_print_parser_recovered_after_worker_error = true` (`:7667`)
- Danach: **kein** generisches Fail/Retry der Lease (`:7793–7794` skipt den normalen Error-Pfad), sondern `ack_leased_messages(..., "handled")` (`:7890–7903`)

Erkennung des Jobs (`is_cv_print_parser_queue_job`, `:9653–9657`):

- `suggested_skill == "ctox-cv-print-parser"` **oder**
- Prompt enthält `"ctox-cv-print-parser"` / `"ctox.cv_print.apply_parse"` / `"CV PDF extracted text:"`

### Wer den Zustand „richtig“ hätte schreiben müssen

| Schritt | Schreibpfad | datei:zeile |
|---|---|---|
| Modell-Antwort (Soll) | Agent liefert minified JSON `ctox.cv_print_profile.v1` | Prompt-Vertrag: `business_os_cv_print_execution_prompt` `service.rs:9660–9664`; Skill `src/skills/packs/business/ctox-cv-print-parser/SKILL.md` |
| Happy-Path-Writeback | `complete_business_command_from_queue_reply(root, message_key, &reply)` | `service.rs:7275–7278` → `store.rs:14291+` → bei CV: `process_cv_print_parse_command` `store.rs:23480` → `writeback_cv_print_parse` `store.rs:23254` |
| Recovery-Writeback (Ist-Fallback) | `complete_cv_print_parser_recovery_to_leased_queue` → lokal erzeugte `reply` | `service.rs:9674–9763` → `complete_business_command_from_queue_reply` / `complete_cv_print_command_from_reply` (`store.rs:14291`, `store.rs:14403`) |
| Persistenz bei Erfolg | `business_commands` accepted + Document-Version | `process_cv_print_parse_command` `store.rs:23488–23537`; `writeback_cv_print_parse` schreibt Document-Version/Envelope |

**Ursache des reparierten Zustands:** kein (vollständiges) Modell-JSON, weil der Turn an **Runtime/Provider-Grenzen** stirbt — nicht, weil ein erfolgreiches Modell-JSON den Parser-Vertrag verletzt.

Gate (`cv_print_parser_error_allows_compact_recovery`, `service.rs:9667–9672`) matcht **nur**:

1. `max_output_tokens`
2. `incomplete response`
3. `stream disconnected before completion`

Das sind dieselben Klassen, die der Runtime-Stack ohnehin als transient/limit behandelt (`runtime_error_is_transient_api_failure` `service.rs:21394–21412`; `hard_runtime_blocker_retry_cooldown_secs` `turn_loop.rs:1881–1885`).

**Ursache existiert weiter:** Ja. Output-Token-Limits, abgebrochene SSE-Streams und incomplete Responses sind inhärent unzuverlässige Provider-/Modell-Effekte. Der Execution-Prompt verlangt zudem „Preserve all clearly extracted stations; do not artificially truncate“ (`service.rs:9662`) — große CVs erhöhen die `max_output_tokens`-Wahrscheinlichkeit. Der Skill-Vertrag und `extract_json_object` (`store.rs:23184+`) bleiben für den Erfolgsfall gültig.

### String-Matching vs. typisierter Fehlerkanal

**Befund:** Der Recovery-Gate entscheidet auf `err.to_string()` — verbotenes Muster auf dem **Error-String**, nicht auf Modell-Output.

Typisierte Kanäle **existieren upstream**, werden am Gate aber **nicht** genutzt:

| Typ | Ort | Inhalt |
|---|---|---|
| `CodexErr::Stream(String, …)` | `src/core/harness/core/src/error.rs:74` | Display: `"stream disconnected before completion: {0}"` |
| SSE incomplete | `src/core/harness/ctox-api/src/sse/responses.rs:337–346` | `"Incomplete response returned, reason: {reason}"` (reason oft `max_output_tokens`) |
| `CodexErrorInfo::ResponseStreamDisconnected` | `src/core/harness/protocol/src/protocol.rs:1568–1570` | typisierte App-Server-Variante |
| `lcm::AgentOutcome` | `src/core/context/lcm/mod.rs:107–123` | Success / TurnTimeout / ExecutionError / ContextRejected / Aborted / Cancelled |

`classify_agent_failure` (`service.rs:25317–25346`) mappt Timeout/Abort/Cancel/exact-overflow — **`max_output_tokens` / Stream-Disconnect landen als generisches `ExecutionError`**. Am Recovery-Call-Site steht nur der stringifizierte Fehler (`service.rs:7642`). Kommentar bei `classify_agent_failure` (`:25318–25320`): Error-Text kommt vom Harness (owned format), nicht von freiem Prompt-Content — das mildert die String-Matching-Schuld, ersetzt aber keinen typisierten Recovery-Kanal.

**Klassifikation der Maßnahme:**

- **Nicht** Kompensation für schwachen Prompt/Parser-Vertrag (Gate greift nicht bei Bad-JSON/Markdown/Prose nach erfolgreichem Turn).
- **Doch** legitime Fehlerbehandlung / graceful degradation einer **inhaerent unzuverlaessigen Quelle** (LLM-Stream + Token-Budget), mit lokalem Heuristik-Parser (`cv_print_compact_recovery_reply` `service.rs:9782–9874`) und Diagnostic-Warn `"Kompakter CTOX-Recovery-Parse nach Runtime-Abbruch: …"` (`:9808–9813`).

---

## verblieben

### Feuert der Belang real? Dauerhafte Spuren

**Erwartete Dauerhaftigkeit bei erfolgreichem Recovery** (Pfad schreibt permanent):

- `writeback_cv_print_parse` → Document-Version / business records (`store.rs:23254+`)
- `process_cv_print_parse_command` → `business_commands` / Projection (`store.rs:23491–23529`)
- Queue ack `"handled"` (`service.rs:7896–7899`)
- Diagnostic-Text in Modell-JSON: `Kompakter CTOX-Recovery-Parse…`

**Nicht dauerhaft:**

- `push_event_locked` → nur In-Memory-Ring `recent_events` (max 24, `service.rs:24649–24653`)
- `eprintln!("Recovered CV print parser writeback…")` (`service.rs:9717–9750`) — Prozess-Log, nicht SQLite

**Messungen (runtime, Stand Abfrage 2026-08-05):**

| Persistenz | Query / Beobachtung | Zahl / Zeitraum |
|---|---|---|
| `runtime/business-os.sqlite3` | Commands mit Modul `cv-print-builder` | **0** |
| `runtime/business-os.sqlite3` | `business_document_versions` / `business_documents` Rows | **0** / **0** |
| `runtime/business-os.sqlite3` | `business_records` mit `ctox.cv_print` / `ctox-cv-print` / `Kompakter CTOX` | **0** echte CV-Hits; 1 irrelevanter Treffer `payload_json LIKE '%cv_print%'` = AppSec-Finding `F-050` (String im Pfad/Title, kein Command) |
| `runtime/ctox.sqlite3` | `business_command_results` mit `Kompakter` / `cv_print_profile` | **0** (Tabelle hat 243 Rows, **2026-07-10 → 2026-07-18**) |
| `runtime/ctox.sqlite3` | `ctox_harness_flow_events` mit „Recovered CV print“ / „CV print parser“ | **0** (1859 Events, **2026-06-15 → 2026-07-24**) |
| `runtime/ctox.sqlite3` | `business_command_aggregates` intent/module/result cv_print | **0** CV-Print-Aggregate |
| Backups `/Volumes/tmp/ctox-state-backups/update-20260718*/runtime/ctox.sqlite3` | Recovered-CV-Events | **0** (leere/fehlende Treffer) |
| Backup `/Volumes/tmp/ctox-runtime-ausgelagert/backups/update-20260724T072333Z/…/ctox.sqlite3` | Recovered-CV-Events | **0** |

**Lehre angewandt:** Null-Ereignisse beweisen Totsein nur, wenn der Pfad bei Erfolg **dauerhaft** schreibt. Hier schreibt der Erfolgspfad dauerhaft (Commands + Document-Versions). **0 Document-Versions und 0 cv-print Commands** belegen daher: **in diesem Runtime/Backup-Korpus hat Recovery keinen erfolgreichen Writeback hinterlassen** und der CV-Print-Ablauf wurde hier faktisch nicht produktiv mit Writeback genutzt (oder Daten wurden bereinigt).

Einschränkung: Gate-Match mit `updated == 0` hinterlässt nur `eprintln` (`service.rs:9756–9761`) — das wäre unsichtbar in SQLite. Ohne Service-Logs nicht messbar. Für „hat Recovery **erfolgreich** gefeuert“ gilt: **nein, 0 Spuren im messbaren Fenster**.

Code/Unit-Tests existieren und belegen Intent, nicht Produktion:

- `cv_print_compact_recovery_reply_stays_parseable…` `service.rs:29355+`
- `cv_print_runtime_retry_keeps_original_parser_prompt` `service.rs:39222+`
- Fixture-Errors mit `stream disconnected… max_output_tokens` `service.rs:29385`, `39404`

### Ursache noch da?

Ja. Provider-Stream-Abbrüche und `max_output_tokens` bleiben Teil des Runtime-Vertrags; typisierte Upstream-Fehler werden am Service-Rand weiterhin zu Strings.

---

## pfade

### Runde-2-Empfehlung

**Begründetes Nein zur Entfernung als „Kompensationsnetz“.**

Das Netz ist **legitim**: harte Runtime-Abbrüche / inhärent unzuverlässige Modell-Stream-Quelle; Fallback liefert review-fähiges Minimalprofil aus dem bereits im Prompt stehenden extrahierten CV-Text, statt den Command hängen zu lassen oder nur zu retryen. Es ist **kein** Pflaster für „Modell hat Prose statt JSON geliefert“ — das würde den Happy-Path-Parser treffen, nicht diesen Gate.

Optional (nur wenn man härten will, **nicht** als Pflicht-Migration):

1. **Typisierter Gate** statt Substring: z. B. `CodexErr::Stream` + incomplete-reason `max_output_tokens`, oder feineres `AgentOutcome` (z. B. Incomplete/TokenLimit), statt `error_text.contains(...)` in `service.rs:9667–9672`.
2. **Produktentscheid** dokumentieren: Compact-Recovery vs. reiner Retry bei transientem Stream (heute short-circuited Recovery vor dem Retry-Pfad `:7793–7794`).
3. **Telemetrie**: Recovery-Erfolg/Fail dauerhaft (command result diagnostic / harness flow event), damit „feuert real?“ messbar wird — aktuell nur Ringbuffer + eprintln.
4. **Nicht** nötig: Prompt/Skill-Vertrag neu erfinden — der Vertrag ist klar (`service.rs:9662`, SKILL.md); Limit-Fehler sind orthogonal.

### Kurzfazit

| Frage | Antwort |
|---|---|
| Kompensation schwacher Parser-Vertrag? | **Nein** — Gate = Transport/Token-Limit |
| Legitime Fehlerbehandlung? | **Ja** — unzuverlässige Modell-/Stream-Quelle |
| Typisierter Kanal vorhanden, ungenutzt? | **Ja** (`CodexErr::Stream`, incomplete reason, `AgentOutcome`) |
| Produktiv gefeuert (Dauerhaft)? | **0 Spuren** in runtime + geprüften Backups (Fenster ca. 2026-06-15–2026-07-24 für Flow-Events; CV-Domain leer) |
| Runde 2 Pflicht? | **Nein** (optional: typed gate + durable metric) |

