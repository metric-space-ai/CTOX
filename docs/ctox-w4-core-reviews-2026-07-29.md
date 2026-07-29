# W4 — Erstreviews der nie geprüften CTOX-Kerne (29.07.2026)

Unabhängige Kimi-Reviews auf HEAD-Snapshot (Commit 469b48628), je Paket ein
Reviewer mit der globalen A-Definition aus
`docs/ctox-a-grade-masterplan-2026-07-29.md` als Maßstab.

| Paket | Umfang | Note | Aufwand bis A |
|---|---|---|---|
| P8a mission (queue/tickets/schedules/plan) | 53.437 Z. | **C** | L |
| P8b context/continuity + autonomy | 13.012 Z. | **B−** | M (grenzwertig L) |
| P9a communication (Kanäle, Adapter) | 68.543 Z. | **D+** | L |
| P9b mailserver | 4.190 Z. | **D** | M |
| P10 capabilities (scrape/web/doc) | 7.384 Z. | **C** | L |

Kampagnen-Querschnitt: Die vier Schuldklassen der Sync-Reviews (S1–S4)
wiederholen sich in jedem Paket. Neu und schwerwiegend: **behauptete
Garantien** — communication meldet `ok: true` bei Sendefehlern ohne Outbox,
der Mailserver quittiert verworfene Mails mit `250 Ok`, capabilities setzt
`ok: true` bei blocked-Status. Sicherheitsrelevant (sofortige Tickets, nicht
erst A-Welle): Klartext-Passwörter in `password_hash`, Backdoor-Token
`ctox_secret_token` (directory/mod.rs), untrusted Skript kontrolliert
Domain-Allowlist (scrape.rs).

---

## P8a — Mission-Kern: Queue, Tickets, Schedules

### grade
**C** — Der Mission-Kern ist funktional ernstzunehmend: Leasing läuft über atomare SQL-CAS mit Verlierer-Abbruch, Retry/Backoff ist budgetiert und transaktional, Ingest-Schlüssel sind deterministisch-idempotent, und die Tests sind überwiegend echte SQLite-Verhaltenstests (Race-, Idempotenz-, Drift-Szenarien). Aber die strukturellen Schulden dominieren: zwei God-Files (channels.rs 20,5k, tickets.rs 16,4k Zeilen) tragen das gesamte Paket; der Task-/Ticket-Zustand existiert in **~7 parallel gepflegten String-Status-Vokabularen** mit vier separaten, handgewarteten `String→CoreState`-Abbildungen, die schon jetzt auseinanderlaufen; load-bearing `repair_`/`reconcile_`-Funktionen flicken Drift, den die Schreibpfade selbst erzeugen (A-Definition #3 verletzt); und die „terminal policy proofs" werden über Selbstdeklarations-Strings (actor/reason-Prefix) und `request_json LIKE`-Proben vergeben — Text-Dispatch an einem sicherheitsrelevanten Gate. Das Paket war noch nie reviewt, und man sieht es: Die Verhaltensklasse „falsch schreiben → später reparieren" ist hier institutionalisiert (inkl. agentischem LLM-Repair-Pass als Dauerbetrieb).

### findings
`[HIGH] tickets.rs:4799-4832 + channels.rs:13289-13315 + tickets.rs:10773-10805 + tickets.rs:8225-8245 — Vier handgepflegte String→CoreState-Karten, ~7 Status-Vokabulare`
Queue-`route_status` (8 schreibbare + 6 Legacy-Stringwerte, channels.rs:13324-13335), Ticket-Self-Work-State (~27 Strings), Ticket-Case-State (~20 Strings), Ticket-Event-Route (6 Strings), Workflow-Step-Status (eigenes Vokabular in `workflow_step_satisfied`, tickets.rs:4271-4284), Plan-Goal/Step-Konstanten, Schedule-Run-Strings. Kein einziges Status-Enum im Paket — alles `String` + `matches!(as_str(), …)`. Konsequenz: Jede neue Statusausprägung muss an N Stellen nachgetragen werden; die Workflow-Helper zeigen bereits Drift (sie kennen `verified/passed/satisfied`, die Case-Karte nicht). Feld-Folge: gleicher fachlicher Zustand verhält sich je nach Subsystem unterschiedlich, und die Core-State-Machine-Integration bails zur Laufzeit statt zur Compile-Zeit.

`[HIGH] channels.rs:13251-13270 + tickets.rs:8216-8223 — Terminal-Policy-Proofs per Actor/Reason-String-Matching`
`queue_terminal_policy_proof` gewährt den Completed-Gate-Proof, wenn `actor == "ctox-queue-update" && reason.starts_with("business-os:terminal-success:")` (bzw. `"appsec:…"`, und tickets.rs matcht exakt `("ctox-ticket-routing", "force_ticket_event_routed_state")`). Actor und Reason sind frei wählbare Aufrufer-Strings — jeder Schreibpfad, der die magischen Strings kennt, „beweist" terminalen Erfolg. Die Gegenprobe ist ebenfalls Text: `request_json LIKE '%"terminal_policy_proof"%'` (channels.rs:13221, tickets.rs:4790, 8206), fragil gegenüber JSON-Serialisierungsdetails. Konsequenz: Das Beweis-Gate, das Queue-Completion ohne Review verhindern soll, ist eine Namenskonvention, kein Beweis.

`[HIGH] plan.rs:1012-1059 + channels.rs:5569-5708 + queue.rs:570-595 — Reparatur als Dauerbetrieb statt korrekter Schreibpfad`
`repair_stale_step_routing_state` schreibt Routing-Status nach, den die Plan-Emission selbst hätte atomar setzen müssen; `reconcile_business_command_invariants` flickt „cancelled queue command drift" und fehlende Outbox-Zeilen; `repair_queue_state` läuft bei jedem `ctox queue repair` inkl. LLM-Agenten-Pass. Alle drei haben dedizierte Tests, die das Reparaturverhalten als Feature pinnen. Konsequenz: Die echten Schreibpfade dürfen driftend bleiben, weil der Reconciler es schon richtet — genau die Schuldklasse, deren A-Akzeptanz die Löschung ist.

`[HIGH] channels.rs:1-20494 + tickets.rs:1-16385 — Zwei God-Files mit Verantwortungs-Salat`
channels.rs (14,2k Produktivzeilen): Queue-Store, Business-Command-Saga/Outbox, Outbound-Send-Pipeline, Review-Approvals, Founder-Deliverables, **PDF-Generierung** (channels.rs:7926-8047), TUI-Ingest, Pairing, Business-OS-Projektionen. tickets.rs (12,9k Produktivzeilen): 930-Zeilen-CLI-Dispatcher (tickets.rs:857-1786), 465-Zeilen-`ensure_schema` (11570), Knowledge-Base mit Embedding-Socket-RPC (6911), Workflow-Engine, Cases, Writebacks, Lernkandidaten. Konsequenz: Jede Queue-Änderung serialisiert sich an channels.rs; die vom Masterplan geforderten Move-only-Schnitte sind hier noch dringender als im Store.

`[HIGH] plan.rs:1920-1962 — Doppelbesitz am Queue-Zustand: plan.rs schreibt channels-Tabelle direkt`
`set_queue_routing_status_tx` schreibt `communication_routing_state` (channels.rs-Eigentum) über die Plan-Connection mit einem eigenen, fünften akzeptierten Status-Subset (`pending|blocked|failed|handled|cancelled`). Gleiche SQLite-Datei, aber zwei Besitzer mit divergierendem Vokabular und divergierenden Guard-Aufrufen. Konsequenz: „Wer besitzt den Task-Zustand?" ist faktisch unbeantwortet — channels.rs nominal, plan.rs/queue.rs/schedule.rs faktisch mit-schreibend; genau daraus entsteht die Drift, die Finding 3 repariert.

`[HIGH] schedule.rs:482-489 — One-Shot-Meeting-Join: 1-Minuten-Retry ohne Cap, ohne Backoff`
Bei Spawn-Fehler liefert `next_task_state_after_emit` `now + 1min, enabled=true` — unendlich, minütlich, ohne Attempt-Zähler. schedule.rs:1122 pinnt dieses Verhalten per Test. Konsequenz: Ein permanent scheiternder Join (kaputte URL, fehlender Bot) erzeugt einen Dauer-Retry-Sturm und müllt `scheduled_task_runs` zu — im Gegensatz zur sonst überall disziplinierten Budgetierung (5/3 Attempts mit Backoff).

`[MED] channels.rs:3212-3216 vs. tickets.rs:5497-5503 — Backoff-Arithmetik kopiert und bereits divergiert`
Identische Formel (300s·2ⁿ, Cap 3600s), aber Attempt-Limit 5 (Queue-Holds) vs. 3 (Ticket-Events), plus `failure_proof` als Freitext-Format-String inkl. unbegrenzter `reason`. Konsequenz: Retry-Politik ist kein Ort, sondern zwei Kopien; die nächste Änderung driftet weiter.

`[MED] schedule.rs:1367-1374 — Handgerollter Cron-Parser mit falscher dom/dow-Semantik`
POSIX-Cron verknüpft eingeschränktes `day_of_month` und `day_of_week` mit ODER; `CronExpr::matches` UND-et sie. `0 9 1 * 1` feuert hier nur am 1., wenn dieser ein Montag ist — statt am 1. und jedem Montag. Konsequenz: stille Fehl-Scheduling für genau die Ausdrücke, die Operatoren aus cron-Gewohnheit schreiben.

`[MED] queue.rs:1966-1975 + 2008-2016 — Follow-up-Identität über Titel-Präfix „spill restore: "`
Restore-Follow-ups werden erkannt und dedupliziert über `title.starts_with(SPILL_RESTORE_TITLE_PREFIX)` statt über ein Metadata-Feld. Ein umbenanntes oder lokalisiertes Follow-up wird doppelt angelegt bzw. nie abgeschlossen. Gleiche Klasse: `legacy_workspace_root_from_prompt` (channels.rs:13371-13400) parst Workspace-Pfade aus Prompt-Textmarkern.

`[MED] plan.rs:767-823 + schedule.rs:432-471 — Ingest vor Commit auf getrennter Connection`
`ingest_plan_message`/`ingest_cron_message` committen über eine zweite Connection, bevor die Plan-/Schedule-Transaktion den Schritt als queued markiert. Crash-Fenster dazwischen; die deterministischen Message-Keys + UPSERT machen die Wiederholung idempotent (das rettet es), aber die Run-/Step-Zeile bleibt beim Retry vom Überschreiben abhängig. Konsequenz: Korrektheit hängt an Schlüssel-Disziplin statt an Atomarität — eine Transaktion über eine Connection wäre möglich (gleiche Datei).

`[MED] plan.rs:31 + schedule.rs:37 + tickets.rs:56 + ticket_local_native.rs:20 — Vier Kopien desselben DB-Pfads, sechs open_*_db-Funktionen, ~40 duplizierte Helfer`
`DEFAULT_DB_RELATIVE_PATH = "runtime/ctox.sqlite3"` vierfach, `open_channel/plan/schedule/ticket/queue_bridge_db` mit je eigener `ensure_schema_once`-Cache-Maschinerie; `now_iso_string`, `stable_digest`, `required_flag_value`, `find_flag_value`, `print_json`, `clip_text` in bis zu 6 Kopien. Konsequenz: Parallelwahrheiten im Kleinen — die Datei ist eine, die Infrastruktur behauptet sechs.

`[LOW] verification.rs:22-58 — Closure-/Operational-/Artifact-Wortlisten im Code`
Keyword-Heuristiken (`CLOSURE_WORDS`, `OPERATIONAL_WORDS`) als harte Konstanten; deutschsprachige Ergebnistexte fallen durchs Raster. Für A eher Daten/Skill als Code.

`[LOW] approval_nag.rs:243-251 + channels.rs:11568-11572 — „duplicate column name"-Textmatch in Migrationen`
Probe-then-ALTER mit Fehlertext-Fallback für die Race dazwischen. Toleriertes Idiom, aber formal Kontrollfluss über Fehlertext; mit transaktionalem Schema-Lock oder `ALTER … ADD COLUMN IF NOT EXISTS`-Ersatz lösbar.

### healthiest_aspects
- channels.rs:6283-6305 — Lease per `INSERT … ON CONFLICT … WHERE route_status='pending'`: atomare CAS-Transition, der verlierende Racer schreibt keinen Phantom-Proof und bekommt keinen Task. Vorbildlich.
- channels.rs:3129-3270 — `hold_leased_messages`: typisierter `HoldReason`, budgetierter exponentieller Backoff mit Terminalisierung, Post-Write-`ensure!`-Invarianten, alles in einer Transaktion.
- channels.rs:3357-3376 + channels.rs:6706 — Idempotentes Enqueue: Caller-`idempotency_key` foldet Crash-Retries auf denselben `message_key`; Plan-/Cron-/IoT-Ingests haben deterministische Keys + Spawn-Budget (`max_attempts: 8`, channels.rs:6737).
- queue.rs:1506-1615 — Agentischer Repair sauber getrennt: LLM **proponiert** Textplan, deterministischer Applier **verweigert** Hot-Path-Items und unproven Completes (queue.rs:1529-1546). Propose/Apply mit Guards statt LLM-mutiert-direkt.

### god_files
- **channels.rs (20.494 Zeilen, ~14,2k produktiv)** — Mix: Queue-Store, Command-Saga, Send-Pipeline, Review, Founder-PDF. Schnitt: `queue_store.rs` (QueueTaskView/Leasing/Routing/Backoff), `command_saga.rs` (3444-5708), `outbound_send.rs` (7266-10400), `review_approvals.rs` (8613-10250), `business_os_projection.rs` (1577-2516). PDF-Generierung (7926-8047) gehört ganz raus (artifacts).
- **tickets.rs (16.385 Zeilen, ~12,9k produktiv)** — Schnitt: `ticket_cli.rs` (handle_ticket_command 857-1786, ~930 Zeilen → in Subcommand-Module zerlegen), `ticket_schema.rs` (ensure_schema 11570-12035 + Migrationen), `self_work.rs` (2916-5148), `workflow.rs` (3515-4370), `knowledge.rs` (2403-2725 + Embeddings 6814-6975), `cases.rs` (8931-10805), `source_skills.rs` (5887-7665).
- **God-Funktionen**: tickets.rs:857 (~930 Z., CLI-Dispatch), tickets.rs:11570 `ensure_schema` (~465 Z.), channels.rs:11056 `ensure_schema` (~448 Z.), review.rs:840 `build_review_prompt` (~329 Z.), tickets.rs:2403 `refresh_observed_ticket_knowledge` (~322 Z.).

### text_dispatch
**~10 Produktions-Stellen** (Test-Assertionen auf Fehlertexte ausgenommen, davon ~14). Schlimmste:
1. channels.rs:13251-13270 / tickets.rs:8216-8223 — Policy-Proof-Vergabe über Actor-/Reason-Strings (`starts_with("business-os:terminal-success:")`), sicherheitsrelevant.
2. channels.rs:13221-13228 / tickets.rs:4790-4797 / tickets.rs:8206-8214 — Proof-Existenz via `request_json LIKE '%"…":"true"%'`-Substring.
3. review.rs:780-784 — `err.to_string().contains("turn completed without assistant message")` steuert Recovery-Pfad.
4. queue.rs:1973/2013 — Identität per Titel-Präfix (s. Finding).
5. approval_nag.rs:246 / channels.rs:11569 — „duplicate column name".

### test_coverage
Überwiegend **Verhaltens**-Regressionen mit echtem SQLite: CAS-Lease-Races (channels.rs:17522), Idempotenz unter gleichem Key (16802), Reconciler-Drift (19903, 19995, queue.rs:3109), Outbox-Backoff→Dead-Letter (20454), Review-Rework-Requeue (20072), Cron-Parser + Emit-Gate (schedule.rs:1070-1240), Repair-Parsing/Apply-Guards (queue.rs:3150-3200+). **Impl-Pins**: EXPLAIN-QUERY-PLAN-Assertion (channels.rs:14328-14341), ~15 Test-only-Zähler für DB-Opens/Cache-Misses (kanalübergreifend kopiert), ~14 `err.to_string().contains(…)`-Assertionen, die Fehlertexte als API zementieren. **Lücken**: ticket_local_native.rs (585 Z., **0 Tests**), ticket_gateway.rs/ticket_protocol.rs (0), approval_nag.rs (12 % Testanteil; Business-Hours/Reply-Parsing kaum belegt), die unbegrenzte Meeting-Retry-Schleife ist gepinnt statt hinterfragt.

### effort
**L** (mehrere virtuelle Wochen). Treiber: (1) Move-only-Schnitt der zwei God-Files entlang der genannten Nähte — Vorbedingung für parallele Arbeit; (2) ein typisiertes Status-Enum je Entität mit genau einer `→CoreState`-Karte, inkl. Migration der Legacy-Strings und Löschung dreier von vier Mapping-Funktionen; (3) Schreibpfade atomar machen (eine Connection/eine Transaktion für Plan-Emission + Routing) und danach `repair_stale_step_routing_state`/`reconcile_business_command_invariants`/Agentic-Repair **löschen** — A-Akzeptanz ist die Löschung; (4) Policy-Proofs von String-Matching auf strukturierte Evidence umstellen; (5) Meeting-Retry-Cap, Cron-dom/dow-Fix, Helper-Deduplizierung, Testlücken (ticket_local_native) schließen.

## P8b — Context & Continuity + Autonomie

### grade
**B-** — Das Paket hat eine echte Stärke: Die Compaction-Pipeline ist durchdacht verlustarm konstruiert (rohe Messages bleiben persistent, Summaries sind content-gehasht, token-gated inserts mit Savepoint-Rollback, deterministischer Fallback-Summarizer, adversarialer Stress-Harness). `compact.rs` und `autonomy.rs` sind saubere, kleine Zustandsmaschinen. Der dominante Verfallsmuster sitzt aber genau im Continuity-Kern: Der Mission-State — das zentrale Steuersignal für Watcher, Autonomie und Health-Governor — wird per `derive_mission_state_from_continuity` aus freiem, modellgeschriebenem Markdown mit bilingualen Namensheuristiken und Legacy-Sektions-Fallback-Ketten geparst; bei Konflikten wird das Dokument nachträglich „repariert" und in beiden Richtungen synchronisiert (Text↔Tabelle). Das ist die Schuldklasse „falsch schreiben→später reparieren" plus Sonderfall-Stapel in Reinform, dazu ein 7.788-Zeilen-God-File und Text-Dispatch auf SQLite-Fehlertexte. Garantie gegen stillen Kontextverlust: Speicherseitig stark (nichts wird gelöscht, forgotten-Ledger, Clobber-Guard); prompt-seitig nur heuristisch (Relevanz-Ranking und Mission-Term-Filter können Zeilen still fallen lassen, immerhin mit omitted-Counter-Notice). Determinismus: Summary-IDs sind reproduzierbar, Continuity-Commit-IDs enthalten Wall-Clock-Millis — kein Bit-Replay. (Hinweis: Der Masterplan liegt nicht im Snapshot, nur `src/`; Bewertung nach den genannten Schuldklassen.)

### findings
`[HIGH] lcm.rs:1-7788 — God-File mit ~10 Verantwortungen`
Engine/Schema, Compaction-DAG, Continuity-Dokumente+Diffs, Mission-State, Verification-Runs, Strategische Direktiven, Claims, Secret-Rewrite, FTS/Regex-Suche, CLI-`run_*`-Fassade und Fixture-Harness in einer Datei (plus ~1.400 Zeilen Tests). Jede Änderung an Mission-State zwingt zum Lesen von Compaction-Internals; Merge- und Review-Kosten steigen monoton. Siehe god_files für Schnittvorschlag.

`[HIGH] lcm.rs:5102-5330 — Mission-State aus Freitext geparst, dann repariert`
`derive_mission_state_from_continuity` extrahiert das zentrale Laufzeitsignal (`is_open`, `allow_idle`, Blocker, Next-Slice) über `last_named_value`-Heuristiken mit zwei parallelen Sektionsvokabularen („Contract/State" vs. Legacy „Status/Blocker/Next/Done / Gate") und deutschen/englischen Feldnamen. Ein Modell, das „Status: blocked" statt „Mission state: blocked" schreibt, landet in einer anderen Fallback-Kette — Fehlparse ⇒ Watcher feuert nicht oder Mission gilt fälschlich offen/geschlossen. `maybe_repair_focus_continuity_with` (lcm.rs:5202) und `rewrite_focus_continuity_from_mission_state` (lcm.rs:2125) schreiben das Dokument anschließend kanonisiert zurück — Zwei-Wege-Sync zwischen Textwahrheit und Tabellenwahrheit mit Reparaturschleife statt strukturiertem Zustandsautomaten.

`[MED] lcm.rs:4141-4146, 1093-1101, 5496 — Text-Dispatch auf SQLite-Fehlertexte`
Drei Stellen matchen Kontrollfluss auf Fehlerstrings: WAL-Fallback via „xShmMap"/„shared-memory"/„disk I/O error && resize", Migrationstoleranz via „duplicate column name", Claim-Zählung schluckt „no such table" als `Ok(0)`. Letzteres maskiert Schema-Regressionen still als „keine blockierenden Claims" — ein Closure-Gate, das bei kaputtem Schema einfach aufgeht. Rusqlite liefert `ErrorCode`-Enums; mindestens das „no such table"-Schlucken gehört auf eine echte Schema-Prüfung.

`[MED] lcm.rs:4629-4667 — Audit-Events in prozessflüchtigem Thread-Local`
Der Clobber-Guard (gute Idee: Felder nicht leer überschreiben) puffert blockierte Schreibversuche in einem thread-lokalen `RefCell<Vec>` und publiziert sie erst beim nächsten Maintenance-Pass an Governance. Prozessabsturz oder Schreiben aus einem anderen Thread ⇒ Audit-Event verloren, genau bei der Klasse von Vorfällen, die man später rekonstruieren will. Der Kommentar „Failures are swallowed" dokumentiert die Einbahnstraße.

`[MED] live_context.rs:1170-1201, 1837-1881 — Stilles Prompt-Line-Dropping per Substring-Heuristik`
`rank_context_lines` sortiert Kontextzeilen nach Vorkommen von ≥4-Zeichen-Termen aus dem User-Prompt (mit handgepflegter deutscher/englischer Stopword-Mini-Liste); `workflow_matches_terms` filtert offene Arbeiten nach Missions-Termen. Ein wichtiger Kontextfakt ohne lexikalischen Overlap mit der aktuellen Frage fällt im Budget-Fall hinten runter bzw. raus. Abgefedert durch omitted-Counter und Empty-Filter-Fallback (getestet), aber „kein Kontext geht still verloren" gilt hier nur für die DB, nicht für den gerenderten Prompt.

`[MED] lcm.rs:1559-1566, 1642 — Zeitstempel als ungepadete Millisekunden-Strings`
`iso_now()` liefert `u128`-Millis als String; `latest_continuity` nimmt `std::cmp::max` und `continuity_log` sortiert lexikographisch darüber. Funktioniert nur, solange alle Werte 13-stellig bleiben, und macht Commit-IDs (lcm.rs:5774, hashen `created_at` mit) wall-clock-abhängig — derselbe logische Diff ergibt bei Replay andere IDs, kein deterministisches Rebuild. Zumal `count_open_closure_blocking_claims` denselben String mit `parse().unwrap_or(i64::MAX)` zurückholt (Parse-Fehler ⇒ „alles abgelaufen" — fail-open in die falsche Richtung).

`[LOW] autonomy.rs:97-141 — Policy-Texte doppelt gepflegt, nur 1 Test`
`runtime_policy_block` und `step_prompt_clause` duplizieren pro Level dieselbe Policy in zwei Wortlauten; Drift zwischen beiden ist vorprogrammiert (kein Test prüft Konsistenz, `from_str_lossy`-Aliase und Nag-Kadenz sind ungetestet). Ansonsten sauberste Datei des Pakets.

`[LOW] lcm.rs:761, live_context.rs:1458 — Benennungs-/Signatur-Schönheitsfehler`
`initialized_lcm_paths` nennt die Variable „canonical", macht aber nur `to_path_buf()` — gleiche DB über verschiedene Pfadschreibweisen re-initialisiert (benign, aber irreführend). `derive_mission_id` ignoriert `_workspace_root` — Missions-IDs kollidieren prinzipiell über Workspaces hinweg.

### healthiest_aspects
- `compact.rs:189-257` — echte geschichtete Zustandsmaschine (Emergency → Adaptive → Fixed-Turns) mit Refire-Suppression, Integer-Arithmetik und Kommentaren, die konkrete Feld-Regressionen benennen (MidTask-Overflow-400, Reasoning-Token-Verzerrung).
- `lcm.rs:3165-3231` — `insert_summary_token_gated`: Write-Validate-Commit über Savepoint; Compaction kann den Kontext nie vergrößern, Verhalten durch `compact_never_enlarges_context_under_regressing_summarizer` gepinnt.
- `lcm.rs:3418-3448` — `summarize_with_escalation`: leere/regressive Modell-Summaries fallen auf einen deterministischen Fallback zurück statt den DAG zu vergiften.
- `context_stress.rs:186-523` — adversarialer Summarizer + 20-forced-compactions-Harness: Property-Level-Robustheitstests, wie man sie in den anderen Kernen selten findet.
- Continuity-Forgotten-Ledger (`lcm.rs:2666-2741`) + Anchor-Literal-Preservation mit Respekt vor deliberate deletions (`lcm.rs:2820-2843`) — der „nichts geht still verloren"-Wille ist architektonisch verankert, nicht nur kommentiert.

### god_files
- **lcm.rs (7.788 Zeilen, ~6.400 non-test)** — klar über der 5k-Grenze mit massivem Verantwortungs-Mix. Schnittvorschlag entlang vorhandener Nahtstellen: `lcm/engine.rs` (open, init_schema, ensure_column, busy-timeout), `lcm/compaction.rs` (compact, *_pass, insert_summary*, resequence, token counts, summarize_with_escalation), `lcm/continuity.rs` (Dokumente, Diffs, Commits, Prompts, Forgotten, Anchor-Literale), `lcm/mission_state.rs` (derive/persist/repair/Clobber-Guard/canonicalize_*), `lcm/assurance.rs` (verification runs, claims, strategic directives), `lcm/search.rs` (FTS/Regex/grep/expand), `lcm/cli.rs` (`run_*`-Fassade), `lcm/fixture.rs`. Größte Einzelfunktionen bleiben unter 300 Zeilen, aber `build_continuity_prompt_text` (lcm.rs:6012, 141 Zeilen), `persist_mission_state_with` (lcm.rs:4709, 117) und `derive_mission_state_from_continuity` (100) gehören nach dem Split jeweils in Parsing- vs. Persistenz-Hälften geteilt.
- **live_context.rs (2.569)** — unter der Dateigrenze, aber `render_workflow_state_block` (1624-1837, 213 Zeilen) mischt SQL, Term-Filterung und Rendering: in `load_open_work` / `filter_relevant` / `render` schneiden.
- **context_health.rs (1.430)** — `build_warnings` (625-769, 144 Zeilen) ist ein Warnungs-Sonderfall-Stapel; pro Warning-Code ein kleiner Emitter würde die Datei flach halten.

### text_dispatch
**3 harte Kontrollfluss-Stellen** (alle lcm.rs): 4141-4146 `is_shared_memory_io_error` (String-Match entscheidet WAL→Delete-Fallback — Match-Lücke ⇒ Datenbank öffnet nicht statt degraded zu laufen), 1093-1101 „duplicate column name" (Migrationstoleranz), 5496 „no such table" ⇒ `Ok(0)` — die schlimmste, weil sie ein Closure-Gate bei Schema-Bruch still öffnet. Weiche Varianten (Heuristik über Nachrichtentexte, die in Scores/Governor-Entscheidungen mündet): `context_health.rs:1060-1066` (blocked-Erkennung über en/de Phrasen wie „still blocked"/„bleibt blockiert") und 1074-1084 (Repair-Churn über Source-Label-Strings). Gesamt: ~5 nennenswerte Stellen, davon 1 gefährlich (5496).

### test_coverage
74 Tests gesamt (lcm 35, live_context 18, context_health 10, compact 7, stress 3, autonomy 1). Stärke: echte Verhaltens-Pins an den kritischen Invarianten — Token-Gate-Rollback, Adversarial-Summarizer, Secret-Rewrite über alle Speicher, Clobber-Guard mit Audit, Empty-Filter-Fallback, Multibyte-Unicode-Grenzen. Schwäche: Die lcm-Tests pinnen zu einem guten Teil die einzelnen Fallback-Sonderfälle des Freitext-Parsers (`mission_state_accepts_open_and_partial...`, `..._keeps_explicit_blank...`, `..._ignores_empty_focus_template_placeholders`) — Impl-Pins, die den Sonderfall-Stapel zementieren statt ein Verhaltenskontrakt („Mission-State ist strukturiert") zu fordern. Lücken: (a) kein Determinismus-/Replay-Test (gleicher Message-Strom ⇒ gleiche Summary-IDs und stabile Continuity-Revision über Rebuild), (b) autonomy: Aliase, Kadenz-Mapping, Konsistenz der zwei Policy-Texte ungetestet, (c) kein Test für „no such table"-Pfad in `count_open_closure_blocking_claims`, (d) kein Test, dass der WAL-Fallback bei echtem I/O-Fehler greift (nur via Text-Match erreichbar).

### effort
**M (grenzwertig L)** — Der große Block ist der lcm.rs-Split (mechanisch, aber difficil wegen der vielen `pub(crate)`-Querverweise; mit den vorhandenen 35 Verhaltens-Tests als Sicherheitsnetz machbar) plus die Ablösung des Freitext-Mission-State durch ein strukturiertes Feldset mit einmaliger Migration (der riskanteste Eingriff, weil Watcher/Autonomie daran hängen). Text-Dispatch-Fixes, Timestamp-Typisierung, Audit-Puffer-Persistenz und autonomy-Testlücken sind klein. Realistisch: 2 Pakete à M (Split; Mission-State-Strukturierung) + 1 S (Rest).

## P9a — Communication-Kern: Kanäle & Zustellung

### grade

**D+** — Der Communication-Kern trägt funktional (12 Kanäle, Sync-Dedup über `message_key`-Cursors, zentrale DB-Helfer), ist aber die reinste Ansammlung der bekannten Schuldklassen, die ich bisher in dieser Kampagne gesehen habe. Dominantes Verfallsmuster: **behauptete statt echter Zustellgarantien** — Chat/Teams/E-Mail kennen keine Outbox und keinen Retry; bei Provider-Fehler wird `"ok": true` mit Status `"failed"` zurückgegeben und eine `"queued-<digest>"`-ID fabriziert, obwohl nichts queued; `meeting send` schreibt `"sent"` in die DB und verwirft danach den Weiterleitungsfehler mit `let _ =` (meeting_native.rs:374-378). Zweites Muster: **God-Files** — meeting_native.rs (8.234 Zeilen, davon ~3.200 Zeilen eingebettetes JavaScript mit `__PLACEHOLDER__`-String-Templating) und chat_native.rs (6.213 Zeilen, 7 Plattformen × sync/send/test als Match-Arme statt Daten). Dazu der schlimmste Text-Dispatch bisher: HTTP-`Retry-After` wird in einen anyhow-Fehlertext hineinformatiert (chat_native.rs:4266) und 250 Zeilen später per `split("retry_after=")` wieder herausgeparst, um Backoff zu schedulen (chat_native.rs:4110). Gut: das vendored `whatsapp_rust` (42k der 68k Zeilen) ist sauber in 12 benannte Crates geschnitten mit getyptem Retry-Receipt-Protokoll; Jami ist der einzige ehrliche Sender (`"submitted"`, `confirmed: false`, jami_native.rs:282); Slack-Socket-Mode-Backoff ist ein persistenter, gecappter Zustand.

### findings

`[HIGH] chat_native.rs:4266-4269 + 4110-4119 + 3156 — Retry-After-Roundtrip durch Fehlertext` `http_json_response` formatiert `retry_after={retry_after}` in den anyhow-String; `rate_limited_until_ms_from_error` parst denselben String per `split("retry_after=")`, um den Realtime-Backoff-Zeitpunkt zu berechnen. Kontrollfluss (Backoff-Scheduling) hängt an einem selbstformatierten Fehlertext — jede Umformulierung des bail!-Texts bricht das Rate-Limit-Verhalten still. ureq liefert Status + Header getypt; die Information war an der Quelle vorhanden.

`[HIGH] chat_native.rs:595-731, teams_native.rs:1083-1221 — Zustellgarantie behauptet, nicht implementiert` Bei Sendefehler wird die Nachricht mit `"queued-<digest>"`-Remote-ID und Status `"failed"` persistiert und `"ok": true` zurückgegeben — es gibt keinerlei Outbox, Wiederholung oder Dead-Letter-Pfad; „failed" ist terminal. Zusätzlich ist die Fallback-ID ein Inhalts-Digest aus `plattform:ziel:body`: zweimal derselbe Text an dasselbe Ziel erzeugt denselben `message_key` und überschreibt per Upsert den früheren Datensatz (chat_native.rs:609-617, 682). Feld-Konsequenz: stille Nachrichtenverluste, die in der UI als „Zustellversuch" erscheinen.

`[HIGH] meeting_native.rs:309-380 — „sent" geschrieben, Weiterleitung danach mit `let _ =` verworfen` Der Meeting-Chat-Send persistiert zuerst Status `"sent"` in SQLite und schreibt erst danach das Kommando in die Session-Command-Datei; ein Fehler des Playwright-Runners ändert den Datensatz nicht mehr. Falsch-schreiben→nie-reparieren: die DB behauptet Zustellung, der Teilnehmer sieht nichts.

`[HIGH] meeting_native.rs:3291-6533 — ~3.200 Zeilen eingebettetes JavaScript mit Placeholder-Templating` `MEETING_RUNNER_TEMPLATE` plus drei Provider-Join-Skripte, Chat-Scraper und Sender als Rust-String-Literale mit `__MEETING_URL__`-Substitution. Unlesbar, ungelintet, keinerlei Syntaxprüfung; Status-Literale (`"joining"`, `"active"`, `"ended"`) existieren parallel im Rust- und im JS-Text (Parallelwahrheit über eine String-Grenze, meeting_native.rs:107/453 vs. 4004/6325). Jede Selektor-Änderung an Google Meet/Teams/Zoom ist ein Blindflug-Edit in einem Fremdsprachen-String.

`[HIGH] chat_native.rs:3878-3980 — Fehlerklassifikation über ~50 Substring-Probes` `classify_provider_error` matcht lowercase-Substringe (`"unauthorized"`, `"scope"`, `"intent"` …) auf Fehlertexte aller 7 Provider, inkl. Plattform-Sonderfall `Discord if contains("intent")`. Die Substrings überlappen (`"scope"` schlägt vor `"insufficient_scope"` zu, Reihenfolge ist Semantik), und jede Provider-Umformulierung landet im generischen `"failed"`-Bucket mit falscher Remediation.

`[HIGH] meeting_native.rs:449-477 — reconcile_stale_running_session repariert beim Lesen` Sync mutiert Session-JSON-Dateien on-read (`status → "ended"`, `end_reason: "process_not_running"`), wenn `kill -0` den PID nicht mehr findet. Exakt die verbotene Repair-Funktion: der Schreibpfad (Runner-Crash ohne Finalize) bleibt defekt, die Reparatur kaschiert es — und `kill -0` ist nicht portabel (Windows) und PID-Reuse-anfällig.

`[MED] chat_native.rs:3003-3004 + 27-28 — Fake-Provider per Magic-String in Produktionspfaden` `is_fake_mode` wird über Token == `"ctox-fake"` oder Base-URL-Präfix `"ctox-fake://"` aktiviert und verzweigt in allen `execute_*`-Pfaden (383, 489, 618, 1230, 2988, 3214, 3795). Testrückgrat als Laufzeit-Verhaltenszweig: ein geleakter/gesetzter Fake-Token macht den Adapter lautlos funktionslos bei `"ok": true`.

`[MED] whatsapp_native.rs:427 — „delivery confirmed: true" nach bloßem Server-Ack` `send_text` liefert die Message-ID nach Server-Annahme; es wird weder Zustell- noch Lese-Receipt abgewartet, trotzdem wird `"delivery": {"confirmed": true}` behauptet. Dieselbe Operation meldet Jami ehrlich als `confirmed: false` — inkonsistente Semantik desselben JSON-Felds je Kanal.

`[MED] email_native.rs:2458 — paketweiter HTTP-Client wohnt im E-Mail-Adapter` `http_request` (ureq-Wrapper) wird von chat_native, teams_native und gateway (STT/TTS) querimportiert; email_native ist zugleich IMAP-Client, SMTP-Client, EWS-SOAP-, ActiveSync- und Graph-Stack (4 Protokoll-Implementierungen in einer Datei). Modulgrenze faktisch nicht vorhanden.

`[MED] gateway.rs:20-22 + adapters.rs:106-211 — Abstraktion, die nicht abstrahiert` `CommunicationAdapterBackend` hat genau eine Variante (`NativeRust`), `CommunicationAdapterSpec.backend` ist damit Konstante; gleichzeitig existieren sieben fast identische Send-Request-Structs (Email/Jami/Chat/Teams/Whatsapp/Meeting…), weil das Trait nur `kind()` kennt. Kanal-Sonderfälle sind Code (12 leere Adapter-Structs + per-Kanal-impl-Blöcke), nicht Daten — eine Plattform-Tabelle (Env-Keys, Cursor-Art, Endpunkte, Capabilities) würde adapters.rs auf ~150 Zeilen schrumpfen.

`[LOW] meeting_native.rs:434 + 353 — Hardcodierter Produktname „INF Yoda Notetaker" im Kern` Bot-Name als Default in Eigenheits-Erkennung und als `sender_display`; deutsche Ack-Texte hartkodiert (382-384). Branding-Änderung = Core-Patch.

`[LOW] Cargo.toml:10 (Repo-Root) — `_upstream.backup/` als ausgeschlossene Parallelkopie` Das vendored whatsapp_rust trägt eine aus dem Workspace ausgeschlossene Backup-Kopie des Upstreams im Source-Tree — Parallelwahrheit ohne Drift-Check; die bessere Form wäre ein Git-Remote/Pin, kein Datei-Duplikat.

### healthiest_aspects

- `whatsapp_rust/crates/wha-client/src/retry.rs:1-53` — getyptes Retry-Receipt-Protokoll mit Max-Retry-Zähler und reinen Build-Funktionen; die 12-Crate-Struktur (types/binary/crypto/socket/store/signal/client/media) ist der beste Modulschnitt des gesamten Pakets.
- `chat_native.rs:21-22` — DB-Zugriff über zentrale Helfer (`ensure_account`, `upsert_communication_message`, `refresh_thread`, `record_communication_sync_run`) statt per-Adapter-Duplikate; Sync-Dedup über Cursors mit getesteter Max-Cursor-Logik (4071-4108).
- `jami_native.rs:282, 305-308` — einziger Adapter mit ehrlicher Zustellsemantik (`"submitted"`, `confirmed: false`, benannter `state: "submitted_to_daemon"`).
- `chat_native.rs:3289-3312` — Slack-Realtime-Backoff als persistenter Zustand (until/attempt/reason) mit Exponential-Kappe und Clear-Pfad — der Ansatz, den die Outbox bräuchte.
- `runtime.rs:6-30` — saubere Pfad-Namensräume pro Kanal inkl. Migrations-aware Legacy-Auflösung.

### god_files

- **meeting_native.rs — 8.234 Zeilen.** Verantwortungs-Mix: Session-JSON-Lifecycle + Repair-on-Read, Sync-Ingest in SQLite, STT/TTS-Runtime-Guard, PulseAudio-Setup, Preflight-Probes (inkl. eingebettetem Python, 1664-1730), Node/Playwright-Spawn + Xvfb, ~3.200 Zeilen eingebettetes JS (3291-6533), ICS-Parsing (6558-6665), CLI-Dispatch, Mention-Handling. God-Funktion `run_meeting_session` (1813-2265, ~450 Zeilen): Setup, Guard, Env, Spawn, Event-Loop über JSON-`type`-Strings, Finalize. **Schnitt:** `session_state.rs` (Status als Enum, kein String-Patchwork), `sync_ingest.rs`, `stt_runtime.rs` (Guard + transcribe/synthesize), `preflight.rs` (Probes, PulseAudio, Mistral), `ics.rs`, und `runner/*.js` als echte Dateien via `include_str!` — damit werden JS-Lint/Tests möglich und die Status-Parallelwahrheit hat eine Seite.
- **chat_native.rs — 6.213 Zeilen.** 7 Plattformen × {sync, send, test, normalize, payload} + Slack Socket Mode + Discord Gateway + Zulip Event-Queue + Backoff + Fehlerklassifikation + Fake-Provider. **Schnitt:** `platform.rs` (Daten-Tabelle: Env-Keys, Cursor-Strategie, Endpunkte, Auth-Form), `providers/{slack,discord,telegram,matrix,mattermost,zulip,google_chat}.rs`, `realtime.rs` (Socket/Gateway/Queue), `error.rs` (getypter `ProviderError { status, retry_after, code }` statt String-Probes), `outbox.rs` (Zustell-Zustandsautomat). Damit schrumpft der Kern auf den Sync/Send-Orchestrator.
- **email_native.rs — 4.220 Zeilen.** IMAP-, SMTP-, EWS-SOAP-, ActiveSync- und Graph-Client + MIME-Bau + Sent-Verifikation + paketweites HTTP. **Schnitt:** `http.rs` (paketweit), `imap.rs`, `smtp.rs`, `ews.rs`, `activesync.rs`, `graph.rs`, `mime.rs`, `delivery_verify.rs`.

### text_dispatch

~10 Stellen, davon 3 strukturell schwer:
1. **chat_native.rs:4266/4307 → 4110 (Caller 3156):** `retry_after=` wird in den Fehlertext formatiert und per `split` zurückgeparst — Backoff-Scheduling hängt am eigenen Error-String. Schlimmste Stelle.
2. **chat_native.rs:3878-3980:** ~50 Substring-Probes zur Provider-Fehlerklassifikation mit Reihenfolge-Semantik und Discord-Intent-Sonderfall (3963, 4006).
3. **chat_native.rs:4111:** Rate-Limit-Erkennung per `contains("HTTP 429") || lowercase.contains("rate")`.
4. email_native.rs:2966 (`contains("responsemessage")` in EWS-XML), 6102 (Test asserted auf bail-Text „text-only"). Die IMAP-Greeting-Prüfungen (2070, 2096) sind Protokoll-Parsing, kein Text-Dispatch im verbotenen Sinn. whatsapp_rust: keine Fundstelle.

### test_coverage

109 `#[test]` in den nativen Adaptern (meeting 33, chat 20, email 20, teams 14, jami 7, whatsapp 5, gateway/adapters/runtime je 3/3/1) plus eigene Tests im whatsapp_rust-Baum (Crypto/JID/Store). **Verhaltens-Tests vorhanden:** Backoff-Kappe und Persistenz (chat_native.rs:5372-5425), Discord-Dedup per `message_key` (5501), Zulip-Event-Normalisierung (5831), Meeting-Reconcile (7264), Cursor-Max. **Aber viele Impl-Pins:** Registry-Gleichheit (adapters.rs:605-656), „backend == NativeRust" auf einer Ein-Varianten-Enum (gateway.rs:627-649), Template-Substring-Asserts (chat_native.rs:5364, 5831), `resolve_account_request_stays_typed` pinnt eine Struct-Definition. **Lücken:** kein Test für den Sendefehler-Pfad (kein Retry → nichts zu testen, Symptom des HIGH), kein Test für die `"queued-"`-Digest-Kollision, kein Test, der „ok:true bei failed" als Verhalten festnagelt (oder besser: rot macht), keinerlei Test der eingebetteten JS-Skripte.

### effort

**L.** Schnitt zuerst (move-only, ~2-3 Commits): JS aus meeting_native extrahieren, meeting/chat/email nach obiger Karte teilen. Danach Semantik: (1) getypter ProviderError mit Status/Retry-After — ersetzt den Format-Parse-Roundtrip und die 50-Probes-Klassifikation; (2) echter Outbox-Zustandsautomat (queued→sent→confirmed/failed mit Retry/Backoff — der Slack-Backoff-Code ist die Vorlage) inkl. Löschung der `"queued-<digest>"`-Fabrikation und des `let _ =`-Sendepfads; (3) Session-Status als Enum an einer Seite der Rust/JS-Grenze, `reconcile_stale_running_session` durch Schreiben beim Runner-Exit ersetzen und löschen; (4) Plattform-Tabelle statt 12-Struct-Registry. Realistisch die größte Einzelbaustelle der bisher reviewten Kerne, aber dank whatsapp_rust-Vorbild und vorhandener Verhaltens-Tests mit klarer Landebahn.

## P9b — Mailserver

### grade
**D** — Der Mailserver ist eine funktionierende Protokoll-Hülle mit ernsthaft gefälschten Sicherheitssemantiken: Die dominante Verfallsklasse ist „sieht aus wie X, ist aber nicht X" — die Spalte `password_hash` speichert Klartext-Passwörter und vergleicht sie per String-Gleichheit, der DKIM-Signer erzeugt konstruktionsbedingt ungültige Signaturen und fällt bei Schlüsselproblemen still auf `"MOCK_SIGNATURE_FAIL"` bzw. einen SHA-256-Hash als „Signatur" zurück, der SMTP-Server antwortet `250 Ok` auch dann, wenn kein einziger Empfänger zugestellt wurde (Mail-Verlust mit Erfolgs-Ack), und IMAP deklariert `UIDVALIDITY 1` mit UID = Sequenznummer, `EXPUNGE` als No-Op. Dazu kommt ein hartkodierter Backdoor-Token (`ctox_secret_token` für jedes `admin@*`-Konto) in einer ungenutzten Parallel-Auth. Gut ist die Store-Schicht: WAL, busy_timeout, Connection-Cache, Change-Stamp-Idle-Gate, Retry-Backoff mit terminalem `failed_permanent` und ein echter SMTP→Store→IMAP-Verhaltens-Test. Offenes Relay ist es nicht (unbekannte RCPTs werden für Unauthentifizierte abgelehnt), aber fail-closed ist das Paket an mehreren Stellen nicht.

### findings
`[HIGH] src/core/mailserver/src/store/sqlite.rs:458-470 — Klartext-Passwörter in Spalte namens password_hash`
`authenticate_user` vergleicht `db_pass == password_hash` direkt; der Konformanztest speichert `"securepass"` im Klartext. Wer die SQLite-Datei liest (Backup, Support-Bundle), besitzt alle Mail-Credentials sofort. Der Name `password_hash` kaschiert das zusätzlich — Audits würden hier Hashing annehmen.

`[HIGH] src/core/mailserver/src/directory/mod.rs:32-41 — Hartkodierter Backdoor-Token als Parallel-Auth`
`email.starts_with("admin@") && secret == "ctox_secret_token"` gewährt jedem beliebigen `admin@<irgendwas>` Zugang mit einem im Quelltext stehenden Shared Secret. Der Resolver ist `pub`, wird im Paket selbst nicht aufgerufen — eine schlafende Zweit-Wahrheit neben der Store-Auth, die jeder externe Konsument scharf schalten kann.

`[HIGH] src/core/mailserver/src/smtp/dkim.rs:40-69 — DKIM-Signatur per Konstruktion ungültig, fail-open`
Es wird `relaxed/relaxed` deklariert, aber nur der Body kanonikalisiert und ein ad-hoc-String (`"from: {}\r\n{}"`) statt der RFC-Header-Menge signiert; bei Nicht-PKCS8-Schlüsseln wird still ein SHA-256-Hash als Signatur kodiert, bei Signaturfehler wörtlich `"MOCK_SIGNATURE_FAIL"` ins Header-Feld geschrieben. Konsequenz: Jede signierte Ausgangsmail schlägt DKIM/DMARC-Validierung fehl (Zustellbarkeit, Spam-Score) — und niemand erfährt es, weil der Signer `Ok` zurückgibt. Klassisches „falsch schreiben, nie reparieren".

`[HIGH] src/core/mailserver/src/smtp/server.rs:160-167 — 250 Ok für still verworfene Mails; 550-Zweig ist toter Code`
`if delivered || !rcpt_to.is_empty()` ist immer wahr, sobald irgendein RCPT akzeptiert wurde — auch wenn null Zustellungen erfolgten (authentifizierter Nutzer mit externem Empfänger, Mailbox weg zwischen RCPT und DATA). Der Server bestätigt den Empfang und verwirft die Mail ohne Bounce. Das ist Datenverlust mit Erfolgsquittung — die schlimmstmögliche Fail-Richtung für einen Mailserver.

`[HIGH] src/core/mailserver/src/smtp/server.rs:254,308-312 — Command-Match case-insensitiv, Envelope-Parse case-sensitiv`
Gematcht wird auf einer uppercase Kopie (`line_upper`), extrahiert wird aber per `line.replace("MAIL FROM:", …)` auf der Originalzeile. Ein Client, der `mail from:<a@b>` schreibt (RFC-erlaubt), erzeugt `mail_from = "mail from:<a@b>"` als Envelope-Adresse — danach brechen Greylist-Key, DKIM-Domain-Split und From-Header still. Tests decken nur Großschreibung ab.

`[MED] src/core/mailserver/src/smtp/server.rs:147-158 + dsn.rs:44-54 — DSN-Parsing auf jeder Inbound-Mail mit Raten-Fallback`
Jede eingehende Mail wird als Bounce interpretiert, egal ob `MAIL FROM:<>`; der Regex-Fallback nimmt irgendein Wort mit `@` (`ends_with('.') || len() > 5`) als Empfänger — false positives erzeugen Phantom-Bounces. Der Bounce geht per `queue_email` an `admin@localhost`, der typischerweise nicht existiert: fünf Retry-Zyklen bis `failed_permanent`, für jede einzelne Bounce-artige Mail. Bounce-Handling ohne Bounce-Adressaten-Verwaltung.

`[MED] src/core/mailserver/src/imap/mod.rs:243-247,329-393 — IMAP-UID-/Flag-Semantik falsch`
UID wird als `idx + 1` aus der aktuellen Sortierung abgeleitet (nicht persistent), `UIDVALIDITY` hartkodiert 1, `STORE +FLAGS \Deleted` löscht sofort physisch statt zu flaggen, `EXPUNGE` ist eine No-Op. Clients (Thunderbird, Apple Mail) bauen auf persistente UIDs — nach jeder Löschung zeigen alle UIDs auf andere Mails: Fehlsync, falsch gelöschte/gelesene Mails clientseitig.

`[MED] src/core/mailserver/src/smtp/client_queue.rs:353 — Hartkodiertes 8.8.8.8 + handgerollter DNS-Parser`
Der MX-Lookup umgeht den System-Resolver komplett (Privacy, kaputt in Netzen ohne offenen UDP-53-Ausgang) und parst DNS-Responses per Byte-Index selbst, obwohl die Crate-Landschaft fertige Resolver hat. Bei DNS-Ausfall: stille A-Record-Fallback-Zustellung an den falschen Host. Genau die Klasse „Sonderfall-Stapel statt bewährter Komponente".

`[MED] src/core/mailserver/src/directory/mod.rs:24-25 vs. store/sqlite_schema.rs:9-10 — SPF/DMARC als Parallelwahrheit`
Das Schema hat `spf_record`/`dmarc_record`-Spalten; `resolve_domain` ignoriert sie und gibt hartkodierte Strings zurück. Zwei Wahrheitsquellen für dieselbe Information, die gespeicherte wird nie konsumiert.

`[MED] src/core/mailserver/src/store/sqlite.rs:218-236 + smtp/client_queue.rs:181,212 vs. sqlite.rs:852 — Delivery-Outcome-Vokabular als freie Strings`
Produktion schreibt `"delivered"`/`"failed"`, der eigene Test schreibt `"success"` — kein Enum, kein CHECK-Constraint, und der externe „outbound reconciler" muss diese Strings matchen. Text-als-API über eine Modulgrenze hinweg; Tippfehler werden nie entdeckt.

`[MED] src/core/mailserver/src/smtp/server.rs:255-307 + imap/mod.rs:120-147 — Kein TLS, kein Auth-Rate-Limit`
AUTH PLAIN/LOGIN und IMAP LOGIN laufen im Klartext ohne Versuchsbegrenzung. Die Loopback-Bindung (lib.rs:45,50) mildert das, aber die Ports sind per Env konfigurierbar und AUTH wird trotzdem angeboten — Brute-Force gegen die Klartext-Passwort-Tabelle ist ungedrosselt möglich.

`[MED] src/core/mailserver/src/smtp/client_queue.rs:96-98 — DB-Fehler flippt lokale Domains nach extern`
`self.store.user_exists(to).unwrap_or(false)`: Ein SQLite-Fehler macht aus einer lokalen Empfänger-Domain eine „remote" Domain, danach MX-Lookup und Versand ins Internet. Fail-open statt fail-closed genau an der Local/Remote-Sicherheitsgrenze.

`[LOW] src/core/mailserver/src/smtp/server.rs:54-93,169-173 — Unbegrenzte Puffer, kein Dot-Stuffing`
`line_buffer` und `mail_body` wachsen ohne Limit (Memory-DoS über eine einzige newline-freie Verbindung), und DATA-Zeilen mit führendem `..` werden nicht entstuffed — gespeicherte Bodies sind byte-falsch.

`[LOW] src/core/mailserver/src/config.rs:23-24 — Tote Konfiguration`
`outbound_throttle_per_min` und `max_connections` werden nirgends gelesen; weder Throttle noch Connection-Limit existieren. Konfiguration suggeriert Schutz, der nicht da ist.

`[LOW] src/core/mailserver/src/lib.rs:27-93 — Doppeltes Logging (tracing + println/eprintln)`
Jede Startmeldung wird doppelt emittiert; Fehlerpfade dreifach. Rausch in den Logs, zwei Wahrheitsquellen für denselben Zustand.

`[LOW] src/core/mailserver/src/smtp/server.rs:179-208 vs. 257-289 — AUTH-PLAIN-Parsing doppelt`
Initiale-Response- und Challenge-Variante duplizieren denselben Decode/Split/Authenticate-Block wörtlich; jede Auth-Fix müsste zweimal passieren.

`[LOW] src/core/mailserver/tests/conformance_test.rs:151-152,292 — Feste Test-Ports`
25250/11430/25251 hartkodiert: parallele Testläufe oder belegte Ports lassen die Suite flaky werden statt Port 0 zu binden.

### healthiest_aspects
- `store/sqlite.rs:76-102` — `with_connection`-Cache + WAL + busy_timeout + Schema-Init in einer Hand; die Hot-Path-Connection-Wiederverwendung ist bewusst gebaut und getestet.
- `smtp/client_queue.rs:231-271,298-325` — Change-Stamp-Idle-Gate gegen Poll-Hammering und ein sauberer Retry-/Backoff-Pfad mit terminalem Status und append-only Delivery-Log: ein echter Zustandsverlauf statt Sonderfall-Stapel.
- `store/sqlite.rs:644-670` + `tests/conformance_test.rs:355-382` — Greylisting mit Loopback-Bypass ist klein, korrekt gerichtet (fail-closed für Externe) und verhaltensgetestet.
- `tests/conformance_test.rs:139-278` — Ein echter End-to-End-Test über Socket: SMTP rein → SQLite → IMAP raus. Das ist die richtige Testform, nur zu wenig davon.

### god_files
Keine Datei >5k Zeilen (größte: `store/sqlite.rs` 965). Zwei God-Funktionen mit klarem Verantwortungs-Mix:
- `smtp/server.rs:49-371` `handle_connection` (~322 Zeilen): Line-Splitting + Auth-Zustandsautomat + Envelope-Parsing + Delivery + DSN-Handling in einem Loop. Schnitt: `session_io` (Zeilenrahmen, Dot-Terminierung), `auth_flow` (PLAIN/LOGIN-Automat — existiert als Enum schon ansatzweise), `message_ingest` (Parsing, Zustellung, Bounce).
- `imap/mod.rs:58-406` `handle_connection` (~348 Zeilen): Tokenizer + Command-Dispatch + FETCH-Rendering + STORE-Semantik. Schnitt: `imap_protocol` (Tokenizer, Sequence-Set), `imap_commands` (pro Kommando ein Handler), `fetch_render` (message_full_raw/header_only).

### text_dispatch
Kontrollfluss über **Fehlertexte**: **0 Stellen** — `err_text` (client_queue.rs:198-213) wird nur persistiert, nie verzweigt; `StalwartError::Smtp { code, .. }` trägt den Code strukturiert. Das Paket ist in dieser Schuldklasse sauber.
Auffällig ist stattdessen Kontrollfluss über **Protokoll-Substring-Matching**: `imap/mod.rs:251-304` (`query_upper.contains("HEADER"/"TEXT"/"UID"/"FLAGS")`) — eine Query, die mehrere Tokens enthält, gewinnt willkürlich der erste `else-if`-Zweig; `client.rs:113` Sonderfall `expected_code == 250 && code == 200` (SMTP kennt kein 200); `dsn.rs:44-54` ratet Bounce-Empfänger aus Freitext.

### test_coverage
Verhaltens-Tests vorhanden und wertvoll: SMTP→IMAP-Roundtrip, Queue-Pooling über zwei Domains, Greylist, DSN-Parsing, iCal/vCard. Aber: `test_dkim_signing` (conformance_test.rs:52-57) pinnt nur das Vorhandensein des Strings `DKIM-Signature:` — es pinnt die gefälschte Signatur statt sie zu verifizieren; `sqlite.rs:918-964` pinnt `EXPLAIN QUERY PLAN`-Texte (Implementierungs-Pin); `sqlite.rs:765-818` zählt Connection-Öffnungen (reiner Impl-Pin); `sqlite.rs:852` nutzt ein Outcome-Vokabular, das die Produktion nie schreibt. Lücken: kein Negativ-Test für Relay-Ablehnung (unauthentifiziert → externe RCPT → 550), kein Auth-Fehler-Test, kein Test für Kleinschreib-Kommandos, kein Test für den Silent-Drop-250-Pfad, kein Test für UID-Persistenz nach Löschung — also genau die fünf HIGH/MED-Verhalten ungetestet.

### effort
**M** — Das Paket ist klein (~4.200 Zeilen), die Architektur-Nähte sind sichtbar. A-Grade erfordert: echtes Passwort-Hashing (argon2/bcrypt) + Migration, Backdoor-Resolver löschen, DKIM entweder RFC-konform (Header-Canonikalisierung, Verifikations-Test) oder Header weglassen statt Müll-Signatur, 250-nur-bei-Zustellung + echten Bounce-Pfad, Case-sensitives Envelope-Parsing, persistente IMAP-UIDs + Flag-/EXPUNGE-Semantik, Resolver-Crate statt Byte-DNS, Outcome-Enum mit CHECK-Constraint, Impl-Pin-Tests durch Verhaltens-/Negativ-Tests ersetzen. Kein God-File-Schnitt nötig vorab; zwei Funktions-Schnitte (SMTP/IMAP `handle_connection`) emitteln sich nebenbei. Kein L, weil keine 25k-Zeilen-Trägheit; kein S, weil Protokoll-Korrektheit (DKIM, IMAP-UIDs) echte Semantik-Arbeit ist.

## P10 — Capabilities

### grade
**C** — Funktional überraschend gereift (Sandbox-Env-Hygiene, Credential-Referenz-Disziplin, SHA-gepinnte Revisionen, echte Verhaltenstests inkl. End-to-End-Execute), aber strukturell tief verschuldet. Das Paket besteht faktisch aus einer einzigen Datei: `scrape.rs` mit 7354 Zeilen (~5380 Produktiv + ~2000 Test), dazu drei 6–11-zeilige Durchreich-Shims (`browser.rs`, `web.rs`, `doc.rs`), die nur `web_stack`/`doc_stack` unter zweitem Namen re-exportieren. Dominante Verfallsmuster: (1) **Sonderfall-Stapel statt Zustandsautomat** in `classify_outcome` — reihenfolgeabhängige Kette aus Payload-Strings, HTTP-Heuristiken und Fehlertext-Matching, mit nachweisbarem Datenverlust; (2) **falsch schreiben → später reparieren**: absolute Skript-Pfade gehen bei jedem Release-Wechsel per Design kaputt und werden im *Lese*pfad repariert; (3) Text-Dispatch auf stderr/Body-Text steuert Status und Materialisierung. Gut ist die Trust-Boundary zum untrusted Runner (env-clear + Allowlist, Prozessgruppen-Kill) und die Credential-Hygiene (nur `ctox-secret://`-Referenzen, per Test abgesichert).

### findings
`[HIGH] scrape.rs:3384-3394 + 774-800` — **Transient-Wort im stderr verschluckt erfolgreiche Runs.** `classify_outcome` prüft `contains_transient_hint(stderr)` *vor* Exit-Code und Records; ein Run mit exit 0 und N gefundenen Records, dessen Skript „timeout"/„ssl"/„429" nach stderr loggt, wird `temporary_unreachable` — und weil Materialisierung (784) und Enrichment (775) nur bei `Succeeded` laufen, gehen die Records **lautlos verloren**, ohne Repair-Queue (`should_queue_repair=false`). Test 6998 deckt nur den Fall „Wörter im Record-Inhalt", nicht den stderr-Fall — die Lücke ist genau der Bug.

`[HIGH] scrape.rs:2547-2575, 2602-2651` — **Reparatur im Lesepfad statt korrekter Schreibpfad.** `load_registered_target` führt bei jedem Load UPDATEs aus und `resolve_registered_script_path` materialisiert Skripte aus `script_body` neu, weil `script_path` absolut in Release-Verzeichnisse zeigt und per Design stale wird (`workspace_belongs_to_stale_release`). Parallelwahrheit `script_body` (Wahrheit) vs. `script_path` (Cache, der lügt) — fünf Tests (5838-6052) pinnen die Reparatur statt des korrekten Schreibens. Relativ-Workspace-Pfade beim Schreiben würden die ganze Reparaturklasse löschen.

`[MED] scrape.rs:956-968, 3292-3331` — **Fehler-Semantik: `ok: true` bei fehlgeschlagenem Run.** Das Outcome setzt `ok` hart auf `true`, auch wenn `status` `blocked`/`portal_drift` ist und `error: Some(...)` gefüllt — der CLI-Prozess exited 0. Automation, die nur Exit-Code/`ok` prüft, hält jeden Fehlschlag für Erfolg; `ok` und `error` widersprechen sich im selben JSON. Zusätzlich ist der `failure_mode`-String-Vertrag aus untrusted Script-stdout doppelt gespiegelt (classify + Payload-Passthrough 842).

`[MED] scrape.rs:5060-5079` — **Run-Lock ohne Liveness-Check.** `create_new(true)`-Lock mit PID drin, die nie geprüft wird: Kill -9 des Executors hinterlässt das Lock dauerhaft, jeder Folge-Run bailt mit „already has an active run" bis zur manuellen Löschung. Kein Test für Lock-Recovery.

`[MED] scrape.rs:5081-5090, 5094-5097` — **Probe-Wahrheit fabriziert + SSRF-Grenze fehlt.** `skip_probe` erzeugt `reachable=true, status=200` aus Konfiguration — die gesamte Klassifikation (inkl. `AuthorizationRequired`-Upgrade über `probe.final_url`) vertraut einer erfundenen Beobachtung. Ohne skip: `ureq` folgt Redirects auf beliebige Hosts aus operator-configurierter URL, kein Scheme-/Host-Allowlist, kein Check gegen link-local/metadata-IPs — für Single-User-CLI grenzwertig akzeptabel, aber undokumentiert und ungetestet.

`[MED] scrape.rs:3472-3488, 3527-3583` — **Sicherheitsrelevante Allowlist per String-Parsing aus untrusted Skript.** `protected_config_from_script` extrahiert `allowed_domains`/`login_url` per `find("const PROTECTED_SOURCE_CONFIG = Object.freeze({")` aus dem hot-revisierbaren (Repair-LLM-beschreibbaren) Adapter-Skript. Die compiled-in Recipe gewinnt nur, wenn sie existiert; bei script-only Adaptern kontrolliert das untrusted Skript die Domain-Allowlist des Reauth-Handoffs, und Formatänderungen (Quotes, Spacing) brechen das Parsing still. Parallelwahrheit Recipe vs. Skripttext, genau die Schuldklasse.

`[MED] scrape.rs:3285-3424` — **Sonderfall-Stapel statt Zustandsautomat.** Zwölf reihenfolgeabhängige Early-Returns mischen explizite Payload-Strings, HTTP-Status-Heuristiken, stderr-Textmatching, Exit-Codes und Record-Zählung; „human verification"/„captcha"/„access denied" im Body (5149-5160) flippen jede Seite mit solchen Wörtern zu `blocked` + Repair-Queue. Jede neue Failure-Art wird als weiterer Ast davor/dazwischen/dahinter geklebt (Capability 10 als nachträgliche Re-Klassifikation in 757-767 zeigt das Muster).

`[LOW] browser.rs:1-11, web.rs:1-6, doc.rs:1-6` — **Null-Wert-Indirektion.** Reine Durchreiche an `web_stack`/`doc_stack` inkl. Re-Export einzelner Typen — zweite Namen für dieselbe Sache, ohne eigenen Vertrag. Entweder echte Capability-Grenze mit eigener Semantik oder löschen.

`[LOW] scrape.rs:460-637, 5326-5366` — **Handgerolltes CLI-Parsing.** `find_flag_value(s)` + in jedem Arm duplizierte mehrzeilige Usage-Strings statt clap-Subcommands; der 634er-Bail wiederholt alle Usages nochmal komplett. Fehleranfällig bei jeder Flag-Erweiterung.

### healthiest_aspects
- `scrape.rs:2949-2977, 3023-3028` — env-clear + explizite Allowlist für den untrusted Runner, mit Test 5379 („drops secrets"); sauber dokumentierte Trust-Boundary.
- `scrape.rs:3490-3501, 6823-6834, 6908` — Credential-Disziplin: nur `ctox-secret://<scope>/<NAME>`-Referenzen, strikt validiert, Test assertiert, dass kein Secret-Wert serialisiert wird.
- `scrape.rs:2640-2643, 2653-2657` — SHA-256-Integritätsgatter, bevor ein Skript wiederhergestellt/ausgeführt wird; Revision-Dedup über Content-Hash.
- `scrape.rs:3964-3968, 4803-4807` — Private-IPC-Erzwingung für lokale Inferenz (kein Loopback-HTTP), plus typisierte Socket-Verträge mit Contract-Tests (5578, 5614).

### god_files
- **`scrape.rs`: 7354 Zeilen** (God-File >5k). Verantwortungs-Mix: SQLite-Schema+Registry, CLI-Dispatch, Prozess-Ausführung, Portal-Probe, Klassifikation, Reauth/Credentials, LLM-Enrichment, Embeddings/semantische Suche, Materialisierung/Delta, Template-Promotion, Repair-Bundles, Locks — plus 2000 Zeilen Tests in-file. **Schnittvorschlag (move-only, seriell):** `cli.rs` (Dispatch/Flags) · `registry.rs` (Schema, targets/scripts/sources, Pfade) · `execute.rs` (Runner-Spawn, Probe, Lock) · `classify.rs` (Outcome→Status als echter Zustandsautomat) · `reauth.rs` (Protected-Config, Session-Expiry, Handoff) · `enrichment.rs` + `semantic.rs` (LLM/Embeddings) · `materialize.rs` · `templates.rs` · `repair.rs`; Tests raus in `tests/`.
- **`execute_scrape_with_outcome` (644-980, ~336 Zeilen)**: Orchestrierung von Lock, Probe, Spawn, Parse, Classify, Reauth, Enrich, Materialize, Repair-Queue, Handoff, Record, Template — nach dem Datei-Schnitt als schlanke Pipeline über die Module neu fassen.
- `register_script` (1843-2044, ~200 Z.) und `handle_scrape_command` (460-637, ~178 Z.) sind die nächsten Kandidaten, bleiben aber unter der 300er-Schwelle.

### text_dispatch
**5 Stellen**, davon 2 mit direkter Feld-Konsequenz:
1. `scrape.rs:3342-3394` — stderr/Probe-Error per Substring („timeout", „ssl", „429", „net::err_") → Status `temporary_unreachable`; **schlimmste Stelle** (silent record drop, s. HIGH #1).
2. `scrape.rs:5149-5160` — HTML-Body-Substrings („captcha", „access denied") → `blocked` + Repair-Queue; False-Positive-Queue-Flut bei legitimem Content.
3. `scrape.rs:3292-3331` — `failure_mode`-String aus untrusted Script-stdout als Status-Dispatch (deklarierter Vertrag, aber stringly und doppelt ausgewertet in 839-843).
4. `scrape.rs:3589-3611` — URL-Path-Substrings („login", „anmeld") für Login-Landing-Erkennung.
5. `scrape.rs:3444-3488` — String-Parsing von JS-Quelltext für die Security-Allowlist (s. MED).

### test_coverage
**35 Tests, überwiegend Verhaltenstests** — das ist die Stärke des Pakets: End-to-End-Multi-Source-Execute mit Fixture-Skripten (6326), Materialisierungs-Deltas insert/update/delete (7027), Klassifikations-Matrix (6652-7024), Reauth-/Login-Landing-Verhalten (6718-6977), Credential-Hygiene (6823, 6908), Socket-Contracts (5578, 5614). Impl-Pins in der problematischen Ecke: fünf Tests (5838-6052) pinnen die *Reparatur* staler Release-Pfade statt korrektes Schreiben. **Lücken:** kein Test für Transient-Wörter im stderr bei Records>0 (maskiert HIGH #1), kein Lock-Stale-Recovery-Test, keine Probe-Tests (skip_probe-Fabrikation, Redirects, Nicht-HTML), kein Test für Exit-Code/`ok`-Semantik des CLI.

### effort
**L** — Datei-Schnitt (move-only, ~8 Module + Test-Auszug) ist Voraussetzung für alles Weitere; danach Klassifikation als Zustandsautomat mit Materialisierungs-Garantie (HIGH #1), Workspace-Pfade relativ schreiben und Read-Path-Repair löschen (HIGH #2), Fehler-Semantik (`ok`/Exit-Code) geradeziehen, Lock-Liveness, CLI auf clap. Realistisch die größte Einzelpaket-Baustelle der W4-Wundertüte.

