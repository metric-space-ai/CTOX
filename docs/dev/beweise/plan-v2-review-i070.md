# Adversariales Review — I-070 (`1cbefb0bc`) + Kampagnen-Nebencommits

Read-only, commit-basiert (`git show 1cbefb0bc:<…>`), nicht Arbeitsbaum.

---

## 1) Überschreibt der neue Import-/Reconciliation-Pfad modellgeschriebene Focus-Diff-Felder? (e10735662-Muster)

### ENTKRÄFTET (für Control-Felder auf dem Diff-Pfad) · REST-RISIKO (Text-Asymmetrie)

**Was der Commit tut**

- `import_legacy_mission_state` erzwingt bei leerem Mission/Next/DoneGate jetzt `dormant/dormant/archive` statt Defaults `active/continuous` (`mission_state.rs:563–567`).
- Bei Focus-Diff **nach** One-Time-Import: neuer Pfad `apply_imported_focus_diff_controls` statt reines „imported record stehen lassen“ (`lcm/mod.rs:1728–1743`).
- Explizite Control-Felder aus dem kanonischen Diff werden aus dem **applied** Focus gelesen und gesetzt (`mission_state.rs:710–759`).
- Wenn Mission-Text im Diff berührt wird und Controls unberührt blieben, werden Dormant-Defaults auf `active/continuous/hot` angehoben (`mission_state.rs:765–775`).
- Kommentar und Struktur adressieren e10735662 explizit: Diff bleibt last writer für Controls (`lcm/mod.rs:1726–1729`).

**Warum e10735662 hier nicht 1:1 greift**

- `apply_canonical_focus_diff_to_mission_state` bleibt für Nicht-Import-Pfad der Writer für Mission/Next/Done/Controls (`mission_state.rs:634–671`).
- Import-Pfad liest den Focus **nach** Diff-Anwendung; Legacy-Parser sieht den Modelltext im Dokument.

**Rest-Risiko (nicht entkräftet)**

- `apply_imported_focus_diff_controls` schreibt **keine** Mission/NextSlice/DoneGate-Texte aus dem Diff (`mission_state.rs:707–709` setzt nur `mission_text_touched`). Danach folgt `render_focus_continuity_with` aus dem structured Record (`lcm/mod.rs:1748–1754`) → bei Parser-Missmatch kann der Render den gerade geschriebenen Focus mit dem Import-Record **überschreiben**.
- Andere Import-Call-Sites (`continuity_init_documents`, full-replace, string-replace, `sync_mission_state_from_continuity_with_repair`) rufen `apply_imported_focus_diff_controls` **nicht** (`lcm/mod.rs:1628–1634`, `:1810–1816`, `:1894–1900`, `:2038–2045`). Relevant vor allem für reinen Init/Sync ohne Diff — dort gibt es kein Modell-Diff zu schützen.

**Kleinster entscheidender Check:** Integrationstest „erster Focus-Diff setzt `mission:` + `mission_state: blocked` auf leerem Template“ und assert structured + rendered Focus; plus Negativfall „Legacy-Alias nur im Body, kein kanonischer Diff-Name“.

---

## 2) Queue-Seed in derselben Transaktion? Rollback? Fremde Conversation?

### KONFIRMIERT: gleiche Transaktion · ENTKRÄFTET: Cross-Conversation ohne Hash-Kollision

**Transaktion**

- `create_queue_task_with_metadata`: `conn.transaction()?` → `create_queue_task_with_metadata_tx` → `tx.commit()` (`channels/mod.rs:2714–2718`).
- Innerhalb der TX: Message-Upsert, Thread-Refresh, Routing, **dann** Seed (`channels/mod.rs:2795–2808`).
- Seed ist pure SQL auf dem übergebenen `conn`/`tx` (`lcm/mod.rs:4614–4665`).
- Fehler vor `commit` → Rollback von Message **und** Seed. Kein „halber“ Seed außerhalb der Queue-TX.

**Fremde Conversation**

- `conversation_id = conversation_id_for_thread_key(thread_key)` = SHA-256-Prefix, 62 Bit (`turn_loop.rs:1699–1711`).
- Seed und Turn-Loop nutzen **dieselbe** Funktion; der Seed trifft genau die Conversation, die dieser Thread-Key später sieht.
- Fremd-Treffer nur bei Hash-Kollision zweier thread_keys (vorbestehendes Modell, nicht neu eingeführt). Migration mappt Queue-Titel ebenfalls so (`lcm/mod.rs:4736–4741`).
- Idempotenz: `ON CONFLICT … WHERE trim(mission)='' AND trim(next_slice)='' AND trim(done_gate)=''` (`lcm/mod.rs:4649–4662`) — modellgeschriebene nicht-leere Mission wird nicht überschrieben (channels-Test `channels/tests.rs:2717–2741`).

**Lücke:** Seed setzt bei leerem Tripel **immer** `mission = next_slice = title` und erzwingt active/hot; das ist Absicht, aber kein Schutz gegen Seed auf „bewusst leerem, aber bereits mit Control-Feldern belegtem“ Record (nach Dormant-Import: mission/next/done leer → Seed greift und überschreibt Controls — gewollt für Queue-Promote).

---

## 3) Migration `i070_empty_active_continuous_mission_state_v1`

### KONFIRMIERT: einmal pro DB · TEILWEISE KONFIRMIERT: Kandidatenfilter · REST: Plan-only / Race mit Terminalität

**Einmal pro DB**

- Marker-Tabelle `lcm_data_migrations` + Immediate-TX + Early-Exit wenn Marker existiert (`lcm/mod.rs:4677–4698`, Insert `:4770–4782`).
- Aufgerufen in `LcmEngine` Open-Pfad nach Schema (`lcm/mod.rs:1031–1032`).
- Test: zweiter Pass lässt manuell re-vergiftete `active` unangetastet (`lcm/tests.rs:973–989`) — Marker, nicht Prozess-Cache.

**Was sie erwischt**

```
trim(mission/next/done)='' AND is_open=0
AND status=active AND continuation=continuous
```
(`lcm/mod.rs:4702–4709`, UPDATE `:4747–4761`)

- Passt zur i-064-Messung (662 sparse, active/continuous, is_open=0).
- Legitim „leer + offen“ mit `is_open=1` wird **nicht** angefasst.
- Legitim leere Dormant-Zeilen (nach Fix) ebenfalls nicht (Filter active/continuous).

**Reihenfolge in der Migration**

1. Kandidaten sammeln  
2. Open-Queue-Titel je conversation_id (pending/leased/blocked)  
3. **Alle** Kandidaten → dormant/archive/allow_idle=1  
4. Danach Queue-Seed aus den vorher gelesenen Titeln (`lcm/mod.rs:4746–4768`)

Alles in einer Immediate-TX → kein TOCTOU innerhalb SQLite-Writer.

**Was bleibt problematisch**

| Fall | Wirkung |
|---|---|
| Open **Plan**, keine open Queue | → dormant; Repair-Netz muss später reopenen |
| Queue terminal **vor** Migration | → dormant (korrekt, kein open Work) |
| Queue terminal **während** langer Migration | in derselben TX: Status zum TX-Start; nach Commit inkonsistent bis nächstem Repair/Seed |
| Empty + `active` + `is_open=0` war „warten auf Modell“ ohne Queue | → dormant/allow_idle — **gewollte** ehrliche Lesart |

**Testlücke:** Migrationstest hat **keine** `communication_messages`/Queue-Seed-Spur (`lcm/tests.rs:924–993`) — der „queue_seeded_count“-Zweig ist ungetestet.

---

## 4) Leser von dormant / archive / allow_idle

### KONFIRMIERT: Verhalten ändert sich für mehrere Konsumenten

| Konsument | Datei:Zeile (HEAD/1cbefb0bc-Logik) | Unterschied zu active/continuous |
|---|---|---|
| `mission_is_open` | `mission_state.rs:1315–1331` | `MissionStatus::Dormant` **oder** `ContinuationMode::Dormant` → `false` |
| `mission_allows_idle` | `mission_state.rs:1334–1346` | Dormant status/mode → `true` |
| State-Invariants | `state_invariants.rs:117–145` | `continuation_mode==dormant` + open Work → `closed_mission_with_open_runtime_work`; `allow_idle` + open Work → `idle_allowed_with_open_runtime_work` |
| Turn/Boot-Repair | `service.rs:1683–1694` | setzt `is_open=true`, `allow_idle=false`, continuation `dormant→continuous`; **`mission_status=dormant` wird nicht auf `active` gesetzt** (nur done/closed/complete*) |
| `defer_mission_for_reason` | `lcm/mod.rs:2155–2166` | schreibt deferred + allow_idle (unverändert; anderes Vokabular) |
| Mission-Governor-Prompt | `mission_governor.rs:76–102` | gibt `mission_status` wörtlich in den Prompt („Current mission state: dormant“) |
| Live-Context | `live_context.rs:1415–1418` | rendert Status-String „dormant“ in Focus-Synthese |
| TUI | `tui/render.rs:4217–4221` | bei `is_open && allow_idle` anderer Health-Hint |
| Desktop-UI | `apps/desktop/.../mission.rs`, `overview.rs` | Badges/Farben aus status/mode/trigger; `allow_idle`-Badge |
| Focus-Render | `mission_state.rs:935–954` | kanonischer Focus zeigt dormant/archive |

**Kritischer Folgefehler (KONFIRMIERT)**

Repair bei open Work + Dormant-Status:

```text
is_open = true, allow_idle = false, continuation = continuous,
mission_status bleibt "dormant"
```

(`service.rs:1683–1694`)

Das widerspricht `mission_is_open`/`mission_allows_idle`. Nächster kanonischer Diff-Apply würde `is_open` aus Dormant-Status wieder auf false rechnen. Queue-Seed-Pfad setzt status korrekt auf `active` (`lcm/mod.rs:4645–4652`) — der Repair-Pfad holt den neuen ehrlichen Status **nicht** nach.

---

## 5) Tests: Beweis oder Beschreibung?

### TEILWEISE KONFIRMIERT — beweisen Teile, lassen zentrale Defekte unentdeckt

| Test | Beweist | Unentdeckt bliebe |
|---|---|---|
| `legacy_mission_state_import_…placeholders` (`lcm/tests.rs:909–915`) | Leeres Template → dormant/allow_idle | Parser vs. Diff-Text-Clobber nach Import |
| `empty_active_continuous_rows_migrate_once_per_database` (`:924–993`) | Marker einmal pro DB; re-poison no-op | Queue-Seed-Zweig der Migration; Plan-only; false-positive mit non-empty whitespace-only in einem Feld? (trim greift) |
| `create_queue_task_is_idempotent_…` Erweiterung (`channels/tests.rs:2698–2741`) | Seed schreibt Title/is_open; zweiter Create überschreibt Modelltext nicht | TX-Rollback; Hash-Kollision; Seed bei nur-done_gate |
| `queue_create_seeds_mission_before_turn_end_…` (`service.rs:29978–30029`) | Happy Path: create → continuity_init → clean invariants, kein Repair-Event | Repair mit status=dormant; Migration+open queue; first-import+model-diff |
| `queue_task_metadata_round_trips_…` (`channels/mod.rs:7181–7228`) | **Drive-by**, nicht I-070 | — |

**Gedankliche Gegenprobe:** Defekt „Repair lässt `mission_status=dormant` bei forced open“ wäre grün. Defekt „Migration seeded open queue nicht“ wäre grün. Defekt „apply_imported vergisst Mission-Text und Render clobbert Focus“ wäre grün.

---

## 6) `service.rs`-Testhunk vs. committeter HEAD

### ENTKRÄFTET: zusätzliche uncommittete Symbole · KONFIRMIERT: Commit bricht Compile durch Drive-by `QueueTaskView.metadata`

**Test-Hunk selbst**

Nutzt in `1cbefb0bc` vorhandene Symbole:

- `temp_root`, `SharedState`, `run_turn_end_state_invariant_check` (Modul `tests` mit `use super::*`, `LcmEngine`/`LcmConfig` Import `service.rs:26208–26209`)
- `channels::create_queue_task`, `QueueTaskCreateRequest`
- `turn_loop::conversation_id_for_thread_key`
- `state_invariants::evaluate_runtime_state_invariants`
- `governance::list_recent_events`
- `continuity_init_documents`

Keine Abhängigkeit auf uncommittete Worktree-Symbole.

**Zusätzlicher Compile-Bruch durch denselben Commit**

`QueueTaskView` bekommt Pflichtfeld `metadata: Value` (`channels/mod.rs:286`), befüllt nur in `queue_task_from_message` (`:6286`).

Unvollständige Struct-Literale **im gleichen Tree**:

- `store_projections.rs:1310–1330` (Test)
- `service.rs:44017–44035`, `:44038–44055` (Tests)

→ `missing field metadata` — **I-070 selbst** macht den Tree unbaubar, zusätzlich zu bekannten `1a0fd9fc3`-Lücken. Der Metadata-Roundtrip-Test ist fachfremd zu Split-Brain.

---

## Nebenobjekte (Plausibilität, kurz)

### `abb0c3aea` (origin-Merge, ours CSS)

- Merge `345855c6b` + `c5f13fc91`; Message beschreibt Konflikt in `research/index.css` (border-left `.research-run-note`), lokale Fassung gewinnt.
- First-parent-Diff zeigt `research/index.js`, `queue.rs`, `service.rs` — kein CSS-Restkonflikt im Resultat; CSS existiert weiter in beiden Parents.  
**Plausibel / kein Landungsblocker** für I-070. Kein Beweis, dass die gelöschte Regel regressiv fehlt (out of scope).

### `2b230a3f3` / `e2a8b71b4` (R-01-Methodik)

- `2b230a3f3`: ehrlich partial — HEAD-Scratch unbaubar (`1a0fd9fc3`), 59× unklar mangels Testlauf.
- `e2a8b71b4`: Klassifikation auf **Arbeitsbaum-Snapshot**, nicht HEAD-rein; 0 unklar, aber Basis explizit dirty.  
**Methodik akzeptabel als dokumentierte Einschränkung**, nicht als HEAD-Wahrheit. Kein I-070-Beweis.

---

## Gesamturteil

### **nacharbeit_noetig**

**Nicht** `landung_haelt`: der Kern (ehrlicher Leerzustand + TX-lokaler Queue-Seed + DB-Marker-Migration) adressiert die i-064-Kausalkette richtig, aber die Landung ist **nicht abnahmebereit**:

1. **Harter Compile-Bruch** durch Drive-by `QueueTaskView.metadata` ohne Anpassung aller Struct-Literale.  
2. **Repair-Semantik-Lücke:** `mission_status=dormant` wird bei open Work nicht auf `active` gehoben → inkonsistente Records, Prompt/UI zeigen „dormant“ bei `is_open=true`.  
3. **Testnetz zu dünn** für Migration×Queue, Import×Model-Diff-Text, Repair×Dormant.  
4. **Asymmetrie** Import-Diff vs. Canonical-Diff bei Mission-Text + anschließendem Focus-Render.

**Nicht** `zurueckrollen` als Default: die Richtung ist die richtige Antwort auf i-064 (Erstimport-Split-Brain + fehlender Queue-Writer). Zurückrollen nur, wenn der Compile-Bruch und die Dormant-Repair-Lücke nicht zeitnah in einem Folge-Commit (noch vor I-071) geschlossen werden.

**Minimale Nacharbeit**

1. Metadata-Literale fixen oder Metadata-Hunk aus I-070 herauslösen.  
2. `attempt_state_invariant_repair`: bei Reopen auch `mission_status in {dormant, deferred?}` → `active` (mindestens dormant).  
3. Tests: Migration mit open Queue; first-import Focus-Diff mit Mission+Status; Repair von dormant+open queue.

---

```workjet-completion-receipt-v1
{"schemaVersion":1,"status":"completed","summary":"Adversariales Review I-070 (1cbefb0bc): Kernrichtung (Dormant-Leerimport, TX-Queue-Seed, DB-Marker-Migration) stützt i-064, aber Landung hält nicht — Compile-Bruch durch Drive-by QueueTaskView.metadata, Repair lässt mission_status=dormant bei forced open, Tests beweisen Migration×Queue und Import×Model-Diff nicht. Urteil: nacharbeit_noetig.","changedFiles":[],"verification":[{"command":"git status --porcelain | wc -l","result":"unverändert"},{"command":"git show 1cbefb0bc --stat","result":"6 files, +576/-15 (mission_state,lcm,channels,service tests)"},{"command":"git grep QueueTaskView 1cbefb0bc","result":"3 Struct-Literale ohne metadata-Feld"}],"concerns":["QueueTaskView.metadata Drive-by bricht store_projections+service Tests compile","attempt_state_invariant_repair flipped continuation/is_open/allow_idle but not mission_status=dormant","apply_imported_focus_diff_controls writes no mission text; render can clobber focus","Migration queue-seed branch untested; plan-only open work relies on repair net","R-01 Klassifikation (e2a8b71b4) auf dirty Worktree, nicht HEAD-rein"],"producedPaths":[]}
```
