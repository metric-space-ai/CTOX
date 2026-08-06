# I-064 — RUNDE 1: State-Invariant-Repair — Messung (read-only)

Datum Messung: 2026-08-06  
Arbeitsbaum: `/Users/michaelwelsch/Documents/ctox` (geteilter Checkout, **keine** Datei geaendert)  
Persistenz: `runtime/ctox.sqlite3` (mtime 2026-08-06 00:39, ~1.1 GB)  
Snapshots unter `/Volumes/tmp/ctox-state-backups/` vorhanden, aber fuer den Befund nicht noetig — die Live-DB traegt die Governance-Spuren.

---

## was_geaendert

nichts.

---

## ursache_belegt

### A) `attempt_state_invariant_repair` (`src/core/service/service.rs:1659`)

#### A.1 Welchen Zustand repariert er?

Ausloeser (Boot + Turn-Ende):

- Boot: `run_boot_state_invariant_check` (`service.rs:1426` → `:1460`) nur fuer `CHAT_CONVERSATION_ID = 1`
- Turn-Ende: `run_turn_end_state_invariant_check` (`service.rs:1713` → `:1735`, Callsite `:6484`) pro Job-`conversation_id` (Thread-Key-Hash, `turn_loop.rs:1699`)

Repairable Codes (`has_repairable_state_invariants`, `service.rs:1598`):

| Code | Detektion | Datei:Zeile |
|------|-----------|-------------|
| `closed_mission_with_open_runtime_work` | open Queue (`pending`/`leased`/`blocked`) **oder** nicht-`completed` Plan **und** (`!is_open` **oder** status done/closed/complete **oder** continuation closed/dormant) | `state_invariants.rs:116-134` |
| `idle_allowed_with_open_runtime_work` | open Runtime-Work **und** `allow_idle` | `state_invariants.rs:137-145` |
| `mission_focus_head_mismatch` | `mission_state.focus_head_commit_id != continuity.focus.head_commit_id` | `state_invariants.rs:148-156` |
| `focus_semantic_conflict` | doppelte/konfliktierende Focus-Felder im Focus-Text | `state_invariants.rs:159-166` |

Repair-Schritte (`attempt_state_invariant_repair`, `service.rs:1659-1710`):

1. **Immer:** `engine.sync_mission_state_from_continuity_with_repair` (`lcm/mod.rs:2010`) — laedt/importiert structured state, rendert Focus daraus (`render_focus_continuity_with`, `lcm/mod.rs:4604`). `reopened_for_open_runtime_work` startet hier als `false` (`:2036`).
2. **Bedingt** (nach Re-Eval, Codes `closed_mission_with_open_runtime_work` \| `idle_allowed_with_open_runtime_work` **und** `open_plan_count+open_queue_count > 0`):
   - `record.is_open = true` (`service.rs:1683`)
   - `record.allow_idle = false` (`:1684`)
   - status `done|closed|complete|completed` → `active` (`:1685-1690`)
   - continuation `closed|dormant` → `continuous` (`:1692-1694`)
   - closure_confidence `complete|completed|high` → `low` (`:1696-1701`)
   - sparse Hydration: leeres `mission`/`next_slice` aus open Plan-/Queue-Titles (`hydrate_sparse_open_mission_state_from_runtime`, `:1641-1656`, Call `:1703`)
   - **dauerhaft:** `engine.overwrite_mission_state(&record)` → `persist_mission_state_with` (`service.rs:1705`, `lcm/mod.rs:2154`, `mission_state.rs:384`)

Action-Label: `canonicalized_focus_and_reopened_mission_state` wenn `focus_repaired && reopened_for_open_runtime_work` (`service.rs:1610-1612`).

#### A.2 Wer haette den Zustand richtig schreiben muessen? Ursache noch da?

**Produktions-Before-State (messbar, nicht Review-Vermutung):**

| Metrik | Zahl | Quelle |
|--------|------|--------|
| `turn_state_invariants_repaired` | **54** | `governance_events` `mechanism_id='state_invariant_guard'` |
| davon `violation_codes_before = closed_mission_with_open_runtime_work` | **54/54** | details_json |
| davon `reopened_for_open_runtime_work=true` | **54** | details_json |
| Before-Kombo `status=active \| mode=continuous \| is_open=false \| allow_idle=false` | **54/54** | details_json |
| Before: `mission`+`next_slice`+`done_gate` **alle leer** | **54/54** | details_json |
| Before: `mission_status=done\|closed\|complete` | **0** | details_json |
| After: `is_open=true`, mission aus Queue-Title befuellt | **54** | details_json |
| `boot_state_invariants_repaired` nur `focus_semantic_conflict` (kein Reopen) | **1** (2026-07-05T15:59:18Z) | details_json |
| Repair-Fehlschlag `database is locked` | **1** (2026-06-23T20:21:23Z, Title `Create Create Tiny Business App For`) | `turn_state_invariants_violation` |
| Zeitraum Repair-Feuer | **2026-06-09 … 2026-07-21** (21 Tage mit Events) | created_at epoch-ms |
| Unique conversation_ids mit Repair | **55** (54 turn + 1 boot-focus) | conversation_id |
| Nachher: alle 55 Repair-CIDs `is_open=1` und mission non-empty | **55/55** | `mission_states` |

**Kausalkette (code + Persistenz):**

1. Queue-Tasks leben in `communication_messages` + `communication_routing_state` (`channels/mod.rs:2935+`, JOIN-Messung: aktuell **37 blocked**, **1 pending**, **0 leased**; `planned_goals` total **0**).
2. `conversation_id` = SHA256-Hash des `thread_key` (`turn_loop.rs:1699-1711`), nicht Chat-ID 1 — deshalb viele Mission-Zeilen.
3. Erster Zugriff auf eine neue Conversation initialisiert Focus aus **leerem Template** (`continuity_template` Focus, `lcm/mod.rs:4941-4942`: `Mission:` / `Next slice:` / Contract-Felder ohne Werte).
4. `load_or_import_mission_state_with` (`lcm/mod.rs:4590-4599`) persistiert beim ersten Import `import_legacy_mission_state` → leere Strings.
5. `mission_is_open` (`mission_state.rs:1202-1218`) liefert **`false`**, sobald mission/next_slice/done_gate alle leer sind — **auch wenn** `mission_status`/`continuation_mode` spaeter als active/continuous gelesen werden. Defaults aus leerem Focus-Template + Canonicalisierung erzeugen genau die gemessene Kombo: `active/continuous` + `is_open=0`.
6. Queue-Create schreibt **kein** `mission_states`-Row mit Title/is_open (kein Writer an `create_queue_task` → Mission).
7. Turn-Ende: erst `sync_mission_state_from_continuity` (`service.rs:6479`) — kann sparse Import stabilisieren — dann `run_turn_end_state_invariant_check` (`:6484`) erkennt `closed_mission_with_open_runtime_work` und **zwingt** `is_open=true` + Title-Hydration.

**Wer haette richtig schreiben muessen (Schreibpfade mit datei:zeile):**

| Pflicht | Existierender Pfad | Luecke |
|---------|--------------------|--------|
| Structured Mission beim ersten Fokus-Import nicht „active + is_open=false + leer“ speichern | `import_legacy_mission_state` `mission_state.rs:512` + `mission_is_open` `:1202` + `load_or_import_mission_state_with` `lcm/mod.rs:4590` | Template/Import erzeugt inkonsistente Flags |
| Bei durable open Queue-Work Mission seed-en | `channels::create_queue_task` / Lease→Turn-Start; Turn nutzt `conversation_id_for_thread_key` | **kein** Mission-Seed beim Queue-Create |
| Agent/Focus-Diff setzt Mission-Text nur bei explizitem Diff | `apply_canonical_focus_diff_to_mission_state` `mission_state.rs:624` + `continuity_apply_diff` `lcm/mod.rs:1721` | leerer Startzustand bis Agent schreibt; Queue-Titel kommt erst im Repair |
| Intentionales Schliessen | `defer_mission_for_reason` `lcm/mod.rs:2140` setzt `is_open=false`/`allow_idle=true`/`deferred` | **nicht** der Produktions-Before-State der 54 Repairs |

**Ursache existiert noch:**

- `mission_states`: **662** Rows mit leerem mission+next_slice+done_gate und `is_open=0` (davon **660** `active`+`continuous`) — Stand 2026-08-06.
- `is_open=1`: **55** (genau die reparierten CIDs, alle non-empty).
- Open durable Work weiterhin **37 blocked** Queue-Tasks (Stand Messung; viele `Deployment audit stage:…`, aelteste blocked `2026-06-22`, juengste `2026-07-17`).
- Letzter erfolgreicher Turn-Repair: **2026-07-21T13:14:25Z** (`Freigabe: Filter-Tray…`).
- Kein Clobber-Event (`mission_state_field_clobbered_blocked` = **0**) — Re-Emptying von next_slice/done_gate ist nicht der beobachtete Trigger; der Trigger ist **Erstimport sparse + open Queue**.

#### A.3 Review-Behauptung geprueft (nicht uebernommen)

> „Invariant-Repair halte Missionen aktiv an nie ausfuehrbaren Zeilen offen (`is_open=true` erzwungen).“

| Teil | Urteil | Beleg |
|------|--------|-------|
| Code erzwingt `is_open=true` bei open Runtime-Work | **wahr** | `service.rs:1683` |
| Produktion reoeffnet intentional geschlossene Missionen (`status=done` etc.) | **falsch** | 0/54 Before-States hatten done/closed; alle waren `active/continuous` + leerer Text + `is_open=false` |
| „nie ausfuehrbar“ als generelle Diagnose | **unbelegt / zu weit** | Repair hydriert aus **realen** open Queue-Titles; viele sind spaeter handled/cancelled. **5/20** juengste Repair-Titles liegen **noch** in den aktuellen open subjects (blocked deployment-audit etc.) — das haelt Missionen offen **weil** durable Work `blocked` als open zaehlt (`state_invariants.rs:13`, OPEN_QUEUE_STATUSES), nicht weil der Repair phantom-Work erfindet |
| Unit-Test kann done→reopen simulieren | ja, Test-only | `boot_state_invariant_check_reopens_mission_when_runtime_work_is_still_open` `service.rs:29953` schreibt per SQL `mission_status=done, is_open=0` — **kein** Produktionsereignis dieses Musters in governance_events |

**Fazit Behauptung:** Mechanisch erzwingt der Repair `is_open=true`, aber in Produktion kompensiert er **Split-Brain „active/continuous + leerer sparse Import + is_open=false“ vs. offene Queue**, nicht das Wiederbeleben intentional geschlossener Missionen an toten Zeilen.

#### A.4 Feuert er real? (dauerhafte Spuren)

| reason | action_taken | n | Zeitraum (UTC) |
|--------|--------------|---|----------------|
| `turn_state_invariants_repaired` | `canonicalized_focus_and_reopened_mission_state` | 54 | 2026-06-09 … 2026-07-21 |
| `boot_state_invariants_repaired` | `canonicalized_focus_and_resynced_mission_state` | 1 | 2026-07-05 |
| `turn_state_invariants_violation` | `recorded_state_integrity_alert` | 1 | 2026-06-23 (DB locked) |
| `boot_state_invariants_clean` | `recorded_state_integrity_snapshot` | 1 | 2026-06-10 |
| `boot_state_invariants_not_ready` | `recorded_state_integrity_skip` | 1 | 2026-06-09 |

Top-Title-Haeufigkeiten in Repair-Details (Auszug): `Deployment audit stage: authenticated-multi-user-authz` (10), `app create · new task` (5), diverse `Build/Create *` Business-OS-Apps.

Persistenz der Wirkung: `overwrite_mission_state` schreibt `mission_states` (nicht nur Ringpuffer). Alle 55 Repair-CIDs sind heute noch non-empty + `is_open=1`.

---

### B) `preserve_stale_service_communication_lease_for_specialized_recovery` (`service.rs:2054`)

#### B.1 Welchen Zustand schuetzt er?

Call-Kette Boot (`service.rs:1306` → `release_stale_service_communication_leases_on_boot` `:1870`):

1. **Zuerst** `recover_abandoned_business_os_app_queue_tasks` (`:1874`) — specialized Recovery (Validation/Abschluss grüner App-Module).
2. `channels::release_stale_queue_task_leases` (anderer Pfad, governance `boot_queue_lease_reclaim`).
3. `release_stale_service_communication_leases` (`:1936` / `:1984`):
   - SELECT `communication_routing_state` WHERE `route_status='leased' AND lease_owner=CHANNEL_ROUTER AND acked_at IS NULL` (`:1988-1995`)
   - pro Key: wenn `preserve_…` true → **nicht** releasen (`:2005-2006`)
   - sonst: ggf. business_command transition pending + UPDATE route_status pending, lease null, `last_error='released stale service lease during service boot'` (`:2011-2049`)

Preserve-Praedikat (`:2054-2061`): `load_queue_task` + `business_os_app_module_target_from_metadata` some — nur `ctox.business_os.app.create` \| `ctox.business_os.app.modify` (`service.rs:10690-10698`).

#### B.2 Wer haette die Lease „richtig“ beenden muessen?

| Pfad | Rolle | Datei:Zeile |
|------|-------|-------------|
| Specialized Recovery (Validation green → complete/handle) | **Soll** die App-Lease finalisieren, bevor/statt generic release | `recover_abandoned_business_os_app_queue_tasks` `:18579`, `recover_business_os_app_queue_task_from_validation` `:18875`; Boot ruft das **vor** generic release (`:1874` vor `:1936`) |
| Generic boot reclaim | Crash-Recovery fuer **nicht**-App-Leases | `:1984` + governance `crash_recovery_stale_lease_reclaim` `:1952` |
| Worker mid-slice | separate stuck-lease Protection (andere Mechanismen) | governance: 6× stuck protected lease escalate |

Ursache/Need: **hartes Prozessende** mit lease_owner=service, acked_at null — generic release waere richtig fuer normale Tasks, aber wuerde App-Create/Modify mitten in Validation auf `pending` zuruecksetzen und die specialized green-completion verlieren. Unit-Test `boot_release_preserves_business_os_app_leases_for_specialized_recovery` (`service.rs:40123`) und `boot_recovery_completes_green_business_os_app_lease_before_generic_release` (`:40160`) belegen die Absicht.

#### B.3 Feuert er real? (dauerhafte Spuren)

| Spur | Zahl | Lesart |
|------|------|--------|
| `governance_events` reason=`crash_recovery_stale_lease_reclaim` | **38** Events, Summe `released_count`=**40** | Zeitraum created_at min…max ≈ 2026-06-… bis **2026-07-18T08:56:08Z** — das sind **nicht**-preserved Releases |
| `communication_routing_state` `route_status=leased` jetzt | **0** | keine hangenden preserved Leases im Snapshot |
| historische `last_error` contains `released stale service lease during service boot` | **1** Row (`queue:system::d9305e09…`, jetzt `pending`, updated 2026-06-09) | generic release hat geschrieben |
| Durable Event **wenn** preserve=true | **keines** | Function returned true → `continue` ohne governance (`:2005-2006`) — I-051-Lehre: **kein** dauerhafter Zaehler fuer Preserve-Hits |
| Unit-Test deckt Preserve ab | ja | `service.rs:40123` |

**Strenges Urteil Feuer:** Generic reclaim **feuert** dauerhaft (38×). Preserve-Pfad ist **code-real + unit-getestet**, aber **Produktions-Hit-Count nicht mineable**, weil bewusst still. Indiz: Boot-Reihenfolge laesst specialized recovery zuerst laufen; aktuelle leased=0 impliziert, dass preserved Leases spaeter completed oder nie gleichzeitig stale waren — **kein** Beweis fuer haeufige Preserve-Treffer.

---

## verblieben

1. **Sparse-Import-Split-Brain lebt:** 662× empty active/continuous/`is_open=0` Mission-Rows; naechster Queue-Turn auf neuem thread_key → erneuter Repair (Muster bis mind. 2026-07-21 aktiv).
2. **Repair ist Symptom-Netz, nicht Root-Fix:** Queue-Create seedet Mission nicht; Import schreibt inkonsistente is_open.
3. **Open=`blocked` haelt Missionen offen nach Repair** solange blocked Tasks existieren (37 jetzt) — invariant-korrekt, aber deployment-audit-Blockeds sind langlebig (Juni–Juli).
4. **Ein Repair-Miss** durch DB-lock (2026-06-23) — transient, nicht strukturell.
5. **Preserve-Lease:** kein Audit-Event bei Skip; specialized recovery bleibt noetig vor generic release (Reihenfolge ok).
6. Review-Behauptung „haelt nie ausfuehrbare Missionen kuenstlich offen“ ist **als Produktionsursache widerlegt**; als Nebenwirkung von long-lived `blocked` open work **teilweise beobachtbar**, aber Ursache ist Queue-Lifecycle, nicht der is_open-Zwang allein.

---

## pfade (Runde 2 — Vorschlag, nicht ausgefuehrt)

### `attempt_state_invariant_repair`

| Option | Pfad | Bewertung |
|--------|------|-----------|
| **Root-Writer Queue→Mission** | Bei `create_queue_task` / Turn-Start fuer `conversation_id_for_thread_key`: seed `mission`/`next_slice` aus Title, `is_open=true`, `allow_idle=false` **vor** Turn-Ende | eliminiert 54/54 gemessene Before-States; Repair wird selten |
| **Import-Semantik** | `import_legacy_mission_state` / Defaults: leerer Focus nicht als `active`+`is_open=false` speichern; oder `mission_is_open` an Status koppeln wenn Work open (Vorsicht) | behebt Flag-Inkonsistenz an der Quelle `mission_state.rs:512`+`:1202` |
| **Repair behalten** | `service.rs:1675-1708` als Crash/Split-Brain-Netz | **legitim** — offene durable Work + geschlossene Mission ist echter Integritaetsbruch (Test `:29953`); nicht entfernen nur weil Review „is_open erzwungen“ sagt |
| **Nicht Runde-2-Ziel** | Repair streichen oder `is_open` Force verbieten | wuerde 54 dokumentierte Split-Brains unrepariert lassen |
| **Separat** | Blocked-Queue-Lifecycle (deployment-audit) | verhindert „Mission zeigt ewig blocked Stage“, ist aber **nicht** Invariant-Repair-Bug |

### `preserve_stale_service_communication_lease_for_specialized_recovery`

| Option | Pfad | Bewertung |
|--------|------|-----------|
| **Netz behalten** | `service.rs:2054` + Boot-Order `:1874` vor `:1936` | **legitim** (Crash + specialized App-Validation); Unit-Tests `:40123`, `:40160` |
| Optional Audit | governance_event wenn preserve=true (message_key, module_id) | macht Feuer mineable (I-051) |
| Nicht noetig | Preserve entfernen | wuerde App-Create/Modify Leases nach Crash auf pending werfen und green-completion riskieren |

---

## Methoden-Notiz (I-051)

- Alle Frequenzen aus `runtime/ctox.sqlite3.governance_events` und `mission_states` / `communication_routing_state` / `communication_messages` — **nicht** aus In-Memory-Events.
- `push_event`-Feed allein wurde nicht als Nachweis gewertet.
- Preserve-Skip hat **keine** dauerhafte Spur → „feuert real“ dort nur als Code+Test+Reclaim-Kontext, nicht als Hit-Zaehler.

## Referenzen (Kern)

- Repair: `src/core/service/service.rs:1598-1710`, `:1713-1786`, `:1426-1508`, `:6479-6484`
- Invariants: `src/core/service/state_invariants.rs:58-179`
- Mission open-Logik: `src/core/context/lcm/mission_state.rs:512+`, `:1202-1234`, persist `:384`
- Sync/Import: `src/core/context/lcm/mod.rs:2010-2037`, `:4590-4654`, Template `:4933-4944`
- Lease preserve: `src/core/service/service.rs:1870-2061`, Target-Meta `:10690`, Tests `:40123+`
- Conversation-ID: `src/core/execution/agent/turn_loop.rs:530`, `:1699`
