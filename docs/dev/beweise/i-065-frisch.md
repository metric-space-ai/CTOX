# I-065 — RUNDE 1: Stalled-Communications-Repair

Messung (nur Lesen) im geteilten Checkout `/Users/michaelwelsch/Documents/ctox`.
Zeitpunkt: 2026-08-06. Service-PID 763 laeuft seit 2026-07-31 08:34 (`runtime/ctox_service.pid`).

## was_geaendert

Nichts. Keine Datei angefasst, kein Commit, kein cargo.

## ursache_belegt

### Was die drei Funktionen reparieren

1. **`repair_stalled_founder_communications`** — `src/core/service/service.rs:20246`
   - Zuerst: stale self-work nach bereits reviewed+sent Reply schliessen (`close_stale_founder_communication_self_work_after_reviewed_reply`, :20901).
   - Kandidatensatz A — **falsch-terminal handled**: `channels::list_unreviewed_handled_inbound_messages` (`src/core/mission/channels/mod.rs:2300`)
     - `direction=inbound`, `channel IN ('email','jami')`, `route_status='handled'`, **kein** `communication_founder_reply_reviews` mit `sent_at IS NOT NULL`.
   - Kandidatensatz B — **steckende Review/Fail-Zustande**: `channels::list_stalled_inbound_messages` (`mod.rs:2240`)
     - externe Kanaele (`email,jami,teams,whatsapp,meeting,slack,discord,telegram,matrix,mattermost,zulip,google_chat`),
     - `route_status IN ('failed','review_rework')`.
   - Nur **Founder/Owner/Admin-Email** (via `is_founder_or_owner_inbound_message`, `service.rs:20094`, `channel=='email'` + role) laufen den Founder-Pfad.
   - Nicht-Founder mit reviewed external chat: `repair_stalled_external_chat_message` (`service.rs:20607`).
   - Aktionen u.a.: `ack` → `pending`/`handled`/`cancelled`, Rework spawn (`create_founder_communication_repair_rework`, :21096), Budget-Eskalation (`escalate_exhausted_founder_communication`, :20526).

2. **`repair_stalled_external_chat_message`** — `service.rs:20607`
   - Spiegel des Founder-Sweeps fuer `is_reviewed_external_chat_channel` (`outbound_review.rs:1350`: teams/jami/whatsapp/meeting/slack/discord/telegram/matrix/mattermost/zulip/google_chat).
   - Terminal NO-SEND / auto-submitted → `handled`; reviewed send → `handled`; neueres Inbound im Thread → `cancelled`; Budget leer → Chat-Eskalation; sonst `pending`.

3. **`create_founder_communication_repair_rework`** — `service.rs:21096`
   - Schreibt **dauerhaft** ein self-work-backed Queue-Item (`kind=founder-communication-rework`, `dedupe_key=founder-communication-rework:{inbound}`, `repair_reason` in metadata) ueber `create_self_work_backed_queue_task` (:23609).

Aufruf: Channel-Router `route_external_messages` (`service.rs:17386`) — unter Queue-Pressure (:17436) und im normalen Tick vor Leasing (:17567).

### Wer haette den Zustand richtig schreiben muessen (Schreibpfad)

**Normaler Zustell-/Abschluss-Pfad (Founder-Email):**

| Schritt | Ort | Soll-Zustand |
|--------|-----|--------------|
| Inbound-Lease → Worker | `route_external_messages` lease/enqueue um `service.rs:17763–17850` | `route_status=leased` |
| Completion-Review PASS | `run_completion_review` `service.rs:8826–8841` | `record_founder_reply_review_approval` + **`send_reviewed_founder_reply`** (`outbound_review.rs:1978–2041`) setzt `communication_founder_reply_reviews.sent_at` |
| Terminal Ack | `should_handle_messages` → `terminalize_reviewed_queue_messages` `service.rs:7313–7324`, `:9503–9552` | `ack_leased_messages(..., "handled")` |
| Guard gegen vorzeitiges handled | `guard_founder_handled_ack` `outbound_review.rs:1941–1975` | bail, wenn Founder-Email handled **ohne** non-synthetic sent review (Ausnahmen: auto-submitted, terminal NO-SEND) |

**Fehl-/Stuck-Pfade, die Kandidaten fuer den Repair erzeugen (ursache im Code noch vorhanden):**

| Ursache | Schreibpfad | Zustand, den Repair sieht |
|--------|-------------|---------------------------|
| Review FAIL / FeedbackRetry mit `persist_on_leased_queue` | `service.rs:7368–7441` → `ack ... "review_rework"` | stalled (`review_rework`) |
| Worker-Crash / Lease-Leak | Drop-Cleanup `service.rs:5620–5628` → `ack ... "failed"` + failure_reason | stalled (`failed`) |
| Review PASS, Send scheitert | `service.rs:8880–8893` → `Hold{Technical: reviewed-communication-send}` | `hold_leased_messages` (`mod.rs:2476`): Technical/Missing* → nach Budget **`pending`** (Retry) oder **`failed`** (exhausted); **nicht** `blocked` |
| WaitingExternal NO-SEND | `service.rs:8937–8953` → `Hold{WaitingExternal}` | **`blocked`** — **nicht** in `list_stalled_inbound_messages` |
| handled ohne sent review (Bug/Race) | waere gegen `guard_founder_handled_ack` — Repair A hebt es wieder an | unreviewed handled |

**Fazit Ursache im Code:** Ja, die Mechanismen, die `failed`/`review_rework`/„handled ohne sent review“ erzeugen koennen, existieren weiter. Der Repair ist ein bewusster Safety-Net, kein toter Kompensations-Stub.

### Feuert er real? (nur dauerhafte Spuren)

`push_event` (`service.rs:24644–24653`) schreibt in einen **In-Memory-Ringpuffer `recent_events` (Kapazitaet 24)** — zaehlt **nicht** als dauerhafte Spur (Lehre I-051).

**Live-DB `runtime/ctox.sqlite3`** (mtime 2026-08-06 00:39, WAL aktiv):

| Metrik | Zahl | Zeitraum / Hinweis |
|--------|------|--------------------|
| `communication_messages` channel/direction | nur `queue/inbound=1099` (observed 2026-06-09 … 2026-07-21), `tui/outbound=3` | **0 email/chat inbound je** |
| Accounts | `queue:system`, `tui:default`, `tui:local` | **kein email-/chat-Account** |
| `communication_sync_runs` | **0** | nie Sync-Lauf protokolliert |
| `owner_profiles` | **0** | |
| stalled-Kandidaten (`list_stalled…`-Semantik) | **0** | |
| unreviewed handled email/jami | **0** | |
| `communication_founder_reply_reviews` | **0** | |
| Queue-Subject/Body/metadata `Founder communication rework` / `repair_reason` / `inbound_message_key` like `email:%` | **0** | |
| `ticket_self_work_items` kind `founder-communication-rework` | **0** | kinds nur build/compliance/execute/harness/… |
| `ticket_self_work_notes` mit Founder/repair | **0** | 4 Notes total, authors `ctox`/`ctox-agent` |
| Routing gesamt | cancelled 567, handled 430, failed 68, blocked 37, pending 1 — **alles queue(+3 tui handled)** | failed max_u 2026-07-21; blocked max_u 2026-07-17 |
| failed last_error (Stichprobe) | Session-Timeouts, finite review budget, Business-OS app validation — **queue:system::…** | keine Founder-Mail |

**Backups** `/Volumes/tmp/ctox-state-backups/update-20260718T{073042,101507,115230}Z/ctox.sqlite3`:
- jeweils nur `queue/inbound` (~1094–1096) + `tui/outbound=3`
- stalled=0, unreviewed_handled=0, founder_reviews=0, founder_rework=0  
→ **kein historischer Founder-/Chat-Inbound in diesem State-Baum seit mindestens 2026-07-18**.

**Logs:**
- `runtime/ctox_service.log` (30 942 976 B, mtime 2026-08-06 00:26): Treffer **0** auf `Repaired .*stalled founder`, `Restored stalled founder`, `Escalated exhausted`, `founder-communication-rework`, `review_rework`.
- `runtime/context-log.jsonl` (121 523 661 B, Events bis ~ts 1784883335 ≈ 2026-07-24): **0** `Founder communication` / `Restored stalled` / `email:founder`; `review_rework` nur als Agent-Tool-`ctox queue list` auf Queue-Items.
- `governance_events`: kein Mechanism fuer Founder-Repair. Einmal `queue_pressure_router_skip` (2026-07-13T00:41:17Z), einmal `channel_router_loop_active` (2026-06-15T13:17:04Z) — Router laeuft, hat aber **keine** Repair-Wirkung hinterlassen.

**SYNC-D (I-016 Outbox fail-closed Ack, I-052 dauerhafte Ack-Fehler):** betreffen `business_command_*`/Outbox-Projektion, nicht `communication_routing_state` der Founder-Mail. In der Live-DB gibt es **keine** Founder-Stalls, die durch Outbox-Ack-Fixes haetten „verschwinden“ oder „auftauchen“ koennen. Die 68 `failed` Rows sind durable-queue Business-OS/AppSec-Arbeit, **ausserhalb** der Repair-Selektoren (Selektoren verlangen non-queue Channel-Listen).

**Antwort auf „findet der Repair seit SYNC-D noch Kandidaten?“:** In diesem Runtime-Zustand **nein — und er hat nach messbarer Persistenz (Juni–Aug 2026, inkl. Jul-18-Backups) nie Founder-/Chat-Kandidaten gefunden**, weil der Ingest-Pfad nie email/chat Messages geschrieben hat. Das ist **kein** „null Ereignisse = toter Code“-Fehlschluss: die Repair-Wirkung waere in SQLite sichtbar (route_status-Flip, rework-queue, self-work, founder_reviews, repair_reason); nichts davon existiert. Der Code-Pfad bleibt live und wird pro Router-Tick aufgerufen — er laeuft ueber leere Selektoren.

### Existiert die Ursache noch?

- **Ja im Code:** incomplete Review/Send → `failed`/`review_rework`; Crash → `failed`; Guard verhindert handled-ohne-Send fuer Founder-Email; Repair A bleibt fuer den restlichen Fall.
- **Nein als beobachtetes Live-Problem in dieser DB:** Es gibt schlicht **keine** Founder-/Chat-Inbounds. „Nachrichten stecken“ in diesem Checkout ist **nicht** durch steckende `communication_routing_state` fuer email/chat belegt.
- Jami-Filesystem `runtime/communication/jami/{inbox,outbox,raw,archive}`: leer (nur Verzeichnisse seit 2026-06-21).

## verblieben

1. **Upstream-Luecke, nicht Repair-Luecke:** Ohne email-Account / Sync (`communication_accounts` ohne email, `sync_runs=0`, `owner_profiles=0`) entsteht der zu reparierende Zustand nie. Wenn der User subjektiv „Founder-Mails stecken“, liegt der naechste Messpunkt am Adapter (`sync_configured_channels` `service.rs:23912`, email `service_sync`) und Settings — nicht am Safety-Net.
2. **Selektor-Blindspot `blocked`:** `list_stalled` sieht nur `failed|review_rework`. `WaitingExternal` → `blocked` (`hold_leased_messages` `mod.rs:2491–2533`) wird vom Repair **nicht** angefasst. Technical-Holds nach Send-Fail gehen nach Budget in `pending`/`failed` und koennen wieder Kandidaten werden — das Netz ist hier teilweise legitim (Retry/Budget).
3. **Observability:** Erfolgreiche Founder-Repairs hinterlassen bei Erfolg vor allem `push_event` (RAM) plus SQLite-Seiteneffekte. Es gibt **keinen** `governance_events.mechanism_id` fuer den Repair selbst → spaetere Messungen muessen SQLite (rework metadata, route flips) abfragen, nicht nur Events.
4. **Test-Drift-Hinweis (nur Code-Lesen):** Fixture um `service.rs:40965` kommentiert absichtlich „red“: blocked rework vs. neue Eskalations-Semantik (`founder_communication_review_budget_exhausted` behandelt self-work `blocked` als exhausted, :20482–20491). Produktverhalten und alter Test-Kommentar koennen auseinanderlaufen — **keine** Live-Wirkung ohne Founder-Daten.
5. **Queue-failed vs. Founder-stalled:** 68 durable `failed` Queue-Tasks sind **kein** Beleg fuer Founder-Stall; Repair filtert sie aus.

## pfade

### Runde 2 — nur wenn Produktziel „Repair haertet Zustellung“

Unter diesen Pfaden messen/aendern (hier **nicht** getan):

1. **Ingest zuerst** (sonst bleibt jeder Repair-Test syntetisch):
   - `src/core/service/service.rs:23912` `sync_configured_channels` / email adapter
   - `communication_accounts`, `communication_sync_runs`, `owner_profiles` in `runtime/ctox.sqlite3`
2. **Normaler Schreibpfad haerten / bezeugen:**
   - `run_completion_review` `service.rs:8540` (Send + Hold)
   - `send_reviewed_founder_reply` `outbound_review.rs:1978`
   - `terminalize_reviewed_queue_messages` `service.rs:9503`
   - `guard_founder_handled_ack` `outbound_review.rs:1941`
3. **Repair-Kandidaten erweitern (falls blocked-Waits stecken bleiben sollen):**
   - `list_stalled_inbound_messages` `channels/mod.rs:2240` — heute ohne `blocked`
   - Spiegel in `repair_stalled_external_chat_message` `service.rs:20607`
4. **Dauerhafte Observability:**
   - analog `queue_pressure_router_skip` / `channel_router_loop_active` ein governance mechanism beim realen Repair-Commit (route flip / rework create), damit „null Events“ nicht wieder missverstanden wird.
5. **Nicht noetig als Runde-2-„Netz entfernen“:** Das Safety-Net ist fuer **harten Crash / Review-Budget / Send-Fail** legitim. In dieser Persistenz ist es unbenutzt, aber nicht widerlegt. Entfernen waere spekulativ.

### Klare Einordnung

| Frage | Antwort |
|-------|---------|
| Welchen Zustand? | Founder/Owner-Email (und reviewed Chat) in `failed`/`review_rework`, oder `handled` ohne sent reviewed reply |
| Wer haette richtig schreiben muessen? | Completion-Review PASS → `send_reviewed_founder_reply` + danach `ack handled`; sonst review_rework/failed/hold |
| Ursache noch im Code? | Ja |
| Feuert real (dauerhaft)? | **Nein** in Live+Backups 2026-06-09…2026-08-06 — **0 Kandidaten, 0 Rework-Artefakte** |
| SYNC-D Einfluss messbar? | **Nein** auf diesem Belang |
| Runde 2 noetig zum Entfernen des Netzes? | **Nein** — Netz ist legitim; leere Kandidaten = fehlender Email/Chat-Ingest, kein toter Code |

## Zahlen-Kurzprotokoll (Copy-Paste)

```
DB: runtime/ctox.sqlite3
  messages: queue/inbound=1099 (2026-06-09T16:10:40Z..2026-07-21T12:48:59Z), tui/outbound=3
  email|jami|chat inbound: 0
  stalled failed|review_rework (external channels): 0
  unreviewed handled email|jami: 0
  founder_reply_reviews: 0
  founder-communication-rework self_work|queue|notes: 0
  accounts: queue:system, tui:default, tui:local
  sync_runs: 0
  owner_profiles: 0
  routing: cancelled=567 handled=430 failed=68 blocked=37 pending=1 (queue-dominated)

Backups 2026-07-18: same shape, 0 founder/chat

Logs: ctox_service.log + context-log.jsonl → 0 Restored/Repaired stalled founder
governance: no founder-repair mechanism_id
push_event: in-memory only (capacity 24) — not durable
```
