# GROK-6 — Runde 1 (messen): Ticket-Reconcile in service.rs

Arbeitsbaum: `/Users/michaelwelsch/Documents/ctox` (nur lesen).  
Messzeitpunkt: 2026-08-05 (~20:50–20:55 UTC).  
Persistenz: `runtime/ctox.sqlite3` (mtime 2026-08-05 12:25 local; WAL 2026-08-05 22:05; process-events bis 2026-08-05T20:05:15Z — DB ist lebendig).  
Snapshots unter `/Volumes/tmp/ctox-state-backups/` vorhanden (update-20260718\*), für die Kernaussagen nicht nötig: die Dauer-Spuren sitzen in der Live-Core-DB.

---

## ursache_belegt

### Was der Belang ist

`reconcile_ticket_runtime_state` (`src/core/service/service.rs:19061–19245`) läuft im Idle-Channel-Router vor neuer External-Route (`service.rs:17415–17417`). Davor Idle-Gate:

- `should_skip_idle_ticket_reconcile` `service.rs:17302–17317`
- `mark_ticket_reconcile_ran` `service.rs:17319–17330`
- Safety-Fenster `TICKET_RECONCILE_IDLE_SAFETY_SECS = 3600` (`service.rs:116`)
- Gate-Key: `core_db`-Pfad + Datei-Stamp (main/wal/shm) + sortierte `active_keys` + `last_run`
- Gate-Test pinnt das: `ticket_reconcile_gate_skips_unchanged_idle_state_until_core_db_changes` `service.rs:28145–28180` (erster Lauf muss, idle+unchanged skippt, Queue-Task-Insert invalidiert Gate)

### Welchen Zustand er repariert (vier + ein Side-Effect)

| # | Drift | Reparatur | Code |
|---|--------|-----------|------|
| 1 | Abandoned Business-OS-App-Queue-Tasks | Recovery-Sweep | `recover_abandoned_business_os_app_queue_tasks` `service.rs:19078` |
| 2 | Queue-Task-Lease `leased`, aber abgelaufen / unvollständig und **nicht** in `active_keys` | → `pending`, Lease-Felder null | `channels::release_stale_queue_task_leases` `channels/mod.rs:3494–3654`, Call `service.rs:19094–19138` |
| 3 | Queue-Lease älter als 15 min, **noch** in `active_keys` (Worker mid-slice) | **kein** Release; durable `stuck_lease_escalation` | `list_stale_queue_task_leases` `channels/mod.rs:3462–3485` + `service.rs:19147–19179`; `STALE_QUEUE_LEASE_AGE_MINUTES=15` `channels/mod.rs:3452` |
| 4 | Ticket-Event-Lease `leased` mit `lease_expires_at <= now`, nicht in `active_keys` | → `pending` | `tickets::release_stale_ticket_event_leases` `tickets/mod.rs:2558–2612`, Call `service.rs:19189–19214` |
| 5 | Ticket-Event `blocked` (nicht `waiting_external`), Knowledge/Control inzwischen bereit | → `pending` | `tickets::release_ready_blocked_ticket_events` `tickets/mod.rs:2640–2694`, Call `service.rs:19217–19242` |

Governance-Beschreibung bestätigt denselben Scope: `governance.rs:278–284`  
„Reconciles stale ticket leases and previously blocked ticket events before new external routing.“

### Wer den Zustand richtig hätte schreiben müssen (Schreibpfade)

**Lease anlegen (durabel `leased` + 15-min Expiry):**

- Queue: `channels::lease_queue_task` `channels/mod.rs:3374–3445`  
  schreibt `route_status='leased'`, `lease_owner`, `leased_at`, `lease_expires_at = now+15m` (`:3395`, INSERT/UPSERT `:3412–3414`).
- Ticket-Events: `tickets::lease_pending_ticket_events*` `tickets/mod.rs:2253+`, Expiry `:2316`, UPSERT leased `:2331–2342`.

**Lease am Leben halten (hätte Abort der Stale-Kandidaten):**

- Heartbeat im Prompt-Worker: `PromptWorkerActivity::start` `service.rs:5430–5452`  
  alle 60s `channels::renew_message_leases` (`channels/mod.rs:3657–3679`) und `tickets::renew_ticket_event_leases` (`tickets/mod.rs:2615–2637`) — setzt `lease_expires_at` erneut auf now+15m.
- Worker-Identität (lease-3 / F-002): `channels::record_queue_lease_worker` aus `service.rs:5454–5474` → `channels/mod.rs:3689–3712`.

**Terminaler / korrekter Abschluss (hätte Reconcile unnötig gemacht):**

- In-Memory-Schutz: `leased_message_keys_inflight` / Prompt-Keys → `active_keys` in Reconcile `service.rs:19062–19072`; `release_leased_keys_locked` `service.rs:24796–24806` löscht **nur** den In-Memory-Satz, **nicht** die DB-Route.
- Durable Ack/Handled: z. B. `channels::ack_leased_messages(..., "handled")` (`service.rs:9552`, `20614+`), `set_queue_task_route_status` / Business-OS-Complete-Pfade — das sind die Pfade, die `leased` verlassen sollen, bevor der Lease verfällt.

**`active_keys`-Semantik (wer „richtig“ schützt):**

- Keys aus `pending_prompts.leased_message_keys` / `leased_ticket_event_keys` und, wenn `busy`, `leased_message_keys_inflight` (`service.rs:19062–19072`).
- Fehlt der Key im Prozess (Crash, Drop ohne durable Transition, anderer Prozess-Stand) und Expiry ist durch → Reconcile (oder paralleler Sweep) gibt die Arbeit frei.

### Ursache der Drift — existiert sie noch?

**Ja, und zwar inhärent / legitim als Crash-/Abbruch-Netz, nicht als reiner Schreib-Bug:**

1. **Hard Crash / Prozess-Tod** während `leased`: Heartbeat-Thread stirbt mit dem Prozess → `lease_expires_at` wird nicht erneuert → nach ≤15 min ist die Zeile Sweep-Kandidat. Boot-Pfad greift zusätzlich: `release_stale_service_communication_leases_on_boot` `service.rs:1870–1964` (inkl. `release_stale_queue_task_leases` mit leerem `active_keys` und breiterem non-queue-Lease-Reclaim).
2. **Worker endet ohne durable Transition** (Panic/Drop-Pfad räumt In-Memory, DB bleibt `leased` bis Expiry). Drop-Hook `PromptWorkerActivity::drop` `service.rs:5504+` räumt In-Memory und hat spezielle App-Validation-Cleanup — nicht alle Queue-Leases werden dort hart auf terminal gesetzt.
3. **Mid-slice-Hang mit noch gesetztem active_key**: intentional **nicht** auto-released (`service.rs:19140–19146`); nur Evidence via `stuck_lease_escalation`.
4. **Asynchrone Ticket-Gates**: Block auf Knowledge/Control (`route_ticket_events` `service.rs:19315–19355`); Ready-Release braucht einen späteren Reconcile-Pass, wenn Domains/Control nachgezogen wurden (`tickets/mod.rs:2697–2715`).

**SYNC-D / lease-2/3 haben die Queue-Seite parallelisiert, nicht die Physik entfernt:**

- Dedizierter periodischer Sweep: `run_orphaned_queue_lease_sweep` `service.rs:16791–16881`, Intervall `ORPHANED_QUEUE_LEASE_SWEEP_SECS=60` (`:16788`), **derselbe** `release_stale_queue_task_leases`.
- Pro-Kandidat-Fehlerisolation und CAS auf Lease-Identität: `channels/mod.rs:3550–3578`, `3608–3638`.
- Heartbeat + `lease_worker_id` existieren (s. o.).

Damit: Die **Ursache** (Lease als TTL-Claim über Prozessgrenzen) ist **by design**; die **Schreibpfade** (Lease + Renew + Ack) existieren. Reconcile ist Kompensation, wenn Renew/Ack/Prozess ausbleiben — kein reiner „vergessener Writer“ im Happy Path.

---

## verblieben

### Feuert er real? Dauerhafte Spuren

**Pfad schreibt dauerhaft, wenn er etwas tut:**

- Route-Updates in `communication_routing_state` / `ticket_event_routing_state`
- Governance: `mechanism_id='ticket_reconciliation'` und `'stuck_lease_escalation'` (`service.rs:19121–19237`, Registrierung `governance.rs:278–292`)
- Process-Mining-Trigger auf `communication_routing_state` (Live-DB)

#### A) Governance `ticket_reconciliation` — **13 Ereignisse**, alle Queue-Lease-Releases

| Metrik | Wert |
|--------|------|
| COUNT | **13** |
| reason (alle 13) | `leased ticket-backed queue tasks had no active in-process worker or queued prompt` |
| action_taken (alle 13) | `released stale queue task leases back to pending` |
| MIN(created_at) | `1781493066632` → **2026-06-15T03:11:06.632Z** |
| MAX(created_at) | `1783929861035` → **2026-07-13T08:04:21.035Z** |
| Ticket-Event-Lease-Releases | **0** (kein Event mit „released stale ticket event leases…“) |
| Blocked-Ticket-Event-Releases | **0** (kein Event mit „released blocked ticket events…“) |

Letztes Event-Detail: `{"released_message_keys":["queue:system::03a59d7ab0e2e6ca942e04d8"]}`  
→ **seit ≥23 Tagen kein `ticket_reconciliation`-Produktionsereignis** (Messung 2026-08-05).

Nachlauf der 13 Keys in `communication_routing_state`: alle inzwischen terminal (`handled` / `failed` / `cancelled`) — Reconcile hat sie freigegeben; Downstream hat später abgeschlossen. Beispiel `queue:system::3e77526384577773b8728a02`: process-event `leased→pending` 2026-07-13T04:52:19.753Z (Lease-Alter ~16.7 min) deckungsgleich mit Governance `1783918340007`.

#### B) Governance `stuck_lease_escalation` — **6 Ereignisse**, nur 2026-06-17

| MIN | MAX |
|-----|-----|
| `1781656323099` → 2026-06-17T00:32:03.099Z | `1781698472343` → 2026-06-17T12:14:32.343Z |

Danach: **0**. Keys endeten später `handled`.

#### C) Parallelpfade (gleiche/überlappende Drift)

| Mechanism / Spur | COUNT | Zeitraum (UTC) |
|------------------|-------|----------------|
| `boot_lease_reclaim` (Crash-Boot, non-queue-breiter Claim) | **38** | 2026-06-17 … **2026-07-18T08:56:08.259Z** |
| `boot_queue_lease_reclaim` | **0** | — |
| `orphaned_queue_lease_sweep` | **0** in `governance_events` | Code existiert `service.rs:16850–16866`, Mechanism **nicht** in `DEFAULT_MECHANISMS` (`governance.rs:101+` listet u. a. `boot_lease_reclaim` `:375`, aber **kein** `orphaned_queue_lease_sweep` / `boot_queue_lease_reclaim`) — Insert hat **kein** FK; Fehlen heißt eher: entweder Sweep fand keine `released`-Menge, oder Audit ging verloren unter `record_event_or_count`-Drop (`governance.rs:507–537`) |

#### D) Process-Mining `communication_routing_state` leased→pending

Gesamt **54** Transitions (Fenster process-events ~2026-07-13 … 2026-07-21 für diese Transition; PM-Gesamt-MAX 2026-08-05T20:05:15Z, n=500927).

Buckets nach Lease-Alter (`observed_at - leased_at`):

| Bucket | n | MIN observed | MAX observed |
|--------|---|--------------|--------------|
| age &lt; 1 min (Requeue/sofort) | 22 | 2026-07-13T04:47:29Z | 2026-07-18T12:40:20Z |
| 1–5 min | 14 | 2026-07-13T05:29:23Z | 2026-07-18T09:52:36Z |
| 5–14 min | 11 | 2026-07-13T06:37:51Z | 2026-07-18T12:35:08Z |
| **≥14 min (stale-Kandidat)** | **4** | 2026-07-13T04:52:19Z | **2026-07-21T13:14:25Z** |
| boot_marker (`last_error` boot-Text) | 3 | 2026-07-13T07:56:07Z | 2026-07-18T08:56:08Z |

Die 4 echten Stale-Kandidaten (≥14 min):

1. `queue:system::3e77526384577773b8728a02` — 16.7 min — **mit** `ticket_reconciliation` (s. o.)
2. `queue:system::ec089936cd994d27201f124a` — 19.8 min — 2026-07-16T10:39:33Z — **ohne** `ticket_reconciliation`-Row
3. `queue:system::7d4937f284aae9cbc96cf8b6` — 16.3 min — 2026-07-18T09:13:14Z — ohne
4. `queue:system::009d85857b3b994cd375fb47` — 21.9 min — 2026-07-21T13:14:25Z — ohne

→ Nach 2026-07-13 laufen Stale-Recoveries **noch** (bis mind. 2026-07-21), aber **nicht** mehr unter dem Audit-Label `ticket_reconciliation`. Attribution unscharf (orphaned-Sweep vs. Reconcile vs. andere `pending`-Setter); durable **Wirkung** (leased→pending) ja, durable **ticket_reconciliation-Governance** nein.

Hinweis: Process-Event-JSON deckt nur ein Kernspalten-Subset ab (`message_key, route_status, lease_owner, leased_at, acked_at, last_error, updated_at`) — **kein** `lease_expires_at` im Trigger-Payload. Alter basiert daher auf `leased_at`, nicht auf Expiry-Feld; für ≥14 min ist das trotzdem ein starkes Stale-Signal (TTL=15, Heartbeat=60s).

#### E) Ticket-Event-Seite in Persistenz — kalt

- `ticket_event_routing_state`: nur Status `observed` × **3**; **0** `leased`, **0** `blocked`, **0** `leased_expired`
- `ctox_process_events` für `ticket_event_routing_state`: **0** Rows
- Kein Governance-Event für Ticket-Event-Lease- oder Blocked-Release

→ Die Ticket-Event-Zweige des Reconciles haben in dieser DB **keine** dauerhafte Produktionswirkung hinterlassen. Das ist **kein** Beweis für Totsein des Codes (Tests in `tickets/tests.rs:748+`, `869+`), aber **Beweis für Null-Kandidaten / Null-Wirkung** im Live-Store.

#### F) Live-Inventar jetzt (2026-08-05)

`communication_routing_state` Status-Counts:

- blocked 37, cancelled 567, failed 68, handled 430, **pending 1**, **leased 0**
- Stale-leased-Kandidaten (Query analog Sweep): **0**
- Aktuelle `leased`/`running`-Rows: **0**

`ticket_event_routing_state`: kein Sweep-Futter (s. E).

### Interpretation „null Ereignisse“

- **Nicht** tot wegen fehlender Governance allein: Process-Events belegen leased→pending bis 2026-07-21; Boot-Reclaim bis 2026-07-18.
- **Ticket-Reconcile-spezifisches Audit** (`ticket_reconciliation`) endet 2026-07-13; seitdem **0** unter diesem Mechanism — parallel existiert derselbe Queue-Sweep im 60s-Orphan-Pfad.
- **Ticket-Event- und blocked-Release-Zweige**: dauerhaft **0** Wirkung in dieser Persistenz über den gesamten Governance-Horizont (ab 2026-06-15).
- Live: **0** offene Stale-Kandidaten — Reconcile findet derzeit nichts zu heilen, selbst wenn er läuft (Gate erlaubt Lauf bei DB-Churn; Wirkung wäre no-op + `mark_ticket_reconcile_ran`).

### SYNC-D-Bezug (kurz)

lease-2/3 + router-3 (Kommentare `service.rs:16791+`, `19040+`, `channels/mod.rs:3448+`) haben:

- denselben Queue-Release **außerhalb** des Ticket-Reconciles verdrahtet,
- Stuck-Protected-Leases als Evidence modelliert,
- Heartbeat/Worker-Id als Proper Writer gestärkt.

Das erklärt, warum `ticket_reconciliation` nach Mitte Juli **audit-seitig** abflacht, ohne dass Lease-Physik verschwindet: das Netz ist **verteilt** (boot + orphan-sweep + ticket-reconcile).

---

## pfade

### Runde-2-Empfehlung

**Kein blindes Streichen.** Begründetes Ergebnis:

#### 1) Queue-Stale-Lease-Release im Ticket-Reconcile — **Netz legitim, aber redundant**

- **Ursache bleibt:** Prozess-/Worker-Ausfall über TTL hinweg ist real (4 Stale-Kandidaten process-side bis 2026-07-21; 13 historische ticket_reconciliation; 38 boot_lease_reclaim).
- **Proper Writer** existieren (Lease/Renew/Ack) und sind post-SYNC-D besser; sie können Hard Crash nicht ersetzen.
- **Doppelung:** `reconcile_ticket_runtime_state` und `run_orphaned_queue_lease_sweep` rufen **dieselbe** `release_stale_queue_task_leases` auf. Runde 2, falls Kompensation abbauen: **eine** kanonische Owner-Schleife (vermutlich orphan-sweep + boot), Ticket-Reconcile von Queue-Lease-Sweep **entkoppeln** — aber nur mit Nachweis, dass orphan-sweep wirklich durable audited (aktuell 0 `orphaned_queue_lease_sweep`-Events; Mechanism-Registrierung fehlt in `DEFAULT_MECHANISMS`).

Pfade Runde 2 (Queue-Redundanz):

1. `src/core/service/service.rs:19094–19138` vs `:16797–16881` — Ownership klären, eine Call-Site behalten.
2. `src/core/service/governance.rs` — `orphaned_queue_lease_sweep` / `boot_queue_lease_reclaim` in `DEFAULT_MECHANISMS` nachziehen, sonst bleibt Audit dunkel.
3. Gate/Test: Idle-Gate `28145` bleibt gültig für den verbleibenden Reconcile-Rest.

#### 2) Stuck-Lease-Escalation — **legitim (Evidence-only), kalt seit 2026-06-17**

- Design: nicht releasen, wenn Key protected (`service.rs:19140–19146`).
- Live: 6 Events an einem Tag; seither 0. Behalten als Audit-Netz; Runde 2 nur wenn Process-Mining denselben Fall schon hard-bindet und Duplikat stört.

#### 3) Ticket-Event-Lease-Release + ready-blocked-Release — **Netz legitim (asynchron), in dieser Persistenz wirkungslos**

- Proper Writer: Lease `tickets/mod.rs:2316+`, Renew `2615+`, Block in `route_ticket_events` `service.rs:19315+`, Ready-Check `ticket_event_ready_for_preparation` `tickets/mod.rs:2697+`.
- Ursache „blocked bis Knowledge/Control da ist“ und „Lease ohne Renew nach Crash“ bleiben architektonisch möglich.
- Persistenz: **0/0** dauerhafte Releases. Das ist **kein** Totenschein des Codes, aber **kein** Produktionsdruck zum Erhalt **innerhalb** des dicken Router-Reconciles.

Runde 2 Optionen (nur messen→entscheiden, hier schon vorbereitet):

- **Behalten** als dünnes Safety-Netz vor Ticket-Routing (Beschreibung in `governance.rs:284` trifft genau das), **oder**
- **Verschieben** an den Ticket-Sync-/Route-Eingang (call-site näher am Writer), wenn der Router-Reconcile abgespeckt wird.
- **Nicht** streichen ohne Ersatz, solange Lease-TTL + asynchrone Gates existieren — sonst hängen Events nach Crash/Gate-Heilung dauerhaft.

#### 4) Abandoned BOS-App-Queue-Recovery im Reconcile

- Ebenfalls mehrfach verdrahtet (boot `1874`, idle snapshot, pre-lease, reconcile `19078`).  
- Runde 2: gleiche Ownership-Frage wie Queue-Lease; hier nicht weiter aufgelöst (nicht Kern der Ticket-Lease-Frage).

### Klare Gesamtaussage

| Frage | Antwort |
|-------|---------|
| Welchen Drift heilt er? | Vor allem **verwaiste Queue-Task-Leases**; sekundär Stuck-Evidence; Ticket-Event-Lease/Blocked in Code, in Live-DB **ohne Spur**. |
| Wer hätte richtig schreiben müssen? | Lease-Owner-Heartbeat + durable Ack/Handled (s. Schreibpfade oben). Crash bricht das absichtlich. |
| Ursache noch da? | **Ja** — TTL-Lease über Prozessgrenzen + asynchrone Ticket-Gates. Proper Writer post-SYNC-D besser, ersetzen Crash-Recovery nicht. |
| Feuert real? | **Historisch ja** (13× ticket_reconciliation 2026-06-15–07-13; 6× stuck 2026-06-17). **Seit 2026-07-13 kein ticket_reconciliation-Event**; Stale-leased→pending process-side noch bis 2026-07-21 (Attribution unscharf). **Jetzt: 0 Kandidaten.** |
| Runde 2 nötig? | **Nicht** als „Bugfixen“, sondern optional **Ownership/Redundanz** (Queue-Sweep doppelt) und **Audit-Lücken** (orphan mechanism). **Ticket-Event-Zweige:** behalten als legitimes Netz oder enger an Ticket-Route hängen — **kein** begründetes „tot, löschen“ aus Null-Events, weil der Pfad bei Kandidaten dauerhaft schreiben *würde* (Code+Tests), aber Live nie Kandidaten hatte. |

**Begründetes Nein zur Entfernung des gesamten Belangs:** Hard-Crash- und Mid-slice-Evidence-Netze sind legitim.  
**Begründetes Ja zur Runde-2-Verkleinerung nur der Queue-Lease-Hälfte im Ticket-Reconcile**, *falls* orphan-sweep+boot als einzige Owner mit funktionierendem Audit nachgewiesen werden.

---

## Mess-Anker (Roh)

- Funktion: `src/core/service/service.rs:19061`
- Gate: `service.rs:17302–17330`, Test `:28145`, Safety 3600s `:116`
- Queue release: `src/core/mission/channels/mod.rs:3494`
- Ticket event release: `src/core/mission/tickets/mod.rs:2558`, blocked `:2640`
- Heartbeat writers: `service.rs:5441–5450`
- Orphan twin: `service.rs:16797`
- DB: `runtime/ctox.sqlite3`
  - `ticket_reconciliation` n=13, last `1783929861035` (2026-07-13T08:04:21.035Z)
  - `stuck_lease_escalation` n=6, last `1781698472343` (2026-06-17T12:14:32.343Z)
  - `boot_lease_reclaim` n=38, last `1784364968259` (2026-07-18T08:56:08.259Z)
  - leased→pending process n=54; age≥14min n=4 (last 2026-07-21T13:14:25.202Z)
  - live stale leased queue candidates: 0; ticket leased/blocked: 0
