```
╔════════════════════════════════════════════════════════════════════════════╗
║                    CTOX — Harness explained                                ║
╠════════════════════════════════════════════════════════════════════════════╣
║                                                                            ║
║  ctox start → systemd Service → Persistent Loop                            ║
║                                                                            ║
║  ┌─────────────────────────────────────────────────────────────────────┐   ║
║  │  Mission Queue (SQLite: runtime/ctox.db, mission-side tables)       │   ║
║  │                                                                     │   ║
║  │  Produzenten (schreiben via mission/channels):                      │   ║
║  │   • TUI Chat-Eingabe       "Deploy the new API version"             │   ║
║  │   • Email-Adapter          "Server disk full - please fix"          │   ║
║  │   • Cron-Schedule          "Run nightly backup verification"        │   ║
║  │   • Plan-Steps             "Step 3: Run integration tests"          │   ║
║  │   • Ticket-Sync (Zammad)   "JIRA-4521: Fix login timeout"           │   ║
║  └────────────────────────┬────────────────────────────────────────────┘   ║
║                           │ lease_next_for_thread()                        ║
║                           ▼                                                ║
║  ┌─────────────────────────────────────────────────────────────────────┐   ║
║  │  CTOX Mission Loop (while service running)                          │   ║
║  │                                                                     │   ║
║  │  FOR EACH task from queue:                                          │   ║
║  │                                                                     │   ║
║  │   ┌────────────────────────────────────────────────────────────┐    │   ║
║  │   │ A. Context aufbauen                                        │    │   ║
║  │   │                                                            │    │   ║
║  │   │ ┌─ System Prompt ──────────────────────────────────────┐   │    │   ║
║  │   │ │  CTOX Mission Control Contract                       │   │    │   ║
║  │   │ │  "You are CTOX, an autonomous agent on this host..." │   │    │   ║
║  │   │ │  Tool-Beschreibungen, Governance-Regeln              │   │    │   ║
║  │   │ └──────────────────────────────────────────────────────┘   │    │   ║
║  │   │                                                            │    │   ║
║  │   │ ┌─ Continuity: Focus (aktualisiert nach jedem Turn) ───┐   │    │   ║
║  │   │ │  "Active task: Deploy API v2.3 to production         │   │    │   ║
║  │   │ │   Status: in-progress                                │   │    │   ║
║  │   │ │   Next: Run smoke tests on staging                   │   │    │   ║
║  │   │ │   Blocker: None                                      │   │    │   ║
║  │   │ │   Done gate: curl /health returns 200"               │   │    │   ║
║  │   │ └──────────────────────────────────────────────────────┘   │    │   ║
║  │   │                                                            │    │   ║
║  │   │ ┌─ Continuity: Anchors (aktualisiert bei Erkenntnissen) ┐  │    │   ║
║  │   │ │  "• Repo: /opt/api at branch release/2.3              │  │    │   ║
║  │   │ │   • Config: /etc/api/config.yaml (port 8443)          │  │    │   ║
║  │   │ │   • DB migration applied: 0047_add_rate_limits.sql    │  │    │   ║
║  │   │ │   • Decision: use blue-green deploy via nginx"        │  │    │   ║
║  │   │ └───────────────────────────────────────────────────────┘  │    │   ║
║  │   │                                                            │    │   ║
║  │   │ ┌─ Continuity: Narrative (komprimierte Historie) ──────┐   │    │   ║
║  │   │ │  "Turn 1: Pulled latest, ran tests — 2 failures.     │   │    │   ║
║  │   │ │   Turn 2: Fixed auth test, DB migration applied.     │   │    │   ║
║  │   │ │   Turn 3: Built Docker image, pushed to registry.    │   │    │   ║
║  │   │ │   Turn 4: Deployed to staging, smoke test passed."   │   │    │   ║
║  │   │ └──────────────────────────────────────────────────────┘   │    │   ║
║  │   │                                                            │    │   ║
║  │   │ ┌─ Task Prompt ───────────────────────────────────────┐    │    │   ║
║  │   │ │  "Deploy the new API version"                       │    │    │   ║
║  │   │ │  (oder re-enqueued: "Continue working on the task") │    │    │   ║
║  │   │ └─────────────────────────────────────────────────────┘    │    │   ║
║  │   └────────────────────────────────────────────────────────────┘    │   ║
║  │                           │                                         │   ║
║  │                           ▼                                         │   ║
║  │   ┌────────────────────────────────────────────────────────────┐    │   ║
║  │   │ B. ctox-core Inner Loop (ein PersistentSession-Client)     │    │   ║
║  │   │                                                            │    │   ║
║  │   │    Think → ToolCall → ToolResult → Think → ToolCall → ...  │    │   ║
║  │   │                                                            │    │   ║
║  │   │    Beispiel:                                               │    │   ║
║  │   │    🧠 "I need to check the current deployment status"      │    │   ║
║  │   │    🔧 shell: kubectl get pods -n api                       │    │   ║
║  │   │    📋 "pod/api-v2.2-abc running since 3d"                  │    │   ║
║  │   │    🧠 "Old version running. I'll deploy v2.3..."           │    │   ║
║  │   │    🔧 shell: docker build -t api:2.3 .                     │    │   ║
║  │   │    📋 "Successfully built abc123"                          │    │   ║
║  │   │    🔧 shell: kubectl apply -f deploy-v2.3.yaml             │    │   ║
║  │   │    📋 "deployment.apps/api configured"                     │    │   ║
║  │   │    ...                                                     │    │   ║
║  │   │                                                            │    │   ║
║  │   │    ┌─ Compact Policy (beobachtet jeden Event) ─────────┐   │    │   ║
║  │   │    │                                                   │   │    │   ║
║  │   │    │  Schicht 1: EMERGENCY                             │   │    │   ║
║  │   │    │    call_input >= 75% × 128K = 98304 Tokens        │   │    │   ║
║  │   │    │    (DEFAULT_CONTEXT_THRESHOLD = 0.75)             │   │    │   ║
║  │   │    │    → ThreadCompactStart (Context wird komprimiert)│   │    │   ║
║  │   │    │    → Context schrumpft z.B. 30K → 5K (83%↓)       │   │    │   ║
║  │   │    │    → Inner Loop läuft auf kompaktem Context weiter│   │    │   ║
║  │   │    │                                                   │   │    │   ║
║  │   │    │  Schicht 2: ADAPTIVE (default 15%)                │   │    │   ║
║  │   │    │    Modell-Output >= 15% des per-call Input        │   │    │   ║
║  │   │    │    (output_budget_pct = 15)                       │   │    │   ║
║  │   │    │    (Drift-Signal: Modell wiederholt sich)         │   │    │   ║
║  │   │    │    → ThreadCompactStart                           │   │    │   ║
║  │   │    │    → Max 1 Compact pro Turn (suppress danach)     │   │    │   ║
║  │   │    └───────────────────────────────────────────────────┘   │    │   ║
║  │   │                                                            │    │   ║
║  │   │    TurnComplete → reply text                               │    │   ║
║  │   └────────────────────────────────────────────────────────────┘    │   ║
║  │                           │                                         │   ║
║  │                           ▼                                         │   ║
║  │   ┌────────────────────────────────────────────────────────────┐    │   ║
║  │   │ C. Continuity Refresh (3× run_turn auf selbem Client)      │    │   ║
║  │   │                                                            │    │   ║
║  │   │  Narrative: "Turn 5: Deployed to prod, smoke test passed"  │    │   ║
║  │   │  Anchors:   + "Production: api-v2.3 running on port 8443"  │    │   ║
║  │   │  Focus:     "Status: done. Done gate: ✓ curl returns 200"  │    │   ║
║  │   │                                                            │    │   ║
║  │   │  → Modell ruft `ctox continuity-update` CLI Tool auf       │    │   ║
║  │   │  → Änderungen als Commits in ctox_lcm.db gespeichert       │    │   ║
║  │   └────────────────────────────────────────────────────────────┘    │   ║
║  │                           │                                         │   ║
║  │                           ▼                                         │   ║
║  │   ┌────────────────────────────────────────────────────────────┐    │   ║
║  │   │ D. Mission-Status entscheiden                              │    │   ║
║  │   │                                                            │    │   ║
║  │   │  Focus sagt status=done  → Mission abschließen             │    │   ║
║  │   │  Focus sagt status=active + Queue leer                     │    │   ║
║  │   │     → Re-Enqueue "Continue working on the task"            │    │   ║
║  │   │  Focus sagt status=blocked → Watchdog / Ticket / Email     │    │   ║
║  │   │  Queue hat nächsten Task → nächste Iteration               │    │   ║
║  │   └────────────────────────────────────────────────────────────┘    │   ║
║  │                                                                     │   ║
║  │  → LOOP zurück zu A. mit nächstem Task                              │   ║
║  └─────────────────────────────────────────────────────────────────────┘   ║
║                                                                            ║
║  ┌─────────────────────────────────────────────────────────────────────┐   ║
║  │  Persistenz (runtime/)                                              │   ║
║  │   ctox.db         Core-State (Queue, Tickets, Plan, Governance,     │   ║
║  │                   Secrets, Communication, Knowledge/Skillbooks/     │   ║
║  │                   Runbooks, LCM: Focus/Anchors/Narrative, Mission-  │   ║
║  │                   State, Verification, Claims)                      │   ║
║  │   ticket_local.db Tool-Store: lokaler Ticket-Adapter                │   ║
║  │   ctox_scraping.db Tool-Store: Scrape-Capability                    │   ║
║  │   documents/ctox_doc.db  Tool-Store: Doc-Stack (FTS5 + Embeddings)  │   ║
║  │   backup/<ISO>/   Archivierte Vor-Merge-Kopien (cto_agent.db,       │   ║
║  │                   ctox_lcm.db) vom einmaligen ctox.db-Merge         │   ║
║  │   engine.env      Settings (CTOX_CHAT_MODEL_MAX_CONTEXT=131072)     │   ║
║  └─────────────────────────────────────────────────────────────────────┘   ║
║                                                                            ║
║  Context Window: CTOX_CHAT_MODEL_MAX_CONTEXT (default 131072 = 128K)       ║
║  Alle Schwellenwerte relativ — funktioniert bei 128K, 256K, jeder Größe    ║
╚════════════════════════════════════════════════════════════════════════════╝
```

## Quellen im Code

| Element | Datei |
|---|---|
| `ctox start` | [src/main.rs:202](src/main.rs#L202) |
| Mission-Loop / systemd-Service | [src/service/service.rs](src/service/service.rs) |
| Zentrale DB-Pfade | [src/paths.rs](src/paths.rs) |
| Einmalige `cto_agent.db` + `ctox_lcm.db` → `ctox.db` Merge-Migration | [src/service/db_migration.rs](src/service/db_migration.rs) |
| `lease_next_for_thread()` | [src/main.rs:1389](src/main.rs#L1389) |
| System Prompt "Mission Control Contract" | [assets/prompts/ctox_chat_system_prompt.md:97](assets/prompts/ctox_chat_system_prompt.md#L97) |
| Continuity-Kinds Loop (Focus/Anchors/Narrative) | [src/execution/agent/turn_loop.rs:764](src/execution/agent/turn_loop.rs#L764) |
| EMERGENCY-Schwelle 75% (`DEFAULT_CONTEXT_THRESHOLD`) | [src/context/lcm.rs:16](src/context/lcm.rs#L16) |
| ADAPTIVE-Schwelle 15% (`output_budget_pct`) | [src/execution/agent/turn_loop.rs:668](src/execution/agent/turn_loop.rs#L668) |
| Context-Window Default 131072 | [src/execution/agent/turn_loop.rs:1232](src/execution/agent/turn_loop.rs#L1232) |
| `ctox continuity-update` CLI | [src/main.rs:509](src/main.rs#L509) · Driver [src/execution/agent/turn_loop.rs:824](src/execution/agent/turn_loop.rs#L824) |
| `runtime/engine.env` | [src/execution/models/runtime_env.rs:9](src/execution/models/runtime_env.rs#L9) |
| `CTOX_CHAT_MODEL_MAX_CONTEXT=131072` seed | [install.sh:1276](install.sh#L1276) |
| Producer Email / Cron / Tickets | [src/mission/communication_email_native.rs](src/mission/communication_email_native.rs) · [src/mission/schedule.rs](src/mission/schedule.rs) · [src/mission/tickets.rs](src/mission/tickets.rs) · [src/mission/ticket_zammad_native.rs](src/mission/ticket_zammad_native.rs) |

## Persistenz-Policy

- **Core-State** lebt in einer einzigen Datei `runtime/ctox.db` (Mission-Queue, Tickets, Plan, Schedule, Governance, Secrets, Communication, Knowledge/Skillbooks/Runbooks, LCM, Verification). Alle Pfade werden über [src/paths.rs](src/paths.rs) zentral aufgelöst — `paths::core_db(root)` ist die Single-Source-of-Truth.
- **Tool-Stores** behalten ihre eigenen Dateien, weil sie in ihrem Tool gekapselt sind: `runtime/ticket_local.db` (lokaler Ticket-Adapter), `runtime/ctox_scraping.db` (Scrape-Capability), `runtime/documents/ctox_doc.db` (Doc-Stack).
- **System-Skills** liegen unter [skills/system/](skills/system) im Repo und werden via `include_dir!` in `tools/agent-runtime/skills/src/assets/samples` in die Binary eingebettet; bei Service-Start extrahiert der Codex-Skill-Manager sie nach `$CODEX_HOME/skills/.system/`. User-Skills — darunter die initial aus [skills/packs/](skills/packs) ausgerollten Packs — bleiben lose im Ordner `$CODEX_HOME/skills/<name>/`, sichtbar und dynamisch veränderbar.
- **Runtime-erzeugte Desk-Skills** (Output von `dataset-skill-creator` / `system-onboarding`) landen als User-Skills im gleichen Ordner, nicht im Repo. Metadaten, Bindings und Embeddings dieser Skills werden über die `knowledge_*`-Tabellen in `ctox.db` persistiert.
- **Erste Start nach dem ctox.db-Merge:** die historischen Dateien `cto_agent.db` und `ctox_lcm.db` werden einmalig in `runtime/ctox.db` konsolidiert und nach `runtime/backup/<ISO8601>/` verschoben — ausgelöst von [src/service/db_migration.rs](src/service/db_migration.rs), aufgerufen früh in `main()` ([src/main.rs](src/main.rs)).

## Änderungen gegenüber der Ursprungsversion

1. **`context-log.jsonl` entfernt** — keine solche Datei im Repo; Token-Metriken leben als `token_count`-Spalten in `ctox.db`.
2. **"Auto-Continuation" aus der Producer-Liste entfernt** — es ist kein Producer, sondern der Status-Pfad "status=active + Queue leer → Re-Enqueue" in Sektion D.
3. **Core-DB-Merge** von `cto_agent.db` + `ctox_lcm.db` auf `runtime/ctox.db` mit Auto-Migration beim ersten Start.
4. **`lease_next()` → `lease_next_for_thread()`** (tatsächlicher Symbolname).
5. **Producer-Zeile Ticket-Sync** konkretisiert auf Zammad (nativer Adapter im Code).
6. **Compact-Policy-Schwellen** mit den echten Konstantennamen annotiert (`DEFAULT_CONTEXT_THRESHOLD`, `output_budget_pct`).
