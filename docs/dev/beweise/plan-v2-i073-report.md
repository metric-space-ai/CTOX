## Ergebnis

I-073 ist auf der Arbeitsbaum-Basis umgesetzt.

### Entscheidungen

- **Mechanism-ID:** `orphaned_queue_lease_sweep`
  - Der Call-Site-Audit existierte bereits, war aber nicht in `DEFAULT_MECHANISMS` registriert.
  - Registrierung war notwendig, weil Governance-Abfragen die Ereignisse über einen Inner Join mit `governance_mechanisms` ausgeben.
  - Eingetragen als:
    - Klasse: `recovery`
    - Autonomie: `autonomous_queue_lease_recovery`
    - Sichtbarkeit: `inventory_only`
  - Die Beschreibung hält fest: Audit nur bei realen Releases, Null-Sweeps bleiben still, und der periodische Sweep ist außerhalb der Boot-Recovery der alleinige Owner des Queue-Lease-Releases.

- **Dedupe:** Der Aufruf von `release_stale_queue_task_leases` samt Queue-Projektion und `ticket_reconciliation`-Audit wurde vollständig aus `reconcile_ticket_runtime_state` entfernt.

- **Reconcile bleibt verantwortlich für:**
  1. Recovery verlassener Business-OS-App-Tasks
  2. `stuck_lease_escalation`
  3. Release abgelaufener Ticket-Event-Leases
  4. Release wieder bereiter blockierter Ticket-Events

- **Gate-Anpassung:** Der Gate-Test wurde in
  `ticket_reconcile_gate_skips_idle_state_and_reopens_for_remaining_recovery_work`
  umbenannt. Er prüft weiterhin unverändert streng:
  - unveränderter Idle-Zustand wird innerhalb des Safety-Fensters übersprungen;
  - Core-DB-Änderungen öffnen das Gate wieder – nun ausdrücklich für App-Recovery, Stuck-Evidence und Ticket-Event-Reparatur, nicht für Queue-Lease-Release.

- **Governance-Beschreibung:** `ticket_reconciliation` beschreibt jetzt ausschließlich Ticket-Event-Leases und blockierte Ticket-Events und verweist explizit auf den Orphan-Sweep als Queue-Lease-Owner.

### Geänderte Dateien

- `src/core/service/service.rs`
- `src/core/service/governance.rs`

Die bereits vorher dirty vorhandenen Änderungen unter `channels` und `tickets` wurden nicht angefasst. Ebenso blieben Attempt-Logik, Lease-Alter und Sweep-Intervall unverändert. Kein `git add` und kein Commit.

Alle verwendeten, bereits existierenden Produktions- und Testreferenzen wurden mit `git grep … HEAD` gegen `HEAD` geprüft.

## Beweistests

Neu und isoliert grün:

- `orphaned_queue_lease_sweep_audits_one_real_release_and_noop_stays_silent`
  - reale verwaiste Lease: `leased → pending`
  - genau ein sichtbares, dauerhaftes `orphaned_queue_lease_sweep`-Ereignis
  - `released_message_keys` enthält den freigegebenen Key
  - erzwungener zweiter Sweep ohne Kandidaten erzeugt kein weiteres Ereignis

- `ticket_reconcile_leaves_stale_queue_lease_for_orphaned_sweep_owner`
  - Reconcile lässt die abgelaufene Queue-Lease auf `leased`
  - kein falsches `ticket_reconciliation`-Ereignis
  - anschließender Orphan-Sweep setzt sie auf `pending` und schreibt genau ein Sweep-Ereignis

Zusätzlich grün:

- `ticket_reconcile_gate_skips_idle_state_and_reopens_for_remaining_recovery_work`
- `inventory_covers_router_and_plan_repair_mechanisms`

Vier verbleibende Reconcile-Aufgaben, bestehende Tests:

- `router_reconcile_completes_green_leased_business_os_app_task`
- `stuck_protected_queue_lease_is_escalated_not_released`
- `stale_ticket_event_lease_releases_to_pending`
- `blocked_ticket_event_releases_after_knowledge_and_control_are_ready`

## Vorher/Nachher

| Prüfung | Vorher | Nachher |
|---|---:|---:|
| `service::service_loop` | Exit 101, 3 greppy-Fehlerblöcke, 299 Warnungen | Exit 101, dieselben 3 Fehlerblöcke, 299 Warnungen |
| `service::state_invariants` | Exit 0, 298 Warnungen | Exit 0, 298 Warnungen |
| Neue Beweistests | – | 2/2 grün |
| Vier verbleibende Reconcile-Aufgaben | – | 4/4 isoliert grün |
| `cargo fmt --check` | – | grün |
| `git diff --check` | – | grün |

Die Aggregate-Service-Loop-Fehler waren bereits in der Baseline vorhanden; beobachtet wurden unverändert die MCP-Options-Assertion und der Fehler „File opened that is not a database file“. Es kamen keine neuen roten Fehlerblöcke hinzu.

## Offene Bedenken

- Der stark dirty Arbeitsbaum bleibt die einzige wesentliche Einschränkung bei der Bewertung der Aggregate-Suite.
- Die bekannten Baseline-Fehler wurden nicht bearbeitet, da sie außerhalb von I-073 liegen.
- Fortschritt wurde unter `/Volumes/tmp/ctox-pipeline/i073-fortschritt.md` protokolliert.

**Workjet-Completion-Receipt v1**
```json
{
  "task_id": "I-073",
  "status": "completed",
  "acceptance": "met_on_worktree_baseline",
  "files_written": [
    "src/core/service/service.rs",
    "src/core/service/governance.rs"
  ],
  "proof_tests_added": 2,
  "proof_tests_passed": 2,
  "remaining_reconcile_regressions_passed": 4,
  "aggregate_regression_delta": 0,
  "format_check": "passed",
  "commit_created": false
}
```
