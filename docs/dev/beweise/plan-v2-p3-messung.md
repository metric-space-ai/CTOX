# P3-Baseline: Idle-Wake-Raten der Peer-Loops — Hypothese widerlegt

Messung 11.08.2026 07:53–08:03 (10-min-Differenz zweier Snapshots von
`~/.local/state/ctox/business-os-rxdb-peer.status.json`, performance.loops;
Rohdaten: /Volumes/tmp/ctox-pipeline/reports/p3-messung-{1,2}.json).
Lebender Daemon, keine Nutzeraktivität am Business OS.

| Loop | ΔTicks | ΔAktiv | Ticks/min |
|---|---:|---:|---:|
| business_commands | 21 | 0 | 2,10 |
| desktop_file_index | 2 | 0 | 0,20 |
| channel_state | 1 | 0 | 0,10 |
| knowledge_tables | 1 | 0 | 0,10 |
| module_catalog | 1 | 0 | 0,10 |
| browser_runtime, business_records, business_users, notes, runtime_settings, ticket_state, workspace_branding | 0 | 0 | 0,00 |

**Urteil:** Die Discovery-Hypothese H-4 („12+ Loops à 3 s, Idle-Backoff
unwirksam, ein aktiver Tick resettet") gilt im Idle NICHT: 7/12 Loops
schlafen vollständig, der Command-Consumer hält sein 30-s-Idle-Intervall,
kein Loop zeigt 3-s-Polling in Ruhe. P3 als Optimierungshebel ist im
Ist-Zustand ein NEGATIVES ERGEBNIS (wie P1). Nebenbefund: auch die
Business-Record-Projektion (P6-Fläche) ist im Idle still (0,0/min) —
P6 lohnt nur unter Änderungslast, nicht als Idle-Fix.
Offen bleibt das Verhalten UNTER Last (Reset-Empfindlichkeit) — das ist
ein anderes, kleineres Ticket als das geplante P3.
