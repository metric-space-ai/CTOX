# Plan-v2 Baseline — 10.08.2026 (S0)

Referenzpunkt für alle Tore aus `docs/ctox-sync-plan-2026-08-10.md`.
HEAD bei Erhebung: `abb0c3aea` (nach origin-Merge).

## Statisch (heute gemessen, Orchestrator)

| Metrik | Wert | Methode |
|---|---|---|
| Budget-Verstöße auf HEAD | **0** (store 27.316/27.516 · office 13.953/13.953 · outbound 5.195/5.270 · channels 7.174/7.221 · rxdb_peer 12.718/12.718 · service 26.177/26.237) | Guard-Regel (letzter `#[cfg(test)]`) auf `git show HEAD:` |
| Physische Zeilen | store 43.781 · service 46.039 · rxdb_peer 22.658 · app.js 12.321 · sync.js 3.026 | `wc -l` |
| Rote Tests (seriell) | **59**, unklassifiziert | `docs/dev/beweise/rot-basis.txt` (Basis-Messung der Vorwoche; R-01 verifiziert Aktualität) |
| Nur-parallel-Rote | ~50 zusätzlich | Board/Frühere Messung; nicht Tor-relevant (Vergleiche seriell) |
| Collections im Schema-Vertrag | 178 | `business_os_schema_contract.json` |
| app.js-Exports | 0 | `grep -c "^export "` |
| Dirty Arbeitsbaum | ~140 Einträge (102 M · 35 MM · 34 ??; Stand vormittags) | `git status --porcelain` — Triage = Owner-Frage §6.1 |

## Laufzeit (Board-Werte; Nachmessung ausstehend, je vor dem zugehörigen Hebel Pflicht)

| Metrik | Baseline | Quelle | Nachzumessen vor |
|---|---|---|---|
| Shell-Boot-HTTP-Requests | 129 (vorher 208) | Missionstafel 09.08. | P1 |
| Maintenance-Polls | 1,6/min idle | Missionstafel | P1 |
| Initial-Sync-Dauer bis CRITICAL live | — (nie gemessen) | — | P4 |
| Projektionen/min bei Null-Last | — | — | P3/P6 |
| `cached_document_count()` + RSS nach 1 h Churn | — | — | P2 |

## Betriebszustand bei Erhebung

- Platte: 100 % voll vorgefunden (8,5 GiB frei); nach Räumung (incremental
  8,3 G + ctox-Testbinaries aus deps) ~16 GiB frei. **Platten-Check bleibt
  Pflicht vor jedem Worker-Testlauf.**
- Load ~5–6 (ok). Kimi-Worker 403-Quota (Review-Gate S4 wartet).
- origin/main eingeholt (`abb0c3aea`); ein CSS-Konflikt zugunsten der lokalen
  Umstrukturierung aufgelöst (Regel `.research-run-note` existiert nicht mehr).
