# P4-Baseline: Frischer Shell-Boot — die 178-Collections-Prämisse gilt nicht

Messung 11.08.2026 08:05–08:10, frisches Browser-Profil (kein IndexedDB),
lokale Instanz http://127.0.0.1:8765, Load 3,3.

| Metrik | Wert |
|---|---|
| Boot-HTTP-Requests bis readyState complete | **125** (Board-Baseline 129 ✓) |
| Eager registrierte Collections | **15** — nicht 178 |
| Zustand nach ~1 min | alle 15 `complete`, phase `collection-sync`, mode webrtc |
| Zustand nach 4 min | unverändert 15/complete; +57 Requests (Maintenance/Assets) |
| JS-Heap | 19 MB (weit unter der 2-GB-Regel) |
| lastError | null |

**Urteil:** Die Discovery-Prämisse „Initial-Sync über alle 178 Collections,
serieller Start mit 500-ms-Gap = Worst Case ~89 s" beschreibt den
Shell-Boot NICHT: Das Demand-Loading beschneidet den Eager-Sync auf 15
Prioritäts-Collections; Boot-Sync komplett in ~1 Minute bei 19 MB Heap.
P4 als „Initial-Sync-Priorisierung bauen" ist damit WIDERLEGT — die
Priorisierung existiert und wirkt. Dritter durch Messung getöteter Hebel
(nach P1, P3). Verbleibende echte Hebel: P5 (Batch-/Checkpoint-Matrix,
erst messen) und P6 (Projektions-Scan NUR unter Änderungslast).
Offene Folgefrage (klein): Wie schnell werden die 163 Demand-Collections
beim ersten Modul-Öffnen live — das ist die Nutzer-sichtbare Latenz.
