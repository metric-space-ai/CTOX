# APPSTORE-V2 — Kampagnen-Board

**Headline:** Kritischer Pfad = Phase 0 abschließen (Registry-Generator), dann Phase 1 Loader-Vereinigung an Sol.

Zielbild (entscheidungsfrei bis auf OWNER-Karten unten):
https://claude.ai/code/artifact/00f7ce23-1cb4-450d-bb26-a1f1f5fc2898
Phasen: 0 Boden · 1 Katalog/Loader · 2 Origin+Core-Repair · 3 Store-Kanal appstore.ctox.dev · 4 Nutzer-Apps-Pipeline · 5 Code-Modus Shell · 6 Rubriken.

---

## Done

- **[P0.1] Bundle-Wächter grün** — `coding-agents` Manifest: `deletable=false`, Store-Block auf
  `installable:false / editable_after_install:false / distribution:system-module` (wie registry
  und alle System-Apps). Verifiziert: `node src/apps/business-os/scripts/assert-standard-app-bundle.mjs`
  → `standard_app_bundle_ok=1 selected=17`. Commit `f1f5d84d0`.
- **[P0-Messung] kundenpipeline-Doppelgänger vermessen** — `src/apps/business-os/installed-modules/`
  ist gitignored (.gitignore:111) und wird von keinem Loader gelesen (Serving-Root ist
  `<state>/runtime/business-os`). Kein Laufzeitkonflikt. ABER: die ignorierte Kopie (0.3.1) ist
  gegenüber der Core-Kopie (0.1.0) echt divergiert (`core/`-Verzeichnis, records.test.mjs,
  audit-scenarios.json nur dort; alle Kernddateien differieren). → OWNER-Karte unten.

## Working

- **[P0.2] Registry-Generator** — Fable direkt. `registry.json` + Offline-Fallback aus den
  Manifesten generieren, Wächter auf Byte-Stabilität. Fertig = Generator deterministisch,
  Wächter grün, `app.js`-Fallback aus derselben Quelle, alle drei Cache-Buster-Regeln beachtet.

## To-Do

- **[P0.3] ID-Kollision → harter Fehler** in beiden Loadern (store.rs:4434, server.rs:2336).
  TRIGGER: kann sofort; sinnvoll zusammen mit P1.1 (Loader-Vereinigung), sonst doppelte Arbeit.
- **[P1.1] Loader-Vereinigung** store.rs:4385/server.rs:2284 + upsert-Duplikat (store.rs:4684 /
  server.rs:2804) auf eine Funktion. Route: Sol. Rust-Diff selbst im Hauptbaum kompilieren.
  TRIGGER: startet nach P0.2-Commit (gemeinsame Basis).
- **[P1.2] Manifeste für explorer/file-viewer/creator**, DESKTOP_APPS (app.js:4346) entfernen.
  TRIGGER: nach P1.1.
- **[P1.3] Store-UI-Katalogquellen auf Server-Projektion reduzieren** (app-store/index.js:468–508).
  TRIGGER: nach P1.2.
- **[P2] origin-Feld + Core-Repair + local-catalog-Install.** TRIGGER: nach P1.
- **[P3] Store-Kanal V1** (CI-Publisher + nativer Client + Kimi-Cyber-Review). TRIGGER: nach P2;
  DNS appstore.ctox.dev = OWNER-Handgriff.
- **[P4] Nutzer-Apps-Pipeline** (Zip/GitHub-Link als origin:user, Importer auf Kommando,
  Deinstallation). TRIGGER: nach P2, parallel zu P3 möglich.
- **[P5] Code-Modus in der Shell**, danach code-editor stilllegen. Route: Kimi UI/UX + Sol.
  Geometrie-Labor + assert-shell-v2-contract Pflicht. TRIGGER: parallel ab sofort möglich.
- **[P6] Rubriken kanonisieren** (16 Slugs, Mapping, PUBLIC_DISTRIBUTIONS raus). TRIGGER: nach P1.

## Backlog + Owner

- **OWNER: kundenpipeline-Fork auflösen.** Ignorierte Kopie 0.3.1 (installed-modules/, mit
  core/-Dir + Zusatztests) vs. Core-Kopie 0.1.0 (modules/). Welche ist die Wahrheit? Bis zur
  Entscheidung wird nichts gelöscht; Serving nutzt die Core-Kopie.
- **OWNER: DNS/Subdomain appstore.ctox.dev anlegen** (Voraussetzung P3-Deploy; Deploy-Mechanik
  wie ctox.dev, siehe Memory project_ctox_dev_deploy).
- **OWNER 3 (Default läuft): Rubrikliste** = 16 Workjet-Slugs, „REM Capital" wird
  Kunden-Sichtbarkeit statt Rubrik. Veto möglich bis P6-Start.
- **Beobachtung:** `src/apps/business-os/runtime/qa/` (untracked Screenshots) verletzt
  „src/ ist nur Quellcode"; nicht Teil dieser Kampagne, nichts angefasst.
- **Beobachtung:** rem-* Apps liegen in gitignoriertem installed-modules/ ohne
  customer-app-binding.json — würden selbst am richtigen Ort von authorize_runtime_module
  abgelehnt. Gehört zu P4/K5-Kontext.

---

## Umgebungsfallen

- Worker-Worktrees können kein Rust bauen (gitignorierte Artefakte fehlen) → Diffs selbst im
  Hauptbaum kompilieren.
- Workjet-Snapshots: nur HEAD, kein origin, detached HEAD; 64-MiB-Archivlimit → ggf. Launchpad
  unter ~/.local/state/workjet-launchpads/ (nie Scratchpad). Ergebnisse via
  `workjet result import <run-id>` als refs/workjet/<run-id>.
- Kein absoluter Repo-Pfad in Briefs; Briefs prüfen Baum-Identität über Dateiinhalt, nicht Refs.
- Shell-Arbeit: Slots nur aus main; scripts/shell-v2-geometry-lab.mjs +
  scripts/assert-shell-v2-contract.mjs für shell-wirksame Änderungen.
- `src/apps/business-os/installed-modules/` ist gitignored — Änderungen dort landen nie in git.

## Fehlermuster

1. Prämisse vor dem Auftrag messen (kundenpipeline: vermeintlicher Laufzeitkonflikt war keiner).

## Evidenzkarte

- Board: diese Datei (docs/appstore-v2-board.md, committed) + Artifact-URL (stabil, siehe unten).
- Zielbild-Artifact: https://claude.ai/code/artifact/00f7ce23-1cb4-450d-bb26-a1f1f5fc2898
- Codekarte (18 Befunde A–R): Explorer-Report, kondensiert im Zielbild §2.
- Wächterlauf P0.1: lokal reproduzierbar via assert-standard-app-bundle.mjs.
