# APPSTORE-V2 — Kampagnen-Board

**Headline:** Phase 0 komplett; kritischer Pfad = Sol-Lauf P1.1 (Loader-Vereinigung) landet, danach P1.2/P2.

Zielbild (entscheidungsfrei bis auf OWNER-Karten unten):
https://claude.ai/code/artifact/00f7ce23-1cb4-450d-bb26-a1f1f5fc2898
Phasen: 0 Boden · 1 Katalog/Loader · 2 Origin+Core-Repair · 3 Store-Kanal appstore.ctox.dev · 4 Nutzer-Apps-Pipeline · 5 Code-Modus Shell · 6 Rubriken.

---

## Done

- **[P0.1] Bundle-Wächter grün** — `coding-agents` Manifest: `deletable=false`, Store-Block auf
  `installable:false / editable_after_install:false / distribution:system-module` (wie registry
  und alle System-Apps). Verifiziert: `node src/apps/business-os/scripts/assert-standard-app-bundle.mjs`
  → `standard_app_bundle_ok=1 selected=17`. Commit `f1f5d84d0`.
- **[P0.2] Registry-Generator gelandet** — `scripts/generate-module-registry.mjs` projiziert
  die 36 Manifeste deterministisch nach `modules/registry.json` UND in den markierten
  `OFFLINE_FALLBACK_CATALOG`-Block in app.js (jetzt alle 18 System-Apps statt 10);
  `--check` = Drift-Wächter; harte Validierung der Core-Mitgliedschaft. 7 Manifest-Versionen
  auf SemVer normalisiert. APP_BUILD → `20260831-ctox-appstore-registry-v326` (11 Träger).
  Verifiziert: generate --check, assert-standard-app-bundle, assert-shell-v2-contract
  (33/35, 2 Restbefunde calendar/importer sind vorbestehend), node --check app.js — grün.
  Commit `393ba88ce`.
- **[P5-Messung] Shell-Kartierung Code-Modus** — KORREKTUR zweier Annahmen aus der Codekarte:
  (1) `code-editor` ist HEUTE SCHON keine startbare App — er fehlt in DESKTOP_APPS
  (app.js:4346 enthält nur explorer+file-viewer) und wird ausschließlich als integrierte
  Source-Ansicht ins App-Fenster gemountet (`ensureIntegratedModuleToolSession`, Modi
  app|source|versions, app.js:2598ff; Mount app.js:2713). Der Fenster-Modus-Umschalter,
  den OWNER 2 fordert, existiert im Kern bereits. (2) `creator` ist eine TOTE Referenz:
  kein Modul, kein DESKTOP_APPS-Eintrag, aber der App-Store ruft `ctx.openApp('creator')`
  → der „Neue App per KI-Prompt"-Weg über den Store ist vermutlich defekt (in P1.2 messen).
  Coding läuft heute als separates Fenster (coding-agents) aus dem Titel-Versionsmenü
  (app.js:2450); code-editor hat aber ein eingebautes Agent-Panel mit `ctox.coding.turn`
  (desktop-apps/code-editor/app.js:890). P5 = Umschalter sichtbar machen + Agent-Panel im
  integrierten Modus + Reste (`DESKTOP_APP_DB_COLLECTIONS` browser/creator, app.js:6531).
- **[P0-Messung] kundenpipeline-Doppelgänger vermessen** — `src/apps/business-os/installed-modules/`
  ist gitignored (.gitignore:111) und wird von keinem Loader gelesen (Serving-Root ist
  `<state>/runtime/business-os`). Kein Laufzeitkonflikt. ABER: die ignorierte Kopie (0.3.1) ist
  gegenüber der Core-Kopie (0.1.0) echt divergiert (`core/`-Verzeichnis, records.test.mjs,
  audit-scenarios.json nur dort; alle Kernddateien differieren). → OWNER-Karte unten.

## Working

- **[P1.1 + P0.3] Loader-Vereinigung + Kollisionsdiagnose** — Sol · Completion, Workjet-Run
  `local-2026-08-31T121437Z-2d73f775-d971-4baa-8ce6-0fab405b222c`, gestartet 12:14Z aus
  Launchpad ~/.local/state/workjet-launchpads/appstore-v2 (business_os-Subtree, Baseline
  9f618cb; Voll-Repo-Archiv 408 MB > 64-MiB-Limit). Brief: eine gemeinsame Loader-/Upsert-
  Implementierung, store.rs-Semantik kanonisch; ID-Kollision = laute Diagnose + erhaltener
  Vorrang, KEIN Bail (Tenant-Altbestand darf nicht sterben). Kein cargo im Workspace (kann
  dort nicht bauen) — Fable kompiliert den Diff im Hauptbaum. Fertig = Diff importiert,
  cargo check grün, eine Definition je Funktion.
- **[P3A] Statischer Appstore-Publisher** — Sol · Completion, Workjet-Run
  `local-2026-08-31T122315Z-09c728d1-3802-4cfc-b628-23036c22b7c3`, gestartet 12:23Z aus
  Launchpad ~/.local/state/workjet-launchpads/appstore-v2-publisher (13 MB, Baseline von
  393ba88ce). Liefert build-appstore-index.mjs + node:test-Suite: deterministische Zips,
  Ed25519-Signaturen, index.json v1, 18 Store-Apps. Fertig = Tests grün im Hauptbaum,
  Proof-Lauf mit 18 Apps.

## To-Do
- **[P1.2] explorer/file-viewer/creator als echte Module** — explorer+file-viewer aus
  DESKTOP_APPS (app.js:4346) zu modules/<id>/ portieren (mount(container,ctx) →
  mount(ctx)-Vertrag), creator aus desktop-apps/ als Modul wiederbeleben (App-Store-Pfad
  `openApp('creator')` ist tot — erst messen). System-Apps-Listen + Registry-Generator
  laufen automatisch mit. TRIGGER: nach P1.1-Import (Loader-Landung).
- **[P1.3] Store-UI-Katalogquellen auf Server-Projektion reduzieren** (app-store/index.js:468–508).
  TRIGGER: nach P1.2.
- **[P2] origin-Feld + Core-Repair + local-catalog-Install.** TRIGGER: nach P1.
- **[P3] Store-Kanal V1** (CI-Publisher + nativer Client + Kimi-Cyber-Review). TRIGGER: nach P2;
  DNS appstore.ctox.dev = OWNER-Handgriff.
- **[P4] Nutzer-Apps-Pipeline** (Zip/GitHub-Link als origin:user, Importer auf Kommando,
  Deinstallation). TRIGGER: nach P2, parallel zu P3 möglich.
- **[P5] Code-Modus sichtbar + Agent-Panel integriert** — deutlich kleiner als geplant
  (siehe P5-Messung): sichtbarer Modus-Umschalter am v2-Fenster (heute nur über
  Titel-Klick-Menü erreichbar), Agent-Panel des eingebetteten Editors als Coding-Weg im
  Fenster, coding-agents-Querstart bleibt. desktop-apps/code-editor bleibt als
  Shell-Komponente (nicht App) — Doku entsprechend. Route: Fable direkt + Kimi-Pixelreview.
  Geometrie-Labor + assert-shell-v2-contract Pflicht. TRIGGER: nach P1.2 (app.js-Konflikt).
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

- Board: diese Datei (docs/dev/appstore-v2-board.md, committed) + stabile Artifact-URL:
  https://claude.ai/code/artifact/e9ffba86-1126-4663-ad64-0213db76ceb3
  (Rendering via Scratchpad-Skript render-board.mjs; bei Board-Updates neu publizieren).
- Zielbild-Artifact: https://claude.ai/code/artifact/00f7ce23-1cb4-450d-bb26-a1f1f5fc2898
- Codekarte (18 Befunde A–R): Explorer-Report, kondensiert im Zielbild §2.
- Wächterlauf P0.1: lokal reproduzierbar via assert-standard-app-bundle.mjs.
