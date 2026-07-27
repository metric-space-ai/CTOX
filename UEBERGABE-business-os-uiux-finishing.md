# Übergabe — Business OS UI/UX-Feinschliff (Shell-Phase + Referenz-App-Runde)

Stand: 2026-07-22. Autor: Codex (Kimi-Work-Session). Auftrag: `uiux-finishing-pass.md`
(Skill `business-os-app-module-development`), freigegeben mit vier Owner-Korrekturen
und Rahmenbedingungen (Tabu-Zonen, ein Commit pro Baustein, Spiegelung + Proof).

Alles ist auf `origin/main` gepusht (letzter Push: `d5ebf06a7`). Worktree-Dirt in
diesem Checkout stammt von Parallel-Sessions und wurde konsequent nicht angefasst
(Tabu: documents, importer, customers, rxdb/, sync.js, db.js).

---

## 1. Shell-Phase (12 Commits)

| Commit | Baustein | Kern |
|---|---|---|
| `b344b4fb2` | Motion-System | `--motion-fast/base/slow` (120/160/200 ms) + `--ease-standard/spring`; 55 Transition-Sites gesweept; 9× `transition: all` entfernt; Fenster-Öffnen-Animation `.is-opening` neu; Minimize/Close auf Token-Klassen; Reduced-Motion von 3 Blöcken auf 1 globalen konsolidiert + `matchMedia`-Helper in window-manager.js |
| `0a453a36f` | Toter Dock | `shared/taskbar.js` + 471 Zeilen `.shell-taskbar`-CSS gelöscht (Owner-Entscheid: `.module-tabs` ist der lebende Pfad); toter `data-shell-taskbar-open`-Pfad inkl. Safe-Area-Token mit entfernt |
| `3a83ac38c` | Typo-Sweep | 192 → 10 rohe `font-size` in app.css; Body 14 → 13 px auf Kit-Skala (`--fs-*`); `tabular-nums` für Badges/Zähler; Kicker-Tracking vereinheitlicht (0.09em) |
| `90475f2a8` | Spacing-Sweep | 453 → 60 rohe px-Abstände; 409 Deklarationen auf `--space-1..5`; Rest = begründete Geometrie (optische Nudges, >24 px Layout) |
| `2a9c77015` | Zustände | Ein Focus-Ring (`var(--focus-ring)`, Doppel-Ring und `--shell-focus-ring` entfernt); `::selection` neu (accent 24 %); Disabled-Rezept vereinheitlicht (0.55, grayscale weg); erste Hover-Fills auf Kit-Tokens |
| `cb5ccf689` | Theme-Enthärtung | Hex 100 → 27 (Rest: Token-Definitionen, macOS-Ampeln, Mix-Pole); dark-kalibrierte Badge-Paletten auf `--warning/--danger` (Light-Break gefixt); Tailwind-Fehler-Palette entfernt; Dark-Chrome token-abgeleitet (Override 60 → 45 Zeilen); Startup-Gradient als Tokens |
| `290a5f6cd` | Fenster-Chrome | `--win-elev-dim` in beiden Themes; Dark-Override kollabiert; `updateDynamicShadow()` im ctox-Style abgestellt (war tote Per-Frame-Arbeit, `!important`-überstimmt); write-only `--win-shadow-x` entfernt |
| `86eb27719` | Skeleton-Treue | Loading-Shadow-Radien auf `--panel-radius` (14 px stimmte mit nichts überein); toter Selektor repariert; 15 stale `var()`-Hex-Fallbacks entfernt; Desktop-Tile-Skeleton 4 → 16 px (matcht echte Tiles) |
| `77324af7f` | Chat-Dock | business-chat.js eingebettetes CSS auf `--motion-*`/`--ease-*` (28 Deklarationen) und `--fs-*`/`--space-*`; Toast-Timer 360 → 200 ms an `--motion-base` gebunden + Reduced-Motion = 0 ms |
| `1494adb8a` | Reste | Tote `.shell-taskbar`-Selektoren aus app.js-Kontextliste; `.ctox-pane-tab:disabled` 0.5 → 0.55 (Guard geprüft: Drift, keine Policy) |
| `20fea6e09` | Branding-Fixture | Design-Matrix um Custom-Brand-Dimension erweitert (24 Captures; violettes Fixture über exakt dem Runtime-Mechanismus `workspaceBrandingStyleText()`); Baselines regeneriert (Default-Shift 2,8–5,9 % = erwarteter Finishing-Effekt); Visual-Gate grün. **Linux-Baseline muss auf Linux/CI nachgezogen werden** (`npm run qa:visual-baseline`) |
| `4a2cef194` | APP_BUILD | Stempel `20260721-shell-finishing-v1` (inzwischen von Parallel-Session überstampft — deren Hoheit) |

Verifikation Shell-Phase: `window-manager.test.mjs` 3/3, `taskbar-pins.test.mjs`,
`business-chat.test.mjs` 26/26, `assert-design-system.mjs` (nur pre-existing
coding-agents-Assertion rot), `assert-visual-diff` 24/24.

## 2. Release-Spiegelung + Browser-Proof

- **Selektiver rsync** (nur eigene Dateien) nach `~/.local/state/ctox/business-os/`
  und `~/.local/lib/ctox/current/src/apps/business-os/` — kein Voll-rsync, damit
  uncommittete Fremd-Arbeit nicht ins Release rutscht.
- **Browser-Proof gegen die laufende Instanz (Port 8765): 14/14 bestanden** —
  Stempel live, Motion-Tokens computed exakt, `.is-opening` live beobachtet,
  Resize-Clamp 640×480, Theme-Switch mit 3 offenen Fenstern, kein `.shell-taskbar`
  im DOM. Artefakte: `tmp/browser-proof/` (proof.json + 6 Screenshots).
- App-Runde nachgeliefert: sechs Modulverzeichnisse gespiegelt (gleiche Ziele).
- `assert-managed-business-os-assets.mjs` meldet Rest-Drift ausschließlich auf
  Fremd-Dateien (app.js/index.html/registry/business-chat/documents/spreadsheets)
  — uncommittete Parallel-Sessions-Arbeit, absichtlich nicht gespiegelt.

## 3. App-Runde (6 Commits, Brief-Reihenfolge)

| Commit | App | Kern |
|---|---|---|
| `9968118db` | ctox | Import/Export-Header-Icons; stale `right`-Pane aus module.json entfernt; sessionStorage-Handoff geprüft (etablierte Inter-App-Konvention, belassen) |
| `10cb4096e` | coding-agents | Vollmigration auf `data-pg-*` + `ctox-pane-grammar-change` (Chrome-Verdrahtung gelöscht); Kontext-Trio auf App-Rows + Turn-Rows (`business_command`); Scroll-Erhalt in Transcript/Turns; `aria-selected`; 0.2.1 → 0.2.2. Tests 12/12, Static-Check OK |
| `b93b0145c` | knowledge | In-Place-Flip `applyKnowledgeSelection()`; localStorage-Dead-Code entfernt (grep-bewiesen unerreichbar); **Ladepfad-Beweis**: Shell importiert nur index.js → index.html ist jetzt single source (data-pg-Markup, via `loadModuleMarkup` geladen), `documentTemplate()` gelöscht; Kit-Token-Redefinitionen entfernt; Footer auf Per-Pane-`data-pg-footer`; `aria-current` → `aria-selected`; v1 → 1.1.0. Tests 21/21 |
| `b090bf407` | threads | In-Place-Flip `applyThreadSelection()`; kanonisches Kontext-Trio auf allen Rows (Legacy-Attribute entfernt, Shell-Fallback bevorzugt Trio); 4. Band-Tab „Erledigt" gezählt; Header-Icons (create-note, export-threads JSON — **kein Import**: Domäne hat keinen); Briefing-Strip entfernt (verbotener Dauerstatus); `third_pane_justification`; Markup-Cache-Buster; sessionStorage → `ctx.storageScope`; v0.1 → 0.2.0; schema.js belassen (Re-Export aus ctox = echte Registrierungsquelle, Validator source-mode grün) |
| `3e725b5d6` | app-store | **Refresh → Header-Icon** (GitHub-Discovery bleibt als Fähigkeit, Owner-Regel); Text-Buttons aus Content-Fluss → Header-Icons; Sync-Dauerzeile entfernt; Grid-Selektion In-Place + Scroll-Erhalt; Shard-Badge-Wall → kanonische Selector-Anatomie; Cache-Buster für Markup + CSS; v1 → Semver. Tests 25/25 |
| `d5ebf06a7` | reports | In-Place-Flip `applyReportsSelection()`; tote Refresh-Verdrahtung end-to-end gelöscht; Export-JSON-Header-Icon; Cache-Buster; `third_pane_justification`; v1 → 1.1.0; schema.js-Re-Export belassen (Validator source-mode exempt). Tests 8/8 |

Systemisch über alle sechs: kanonische Spalten-Grammatik (Shell verdrahtet, Modul
hört nur noch), Interaktions-Gesetz „Re-Renders bewegen den Operator nie",
Kontext-Trio flächendeckend, dreiteilige Semver-Versionen, Cache-Buster erbt.

## 4. Offene Flanken / Deferrals

1. **`assert-shell-chat-composition.mjs` rot auf HEAD** — nachweislich durch
   Chat-Dock-Arbeit einer Parallel-Session (mit gestashten eigenen Änderungen
   identisch rot). Fremd-Baustelle.
2. **`assert-rxdb-only.mjs`: `conversations/index.js: frontend-local-only-sync-mode`**
   — fremd, nicht angefasst.
3. **`BusinessOsPermissionError: Kein Leserecht für business_commands`** beim
   coding-agents-Mount in der `local-dev`-Session (Instanz-Policy, kein Code-Bug
   dieser Kampagne). Policy-seitig klären.
4. **Linux-Design-Matrix-Baseline** regenerieren (darwin ist aktuell, linux würde
   sonst am File-Set 12 ≠ 24 scheitern).
5. **`focus_ring` im Branding-Whitelist-Mechanismus**: `isSafeBrandingColor`
   akzeptiert nur Bare-Colors, `--focus-ring` ist ein Box-Shadow — pre-existing
   Lücke in branding.js, Fixture überspringt ihn bewusst.
6. **modules/desktop/index.css** (rohe Dauern/`transition: all !important`) und
   **business-chat.js Theme-Fallbacks** (`--accent: var(--theme-accent, #10b981)`)
   — bewusst späteren Runden zugeordnet (Modul-Runde / Theme-Runde).
7. **Nebeneffekt des Proofs**: CTOX-Fenster der Owner-Instanz steht auf 640×480
   (serverseitig persistierte Geometrie), Theme wurde auf Dark zurückgesetzt.
8. **Import-Icons** nur dort, wo die Domäne einen echten Import hat (knowledge ja,
   threads/reports nein — Export überall real verdrahtet).

## 5. Prozess-Einhaltung

- Ein Commit pro Baustein/App; ausschließlich explizit benannte Pfade gestaged;
  keine Fremd-Datei in keinem Commit (`git show --stat` jeweils geprüft).
- Tests erweitert, nie abgeschwächt (alte Chrome-Kontrakt-Assertionen wurden auf
  den neuen `data-pg-*`-Kontrakt migriert, Abdeckung gewachsen: +1
  Grammatik-Kontrakt-Test in knowledge, +1 in reports, +Chrome-Block in threads).
- Owner-Korrekturen umgesetzt: threads-Rechtsklick als Degradierung behandelt,
  app-store-Refresh verschoben statt gelöscht, `business_commands`-Entzug nur mit
  Validator-Nachweis (überall exempt → nirgends entfernt), knowledge-Löschung nur
  mit Ladepfad-Beweis.
- Keine Fachlogik, keine Command-Flows, keine Schema-Änderungen, keine
  Lifecycle-Kommandos (`ctox stop/start/upgrade`) — der Proof lief gegen die
  bereits laufende Owner-Instanz.

## 6. Nächste sinnvolle Schritte

1. Restliche Apps nach IA-Karte durch dieselbe Migration (tickets, conversations,
   notes, calendar, … — tickets/conversations/notes haben schon `data-pg-*`-Anteile,
   Voll-Audit nötig).
2. Browser-Proof der sechs gefixten Apps im echten Shell (Interaktions-Proofs
   §5: scrollen → unteres Element wählen → Offset bleibt; Tippen während Refresh).
3. Fremd-Flanken 1–3 mit den Parallel-Sessions koordinieren.
