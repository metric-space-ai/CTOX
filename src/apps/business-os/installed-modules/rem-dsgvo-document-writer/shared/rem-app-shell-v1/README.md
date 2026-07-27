# REM App Shell v1

REM App Shell v1 ist der gemeinsame Shell-Vertrag fuer die Hypoport/REM Apps. Die Shell bleibt funktional kompatibel zur CTOX Business OS Shell: gleiche zentrale Klassen, gleiche `data-*` Attribute fuer Resizer und gleiche Pane-Semantik. REM-spezifisch sind nur Farben, Typografie-Feinschliff und Statusdarstellung.

## Layout-Vertrag

Standalone und SharePoint nutzen zwei Spalten:

```html
<div class="app-shell rem-app-shell">
  <header class="topbar">...</header>
  <main class="ctox-workspace ctox-workspace--two-pane" data-resize-frame>
    <section class="ctox-pane">...</section>
    <button class="ctox-column-resizer"
      data-resizer="right"
      data-resizer-var="--ctox-right-width"
      data-resizer-min="300"
      data-resizer-max="560"></button>
    <aside class="ctox-pane">...</aside>
  </main>
</div>
```

CTOX Business OS nutzt drei Spalten und fuegt nur die linke Persistenzspalte hinzu:

```html
<main class="ctox-workspace" data-resize-frame>
  <aside class="ctox-pane ctox-pane--glass">Persistenz</aside>
  <button class="ctox-column-resizer" data-resizer="left" data-resizer-var="--ctox-left-width"></button>
  <section class="ctox-pane">Hauptansicht</section>
  <button class="ctox-column-resizer" data-resizer="right" data-resizer-var="--ctox-right-width"></button>
  <aside class="ctox-pane">Agent</aside>
</main>
```

## Klassen, die Apps verwenden sollen

- `app-shell`, `topbar`, `brand`, `top-actions`, `api-status`, `theme-toggle`
- `ctox-workspace`, `ctox-workspace--two-pane`, `ctox-pane`, `ctox-pane--glass`
- `ctox-pane-header`, `ctox-pane-title-row`, `ctox-pane-titles`, `ctox-pane-title`, `ctox-pane-kicker`, `ctox-pane-actions`
- `ctox-pane-body`, `ctox-pane-scroll`, `ctox-pane-band`, `ctox-pane-tools`, `ctox-pane-search`
- `ctox-input`, `ctox-select`, `ctox-textarea`, `ctox-field-label`
- `ctox-button`, `ctox-button--primary`, `ctox-icon-button`, `ctox-column-resizer`
- REM-Erweiterungen fuer Agent-UX: `rem-agent-status`, `rem-progress`, `rem-activity-list`, `rem-question-card`, `rem-options-panel`

## JS-Vertrag

`rem-app-shell-v1.js` stellt bereit:

- `RemAppShellV1.initTheme()` fuer den gemeinsamen Light/Dark-Wechsel.
- `RemAppShellV1.initColumnResizers(root)` fuer CTOX-kompatible Resizer ueber `data-resize-frame` und `data-resizer-var`.
- `RemAppShellV1.initDisclosure(toggle, panel)` fuer einheitliche ein-/ausklappbare Bereiche.

## Qualitaetsregeln

- Standalone/SharePoint: genau zwei Hauptspalten, Hauptansicht links, Agent/Human-in-the-loop rechts.
- CTOX: genau drei Hauptspalten, Persistenz links, Hauptansicht mittig, Agent/Human-in-the-loop rechts.
- Keine Rahmen-in-Rahmen-Ketten: Pane ist der sichtbare Rahmen; innere Inhalte sind flach.
- Fortschritt erscheint erst, wenn ein Lauf gestartet ist.
- Aktivitaetslogs enthalten Zeitstempel.
- API-Modus: eingebaute API ist gesperrt; eigene API erlaubt eigene Provider-/Modellwerte.
