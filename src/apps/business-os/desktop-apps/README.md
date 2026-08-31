# desktop-apps/ — Status (APPSTORE-V2, 31.08.2026)

Dieses Verzeichnis enthält **keine startbaren Apps** mehr. `DESKTOP_APPS` in
`app.js` ist leer; jede startbare App ist ein Modul unter `modules/<id>/` und
kommt aus dem Katalog.

- `code-editor/` — **Shell-Komponente, keine App.** Der Motor der integrierten
  Code-Ansicht: `ensureIntegratedModuleToolSession` (app.js) mountet ihn als
  Fenster-Modus `source` in jedes App-Fenster (Titelmenü → „Code-Modus").
  Enthält auch das Agent-Panel (`ctox.coding.turn`).
- `explorer/`, `file-viewer/`, `creator/`, `browser/` — **Legacy.** Die Shell
  lädt sie nicht mehr; explorer/file-viewer/creator leben als Module weiter.
  `file-viewer/` bleibt liegen, weil `src/core/rxdb/tools/browser_rust_smoke.js`
  ihn in vier Smokes direkt importiert (gemessen 31.08.2026). Entfernung der
  übrigen Verzeichnisse erst nach erneuter Referenzmessung.
