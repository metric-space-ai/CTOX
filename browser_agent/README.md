# Browser Agent

Dieses Verzeichnis enthaelt den entkoppelten Browser-Agenten des CTO-Agenten.

- `extension/`
  Enthält die Chrome-Extension mit eigenem Agent-Loop, Browser-Runtime, visueller Navigation, Tab-Management, Playwright-CRX-Attach und einem Sidepanel fuer Live-Status.
- `../runtime/browser-agent-bridge/`
  Persistente lokale Job-Bridge zwischen Rust-Supervisor und Chrome-Extension.

Staging:

```sh
sh scripts/install_browser_agent_extension.sh
```

Komplettes Browser-Bootstrap auf Linux:

```sh
sh scripts/install_browser_engine.sh
sh scripts/launch_browser_agent_chrome.sh
```

Auf Linux nutzt der Launcher fuer den Browser-Agent bevorzugt `Chrome for Testing`, weil der offizielle Stable-Channel `--load-extension` fuer diesen Automationspfad nicht mehr verlaesslich umsetzt.
Danach laeuft Chrome mit der aus `runtime/browser-agent-extension` geladenen unpacked Extension gegen die lokale Bridge auf `http://127.0.0.1:8765`.
Wenn Chrome die unpacked Extension wegen deaktiviertem Developer Mode im dedizierten Browser-Agent-Profil ausknipst, erkennt der Launcher diesen Zustand, setzt das isolierte Profil einmal sauber zurueck und startet Chrome mit dem passenden Feature-Override erneut.
Das sichtbare Sidepanel ist in der Extension eingebaut; der sichere Oeffnungspfad ist ein Klick auf das Extension-Icon. Der Startup-Autopen ist nur best effort, weil Chrome das je nach Fenster-/Gesture-Zustand einschränken kann.
