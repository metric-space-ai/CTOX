# Browser Agent

This directory contains the decoupled browser agent of the CTO-Agent.

- `extension/`
  Contains the Chrome extension with its own agent loop, browser runtime, visual navigation, tab management, Playwright CRX attach flow, and a side panel for live status.
- `../runtime/browser-agent-bridge/`
  Persistent local job bridge between the Rust supervisor and the Chrome extension.

Staging:

```sh
sh scripts/install_browser_agent_extension.sh
```

Full browser bootstrap on Linux:

```sh
sh scripts/install_browser_engine.sh
sh scripts/launch_browser_agent_chrome.sh
```

On Linux, the launcher prefers `Chrome for Testing` for the browser agent because the official stable channel no longer handles `--load-extension` reliably enough for this automation path.
After that, Chrome runs with the unpacked extension loaded from `runtime/browser-agent-extension` against the local bridge at `http://127.0.0.1:8765`.
If Chrome disables the unpacked extension because developer mode is off in the dedicated browser-agent profile, the launcher detects that state, resets the isolated profile once, and restarts Chrome with the matching feature override.
The visible side panel is built into the extension; the safe opening path is clicking the extension icon. Startup auto-open is only best effort because Chrome may restrict it depending on window and gesture state.
