# CTOX Web Stack

This crate is the owned compile boundary for the CTOX web surface:

- `ctox_web_search`
- `ctox_web_read`
- `ctox_browser_prepare`
- `ctox_browser_automation`

The root `ctox` binary now keeps only thin adapters plus the durable scrape
executor injection, so search/read/browser work can evolve without dragging
unrelated CTOX execution modules into the same edit surface.

`bench/` contains the standalone regression bench for this module. It is
binary-first and data-driven so fixture and live checks can run against a built
`ctox` binary without recompiling the whole repository for every iteration.

Current ownership boundary:

- `search`, `read`, `browser-prepare`, and `browser-automation` are owned here.
- the `web scrape` request shape and CLI contract are owned here.
- the durable scrape runtime/database still stays in the wider CTOX scrape
  subsystem, so the root injects only that executor.

## Google bootstrap profile

`ctox_web_search` with the default `google_bootstrap_native` provider fronts
Google search with a cookie profile sampled from a headed Chrome session. The
profile is persisted at `runtime/google_bootstrap_native_profile.json` with
mode `0600` — **it holds live Google session cookies (SID, __Secure-1PSID,
SAPISID, …) that are equivalent to a logged-in auth token for the signed-in
Google account**. Treat the file with the same care as an OAuth refresh
token:

- Do not commit it, back it up unencrypted, or copy it to shared hosts.
- On headless servers, sample the profile on a GUI host and transfer it via
  `ctox web google-bootstrap-import --file <path>` — the import re-applies
  `0600` permissions.
- Run `ctox web google-doctor` to check pipeline readiness (Chrome binary,
  Playwright workspace, helper binary, profile freshness, DISPLAY availability).

### Environment overrides

| Variable | Purpose |
| --- | --- |
| `CTOX_WEB_SEARCH_OPENAI_MODE` | `ctox_primary` (default) routes OpenAI `web_search` tool calls through CTOX; `passthrough` forwards them upstream unchanged. |
| `CTOX_WEB_SEARCH_PROVIDER` | `google_bootstrap_native` (default), `google`, `google_browser`, `bing`, `searxng`, or `mock`. |
| `CTOX_WEB_GOOGLE_BOOTSTRAP_TTL_SECS` | Proactive profile refresh window. Default `21600` (6h). |
| `CTOX_WEB_GOOGLE_BOOTSTRAP_PROFILE_PATH` | Override persisted profile location. |
| `CTOX_WEB_GOOGLE_BOOTSTRAP_PROBE` | Override probe script (used by tests). |
| `CTOX_WEB_BROWSER_REFERENCE_DIR` | Directory containing `node_modules/playwright`. Defaults to `runtime/browser/interactive-reference`. |
| `CTOX_WEB_GOOGLE_BOOTSTRAP_LEAVE_CHROME_RUNNING` | `1` to skip the auto-quit of the user's Chrome during a probe refresh. |
| `CTOX_WEB_CHROME_BIN` | Path to the Chrome/Chromium executable (auto-discovered on macOS/Linux). |
| `CTOX_WEB_CHROME_USER_DATA_DIR` | Path to the Chrome profile to clone (auto-discovered on macOS/Linux). |
