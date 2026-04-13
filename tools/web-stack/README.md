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
