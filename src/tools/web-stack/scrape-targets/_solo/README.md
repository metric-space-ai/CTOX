# Solo adapter harness

Adapters are developed **solo first**: build `<target>/solo/probe.mjs` as plain
Playwright, prove extraction against the live site, and only then port the
working logic into `<target>/scripts/v1.js`.

Setup (once per checkout):

    cd scrape-targets/_solo && npm install && npx playwright install chromium
    ln -sfn _solo/node_modules ../node_modules

The symlink lets every target resolve a bare `import { chromium } from 'playwright'`.
Both the install and the symlink are gitignored.

Run a probe:

    node scrape-targets/northdata.de/solo/probe.mjs "BNT Chemicals GmbH"

A probe prints one JSON object with `fields` and exits 0 only on real
extraction; on a bot challenge it exits non-zero with a `reason`. Challenges are
never solved or evaded — a hard block is a finding, not a bug to route around.
