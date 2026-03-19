# Homepage Bootstrap Tool Contract

This contract describes what a homepage-setup agent may read, patch and wire when it receives a homepage bootstrap task.

## Goal

Set up or reshape the homepage from the canonical BIOS bridge template resource, not from an ad-hoc heuristic or a blank generic homepage.

## Required read inputs

The agent should read these first:

- `../assets/homepage-bios-bridge-template.html`
- `./homepage-bios-bridge-contract.md`
- `../../../contracts/homepage/homepage-policy.json`
- `../../../contracts/bios/bios.json`
- `../../../contracts/history/origin-story.md`
- `../../../src/app.rs`
- `../../../src/pages.rs`
- `../../../src/runtime_db.rs`
- `GET /api/bios/template-data` when the runtime is live

If the runtime is unavailable, the agent may fall back to:

- `../../../runtime/cto_agent.db`
- the sample payload embedded in the template resource

## Allowed write surfaces

The agent may patch these implementation surfaces when needed:

- `src/pages.rs`
- `src/app.rs`
- `contracts/homepage/homepage-policy.json`
- skill resources under `.agents/skills/homepage-bootstrap/`
- skill resources under `.agents/skills/bios-interface-bootstrap/` when the shared shell changes materially

## Bound UI routes

The homepage template may bind these routes explicitly:

- `POST /homepage/update`
- `POST /bios/chat`
- `POST /bios/upload`
- `POST /homepage/branding-lock`

The homepage may also expose BIOS or root routes as linked boundaries:

- `POST /bios/update`
- `POST /bios/brain-access`
- `POST /bios/freeze`
- `POST /root-auth/set`

## Non-negotiable safety rules

- Do not surface raw secret values.
- Do not expose the superpassword itself.
- Do not fake write paths that do not exist.
- Do not downgrade the page into a generic corporate homepage while it still claims to be BIOS.
- Do not depend on hardcoded mock data once the runtime snapshot is available.

## Anti-heuristic rule

The agent should not invent the shell from scratch for each task.
It should start from the canonical resource and mutate from there.

That keeps the output agentic while avoiding random layout drift.
