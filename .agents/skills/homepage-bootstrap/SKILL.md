---
name: homepage-bootstrap
description: Use when the CTO-Agent must create, reshape, or simplify its homepage as an early communication bridge from the terminal to the BIOS, especially after owner feedback from the terminal or BIOS chat.
---

# Homepage Bootstrap

This skill exists for the first communication bridge of the CTO-Agent.

The homepage is not a fixed product page.
It is a mutable bootstrap surface that the agent should be able to reshape when the owner says:

- this communication path is not good enough
- the look is wrong
- the homepage should be simpler, clearer, stronger, or more trustworthy
- BIOS should be more visible
- terminal fallback should be emphasized

## Required sources

Read these files first:

1. `../../../contracts/homepage/homepage-policy.json`
2. `../../../contracts/bios/bios.json`
3. `../../../contracts/history/origin-story.md`
4. `./references/homepage-bios-bridge-contract.md`
5. `./references/homepage-bootstrap-tool-contract.md`
6. `./assets/homepage-bios-bridge-template.html`
7. `../bios-interface-bootstrap/SKILL.md`
8. `../bios-interface-bootstrap/assets/bios-start-template.html`
9. `../bios-interface-bootstrap/references/bios-interface-template-contract.md`

If the request is about current runtime trust state, also inspect:

10. `../../../runtime/cto_agent.db`
11. `/api/bios/template-data` when the runtime is live

## Core rules

- Terminal is the first and permanent fallback communication layer.
- The always-on bootstrap loop should be able to react to terminal feedback immediately after startup.
- The homepage should evolve into the preferred 1:1 trust surface when a topic should leave low-trust external channels.
- The homepage should offer a more comfortable first trust surface than the raw terminal, including direct 1:1 chat and image upload.
- The homepage starts as a bridge, not as final owner branding.
- BIOS must remain visible from the homepage.
- The homepage must not collapse into a generic startup, product or marketing landing page while it still acts as BIOS.
- Owner branding must not be locked in before BIOS communication has actually been adopted.
- If the owner gives feedback from the terminal, the agent may still use this skill to reshape the homepage.
- If the owner gives the same feedback later through BIOS chat, the same skill may reshape the homepage again.
- Email, WhatsApp and similar channels are lower-trust surfaces and may trigger a redirect into the homepage 1:1 chat.
- Deep system-foundation changes should not be accepted over low-trust channels; they belong to terminal-level escalation.
- Changes should modify the homepage policy and, when available, append a homepage revision record.
- Start from the bundled `homepage-bios-bridge-template` resource instead of inventing a fresh layout from scratch.
- Treat the BIOS visual shell and the SQLite-backed snapshot contract as one bundle.
- Keep the shell recognizable: top bar, tabs, left rail, right workspace, terminal-fallback footer.
- Prefer `/api/bios/template-data` as the live runtime adapter instead of embedding hardcoded state blobs.
- This skill should ship resources and contracts; later turns should instantiate or mutate from those resources rather than rely on a deterministic homepage generator.

## Build pattern

1. Load the canonical homepage BIOS bridge template resource.
2. Read the current contract state and runtime snapshot.
3. Mutate the template while preserving the shell and explicit write routes.
4. Keep chat, uploads, runtime and trust state inside the same surface.
5. If the shell changes materially, update the bundled resource rather than leaving the improvement only in one page patch.

## What to change

Prefer changing:

- the copy and panel emphasis inside the canonical shell
- `currentTitle`
- `currentHeadline`
- `currentIntro`
- `communicationNote`
- `terminalFallbackNote`
- stage flags in `homepage-policy.json`

Only lock branding after BIOS-based communication and root verification are established.

## Never do

- Never ship a generic homepage just because the owner asked for "simpler".
- Never leave the design intent only in `bios-vanilla-mockup.html` while the live template resource says something else.
- Never use hardcoded mock records when `/api/bios/template-data` or the runtime DB is available.
