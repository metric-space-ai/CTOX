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
4. `../bios-interface-bootstrap/SKILL.md` if the request is specifically about the BIOS interface shell or a stable BIOS start template

If the request is about current runtime trust state, also inspect:

5. `../../../runtime/cto_agent.db`

## Core rules

- Terminal is the first and permanent fallback communication layer.
- The always-on bootstrap loop should be able to react to terminal feedback immediately after startup.
- The homepage should evolve into the preferred 1:1 trust surface when a topic should leave low-trust external channels.
- The homepage should offer a more comfortable first trust surface than the raw terminal, including direct 1:1 chat and image upload.
- The homepage starts as a bridge, not as final owner branding.
- BIOS must remain visible from the homepage.
- Owner branding must not be locked in before BIOS communication has actually been adopted.
- If the owner gives feedback from the terminal, the agent may still use this skill to reshape the homepage.
- If the owner gives the same feedback later through BIOS chat, the same skill may reshape the homepage again.
- Email, WhatsApp and similar channels are lower-trust surfaces and may trigger a redirect into the homepage 1:1 chat.
- Deep system-foundation changes should not be accepted over low-trust channels; they belong to terminal-level escalation.
- Changes should modify the homepage policy and, when available, append a homepage revision record.
- If the request is BIOS-first and needs a stable shell, reuse the bundled BIOS start template instead of inventing a fresh layout from scratch.

## What to change

Prefer changing:

- `currentTitle`
- `currentHeadline`
- `currentIntro`
- `communicationNote`
- `terminalFallbackNote`
- stage flags in `homepage-policy.json`

Only lock branding after BIOS-based communication and root verification are established.
