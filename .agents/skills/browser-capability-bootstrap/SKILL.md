---
name: browser-capability-bootstrap
description: Use when the CTO-Agent needs a real browser automation surface, must decide whether browser work is safe to perform on the current trust level, or must shape recurring browser work into reviewed browser capabilities.
---

# Browser Capability Bootstrap

This skill governs how the CTO-Agent should think about browser automation.

It is inspired by the reviewed browser capability and browser-agent job ideas in `local_ai_tunes`, but it is not tied to the browser-side WebGPU training path from that project.

## Required sources

Read these first:

1. `../../../contracts/browser/browser-capability-policy.json`
2. `../../../contracts/bios/bios.json`
3. `../../../contracts/org/organigram.json`
4. `../../../contracts/history/origin-story.md`

If the request concerns specialization or repeated work, also read:

5. `../../../contracts/pipelines/specialist-model-pipeline.json`

## Core rules

- Browser work must happen in a real browser runtime, not from invented page state.
- The main CTO-Agent should not swallow whole browser traces into its own context.
- Prefer reviewed browser capabilities with clear input/output contracts.
- Treat the currently published reviewed capability path as the champion until a challenger proves stronger.
- Prefer challenger smoke/eval against a fixed mini-suite before overwriting a stable reviewed path.
- Keep each browser-capability experiment narrow: one artifact family per round unless a directly blocking defect forces a coupled repair.
- If a browser request is sensitive and came from a low-trust channel, redirect to BIOS chat or terminal depending on impact.
- Repeated successful browser flows should graduate into:
  - a reviewed browser capability
  - a deterministic worker
  - or a small specialist model

## Output style

When using this skill, prefer answers in this shape:

1. what browser capability is needed
2. whether the current trust level is sufficient
3. whether the task should stay a generic browser action or be promoted into a specialist path

## Never do

- Never pretend browser work happened if no real browser runtime exists.
- Never promote a recurring browser task straight from raw traces into training.
- Never let low-trust external channels directly authorize high-impact browser actions.
