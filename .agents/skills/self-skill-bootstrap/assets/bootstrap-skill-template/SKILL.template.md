---
name: <capability>-bootstrap
description: Use when the CTO-Agent needs to create, replace or re-architect the <capability> capability in this repo.
---

# <Capability> Bootstrap

Use this skill when no suitable tool exists yet, the current tool is too rigid, or the capability must be rebuilt in a form the CTO-Agent can patch later.

## Required sources

Read these first:

1. `<governing-policy-path>`
2. `<shared-schema-or-contract-path>`
3. `<template-or-asset-path>`

## Core rules

- Prefer the smallest reliable local tool that solves the current capability.
- Keep persistence and policy contracts separate from transport or provider adapters.
- If the resulting tool will be reused, also create or update the paired operations skill.
- Verify with a real bounded test before declaring the bootstrap complete.
- State an explicit Definition of Done so later turns know when the capability is really complete, still blocked or only partially advanced.

## Build pattern

1. Confirm the concrete need and trust boundary.
2. Check whether an existing local tool can be patched instead of replaced.
3. Start from the bundled template or the smallest viable scaffold.
4. Keep the first version easy to modify from the CLI.
5. Persist state in the shared contract or database chosen for this capability.
6. Add interrupt or queue hooks only through the canonical runtime bridge.
7. Create or update the paired operations skill.

## Definition of Done

- The capability objective and claimed scope are explicit.
- At least one fresh bounded verification step or artifact proves the capability works.
- The skill records the success evidence, remaining blockers and safe failure boundaries.
- Later turns can tell the difference between `done`, `blocked` and `continue` without guessing from tone.

## Never do

- Never bury reusable architecture knowledge only in transient task output.
- Never lock the capability into a framework that the CTO-Agent cannot patch quickly.
