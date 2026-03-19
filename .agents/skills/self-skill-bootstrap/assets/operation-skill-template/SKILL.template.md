---
name: <capability>-operations
description: Use when the CTO-Agent needs to operate, inspect or patch the current <capability> tool in this repo.
---

# <Capability> Operations

Use this skill when the concrete tool already exists and later turns need to run it, inspect its state, or patch it safely.

## Required sources

Read these first:

1. `<tool-path>`
2. `<schema-or-policy-path>`
3. `<runtime-db-or-config-path>`

## Commands

- `<command-1>`
- `<command-2>`
- `<command-3>`

## Runtime contract

- Tool path: `<tool-path>`
- Runtime storage: `<db-or-storage-path>`
- Inputs: `<main-inputs>`
- Outputs: `<main-outputs>`
- Side effects: `<interrupts / queue writes / files / network>`

## Failure handling

- If `<failure-mode>`, inspect `<log-or-db-path>`
- If `<provider-or-auth-problem>`, patch `<tool-path>` before retrying
- Verify with `<bounded-verification-step>`

## Never do

- Never assume the tool still behaves like an older version without checking the current file.
- Never claim success without running the bounded verification step.
