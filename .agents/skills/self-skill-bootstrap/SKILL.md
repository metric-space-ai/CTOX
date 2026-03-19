---
name: self-skill-bootstrap
description: Use when the CTO-Agent needs to create, update or operationalize its own repo-local skills so new tools and workflows remain available in later turns.
---

# Self Skill Bootstrap

This skill is the repo-local equivalent of an external skill-creator. Use it when the CTO-Agent must turn a reusable tool, workflow or repeated repair pattern into a persistent repo skill.

## Required sources

Read these first:

1. `./references/skill-lifecycle.md`
2. `./assets/operation-skill-template/SKILL.template.md`
3. `./assets/bootstrap-skill-template/SKILL.template.md`
4. `./assets/email-operations-example/SKILL.template.md` if the current capability is email and you need a concrete operations-skill starting point without precommitting the final live skill for yourself

Then read the domain-specific bootstrap skill if one already exists, for example:

- `../communication-client-bootstrap/SKILL.md` for mail, chat or webhook clients
- other repo-local skills under `../`

## Core rules

- Do not leave reusable capability only in ad-hoc shell memory. Encode it as a repo-local skill under `.agents/skills/<slug>/`.
- Prefer an operations skill when a concrete tool already exists and later turns mainly need to run, inspect or patch it.
- Prefer a bootstrap skill when later turns may need to rebuild or replace the capability itself, not just operate it.
- When a new tool becomes usable, create or update the paired operations skill in the same work pass.
- If the capability owner should earn the build milestone itself, keep only a template or example as a resource and let the agent instantiate the final live skill after the tool really works.
- Keep `SKILL.md` lean. Put detailed schemas, policies or templates into `references/` or `assets/`.
- Write a concrete frontmatter `description`; later turns rediscover skills first through name and description.
- Do not create auxiliary READMEs or process documents around a skill. The skill folder should stay minimal and operational.
- Never hardcode live credentials into skill templates or committed tool code.

## Binding rule

The runtime rescans `.agents/skills` while preparing later turns.

That means a skill becomes available to later turns when:

1. the folder exists under `.agents/skills/<slug>/`
2. `SKILL.md` exists with valid frontmatter
3. the next context package is prepared

You do not need a separate registration ritual. Writing the repo-local skill files is the binding step.

## Build pattern

1. Confirm the missing or fragile capability is truly reusable across future turns.
2. Build or patch the concrete tool first in the workspace.
3. Decide whether the future need is mostly operations or mostly rebuilding.
4. Start from the matching template in `assets/`.
5. Fill in exact paths, commands, environment variables, databases, interrupt hooks and failure boundaries.
6. If the tool is user-visible or channel-facing, reference the governing policy or schema files directly from the skill.
7. Keep the description trigger-focused so future turns can match it quickly.
8. If the capability already has a related skill, update that skill instead of forking duplicates.
9. If a concrete example template exists, copy and specialize it instead of hand-writing the skill from scratch.

## Two-skill rule

For a non-trivial new capability, prefer this split:

- bootstrap skill: how to create, replace or re-architect the capability
- operations skill: how to run, monitor, inspect and patch the current concrete tool

Example:

- `communication-client-bootstrap` teaches how to build a communication client class
- `email-operations` would teach how to operate the current mail CLI

If the owner wants the CTO-Agent to claim that success itself, do not precreate the final `email-operations` skill. Ship the example template as a resource and let the agent instantiate the real skill after bounded validation.

## Never do

- Never assume a new tool is self-explanatory to later turns.
- Never write vague descriptions like "general helper skill".
- Never create five near-identical skills when one precise operations skill would do.
- Never claim a capability is bound for later turns if you did not actually write the `.agents/skills/<slug>/SKILL.md` files.
- Never steal the agent's own milestone by prewriting the final live operations skill when a template resource is sufficient.
