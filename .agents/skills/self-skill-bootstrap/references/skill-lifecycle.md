# Skill Lifecycle

Use this reference when deciding whether a new capability should become a repo-local skill and which kind of skill to create.

## Decision rule

Create or update a skill when at least one is true:

- the tool or workflow will probably be needed again
- later turns would otherwise have to rediscover commands, schemas or paths
- the capability includes trust, interrupt or storage rules that should not be improvised
- the capability is likely to be patched in place by the CTO-Agent later

## Skill kinds

Operations skill:

- use when the tool already exists
- teach exact commands, paths, env vars, outputs and repair entry points
- keep it close to the current implementation
- if you want the agent itself to earn the operationalization step, provide only a template until the tool is actually proven

Bootstrap skill:

- use when future turns may need to rebuild or replace the capability itself
- include selection rules, architecture constraints and template starting points
- point to assets or references for concrete scaffolding

## Minimal folder shape

```text
.agents/skills/<slug>/
├── SKILL.md
├── references/      # optional
└── assets/          # optional
```

## Binding into later turns

There is no separate plugin registry.

The binding path is:

1. write the skill files under `.agents/skills/<slug>/`
2. let the runtime prepare the next context package
3. the next turn sees the skill in the available skill catalog
4. if the skill is relevant, the turn can read `SKILL.md` over bounded exec work and use it

Templates and examples under another skill's `assets/` are not yet bound capabilities. They are only scaffolds until the agent writes the real `.agents/skills/<slug>/SKILL.md`.

## Description guidance

The description should say:

- when the skill should be used
- what capability it covers
- whether it is about operations, building or replacing a capability

Good:

- `Use when the CTO-Agent needs to operate the current IMAP/SMTP mail CLI, inspect inbox sync state or patch the interrupt hook.`

Weak:

- `Mail helper`

## Operational completeness

An operations skill should usually include:

- concrete tool path
- main commands
- important runtime files or databases
- interrupt or queue side effects
- explicit Definition of Done with success evidence, blocked conditions and verification path
- bounded verification step
- failure handling guidance

## Example-first pattern

If a concrete operations skill should still be authored by the CTO-Agent itself:

1. ship a domain example as a template resource
2. let the agent build or stabilize the tool
3. let the agent instantiate the final operations skill from that template
4. only then does the agent get the persistent later-turn capability
