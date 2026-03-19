# BIOS Interface Start Template Contract

This reference turns `bios-vanilla-mockup.html` into a reusable starting point instead of a one-off sketch.

## Review of `bios-vanilla-mockup.html`

What the mockup gets right:

- classic BIOS utility visual language
- stable mental model with tabs on top, controls on the left and the active view on the right
- BIOS, organigram, agent and root concerns are all visible in one place
- chat is treated as part of the BIOS surface rather than an unrelated extra

What the mockup is still missing:

- hardcoded state instead of a defined runtime snapshot contract
- no separation between constitutional data and operational SQLite data
- secrets are shown as editable text instead of being handled as masked or summarized state
- write paths are implied but not anchored to the real Rust routes
- no stable mapping from runtime tables to UI panels

## Canonical shell

Every BIOS start template should keep these shell elements unless the owner explicitly asks otherwise:

1. top status bar for runtime and source state
2. fixed tab row for major BIOS areas
3. left control rail for the active tab
4. right work area for previews, tables, chat logs or action cards
5. footer note that terminal fallback remains available

## Source split

Use two source classes:

- contracts for constitutional truth
  - BIOS identity, mission, communication policy
  - homepage policy and BIOS visibility
  - organigram
  - root-auth policy and freeze rules
- SQLite for live operational truth
  - trust state
  - active focus and open tasks
  - recent BIOS dialogue
  - memory and learning summaries
  - resources and discovered skills
  - uploads and homepage revisions

Do not try to force every constitutional field into SQLite only to satisfy the template.

## Shared snapshot shape

The template should expect a JSON payload with this shape:

```json
{
  "ok": true,
  "generatedAt": "ISO-8601 timestamp",
  "template": {
    "id": "bios-start-template",
    "version": 1,
    "snapshotUrl": "/api/bios/template-data",
    "sqlitePrimary": true
  },
  "sources": {
    "runtimeDbPath": "runtime/cto_agent.db",
    "contracts": {
      "bios": "contracts/bios/bios.json",
      "homepage": "contracts/homepage/homepage-policy.json"
    }
  },
  "contracts": {
    "bios": {},
    "homepage": {},
    "organigram": {},
    "rootAuth": {},
    "genome": {},
    "modelPolicy": {}
  },
  "runtime": {
    "trust": {},
    "agentState": {},
    "browserState": {},
    "focusState": {},
    "openTasks": [],
    "dialogue": [],
    "uploads": [],
    "memorySummary": {
      "ownerCalibration": null,
      "learningWorkingSet": null,
      "learningOperational": null,
      "learningGeneral": null,
      "learningNegative": null
    },
    "memoryItems": [],
    "learningEntries": [],
    "resources": [],
    "skills": [],
    "homepageRevisions": []
  }
}
```

## Tab mapping

- `overview`
  - BIOS identity, homepage title, mission, freeze state, trust state
- `organigram`
  - owner, board, peers, subordinates, reporting line
- `runtime`
  - focus state, queue, tasks, browser/runtime health, resources
- `chat`
  - BIOS dialogue, upload history and BIOS chat handoff controls
- `memory`
  - memory summaries, memory items, learning entries, skills
- `root`
  - superpassword status, brain access, branding lock, BIOS freeze actions

## Write boundaries

The template may prepare these write paths, but should not fake them:

- `POST /bios/update`
- `POST /bios/chat`
- `POST /bios/upload`
- `POST /bios/brain-access`
- `POST /root-auth/set`
- `POST /bios/freeze`

If a template preview is rendered without a live backend, treat those controls as placeholders and keep their intended route visible.

## Stability rule

The owner can always reshape the BIOS interface, but the agent should start from this shell and data contract first.
That keeps installs recognizable while still allowing later owner-specific mutation.
