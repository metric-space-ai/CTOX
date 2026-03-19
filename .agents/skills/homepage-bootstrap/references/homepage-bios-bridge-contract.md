# Homepage BIOS Bridge Contract

This reference combines two things that had drifted apart:

- the visual intent from `bios-vanilla-mockup.html`
- the live runtime binding from the SQLite-backed BIOS template contract

The result is the canonical homepage starting point for future agent turns.

## Purpose

The homepage is not a normal marketing site.
It is the first comfortable 1:1 trust bridge above the terminal.

That means:

- it should look like a BIOS utility when it still carries the BIOS name
- it should expose chat, runtime and trust state inside the same shell
- it should be able to reshape itself later without abandoning the shell too early

## Canonical shell

Keep this shell unless the owner explicitly asks to abandon it:

1. top status bar
2. tab row
3. left control rail
4. right active workspace
5. footer note that terminal fallback remains available

This shell is not decoration. It is the agent's stable mental model for homepage work.

## Visual rules

- Favor BIOS utility language over a generic startup or product landing page.
- Keep the page dense, operational and trust-oriented.
- Chat belongs inside the BIOS-shaped surface, not in a detached widget.
- Runtime state, trust state and owner-facing controls must be visible in named panels.
- If the owner asks for simplification, simplify inside the BIOS shell first.

## Data rules

Treat the homepage template as a live adapter between:

- contracts for constitutional truth
- `/api/bios/template-data` for live SQLite-backed runtime truth

Do not rebuild the page around hardcoded mock values.

## Homepage-specific focus

The homepage version of the BIOS shell should prioritize:

- current title, headline and communication note
- direct BIOS chat handoff
- image upload into the same trust surface
- owner trust and root boundary visibility
- runtime queue and active task visibility
- terminal fallback visibility

## Template identity

Until the owner explicitly chooses another shell, the canonical template id is:

- `homepage-bios-bridge-template`

This name is the contract marker that the task should start from the bundled resource instead of a fresh blank-page design.

## Mutation rule

The agent may mutate copy, wording, panel contents and emphasis.
It should not throw away the shell or its data contract by default.
