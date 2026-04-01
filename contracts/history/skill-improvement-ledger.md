# Skill Improvement Ledger

This ledger records successful or partially successful CTOX skill self-improvement reviews.

Use it for:

- skill creations
- skill refinements
- skill helper or contract fixes
- validated behavior improvements in the skill family

Do not use it for broad architecture history; that belongs in `creation-ledger.md`.

## 2026-04-01
- Status: successful
- Summary: Added the `interactive-browser` skill and a native `ctox browser` bridge so CTOX can treat real browser interaction as an explicit fourth web path instead of an ad hoc external trick.
- Goal: Distinguish browser-backed live interaction from `WebSearch`, `WebRead`, and `WebScrape`, and make the Playwright/js_repl path repeatable enough for normal CTOX use.
- Evidence: `src/browser.rs` now scaffolds and diagnoses the standard Playwright reference workspace, `src/inference/chat.rs` now injects `features.js_repl=true` plus Playwright module search-path overrides into Codex sessions, the repo now contains `skills/.system/interactive-browser/SKILL.md`, and focused Rust tests cover the reference package contract and js_repl override generation.
- Skills: interactive-browser
