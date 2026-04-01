# Skill Lifecycle Ledger

## Skill Transition
- Skill: universal-scraping
- From: none
- To: draft
- Reason: Added the first CTOX-native universal scraping skill plus helper tooling for target registration, script revisioning, template promotion, and run/artifact tracking.
- Evidence: Repo now contains skills/.system/universal-scraping with SKILL.md, references, registry/query/store helpers, and focused registry tests.

## Skill Transition
- Skill: interactive-browser
- From: none
- To: draft
- Reason: Added the first CTOX-native specialist skill for real browser interaction through js_repl-backed Playwright and paired it with a native `ctox browser` setup/doctor bridge.
- Evidence: Repo now contains skills/.system/interactive-browser/SKILL.md, chat runtime now injects js_repl-related Codex overrides, and the new browser module has focused unit tests for the reference workspace contract.
