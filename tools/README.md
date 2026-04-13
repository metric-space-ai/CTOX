# Tools Tree

The `tools/` tree is where CTOX keeps its integrated hard-fork runtime trees.

- `tools/agent-runtime/`
  Integrated execution hard fork derived from `openai/codex`.
- `tools/model-runtime/`
  Integrated local serving hard fork with Candle-focused lineage and CTOX-specific custom code.

Rules:

- Code under `tools/` is not described as loose third-party dependencies.
- Each integrated subtree keeps its own provenance and license files.
- CTOX-specific patches inside these trees are part of the CTOX fork state, not floating dependency overrides.
- These trees remain source-owned in the main repository, without nested `.git` metadata or automatic upstream sync paths.
