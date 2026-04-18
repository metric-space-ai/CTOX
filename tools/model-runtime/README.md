# CTOX Candle Fork

`src/execution/models/runtime/` is CTOX's integrated Candle-derived serving tree.

This subtree carries historical naming debt: parts of the fork lineage were previously carried
through `mistral.rs` and later under the generic `engine` label. In CTOX, the canonical name for
this serving subtree is now `candle-fork` so the fork is described by its real role instead
of an accidental early model choice or a meaningless umbrella name.

The workspace is currently organized by serving-role modules:

- `cli/`
- `server/`
- `core/`
- `sdk/`
- `vision/`
- `quant/`
- `paged-attn/`
- `audio/`
- `mcp/`
- `macros/`

Current active support in CTOX is intentionally narrow:

- `openai/gpt-oss-20b`
- `Qwen/Qwen3.6-35B-A3B`

Other upstream model families or larger variants may still exist in the source tree, but they are
not part of the active CTOX support matrix until they are validated end-to-end on target hosts.

This subtree is integrated source with CTOX-specific patches. It is not presented as a pristine
upstream checkout, and it is not the same thing as the CTOX agent/orchestration layer.
