# CTOX Candle Engine

`engine/candle/` is the CTOX-owned Candle-based local inference engine tree.

This code no longer represents a generic vendored `mistral.rs` checkout. It is the actively
maintained engine implementation that powers the CTOX gateway/runtime path.

The workspace is now organized by CTOX engine role instead of inherited upstream folder names:

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
- `Qwen/Qwen3.5-27B`

Other upstream model families or larger variants may still exist in the source tree, but they are
not part of the active CTOX support matrix until they are validated end-to-end on target hosts.

The CTOX gateway, runtime launcher, and Candle engine are expected to evolve together as one
first-party serving stack.
