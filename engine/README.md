# CTOX Engine

The `engine/` tree contains CTOX's local inference engine implementation.

- `engine/candle/` is the active Candle-based engine codebase used by CTOX.
- The Candle workspace is now organized by engine role: `cli/`, `server/`, `core/`, `sdk/`,
  `vision/`, `quant/`, `paged-attn/`, `audio/`, `mcp/`, and `macros/`.
- CTOX treats the whole tree as one first-party engine layer rather than as a loose vendor drop.
