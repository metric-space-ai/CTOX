# ctox-vllm-serve

`ctox-vllm-serve` is the CTOX-owned fork of the former `mistral.rs` runtime tree.

Inside CTOX, this engine is no longer treated as a generic vendored dependency. It is the
actively maintained local inference backend that powers the CTOX proxy/runtime path.

Current active support in CTOX is intentionally narrow:

- `openai/gpt-oss-20b`
- `Qwen/Qwen3.5-27B`

Other upstream model families or larger variants may still exist in the source tree, but they are
not part of the active CTOX support matrix until they are validated end-to-end on target hosts.

The CTOX proxy and runtime launcher are expected to evolve together with this engine fork.
