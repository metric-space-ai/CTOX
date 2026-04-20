# Candle Metal Performance Patches

Drop-in changes to candle-metal-kernels that we've verified improve Metal
decode throughput. Not yet upstreamed because they require discussion with
the candle maintainers. Reproduce by:

1. Clone candle at rev `c3bb5bf` somewhere outside the CTOX workspace.
2. Apply the patches in order: `quantized.rs.patch`, `sdpa.rs.patch`,
   then `0003-metal-sticky-encoder.patch`.
3. Point `tools/model-runtime/Cargo.toml` at the patched candle via
   `[patch."https://github.com/huggingface/candle.git"]`.

## Rationale

candle-metal-kernels' quantized matmul and vector-SDPA dispatch paths
call `encoder.use_resource(...)` for each of lhs / rhs / output after
`setBuffer`-binding the same resources. `useResource` is only needed for
indirectly-bound resources (argument buffers, heap-backed allocations);
for direct `setBuffer` bindings, Metal already tracks residency from
the binding tables, so the `useResource` calls are redundant driver
round-trips.

Removing them measured on Apple M5 with Gemma 4 Q4K bench:

  E2B decode:    241 → 263 T/s   (+9 %)
  E4B decode:     17 →  18 T/s   (+5 %)
  E2B prefill:   489 → 511 T/s   (+4 %)
  E4B prefill:   225 → 242 T/s   (+7 %)

Correctness: diff-tested against the default build — identical outputs.

## Files

- `quantized.rs.patch` — strip useResource from call_quantized_matmul_mv_t
  and call_quantized_matmul_mm_t (the Q-quant paths).
- `sdpa.rs.patch` — strip useResource from call_sdpa_vector (the decode
  hot-path SDPA kernel).
- `0003-metal-sticky-encoder.patch` — opt-in (default-on) sticky
  `MTLComputeCommandEncoder` reuse across dispatches on the same command
  buffer. Adds `ComputeCommandEncoder::new_sticky`, caches the encoder in
  `EntryState.current_encoder`, and ends it explicitly before blit or
  commit. Saves 2 Metal syscalls per dispatch (~6300/token on Gemma 4
  E4B). Disable with `CANDLE_METAL_STICKY_ENCODER=0`. Perf impact not yet
  measured — user to bench on Apple M5.

The mask and intermediate-buffer useResource calls in other SDPA
variants are left untouched because they're on less-hot paths and
removing them would need more testing.
