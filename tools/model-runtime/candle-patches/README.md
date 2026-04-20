# Candle Metal Performance Patches

Drop-in changes to candle-metal-kernels that we've verified improve Metal
decode throughput. Not yet upstreamed because they require discussion with
the candle maintainers. Reproduce by:

1. Clone candle at rev `c3bb5bf` somewhere outside the CTOX workspace.
2. Apply the patches in order: `quantized.rs.patch`, then `sdpa.rs.patch`.
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

The mask and intermediate-buffer useResource calls in other SDPA
variants are left untouched because they're on less-hot paths and
removing them would need more testing.

## Falsified hypothesis: sticky compute-encoder reuse

We also tried caching one `MTLComputeCommandEncoder` across all
dispatches on a command buffer (only ending it at blit or commit
boundaries), to eliminate the ~6300 create/endEncoding syscalls per
E4B decode token. Implemented in fork commit `053097d3` and exported
as `0003-metal-sticky-encoder.patch` in an earlier revision of this
directory.

Measured on Apple M5 with Gemma 4 E4B Q4K, 3 iter + 1 warmup:

  decode sticky-off:  18.0 ± 0.1 T/s
  decode sticky-on:   18.1 ± 0.2 T/s    (within noise)
  prefill sticky-off: 226.1 ± 2.3 T/s
  prefill sticky-on:  222.5 ± 6.3 T/s   (within noise)

CPU-side syscall overhead is *not* the binder on base Apple M5 for
E4B decode. The remaining wall-time gap vs E2B is GPU-bound —
memory-bandwidth- and/or SIMD-group-saturation-limited, not CPU
serialization. The sticky-encoder patch was reverted to keep the
fork minimal. See `docs/METAL_E4B_INVESTIGATION.md` for the broader
investigation.
