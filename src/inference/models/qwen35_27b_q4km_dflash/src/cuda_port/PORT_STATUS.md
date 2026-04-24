# Bare-Metal CUDA-Dispatcher Port — Status

Byte-für-byte Port von llama.cpp's ggml-cuda Host-Side Dispatchern
nach Rust, pro CLAUDE.md Inference-Engine Architecture Rules.

## Was verifiziert läuft (A6000)

| Op | Source | Verify | Max-Drift |
|---|---|---|---:|
| `rms_norm` (f32) | norm.cu:297-475 | `qwen35-27b-q4km-dflash-rms-norm-verify` | 2.4e-7 (~1 ULP) |
| `silu` (f32) | unary.cu:124-178 | `qwen35-27b-q4km-dflash-unary-verify` | 2.4e-7 |
| `neg` (f32) | unary.cu:157 | (same bin) | 0 (exact) |
| `exp` (f32) | unary.cu:201 | (same bin) | 9.5e-7 |

## Infrastruktur (bewiesen, reusable)

- **Vendored tree self-build.** `build.rs::compile_kernel_to_ptx(stem)` feuert
  nvcc gegen `vendor/ggml-cuda/<stem>.cu` mit exakt den Flags die ggml's
  CMake nutzt; kein externer ggml-Build mehr nötig. Getestet: 243 KB
  norm.ptx, ~400 KB unary.ptx auf A6000.
- **Mangled-name extraction.** `build.rs::generate_ptx_entries_module(stem)`
  parst alle `.entry <mangled>(...)` aus der compiled PTX und emittiert
  `$OUT_DIR/<stem>_entries.rs` mit einer `&[&[u8]]` Tabelle NUL-terminierter
  Mangled-Names. Bypassed nvcc's per-translation-unit hash (wichtig für
  internal-linkage `static` op-functors wie `op_silu`).
- **Runtime lookup.** `ptx::find_entry(entries, &[needle1, …])` macht
  substring-AND-matching mit Uniqueness-Check. Needles = stabile Itanium
  mangled-name-Fragmente (z.B. `b"7op_siluE"` für den Functor-Namen +
  `b"EfEvPK"` für den T=float Discriminator).
- **Kernel-Handle-Cache.** `cuda_port::module::porter()` lazy-init via
  `OnceLock`, resolved alle Handles einmal pro Prozess, cached sie in
  `PortedKernels`.
- **Context binding.** `driver::ensure_current_context(ordinal)` setzt
  den Device-Primary-Context current auf dem rufenden Thread, idempotent.
  Nötig weil ggml_backend_cuda_init den Context nicht auf beliebige Threads
  pushed.
- **Verifier-Pattern.** `src/bin/<op>_verify.rs` = standalone Binary das
  den Port gegen CPU-f64-Referenz-Impl vergleicht. Tolerance 1e-5 deckt
  fast-math drift ab.

## Was noch fehlt (priorisiert)

### Phase A — Restliche Op-Dispatcher (~20 ops)

Geschätzt 30-90 Min pro Op je nach Komplexität:

- **einfach** (~30 min): `scale`, `cpy`, `cont` (+ `cont_2d`/`cont_4d`),
  `fill`, `diag`, `get_rows` (nur f32/bf16 paths)
- **mittel** (~60 min): `add`/`sub`/`mul` (binbcast — template-varadic mit
  dim-collapsing), `concat`, `pad`, `repeat_4d`, `cumsum`, `tri`,
  `solve_tri`
- **komplex** (~90 min): `soft_max_ext`, `rope` (M-RoPE section-aware
  variant), `ssm_conv`
- **sehr komplex** (Tage): `mmq_{q4k,q5k,q6k,q8_0,iq4_xs}`,
  `mm_f16`/`mm_bf16` (matmul-family — viele kernel-variants, kompliziertes
  auto-tuning nach Shape), `flash_attn_ext` (custom-FA2 kernels),
  `gated_delta_net` (+ `_tree`, `_tree_persist` — Qwen3.5-spezifische
  DeltaNet-Kernels von lucebox)

### Phase B — Rust-Side Graph-Executor

Heute konstruiert `graph.rs` einen `ggml_cgraph`, der via
`ggml_backend_graph_compute(backend, gf)` ausgeführt wird — das ist genau
die Library-Dispatch-Logik die wir loswerden wollen. Ohne Graph-Executor
existieren die port-Ops nur als standalone-Verifier; der echte
Qwen3.5-Forward geht weiter durch libggml-cuda.so.

Nötig:
- Rust-seitiger `Tensor` struct (device ptr, shape, strides, dtype)
- Op-DAG-Builder (wie ggml's `ggml_mul_mat(ctx, a, b)` → Rust
  `graph.mul_mat(a, b) -> TensorId`)
- Topological sort + Executor der pro Node die passende cuda_port op
  aufruft
- Memory allocator für intermediates (ggml_gallocr Ersatz)
- KV-Cache handling

Geschätzt ~1500-2000 LoC Rust, 3-5 Tage.

### Phase C — graph.rs Cutover

`graph.rs` von ggml-APIs (1912 Zeilen, ~300 op-calls) auf den neuen
Rust-Graph-Builder umschreiben. 2-3 Tage.

### Phase D — Link-Layer-Cutover

`build.rs` trimmt die `libggml*` linkage raus, nur noch `cudart` + `cuda`
(Driver-API) bleiben. CTOX bench-bin + server-bin bauen weiter, nur gegen
Rust-native Pfad.

## Commit-Reihenfolge

```
834e4fd  fix(cuda_port/module): tighten unary needle — op_exp was matching op_expm1
6e82d81  fix(unary_verify): CUresult not u32 for dispatch closure return type
1b31ac0  test(qwen35-bare-metal): unary_verify bin — silu/neg/exp vs CPU reference
71cfc38  port(qwen35-bare-metal): unary ops + PTX-entry-table code-gen
988925b  fix(cuda_port/norm): push all 23 kernel args — C++ defaults don't apply at PTX level
3b662f2  fix(cuda_port): ensure primary context is current before Driver API calls
0cff2b8  test(qwen35-bare-metal): rms_norm_verify bin — ported path vs CPU reference
bbabb32  wire(qwen35-bare-metal): PTX compile + load + kernel-handle resolve
180ffa4  port(qwen35-bare-metal): rms_norm dispatcher (norm.cu → Rust)
84e5758  build(qwen35-bare-metal): add compile_kernel_to_ptx helper
fc8b4ba  scaffold(qwen35-bare-metal): cuda_port module + CUDA Driver API FFI
32e26c1  docs(CLAUDE.md): HARD rules for per-model inference-engine architecture
```
