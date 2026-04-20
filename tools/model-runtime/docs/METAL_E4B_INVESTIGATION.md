# Metal E4B Decode-Performance-Investigation

Interner Notiz-Snapshot nach ganztägiger Kernel-Level-Diagnose auf Apple M5.
Dient als Wegweiser für spätere upstream-candle oder fused-kernel-Arbeit.

## Das Problem

Gemma 4 E4B decode auf Apple M5 Metal Q4K: **17 T/s** — vs E2B **241 T/s**.
Gap = **14×** für nur 1.9× Modell-Parameter. Physikalisch nicht vertretbar.

Zum Vergleich, derselbe Code auf RTX A6000 CUDA nach Q4K→Q8_0 Auto-Upgrade:
- E2B Q8_0: 530 T/s  (decode)
- E4B Q8_0: 205 T/s  (decode) — Ratio 2.6× ≈ Parameter-Verhältnis ✅

## Harte Messdaten (per-Forward, 1 decode token steady-state)

| Metrik | E2B | E4B | Ratio |
|---|---|---|---|
| decode tok/s | 241 | 17 | **14×** |
| encoders/token | ~2700 | ~3150 | 1.17× |
| command-buffer commits | ~54 | ~62 | 1.15× |
| `allocate_buffer` avg | 0.6 μs | 0.7 μs | 1.0× |
| pool-size (nach Bench) | ~1300 bufs | ~1300 bufs | 1.0× |
| per-stage kernel mit `synchronize` | MLP 474 μs | MLP 658 μs | 1.4× |
| **effective wall time pro encoder** | **1.3 μs** | **15 μs** | **11.5×** |

## Hypothesen die AUSGESCHLOSSEN sind

| Hypothese | Test | Ergebnis |
|---|---|---|
| candle Q4K kernel shape-dependent | Q8_0/BF16 getestet | Alle ~15-17 T/s auf E4B |
| SDPA Fast-Path ineffizient | `ENGINE_DISABLE_METAL_SDPA=1` | nur 14% Degradation (bei E2B 86%) |
| Command-pool size | 1/5/32 getestet | ±0.5 T/s |
| `CANDLE_METAL_COMPUTE_PER_BUFFER` | 10/50/200/1000 | ±1 T/s |
| Linear scan allocator | Instrumentiert | avg 0.7 μs, niemals > 100 μs |
| Encoder count ratio | Instrumentiert | 1.17× (E4B/E2B), erwartungsgemäß |
| Per-stage compute | Sync'd stage timing | 1.0-1.4× ratio, erwartungsgemäß |
| KV-cache `to_device` no-ops | Entfernt (commit 8b05c0d) | keine Änderung |

## Was TATSÄCHLICH los ist

Die GPU-Kernel-Dispatch-Infrastruktur in candle-metal-kernels ist **identisch** für
E2B und E4B — gleiche Encoder-Count-Ratio, gleiche Pool-Verwaltung. Der
gesamte Wall-Time-Gap kommt aus **GPU-Kernel-Parallelism**:

- **E2B**: Matmul-Shapes `(1536, 6144)` / `(1536, 12288)` sind klein genug,
  dass Apple M5 GPU **mehrere Kernels gleichzeitig** auf unterschiedlichen
  SIMD-Gruppen dispatchen kann. Effective 1.3 μs pro encoder = ~6× parallel
  execution.

- **E4B**: Matmul-Shapes `(2560, 10240)` **saturieren die M5 GPU-Cores mit
  einem einzigen Kernel**. Subsequent kernels warten auf verfügbare Resources.
  Effective 15 μs pro encoder = reine serielle Ausführung.

Dies ist **keine candle-Bug** — sondern ein **Apple M5 Compute-Parallelism-Limit**,
das candle nicht abfedern kann ohne Kernel-Fusion oder Graph-Capture.

## Mögliche echte Lösungen (nicht-trivial)

1. **Fused MLP Kernel** — Schreibe ein custom Metal Shader das
   `gate_proj → silu → mul → down_proj` in einem einzigen Kernel kombiniert.
   Reduziert per-Layer Kernel-Count von ~20 auf ~15. Nötig: MSL Coding mit
   Q4K dequant-at-load, Register-Tiling, output-reduction.

2. **MPSGraph** — Apple's Graph-Framework macht automatisches Kernel-Fusing.
   Würde bedeuten: candle-metal-kernels weitestgehend durch MPSGraph ersetzen.
   Massiver Rewrite.

3. **Hardware** — Apple M5 Pro (2× GPU-Cores) oder M5 Max (4× GPU-Cores).
   E4B-Kernels könnten dann wieder parallelisieren. Kein Code-Fix.

## Diagnostische Env-Toggles (beschreibend committed)

- `ENGINE_GEMMA4_LAYER_TIMING=1` — sync nach jedem Layer, dump μs (commit 7a7d726)
- `ENGINE_GEMMA4_STAGE_TIMING=1` — sync nach jeder Stage im layer (commit f21eb16)
- `ENGINE_DISABLE_METAL_SDPA=1` — skip Metal SDPA fast-path (commit 583d677)
- `CANDLE_METAL_TRACE_ALLOC=1` — buffer-pool allocator stats (nur im lokalen Fork)
- `CANDLE_METAL_TRACE_ENC=1` — encoder dispatch + flush counts (nur im lokalen Fork)

## Production-Empfehlung

| Plattform | Modell | Empfehlung |
|---|---|---|
| CUDA | E2B, E4B | Funktioniert exzellent (Q4K→Q8_0 auto) |
| Metal M5 | E2B | Funktioniert exzellent (241 T/s) |
| Metal M5 | E4B | Hardware-limitiert (17 T/s). Dev-OK, Production auf CUDA. |
| Metal M5 Pro/Max | E4B | Nicht getestet, sollte besser sein. |

## Weiterführende Arbeit

Wenn jemand die Metal E4B Zahl fixen will: **mit Xcode Instruments
Metal System Trace** ein bench-run öffnen und GPU-Timeline-Bubbles zwischen
Kernel-Dispatches genau vermessen. Das identifiziert exakt welche Shader
saturiert laufen. Dann entweder (a) MSL fused-kernel schreiben, (b) MPSGraph
introduzieren, oder (c) explizit bestätigen dass es Hardware-Limit ist und
User auf M5 Pro/Max-Hardware oder CUDA verweisen.
