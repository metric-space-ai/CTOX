# ctox-qwen35-27b-q4km-dflash

Self-contained Rust inference crate for the **Qwen3.5-27B Q4_K_M** target
model paired with the **z-lab DFlash** speculative-decoding block-diffusion
draft, treated as one curated model.

Byte-exact port of the [lucebox/dflash](https://github.com/lucebox/dflash)
C++ reference. Every Rust module corresponds 1:1 to a `.cpp`/`.h`/`.cu`
file in the reference; ops are built on top of the ggml C API via the
in-crate FFI bindings in `src/ffi.rs`.

## Layout

```
src/inference/models/qwen35_27b_q4km_dflash/
├── Cargo.toml                  standalone crate (no parent workspace)
├── README.md                   this file
├── build.rs                    ggml link + f16_convert.cu nvcc step
├── vendor/
│   ├── ggml-cuda/              61 ggml-cuda .cu files + f16_convert.cu + .cuh headers
│   │                            (source-of-truth: llama.cpp b16de65)
│   ├── ggml-include/           ggml C API headers (ggml.h, gguf.h, etc.)
│   ├── llama-cpp.version       upstream commit pin
│   └── dflash.version          lucebox commit pin
└── src/
    ├── lib.rs                  re-exports + constants + last_error
    ├── ffi.rs                  raw ggml / ggml-backend / ggml-cuda / gguf bindings
    ├── model.rs                TargetWeights / DraftWeights / TargetCache
    │                            (ref: internal.h)
    ├── loader.rs               gguf target + safetensors draft loaders
    │                            (ref: gguf_target_loader.cpp + safetensors_draft.cpp)
    ├── graph.rs                all ggml graph builders + delta-net chunking
    │                            (ref: qwen35_target_graph.cpp +
    │                                  qwen3_dflash_graph.cpp +
    │                                  delta_net_chunked.cpp)
    ├── ddtree.rs               DDTree tree-verify helpers
    ├── driver.rs               3-mode spec-decode driver
    │                            (ref: test_dflash.cpp)
    └── bin/bench.rs            `qwen35-27b-q4km-dflash-bench` CLI
```

## Self-containment

Every file the crate needs for target + draft forward passes lives inside
this directory — **no code is shared with other models**. The vendored
`ggml-cuda/` tree (61 `.cu` files) is llama.cpp's ggml-cuda backend pinned
to the `llama-cpp.version` commit and is consumed via the lucebox-built
`libggml-cuda.so` at link time. The one vendored kernel authored
outside llama.cpp — `f16_convert.cu` — sits alongside them and is
compiled by `build.rs` via nvcc.

Per-compute-capability optimization happens at the nvcc layer:
`CTOX_CUDA_SM` (default `86`) is passed as `-arch=sm_XX` when compiling
`f16_convert.cu`, and the linked `libggml-cuda.so` is itself built with
the same SM target.

## Building

```bash
# dev box with lucebox reference build tree available:
GGML_LIB_DIR=/home/metricspace/dflash-ref/dflash/build/deps/llama.cpp/ggml/src \
    cargo build --release --features=cuda

# just the Rust surface, no CUDA toolchain required:
cargo check
```

## Running the bench

```bash
GGML_LIB_DIR=<path>                                                          \
    LD_LIBRARY_PATH=$GGML_LIB_DIR:$GGML_LIB_DIR/ggml-cuda:$LD_LIBRARY_PATH   \
    cargo run --release --features=cuda --bin qwen35-27b-q4km-dflash-bench   \
        -- <target.gguf> <draft.safetensors> <prompt.bin> <n_gen> <out.bin>  \
        --fast-rollback                                                      \
        --ddtree --ddtree-budget 22
```

Verifies bit-exact against the reference's `test_dflash` output via `cmp`
on the `out.bin` file.

## Performance (as of last A6000 run, before this migration)

| Mode                     |   Rust |   Ref | Gap |
|--------------------------|-------:|------:|----:|
| chain-verify replay      |  61.40 | 62.40 | 1.6% |
| `--fast-rollback`        |  73.54 | 74.66 | 1.5% |
| `--ddtree --budget=22`   |  80.99 | 86.35 | 6.2% |

All three modes `cmp`-byte-identical with the C++ reference output.

## Porting discipline

Every function carries a `// ref: <file>:<line-range>` doc annotation so
reviewers can diff against the C++ source line-by-line. Variable names
match the reference (`ne[0..3]` / `nb[0..3]` etc.). Comments from the
reference are translated verbatim when they describe algorithm;
paraphrased only when they reference C/C++ constructs that don't exist
in Rust.
