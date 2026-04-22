// Dense bf16 × bf16 matmul, f32 accumulation.
//
// Math (standard row-major, NOT transposed-B):
//   C[i, j] = sum_k A[i, k] * B[k, j]
//
// Shapes (row-major, C order):
//   A : [M, K]   bf16
//   B : [K, N]   bf16
//   C : [M, N]   bf16 or f32  (two entry points)
//
// Accumulation is f32: each fetched bf16 is cast via __bfloat162float
// before the multiply-add, and the running sum stays in f32 for the
// whole reduction. This gives us attention-score-usable precision
// without paying the bandwidth cost of f32 weights.
//
// Launch convention (32x32 output tile):
//   grid_dim  = (N / 32, M / 32, 1)
//   block_dim = (32, 32, 1)      — 1024 threads, one per output elem
//   shmem     = 2 * (TILE_K + 1) * 32 * sizeof(bf16)
//                 two buffers, A and B, padded to kill bank conflicts.
//
// This is the plain-fp32-FMA correctness port. No tensor-core MMA.
// Tensor-core mmq is a later optimization; get the math right first.
//
// Extern "C" entry points:
//   * matmul_bf16_bf16_out — C is bf16 (output projections)
//   * matmul_bf16_f32_out  — C is f32  (Q·K^T attention scores)
// Load via cudarc's `module.load_function(...)`.

#include <cuda_bf16.h>

// Tile dimensions. 32x32 output tile matches the 32x32 thread block so
// every thread owns exactly one output element and computes one dot-
// product across K via staged tile loads. TILE_K = 32 keeps shared-
// memory footprint modest (~2 * 33 * 32 * 2B = 4.1 KiB) and lets us
// unroll the inner K loop tightly.
#define TILE_M 32
#define TILE_N 32
#define TILE_K 32

// +1 padding on the K dimension of the shared tile kills 32-way bank
// conflicts when column j of B is read by all 32 threads in a warp.
// Each shared row is (TILE_K + 1) bf16 = 66 bytes → stride that the
// bank hash spreads cleanly across banks.
#define SH_K_STRIDE (TILE_K + 1)

// ---------------------------------------------------------------------------
// Device-side tile loader: pull one TILE_M × TILE_K slab of A and one
// TILE_K × TILE_N slab of B into shared. Each thread loads one element
// of A (its (ty, tx)-th entry of the slab) and one element of B. Out-
// of-range positions would corrupt memory — we assert divisible dims
// at the host so this can't happen at runtime; still zero-fill to make
// the unrolled accumulation a no-op on any trailing partial tile.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Core matmul, templated on output dtype via a tiny tag-dispatch macro.
// We avoid a real template so we can keep extern "C" symbols.
// ---------------------------------------------------------------------------

// Store helpers: collapse the two output-dtype variants to a single
// inline path. Host wrappers pick the entry point; the kernel body is
// shared.
__device__ __forceinline__ void store_bf16(__nv_bfloat16 * c, float v) {
    *c = __float2bfloat16_rn(v);
}
__device__ __forceinline__ void store_f32(float * c, float v) {
    *c = v;
}

// Shared kernel body. Templated would be cleaner; we use two extern "C"
// entry wrappers below so cudarc can find them by symbol.
template <typename OutT, void (*Store)(OutT *, float)>
__device__ __forceinline__ void matmul_bf16_impl(
    const __nv_bfloat16 * __restrict__ A,
    const __nv_bfloat16 * __restrict__ B,
    OutT * __restrict__ C,
    int M, int K, int N
) {
    const int tx = threadIdx.x;   // 0..31, column within tile
    const int ty = threadIdx.y;   // 0..31, row within tile
    const int bx = blockIdx.x;    // tile column: N-direction
    const int by = blockIdx.y;    // tile row:    M-direction

    const int row = by * TILE_M + ty;   // global row in C / A
    const int col = bx * TILE_N + tx;   // global col in C / B

    // Shared tiles. A_tile is row-major [TILE_M, SH_K_STRIDE]; B_tile
    // is row-major [TILE_K, TILE_N + 1] with a +1 pad on the N
    // dimension so warp reads down column `tx` hit different banks.
    __shared__ __nv_bfloat16 A_tile[TILE_M][SH_K_STRIDE];
    __shared__ __nv_bfloat16 B_tile[TILE_K][TILE_N + 1];

    float acc = 0.0f;

    // Walk K in TILE_K-sized chunks. Host validation guarantees
    // K % TILE_K == 0, so no tail branch needed.
    for (int k0 = 0; k0 < K; k0 += TILE_K) {
        // Each thread loads A[row, k0 + tx] into A_tile[ty][tx].
        // ty is the slab row (0..TILE_M-1), tx is the slab column
        // (0..TILE_K-1). That's one bf16 per thread, coalesced on the
        // K axis (threads in a warp share ty and vary tx).
        const int a_k = k0 + tx;
        A_tile[ty][tx] = (row < M && a_k < K)
            ? A[(size_t)row * K + a_k]
            : __float2bfloat16_rn(0.0f);

        // Each thread loads B[k0 + ty, col] into B_tile[ty][tx]. ty is
        // the K-row within the tile, tx is the N-column.
        const int b_k = k0 + ty;
        B_tile[ty][tx] = (b_k < K && col < N)
            ? B[(size_t)b_k * N + col]
            : __float2bfloat16_rn(0.0f);

        __syncthreads();

        // Inner K reduction: each thread dot-products A_tile row `ty`
        // against B_tile column `tx` for 32 positions. Unrolled to let
        // the compiler schedule FMAs back-to-back and issue all 32
        // shared loads as bank-conflict-free broadcasts (A) and
        // columnar fetches (B, safe due to the +1 pad).
        #pragma unroll
        for (int kk = 0; kk < TILE_K; ++kk) {
            const float a = __bfloat162float(A_tile[ty][kk]);
            const float b = __bfloat162float(B_tile[kk][tx]);
            acc += a * b;
        }

        __syncthreads();
    }

    // Store. Out-of-range threads (when M or N aren't multiples of 32 —
    // which host validation forbids, but we keep the guard for safety)
    // simply skip. Accumulated acc stays in f32 up to the final store.
    if (row < M && col < N) {
        Store(C + (size_t)row * N + col, acc);
    }
}

// ---------------------------------------------------------------------------
// Entry points. cudarc loads these by `extern "C"` symbol; template
// instantiations live inside.
// ---------------------------------------------------------------------------

extern "C" __global__ void matmul_bf16_bf16_out(
    const __nv_bfloat16 * __restrict__ A,
    const __nv_bfloat16 * __restrict__ B,
    __nv_bfloat16 * __restrict__ C,
    int M, int K, int N
) {
    matmul_bf16_impl<__nv_bfloat16, store_bf16>(A, B, C, M, K, N);
}

extern "C" __global__ void matmul_bf16_f32_out(
    const __nv_bfloat16 * __restrict__ A,
    const __nv_bfloat16 * __restrict__ B,
    float * __restrict__ C,
    int M, int K, int N
) {
    matmul_bf16_impl<float, store_f32>(A, B, C, M, K, N);
}
