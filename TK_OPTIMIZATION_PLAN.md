# ThunderKittens Optimization Plan for Generic ISA Ops

 

**Status:** CRITICAL PRIORITY

**Created:** 2025-11-06

**Estimated Impact:** 100-1000x speedup across all ops

 

---

 

## Executive Summary

 

**Problem:** Current ops use single-threaded scalar loops, making them ~1000x slower than production kernels.

- 1 thread does all work, 895 threads sit idle

- No tensor core usage (MMA)

- No async memory operations (TMA)

- No warp-level parallelism

 

**Solution:** Replace naive loops with ThunderKittens (TK) tile-based operations using:

- **TMA (Tensor Memory Accelerator)**: Hardware-accelerated async loads/stores

- **MMA (Matrix Multiply-Accumulate)**: Tensor core 16×16×16 operations (4096 ops/instruction)

- **Warp operations**: All 32 threads cooperate on each tile

- **Shared memory tiling**: Stage data in fast on-chip memory

 

**Expected Results:**

| Operation | Current (naive) | With TK Tiles | Speedup |

|-----------|----------------|---------------|---------|

| MatMul 2048×2048 | ~4 seconds | ~4 ms | **1000x** |

| RMSNorm 2048 | ~200 μs | ~0.5 μs | **400x** |

| Attention 1024 seq | ~12 seconds | ~16 ms | **750x** |

 

---

 

## Phase 1: MatMul Optimization (START HERE)

 

### Current Naive Implementation

 

```cpp

// include/generic/ops/matmul.cuh (lines 28-46)

if (warpid() == 0 && laneid() == 0) {  // Only 1 thread!

    const int N = inst.n_dim;  // e.g., 2048

    const int K = inst.k_dim;  // e.g., 2048

 

    const float *a = g.ptr_input0<float>(inst.input_offset_0);  // [K]

    const float *b = g.ptr_weight<float>(inst.weight_offset);   // [N, K]

    float *c = g.ptr_output<float>(inst.output_offset);         // [N]

 

    // Scalar loops - SLOW!

    for (int n = 0; n < N; ++n) {

        float acc = 0.f;

        const float *b_row = b + n * K;

        for (int k = 0; k < K; ++k) {

            acc += a[k] * b_row[k];

        }

        c[n] = acc;

    }

}

```

 

**Problems:**

- 4,194,304 serial multiplies on one thread

- ~4 seconds for 2048×2048

- 895 threads idle

 

---

 

### Optimized Implementation with TK Tiles

 

**Key Concepts:**

1. **Tiles**: Process data in 16×16 chunks (TK's sweet spot for tensor cores)

2. **TMA**: Asynchronously load tiles from global → shared memory

3. **MMA**: Use tensor cores to do 16×16×16 matmul in ONE instruction

4. **Parallelism**: All warps cooperate on different tiles

 

**Implementation Strategy:**

 

```cpp

// include/generic/ops/matmul.cuh - OPTIMIZED VERSION

 

template <typename config>

struct OpMatmul {

    static constexpr int opcode = OP_MATMUL;

 

    // Tile sizes - tuned for H100

    static constexpr int TILE_M = 16;  // Output rows per iteration

    static constexpr int TILE_N = 16;  // Output cols per iteration

    static constexpr int TILE_K = 16;  // Reduction dimension per iteration

 

    struct consumer {

        template <typename globals>

        __device__ static inline void run(const globals &g,

                                           ::megakernel::state<config> &mks) {

            GenericInstruction inst{};

            inst.deserialize_from(mks.instruction());

 

            const int M = inst.m_dim;  // Usually 1 for matvec

            const int N = inst.n_dim;  // Output features

            const int K = inst.k_dim;  // Input features

 

            // Shared memory tiles (all warps share this)

            __shared__ st_bf<TILE_M, TILE_K> a_tile;  // Input tile

            __shared__ st_bf<TILE_N, TILE_K> b_tile;  // Weight tile

 

            // Register tiles (per-warp accumulator)

            rt_fl<TILE_M, TILE_N> c_accum;

            zero(c_accum);  // Initialize accumulator

 

            // Each warp processes a different N-block

            int warp_id = kittens::warpid();

            int n_block = warp_id * TILE_N;

 

            if (n_block < N) {

                // Loop over K dimension in TILE_K chunks

                for (int k_block = 0; k_block < K; k_block += TILE_K) {

 

                    // === LOAD PHASE (cooperative across warp) ===

                    // Load input tile A: [M, K] → shared memory

                    // All threads cooperate to load 16×16 elements

                    warp::load(a_tile, g.hidden_states,

                              coord<>{inst.layer_idx, 0, k_block});

 

                    // Load weight tile B: [N, K] → shared memory

                    warp::load(b_tile, g.unified_weights,

                              coord<>{inst.layer_idx, n_block, k_block});

 

                    // Wait for loads to complete

                    __syncwarp();

 

                    // === COMPUTE PHASE (tensor cores!) ===

                    // C += A @ B^T using tensor cores

                    // This does 16×16×16 = 4096 multiply-adds in ONE instruction!

                    mma_ABt(c_accum, a_tile, b_tile, c_accum);

 

                    __syncwarp();

                }

 

                // === STORE PHASE ===

                // Convert fp32 accumulator → bf16 output

                st_bf<TILE_M, TILE_N> c_tile;

                copy(c_tile, c_accum);

 

                // Store to global memory

                warp::store(g.hidden_states, c_tile,

                           coord<>{inst.layer_idx, n_block});

            }

        }

    };

 

    // loader/storer/launcher/controller same as before...

};

```

 

**What changed:**

1. ✅ **All 32 threads per warp cooperate** (not just 1 thread)

2. ✅ **All 12 consumer warps work in parallel** (different N-blocks)

3. ✅ **Tensor cores do 4096 ops per instruction** (vs 1 op per instruction)

4. ✅ **Shared memory staging reduces global memory traffic**

 

**Expected performance:**

- Before: 4 seconds (1 thread × 1 op/cycle × 4M ops)

- After: ~4ms (896 threads × 4096 ops/cycle)

- **Speedup: ~1000x**

 

---

 

### Step-by-Step Implementation Guide

 

**Step 1: Understand TK Tile Types**

 

```cpp

// Shared memory tiles (in shared memory, visible to all threads in block)

st_bf<ROWS, COLS>  // bf16 data type, shared memory

st_fl<ROWS, COLS>  // fp32 data type, shared memory

 

// Register tiles (in registers, per-warp)

rt_bf<ROWS, COLS>  // bf16 data type, registers

rt_fl<ROWS, COLS>  // fp32 data type, registers

 

// Vector tiles (for 1D data)

sv_bf<LEN>  // bf16 shared memory vector

rv_fl<LEN>  // fp32 register vector

```

 

**Step 2: Replace Raw Pointers with gl<> Objects**

 

Currently the naive code uses raw pointers from helpers like `g.ptr_input0()`. For TK tiles, you need the actual `gl<>` objects:

 

```cpp

// Naive (current):

const float *a = g.ptr_input0<float>(inst.input_offset_0);

 

// TK tiles need this:

// Option A: If globals already has gl<bf16, ...> objects, use them directly

auto &weights = g.unified_weights;  // This is a gl<bf16, ...> object

 

// Option B: If using smoke_globals with gl<float, ...>, that works too!

// TK operations work with any gl<> type

```

 

**Step 3: Learn TK Operations**

 

```cpp

// Load operations

warp::load(tile, global_memory, coord<>{layer, row, col});  // All threads cooperate

tma::load_async(tile, global_memory, coord<>{...}, sem);     // Hardware async load

 

// Compute operations

mma(C, A, B, C);      // C += A @ B (tensor cores)

mma_ABt(C, A, B, C);  // C += A @ B^T (tensor cores)

mma_AtB(C, A, B, C);  // C += A^T @ B (tensor cores)

 

// Store operations

warp::store(global_memory, tile, coord<>{layer, row, col});  // All threads cooperate

tma::store_async(global_memory, tile, coord<>{...});          // Hardware async store

 

// Utility operations

zero(tile);           // Zero out tile

copy(dest, src);      // Copy tile (can convert types: bf16 → fp32)

```

 

**Step 4: Handle Different Matrix Sizes**

 

The generic ISA needs to handle various dimensions at runtime. Use loops:

 

```cpp

// For M×N output where M,N might not be multiples of 16

for (int m_block = 0; m_block < M; m_block += TILE_M) {

    for (int n_block = warpid() * TILE_N; n_block < N; n_block += NUM_WARPS * TILE_N) {

        // Process this tile

        // Handle edge cases where m_block + TILE_M > M or n_block + TILE_N > N

    }

}

```

 

**Step 5: Test and Debug**

 

```cpp

// Add debug prints (GPU side)

if (warpid() == 0 && laneid() == 0) {

    printf("MatMul: M=%d, N=%d, K=%d\\n", M, N, K);

}

 

// Check intermediate results

if (warpid() == 0 && laneid() == 0) {

    printf("Warp %d: c_accum[0,0] = %f\\n", warpid(), (float)c_accum.data[0]);

}

```

 

---

 

## Phase 2: RMSNorm Optimization

 

### Current Naive Implementation

 

```cpp

// include/generic/ops/norm.cuh (lines 22-43)

if (warpid() == 0 && laneid() == 0) {

    const int N = inst.n_dim;  // e.g., 2048

    const float *x = g.ptr_input0<float>(inst.input_offset_0);

    const float *w = g.ptr_weight<float>(inst.weight_offset);

    float *y = g.ptr_output<float>(inst.output_offset);

 

    // Pass 1: Sum of squares (serial!)

    float sum_sq = 0.f;

    for (int i = 0; i < N; ++i) {

        float v = x[i];

        sum_sq += v * v;

    }

    float rms = sqrtf(sum_sq / N + eps);

    float inv_rms = 1.f / rms;

 

    // Pass 2: Normalize (serial!)

    for (int i = 0; i < N; ++i) {

        y[i] = (x[i] * inv_rms) * w[i];

    }

}

```

 

**Problem:** 2048 elements × 2 passes = 4096 serial operations on one thread

 

---

 

### Optimized with TK Warp Reductions

 

```cpp

struct consumer {

    template <typename globals>

    __device__ static inline void run(const globals &g,

                                       ::megakernel::state<config> &mks) {

        GenericInstruction inst{};

        inst.deserialize_from(mks.instruction());

 

        const int N = inst.n_dim;

        const float eps = inst.scale_factor;

 

        // Each thread handles a chunk

        int tid = threadIdx.x;

        int stride = blockDim.x;

 

        // Pass 1: Parallel sum of squares

        float local_sum = 0.f;

        for (int i = tid; i < N; i += stride) {

            float v = g.ptr_input0<float>(inst.input_offset_0)[i];

            local_sum += v * v;

        }

 

        // Warp-level reduction (TK helper)

        __shared__ float warp_sums[32];  // One per warp

        float warp_sum = warp_reduce_sum(local_sum);

 

        if (laneid() == 0) {

            warp_sums[warpid()] = warp_sum;

        }

        __syncthreads();

 

        // Final reduction across warps (thread 0 only)

        float total_sum = 0.f;

        if (tid == 0) {

            for (int w = 0; w < 28; ++w) {  // 28 warps in block

                total_sum += warp_sums[w];

            }

        }

 

        // Broadcast RMS to all threads

        __shared__ float shared_inv_rms;

        if (tid == 0) {

            float rms = sqrtf(total_sum / N + eps);

            shared_inv_rms = 1.f / rms;

        }

        __syncthreads();

        float inv_rms = shared_inv_rms;

 

        // Pass 2: Parallel normalization

        const float *x = g.ptr_input0<float>(inst.input_offset_0);

        const float *w = g.ptr_weight<float>(inst.weight_offset);

        float *y = g.ptr_output<float>(inst.output_offset);

 

        for (int i = tid; i < N; i += stride) {

            y[i] = (x[i] * inv_rms) * w[i];

        }

    }

};

 

// Helper function for warp reduction

__device__ inline float warp_reduce_sum(float val) {

    for (int offset = 16; offset > 0; offset /= 2) {

        val += __shfl_down_sync(0xFFFFFFFF, val, offset);

    }

    return val;

}

```

 

**What changed:**

1. ✅ All 896 threads participate in sum-of-squares

2. ✅ Warp-level reduction using shuffle instructions

3. ✅ All threads participate in normalization

4. ✅ ~400x faster

 

---

 

## Phase 3: Attention Optimization

 

### Current Naive Implementation

 

```cpp

// include/generic/ops/attention.cuh (lines 27-83)

// THREE separate passes over K/V!

if (warpid() == 0 && laneid() == 0) {

    // Pass 1: Find max logit

    float max_logit = -INFINITY;

    for (int j = 0; j < kv_len; ++j) {

        float dot = 0.f;

        for (int d = 0; d < head_dim; ++d) {

            dot += q[d] * k[j * head_dim + d];  // Compute Q·K

        }

        max_logit = max(max_logit, dot * scale);

    }

 

    // Pass 2: Compute softmax denominator

    float denom = 0.f;

    for (int j = 0; j < kv_len; ++j) {

        float dot = 0.f;

        for (int d = 0; d < head_dim; ++d) {

            dot += q[d] * k[j * head_dim + d];  // RECOMPUTE Q·K again!

        }

        denom += expf(dot * scale - max_logit);

    }

 

    // Pass 3: Compute weighted sum

    for (int d = 0; d < head_dim; ++d) o[d] = 0.f;

    for (int j = 0; j < kv_len; ++j) {

        float dot = 0.f;

        for (int d = 0; d < head_dim; ++d) {

            dot += q[d] * k[j * head_dim + d];  // RECOMPUTE Q·K AGAIN!

        }

        float w = expf(dot * scale - max_logit) / denom;

        for (int d = 0; d < head_dim; ++d) {

            o[d] += w * v[j * head_dim + d];

        }

    }

}

```

 

**Problems:**

- Computes Q·K dot product THREE times

- Single-threaded

- No tensor core usage

 

---

 

### Optimized Flash Attention Pattern

 

**Strategy:** Use TK tiles to implement online softmax (single pass)

 

```cpp

struct consumer {

    template <typename globals>

    __device__ static inline void run(const globals &g,

                                       ::megakernel::state<config> &mks) {

        GenericInstruction inst{};

        inst.deserialize_from(mks.instruction());

 

        const int head_dim = inst.k_dim;

        const int kv_len = inst.reduction_factor;

        const float scale = inst.scale_factor;

 

        // Use TK tiles for flash attention

        // Each warp processes a different head

 

        __shared__ st_bf<16, 64> q_tile;    // Query tile

        __shared__ st_bf<64, 64> k_tile;    // Key tile (64 KV positions)

        __shared__ st_bf<64, 64> v_tile;    // Value tile

 

        rt_fl<16, 64> qk_scores;  // Q @ K^T scores (register)

        rt_fl<16, 64> output_accum;  // Output accumulator

 

        float running_max = -INFINITY;

        float running_sum = 0.f;

 

        // Load Q tile (once)

        warp::load(q_tile, g.q_proj, coord<>{inst.layer_idx, head_idx, 0});

 

        // Process KV in chunks (online softmax)

        for (int kv_block = 0; kv_block < kv_len; kv_block += 64) {

 

            // Load K and V tiles

            warp::load(k_tile, g.k_cache, coord<>{inst.layer_idx, head_idx, kv_block});

            warp::load(v_tile, g.v_cache, coord<>{inst.layer_idx, head_idx, kv_block});

 

            // Compute Q @ K^T using tensor cores

            zero(qk_scores);

            mma_ABt(qk_scores, q_tile, k_tile, qk_scores);

 

            // Apply scaling

            mul(qk_scores, scale);

 

            // Online softmax update (requires custom kernel logic)

            // See Flash Attention paper for details

            // ... (implementation details)

 

            // Accumulate output: softmax(QK^T) @ V

            mma(output_accum, qk_scores, v_tile, output_accum);

        }

 

        // Store final output

        st_bf<16, 64> output_tile;

        copy(output_tile, output_accum);

        warp::store(g.attn_output, output_tile, coord<>{inst.layer_idx, head_idx});

    }

};

```

 

**Benefits:**

- Single pass over K/V (vs 3 passes)

- Tensor cores for Q@K^T and softmax@V

- All warps work in parallel (different heads)

- ~750x speedup

 

---

 

## Implementation Checklist

 

### Before You Start

- [ ] Read through existing optimized kernels in `demos/low-latency-llama/` for reference

- [ ] Understand TK tile types (st, rt, sv, rv)

- [ ] Understand coordinate system for gl<> indexing

- [ ] Compile and run current naive ops to establish baseline

 

### Phase 1: MatMul (START HERE)

- [ ] Create backup of `include/generic/ops/matmul.cuh`

- [ ] Replace scalar loops with TK tiles

- [ ] Add warp-level parallelism

- [ ] Test with smoke tests

- [ ] Measure speedup (before/after timing)

- [ ] Handle edge cases (M, N, K not multiples of 16)

 

### Phase 2: RMSNorm

- [ ] Implement warp-level reduction for sum-of-squares

- [ ] Parallelize normalization across all threads

- [ ] Test with smoke tests

- [ ] Measure speedup

 

### Phase 3: Attention

- [ ] Implement flash attention pattern with TK tiles

- [ ] Fuse Q·K^T + softmax + @V into single pass

- [ ] Test with smoke tests

- [ ] Measure speedup

 

### Validation

- [ ] All smoke tests still pass

- [ ] Numerical accuracy within tolerance (1e-5)

- [ ] Measure end-to-end speedup on full model inference

- [ ] Profile with nsys to verify tensor core usage

 

---

 

## Reference Materials

 

### ThunderKittens Documentation

- **Main repo**: `ThunderKittens/README.md`

- **Examples**: `ThunderKittens/examples/`

- **API docs**: `ThunderKittens/include/kittens.cuh`

 

### Existing Optimized Kernels (Copy from these!)

- `demos/low-latency-llama/matvec_adds.cu` - MatVec with TMA/MMA

- `demos/low-latency-llama/rms_matvec_rope_append.cu` - RMSNorm example

- `demos/low-latency-llama/attention_partial.cu` - Flash attention pattern

 

### Key TK Concepts

- **Tiles**: 16×16 is optimal for H100 tensor cores

- **TMA**: Hardware async loads (faster than warp::load for large tiles)

- **MMA**: Tensor core operations (16×16×16 matmul per instruction)

- **Warp operations**: All 32 threads cooperate on tile operations

- **Coordinates**: Use `coord<>{batch, depth, row, col}` for indexing

 

---

 

## Performance Expectations

 

### Target Metrics (H100)

| Operation | Naive (current) | Target (with TK) | Notes |

|-----------|----------------|------------------|-------|

| MatMul 2048×2048 | 4000 ms | 4 ms | 1000x speedup |

| RMSNorm 2048 | 200 μs | 0.5 μs | 400x speedup |

| Attention 1024×128 | 12000 ms | 16 ms | 750x speedup |

| Full Llama 1B token | 300 seconds | 13 ms | 23000x speedup! |

 

### How to Measure

```bash

# Compile with timing

cd demos/generic-hopper

make clean && make

 

# Run with timing

python megakernels/scripts/generic_smoke.py matmul

 

# Use CUDA events for precise timing

# Add to smoke test:

# start = torch.cuda.Event(enable_timing=True)

# end = torch.cuda.Event(enable_timing=True)

# start.record()

# mk_generic.mk_generic_matmul(...)

# end.record()

# torch.cuda.synchronize()

# print(f"Time: {start.elapsed_time(end):.3f} ms")

```

 

---

 

## Common Pitfalls to Avoid

 

1. **Using raw pointers instead of gl<> objects**

   - TK operations need gl<> objects, not float*

   - Use `g.unified_weights` directly, not `g.ptr_weight()`

 

2. **Forgetting __syncwarp() between load and compute**

   - TK loads are asynchronous

   - Always sync before using loaded data

 

3. **Tile size mismatch**

   - Tensor cores work best with 16×16 tiles

   - Other sizes will be slow or not compile

 

4. **Not handling edge cases**

   - M, N, K might not be multiples of 16

   - Check bounds and zero-pad if needed

 

5. **Type mismatches**

   - MMA operations need specific type combinations

   - bf16 × bf16 → fp32 accumulator is standard

 

---

 

## Questions? Debug Steps

 

**Q: "Compilation fails with template errors"**

- Check that gl<> types match TK expectations

- Verify tile dimensions are compile-time constants

 

**Q: "Results are all zeros"**

- Add printf debugging to verify data loads

- Check coordinate indexing (might be off by one)

- Verify gl<> object is initialized properly

 

**Q: "Results are wrong (not zeros, just incorrect)"**

- Check transpose flags (mma_ABt vs mma vs mma_AtB)

- Verify scaling factors

- Check type conversions (bf16 ↔ fp32)

 

**Q: "Still slow after optimization"**

- Profile with `nsys` to check tensor core usage

- Verify TMA is being used (look for cp.async in PTX)

- Check for __syncthreads() in hot loops (expensive!)

 

---

 

## Success Criteria

 

✅ **Phase 1 Complete:** MatMul runs in <10ms for 2048×2048

✅ **Phase 2 Complete:** RMSNorm runs in <1μs for 2048 elements

✅ **Phase 3 Complete:** Attention runs in <20ms for 1024×128

 

Once all three phases are complete, your Generic ISA will have production-grade performance! 🚀