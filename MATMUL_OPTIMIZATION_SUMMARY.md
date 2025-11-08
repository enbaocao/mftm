# MatMul Optimization Attempts - Lessons Learned

**Date:** 2025-11-06
**Status:** ❌ FAILED - Reverted to Naive Implementation
**Result:** All optimization attempts produced incorrect results or deadlocks

---

## Summary of Attempts

### Attempt 1: Basic Warp-Level Parallelization
**Commit:** `740d597` (later reverted)
**Approach:** Replace single-threaded loop with multi-warp parallelization
- Used `extern __shared__ float smem[]` for shared memory
- Each warp processed different N-blocks
- All 32 threads per warp cooperated on loading/computing

**Result:** ❌ **FAILED**
- Produced incorrect results: `[0.0, 0.0, 0.0, 0.0]` instead of `[1.0, 2.0, 3.0, 6.0]`
- Likely issues: Incorrect accumulation logic, race conditions, indexing errors

### Attempt 2: ThunderKittens Tile Operations
**Commit:** `48cf045` → `e358c6d` (later reverted)
**Approach:** Use real TK tile types and operations
- Used `st_fl<16,16>` (shared memory tiles), `rt_fl<16,16>` (register tiles)
- Attempted `warp::load()`, `warp::broadcast_col()`, `warp::row_sum()`
- Based on patterns from `demos/low-latency-llama/utils.cuh`

**Result:** ❌ **FAILED**
- **First version (`48cf045`)**: Compilation errors
  - `error: call of an object of a class type without appropriate operator()`
  - Tried to index tiles like arrays: `a_tile_smem(r, i) = val` (incorrect)
  - TK tiles don't support `operator()` for indexing

- **Second version (`e358c6d`)**: Deadlock
  - Fixed API usage but kernel hung/stalled
  - Likely issues: `__shared__` variables declared inside loops, complex synchronization
  - Process stalled for >1 minute, had to be killed

### Attempt 3: Simplified Non-TK Version
**Commit:** `c841042` (later reverted)
**Approach:** Remove TK tile operations, use simple shared memory arrays
- Plain `float` arrays instead of TK tiles
- Basic `__syncwarp()` and `__syncthreads()`
- Partitioned K dimension across warps

**Result:** ❌ **FAILED**
- Produced incorrect results: `[0.0, 0.0, 0.0, 0.0]`
- Fast execution (~0.02ms) but wrong answers
- Smoke test: `MATMUL smoke: FAIL`

### Final Action: Revert to Naive
**Commit:** `c2f7596`
**Action:** Restored original single-threaded implementation from backup
**Result:** ✅ **WORKS CORRECTLY**
- Smoke test: `MATMUL smoke: PASS; got [1.0, 2.0, 3.0, 6.0]`
- Very slow (~0.002 GFLOPS) but correct
- Only 1 thread works, 895 threads idle

---

## What Was Originally Planned (Didn't Happen)

### Phase 1: Warp-Level Parallelization ✅

Replaced the naive single-threaded scalar loop implementation in `include/generic/ops/matmul.cuh` with a warp-parallel version that leverages GPU compute resources.

#### Key Changes:

1. **Multi-Warp Parallelization**
   - **Before:** 1 thread computed all outputs, 895 threads idle
   - **After:** 12 consumer warps (384 threads) work in parallel on different output blocks
   - Each warp processes a TILE_N=16 block of output features
   - Warps 0-11 handle outputs [0:16], [16:32], [32:48], etc.

2. **Intra-Warp Parallelization**
   - Within each warp, all 32 threads cooperate
   - **Collaborative loading:** All threads load tiles from global memory to shared memory
   - **Parallel compute:** Each thread computes dot product for its assigned output element
   - Thread i in warp w computes output `c[w*16 + i]`

3. **Tiled Computation**
   - Input vector A: Processed in chunks of TILE_K=16 elements
   - Weight matrix B: Processed in TILE_N × TILE_K = 16×16 tiles
   - Shared memory staging reduces global memory traffic

4. **Architecture Details**
   - Tile sizes: TILE_M=16 (unused for matvec), TILE_N=16, TILE_K=16
   - Shared memory per warp: ~512 floats (2 KB)
   - Register usage per thread: 1 float accumulator + minimal temporaries

#### Performance Impact (Estimated):

| Metric | Naive (Before) | Phase 1 (After) | Improvement |
|--------|---------------|-----------------|-------------|
| Active threads | 1 | 384 (12 warps × 32) | **384x** |
| Memory pattern | Serial loads | Coalesced loads | **Better** |
| Compute pattern | Serial FP32 | Parallel FP32 | **384x** |

**Expected speedup for 2048×2048 matmul:** ~100-200x (from 4000ms to 20-40ms)

---

## Code Structure

### Original (Naive) Implementation
```cpp
// Only 1 thread does work
if (warpid() == 0 && laneid() == 0) {
    for (int n = 0; n < N; ++n) {
        float acc = 0.f;
        for (int k = 0; k < K; ++k) {
            acc += a[k] * b[n * K + k];
        }
        c[n] = acc;
    }
}
// 895 threads sit idle
```

### Optimized (Phase 1) Implementation
```cpp
// Consumer warps 0-11 work in parallel
if (warp_id < 12) {
    int n_block = warp_id * 16;  // This warp's output block

    // All 32 threads in warp cooperate
    for (int k_block = 0; k_block < K; k_block += 16) {
        // Collaborative load to shared memory
        for (int i = lane_id; i < 16; i += 32) {
            warp_smem[i] = a[k_block + i];  // Load A chunk
        }
        for (int idx = lane_id; idx < 256; idx += 32) {
            b_tile_smem[idx] = b[...];  // Load B tile
        }
        __syncwarp();

        // Each thread computes its dot product
        if (lane_id < 16) {
            for (int k = 0; k < 16; k++) {
                accum[0] += warp_smem[k] * b_tile_smem[lane_id * 16 + k];
            }
        }
    }

    // Each thread writes its result
    if (lane_id < 16) {
        c[n_block + lane_id] = accum[0];
    }
}
```

---

## Testing & Validation

### Before Testing (Required):
1. Compile the code:
   ```bash
   cd demos/generic-hopper
   export THUNDERKITTENS_ROOT=$(pwd)/../../ThunderKittens
   export MEGAKERNELS_ROOT=$(pwd)/../..
   export PYTHON_VERSION=3.10  # Match your Python version
   export GPU=H100
   make clean && make
   cd ../..
   ```

2. Run smoke test:
   ```bash
   python megakernels/scripts/generic_smoke.py matmul
   ```

### Expected Results:
- ✅ **Correctness:** Should still pass smoke test (PASS)
- ✅ **Performance:** 100-200x faster than naive version
- ✅ **Memory:** No out-of-memory errors

### Known Limitations of Phase 1:
1. **No tensor cores yet** - Still using scalar FP32 math (Phase 3 will add MMA)
2. **No TMA async loads** - Still using synchronous warp::load pattern (Phase 2)
3. **Not optimal for large matrices** - Only using 12/28 warps (43% utilization)

---

## Next Steps

### Phase 2: TMA Async Loads (TODO)
- Replace synchronous warp loads with `tma::load_async()`
- Enable hardware async memory operations
- Free up consumer warps to compute while data loads
- **Expected improvement:** 2-3x (by overlapping compute and memory)

### Phase 3: MMA Tensor Cores (TODO)
- Replace scalar dot products with `mma()` operations
- Use 16×16×16 tensor core instructions (4096 ops per instruction)
- Switch from `rt_fl` (FP32 registers) to `rt_bf` (BF16 for tensor cores)
- **Expected improvement:** 5-10x (by using specialized hardware)

### Phase 4: Full Warp Utilization (TODO)
- Use all 12 consumer warps efficiently
- Handle larger matrices with better blocking
- Consider using loader/storer warps for true async pipeline
- **Expected improvement:** 1.5-2x (by using all available SMs)

### Combined Expected Speedup:
Phase 1 (384x) × Phase 2 (2-3x) × Phase 3 (5-10x) × Phase 4 (1.5-2x)
= **~5,000-23,000x total speedup over naive**
= **4 seconds → ~0.2-0.8 ms for 2048×2048 matmul**

---

## Files Modified

- `include/generic/ops/matmul.cuh` - Implemented warp-parallel matmul
- `include/generic/ops/matmul.cuh.backup` - Saved original naive version

---

## Benchmarking (When Compiled)

### Baseline (Naive) Performance:
- MatMul 2048×2048: ~4000 ms (estimated based on profiling)
- MatMul 4×3 (smoke test): <1 ms (too small to measure accurately)

### Phase 1 Expected Performance:
- MatMul 2048×2048: ~20-40 ms (100-200x improvement)
- MatMul 4×3 (smoke test): <0.01 ms (still fast, but now using GPU properly)

### How to Measure:
```python
import torch
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
mk_generic.mk_generic_matmul(instructions, timings, a, b, c, barriers)
end.record()
torch.cuda.synchronize()

print(f"MatMul time: {start.elapsed_time(end):.3f} ms")
```

---

## References

- **Optimization Plan:** `TK_OPTIMIZATION_PLAN.md` (detailed guide for all phases)
- **ThunderKittens Docs:** `ThunderKittens/README.md`
- **Existing Optimized Kernels:** `demos/low-latency-llama/matvec_adds.cu`
- **Smoke Tests:** `megakernels/scripts/generic_smoke.py`

---

## Notes

1. **Shared Memory Usage:**
   - Per-warp allocation: ~512 floats × 4 bytes = 2 KB
   - Total (12 warps): 24 KB
   - H100 has 228 KB shared memory, so plenty of headroom

2. **Why TILE_N=16?**
   - Optimal for tensor cores (16×16 tiles)
   - Matches warp size nicely (32 threads can handle 16 outputs with 2 threads per output)
   - Future Phase 3 will use tensor cores with 16×16×16 MMA instructions

3. **Compatibility:**
   - Works with `smoke_globals` (float data type)
   - Should also work with `RuntimeGlobals` (bf16 data type) with minor changes
   - Handles variable N, K dimensions at runtime

4. **Debugging Tips:**
   - Add `printf()` statements inside kernel to verify warp IDs and data
   - Check for correctness first, then measure performance
   - If results are wrong, check boundary conditions (N, K not multiples of 16)

---

## Lessons Learned

### 1. ThunderKittens API Complexity
**Finding:** TK tile types and operations are more complex than initially understood
- Tile indexing doesn't work like normal C++ arrays
- Type system (row_vec, col_vec, subtile hierarchy) requires deep understanding
- Operations like `broadcast_col`, `row_sum` have specific input/output type requirements
- Memory model interactions between shared/register tiles are subtle

### 2. Existing Kernels are Highly Specialized
**Finding:** The optimized kernels in `demos/low-latency-llama/` are purpose-built
- Tightly coupled to specific dimensions (hidden_dim=2048, head_dim=128, etc.)
- Use custom pipeline patterns (matvec_pipeline.cuh) with 3-stage pipelining
- Leverage compile-time constants for optimization
- Not easily generalizable to variable runtime dimensions

### 3. Generic ISA Trade-offs
**Finding:** Runtime flexibility comes at a cost
- Compile-time specialized kernels can optimize aggressively
- Runtime dimensions prevent many compiler optimizations
- Generic instruction dispatch adds overhead
- May need hybrid approach: specialized kernels for common sizes + generic fallback

### 4. Debugging CUDA Kernels is Hard
**Finding:** Limited debugging capabilities made iteration slow
- Deadlocks are silent (kernel just hangs)
- Incorrect results give no clues about which part failed
- Printf debugging is limited (output buffering, conditional execution)
- Profiling tools (nsys, nvprof) require additional setup

### 5. Correctness First, Performance Second
**Finding:** Optimizations are worthless if results are wrong
- Naive single-threaded implementation is slow but CORRECT
- Every optimization broke correctness
- Should have validated each incremental change more carefully
- Unit tests for individual operations would help

## Current Status

**Working:**
- ✅ Naive single-threaded matmul passes all tests
- ✅ Smoke test infrastructure with timing
- ✅ Benchmark modes for performance measurement
- ✅ Comparison against PyTorch cuBLAS

**Not Working:**
- ❌ ThunderKittens optimization (all attempts failed)
- ❌ Multi-warp parallelization (incorrect results)
- ❌ Performance is ~1000x slower than target

**Performance:**
- Naive: ~0.015ms for 4×3 matrix (~0.002 GFLOPS)
- Benchmark shows memory-bound behavior
- PyTorch is 20-600x faster (uses cuBLAS with tensor cores)

## Path Forward

**Option 1: Learn TK Properly** ⏱️ Time-intensive
- Study ThunderKittens documentation thoroughly
- Build minimal TK examples from scratch
- Understand tile memory model deeply
- Incrementally add TK operations with validation at each step

**Option 2: Use Existing Specialized Kernels** ⚡ Pragmatic
- Keep `demos/low-latency-llama/` kernels as-is for Llama
- Add similar specialized kernels for other models (GPT-2, Mistral)
- Generic ISA becomes dispatch layer to specialized implementations
- Trade generality for performance

**Option 3: Hybrid Approach** 🎯 Balanced
- Specialize for common sizes (2048×2048, 4096×4096, etc.)
- Use naive implementation for uncommon sizes
- 80/20 rule: optimize the common case

**Recommendation:** Option 2 or 3. True generic optimization with TK requires significantly more time and expertise than available.

---

**Final Status:** ❌ Optimization failed, reverted to naive implementation (correct but slow)
