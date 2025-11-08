# Megakernels for the Masses - MVP Plan

> **⚠️ NOTE: This document describes an early design plan that has been superseded by the Generic ISA implementation.**
> 
> The current implementation uses:
> - **64-byte instructions** (not 128 bytes as described here)
> - **Generic ISA** (`include/generic/`) instead of MVP-specific format
> - **Runtime configuration** instead of compile-time templates
> 
> See [`GENERIC_ISA.md`](GENERIC_ISA.md) for the current architecture specification.
> 
> This document is kept for historical reference and design evolution understanding.

**Working Title**: Megakernels for the Masses  
**Goal**: Extensible instruction set VM for LLM inference  
**MVP Target**: Llama 8B forward pass on single H100  
**Status**: Superseded by Generic ISA implementation

## MVP Requirements

✅ **Must Have**:
1. Run Llama 8B forward pass on one H100
2. Basic unit tests of all instruction types
3. Written in extensible way for future model support

🎯 **Success Criteria**:
- Correctness: Matches PyTorch reference within numerical tolerance
- Performance: Within 20% of hand-tuned Llama kernels
- Extensibility: Can add new instructions/models without major refactoring

## Architecture Overview

```
Python Side:                CUDA Side:
┌─────────────────┐        ┌─────────────────┐
│ Llama Model     │        │ VM Controller   │
│ (PyTorch)       │◄──────►│ (instruction    │
└─────────────────┘        │  dispatch)      │
         │                 └─────────────────┘
         │                          │
         ▼                          ▼
┌─────────────────┐        ┌─────────────────┐
│ Scheduler       │        │ Consumer Warps  │
│ (generates      │───────►│ (execute ops)   │
│  instructions)  │        └─────────────────┘
└─────────────────┘                 │
                                    ▼
                           ┌─────────────────┐
                           │ ThunderKittens  │
                           │ (tile ops)      │
                           └─────────────────┘
```

## Phase 1: Core Instruction Format (Week 1-2)

### 1.1 Instruction Layout (128 bytes total)

Based on your spec, here's the unified format:

```cpp
struct MVPInstruction {
    // ===== Header (16 bytes) =====
    uint32_t opcode;           // Operation type
    uint32_t vm_flags;         // Bits: [0]=timing, [1]=verbose, [2-7]=reserved

    // ===== Barrier Control (24 bytes) =====
    struct BarrierWait {
        uint32_t barrier_id;
        uint8_t condition;     // 0: =, 1: <, 2: >=
        uint32_t expected_count : 24;
    } start_barrier;           // 8 bytes

    BarrierWait epilogue_barrier;  // 8 bytes

    struct BarrierWrite {
        uint32_t barrier_id;
        uint32_t value;
    } barrier_write;           // 8 bytes

    // ===== Operation-Specific Payload (88 bytes) =====
    union {
        // MATMUL instruction (64 bytes used)
        struct {
            struct MatmulInfo {
                uint16_t m, n, k;      // Dimensions
                uint8_t a_transpose;
                uint8_t b_transpose;
                uint8_t multiply_precision;   // FP32=0, BF16=1, FP16=2, E5M2=3, E4M3=4
                uint8_t accumulate_precision;
                uint16_t _padding;
            } info;                    // 16 bytes

            LoadStoreIndex a_matrix;   // 16 bytes
            LoadStoreIndex b_matrix;   // 16 bytes

            uint8_t epilogue_type;     // None=0, ReLU=1, SiLU=2, AddMat=4, MulMat=6
            uint8_t _pad[3];
            LoadStoreIndex epilogue_load;  // 16 bytes (optional, for Add/Mul)
            LoadStoreIndex output;         // 16 bytes
        } matmul;

        // RMS_NORM instruction
        struct {
            LoadStoreIndex input;      // 16 bytes
            LoadStoreIndex weight;     // 16 bytes
            LoadStoreIndex output;     // 16 bytes
            uint32_t hidden_dim;
            float epsilon;
            uint8_t _padding[56];
        } rms_norm;

        // ATTENTION instruction
        struct {
            LoadStoreIndex q, k, v;    // 48 bytes
            LoadStoreIndex output;     // 16 bytes
            uint16_t num_heads;
            uint16_t head_dim;
            uint16_t seq_len;
            uint16_t kv_seq_len;       // For decode vs prefill
            float scale;
            uint8_t mode;              // 0=prefill, 1=decode
            uint8_t _padding[11];
        } attention;

        // COPY/ZERO instruction
        struct {
            LoadStoreIndex src;        // 16 bytes (src offset for copy, unused for zero)
            LoadStoreIndex dst;        // 16 bytes
            uint32_t num_elements;
            uint8_t operation;         // 0=copy, 1=zero
            uint8_t _padding[51];
        } memory_op;

        // NOOP/BARRIER instruction
        struct {
            uint8_t _padding[88];
        } noop;

        uint8_t raw[88];
    };
};
static_assert(sizeof(MVPInstruction) == 128, "Instruction must be 128 bytes");
```

### 1.2 Load/Store Index Format (16 bytes)

```cpp
struct LoadStoreIndex {
    uint8_t tensor_id;     // Which global tensor (0-255)
    uint8_t gpu_id;        // Multi-GPU support (0-255)
    uint8_t dtype;         // FP32=0, BF16=1, FP16=2, E5M2=3, E4M3=4
    uint8_t operation;     // LOAD=0, STORE=1, ADD=2, SUBTRACT=3
    uint16_t idx0;         // Row index (units of 1)
    uint16_t idx1;         // Col index (units of 1)
    uint32_t idx2;         // Batch/layer index (units of 1)
    uint32_t idx3;         // Additional index for 4D+ tensors

    // Shape is implicit from instruction type
    // Debug mode: strict checking, Release mode: skip checks
};
```

### 1.3 Opcode Definitions

```cpp
// Core operations
constexpr uint32_t OP_NOOP           = 0x0000;
constexpr uint32_t OP_BARRIER        = 0x0001;
constexpr uint32_t OP_COPY           = 0x0002;
constexpr uint32_t OP_ZERO           = 0x0003;

// Matrix operations (0x1000-0x1FFF)
constexpr uint32_t OP_MATMUL_BASE    = 0x1000;
// Option 1: Separate opcodes per shape/precision
constexpr uint32_t OP_MATMUL_64x64_FP32_BF16   = 0x1000;
constexpr uint32_t OP_MATMUL_64x128_FP32_BF16  = 0x1001;
constexpr uint32_t OP_MATMUL_128x128_FP32_BF16 = 0x1002;
// ... or Option 2: Use matmul_info field (prefer this for extensibility)
constexpr uint32_t OP_MATMUL         = 0x1000;

// Normalization (0x2000-0x2FFF)
constexpr uint32_t OP_RMS_NORM       = 0x2000;
constexpr uint32_t OP_LAYER_NORM     = 0x2001;

// Attention (0x3000-0x3FFF)
constexpr uint32_t OP_ATTENTION_PREFILL = 0x3000;
constexpr uint32_t OP_ATTENTION_DECODE  = 0x3001;

// Activations (0x4000-0x4FFF)
constexpr uint32_t OP_RELU           = 0x4000;
constexpr uint32_t OP_SILU           = 0x4001;
constexpr uint32_t OP_GELU           = 0x4002;
```

### 1.4 Epilogue Types

```cpp
enum class EpilogueType : uint8_t {
    NONE = 0,           // Just store result
    RELU = 1,           // max(0, x)
    SILU = 2,           // x * sigmoid(x)
    ADD_VECTOR = 3,     // result + vector (bias, low priority)
    ADD_MATRIX = 4,     // result + matrix (residual)
    MUL_VECTOR = 5,     // result * vector (low priority)
    MUL_MATRIX = 6,     // result * matrix
};
```

## Phase 2: VM Modifications (Week 2-3)

### 2.1 Barrier Pre-Wait in Controller

Current: Consumer warps check barriers when executing instruction
**New**: Controller warp checks barriers in advance and marks in shared memory

```cpp
// In controller warp (before dispatching instruction)
__device__ void controller_prefetch_barriers(
    const MVPInstruction& inst,
    volatile bool* barrier_ready  // Shared memory flag
) {
    if (laneid() == 0) {
        // Check start barrier
        if (inst.start_barrier.barrier_id != 0) {
            wait_barrier_condition(
                inst.start_barrier.barrier_id,
                inst.start_barrier.condition,
                inst.start_barrier.expected_count
            );
        }

        // Mark barrier as satisfied
        *barrier_ready = true;
    }
}

// Consumer warps just check the flag (much faster!)
__device__ void consumer_check_barrier(
    volatile bool* barrier_ready
) {
    // Spin on shared memory flag (fast)
    while (!*barrier_ready) {
        __nanosleep(10);
    }
}
```

### 2.2 VM Flags

Add flags to control VM behavior without recompilation:

```cpp
struct VMFlags {
    bool enable_timing : 1;      // Record instruction timings
    bool verbose_print : 1;      // Print from each instruction
    bool strict_checks : 1;      // Debug mode: validate all indices
    bool profile_barriers : 1;   // Track barrier wait times
    uint32_t reserved : 28;
};

// Usage in instruction
if (inst.vm_flags & (1 << 0)) {
    // Record timing
}
if (inst.vm_flags & (1 << 1)) {
    printf("Executing instruction %d\n", inst.opcode);
}
```

### 2.3 Asymmetric, Colocated VMs (Future)

Design to support multiple VMs per SM:
- VM 0: Handles matmuls (high compute)
- VM 1: Handles memory ops (low compute)

```cpp
struct VMDescriptor {
    uint8_t vm_id;
    uint8_t warp_assignment;  // Which warps belong to this VM
    uint32_t instruction_queue_offset;
};

// Each SM can run multiple VMs concurrently
// For MVP: Single VM per SM, but design for extensibility
```

## Phase 3: ThunderKittens Integration (Week 3-4)

### 3.1 Global Layout Handling

**Option 1: Fork TK for Byte-Array GLs** (Recommended for flexibility)

```cpp
// New in forked TK: GL as typed byte array
template<typename MemoryLayout>
struct gl_bytes {
    void* data;
    size_t size_bytes;

    // TMA descriptors handle typing
    template<typename T, int ROWS, int COLS>
    __device__ tma_descriptor<st<T, ROWS, COLS>> get_tma_desc(size_t offset) {
        return make_tma_descriptor<st<T, ROWS, COLS>>(
            reinterpret_cast<T*>(data + offset)
        );
    }
};

// Usage in instruction
auto tma_desc = global_buffer.get_tma_desc<bf16, 64, 64>(inst.matmul.a_matrix.offset());
tma::load_async(a_tile, tma_desc, ...);
```

**Option 2: Multiple GL Arrays** (Simpler, less flexible)

```cpp
struct GlobalTensors {
    gl<bf16, -1, -1> bf16_tensors[256];
    gl<float, -1, -1> fp32_tensors[256];
    gl<half, -1, -1> fp16_tensors[256];
    // ... E5M2, E4M3, etc.
};

// Access based on tensor_id and dtype
auto& tensor = get_tensor(globals, inst.matmul.a_matrix.tensor_id, inst.matmul.a_matrix.dtype);
```

**Decision**: Start with Option 2 for MVP, design for Option 1 migration.

### 3.2 TMA Descriptor Management

Concern: Descriptor explosion in constant cache.

**Solution**: Store frequently-used descriptors in global memory, cache hot ones:

```cpp
struct TMADescriptorCache {
    // Hot descriptors in constant memory (64 max)
    __constant__ tma_descriptor hot_descriptors[64];

    // Cold descriptors in global memory
    tma_descriptor* cold_descriptors;  // Device memory

    __device__ const tma_descriptor& get(uint16_t desc_id) {
        if (desc_id < 64) return hot_descriptors[desc_id];
        return cold_descriptors[desc_id - 64];
    }
};
```

## Phase 4: Core Instructions (Week 4-6)

### 4.1 MatMul with Epilogues

```cpp
template<int M, int N, int K, typename AccumT, typename InputT>
struct matmul_op {
    __device__ static void consumer::run(
        const MVPInstruction& inst,
        const GlobalTensors& globals,
        state<config>& mks
    ) {
        // Load A and B tiles using TMA
        st<InputT, M, K> a_tile;
        st<InputT, K, N> b_tile;

        load_tile(a_tile, inst.matmul.a_matrix, globals);
        load_tile(b_tile, inst.matmul.b_matrix, globals);

        // Compute using WGMMA
        rt<AccumT, M, N> accum;
        mma::wgmma(accum, a_tile, b_tile);

        // Apply epilogue
        switch (inst.matmul.epilogue_type) {
        case EpilogueType::NONE:
            break;
        case EpilogueType::RELU:
            apply_relu(accum);
            break;
        case EpilogueType::SILU:
            apply_silu(accum);
            break;
        case EpilogueType::ADD_MATRIX:
            st<InputT, M, N> residual;
            load_tile(residual, inst.matmul.epilogue_load, globals);
            add(accum, accum, residual);
            break;
        }

        // Store result
        store_tile(accum, inst.matmul.output, globals);
    }
};
```

### 4.2 RMS Norm

```cpp
struct rms_norm_op {
    __device__ static void consumer::run(
        const MVPInstruction& inst,
        const GlobalTensors& globals,
        state<config>& mks
    ) {
        const uint32_t hidden_dim = inst.rms_norm.hidden_dim;
        const float eps = inst.rms_norm.epsilon;

        // Load input vector
        sv<bf16, hidden_dim> input;
        load_vector(input, inst.rms_norm.input, globals);

        // Compute RMS
        float sum_sq = 0.0f;
        #pragma unroll
        for (int i = 0; i < hidden_dim; i++) {
            float val = static_cast<float>(input[i]);
            sum_sq += val * val;
        }
        float rms = sqrtf(sum_sq / hidden_dim + eps);

        // Load weight and scale
        sv<bf16, hidden_dim> weight;
        load_vector(weight, inst.rms_norm.weight, globals);

        sv<bf16, hidden_dim> output;
        #pragma unroll
        for (int i = 0; i < hidden_dim; i++) {
            output[i] = static_cast<bf16>(
                static_cast<float>(input[i]) / rms * static_cast<float>(weight[i])
            );
        }

        // Store result
        store_vector(output, inst.rms_norm.output, globals);
    }
};
```

### 4.3 Attention (Prefill + Decode)

```cpp
struct attention_prefill_op {
    __device__ static void consumer::run(
        const MVPInstruction& inst,
        const GlobalTensors& globals,
        state<config>& mks
    ) {
        // Flash Attention implementation
        // Q @ K^T -> softmax -> @ V
        // Use TK's attention primitives

        const int num_heads = inst.attention.num_heads;
        const int head_dim = inst.attention.head_dim;
        const int seq_len = inst.attention.seq_len;
        const float scale = inst.attention.scale;

        // Implement flash attention tiling
        // ...
    }
};

struct attention_decode_op {
    __device__ static void consumer::run(
        const MVPInstruction& inst,
        const GlobalTensors& globals,
        state<config>& mks
    ) {
        // Single-query decode attention
        // More efficient than prefill for seq_len=1

        const int num_heads = inst.attention.num_heads;
        const int head_dim = inst.attention.head_dim;
        const int kv_seq_len = inst.attention.kv_seq_len;
        const float scale = inst.attention.scale;

        // Compute attention over KV cache
        // ...
    }
};
```

### 4.4 Memory Operations (Copy/Zero)

```cpp
struct memory_op {
    __device__ static void consumer::run(
        const MVPInstruction& inst,
        const GlobalTensors& globals,
        state<config>& mks
    ) {
        const uint32_t num_elements = inst.memory_op.num_elements;

        if (inst.memory_op.operation == 0) {
            // Copy
            copy_async(
                get_ptr(inst.memory_op.dst, globals),
                get_ptr(inst.memory_op.src, globals),
                num_elements
            );
        } else if (inst.memory_op.operation == 1) {
            // Zero
            zero_async(
                get_ptr(inst.memory_op.dst, globals),
                num_elements
            );
        }
    }
};
```

## Phase 5: Llama 8B Implementation (Week 6-7)

### 5.1 Model Architecture

```python
# Llama 8B configuration
config = {
    'num_layers': 32,
    'hidden_dim': 4096,
    'intermediate_dim': 14336,
    'num_heads': 32,
    'num_kv_heads': 8,  # GQA
    'head_dim': 128,
    'vocab_size': 128256,
}
```

### 5.2 Instruction Sequence for One Layer

```python
def build_llama_layer(layer_idx: int) -> List[MVPInstruction]:
    instructions = []

    # 1. RMS Norm (pre-attention)
    instructions.append(MVPInstruction(
        opcode=OP_RMS_NORM,
        start_barrier=wait_previous_layer(layer_idx),
        rms_norm={
            'input': hidden_states,
            'weight': attn_norm_weights[layer_idx],
            'output': normed_hidden,
            'hidden_dim': 4096,
            'epsilon': 1e-5,
        }
    ))

    # 2. QKV Projection (3 matmuls can be fused into 1)
    instructions.append(MVPInstruction(
        opcode=OP_MATMUL,
        matmul={
            'info': {'m': 1, 'n': 6144, 'k': 4096},  # Q(4096) + K(1024) + V(1024)
            'a_matrix': normed_hidden,
            'b_matrix': qkv_weights[layer_idx],
            'output': qkv_proj,
        }
    ))

    # 3. RoPE + Attention
    instructions.append(MVPInstruction(
        opcode=OP_ATTENTION_DECODE,
        attention={
            'q': qkv_proj[:4096],
            'k': kv_cache_k[layer_idx],
            'v': kv_cache_v[layer_idx],
            'output': attn_out,
            'num_heads': 32,
            'head_dim': 128,
            'kv_seq_len': seq_pos + 1,
        }
    ))

    # 4. O Projection + Residual
    instructions.append(MVPInstruction(
        opcode=OP_MATMUL,
        matmul={
            'info': {'m': 1, 'n': 4096, 'k': 4096},
            'a_matrix': attn_out,
            'b_matrix': o_proj_weights[layer_idx],
            'epilogue_type': EpilogueType.ADD_MATRIX,
            'epilogue_load': hidden_states,  # Residual
            'output': hidden_states,
        }
    ))

    # 5. RMS Norm (pre-MLP)
    instructions.append(MVPInstruction(
        opcode=OP_RMS_NORM,
        rms_norm={
            'input': hidden_states,
            'weight': mlp_norm_weights[layer_idx],
            'output': normed_hidden,
            'hidden_dim': 4096,
        }
    ))

    # 6. Gate + Up projections
    instructions.append(MVPInstruction(
        opcode=OP_MATMUL,
        matmul={
            'info': {'m': 1, 'n': 14336, 'k': 4096},
            'a_matrix': normed_hidden,
            'b_matrix': gate_weights[layer_idx],
            'epilogue_type': EpilogueType.SILU,
            'output': gate_out,
        }
    ))

    instructions.append(MVPInstruction(
        opcode=OP_MATMUL,
        matmul={
            'info': {'m': 1, 'n': 14336, 'k': 4096},
            'a_matrix': normed_hidden,
            'b_matrix': up_weights[layer_idx],
            'epilogue_type': EpilogueType.MUL_MATRIX,
            'epilogue_load': gate_out,
            'output': mlp_intermediate,
        }
    ))

    # 7. Down projection + Residual
    instructions.append(MVPInstruction(
        opcode=OP_MATMUL,
        matmul={
            'info': {'m': 1, 'n': 4096, 'k': 14336},
            'a_matrix': mlp_intermediate,
            'b_matrix': down_weights[layer_idx],
            'epilogue_type': EpilogueType.ADD_MATRIX,
            'epilogue_load': hidden_states,
            'output': hidden_states,
        }
    ))

    # Barrier: Signal layer complete
    instructions.append(MVPInstruction(
        opcode=OP_BARRIER,
        barrier_write={'barrier_id': layer_idx, 'value': 1}
    ))

    return instructions
```

### 5.3 Full Forward Pass

```python
def build_llama_forward_pass() -> List[MVPInstruction]:
    instructions = []

    # Embedding lookup (handled by PyTorch for MVP)

    # Process all 32 layers
    for layer_idx in range(32):
        instructions.extend(build_llama_layer(layer_idx))

    # Final RMS norm + LM head
    instructions.append(MVPInstruction(
        opcode=OP_RMS_NORM,
        rms_norm={
            'input': hidden_states,
            'weight': final_norm_weights,
            'output': normed_final,
        }
    ))

    instructions.append(MVPInstruction(
        opcode=OP_MATMUL,
        matmul={
            'info': {'m': 1, 'n': 128256, 'k': 4096},
            'a_matrix': normed_final,
            'b_matrix': lm_head_weights,
            'output': logits,
        }
    ))

    return instructions
```

## Phase 6: Testing & Validation (Week 7-8)

### 6.1 Unit Tests

```python
def test_matmul_epilogues():
    """Test all matmul epilogue types"""
    for epilogue in [None, ReLU, SiLU, AddMatrix, MulMatrix]:
        result_mk = run_matmul_instruction(epilogue)
        result_torch = torch_matmul_with_epilogue(epilogue)
        assert torch.allclose(result_mk, result_torch, rtol=1e-3)

def test_rms_norm():
    """Test RMS normalization"""
    result_mk = run_rms_norm_instruction()
    result_torch = torch_rms_norm()
    assert torch.allclose(result_mk, result_torch, rtol=1e-4)

def test_attention_decode():
    """Test single-query attention"""
    result_mk = run_attention_decode_instruction()
    result_torch = torch_attention()
    assert torch.allclose(result_mk, result_torch, rtol=1e-3)

def test_barrier_conditions():
    """Test barrier wait conditions (=, <, >=)"""
    for condition in ['eq', 'lt', 'gte']:
        test_barrier_condition(condition)
```

### 6.2 Llama 8B Validation

```python
def test_llama_forward_pass():
    """End-to-end test"""
    # Load Llama 8B weights
    model_torch = load_llama_8b_pytorch()
    model_mk = load_llama_8b_megakernel()

    # Test on 100 random inputs
    for i in range(100):
        input_ids = torch.randint(0, 128256, (1, 1))

        logits_torch = model_torch(input_ids)
        logits_mk = model_mk(input_ids)

        # Check numerical accuracy
        assert torch.allclose(logits_torch, logits_mk, rtol=1e-2, atol=1e-3)

        # Check argmax agrees (same next token)
        assert logits_torch.argmax() == logits_mk.argmax()
```

### 6.3 Performance Benchmarking

```python
def benchmark_llama_8b():
    """Measure performance"""
    # Warmup
    for _ in range(10):
        model_mk(input_ids)

    # Measure
    times = []
    for _ in range(100):
        start = time.perf_counter()
        model_mk(input_ids)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    mean_time = np.mean(times)
    std_time = np.std(times)

    print(f"Llama 8B decode: {mean_time*1000:.2f}ms ± {std_time*1000:.2f}ms")
    print(f"Tokens/sec: {1.0/mean_time:.1f}")
```

## Implementation Timeline

| Week | Phase | Deliverables |
|------|-------|--------------|
| 1-2  | Instruction Format | MVPInstruction struct, opcodes, builder helpers |
| 2-3  | VM Modifications | Barrier pre-wait, VM flags, controller updates |
| 3-4  | TK Integration | Global layout handling, TMA descriptors |
| 4-5  | Core Instructions | MatMul, RMSNorm, Attention, Memory ops |
| 5-6  | Llama 8B | Full instruction sequence, weight loading |
| 6-7  | Testing | Unit tests for all ops, integration tests |
| 7-8  | Validation | Correctness vs PyTorch, performance tuning |

## Key Design Decisions

1. **Instruction Size**: 128 bytes (vs 64 generic)
   - Trade-off: Larger instructions, but simpler dispatch
   - Benefit: Explicit barrier control, full matmul info

2. **Barrier Pre-Wait**: Controller handles barriers
   - Benefit: Consumer warps don't stall
   - Trade-off: More controller complexity

3. **Epilogues in MatMul**: Reduce kernel launches
   - Benefit: ~20% speedup from fusion
   - Trade-off: More matmul kernel variants

4. **TK Byte Arrays**: Defer to Phase 3
   - Start: Multiple GL arrays per type
   - Future: Migrate to byte-array GLs with TMA typing

5. **MVP Scope**: Llama 8B only
   - Benefit: Ship faster, validate design
   - Future: Extend to other models using same infrastructure

## Extensibility Path

After MVP, easy to add:
- **New models**: Same instruction set, different sequences
- **New instructions**: Add opcode + implementation
- **Multi-GPU**: Already have gpu_id in LoadStoreIndex
- **Quantization**: Add E5M2, E4M3 paths (dtypes already defined)
- **Prefill**: Add OP_ATTENTION_PREFILL variant

## Success Metrics

- ✅ **Correctness**: Llama 8B output matches PyTorch within 1e-2 relative error
- ✅ **Performance**: Within 20% of specialized Llama megakernel
- ✅ **Extensibility**: Can add new instruction in <1 day
- ✅ **Tests**: >90% code coverage on instruction execution

## Files to Create

```
include/mvp/
├── instruction.cuh       # MVPInstruction struct, opcodes
├── load_store.cuh        # LoadStoreIndex, tensor access
├── barriers.cuh          # Barrier wait/write primitives
├── epilogues.cuh         # Epilogue implementations
└── ops/
    ├── matmul.cuh       # MatMul with epilogues
    ├── rms_norm.cuh     # RMS normalization
    ├── attention.cuh    # Prefill + decode attention
    └── memory.cuh       # Copy/zero operations

megakernels/mvp/
├── instruction_builder.py   # Python instruction builders
├── llama_scheduler.py        # Llama instruction generation
└── tests/
    ├── test_instructions.py
    └── test_llama_8b.py

demos/mvp-llama/
├── Makefile
├── llama_kernel.cu
└── test_llama.py
```

This plan is **concrete, implementable, and scoped for MVP** while maintaining extensibility for future work.
