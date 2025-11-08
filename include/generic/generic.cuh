#pragma once

// ============================================================================
// Generic Megakernel Instruction Set for Hopper (and Blackwell)
// ============================================================================
//
// This header provides a model-agnostic instruction set that works across
// different transformer architectures (Llama, GPT, Mistral, etc.) while
// maintaining high performance on NVIDIA Hopper and Blackwell GPUs.
//
// Key Features:
// - Runtime model configuration (no recompilation needed)
// - Flexible attention patterns (MHA, GQA, MQA, MLA)
// - Support for different normalization types (RMSNorm, LayerNorm)
// - Support for different activation functions (SiLU, GELU, etc.)
// - Portable across H100 and B200 GPUs
//
// Usage:
//   1. Define your model configuration using RuntimeModelConfig
//   2. Create instructions using the builder helpers
//   3. Schedule instructions across SMs
//   4. Execute using the megakernel framework
//

#include "opcodes.cuh"
#include "model_config.cuh"
#include "instruction.cuh"
#include "globals.cuh"
#include "worker_types.cuh"

// Generic op implementations (scaffold)
#include "ops/matmul.cuh"
#include "ops/norm.cuh"
#include "ops/barrier.cuh"
#include "ops/memory.cuh"
// Optional ops for smoke testing
#include "ops/attention.cuh"
#include "ops/rope.cuh"
#include "ops/fused.cuh"

namespace megakernel {
namespace generic {

// ============================================================================
// Version Information
// ============================================================================

constexpr int GENERIC_ISA_VERSION_MAJOR = 0;
constexpr int GENERIC_ISA_VERSION_MINOR = 1;
constexpr int GENERIC_ISA_VERSION_PATCH = 0;

// ============================================================================
// Operation Dispatcher Interface
// ============================================================================
// Each operation implements this interface for the different worker types

template <typename config, typename op>
struct GenericOperationDispatcher {
    // Consumer warp implementation (compute)
    __device__ static void consumer_run(
        const RuntimeGlobals<config> &g,
        ::megakernel::state<config> &mks,
        const GenericInstruction &inst
    ) {
        // Default: no-op
    }

    // Loader warp implementation (async memory loads)
    __device__ static void loader_run(
        const RuntimeGlobals<config> &g,
        ::megakernel::state<config> &mks,
        const GenericInstruction &inst
    ) {
        // Default: no-op
    }

    // Storer warp implementation (async memory stores)
    __device__ static void storer_run(
        const RuntimeGlobals<config> &g,
        ::megakernel::state<config> &mks,
        const GenericInstruction &inst
    ) {
        // Default: no-op
    }

    // Launcher warp implementation (kernel launches)
    __device__ static void launcher_run(
        const RuntimeGlobals<config> &g,
        ::megakernel::state<config> &mks,
        const GenericInstruction &inst
    ) {
        // Default: no-op
    }

    // Cost estimation (for scheduling)
    __host__ __device__ static uint64_t estimate_cost(
        const RuntimeModelConfig &model_cfg,
        const GenericInstruction &inst
    ) {
        return inst.work_elements();
    }
};

// ============================================================================
// Runtime Instruction Decoder
// ============================================================================
// Decodes generic instructions from the instruction buffer

__device__ inline GenericInstruction decode_instruction(
    const int instruction_buffer[32]
) {
    GenericInstruction inst;
    inst.deserialize_from(instruction_buffer);
    return inst;
}

// ============================================================================
// Main Dispatch Function
// ============================================================================
// Routes opcodes to appropriate operation implementations

template <typename config, typename worker_type>
__device__ inline void dispatch_generic_instruction(
    const RuntimeGlobals<config> &g,
    ::megakernel::state<config> &mks,
    const GenericInstruction &inst
);

// This will be specialized for each operation in separate files
// See: generic/ops/*.cuh

// --------------------------------------------------------------------------
// Default worker-specific dispatcher implementation
// --------------------------------------------------------------------------

template <typename config>
__device__ inline void dispatch_generic_instruction_worker(
    const RuntimeGlobals<config> &g,
    ::megakernel::state<config> &mks,
    const GenericInstruction &inst,
    worker_consumer
) {
    switch (inst.opcode) {
    case OP_MATMUL:
        OpMatmul<config>::consumer::run(g, mks);
        break;
    case OP_RMS_NORM:
        OpRmsNorm<config>::consumer::run(g, mks);
        break;
    case OP_LAYER_NORM:
        OpLayerNorm<config>::consumer::run(g, mks);
        break;
    case OP_ATTENTION_PARTIAL:
        OpAttentionPartial<config>::consumer::run(g, mks);
        break;
    case OP_ROPE_EMBED:
        OpRopeEmbed<config>::consumer::run(g, mks);
        break;
    case OP_FUSED_NORM_MATMUL:
        OpFusedNormMatmul<config>::consumer::run(g, mks);
        break;
    case OP_FUSED_NORM_QKV_ROPE:
        OpFusedNormQkvRope<config>::consumer::run(g, mks);
        break;
    default:
        // Unknown opcode for consumer; no-op
        break;
    }
}

template <typename config>
__device__ inline void dispatch_generic_instruction_worker(
    const RuntimeGlobals<config> &g,
    ::megakernel::state<config> &mks,
    const GenericInstruction &inst,
    worker_loader
) {
    switch (inst.opcode) {
    case OP_MATMUL:
        OpMatmul<config>::loader::run(g, mks);
        break;
    case OP_RMS_NORM:
        OpRmsNorm<config>::loader::run(g, mks);
        break;
    case OP_LAYER_NORM:
        OpLayerNorm<config>::loader::run(g, mks);
        break;
    case OP_ATTENTION_PARTIAL:
        // no-op loader in smoke version
        break;
    case OP_ROPE_EMBED:
        // no-op loader in smoke version
        break;
    case OP_FUSED_NORM_MATMUL:
        // no-op loader in smoke version
        break;
    case OP_FUSED_NORM_QKV_ROPE:
        // no-op loader in smoke version
        break;
    default:
        break;
    }
}

template <typename config>
__device__ inline void dispatch_generic_instruction_worker(
    const RuntimeGlobals<config> &g,
    ::megakernel::state<config> &mks,
    const GenericInstruction &inst,
    worker_storer
) {
    switch (inst.opcode) {
    case OP_MATMUL:
        OpMatmul<config>::storer::run(g, mks);
        break;
    case OP_RMS_NORM:
        OpRmsNorm<config>::storer::run(g, mks);
        break;
    case OP_LAYER_NORM:
        OpLayerNorm<config>::storer::run(g, mks);
        break;
    case OP_ATTENTION_PARTIAL:
        // no-op storer in smoke version
        break;
    case OP_ROPE_EMBED:
        // no-op storer in smoke version
        break;
    case OP_FUSED_NORM_MATMUL:
        // no-op storer in smoke version
        break;
    case OP_FUSED_NORM_QKV_ROPE:
        // no-op storer in smoke version
        break;
    default:
        break;
    }
}

template <typename config>
__device__ inline void dispatch_generic_instruction_worker(
    const RuntimeGlobals<config> &g,
    ::megakernel::state<config> &mks,
    const GenericInstruction &inst,
    worker_launcher
) {
    switch (inst.opcode) {
    case OP_MATMUL:
        OpMatmul<config>::launcher::run(g, mks);
        break;
    case OP_RMS_NORM:
        OpRmsNorm<config>::launcher::run(g, mks);
        break;
    case OP_LAYER_NORM:
        OpLayerNorm<config>::launcher::run(g, mks);
        break;
    case OP_ATTENTION_PARTIAL:
        // no-op
        break;
    case OP_ROPE_EMBED:
        // no-op
        break;
    case OP_FUSED_NORM_MATMUL:
        // no-op
        break;
    case OP_FUSED_NORM_QKV_ROPE:
        // no-op
        break;
    default:
        break;
    }
}

template <typename config, typename worker_type>
__device__ inline void dispatch_generic_instruction(
    const RuntimeGlobals<config> &g,
    ::megakernel::state<config> &mks,
    const GenericInstruction &inst
) {
    // Route to the worker-specific dispatcher using tag dispatching
    dispatch_generic_instruction_worker<config>(g, mks, inst, worker_type{});
}

// ============================================================================
// Performance Hints
// ============================================================================

struct PerformanceHints {
    // Suggest block sizes based on model dimensions
    static __host__ inline void suggest_tiling(
        RuntimeModelConfig &cfg,
        bool is_blackwell = false
    ) {
        // Heuristic: larger tiles for larger models
        if (cfg.hidden_dim >= 4096) {
            cfg.matmul_block_m = 32;
            cfg.matmul_block_n = 32;
            cfg.matmul_block_k = 512;
        } else if (cfg.hidden_dim >= 2048) {
            cfg.matmul_block_m = 16;
            cfg.matmul_block_n = 16;
            cfg.matmul_block_k = 512;
        } else {
            cfg.matmul_block_m = 16;
            cfg.matmul_block_n = 16;
            cfg.matmul_block_k = 256;
        }

        // Blackwell has more shared memory and faster tensor cores
        if (is_blackwell) {
            cfg.matmul_block_k *= 2;
            cfg.attn_block_kv = 128;  // vs 64 on Hopper
        }

        // KV cache blocking
        cfg.kv_block_size = 16;  // Good for both Hopper and Blackwell
        cfg.attn_block_q = cfg.matmul_block_m;
    }

    // Suggest SM count based on GPU type
    static __host__ inline void detect_hardware(RuntimeModelConfig &cfg) {
        int device;
        cudaGetDevice(&device);

        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);

        cfg.sm_count = prop.multiProcessorCount;
        cfg.shared_mem_size = prop.sharedMemPerBlock;

        // Detect architecture
        if (prop.major == 9 && prop.minor == 0) {
            cfg.is_hopper = true;
            cfg.is_blackwell = false;
            cfg.has_fp8 = true;
        } else if (prop.major == 10 && prop.minor == 0) {
            cfg.is_hopper = false;
            cfg.is_blackwell = true;
            cfg.has_fp8 = true;
        } else {
            // Ampere or earlier
            cfg.is_hopper = false;
            cfg.is_blackwell = false;
            cfg.has_fp8 = (prop.major >= 9);  // Ada has FP8
        }
    }
};

// ============================================================================
// Debugging Utilities
// ============================================================================

__device__ inline void print_instruction(const GenericInstruction &inst) {
    printf("Instruction ID %u:\n", inst.instruction_id);
    printf("  Opcode: 0x%02x, Layer: %u\n", inst.opcode, inst.layer_idx);
    printf("  Dims: %u x %u x %u\n", inst.m_dim, inst.n_dim, inst.k_dim);
    printf("  Inputs: [%u, %u, %u], Output: %u\n",
           inst.input_offset_0, inst.input_offset_1,
           inst.input_offset_2, inst.output_offset);
    printf("  Weight offset: %u, Scale: %.4f\n",
           inst.weight_offset, inst.scale_factor);
}

__host__ inline const char *opcode_name(uint8_t opcode) {
    switch (opcode) {
    case OP_NOOP: return "NOOP";
    case OP_MATMUL: return "MATMUL";
    case OP_MATVEC: return "MATVEC";
    case OP_RMS_NORM: return "RMS_NORM";
    case OP_LAYER_NORM: return "LAYER_NORM";
    case OP_ATTENTION_PARTIAL: return "ATTENTION_PARTIAL";
    case OP_ATTENTION_REDUCE: return "ATTENTION_REDUCE";
    case OP_ROPE_EMBED: return "ROPE_EMBED";
    case OP_SILU: return "SILU";
    case OP_GELU: return "GELU";
    case OP_SWIGLU: return "SWIGLU";
    case OP_RESIDUAL_ADD: return "RESIDUAL_ADD";
    case OP_FUSED_NORM_MATMUL: return "FUSED_NORM_MATMUL";
    case OP_FUSED_ROPE_APPEND: return "FUSED_ROPE_APPEND";
    default: return "UNKNOWN";
    }
}

} // namespace generic
} // namespace megakernel
