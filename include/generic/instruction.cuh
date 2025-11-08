#pragma once

#include "opcodes.cuh"
#include "model_config.cuh"
#include <cstdint>

namespace megakernel {
namespace generic {

// ============================================================================
// Generic Instruction Format
// ============================================================================
// Designed to fit within 128 bytes (32 ints) for efficient packing

struct alignas(16) GenericInstruction {
    // ========== Core Operation (4 bytes) ==========
    uint8_t opcode;           // Operation type (see opcodes.cuh)
    uint8_t flags;            // Operation modifiers (transpose, accumulate, etc.)
    uint16_t layer_idx;       // Which transformer layer (for weight indexing)

    // ========== Dimensions (12 bytes) ==========
    // Flexible interpretation based on opcode:
    // - MatMul: m_dim x k_dim @ k_dim x n_dim
    // - Norm: operates on m_dim x n_dim
    // - Attention: m_dim = num_heads, n_dim = seq_len, k_dim = head_dim
    uint16_t m_dim;
    uint16_t n_dim;
    uint16_t k_dim;

    // Block indices for tiling (allows operations to be split across SMs)
    uint16_t block_idx_m;
    uint16_t block_idx_n;
    uint16_t block_idx_k;

    // ========== Memory Addressing (24 bytes) ==========
    // All offsets are in elements (not bytes) from buffer base
    uint32_t input_offset_0;   // Primary input (e.g., hidden states)
    uint32_t input_offset_1;   // Secondary input (e.g., residual, or K for attention)
    uint32_t input_offset_2;   // Tertiary input (e.g., V for attention)
    uint32_t output_offset;    // Where to write result
    uint32_t weight_offset;    // Offset into weight buffer
    uint32_t scratch_offset;   // Offset into scratch/temp buffer

    // ========== Configuration (8 bytes) ==========
    uint16_t head_config;      // Attention pattern encoding (ATTN_TYPE_*, etc.)
    uint16_t reduction_factor; // For tree reductions, blocking, etc.
    uint16_t seq_pos;          // Current sequence position (for RoPE, caching)
    uint16_t batch_idx;        // Batch index (for throughput mode)

    // ========== Synchronization (8 bytes) ==========
    uint16_t dependency_mask;  // Bitmask: which prior instructions must complete
    uint8_t sync_slot;         // Which barrier/semaphore to use
    uint8_t sync_count;        // How many arrivals expected at barrier
    uint32_t parent_instr_id;  // For tree reductions: which instruction to reduce with

    // ========== Metadata (8 bytes) ==========
    uint32_t instruction_id;   // Unique ID for debugging and dependency tracking
    float scale_factor;        // For scaling operations, attention scale, etc.

    // ========== Padding to 64 bytes ==========
    uint8_t _padding[64 - 4 - 12 - 24 - 8 - 8 - 8];

    // ========== Helper Methods ==========

    __device__ __host__ inline bool is_memory_op() const {
        return opcode >= 0x10 && opcode < 0x30;
    }

    __device__ __host__ inline bool is_compute_op() const {
        return opcode >= 0x30;
    }

    __device__ __host__ inline bool is_fused_op() const {
        return opcode >= 0xB0 && opcode < 0xD0;
    }

    __device__ __host__ inline bool has_transpose_a() const {
        return flags & FLAG_TRANSPOSE_A;
    }

    __device__ __host__ inline bool has_transpose_b() const {
        return flags & FLAG_TRANSPOSE_B;
    }

    __device__ __host__ inline bool should_accumulate() const {
        return flags & FLAG_ACCUMULATE;
    }

    __device__ __host__ inline bool has_residual() const {
        return flags & FLAG_RESIDUAL;
    }

    __device__ __host__ inline bool wait_for_prev() const {
        return flags & FLAG_WAIT_PREV;
    }

    __device__ __host__ inline uint16_t get_attn_type() const {
        return generic::get_attn_type(head_config);
    }

    __device__ __host__ inline bool is_gqa() const {
        return get_attn_type() == ATTN_TYPE_GQA;
    }

    __device__ __host__ inline bool is_mqa() const {
        return get_attn_type() == ATTN_TYPE_MQA;
    }

    // Calculate how many elements this instruction operates on (for cost estimation)
    __device__ __host__ inline uint64_t work_elements() const {
        switch (opcode) {
        case OP_MATMUL:
        case OP_BATCH_MATMUL:
            // M x N output, K inner dimension
            return (uint64_t)m_dim * n_dim * k_dim * 2;  // 2 FLOPs per multiply-add
        case OP_MATVEC:
            return (uint64_t)m_dim * k_dim * 2;
        case OP_RMS_NORM:
        case OP_LAYER_NORM:
            return (uint64_t)m_dim * n_dim * 3;  // reduction + scale + shift
        case OP_ATTENTION_PARTIAL:
            // Q @ K^T + softmax + @ V
            return (uint64_t)m_dim * n_dim * k_dim * 4;
        case OP_SILU:
        case OP_GELU:
        case OP_RELU:
            return (uint64_t)m_dim * n_dim;
        default:
            return (uint64_t)m_dim * n_dim;
        }
    }

    // Serialize to int array (for compatibility with existing megakernel infrastructure)
    __host__ inline void serialize_to(int *buffer) const {
        const uint32_t *src = reinterpret_cast<const uint32_t *>(this);
        for (int i = 0; i < 16; i++) {  // 64 bytes / 4 = 16 words
            buffer[i] = src[i];
        }
        // Pad remaining space
        for (int i = 16; i < 32; i++) {
            buffer[i] = 0;
        }
    }

    __device__ inline void deserialize_from(const int *buffer) {
        uint32_t *dst = reinterpret_cast<uint32_t *>(this);
        for (int i = 0; i < 16; i++) {
            dst[i] = buffer[i];
        }
    }
};

static_assert(sizeof(GenericInstruction) == 64,
              "GenericInstruction must be exactly 64 bytes");

// ============================================================================
// Instruction Builder Helpers
// ============================================================================

// Build a matrix multiply instruction
__host__ inline GenericInstruction make_matmul_instruction(
    uint16_t layer_idx,
    uint16_t m, uint16_t n, uint16_t k,
    uint32_t input_offset, uint32_t weight_offset, uint32_t output_offset,
    bool transpose_b = false, bool accumulate = false
) {
    GenericInstruction inst = {};
    inst.opcode = OP_MATMUL;
    inst.flags = 0;
    if (transpose_b) inst.flags |= FLAG_TRANSPOSE_B;
    if (accumulate) inst.flags |= FLAG_ACCUMULATE;

    inst.layer_idx = layer_idx;
    inst.m_dim = m;
    inst.n_dim = n;
    inst.k_dim = k;

    inst.input_offset_0 = input_offset;
    inst.weight_offset = weight_offset;
    inst.output_offset = output_offset;

    return inst;
}

// Build a normalization instruction
__host__ inline GenericInstruction make_norm_instruction(
    NormType norm_type,
    uint16_t layer_idx,
    uint16_t size,
    uint32_t input_offset, uint32_t weight_offset, uint32_t output_offset,
    float eps = 1e-5f
) {
    GenericInstruction inst = {};
    inst.opcode = (norm_type == NormType::RMS_NORM) ? OP_RMS_NORM : OP_LAYER_NORM;
    inst.layer_idx = layer_idx;
    inst.m_dim = 1;
    inst.n_dim = size;

    inst.input_offset_0 = input_offset;
    inst.weight_offset = weight_offset;
    inst.output_offset = output_offset;
    inst.scale_factor = eps;

    return inst;
}

// Build an attention partial instruction
__host__ inline GenericInstruction make_attention_partial_instruction(
    uint16_t layer_idx,
    uint16_t num_heads, uint16_t head_dim, uint16_t seq_len,
    uint16_t attn_config,
    uint32_t q_offset, uint32_t k_offset, uint32_t v_offset,
    uint32_t output_offset,
    uint16_t kv_block_idx = 0,
    uint16_t num_kv_blocks = 1,
    float attn_scale = 0.0f
) {
    GenericInstruction inst = {};
    inst.opcode = OP_ATTENTION_PARTIAL;
    inst.layer_idx = layer_idx;

    inst.m_dim = num_heads;
    inst.n_dim = seq_len;
    inst.k_dim = head_dim;

    inst.head_config = attn_config;

    inst.input_offset_0 = q_offset;
    inst.input_offset_1 = k_offset;
    inst.input_offset_2 = v_offset;
    inst.output_offset = output_offset;

    inst.block_idx_k = kv_block_idx;
    inst.reduction_factor = num_kv_blocks;

    inst.scale_factor = (attn_scale != 0.0f) ? attn_scale : (1.0f / sqrtf(head_dim));

    return inst;
}

// Build a fused operation instruction
__host__ inline GenericInstruction make_fused_norm_matmul_instruction(
    NormType norm_type,
    uint16_t layer_idx,
    uint16_t m, uint16_t n, uint16_t k,
    uint32_t input_offset, uint32_t norm_weight_offset,
    uint32_t matmul_weight_offset, uint32_t output_offset,
    float norm_eps = 1e-5f
) {
    GenericInstruction inst = {};
    inst.opcode = OP_FUSED_NORM_MATMUL;
    inst.layer_idx = layer_idx;

    inst.m_dim = m;
    inst.n_dim = n;
    inst.k_dim = k;

    inst.input_offset_0 = input_offset;
    inst.weight_offset = matmul_weight_offset;
    inst.input_offset_1 = norm_weight_offset;  // Norm weights in secondary slot
    inst.output_offset = output_offset;

    inst.scale_factor = norm_eps;

    return inst;
}

// Build residual add instruction
__host__ inline GenericInstruction make_residual_add_instruction(
    uint16_t size,
    uint32_t input_offset, uint32_t residual_offset, uint32_t output_offset
) {
    GenericInstruction inst = {};
    inst.opcode = OP_RESIDUAL_ADD;

    inst.m_dim = 1;
    inst.n_dim = size;

    inst.input_offset_0 = input_offset;
    inst.input_offset_1 = residual_offset;
    inst.output_offset = output_offset;

    return inst;
}

} // namespace generic
} // namespace megakernel
