#pragma once

namespace megakernel {
namespace generic {

// ============================================================================
// Core Opcode Definitions
// ============================================================================
// Design: 8-bit opcode space divided into categories for easy decoding

// Category 0x00-0x0F: No-ops and control flow
constexpr uint8_t OP_NOOP = 0x00;
constexpr uint8_t OP_BARRIER = 0x01;
constexpr uint8_t OP_SYNC = 0x02;

// Category 0x10-0x2F: Memory operations
constexpr uint8_t OP_LOAD_ACTIVATION = 0x10;
constexpr uint8_t OP_STORE_ACTIVATION = 0x11;
constexpr uint8_t OP_LOAD_WEIGHT_TILE = 0x12;
constexpr uint8_t OP_CACHE_READ = 0x13;
constexpr uint8_t OP_CACHE_APPEND = 0x14;
constexpr uint8_t OP_COPY = 0x15;           // Memory copy operation
constexpr uint8_t OP_ZERO = 0x16;           // Zero memory operation

// Category 0x30-0x4F: Linear algebra operations
constexpr uint8_t OP_MATMUL = 0x30;
constexpr uint8_t OP_MATVEC = 0x31;
constexpr uint8_t OP_OUTER_PRODUCT = 0x32;
constexpr uint8_t OP_BATCH_MATMUL = 0x33;

// Category 0x50-0x6F: Normalization operations
constexpr uint8_t OP_RMS_NORM = 0x50;
constexpr uint8_t OP_LAYER_NORM = 0x51;
constexpr uint8_t OP_GROUP_NORM = 0x52;

// Category 0x70-0x8F: Attention operations
constexpr uint8_t OP_ATTENTION_PARTIAL = 0x70;
constexpr uint8_t OP_ATTENTION_REDUCE = 0x71;
constexpr uint8_t OP_ROPE_EMBED = 0x72;
constexpr uint8_t OP_ALIBI_EMBED = 0x73;
constexpr uint8_t OP_SLIDING_WINDOW_MASK = 0x74;

// Category 0x90-0x9F: Activation functions
constexpr uint8_t OP_SILU = 0x90;
constexpr uint8_t OP_GELU = 0x91;
constexpr uint8_t OP_RELU = 0x92;
constexpr uint8_t OP_SWIGLU = 0x93;  // Fused gate * SiLU(up)
constexpr uint8_t OP_GEGLU = 0x94;   // Fused gate * GELU(up)

// Category 0xA0-0xAF: Element-wise operations
constexpr uint8_t OP_ADD = 0xA0;
constexpr uint8_t OP_MUL = 0xA1;
constexpr uint8_t OP_RESIDUAL_ADD = 0xA2;
constexpr uint8_t OP_SCALE = 0xA3;
constexpr uint8_t OP_ELEMWISE_MUL_ADD = 0xA4;

// Category 0xB0-0xCF: Fused operations (for performance)
constexpr uint8_t OP_FUSED_NORM_MATMUL = 0xB0;
constexpr uint8_t OP_FUSED_ROPE_APPEND = 0xB1;
constexpr uint8_t OP_FUSED_GATE_ACT = 0xB2;
constexpr uint8_t OP_FUSED_QKV_PROJ = 0xB3;
constexpr uint8_t OP_FUSED_MATMUL_RESIDUAL = 0xB4;
constexpr uint8_t OP_FUSED_NORM_QKV_ROPE = 0xB5;  // Full attention prep

// Category 0xD0-0xEF: Model-specific operations
constexpr uint8_t OP_MLA_COMPRESS = 0xD0;      // Multi-latent attention compression
constexpr uint8_t OP_MLA_DECOMPRESS = 0xD1;
constexpr uint8_t OP_ROTARY_PARTIAL = 0xD2;     // Partial RoPE (e.g., Qwen)
constexpr uint8_t OP_CROSS_ATTENTION = 0xD3;

// Category 0xF0-0xFF: Reserved for future extensions
constexpr uint8_t OP_CUSTOM_0 = 0xF0;
constexpr uint8_t OP_CUSTOM_1 = 0xF1;

// ============================================================================
// Instruction Flags (8-bit field for operation modifiers)
// ============================================================================

// Bit 0: Transpose
constexpr uint8_t FLAG_TRANSPOSE_A = 0x01;
constexpr uint8_t FLAG_TRANSPOSE_B = 0x02;

// Bit 2-3: Accumulation mode
constexpr uint8_t FLAG_ACCUMULATE = 0x04;      // Add to existing output
constexpr uint8_t FLAG_RESIDUAL = 0x08;         // Add residual connection

// Bit 4-5: Data type override (normally inferred from config)
constexpr uint8_t FLAG_FP32_ACCUM = 0x10;      // Use FP32 accumulation
constexpr uint8_t FLAG_FP8_INPUT = 0x20;       // Input is FP8

// Bit 6-7: Sync/dependency hints
constexpr uint8_t FLAG_WAIT_PREV = 0x40;       // Wait for previous op in layer
constexpr uint8_t FLAG_BROADCAST = 0x80;       // Broadcast result to all SMs

// ============================================================================
// Attention Configuration Encoding (16-bit field)
// ============================================================================

// Bits 0-3: Attention type
constexpr uint16_t ATTN_TYPE_MASK = 0x000F;
constexpr uint16_t ATTN_TYPE_MHA = 0x0000;     // Multi-head attention
constexpr uint16_t ATTN_TYPE_GQA = 0x0001;     // Grouped query attention
constexpr uint16_t ATTN_TYPE_MQA = 0x0002;     // Multi-query attention (1 KV head)
constexpr uint16_t ATTN_TYPE_MLA = 0x0003;     // Multi-latent attention

// Bits 4-7: Position embedding type
constexpr uint16_t POS_EMB_MASK = 0x00F0;
constexpr uint16_t POS_EMB_NONE = 0x0000;
constexpr uint16_t POS_EMB_ROPE = 0x0010;
constexpr uint16_t POS_EMB_ALIBI = 0x0020;
constexpr uint16_t POS_EMB_LEARNED = 0x0030;
constexpr uint16_t POS_EMB_ROPE_PARTIAL = 0x0040;  // Partial RoPE (e.g., Qwen)

// Bits 8-11: Masking strategy
constexpr uint16_t MASK_TYPE_MASK = 0x0F00;
constexpr uint16_t MASK_CAUSAL = 0x0100;
constexpr uint16_t MASK_SLIDING_WINDOW = 0x0200;
constexpr uint16_t MASK_BLOCK_SPARSE = 0x0300;

// Bits 12-15: Special features
constexpr uint16_t ATTN_FEAT_MASK = 0xF000;
constexpr uint16_t ATTN_FEAT_FLASH = 0x1000;      // Flash attention style
constexpr uint16_t ATTN_FEAT_PAGED = 0x2000;      // Paged KV cache
constexpr uint16_t ATTN_FEAT_SOFTMAX_SCALE = 0x4000;  // Custom softmax scaling

// Helper functions for attention config
__device__ __host__ inline uint16_t make_attn_config(
    uint16_t attn_type, uint16_t pos_emb, uint16_t mask_type, uint16_t features = 0
) {
    return attn_type | pos_emb | mask_type | features;
}

__device__ __host__ inline uint16_t get_attn_type(uint16_t config) {
    return config & ATTN_TYPE_MASK;
}

__device__ __host__ inline uint16_t get_pos_emb_type(uint16_t config) {
    return config & POS_EMB_MASK;
}

__device__ __host__ inline uint16_t get_mask_type(uint16_t config) {
    return config & MASK_TYPE_MASK;
}

// ============================================================================
// Activation Function Encoding
// ============================================================================

enum class ActivationType : uint8_t {
    NONE = 0,
    RELU = 1,
    GELU = 2,
    SILU = 3,
    SWIGLU = 4,  // gate * SiLU(up)
    GEGLU = 5,   // gate * GELU(up)
    TANH = 6,
    SIGMOID = 7
};

// ============================================================================
// Normalization Type Encoding
// ============================================================================

enum class NormType : uint8_t {
    NONE = 0,
    RMS_NORM = 1,
    LAYER_NORM = 2,
    GROUP_NORM = 3,
    RMS_NORM_NO_WEIGHT = 4  // RMS without learnable scale
};

} // namespace generic
} // namespace megakernel
