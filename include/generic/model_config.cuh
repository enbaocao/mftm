#pragma once

#include "opcodes.cuh"
#include <cstdint>

namespace megakernel {
namespace generic {

// ============================================================================
// Runtime Model Configuration
// ============================================================================
// Replaces compile-time template parameters with runtime configuration
// This allows a single compiled kernel to work across different models

struct RuntimeModelConfig {
    // ========== Architecture Parameters ==========
    uint16_t num_layers;
    uint16_t hidden_dim;
    uint16_t intermediate_dim;
    uint32_t vocab_size;

    // ========== Attention Configuration ==========
    uint16_t num_q_heads;
    uint16_t num_kv_heads;  // For GQA/MQA: num_q_heads >= num_kv_heads
    uint16_t head_dim;
    uint16_t max_seq_len;

    // Attention pattern encoding
    uint16_t attn_config;   // Uses ATTN_TYPE_*, POS_EMB_*, MASK_TYPE_* flags

    // Sliding window attention (Mistral)
    uint32_t sliding_window_size;  // 0 = no sliding window

    // RoPE configuration
    float rope_theta;       // Base frequency (10000.0 for standard RoPE)
    float rope_scaling;     // Scaling factor for extended context
    uint16_t rope_partial_factor;  // For partial RoPE (Qwen), 0 = full RoPE

    // Attention scaling
    float attn_scale;       // Usually 1/sqrt(head_dim), can be custom

    // ========== MLP Configuration ==========
    ActivationType mlp_activation;
    bool mlp_gated;         // True for SwiGLU/GeGLU style MLPs

    // ========== Normalization ==========
    NormType norm_type;
    float norm_eps;         // Epsilon for numerical stability

    // ========== Tiling Parameters (Performance Tuning) ==========
    // These control how operations are blocked for optimal memory hierarchy usage

    // Matrix multiplication tiles
    uint16_t matmul_block_m;
    uint16_t matmul_block_n;
    uint16_t matmul_block_k;

    // KV cache tiling
    uint16_t kv_block_size;      // Tokens per KV cache block
    uint16_t kv_page_size;       // For paged attention

    // Attention tiling
    uint16_t attn_block_q;       // Query block size
    uint16_t attn_block_kv;      // KV block size for flash attention

    // Batch processing
    uint16_t max_batch_size;
    uint16_t batch_block_size;   // For throughput mode

    // ========== Hardware Configuration ==========
    uint16_t sm_count;           // Number of SMs on target GPU
    uint16_t warp_count;         // Warps per block
    uint32_t shared_mem_size;    // Available shared memory per block

    // GPU generation flags
    bool is_hopper;              // H100
    bool is_blackwell;           // B200
    bool has_fp8;                // FP8 tensor core support

    // ========== Memory Layout ==========
    // Offsets into unified weight buffer (in elements)
    uint64_t qkv_weight_offset;
    uint64_t o_proj_weight_offset;
    uint64_t gate_proj_weight_offset;
    uint64_t up_proj_weight_offset;
    uint64_t down_proj_weight_offset;
    uint64_t embed_weight_offset;
    uint64_t lm_head_weight_offset;

    // Offsets into unified norm weight buffer
    uint64_t attn_norm_offset;
    uint64_t mlp_norm_offset;
    uint64_t final_norm_offset;

    // ========== Helper Methods ==========

    __device__ __host__ inline uint16_t qkv_dim() const {
        return num_q_heads * head_dim + 2 * num_kv_heads * head_dim;
    }

    __device__ __host__ inline uint16_t q_dim() const {
        return num_q_heads * head_dim;
    }

    __device__ __host__ inline uint16_t kv_dim() const {
        return num_kv_heads * head_dim;
    }

    __device__ __host__ inline uint16_t num_kv_groups() const {
        return num_q_heads / num_kv_heads;
    }

    __device__ __host__ inline bool is_mha() const {
        return (attn_config & ATTN_TYPE_MASK) == ATTN_TYPE_MHA;
    }

    __device__ __host__ inline bool is_gqa() const {
        return (attn_config & ATTN_TYPE_MASK) == ATTN_TYPE_GQA;
    }

    __device__ __host__ inline bool is_mqa() const {
        return (attn_config & ATTN_TYPE_MASK) == ATTN_TYPE_MQA;
    }

    __device__ __host__ inline bool has_rope() const {
        return (attn_config & POS_EMB_MASK) == POS_EMB_ROPE ||
               (attn_config & POS_EMB_MASK) == POS_EMB_ROPE_PARTIAL;
    }

    __device__ __host__ inline bool has_sliding_window() const {
        return (attn_config & MASK_TYPE_MASK) == MASK_SLIDING_WINDOW;
    }

    __device__ __host__ inline uint32_t get_sliding_window() const {
        return has_sliding_window() ? sliding_window_size : max_seq_len;
    }

    __device__ __host__ inline float get_attn_scale() const {
        return attn_scale;
    }

    // Calculate total weight memory needed
    __device__ __host__ inline uint64_t total_weight_elements() const {
        uint64_t per_layer =
            qkv_dim() * hidden_dim +           // QKV projection
            q_dim() * hidden_dim +             // O projection
            intermediate_dim * hidden_dim +    // Gate projection
            intermediate_dim * hidden_dim +    // Up projection
            hidden_dim * intermediate_dim;     // Down projection

        uint64_t total = per_layer * num_layers +
                        vocab_size * hidden_dim +   // Embedding
                        vocab_size * hidden_dim;    // LM head

        return total;
    }

    // Calculate KV cache size
    __device__ __host__ inline uint64_t kv_cache_elements_per_layer() const {
        return 2 * num_kv_heads * head_dim * max_seq_len;  // K + V
    }
};

// ============================================================================
// Model Configuration Presets
// ============================================================================
// Factory functions for common model architectures

__host__ inline RuntimeModelConfig make_llama_3_1b_config() {
    RuntimeModelConfig cfg = {};

    // Llama 3.2 1B architecture
    cfg.num_layers = 16;
    cfg.hidden_dim = 2048;
    cfg.intermediate_dim = 8192;
    cfg.vocab_size = 128256;

    cfg.num_q_heads = 32;
    cfg.num_kv_heads = 8;  // GQA with 4 groups
    cfg.head_dim = 64;
    cfg.max_seq_len = 8192;

    cfg.attn_config = make_attn_config(
        ATTN_TYPE_GQA,
        POS_EMB_ROPE,
        MASK_CAUSAL,
        ATTN_FEAT_FLASH
    );

    cfg.sliding_window_size = 0;
    cfg.rope_theta = 500000.0f;  // Llama 3 uses higher theta
    cfg.rope_scaling = 1.0f;
    cfg.rope_partial_factor = 0;

    cfg.attn_scale = 1.0f / sqrtf(64.0f);

    cfg.mlp_activation = ActivationType::SWIGLU;
    cfg.mlp_gated = true;

    cfg.norm_type = NormType::RMS_NORM;
    cfg.norm_eps = 1e-5f;

    // Tiling parameters (tuned for H100)
    cfg.matmul_block_m = 16;
    cfg.matmul_block_n = 16;
    cfg.matmul_block_k = 512;
    cfg.kv_block_size = 16;
    cfg.attn_block_q = 16;
    cfg.attn_block_kv = 64;

    cfg.max_batch_size = 1;
    cfg.batch_block_size = 1;

    // H100 hardware
    cfg.sm_count = 132;
    cfg.warp_count = 20;
    cfg.shared_mem_size = 228 * 1024;
    cfg.is_hopper = true;
    cfg.is_blackwell = false;
    cfg.has_fp8 = true;

    return cfg;
}

__host__ inline RuntimeModelConfig make_gpt2_config() {
    RuntimeModelConfig cfg = {};

    // GPT-2 (124M) architecture
    cfg.num_layers = 12;
    cfg.hidden_dim = 768;
    cfg.intermediate_dim = 3072;
    cfg.vocab_size = 50257;

    cfg.num_q_heads = 12;
    cfg.num_kv_heads = 12;  // MHA
    cfg.head_dim = 64;
    cfg.max_seq_len = 1024;

    cfg.attn_config = make_attn_config(
        ATTN_TYPE_MHA,
        POS_EMB_LEARNED,
        MASK_CAUSAL,
        ATTN_FEAT_FLASH
    );

    cfg.attn_scale = 1.0f / sqrtf(64.0f);

    cfg.mlp_activation = ActivationType::GELU;
    cfg.mlp_gated = false;

    cfg.norm_type = NormType::LAYER_NORM;
    cfg.norm_eps = 1e-5f;

    // Tiling
    cfg.matmul_block_m = 16;
    cfg.matmul_block_n = 16;
    cfg.matmul_block_k = 256;
    cfg.kv_block_size = 16;

    return cfg;
}

__host__ inline RuntimeModelConfig make_mistral_7b_config() {
    RuntimeModelConfig cfg = {};

    // Mistral 7B architecture
    cfg.num_layers = 32;
    cfg.hidden_dim = 4096;
    cfg.intermediate_dim = 14336;
    cfg.vocab_size = 32000;

    cfg.num_q_heads = 32;
    cfg.num_kv_heads = 8;  // GQA
    cfg.head_dim = 128;
    cfg.max_seq_len = 32768;

    cfg.attn_config = make_attn_config(
        ATTN_TYPE_GQA,
        POS_EMB_ROPE,
        MASK_SLIDING_WINDOW,
        ATTN_FEAT_FLASH
    );

    cfg.sliding_window_size = 4096;  // Sliding window attention
    cfg.rope_theta = 10000.0f;
    cfg.rope_scaling = 1.0f;

    cfg.attn_scale = 1.0f / sqrtf(128.0f);

    cfg.mlp_activation = ActivationType::SWIGLU;
    cfg.mlp_gated = true;

    cfg.norm_type = NormType::RMS_NORM;
    cfg.norm_eps = 1e-5f;

    return cfg;
}

} // namespace generic
} // namespace megakernel
