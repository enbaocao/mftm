#pragma once

#include "kittens.cuh"
#include "instruction.cuh"
#include "model_config.cuh"
#include "../config.cuh"

namespace megakernel {
namespace generic {

// ============================================================================
// Runtime Globals Structure
// ============================================================================
// Unlike the compile-time templated globals_t, this structure uses
// runtime configuration and generic buffers

template <typename config>
struct RuntimeGlobals {
    // ========== Runtime Model Configuration ==========
    RuntimeModelConfig model_cfg;

    // ========== Instruction Stream ==========
    using instruction_layout = megakernel::instruction_layout<config>;
    using timing_layout = megakernel::timing_layout<config>;

    instruction_layout instructions;
    timing_layout timings;

    // ========== Unified Weight Buffers ==========
    // All weights stored in contiguous buffers, indexed by layer
    // Layout: [layer_0_weights, layer_1_weights, ..., layer_N_weights]

    // Shape: (num_layers, various_out_dims, hidden_dim)
    // Uses generic pointer to support different dimensions per layer
    kittens::gl<kittens::bf16, -1, -1, -1, -1> unified_weights;

    // Normalization weights
    // Shape: (num_layers * num_norms_per_layer, hidden_dim)
    kittens::gl<kittens::bf16, 1, 1, -1, -1> norm_weights;

    // Embedding and LM head
    kittens::gl<kittens::bf16, 1, 1, -1, -1> embed_weights;
    kittens::gl<kittens::bf16, 1, 1, -1, -1> lm_head_weights;

    // ========== KV Cache ==========
    // Generic KV cache supporting variable head counts
    // Shape: (num_layers, 2, num_kv_heads, max_seq_len, head_dim)
    //        where 2 = K and V
    // Flattened view for KV cache (multi-dim folded into depth/rows/cols)
    kittens::gl<kittens::bf16, -1, -1, -1, -1> kv_cache;

    // ========== Position Embeddings ==========
    // RoPE frequency tables
    kittens::gl<float, 1, 1, -1, -1> rope_cos;
    kittens::gl<float, 1, 1, -1, -1> rope_sin;

    // Learned position embeddings (for GPT-2 style models)
    kittens::gl<kittens::bf16, 1, 1, -1, -1> pos_embed_table;

    // ALiBi slopes (for models using ALiBi positional bias)
    kittens::gl<float, 1, 1, 1, -1> alibi_slopes;

    // ========== Activation Buffers ==========
    // Generic activation buffers supporting variable dimensions
    kittens::gl<kittens::bf16, 1, 1, -1, -1> hidden_states;
    kittens::gl<kittens::bf16, 1, 1, -1, -1> residual_states;
    kittens::gl<kittens::bf16, 1, 1, -1, -1> norm_output;

    // Attention buffers
    kittens::gl<kittens::bf16, 1, -1, -1, -1> q_proj;  // (batch, num_heads, seq_len, head_dim)
    kittens::gl<kittens::bf16, 1, -1, -1, -1> k_proj;
    kittens::gl<kittens::bf16, 1, -1, -1, -1> v_proj;
    kittens::gl<kittens::bf16, 1, -1, -1, -1> attn_output;

    // For split attention (flash attention style)
    kittens::gl<float, 1, -1, -1, -1> attn_lse_intermediates;  // Log-sum-exp for softmax
    kittens::gl<float, 1, -1, -1, -1> attn_out_intermediates;

    // MLP buffers
    kittens::gl<kittens::bf16, 1, 1, -1, -1> gate_output;
    kittens::gl<kittens::bf16, 1, 1, -1, -1> up_output;
    kittens::gl<kittens::bf16, 1, 1, -1, -1> mlp_output;

    // LM head output
    kittens::gl<kittens::bf16, 1, 1, -1, -1> logits;

    // ========== Synchronization ==========
    // Dynamic barriers for inter-SM synchronization
    // Shape: (num_layers, max_ops_per_layer, max_barrier_slots)
    kittens::gl<uint, 1, -1, -1, -1> barriers;

    // ========== Scratch Space ==========
    // Temporary buffer for intermediate computations
    kittens::gl<kittens::bf16, 1, 1, 1, -1> scratch_buffer;

    // ========== Runtime State ==========
    uint32_t current_seq_pos;     // Current position in sequence
    uint32_t current_batch_size;  // Active batch size
    uint32_t max_context_len;     // Maximum context length in current batch

    // ========== Helper Methods ==========

    __device__ __host__ inline uint32_t sm_count() const {
        return model_cfg.sm_count;
    }

    __device__ __host__ inline uint32_t hidden_dim() const {
        return model_cfg.hidden_dim;
    }

    __device__ __host__ inline uint32_t num_layers() const {
        return model_cfg.num_layers;
    }

    // Generic pointer helpers for ops (element offsets)
    template <typename T = kittens::bf16>
    __device__ __host__ inline T *ptr_input0(uint32_t offset_elems) const {
        return reinterpret_cast<T *>(hidden_states.raw_ptr) + offset_elems;
    }

    template <typename T = kittens::bf16>
    __device__ __host__ inline const T *ptr_input0(uint32_t offset_elems) {
        return reinterpret_cast<const T *>(hidden_states.raw_ptr) + offset_elems;
    }

    // Secondary and tertiary inputs (by default, also alias hidden_states)
    template <typename T = kittens::bf16>
    __device__ __host__ inline const T *ptr_input1(uint32_t offset_elems) const {
        return reinterpret_cast<const T *>(hidden_states.raw_ptr) + offset_elems;
    }
    template <typename T = kittens::bf16>
    __device__ __host__ inline const T *ptr_input2(uint32_t offset_elems) const {
        return reinterpret_cast<const T *>(hidden_states.raw_ptr) + offset_elems;
    }

    template <typename T = kittens::bf16>
    __device__ __host__ inline T *ptr_weight(uint32_t offset_elems) const {
        return reinterpret_cast<T *>(unified_weights.raw_ptr) + offset_elems;
    }

    template <typename T = kittens::bf16>
    __device__ __host__ inline T *ptr_output(uint32_t offset_elems) const {
        return reinterpret_cast<T *>(hidden_states.raw_ptr) + offset_elems;
    }

    // Get weight pointer for a specific layer and weight type
    __device__ __host__ inline kittens::bf16 *get_layer_weight(
        uint16_t layer_idx, uint64_t offset_in_layer, uint32_t size
    ) {
        // Calculate offset: sum of all previous layer weights + offset in current layer
        uint64_t total_offset = 0;
        // This would need proper calculation based on model architecture
        // For now, simplified
        return unified_weights.raw_ptr + total_offset + offset_in_layer;
    }

    // Get norm weight for specific layer and norm type
    __device__ __host__ inline kittens::bf16 *get_norm_weight(
        uint16_t layer_idx, uint8_t norm_idx
    ) {
        uint32_t offset = layer_idx * 2 + norm_idx;  // 2 norms per layer (attn, mlp)
        return norm_weights.raw_ptr + offset * model_cfg.hidden_dim;
    }

    // Get KV cache pointer for specific layer, head, and position
    __device__ __host__ inline kittens::bf16 *get_k_cache(
        uint16_t layer_idx, uint16_t head_idx, uint32_t seq_pos
    ) {
        uint64_t offset =
            layer_idx * 2 * model_cfg.num_kv_heads * model_cfg.max_seq_len * model_cfg.head_dim +
            0 * model_cfg.num_kv_heads * model_cfg.max_seq_len * model_cfg.head_dim +  // K cache (0)
            head_idx * model_cfg.max_seq_len * model_cfg.head_dim +
            seq_pos * model_cfg.head_dim;
        return kv_cache.raw_ptr + offset;
    }

    __device__ __host__ inline kittens::bf16 *get_v_cache(
        uint16_t layer_idx, uint16_t head_idx, uint32_t seq_pos
    ) {
        uint64_t offset =
            layer_idx * 2 * model_cfg.num_kv_heads * model_cfg.max_seq_len * model_cfg.head_dim +
            1 * model_cfg.num_kv_heads * model_cfg.max_seq_len * model_cfg.head_dim +  // V cache (1)
            head_idx * model_cfg.max_seq_len * model_cfg.head_dim +
            seq_pos * model_cfg.head_dim;
        return kv_cache.raw_ptr + offset;
    }

    // Grid/block configuration
    __device__ __host__ inline dim3 grid() {
        return dim3(model_cfg.sm_count);
    }

    __device__ __host__ inline dim3 block() {
        return dim3(config::NUM_THREADS);
    }

    __device__ __host__ inline int dynamic_shared_memory() {
        return config::DYNAMIC_SHARED_MEMORY;
    }
};

// ============================================================================
// Buffer Allocation Helpers
// ============================================================================

template <typename config>
__host__ inline void allocate_buffers(
    RuntimeGlobals<config> &globals,
    const RuntimeModelConfig &model_cfg,
    const char *device
) {
    globals.model_cfg = model_cfg;

    // Calculate buffer sizes
    uint64_t weight_size = model_cfg.total_weight_elements();
    uint64_t kv_cache_size = model_cfg.num_layers * model_cfg.kv_cache_elements_per_layer();
    uint64_t activation_size = model_cfg.max_batch_size * model_cfg.hidden_dim;
    uint64_t scratch_size = model_cfg.sm_count * 16384;  // 16KB per SM

    // Allocate on device
    // Note: This is pseudo-code - actual allocation would use PyTorch tensors
    // and bind them through pybind11

    /*
    globals.unified_weights = allocate_tensor<bf16>(weight_size, device);
    globals.kv_cache = allocate_tensor<bf16>(kv_cache_size, device);
    globals.hidden_states = allocate_tensor<bf16>(activation_size, device);
    globals.scratch_buffer = allocate_tensor<bf16>(scratch_size, device);
    // ... etc
    */
}

// ============================================================================
// Model Configuration from PyTorch
// ============================================================================

// Helper to extract RuntimeModelConfig from HuggingFace config
__host__ inline RuntimeModelConfig config_from_hf_llama(
    uint16_t num_layers,
    uint16_t hidden_dim,
    uint16_t intermediate_dim,
    uint32_t vocab_size,
    uint16_t num_attention_heads,
    uint16_t num_kv_heads,
    uint16_t head_dim,
    uint16_t max_position_embeddings,
    float rope_theta,
    float rms_norm_eps
) {
    RuntimeModelConfig cfg = {};

    cfg.num_layers = num_layers;
    cfg.hidden_dim = hidden_dim;
    cfg.intermediate_dim = intermediate_dim;
    cfg.vocab_size = vocab_size;

    cfg.num_q_heads = num_attention_heads;
    cfg.num_kv_heads = num_kv_heads;
    cfg.head_dim = head_dim;
    cfg.max_seq_len = max_position_embeddings;

    cfg.attn_config = make_attn_config(
        (num_attention_heads == num_kv_heads) ? ATTN_TYPE_MHA : ATTN_TYPE_GQA,
        POS_EMB_ROPE,
        MASK_CAUSAL,
        ATTN_FEAT_FLASH
    );

    cfg.rope_theta = rope_theta;
    cfg.rope_scaling = 1.0f;
    cfg.attn_scale = 1.0f / sqrtf(static_cast<float>(head_dim));

    cfg.mlp_activation = ActivationType::SWIGLU;
    cfg.mlp_gated = true;

    cfg.norm_type = NormType::RMS_NORM;
    cfg.norm_eps = rms_norm_eps;

    // Default tiling (will be tuned later)
    cfg.matmul_block_m = 16;
    cfg.matmul_block_n = 16;
    cfg.matmul_block_k = 512;
    cfg.kv_block_size = 16;

    // H100 defaults
    cfg.sm_count = 132;
    cfg.is_hopper = true;

    return cfg;
}

} // namespace generic
} // namespace megakernel
