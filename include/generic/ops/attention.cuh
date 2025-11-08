#pragma once

#include "../instruction.cuh"
#include "../opcodes.cuh"
#include "../globals.cuh"
#include <math.h>

namespace megakernel {
namespace generic {

// ============================================================================
// Attention Partial - Compute attention over a KV block
// ============================================================================
// For prefill: Outputs partial results with LSE (log-sum-exp) for reduction
// For decode: Computes complete attention directly
template <typename config>
struct OpAttentionPartial {
    static constexpr int opcode = OP_ATTENTION_PARTIAL;

    struct consumer {
        template <typename globals>
        __device__ static inline void run(const globals &g,
                                           ::megakernel::state<config> &mks) {
            GenericInstruction inst{};
            inst.deserialize_from(mks.instruction());

            if (::kittens::warpid() == 0 && ::kittens::laneid() == 0) {
                const int head_dim = inst.k_dim;
                const int kv_len = (inst.reduction_factor > 0) ? inst.reduction_factor : 1;
                const float scale = inst.scale_factor == 0.0f ? 1.0f : inst.scale_factor;
                const bool is_partial = (inst.flags & 0x01);  // FLAG_PARTIAL

                const float *q = g.template ptr_input0<const float>(inst.input_offset_0);
                const float *k = g.template ptr_input1<const float>(inst.input_offset_1);
                const float *v = g.template ptr_input2<const float>(inst.input_offset_2);
                float *o = g.template ptr_output<float>(inst.output_offset);

                // Compute logits = (q · K_j) * scale
                float max_logit = -INFINITY;
                // First pass: compute logits and track max
                for (int j = 0; j < kv_len; ++j) {
                    float dot = 0.f;
                    const float *kj = k + j * head_dim;
                    #pragma unroll 1
                    for (int d = 0; d < head_dim; ++d) dot += q[d] * kj[d];
                    float logit = dot * scale;
                    if (logit > max_logit) max_logit = logit;
                }

                // Second pass: softmax denom
                float denom = 0.f;
                for (int j = 0; j < kv_len; ++j) {
                    float dot = 0.f;
                    const float *kj = k + j * head_dim;
                    #pragma unroll 1
                    for (int d = 0; d < head_dim; ++d) dot += q[d] * kj[d];
                    float logit = dot * scale;
                    denom += expf(logit - max_logit);
                }

                // Third pass: Output = sum_j softmax_j * V_j
                for (int d = 0; d < head_dim; ++d) o[d] = 0.f;
                for (int j = 0; j < kv_len; ++j) {
                    float dot = 0.f;
                    const float *kj = k + j * head_dim;
                    #pragma unroll 1
                    for (int d = 0; d < head_dim; ++d) dot += q[d] * kj[d];
                    float logit = dot * scale;
                    float w = expf(logit - max_logit) / denom;
                    const float *vj = v + j * head_dim;
                    #pragma unroll 1
                    for (int d = 0; d < head_dim; ++d) o[d] += w * vj[d];
                }

                // If this is a partial (for reduction), store LSE values
                // LSE values stored after output: [output, max_logit, log(denom)]
                if (is_partial && inst.scratch_offset != 0) {
                    float *lse = g.template ptr_output<float>(inst.scratch_offset);
                    lse[0] = max_logit;  // m_i
                    lse[1] = logf(denom);  // log(s_i)
                }
            }
        }
    };

    // No-op implementations for other workers (required by VM interface)
    struct loader {
        template <typename globals>
        __device__ static inline void run(const globals &, ::megakernel::state<config> &) {}
    };
    struct storer {
        template <typename globals>
        __device__ static inline void run(const globals &, ::megakernel::state<config> &) {}
    };
    struct launcher {
        template <typename globals>
        __device__ static inline void run(const globals &, ::megakernel::state<config> &) {}
    };

    struct controller {
        template <typename globals>
        __device__ static inline int release_lid(const globals &, typename config::instruction_t &, int &query) {
            return query;
        }
        template <typename globals>
        __device__ static inline int init_semaphores(const globals &, ::megakernel::state<config> &) {
            return 0;
        }
    };
};

// ============================================================================
// Attention Reduce - Combine multiple attention partials
// ============================================================================
// Uses log-sum-exp trick for numerical stability
// Input: Multiple (O_partial, m_i, log_s_i) tuples
// Output: Final attention output
template <typename config>
struct OpAttentionReduce {
    static constexpr int opcode = OP_ATTENTION_REDUCE;

    struct consumer {
        template <typename globals>
        __device__ static inline void run(const globals &g,
                                           ::megakernel::state<config> &mks) {
            GenericInstruction inst{};
            inst.deserialize_from(mks.instruction());

            if (::kittens::warpid() == 0 && ::kittens::laneid() == 0) {
                const int head_dim = inst.k_dim;
                const int num_partials = inst.reduction_factor;

                // Input layout:
                // Both partial outputs and LSE values are in buffer 'b' (input1)
                // input_offset_0: Base offset for partial outputs (head_dim per partial)
                // input_offset_1: Base offset for LSE values (2 floats per partial: m_i, log_s_i)
                // output_offset: Final output location

                const float *partial_outputs = g.template ptr_input1<const float>(inst.input_offset_0);
                const float *lse_values = g.template ptr_input1<const float>(inst.input_offset_1);
                float *final_output = g.template ptr_output<float>(inst.output_offset);

                // Step 1: Find global max logit across all partials
                float global_max = -INFINITY;
                for (int p = 0; p < num_partials; ++p) {
                    float m_i = lse_values[p * 2 + 0];  // max_logit for partial p
                    if (m_i > global_max) {
                        global_max = m_i;
                    }
                }

                // Step 2: Compute corrected sum and output
                float corrected_sum = 0.0f;
                for (int d = 0; d < head_dim; ++d) {
                    final_output[d] = 0.0f;
                }

                for (int p = 0; p < num_partials; ++p) {
                    float m_i = lse_values[p * 2 + 0];       // max_logit
                    float log_s_i = lse_values[p * 2 + 1];   // log(sum_exp)

                    // Correction factor: exp(m_i - global_max + log_s_i)
                    // = exp(m_i - global_max) * exp(log_s_i)
                    // = exp(m_i - global_max) * s_i
                    float correction = expf(m_i - global_max + log_s_i);
                    corrected_sum += correction;

                    // Accumulate weighted partial outputs
                    const float *o_i = partial_outputs + p * head_dim;
                    #pragma unroll 1
                    for (int d = 0; d < head_dim; ++d) {
                        final_output[d] += o_i[d] * correction;
                    }
                }

                // Step 3: Normalize by corrected sum
                float inv_sum = 1.0f / corrected_sum;
                #pragma unroll 1
                for (int d = 0; d < head_dim; ++d) {
                    final_output[d] *= inv_sum;
                }
            }
        }
    };

    struct loader {
        template <typename globals>
        __device__ static inline void run(const globals &, ::megakernel::state<config> &) {}
    };

    struct storer {
        template <typename globals>
        __device__ static inline void run(const globals &, ::megakernel::state<config> &) {}
    };

    struct launcher {
        template <typename globals>
        __device__ static inline void run(const globals &, ::megakernel::state<config> &) {}
    };

    struct controller {
        template <typename globals>
        __device__ static inline int release_lid(const globals &, typename config::instruction_t &, int &query) {
            return query;
        }
        template <typename globals>
        __device__ static inline int init_semaphores(const globals &, ::megakernel::state<config> &) {
            return 0;
        }
    };
};

// ============================================================================
// Helper: Build attention partial instruction (with LSE output)
// ============================================================================
__host__ inline GenericInstruction make_attention_partial_with_lse(
    uint16_t layer_idx,
    uint16_t num_heads, uint16_t head_dim, uint16_t kv_len,
    uint16_t attn_config,
    uint32_t q_offset, uint32_t k_offset, uint32_t v_offset,
    uint32_t output_offset,
    uint32_t lse_offset,  // Where to store LSE values
    float attn_scale = 0.0f
) {
    GenericInstruction inst = {};
    inst.opcode = OP_ATTENTION_PARTIAL;
    inst.layer_idx = layer_idx;
    inst.flags = 0x01;  // FLAG_PARTIAL (output LSE)

    inst.m_dim = num_heads;
    inst.k_dim = head_dim;
    inst.reduction_factor = kv_len;
    inst.head_config = attn_config;

    inst.input_offset_0 = q_offset;
    inst.input_offset_1 = k_offset;
    inst.input_offset_2 = v_offset;
    inst.output_offset = output_offset;
    inst.scratch_offset = lse_offset;

    inst.scale_factor = (attn_scale != 0.0f) ? attn_scale : (1.0f / sqrtf(head_dim));

    return inst;
}

// ============================================================================
// Helper: Build attention reduce instruction
// ============================================================================
__host__ inline GenericInstruction make_attention_reduce_instruction(
    uint16_t layer_idx,
    uint16_t head_dim,
    uint16_t num_partials,
    uint32_t partial_outputs_offset,  // Base offset for partial outputs
    uint32_t lse_offset,               // Base offset for LSE values
    uint32_t final_output_offset
) {
    GenericInstruction inst = {};
    inst.opcode = OP_ATTENTION_REDUCE;
    inst.layer_idx = layer_idx;

    inst.k_dim = head_dim;
    inst.reduction_factor = num_partials;

    inst.input_offset_0 = partial_outputs_offset;
    inst.input_offset_1 = lse_offset;
    inst.output_offset = final_output_offset;

    return inst;
}

} // namespace generic
} // namespace megakernel
