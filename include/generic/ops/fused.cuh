#pragma once

#include "../instruction.cuh"
#include "../opcodes.cuh"
#include "../globals.cuh"

namespace megakernel {
namespace generic {

// Fused Norm + MatMul (minimal smoke implementation)
template <typename config>
struct OpFusedNormMatmul {
    static constexpr int opcode = OP_FUSED_NORM_MATMUL;

    struct consumer {
        template <typename globals>
        __device__ static inline void run(const globals &g,
                                           ::megakernel::state<config> &mks) {
            GenericInstruction inst{};
            inst.deserialize_from(mks.instruction());

            if (::kittens::laneid() == 0) {
                const int N = inst.n_dim;  // output size
                const int K = inst.k_dim;  // hidden size
                const float eps = inst.scale_factor;

                const float *x = g.template ptr_input0<const float>(inst.input_offset_0);
                const float *gamma = g.template ptr_input1<const float>(inst.input_offset_1);
                const float *W = g.template ptr_weight<const float>(inst.weight_offset); // rows of length K
                float *y = g.template ptr_output<float>(inst.output_offset);

                float sum_sq = 0.f;
                #pragma unroll 1
                for (int k = 0; k < K; ++k) sum_sq += x[k] * x[k];
                float inv_rms = rsqrtf(sum_sq / (float)K + eps);

                // y[n] = (norm(x) * gamma) dot W[n,:]
                #pragma unroll 1
                for (int n = 0; n < N; ++n) {
                    float acc = 0.f;
                    const float *wrow = W + n * K;
                    #pragma unroll 1
                    for (int k = 0; k < K; ++k) {
                        float xn = x[k] * inv_rms * (gamma ? gamma[k] : 1.f);
                        acc += xn * wrow[k];
                    }
                    y[n] = acc;
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

// Fused Norm + QKV + RoPE (minimal: behaves like Norm+MatMul for smoke)
template <typename config>
struct OpFusedNormQkvRope {
    static constexpr int opcode = OP_FUSED_NORM_QKV_ROPE;

    struct consumer {
        template <typename globals>
        __device__ static inline void run(const globals &g,
                                           ::megakernel::state<config> &mks) {
            // For the smoke implementation, reuse the Norm+MatMul logic
            OpFusedNormMatmul<config>::consumer::run(g, mks);
        }
    };
    struct loader { template <typename globals> __device__ static inline void run(const globals &, ::megakernel::state<config> &) {} };
    struct storer { template <typename globals> __device__ static inline void run(const globals &, ::megakernel::state<config> &) {} };
    struct launcher { template <typename globals> __device__ static inline void run(const globals &, ::megakernel::state<config> &) {} };
    struct controller {
        template <typename globals>
        __device__ static inline int release_lid(const globals &, typename config::instruction_t &, int &query) { return query; }
        template <typename globals>
        __device__ static inline int init_semaphores(const globals &, ::megakernel::state<config> &) { return 0; }
    };
};

} // namespace generic
} // namespace megakernel
