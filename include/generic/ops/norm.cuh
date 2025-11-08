#pragma once

#include "../instruction.cuh"
#include "../opcodes.cuh"
#include "../globals.cuh"

namespace megakernel {
namespace generic {

// Generic RMSNorm operation (scaffold)
template <typename config>
struct OpRmsNorm {
    static constexpr int opcode = OP_RMS_NORM;

    struct consumer {
        template <typename globals>
        __device__ static inline void run(const globals &g,
                                           ::megakernel::state<config> &mks) {
            GenericInstruction inst{};
            inst.deserialize_from(mks.instruction());
            // Minimal RMSNorm over n_dim with epsilon = inst.scale_factor
            if (::kittens::warpid() == 0 && ::kittens::laneid() == 0) {
                const int N = inst.n_dim;
                const float eps = inst.scale_factor;

                const float *x = g.template ptr_input0<const float>(inst.input_offset_0);
                const float *w = g.template ptr_weight<const float>(inst.weight_offset);
                float *y = g.template ptr_output<float>(inst.output_offset);

                float sum_sq = 0.f;
                #pragma unroll 1
                for (int i = 0; i < N; ++i) {
                    float v = x[i];
                    sum_sq += v * v;
                }
                float rms = sqrtf(sum_sq / (float)N + eps);
                float inv_rms = 1.f / rms;

                #pragma unroll 1
                for (int i = 0; i < N; ++i) {
                    y[i] = (x[i] * inv_rms) * (w ? w[i] : 1.f);
                }
            }
        }
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

    struct loader {
        template <typename globals>
        __device__ static inline void run(const globals &g,
                                           ::megakernel::state<config> &mks) {
            (void)g; (void)mks;
        }
    };

    struct storer {
        template <typename globals>
        __device__ static inline void run(const globals &g,
                                           ::megakernel::state<config> &mks) {
            (void)g; (void)mks;
        }
    };

    struct launcher {
        template <typename globals>
        __device__ static inline void run(const globals &g,
                                           ::megakernel::state<config> &mks) {
            (void)g; (void)mks;
        }
    };
};

// Generic LayerNorm operation (scaffold)
template <typename config>
struct OpLayerNorm {
    static constexpr int opcode = OP_LAYER_NORM;

    struct consumer {
        template <typename globals>
        __device__ static inline void run(const globals &g,
                                           ::megakernel::state<config> &mks) {
            GenericInstruction inst{};
            inst.deserialize_from(mks.instruction());
            // Minimal LayerNorm over n_dim with epsilon = inst.scale_factor (gamma only)
            if (::kittens::warpid() == 0 && ::kittens::laneid() == 0) {
                const int N = inst.n_dim;
                const float eps = inst.scale_factor;

                const float *x = g.template ptr_input0<const float>(inst.input_offset_0);
                const float *gamma = g.template ptr_weight<const float>(inst.weight_offset);
                float *y = g.template ptr_output<float>(inst.output_offset);

                float mean = 0.f;
                #pragma unroll 1
                for (int i = 0; i < N; ++i) mean += x[i];
                mean /= (float)N;

                float var = 0.f;
                #pragma unroll 1
                for (int i = 0; i < N; ++i) {
                    float d = x[i] - mean;
                    var += d * d;
                }
                var /= (float)N;
                float inv_std = rsqrtf(var + eps);

                #pragma unroll 1
                for (int i = 0; i < N; ++i) {
                    float nh = (x[i] - mean) * inv_std;
                    y[i] = nh * (gamma ? gamma[i] : 1.f);
                }
            }
        }
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

    struct loader {
        template <typename globals>
        __device__ static inline void run(const globals &g,
                                           ::megakernel::state<config> &mks) {
            (void)g; (void)mks;
        }
    };

    struct storer {
        template <typename globals>
        __device__ static inline void run(const globals &g,
                                           ::megakernel::state<config> &mks) {
            (void)g; (void)mks;
        }
    };

    struct launcher {
        template <typename globals>
        __device__ static inline void run(const globals &g,
                                           ::megakernel::state<config> &mks) {
            (void)g; (void)mks;
        }
    };
};

} // namespace generic
} // namespace megakernel
