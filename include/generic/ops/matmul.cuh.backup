#pragma once

#include "../instruction.cuh"
#include "../opcodes.cuh"
#include "../globals.cuh"

namespace megakernel {
namespace generic {

// Generic MATMUL operation (scaffold)
// Defines per-worker entry points expected by the VM dispatch macros
template <typename config>
struct OpMatmul {
    static constexpr int opcode = OP_MATMUL;  // low 8 bits of instruction word 0

    struct consumer {
        template <typename globals>
        __device__ static inline void run(const globals &g,
                                           ::megakernel::state<config> &mks) {
            // Decode generic instruction from the instruction buffer
            GenericInstruction inst{};
            inst.deserialize_from(mks.instruction());

            // Minimal functional implementation for smoke testing:
            // Computes C = A @ B^T for M=1 (vector-matmul producing N outputs)
            // Data type assumed to be float for the smoke demo globals.
            // Only one lane performs the compute to keep things simple.
            if (::kittens::warpid() == 0 && ::kittens::laneid() == 0) {
                const int N = inst.n_dim;
                const int K = inst.k_dim;

                const float *a = g.template ptr_input0<const float>(inst.input_offset_0);
                const float *b = g.template ptr_weight<const float>(inst.weight_offset);
                float *c = g.template ptr_output<float>(inst.output_offset);

                #pragma unroll 1
                for (int n = 0; n < N; ++n) {
                    float acc = 0.f;
                    const float *b_row = b + n * K;
                    #pragma unroll 1
                    for (int k = 0; k < K; ++k) {
                        acc += a[k] * b_row[k];
                    }
                    c[n] = acc;
                }
            }
        }
    };

    struct loader {
        template <typename globals>
        __device__ static inline void run(const globals &g,
                                           ::megakernel::state<config> &mks) {
            // TODO: Issue async loads for A/B tiles if needed
            (void)g; (void)mks;
        }
    };

    struct storer {
        template <typename globals>
        __device__ static inline void run(const globals &g,
                                           ::megakernel::state<config> &mks) {
            // TODO: Issue async stores for output tiles if needed
            (void)g; (void)mks;
        }
    };

    struct launcher {
        template <typename globals>
        __device__ static inline void run(const globals &g,
                                           ::megakernel::state<config> &mks) {
            // Reserved for any sub-kernel launches (unlikely for matmul)
            (void)g; (void)mks;
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
};

} // namespace generic
} // namespace megakernel
