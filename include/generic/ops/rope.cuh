#pragma once

#include "../instruction.cuh"
#include "../opcodes.cuh"
#include "../globals.cuh"
#include <math.h>

namespace megakernel {
namespace generic {

// Minimal RoPE embedding op for smoke testing
template <typename config>
struct OpRopeEmbed {
    static constexpr int opcode = OP_ROPE_EMBED;

    struct consumer {
        template <typename globals>
        __device__ static inline void run(const globals &g,
                                           ::megakernel::state<config> &mks) {
            GenericInstruction inst{};
            inst.deserialize_from(mks.instruction());

            if (::kittens::warpid() == 0 && ::kittens::laneid() == 0) {
                const int head_dim = inst.k_dim;
                const float angle = inst.scale_factor; // Interpret scale_factor as rotation angle

                const float *x = g.template ptr_input0<const float>(inst.input_offset_0);
                float *y = g.template ptr_output<float>(inst.output_offset);

                float c = cosf(angle);
                float s = sinf(angle);
                // Apply pair-wise rotation: (x0, x1) -> (x0*c - x1*s, x0*s + x1*c)
                for (int d = 0; d < head_dim; d += 2) {
                    float x0 = x[d];
                    float x1 = (d + 1 < head_dim) ? x[d + 1] : 0.f;
                    float y0 = x0 * c - x1 * s;
                    float y1 = x0 * s + x1 * c;
                    y[d] = y0;
                    if (d + 1 < head_dim) y[d + 1] = y1;
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

} // namespace generic
} // namespace megakernel
