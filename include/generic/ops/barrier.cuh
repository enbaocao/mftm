#pragma once

#include "../instruction.cuh"
#include "../opcodes.cuh"
#include "../globals.cuh"

namespace megakernel {
namespace generic {

// ============================================================================
// Barrier Wait Structure (8 bytes)
// ============================================================================
// Encodes barrier synchronization requirements
struct alignas(8) BarrierWait {
    uint32_t barrier_id;      // Which barrier to wait on
    uint8_t condition;        // 0: =, 1: <, 2: >=
    uint8_t _reserved;        // Alignment padding
    uint16_t expected_count;  // Count to wait for (24-bit value stored as 16-bit for alignment)

    // Condition enum
    static constexpr uint8_t COND_EQUAL = 0;
    static constexpr uint8_t COND_LESS = 1;
    static constexpr uint8_t COND_GREATER_EQUAL = 2;

    __device__ __host__ inline bool check(uint32_t actual_count) const {
        switch (condition) {
            case COND_EQUAL:
                return actual_count == expected_count;
            case COND_LESS:
                return actual_count < expected_count;
            case COND_GREATER_EQUAL:
                return actual_count >= expected_count;
            default:
                return false;
        }
    }
};

static_assert(sizeof(BarrierWait) == 8, "BarrierWait must be exactly 8 bytes");

// ============================================================================
// OP_BARRIER - Synchronization Barrier
// ============================================================================
// VM waits on barriers in advance (controller pre-wait) and marks completion
// in shared memory for the instruction
template <typename config>
struct OpBarrier {
    static constexpr int opcode = OP_BARRIER;

    struct consumer {
        template <typename globals>
        __device__ static inline void run(const globals &g,
                                           ::megakernel::state<config> &mks) {
            GenericInstruction inst{};
            inst.deserialize_from(mks.instruction());

            // Barrier wait parameters encoded in instruction fields:
            // - barrier_id: inst.input_offset_0 (reuse as barrier ID)
            // - condition: inst.flags & 0x03
            // - expected_count: inst.input_offset_1 (reuse as count)

            uint32_t barrier_id = inst.input_offset_0;
            uint8_t condition = inst.flags & 0x03;
            uint32_t expected_count = inst.input_offset_1;

            // The controller should have already waited on this barrier
            // Consumer threads just need to synchronize with each other
            __syncthreads();

            // Optional: Signal arrival to global barrier if needed
            if (::kittens::warpid() == 0 && ::kittens::laneid() == 0) {
                // Atomically increment the barrier counter
                if (barrier_id < g.model_cfg.num_layers * 10) {  // Sanity check
                    atomicAdd((int*)&g.barriers[barrier_id], 1);
                }
            }

            __syncthreads();
        }
    };

    struct controller {
        template <typename globals>
        __device__ static inline int release_lid(const globals &g,
                                                  typename config::instruction_t &inst,
                                                  int &query) {
            // Controller pre-waits on barrier before releasing instruction
            uint32_t barrier_id = inst[2];  // input_offset_0
            uint8_t condition = inst[0] & 0x03;  // flags
            uint32_t expected_count = inst[3];  // input_offset_1

            BarrierWait wait;
            wait.barrier_id = barrier_id;
            wait.condition = condition;
            wait.expected_count = expected_count;

            // Spin-wait until barrier condition is met
            if (barrier_id < g.model_cfg.num_layers * 10) {
                while (!wait.check(g.barriers[barrier_id])) {
                    __nanosleep(config::GMEM_SPIN_LOOP_SLEEP_NANOS);
                }
            }

            return query;
        }

        template <typename globals>
        __device__ static inline int init_semaphores(const globals &,
                                                      ::megakernel::state<config> &) {
            return 0;
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
};

// ============================================================================
// OP_SYNC - Simple thread/warp/block synchronization
// ============================================================================
template <typename config>
struct OpSync {
    static constexpr int opcode = OP_SYNC;

    struct consumer {
        template <typename globals>
        __device__ static inline void run(const globals &g,
                                           ::megakernel::state<config> &mks) {
            GenericInstruction inst{};
            inst.deserialize_from(mks.instruction());

            // Sync level encoded in flags:
            // 0: warp sync
            // 1: block sync
            // 2: grid sync (using barriers)
            uint8_t sync_level = inst.flags & 0x03;

            if (sync_level == 0) {
                // Warp sync
                __syncwarp();
            } else if (sync_level == 1) {
                // Block sync
                __syncthreads();
            } else {
                // Grid sync via barrier
                if (::kittens::warpid() == 0 && ::kittens::laneid() == 0) {
                    uint32_t barrier_id = inst.input_offset_0;
                    atomicAdd((int*)&g.barriers[barrier_id], 1);
                }
                __syncthreads();
            }
        }
    };

    struct controller {
        template <typename globals>
        __device__ static inline int release_lid(const globals &,
                                                  typename config::instruction_t &,
                                                  int &query) {
            return query;
        }

        template <typename globals>
        __device__ static inline int init_semaphores(const globals &,
                                                      ::megakernel::state<config> &) {
            return 0;
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
};

// ============================================================================
// Helper: Build barrier instruction
// ============================================================================
__host__ inline GenericInstruction make_barrier_instruction(
    uint32_t barrier_id,
    uint8_t condition,  // BarrierWait::COND_EQUAL, etc.
    uint32_t expected_count
) {
    GenericInstruction inst = {};
    inst.opcode = OP_BARRIER;
    inst.flags = condition & 0x03;

    inst.input_offset_0 = barrier_id;
    inst.input_offset_1 = expected_count;

    return inst;
}

__host__ inline GenericInstruction make_sync_instruction(
    uint8_t sync_level  // 0: warp, 1: block, 2: grid
) {
    GenericInstruction inst = {};
    inst.opcode = OP_SYNC;
    inst.flags = sync_level & 0x03;

    return inst;
}

} // namespace generic
} // namespace megakernel
