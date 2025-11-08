#pragma once

#include "../instruction.cuh"
#include "../opcodes.cuh"
#include "../globals.cuh"

namespace megakernel {
namespace generic {

// ============================================================================
// Load/Store Index Structure (16 bytes)
// ============================================================================
// Flexible memory addressing for arbitrary tensor shapes
struct alignas(16) LoadStoreIndex {
    uint8_t tensor_id;      // Which tensor (0-255)
    uint8_t gpu_id;         // Which GPU (for multi-GPU setups)
    uint8_t dtype;          // Data type (FP32, BF16, FP16, etc.)
    uint8_t operation;      // Operation (LOAD, STORE, ADD, SUBTRACT, etc.)

    // Indices for 4D tensors (in units of 1, not bytes)
    // Specify the lower corner of the tile
    uint16_t idx_0;         // First dimension
    uint16_t idx_1;         // Second dimension
    uint32_t idx_2;         // Third dimension
    uint32_t idx_3;         // Fourth dimension

    // Data type enum
    static constexpr uint8_t DTYPE_FP32 = 0;
    static constexpr uint8_t DTYPE_BF16 = 1;
    static constexpr uint8_t DTYPE_FP16 = 2;
    static constexpr uint8_t DTYPE_E5M2 = 3;  // FP8 E5M2
    static constexpr uint8_t DTYPE_E4M3 = 4;  // FP8 E4M3
    static constexpr uint8_t DTYPE_INT8 = 5;
    static constexpr uint8_t DTYPE_INT32 = 6;

    // Operation enum
    static constexpr uint8_t OP_LOAD = 0;
    static constexpr uint8_t OP_STORE = 1;
    static constexpr uint8_t OP_ADD = 2;
    static constexpr uint8_t OP_SUBTRACT = 3;
    static constexpr uint8_t OP_ATOMIC_ADD = 4;
    static constexpr uint8_t OP_ATOMIC_MAX = 5;

    // Helper: Get element size in bytes
    __device__ __host__ inline int element_size() const {
        switch (dtype) {
            case DTYPE_FP32:
            case DTYPE_INT32:
                return 4;
            case DTYPE_BF16:
            case DTYPE_FP16:
                return 2;
            case DTYPE_E5M2:
            case DTYPE_E4M3:
            case DTYPE_INT8:
                return 1;
            default:
                return 4;
        }
    }

    // Helper: Calculate linear offset
    __device__ __host__ inline uint64_t linear_offset(
        uint32_t dim0, uint32_t dim1, uint32_t dim2, uint32_t dim3
    ) const {
        return (uint64_t)idx_0 * dim1 * dim2 * dim3 +
               (uint64_t)idx_1 * dim2 * dim3 +
               (uint64_t)idx_2 * dim3 +
               (uint64_t)idx_3;
    }
};

static_assert(sizeof(LoadStoreIndex) == 16, "LoadStoreIndex must be exactly 16 bytes");

// ============================================================================
// OP_COPY - Memory Copy Operation
// ============================================================================
// Copies data from source to destination
// Can handle different data types with optional conversion
template <typename config>
struct OpCopy {
    static constexpr int opcode = OP_COPY;

    struct consumer {
        template <typename globals>
        __device__ static inline void run(const globals &g,
                                           ::megakernel::state<config> &mks) {
            GenericInstruction inst{};
            inst.deserialize_from(mks.instruction());

            // Copy parameters:
            // - input_offset_0: source offset
            // - output_offset: destination offset
            // - n_dim: number of elements to copy
            // - flags: data type encoding

            const int num_elements = inst.n_dim;
            const uint32_t src_offset = inst.input_offset_0;
            const uint32_t dst_offset = inst.output_offset;
            const uint8_t dtype = inst.flags & 0x0F;

            // Parallel copy across all threads
            const int tid = threadIdx.x;
            const int num_threads = blockDim.x;

            switch (dtype) {
                case LoadStoreIndex::DTYPE_FP32: {
                    const float *src = g.template ptr_input0<const float>(src_offset);
                    float *dst = g.template ptr_output<float>(dst_offset);
                    for (int i = tid; i < num_elements; i += num_threads) {
                        dst[i] = src[i];
                    }
                    break;
                }
                case LoadStoreIndex::DTYPE_BF16:
                case LoadStoreIndex::DTYPE_FP16: {
                    const __half *src = g.template ptr_input0<const __half>(src_offset);
                    __half *dst = g.template ptr_output<__half>(dst_offset);
                    for (int i = tid; i < num_elements; i += num_threads) {
                        dst[i] = src[i];
                    }
                    break;
                }
                case LoadStoreIndex::DTYPE_INT8: {
                    const int8_t *src = g.template ptr_input0<const int8_t>(src_offset);
                    int8_t *dst = g.template ptr_output<int8_t>(dst_offset);
                    for (int i = tid; i < num_elements; i += num_threads) {
                        dst[i] = src[i];
                    }
                    break;
                }
                default: {
                    // Generic byte copy
                    const uint8_t *src = g.template ptr_input0<const uint8_t>(src_offset);
                    uint8_t *dst = g.template ptr_output<uint8_t>(dst_offset);
                    for (int i = tid; i < num_elements; i += num_threads) {
                        dst[i] = src[i];
                    }
                    break;
                }
            }

            __syncthreads();
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
// OP_ZERO - Zero Memory Operation
// ============================================================================
// Zeros out a region of memory
template <typename config>
struct OpZero {
    static constexpr int opcode = OP_ZERO;

    struct consumer {
        template <typename globals>
        __device__ static inline void run(const globals &g,
                                           ::megakernel::state<config> &mks) {
            GenericInstruction inst{};
            inst.deserialize_from(mks.instruction());

            // Zero parameters:
            // - output_offset: destination offset
            // - n_dim: number of elements to zero
            // - flags: data type encoding

            const int num_elements = inst.n_dim;
            const uint32_t dst_offset = inst.output_offset;
            const uint8_t dtype = inst.flags & 0x0F;

            // Parallel zero across all threads
            const int tid = threadIdx.x;
            const int num_threads = blockDim.x;

            switch (dtype) {
                case LoadStoreIndex::DTYPE_FP32: {
                    float *dst = g.template ptr_output<float>(dst_offset);
                    for (int i = tid; i < num_elements; i += num_threads) {
                        dst[i] = 0.0f;
                    }
                    break;
                }
                case LoadStoreIndex::DTYPE_BF16:
                case LoadStoreIndex::DTYPE_FP16: {
                    __half *dst = g.template ptr_output<__half>(dst_offset);
                    for (int i = tid; i < num_elements; i += num_threads) {
                        dst[i] = __float2half(0.0f);
                    }
                    break;
                }
                case LoadStoreIndex::DTYPE_INT8: {
                    int8_t *dst = g.template ptr_output<int8_t>(dst_offset);
                    for (int i = tid; i < num_elements; i += num_threads) {
                        dst[i] = 0;
                    }
                    break;
                }
                case LoadStoreIndex::DTYPE_INT32: {
                    int32_t *dst = g.template ptr_output<int32_t>(dst_offset);
                    for (int i = tid; i < num_elements; i += num_threads) {
                        dst[i] = 0;
                    }
                    break;
                }
                default: {
                    // Generic byte zero
                    uint8_t *dst = g.template ptr_output<uint8_t>(dst_offset);
                    for (int i = tid; i < num_elements; i += num_threads) {
                        dst[i] = 0;
                    }
                    break;
                }
            }

            __syncthreads();
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
// Helper: Build copy instruction
// ============================================================================
__host__ inline GenericInstruction make_copy_instruction(
    uint32_t src_offset,
    uint32_t dst_offset,
    uint32_t num_elements,
    uint8_t dtype = LoadStoreIndex::DTYPE_FP32
) {
    GenericInstruction inst = {};
    inst.opcode = OP_COPY;
    inst.flags = dtype & 0x0F;

    inst.n_dim = num_elements;
    inst.input_offset_0 = src_offset;
    inst.output_offset = dst_offset;

    return inst;
}

__host__ inline GenericInstruction make_zero_instruction(
    uint32_t dst_offset,
    uint32_t num_elements,
    uint8_t dtype = LoadStoreIndex::DTYPE_FP32
) {
    GenericInstruction inst = {};
    inst.opcode = OP_ZERO;
    inst.flags = dtype & 0x0F;

    inst.n_dim = num_elements;
    inst.output_offset = dst_offset;

    return inst;
}

// ============================================================================
// Helper: Build load/store index
// ============================================================================
__host__ __device__ inline LoadStoreIndex make_load_store_index(
    uint8_t tensor_id,
    uint8_t dtype,
    uint8_t operation,
    uint16_t idx_0 = 0,
    uint16_t idx_1 = 0,
    uint32_t idx_2 = 0,
    uint32_t idx_3 = 0,
    uint8_t gpu_id = 0
) {
    LoadStoreIndex lsi = {};
    lsi.tensor_id = tensor_id;
    lsi.gpu_id = gpu_id;
    lsi.dtype = dtype;
    lsi.operation = operation;
    lsi.idx_0 = idx_0;
    lsi.idx_1 = idx_1;
    lsi.idx_2 = idx_2;
    lsi.idx_3 = idx_3;
    return lsi;
}

} // namespace generic
} // namespace megakernel
