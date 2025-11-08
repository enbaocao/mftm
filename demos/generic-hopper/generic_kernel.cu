#include "kittens.cuh"
#include "megakernel.cuh"
#include "generic/generic.cuh"
#include "pyutils/pyutils.cuh"

using namespace kittens;
using namespace megakernel;

// Minimal globals for a matmul smoke test using float buffers
struct smoke_globals {
    using config = megakernel::default_config;
    using instruction_layout = megakernel::instruction_layout<config>;
    using timing_layout = megakernel::timing_layout<config>;

    instruction_layout instructions;
    timing_layout timings;

    // A: (1, 1, 1, K)
    kittens::gl<float, 1, 1, 1, -1> a;
    // B: (1, N, K) flattened as (1, 1, 1, N*K)
    kittens::gl<float, 1, 1, 1, -1> b;
    // C: (1, 1, 1, N)
    kittens::gl<float, 1, 1, 1, -1> c;
    // Barriers: (1, 1, 1, 256) - 256 barrier slots
    kittens::gl<int, 1, 1, 1, -1> barriers;

    // Model config (needed for barrier bounds checking)
    megakernel::generic::RuntimeModelConfig model_cfg;

    // Pointer helpers expected by generic ops
    template <typename T = float>
    __device__ __host__ inline const T *ptr_input0(uint32_t offset_elems) const {
        return reinterpret_cast<const T *>(a.raw_ptr) + offset_elems;
    }
    template <typename T = float>
    __device__ __host__ inline const T *ptr_input1(uint32_t offset_elems) const {
        return reinterpret_cast<const T *>(b.raw_ptr) + offset_elems;
    }
    template <typename T = float>
    __device__ __host__ inline const T *ptr_input2(uint32_t offset_elems) const {
        return reinterpret_cast<const T *>(b.raw_ptr) + offset_elems;
    }
    template <typename T = float>
    __device__ __host__ inline const T *ptr_weight(uint32_t offset_elems) const {
        return reinterpret_cast<const T *>(b.raw_ptr) + offset_elems;
    }
    template <typename T = float>
    __device__ __host__ inline T *ptr_output(uint32_t offset_elems) const {
        return reinterpret_cast<T *>(c.raw_ptr) + offset_elems;
    }

    dim3 grid() const { return dim3(1); }
    dim3 block() const { return dim3(config::NUM_THREADS); }
    int dynamic_shared_memory() const { return config::DYNAMIC_SHARED_MEMORY; }
};

using matmul_op = megakernel::generic::OpMatmul<megakernel::default_config>;
using rmsnorm_op = megakernel::generic::OpRmsNorm<megakernel::default_config>;
using layernorm_op = megakernel::generic::OpLayerNorm<megakernel::default_config>;
using attn_op = megakernel::generic::OpAttentionPartial<megakernel::default_config>;
using attn_reduce_op = megakernel::generic::OpAttentionReduce<megakernel::default_config>;
using rope_op = megakernel::generic::OpRopeEmbed<megakernel::default_config>;
using fused_nm_op = megakernel::generic::OpFusedNormMatmul<megakernel::default_config>;
using fused_nqkvr_op = megakernel::generic::OpFusedNormQkvRope<megakernel::default_config>;
using barrier_op = megakernel::generic::OpBarrier<megakernel::default_config>;
using sync_op = megakernel::generic::OpSync<megakernel::default_config>;
using copy_op = megakernel::generic::OpCopy<megakernel::default_config>;
using zero_op = megakernel::generic::OpZero<megakernel::default_config>;

PYBIND11_MODULE(mk_generic, m) {
    m.doc() = "Generic ISA smoke test kernel";
    kittens::py::bind_kernel<
        mk<megakernel::default_config, smoke_globals,
           matmul_op, rmsnorm_op, layernorm_op, attn_op, attn_reduce_op, rope_op, fused_nm_op, fused_nqkvr_op,
           barrier_op, sync_op, copy_op, zero_op>>(
        m, "mk_generic_matmul",
        &smoke_globals::instructions,
        &smoke_globals::timings,
        &smoke_globals::a,
        &smoke_globals::b,
        &smoke_globals::c,
        &smoke_globals::barriers
    );
    // Reuse the same binding name for all ops; the kernel inspects the opcode
}
