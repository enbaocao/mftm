# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Megakernels** project for high-performance LLM inference, built on top of **ThunderKittens** - a framework for writing fast CUDA kernels. The project implements optimized GPU kernels for low-latency Llama model inference using a novel megakernel architecture that schedules operations across streaming multiprocessors (SMs).

**NEW**: The project now includes a **Generic Instruction Set Architecture (ISA)** (`include/generic/`) that works across different transformer models (Llama, GPT-2, Mistral, etc.) without recompilation. See `GENERIC_ISA.md` for details.

### Key Architecture Components

1. **Megakernel Framework** (`megakernels/`): Python layer that orchestrates kernel execution
   - `instructions.py`: Defines instruction opcodes and serialization for GPU operations
   - `scheduler.py`: DAG-based scheduler that assigns instructions to SMs using various strategies (round-robin, DAG-based, wave-based)
   - `mk.py`: Main interpreter that loads and executes compiled CUDA megakernels
   - `llama.py`: PyTorch-compatible Llama model implementation with KV-cache management
   - `python_vm.py`: Pure Python reference implementation for validation

2. **CUDA Kernels** (`demos/low-latency-llama/*.cu`): Fused CUDA operations
   - `llama.cu`: Main kernel entry point using pybind11 to expose to Python
   - `attention_partial.cu` / `attention_reduction.cu`: Split attention computation
   - `rms_matvec_rope_append.cu`: Fused RMSNorm + QKV projection + RoPE + KV-cache append
   - `matvec_adds.cu`: Matrix-vector operations with residual connections
   - `upgate.cu`: Fused up/gate projections with SiLU activation
   - `rms_lm_head.cu`: Final RMSNorm + LM head projection

3. **Megakernel Infrastructure** (`include/`): C++ headers for kernel framework
   - `megakernel.cuh`: Core megakernel execution loop with loader/storer/consumer/controller warps
   - `controller/`: Instruction dispatch and pipeline management
   - `config.cuh`: Hardware configuration (SM count, register allocation, pipeline stages)

4. **ThunderKittens** (`ThunderKittens/`): Submodule with tile-based CUDA primitives
   - Provides high-level abstractions for tensor cores, TMA, shared memory management
   - Located at `ThunderKittens/include/kittens.cuh`

## Development Commands

### Initial Setup

**✅ Verified Working on H100:**

```bash
git submodule update --init --recursive

# Create venv with system Python
python3 -m venv venv310
source venv310/bin/activate
pip install -e .

# Login to HuggingFace (for Llama model access)
pip install -U "huggingface_hub[cli]"
huggingface-cli login
```

**Note:** The `pyproject.toml` requires Python >= 3.10 (not 3.12).

### Compile Megakernel (Required Before Running)

**IMPORTANT:** `PYTHON_VERSION` must match your active Python environment exactly!

```bash
# Check your Python version first
python --version  # e.g., Python 3.10.12

# Set environment (adjust PYTHON_VERSION to match above)
export THUNDERKITTENS_ROOT=$(pwd)/ThunderKittens
export MEGAKERNELS_ROOT=$(pwd)
export PYTHON_VERSION=3.10  # Must match: python --version
export GPU=H100  # Options: H100, A100, 4090, or unset for B200

cd demos/low-latency-llama
make
cd ../..

# Verify compilation succeeded
ls demos/low-latency-llama/mk_llama*.so
# Should show: mk_llama.cpython-310-x86_64-linux-gnu.so
```

**Common issue:** If you get "No module named 'mk_llama'", the Python version in the .so filename must match your runtime Python version. Recompile with the correct `PYTHON_VERSION`.

The Makefile compiles CUDA kernels with:
- `-arch=sm_90a` for H100, `-arch=sm_100a` for B200
- pybind11 bindings to create a Python extension module (`mk_llama*.so`)
- Links against ThunderKittens headers and CUDA libraries

### Run Inference

Interactive REPL:
```bash
python megakernels/scripts/llama_repl.py
```

Benchmark generation:
```bash
python megakernels/scripts/generate.py mode=mk prompt="tell me a funny joke" ntok=100
```

Available modes:
- `mode=model`: PyTorch reference implementation
- `mode=pyvm`: Python VM (instruction-based, no CUDA kernel)
- `mode=mk`: Megakernel (compiled CUDA)

### Configuration

The project uses `pydra-config` for configuration. Key parameters in `generate.py`:
- `model`: HuggingFace model ID (default: "meta-llama/Llama-3.2-1B-Instruct")
- `device`: CUDA device (default: "cuda:0")
- `sched`: Scheduler mode - "rr" (round-robin), "dag" (dependency-aware), "wave" (operation grouping), "pool" (memory/compute separation)
- `setting`: "latency" (single-token, low batch) or "throughput" (high batch)
- `interleave_rope`: Whether to fuse RoPE into attention kernels (required for latency mode with mk/pyvm)

## Architecture Deep Dive

### Instruction Model

The megakernel uses a streaming instruction architecture:
1. Python scheduler creates a DAG of operations (e.g., RMSNorm, MatVec, Attention)
2. Instructions are serialized to 32-int arrays (`INTS_PER_INSTRUCTION`)
3. Instructions are assigned to SMs based on dependencies and costs
4. GPU loads instructions from global memory and dispatches via controller warp

Each instruction has:
- Opcode (e.g., `OPCODE_RMS_QKV_MatVecRopeAppend = 1`)
- Layer index, block indices for tiling
- Dependency information for synchronization

### Warp Specialization

Each SM block runs 4 warp groups with distinct roles (see `megakernel.cuh:118-140`):
- **Consumer warps** (0-11): Execute compute operations using high register count
- **Loader warp** (12-15): Asynchronously loads data from global memory
- **Storer warp** (16-19): Asynchronously stores results to global memory
- **Launcher warp** (20-23): Manages asynchronous kernel launches
- **Controller warp** (24-27): Fetches and dispatches instructions

### Scheduling Strategies

`scheduler.py` implements multiple assignment algorithms:
- **Round-robin** (`rr`): Simple rotation across SMs
- **DAG-based** (`dag`): Assigns based on earliest ready time and critical path priority
- **Wave** (`wave`): Groups operations of same type, assigns by cost
- **Pool** (`pool`): Separates memory-bound and compute-bound operations

The scheduler uses `instruction.cost(globs)` to estimate operation latency for load balancing.

### Model Globals

`BaseGlobals` (instructions.py:11) contains all model parameters and state:
- Stacked weights for all layers: `qkv_proj_weights`, `o_proj_weights`, `up_proj_weights`, etc.
- KV-cache tensors: `k_cache`, `v_cache`
- RoPE embeddings: `rope_cos`, `rope_sin`
- Model config: `num_hidden_layers`, `num_attention_heads`, `head_dim`, etc.
- Runtime state: `hidden_states`, `barriers`, `instructions`, `timings`

Weights are stacked along dimension 0 for all layers to enable indexed access.

## Important Development Notes

- **Always recompile** after modifying `.cu` files: `cd demos/low-latency-llama && make`
- **Python version must match** between Makefile and environment (default: 3.13)
- **GPU architecture** must be set correctly via `GPU` env var (H100 vs B200 have different SM counts and architectures)
- The ThunderKittens submodule is essential - ensure it's initialized before building
- For debugging, enable `#define MK_DEBUG` in CUDA files to see kernel execution logs
- Timings are stored in `globs.timings` tensor with shape `[num_sms, max_instructions, TIMING_SLOTS]`

## Extending the System

To add a new fused operation:
1. Define a new instruction class in `megakernels/demos/{latency,throughput}/instructions.py` with opcode and cost estimation
2. Implement CUDA kernel in `demos/low-latency-llama/*.cu` following ThunderKittens patterns
3. Register kernel operation in `llama.cu` pybind11 module
4. Update scheduler in corresponding `scheduler.py` to insert the instruction into the DAG
5. Recompile with `make` in the demo directory

## Testing and Validation

Use `diff_test.py` to compare megakernel outputs against PyTorch reference:
```bash
python megakernels/scripts/diff_test.py
```

This validates that instruction-based execution matches the reference model by computing element-wise differences.

## Generic Instruction Set Architecture (NEW)

The project includes a **model-agnostic instruction set** in `include/generic/` that enables running different transformer architectures without recompilation.

### Key Files
- `include/generic/opcodes.cuh` - ~20 primitive operations (MatMul, Norm, Attention, etc.)
- `include/generic/model_config.cuh` - Runtime model configuration
- `include/generic/instruction.cuh` - 64-byte generic instruction format
- `include/generic/globals.cuh` - Dynamic globals with variable dimensions
- `megakernels/generic_scheduler.py` - Python scheduler for multi-model support

### Quick Example
```python
# Generate instructions for different models
from megakernels.generic_scheduler import ModelConfig, UniversalScheduler

# Llama 3.2 1B
config = ModelConfig.from_llama_3_1b()
scheduler = UniversalScheduler(config)
instructions = scheduler.build_full_model()

# GPT-2 (same scheduler, different config!)
config = ModelConfig.from_gpt2()
scheduler = UniversalScheduler(config)
instructions = scheduler.build_full_model()
```

### Design Principles
1. **Runtime configuration**: Model dimensions specified at runtime, not compile-time
2. **Primitive operations**: 20 opcodes that compose into any transformer
3. **Flexible dispatch**: Single kernel binary handles multiple models
4. **Performance**: Target <10% overhead vs specialized kernels
5. **Portability**: Works on H100 and B200 with architecture-specific optimizations

### Supported Models
- ✅ Llama (RMSNorm + GQA + RoPE + SwiGLU)
- ✅ GPT-2 (LayerNorm + MHA + learned positions + GELU)
- ✅ Mistral (GQA + sliding window attention)
- 📋 Qwen, DeepSeek, others (planned)

### Documentation
- See `GENERIC_ISA.md` for complete architecture specification
- See `include/generic/README.md` for API reference
