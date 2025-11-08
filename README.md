# Megakernels: High-Performance LLM Inference

An extension to megakernels for high-performance LLM inference on NVIDIA GPUs, built on top of **ThunderKittens** - a framework for writing fast CUDA kernels.

## Overview

This project implements optimized GPU kernels for low-latency transformer model inference using a novel megakernel architecture that schedules operations across streaming multiprocessors (SMs). The system includes:

- **Model-specific optimized kernels**: Hand-tuned CUDA kernels for Llama models (`demos/low-latency-llama/`)
- **Generic Instruction Set Architecture (ISA)**: Model-agnostic instruction set that works across different transformer architectures without recompilation (`include/generic/`)
- **Python scheduling layer**: DAG-based scheduler for instruction generation and execution (`megakernels/`)

## Key Features

- **High Performance**: ~1000 tokens/second on H100 for Llama 3.2 1B
- **Low Latency**: ~8.7ms per token on H100
- **Model-Agnostic**: Generic ISA supports Llama, GPT-2, Mistral, and more without recompilation
- **Hardware Portable**: Works on H100, A100, RTX 4090, and B200 with architecture-specific optimizations
- **Extensible**: Add new model architectures in <48 hours

## Quick Start

See [`GETTING_STARTED.md`](GETTING_STARTED.md) for complete setup instructions.

**TL;DR:**
```bash
git submodule update --init --recursive
python3 -m venv venv310 && source venv310/bin/activate
pip install -e .
huggingface-cli login
export THUNDERKITTENS_ROOT=$(pwd)/ThunderKittens MEGAKERNELS_ROOT=$(pwd) PYTHON_VERSION=3.10 GPU=H100
cd demos/low-latency-llama && make && cd ../..
python megakernels/scripts/generate.py mode=mk prompt="Hello world" ntok=10
```

## Project Structure

```
├── include/                    # C++ headers for kernel framework
│   ├── megakernel.cuh         # Core megakernel execution loop
│   ├── generic/                # Generic ISA implementation
│   │   ├── opcodes.cuh        # Operation opcodes
│   │   ├── model_config.cuh   # Runtime model configuration
│   │   ├── instruction.cuh    # Instruction format
│   │   ├── globals.cuh        # Runtime globals
│   │   └── ops/               # Operation implementations
│   └── controller/             # Instruction dispatch and pipeline management
├── megakernels/                # Python scheduling layer
│   ├── scheduler.py            # DAG-based scheduler
│   ├── instructions.py         # Instruction definitions
│   ├── generic_scheduler.py   # Model-agnostic scheduler
│   └── scripts/               # Utility scripts
├── demos/
│   ├── low-latency-llama/     # Optimized Llama kernels
│   └── generic-hopper/        # Generic ISA demo
├── tests/                      # Test suite
│   └── generic/               # Generic ISA tests
└── ThunderKittens/            # Submodule: CUDA primitives
```

## Documentation

- **[GETTING_STARTED.md](GETTING_STARTED.md)** - Complete setup and compilation guide
- **[PROGRESS.md](PROGRESS.md)** - Current project status and what's working
- **[GENERIC_ISA.md](GENERIC_ISA.md)** - Generic Instruction Set Architecture specification
- **[IMPLEMENTATION_DETAILS.md](IMPLEMENTATION_DETAILS.md)** - Implementation details for barrier, memory, and attention operations
- **[TK_OPTIMIZATION_PLAN.md](TK_OPTIMIZATION_PLAN.md)** - ThunderKittens performance optimization roadmap
- **[CLAUDE.md](CLAUDE.md)** - Developer guide and architecture deep dive

## Current Status

### ✅ Working
- Model-specific Llama kernels (1000+ tokens/sec on H100)
- Generic ISA infrastructure (opcodes, instruction format, scheduler)
- Barrier and memory operations (OP_BARRIER, OP_SYNC, OP_COPY, OP_ZERO)
- Attention prefill support (OP_ATTENTION_PARTIAL, OP_ATTENTION_REDUCE)
- Python scheduler for multi-model support
- Comprehensive test suite (24 CPU tests + CUDA smoke tests)

### 🚧 In Progress
- ThunderKittens performance optimization (critical - ops are ~1000x slower than needed)
- Generic operation kernels (currently naive single-threaded implementations)

### 📋 Planned
- Production-grade performance for generic ops
- Support for more model architectures
- Blackwell (B200) optimizations
- Multi-GPU tensor parallelism

See [`PROGRESS.md`](PROGRESS.md) for detailed status.

## Performance

**H100 Benchmarks:**
- Llama 3.2 1B: ~1000 tokens/second, ~8.7ms latency
- Generic ISA: Currently ~1000x slower (naive implementations), optimization in progress

**Expected Performance (after optimization):**
- Generic ISA: <10% overhead vs specialized kernels
- Blackwell (B200): ~2-2.5x speedup vs H100

## Requirements

- NVIDIA GPU (H100, A100, RTX 4090, or B200)
- CUDA 12.x or higher
- Python 3.10+
- Ubuntu/Linux (macOS with CUDA also works)

## Contributing

See [`CLAUDE.md`](CLAUDE.md) for development workflow and architecture details.

To add support for a new model architecture:
1. Define `RuntimeModelConfig` preset in `include/generic/model_config.cuh`
2. If needed, add new opcodes to `include/generic/opcodes.cuh`
3. Implement operation kernels if unique
4. Add Python factory method in `megakernels/generic_scheduler.py`
5. Test correctness against PyTorch reference

Expected time: **<48 hours** (vs. weeks for specialized implementation).

## License

See [LICENSE](LICENSE) file.

## References

- **ThunderKittens**: CUDA tile-based primitives framework
- **Flash Attention**: Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention"
- **GQA**: Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models"
