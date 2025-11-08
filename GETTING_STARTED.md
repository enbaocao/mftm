# Getting Started with Megakernels

**✅ Tested and Working on H100 SXM5**

## Quick Start (Copy-Paste)

```bash
# Clone and setup
git clone <your-repo-url>
cd tvm
git submodule update --init --recursive

# Create Python environment
python3 -m venv venv310
source venv310/bin/activate
pip install -e .

# HuggingFace authentication (for Llama models)
pip install -U "huggingface_hub[cli]"
huggingface-cli login
# Get token: https://huggingface.co/settings/tokens
# Accept license: https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct

# Set environment variables (Python version must match venv)
export THUNDERKITTENS_ROOT=$(pwd)/ThunderKittens
export MEGAKERNELS_ROOT=$(pwd)
export PYTHON_VERSION=3.10  # Must match: python --version
export GPU=H100             # Options: H100, A100, 4090, or unset for B200

# Compile CUDA kernel
cd demos/low-latency-llama
make
cd ../..

# Test it works!
python megakernels/scripts/generate.py mode=mk prompt="Hello world" ntok=10
```

**Expected result:** ~1000 tokens/second on H100

## Prerequisites

- **GPU**: NVIDIA H100, A100, RTX 4090, or B200
- **CUDA**: 12.x or higher
- **Python**: 3.10+ (3.12 recommended)
- **OS**: Ubuntu/Linux (macOS with CUDA also works)

### Verify Prerequisites

```bash
# Check GPU
nvidia-smi
# Should show: NVIDIA H100, A100, RTX 4090, or B200

# Check CUDA compiler
nvcc --version
# Should show: CUDA 12.x or higher

# Check Python version
python3 --version
# Should show: Python 3.9+ (3.12 recommended)
```

## Step-by-Step Setup

### 1. Clone Repository

```bash
git clone <your-repo-url>
cd tvm
git submodule update --init --recursive
```

### 2. Create Python Environment

```bash
# Use system Python (match version below)
python3 -m venv venv310
source venv310/bin/activate
pip install -e .
```

**Note:** The `pyproject.toml` requires Python >= 3.10.

### 3. Install HuggingFace CLI (for Model Access)

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli login
```

**Get your token:** https://huggingface.co/settings/tokens  
**Accept Llama license:** https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct

### 4. Set Environment Variables

```bash
# From repository root
export THUNDERKITTENS_ROOT=$(pwd)/ThunderKittens
export MEGAKERNELS_ROOT=$(pwd)

# Match your Python version (check with: python --version)
export PYTHON_VERSION=3.10  # or 3.11, 3.12, 3.13

# Match your GPU
export GPU=H100  # Options: 4090, A100, H100, or unset for B200
```

**GPU Selection Guide:**
- **RTX 4090**: `export GPU=4090` (compute capability 8.9)
- **A100**: `export GPU=A100` (compute capability 8.0)
- **H100**: `export GPU=H100` (compute capability 9.0a)
- **B200**: Don't set GPU variable, defaults to B200 (compute capability 10.0a)

### 5. Compile CUDA Kernel

```bash
cd demos/low-latency-llama
make
cd ../..
```

**Compilation time:** ~2-5 minutes on first build

**Expected output:**
```
ptxas info    : Used 96 registers, used 16 barriers...
nvlink info    : 0 bytes gmem
```

You should see: `mk_llama.cpython-310-x86_64-linux-gnu.so`

### 6. Verify Installation

```bash
python megakernels/scripts/generate.py mode=mk prompt="Hello world" ntok=10
```

**Expected output:**
```
Average time: 8.70ms
Tokens per second: 1034.04
Output text: ["!\n\nI'm excited to share my first post on"]
```

🎉 **Success!** You're generating 1000+ tokens/second on H100!

## Common Commands

```bash
# Generate text with megakernel
python megakernels/scripts/generate.py mode=mk prompt="Tell me a story" ntok=50

# Compare megakernel vs PyTorch
python megakernels/scripts/diff_test.py

# Interactive REPL
python megakernels/scripts/llama_repl.py

# Benchmark performance
python megakernels/scripts/generate.py mode=mk ntok=100
```

**Available modes:**
- `mode=model`: PyTorch reference implementation
- `mode=pyvm`: Python VM (instruction-based, no CUDA kernel)
- `mode=mk`: Megakernel (compiled CUDA)

## Troubleshooting

### Error: "No module named 'mk_llama'"

**Cause:** Python version mismatch between compilation and runtime

**Fix:**
```bash
# Check your Python version
python --version

# Set PYTHON_VERSION to match
export PYTHON_VERSION=3.10  # or 3.11, 3.12, etc.

# Recompile
cd demos/low-latency-llama
make clean
make
cd ../..
```

### Error: "nvcc: command not found"

**Solution:** Install CUDA Toolkit
```bash
# Check if CUDA is installed
ls /usr/local/cuda

# If not, download from: https://developer.nvidia.com/cuda-downloads
# Or use conda:
conda install -c nvidia cuda-toolkit
```

### Error: "Python.h: No such file or directory"

**Solution:** Install Python development headers
```bash
# Ubuntu/Debian
sudo apt-get install python3-dev

# CentOS/RHEL
sudo yum install python3-devel

# Or use conda environment (recommended)
conda install python=3.12
```

### Error: "cannot find -lpython3.XX"

**Cause:** Python dev libraries not installed or version mismatch

**Fix:**
```bash
# Check available Python versions
ls /usr/lib/libpython* 2>/dev/null || ls /usr/local/lib/libpython*

# Or check with pkg-config
pkg-config --list-all | grep python

# Update PYTHON_VERSION to match
export PYTHON_VERSION=3.11  # or whatever you have

# Or use the system Python version that has dev libs
python3.10 -m venv venv310
```

### Error: "pybind11/pybind11.h: No such file"

**Fix:**
```bash
pip install pybind11
```

### Error: Architecture mismatch (wrong GPU selected)

**Symptoms:**
```
ptxas error: unsupported gpu architecture
```

**Solution:** Set correct GPU variable
```bash
# Check your GPU
nvidia-smi | grep "NVIDIA"

# Set matching GPU variable
export GPU=A100  # or 4090, H100, etc.
```

### Error: "ThunderKittens not found"

**Solution:** Initialize git submodules
```bash
git submodule update --init --recursive
```

### Error: "401 Client Error: Unauthorized"

**Cause:** Need HuggingFace authentication for Llama model

**Fix:**
```bash
huggingface-cli login
# Then accept license at: https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
```

### Compilation is slow or hangs

**Cause:** NVCC compiling for multiple architectures

**Solution:** Already optimized - Makefile only compiles for your selected GPU
- H100: `-arch=sm_90a`
- B200: `-arch=sm_100a`
- A100: `-arch=sm_80`
- 4090: `-arch=sm_89`

## Recompiling After Code Changes

```bash
# Always recompile after modifying .cu files
cd demos/low-latency-llama
make clean
make
cd ../..
```

## Advanced Options

### Multi-GPU Systems

If you have multiple GPUs:

```bash
# See all GPUs
nvidia-smi

# Set specific GPU for testing
export CUDA_VISIBLE_DEVICES=0  # Use first GPU
python3 megakernels/scripts/generate.py
```

### Docker Option (Alternative)

If you have CUDA driver but want isolated environment:

```bash
# Use NVIDIA PyTorch container
docker run --gpus all -it --rm \
  -v $(pwd):/workspace \
  nvcr.io/nvidia/pytorch:24.09-py3 \
  bash

# Inside container
cd /workspace
export THUNDERKITTENS_ROOT=/workspace/ThunderKittens
export MEGAKERNELS_ROOT=/workspace
export PYTHON_VERSION=3.10
export GPU=H100
cd demos/low-latency-llama
make
```

### Remote Compilation

If compiling on a remote server:

```bash
# SSH to GPU server
ssh user@gpu-server

# Clone and compile
git clone <repo-url>
cd <repo>
git submodule update --init --recursive
pip install -e .

# Follow compilation steps above
export THUNDERKITTENS_ROOT=$(pwd)/ThunderKittens
# ... etc
```

## Makefile Details

The Makefile in `demos/low-latency-llama/Makefile`:
- Uses `nvcc` to compile CUDA code
- Links against Python, pybind11, CUDA libraries
- Creates a Python extension module
- Optimizes for your specific GPU architecture
- Includes ThunderKittens headers
- Compiles with `-O3` optimization and fast math

Key compiler flags:
```makefile
NVCCFLAGS=-O3 -std=c++20 -DNDEBUG
NVCCFLAGS+=-I${THUNDERKITTENS_ROOT}/include
NVCCFLAGS+=$(shell python3 -m pybind11 --includes)
NVCCFLAGS+=-arch=sm_90a  # For H100
```

## Performance Benchmarks (H100)

| Model | Tokens/Second | Latency |
|-------|---------------|---------|
| Llama 3.2 1B | ~1000 | ~8.7ms |
| Llama 3.2 3B | ~TBD | ~TBD |

## Next Steps

- See `PROGRESS.md` for current project status
- See `GENERIC_ISA.md` for multi-model architecture
- See `TK_OPTIMIZATION_PLAN.md` for performance optimization roadmap
- See `CLAUDE.md` for developer guide

## Quick Reference Card

**Initial setup:**
```bash
git submodule update --init --recursive
python3 -m venv venv310 && source venv310/bin/activate
pip install -e .
huggingface-cli login
```

**Compile:**
```bash
export PYTHON_VERSION=3.10 THUNDERKITTENS_ROOT=$(pwd)/ThunderKittens MEGAKERNELS_ROOT=$(pwd) GPU=H100
cd demos/low-latency-llama && make && cd ../..
```

**Run:**
```bash
python megakernels/scripts/generate.py mode=mk prompt="test" ntok=10
```

That's it! 🚀

