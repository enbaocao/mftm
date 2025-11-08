# Generic ISA Smoke Demo

Minimal CUDA extension to exercise the Generic ISA MATMUL op via the megakernel.

Build

```bash
export THUNDERKITTENS_ROOT=$(pwd)/ThunderKittens
export MEGAKERNELS_ROOT=$(pwd)
export PYTHON_VERSION=$(python3 -c 'import sys;print("%d.%d"%sys.version_info[:2])')
export GPU=H100  # or 4090/A100; unset for B200
cd demos/generic-hopper
make
```

Run optional smoke test (requires CUDA + PyTorch)

```bash
cd ../..
# Option A: run unit tests:
RUN_GENERIC_SMOKE=1 python -m unittest tests/generic/test_smoke_demo.py -v
RUN_GENERIC_SMOKE=1 python -m unittest tests/generic/test_smoke_fused.py -v

# Option B: use the quick smoke script:
python megakernels/scripts/generic_smoke.py all
```

This test constructs a single MATMUL instruction and checks that the output matches a known matvec result.
