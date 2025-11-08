# Tests for Generic ISA (Python)

This folder contains Python unit tests that validate the Generic ISA scheduler
and instruction packing. These tests do not require CUDA.

How to run:

- From repo root after `pip install -e .`:
  - `python -m unittest discover -s tests`

What’s covered:

- Instruction generation for Llama / GPT-2 via `UniversalScheduler`
- Opcode consistency between Python and C++ headers
- Instruction serialization layout (bit packing, field placement)

Note: GPU dispatch and kernels are not executed here. A CUDA smoke test target
will be added under `demos/generic-hopper/` once the generic kernel is wired.

