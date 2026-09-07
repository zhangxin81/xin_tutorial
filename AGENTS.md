# Agent Instructions

Scope: this file applies to everything under `xin_tutorial/`.

## Project Shape

This repository contains standalone tutorial examples for GPU communication and
compute overlap. Each numbered directory is an independent task with its own
README and, for CUDA examples, its own `build_and_run.sh`.

- `01_stream_overlap/`: PyTorch/NCCL stream overlap example.
- `02_fused_peer_write_rs/`: CUDA C++ fused peer-write reduce-scatter example.
- `03_nvshmem_warp_specialization/`: CUDA C++ NVSHMEM warp-specialization example.
- `04_copy_engine_near_zero_sm/`: CUDA C++ peer-copy/copy-engine example.
- `05_cuda_graph_pitfalls/`: CUDA C++ CUDA Graph capture/replay pitfall-and-fix
  examples (three hard constraints); single-GPU task.
- `06_programmatic_dependent_launch/`: CUDA C++ + Triton Programmatic Dependent
  Launch (PDL) producer/consumer overlap demos (SM90+ required, single GPU).

Keep examples small, readable, and easy to profile. Prefer local fixes inside
the relevant numbered task directory unless a shared document or script clearly
needs an update.

## Running And Profiling

- These examples generally require a GPU worker with at least 2 NVIDIA GPUs on
  one node.
- Keep profiling output outside this repo directory when possible, for example
  under `../worker_results/<run_name>/`.
- Use Nsight Systems for timeline validation. See `docs/NSYS_USAGE.md` for install,
  `nsys profile`, and `nsys export` examples.
- For task 03 NVSHMEM setup and known environment pitfalls, read
  `docs/NVSHMEM_SETUP_NOTES.md` before changing code or rerunning.

## CUDA/NCCL Defaults

For single-node worker runs, these defaults were verified to avoid the worker's
external NCCL network plugin path:

```bash
export NCCL_NET_PLUGIN=none
export NCCL_P2P_LEVEL=NVL
```

For H100 CUDA C++ examples, compile for the native architecture unless the task
README says otherwise:

```bash
CUDA_ARCH=sm_90
```

## NVSHMEM Task 03

Task 03 requires `nvshmrun` from the NVIDIA NVSHMEM binary archive; the pip
package may provide libraries without the launcher.

Use `docs/NVSHMEM_SETUP_NOTES.md` as the source of truth for:

- NVSHMEM binary archive installation.
- CUDA runtime compatibility handling.
- `NVSHMEM_BOOTSTRAP` behavior.
- NVML/P2P detection problems.
- Nsight Systems version issues.

## Documentation Hygiene

- Do not commit personal workspace paths, user names, worker ids, pod IPs, or
  trial names in documentation.
- Prefer relative paths such as `03_nvshmem_warp_specialization/` and
  `../worker_results/<run_name>/`.
- Keep environment-specific absolute paths only when they are generic system
  paths needed to run commands, such as `/tmp`, `/usr/local/cuda`, or
  `/usr/lib/x86_64-linux-gnu/nvidia/current`.

## Commit Gate

- The repository has a pre-commit sensitive-info check in
  `scripts/check_sensitive_paths.sh`.
- The repository-local Git hook path is `.githooks/`, whose `pre-commit` hook
  runs that script against staged text files.
- Before committing manually, run:

```bash
scripts/check_sensitive_paths.sh
```

- Before publishing or packaging a broader tree, run:

```bash
scripts/check_sensitive_paths.sh --all
```

- If the check fails, replace sensitive values with relative paths or
  placeholders. Do not bypass the hook for documentation changes.

## Editing Style

- Keep code and docs ASCII-only unless an existing file already uses non-ASCII
  for user-facing text.
- Use small, explicit changes. Avoid broad refactors across tutorial tasks.
- Preserve the tutorial focus: correctness checks and observable profiling
  behavior are more important than production abstractions.
