# Agent Instructions

Scope: this file applies to everything under `xin_tutorial/`.

## Project Shape

This repository contains standalone tutorial examples for GPU systems topics.
Tasks live under thematic **category** directories; each task directory keeps a
globally sequential task number (publication order) and is independent, with
its own README and, for CUDA examples, its own `build_and_run.sh`. Category
folders own the theme, task numbers own the order; never renumber tasks.

- `communication/01_stream_overlap/`: PyTorch/NCCL stream overlap example.
- `communication/02_fused_peer_write_rs/`: CUDA C++ fused peer-write
  reduce-scatter example.
- `communication/03_nvshmem_warp_specialization/`: CUDA C++ NVSHMEM
  warp-specialization example.
- `communication/04_copy_engine_near_zero_sm/`: CUDA C++ peer-copy/copy-engine
  example.
- `cuda/05_cuda_graph_pitfalls/`: CUDA C++ CUDA Graph capture/replay
  pitfall-and-fix examples (three hard constraints); single-GPU task.
- `cuda/06_programmatic_dependent_launch/`: CUDA C++ + Triton Programmatic
  Dependent Launch (PDL) producer/consumer overlap demos (SM90+ required,
  single GPU).
- `cuda/07_gpu_concurrency_lab/`: Python + CUDA C++ single-GPU concurrency
  benchmark comparing single-stream, multi-stream, and multi-process (NVIDIA
  MPS) modes under a fixed P99 SLA, plus a block-slot occupancy
  microbenchmark; single-GPU task.

Planned categories with no tasks yet: `kernel/` (kernel authoring and
optimization, e.g. warp primitives, Triton/CUTLASS, fusion strategies),
`fundamentals/` (architecture and systems basics, e.g. SM/warp structure,
memory hierarchy, bandwidth/latency, numeric formats), `parallelism/`
(parallelism strategies, e.g. DP/TP/SP/PP/EP, ZeRO/FSDP, sharding and
resharding), and `systems/` (inference/training framework mechanics, e.g.
continuous batching, PagedAttention, KV cache management, scheduling).
Further themes (e.g. profiling methodology) may become categories later.
When adding a task: pick the closest existing category (or create the
directory when a planned or new theme gets its first task), continue the
global numbering, and update the README category table, task index, and
update log.

Keep examples small, readable, and easy to profile. Prefer local fixes inside
the relevant task directory unless a shared document or script clearly
needs an update.

## Running And Profiling

- These examples generally require a GPU worker with at least 2 NVIDIA GPUs on
  one node.
- Keep profiling output outside this repo directory when possible, for example
  under `../worker_results/<run_name>/` relative to the repo root (from inside
  a task directory: `../../../worker_results/<run_name>/`).
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
- Prefer relative paths such as `communication/03_nvshmem_warp_specialization/`
  and `../worker_results/<run_name>/`.
- Keep environment-specific absolute paths only when they are generic system
  paths needed to run commands, such as `/tmp`, `/usr/local/cuda`, or
  `/usr/lib/x86_64-linux-gnu/nvidia/current`.
- Every other absolute path must be rewritten under the shared virtual prefix
  `/path/to` (for example `/path/to/model`). The prefix is user-configurable
  per run with `--virtual-prefix <prefix>` or the `XIN_VIRTUAL_PREFIX`
  environment variable; a configured prefix replaces the default.

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
- To check against a different virtual prefix (replacing the default
  `/path/to`), add `--virtual-prefix <prefix>` to either command or export
  `XIN_VIRTUAL_PREFIX`.

## Editing Style

- Keep code and docs ASCII-only unless an existing file already uses non-ASCII
  for user-facing text.
- Use small, explicit changes. Avoid broad refactors across tutorial tasks.
- Preserve the tutorial focus: correctness checks and observable profiling
  behavior are more important than production abstractions.
