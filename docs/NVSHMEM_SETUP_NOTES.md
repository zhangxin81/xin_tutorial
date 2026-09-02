# NVSHMEM setup notes for task 03

This note records the NVSHMEM setup used to run and profile:

```text
xin_tutorial/03_nvshmem_warp_specialization
```

The verified run used one MLX GPU worker with 2x H100 GPUs. Commands below
assume they are run from the `xin_tutorial` directory unless noted otherwise.

## Install NVSHMEM On Worker

Install NVSHMEM into worker-local `/tmp`. `/tmp` is not shared with the devbox,
so repeat this on each new worker.

```bash
mkdir -p /tmp/nvshmem_install
cd /tmp/nvshmem_install

wget -O nvshmem.tar.xz \
  https://developer.download.nvidia.com/compute/nvshmem/redist/libnvshmem/linux-x86_64/libnvshmem-linux-x86_64-3.1.7_cuda12-archive.tar.xz

tar -xf nvshmem.tar.xz

export NVSHMEM_HOME=/tmp/nvshmem_install/libnvshmem-linux-x86_64-3.1.7_cuda12-archive
export PATH="${NVSHMEM_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${NVSHMEM_HOME}/lib:${LD_LIBRARY_PATH:-}"
```

Verify the installation:

```bash
test -x "${NVSHMEM_HOME}/bin/nvshmrun"
test -f "${NVSHMEM_HOME}/include/nvshmem.h"
test -f "${NVSHMEM_HOME}/lib/libnvshmem_host.so"
```

## CUDA Runtime Compatibility

The task was compiled with the system CUDA 12.9 `nvcc`, but the successful run
used CUDA 12.2 runtime libraries from pip because the worker driver reports
CUDA 12.2 support:

```bash
python3 -m pip install --no-cache-dir --target /tmp/cuda122 \
  nvidia-cuda-runtime-cu12==12.2.140 \
  nvidia-cuda-cupti-cu12==12.2.142
```

Use these libraries first at runtime:

```bash
export LD_LIBRARY_PATH="/tmp/cuda122/nvidia/cuda_runtime/lib:/tmp/cuda122/nvidia/cuda_cupti/lib:/usr/lib/x86_64-linux-gnu/nvidia/current:/usr/lib/x86_64-linux-gnu:${NVSHMEM_HOME}/lib:${LD_LIBRARY_PATH:-}"
export NVSHMEM_CUDA_PATH=/usr/lib/x86_64-linux-gnu/nvidia/current
```

## Compile

Task 03 needs an H100-compatible architecture flag. The local
`build_and_run.sh` supports `CUDA_ARCH` and `NVCC_FLAGS`.

```bash
cd 03_nvshmem_warp_specialization

NVSHMEM_HOME="${NVSHMEM_HOME}" \
NVCC=/usr/local/cuda-12.9/bin/nvcc \
CUDA_ARCH=sm_90 \
NVCC_FLAGS="-I/tmp/cuda122/nvidia/cuda_runtime/include -L/tmp/cuda122/nvidia/cuda_runtime/lib" \
NPES=2 \
./build_and_run.sh 1048576 64
```

Expected success output:

```text
PE 0/2: PASS (N=1048576, compute_iters=64, ...)
PE 1/2: PASS (N=1048576, compute_iters=64, ...)
```

## Runtime Environment For Task 03

Use the following environment for this worker class:

```bash
unset NVSHMEM_BOOTSTRAP

export NVSHMEM_DISABLE_CUDA_VMM=1
export NVSHMEM_SYMMETRIC_SIZE=64M
export NVSHMEM_DISABLE_NCCL=1
export NVSHMEM_DISABLE_LOCAL_ONLY_PROXY=1
export NCCL_NET_PLUGIN=none
export NCCL_P2P_LEVEL=NVL
export NVSHMEM_CUDA_PATH=/usr/lib/x86_64-linux-gnu/nvidia/current
```

Then run:

```bash
"${NVSHMEM_HOME}/bin/nvshmrun" -np 2 \
  ./build/nvshmem_warp_specialization 1048576 64
```

## Profile With Nsight Systems

Use Nsight Systems 2025.6.1 instead of the worker's older system `nsys` if the
system version cannot import `.qdstrm` into `.nsys-rep`.

```bash
export NSIGHT_SYSTEMS_HOME=/tmp/nsys_install/nsight-systems-2025.6.1
export PATH="${NSIGHT_SYSTEMS_HOME}/bin:${PATH}"

OUT=../worker_results/run_<timestamp>_03_nvshmem

nsys profile \
  -o "${OUT}/report03_nvshmem_warp_specialization" \
  --force-overwrite true \
  --trace=cuda,nvtx,osrt \
  "${NVSHMEM_HOME}/bin/nvshmrun" -np 2 \
  ./build/nvshmem_warp_specialization 1048576 64

nsys export \
  --type sqlite \
  --force-overwrite true \
  -o "${OUT}/report03_nvshmem_warp_specialization.sqlite" \
  "${OUT}/report03_nvshmem_warp_specialization.nsys-rep"
```

Verified output files:

```text
../worker_results/run_<timestamp>_03_nvshmem/report03_nvshmem_warp_specialization.nsys-rep
../worker_results/run_<timestamp>_03_nvshmem/report03_nvshmem_warp_specialization.sqlite
```

## Issues Encountered

### Missing `nvshmrun`

The pip package `nvidia-nvshmem-cu12==3.3.20` provides NVSHMEM libraries but did
not provide `nvshmrun` in this worker image. Use the NVIDIA binary archive above
when the launcher is needed.

### Compile Errors From Default GPU Architecture

Initial compilation failed with errors similar to:

```text
identifier "atomicAdd_system" is undefined
namespace "cooperative_groups" has no member "reduce_update_async"
```

The cause was compiling NVSHMEM headers for an older/default CUDA architecture.
Compile task 03 with:

```bash
CUDA_ARCH=sm_90
```

### Wrong UID Bootstrap Setting

Setting `NVSHMEM_BOOTSTRAP=UID` while the code calls plain `nvshmem_init()`
failed with:

```text
Missing init flags for bootstrap UID. Retry with nvshmemx_init_attr and non-zero flags
nvshmem_bootstrap failed
```

For this tutorial binary, leave `NVSHMEM_BOOTSTRAP` unset unless the code is
changed to use `nvshmemx_init_attr`.

### CUDA Runtime Newer Than Driver

On a worker with a bad driver library setup, task 03 failed at `cudaSetDevice`:

```text
CUDA error: CUDA driver version is insufficient for CUDA runtime version
```

That worker had CUDA 12.9 runtime first in the library path while its usable
driver stack was CUDA 12.2 era. Prefer a healthy worker where `nvidia-smi`
works, and put CUDA 12.2 runtime libraries before CUDA 12.9 runtime libraries in
`LD_LIBRARY_PATH`.

### Broken Or Mismatched NVML Libraries

One worker showed:

```text
Failed to initialize NVML: Driver/library version mismatch
```

and also had 0-byte `libcuda.so.*` / `libnvidia-ml.so.*` files in the generic
system library directory. NVSHMEM P2P detection depends on NVML, so this worker
failed transport initialization. The successful worker had valid libraries under:

```text
/usr/lib/x86_64-linux-gnu/nvidia/current
```

Set:

```bash
export NVSHMEM_CUDA_PATH=/usr/lib/x86_64-linux-gnu/nvidia/current
```

### IBRC And NCCL Plugin Warnings

Single-node 2-GPU workers may not expose active IB devices. NVSHMEM printed:

```text
WARN: init failed for remote transport: ibrc
```

This warning was non-fatal once local P2P was available.

The worker environment also injected an external NCCL plugin:

```text
NET/Plugin: Loaded net plugin FasTrak
```

For this single-node NVSHMEM example, avoid that path with:

```bash
export NVSHMEM_DISABLE_NCCL=1
export NVSHMEM_DISABLE_LOCAL_ONLY_PROXY=1
export NCCL_NET_PLUGIN=none
export NCCL_P2P_LEVEL=NVL
```

### Old System Nsight Systems Could Not Import

Worker system `nsys` was:

```text
NVIDIA Nsight Systems version 2022.4.2.1-df9881f
```

It ran the program but printed:

```text
Importer error status: The importer binary and its dependencies were not found.
Unable to retrieve the importer version: skipping importation of the QDSTRM file.
```

Only `.qdstrm` was produced. Installing Nsight Systems 2025.6.1 into `/tmp`
fixed report import and enabled `.nsys-rep` plus sqlite export.
