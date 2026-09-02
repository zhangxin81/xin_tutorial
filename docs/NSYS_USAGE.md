# Nsight Systems CLI setup

This note records the `nsys` setup used for profiling this CUDA/NVSHMEM
tutorial. Commands below assume they are run from the `xin_tutorial` directory
or from a GPU worker that can access the tutorial checkout.

## Download

The tested installer URL is:

```bash
wget -O NsightSystems-linux-public-2025.6.1.190-3689520.run \
  https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2025_6/NsightSystems-linux-public-2025.6.1.190-3689520.run
```

The file is about 387 MiB. NVIDIA redirects this URL to a time-limited
`developer.download.nvidia.com` URL.

## Install Without Root

Install into a shared user-writable directory. The `.run` file is a makeself
archive; use `--noexec` first, then run the internal installer with `-noprompt`
so the EULA and install path prompts do not block a worker script:

```bash
chmod +x NsightSystems-linux-public-2025.6.1.190-3689520.run

./NsightSystems-linux-public-2025.6.1.190-3689520.run \
  --noprogress \
  --noexec \
  --target nsight_extract

perl nsight_extract/install-linux.pl \
  -targetpath=<tutorial-root>/tools/nsight-systems-2025.6.1 \
  -noprompt
```

Add the CLI to `PATH`:

```bash
export NSIGHT_SYSTEMS_HOME=<tutorial-root>/tools/nsight-systems-2025.6.1
export PATH="${NSIGHT_SYSTEMS_HOME}/bin:${PATH}"
```

For a worker-local install, use the same pattern with a `/tmp` target:

```bash
./NsightSystems-linux-public-2025.6.1.190-3689520.run \
  --noexec \
  --target /tmp/nsys_extract \
  --noprogress

cd /tmp/nsys_extract
perl install-linux.pl \
  -noprompt \
  -targetpath=/tmp/nsys_install/nsight-systems-2025.6.1

export NSIGHT_SYSTEMS_HOME=/tmp/nsys_install/nsight-systems-2025.6.1
export PATH="${NSIGHT_SYSTEMS_HOME}/bin:${PATH}"
```

Verify:

```bash
nsys --version
```

The verified version in this workspace is:

```text
NVIDIA Nsight Systems version 2025.6.1.190-256136895201v0
```

## Profile And Export

General pattern:

```bash
nsys profile -o report_name --force-overwrite true <command> [args...]
nsys export --type sqlite --force-overwrite true -o report_name.sqlite report_name.nsys-rep
```

Examples used by this tutorial:

```bash
# Task 01, PyTorch/NCCL. The NCCL environment avoids the worker's external
# gcp-fastrak plugin for single-node runs.
env NCCL_NET_PLUGIN=none NCCL_P2P_LEVEL=NVL TORCH_DISTRIBUTED_DEBUG=DETAIL \
  nsys profile -o report01_stream_overlap --force-overwrite true \
  torchrun --standalone --nproc-per-node=2 01_stream_overlap.py \
    --rows-per-rank 512 --hidden 512 --warmup-iters 1 --iters 2

nsys export --type sqlite --force-overwrite true \
  -o report01_stream_overlap.sqlite report01_stream_overlap.nsys-rep

# Task 03, NVSHMEM.
export NVSHMEM_HOME=/tmp/nvshmem_install/libnvshmem-linux-x86_64-3.1.7_cuda12-archive
export LD_LIBRARY_PATH="/tmp/cuda122/nvidia/cuda_runtime/lib:/tmp/cuda122/nvidia/cuda_cupti/lib:/usr/lib/x86_64-linux-gnu/nvidia/current:/usr/lib/x86_64-linux-gnu:${NVSHMEM_HOME}/lib:${LD_LIBRARY_PATH:-}"
export NVSHMEM_CUDA_PATH=/usr/lib/x86_64-linux-gnu/nvidia/current
export NVSHMEM_DISABLE_CUDA_VMM=1
export NVSHMEM_SYMMETRIC_SIZE=64M
export NVSHMEM_DISABLE_NCCL=1
export NVSHMEM_DISABLE_LOCAL_ONLY_PROXY=1
export NCCL_NET_PLUGIN=none
export NCCL_P2P_LEVEL=NVL
unset NVSHMEM_BOOTSTRAP

nsys profile -o report03_nvshmem_warp_specialization \
  --force-overwrite true \
  --trace=cuda,nvtx,osrt \
  "${NVSHMEM_HOME}/bin/nvshmrun" -np 2 \
  ./build/nvshmem_warp_specialization 1048576 64

nsys export --type sqlite --force-overwrite true \
  -o report03_nvshmem_warp_specialization.sqlite \
  report03_nvshmem_warp_specialization.nsys-rep
```

## Local Output Convention

This tutorial stores profiling results under:

```text
../worker_results/<run_name>/
```

For each profiled task, keep:

- `*.log`: normal program output.
- `*.nsys.log`: `nsys profile` and `nsys export` command output.
- `*.nsys-rep`: Nsight Systems report for GUI inspection.
- `*.sqlite`: exported machine-readable report.

