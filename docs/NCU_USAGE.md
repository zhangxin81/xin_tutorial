# Nsight Compute CLI setup

This note records the Nsight Compute (`ncu`) setup used for kernel-level
profiling in this tutorial. Commands below assume they are run from the
`xin_tutorial` directory or from a GPU worker that can access the tutorial
checkout.

Nsight Compute is different from Nsight Systems (see `docs/NSYS_USAGE.md`):

- `nsys` answers "what happened over time across CPU, CUDA, NCCL, NVTX, and OS
  runtime?"
- `ncu` answers "what happened inside one or a few CUDA kernels?"

Use `ncu` when you need kernel-level metrics, SASS/source views, Tensor Core
instruction evidence, TMA evidence, memory throughput, occupancy, stalls, or
roofline-style analysis.

## Download

For open-source users, start from NVIDIA's official pages instead of copying a
time-limited redirected URL:

- Nsight Compute latest download page:
  <https://developer.nvidia.com/nsight-compute>
- Nsight Compute 2025.2 page:
  <https://developer.nvidia.com/tools-overview/nsight-compute/get-started-2025_2>
- Nsight Compute 2025.2.1 download entry:
  <https://developer.nvidia.com/gameworksdownload#?dn=nsight-compute-2025-2-1>
- Nsight Compute 2025.2 documentation:
  <https://docs.nvidia.com/nsight-compute/2025.2/>

The 2025.2.1 download entry lets you choose platform packages such as Linux
Desktop, Windows, macOS, and Arm server builds. The Linux Desktop installer is
usually the easiest no-root option on a machine where you cannot use `sudo`.
NVIDIA may redirect the browser or `wget` request to a time-limited
`developer.download.nvidia.com` asset URL, so prefer saving the official entry
link in docs and scripts.

If you install CUDA through NVIDIA's package repository, Nsight Compute may also
be available as a package. On Ubuntu/Debian systems configured with the CUDA
repository, the package is commonly named `nsight-compute` or
`nsight-compute-<version>`:

```bash
apt-cache search nsight-compute
apt-cache policy nsight-compute
```

For a root install on a throwaway machine:

```bash
sudo apt-get update
sudo apt-get install nsight-compute
```

For a no-root environment, download the `.deb` package and extract it with
`dpkg-deb -x` as shown below.

One tested package version is:

```text
nsight-compute-2025.2.1_2025.2.1.3-1_amd64.deb
```

This package can be extracted without root. After extraction, the CLI is under:

```text
<extract-dir>/opt/nvidia/nsight-compute/2025.2.1/ncu
```

If your image already has Nsight Compute, first check the usual locations:

```bash
command -v ncu
find /usr/local -path '*nsight*' -o -name ncu 2>/dev/null
find /opt -path '*nsight*' -o -name ncu 2>/dev/null
```

## Install Without Root

For a Debian package, extract it into a user-writable tools directory:

```bash
mkdir -p tools/ncu-2025-deb tools/ncu-2025

dpkg-deb -x \
  tools/ncu-2025-deb/nsight-compute-2025.2.1_2025.2.1.3-1_amd64.deb \
  tools/ncu-2025

export NCU_BIN="$PWD/tools/ncu-2025/opt/nvidia/nsight-compute/2025.2.1/ncu"
"${NCU_BIN}" --version
```

For a `.run` installer, use the same no-root pattern as other NVIDIA tools:

```bash
chmod +x nsight-compute-linux.run

./nsight-compute-linux.run \
  --noprogress \
  --noexec \
  --target /tmp/ncu_extract

cd /tmp/ncu_extract
perl install-linux.pl \
  -noprompt \
  -targetpath=$HOME/tools/NVIDIA-Nsight-Compute

export NCU_BIN=$HOME/tools/NVIDIA-Nsight-Compute/ncu
"${NCU_BIN}" --version
```

A known-good version for the H100 workflow below is:

```text
NVIDIA Nsight Compute Version 2025.2.1.0
```

## Profile And Export

General pattern:

```bash
ncu --force-overwrite --target-processes all --set full \
  --export report_name \
  <command> [args...]

ncu --import report_name.ncu-rep --page raw --csv > raw.csv
ncu --import report_name.ncu-rep --page source --print-source sass > sass.txt
```

The report file `*.ncu-rep` is the binary file to open in the Nsight Compute UI.
The CSV and SASS exports are machine-readable files that can be parsed in CI or
attached to a text-only report.

## H100 GEMM Profiling Workflow

This is the workflow of task `cuda/08_h100_gemm_tile_hierarchy_ncu`, where
`scripts/profile_ncu.sh` automates the steps below. Build or choose a CUDA GEMM
program first. The command below profiles the binary
`./build/h100_cublaslt_gemm` from the task directory, which accepts GEMM shape
arguments:

```bash
export NCU_BIN=${NCU_BIN:-ncu}
mkdir -p reports

"${NCU_BIN}" \
  --force-overwrite \
  --target-processes all \
  --launch-skip 20 \
  --launch-count 1 \
  --set full \
  --export reports/h100_cublaslt_gemm \
  ./build/h100_cublaslt_gemm \
    --m=8192 --n=8192 --k=8192 \
    --warmup=20 --iters=200

"${NCU_BIN}" --import reports/h100_cublaslt_gemm.ncu-rep \
  --page raw --csv > reports/raw.csv

"${NCU_BIN}" --import reports/h100_cublaslt_gemm.ncu-rep \
  --page source --print-source sass > reports/sass.txt
```

This workflow:

1. skips warmup kernels with `--launch-skip 20`;
2. profiles only one kernel with `--launch-count 1`;
3. writes a binary `.ncu-rep` report for the GUI;
4. exports raw metrics to `raw.csv`;
5. exports SASS/source information to `sass.txt`.

Observed H100 cuBLASLt kernel name:

```text
nvjet_tss_320x128_64x3_1x2_h_bz_coopB_NNT
```

You can quickly search for Tensor Core and TMA instruction evidence:

```bash
rg -n "HGMMA|WGMMA|HMMA|MMA\.SYNC|UTMALDG|UTMASTG|UTMACMDFLUSH|CP\.ASYNC\.BULK" \
  reports/sass.txt
```

Observed evidence counts from the H100 BF16 GEMM run:

```text
TMA-related SASS lines found: 52
Tensor Core-related SASS lines found: 40
```

## Local Output Convention

Use a stable output directory:

```text
reports/
```

Keep these files:

- `h100_cublaslt_gemm.ncu-rep`: binary Nsight Compute report for GUI
  inspection.
- `raw.csv`: raw metric export from `--page raw --csv`.
- `sass.txt`: source/SASS export from `--page source --print-source sass`.
- `ncu_summary.md`: parsed evidence summary.
- `profile_stdout.txt`: `ncu` and benchmark stdout from the profiled run.
- `gpu.txt`: GPU name and compute capability.
- `ncu_version.txt`: Nsight Compute version.
- `archive/<UTC timestamp>/`: preserved copy of the NCU files for that run.

The latest files are overwritten on each run for convenience, but the archive
directory keeps prior NCU exports. A simple archive pattern is:

```bash
run_tag=$(date -u '+%Y%m%dT%H%M%SZ')
mkdir -p "reports/archive/${run_tag}"
cp -p reports/h100_cublaslt_gemm.ncu-rep \
      reports/raw.csv \
      reports/sass.txt \
      "reports/archive/${run_tag}/"
```

In this repository, only small evidence excerpts are committed (see
`cuda/08_h100_gemm_tile_hierarchy_ncu/reports/`); the full `sass.txt` and the
`archive/` copies are regenerated by `profile_ncu.sh` and not committed.

## Useful Profiling Options

Common CLI options:

```bash
# Collect the default full section set.
ncu --set full --export report_name <command>

# Profile only one kernel after skipping warmup launches.
ncu --launch-skip 20 --launch-count 1 --export report_name <command>

# Include child processes.
ncu --target-processes all --export report_name <command>

# Collect specific metrics.
ncu --metrics gpu__time_duration.avg,dram__bytes_read.sum \
  --export report_name <command>

# Filter by kernel name substring or regex when many kernels launch.
ncu --kernel-name regex:gemm --export report_name <command>
```

For benchmark workflows, profile after warmup. NCU replay and metric collection
can distort timing, so use the program's own timing for steady-state performance
and use NCU for architectural evidence and bottleneck diagnosis.

## Reading H100 GEMM Evidence

Tensor Core evidence can appear as:

```text
HGMMA.64x64x16.F32.BF16
WGMMA.*
HMMA.*
MMA.SYNC.*
```

For H100 BF16 GEMM, the important instruction often looks like:

```text
HGMMA.64x64x16.F32.BF16
```

Read it as a Hopper Tensor Core matrix instruction. It consumes BF16 fragments
and accumulates FP32 output for a `64 x 64 x 16` instruction tile.

TMA evidence can appear as:

```text
UTMALDG.3D
UTMALDG.3D.MULTICAST
UTMASTG.3D
UTMACMDFLUSH
cp.async.bulk.tensor
cp.async.bulk
```

On Hopper, exported SASS may contain `UTMA*` mnemonics. These show tensor-tile
movement between global memory and shared memory, including multicast loading
and TMA command flushes.

Warpgroup coordination can appear as:

```text
WARPGROUP.ARRIVE
WARPGROUP.WAIT
WARPGROUP.DEPBAR.LE
```

These instructions help order asynchronous warpgroup matrix work around the
HGMMA pipeline.

## Kernel Name Notes

Observed kernel name:

```text
nvjet_tss_320x128_64x3_1x2_h_bz_coopB_NNT
```

This is an internal cuBLASLt name. The most useful fields for explanation are:

- `320x128`: threadblock/CTA output tile hint.
- `64x3`: K-mainloop or staging hint.
- `h`: half-precision family; SASS confirms BF16 with FP32 accumulation.
- `coopB`: cooperative B operand staging/loading hint.
- `NNT`: layout/transpose-style suffix.

Do not depend on the exact name as a stable interface. Use it to explain what a
library kernel selected for a profiled run, and use SASS plus metrics as the actual
evidence.

## Troubleshooting

If `ncu` is missing:

```bash
command -v ncu
find /usr/local -path '*nsight*' -o -name ncu 2>/dev/null
find /opt -path '*nsight*' -o -name ncu 2>/dev/null
```

If an older `ncu` reports an unsupported CUDA driver/API error, use a newer
Nsight Compute version matching the installed driver and CUDA runtime. For H100
and recent CUDA 12.x drivers, prefer a recent Nsight Compute release such as
2025.2.1 or newer.

If SASS export is empty, open the `.ncu-rep` in the GUI and check the Source
page. Some stripped or JIT-generated kernels may need different profile/export
settings, but the binary report should still be retained.

If metrics are missing, rerun with `--set full` first, then narrow to specific
metrics after you know the metric names available on the target GPU.
