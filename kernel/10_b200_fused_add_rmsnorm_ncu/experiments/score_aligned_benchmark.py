"""Official-score B200 benchmark for SOL ExecBench kernel 69.

The primary metric is the public SOL Score formula applied per workload and
then arithmetically averaged. Local CUDA-event timings include an additive
harness bias relative to the public CUPTI pipeline; by default the script also
prints a bias-corrected score using the 2.62 us offset measured from the current
best upload. Legacy aggregate latencies are still emitted as diagnostics only.
"""

import argparse
import inspect
import importlib.util
import json
import math
import statistics
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl


ROOT = Path(__file__).resolve().parent
SHAPES = [
    (1, 8192),
    (1, 131),
    (16, 256),
    (4, 128),
    (4, 541),
    (1, 2048),
    (2, 293),
    (4, 2048),
    (8, 128),
    (8, 512),
    (4, 256),
    (2, 2048),
    (1, 4096),
    (1, 256),
    (2, 1024),
    (1, 1024),
]
EPS = 1e-5
HARNESS_BIAS_MS = 0.00262
BYTES_PER_ROW = 8192 * 2 * 3
OFFICIAL_WORKLOADS = [
    {"batch": 1, "seq": 8192, "baseline_ms": 0.0710075, "sol_ms": 0.0529},
    {"batch": 1, "seq": 131, "baseline_ms": 0.01712, "sol_ms": 0.0012},
    {"batch": 16, "seq": 256, "baseline_ms": 0.040576, "sol_ms": 0.0267},
    {"batch": 4, "seq": 128, "baseline_ms": 0.022016, "sol_ms": 0.0037},
    {"batch": 4, "seq": 541, "baseline_ms": 0.0424, "sol_ms": 0.0143},
    {"batch": 1, "seq": 2048, "baseline_ms": 0.03928, "sol_ms": 0.0135},
    {"batch": 2, "seq": 293, "baseline_ms": 0.022176, "sol_ms": 0.0042},
    {"batch": 4, "seq": 2048, "baseline_ms": 0.070656, "sol_ms": 0.0529},
    {"batch": 8, "seq": 128, "baseline_ms": 0.0271515, "sol_ms": 0.007},
    {"batch": 8, "seq": 512, "baseline_ms": 0.040448, "sol_ms": 0.0267},
    {"batch": 4, "seq": 256, "baseline_ms": 0.027216, "sol_ms": 0.007},
    {"batch": 2, "seq": 2048, "baseline_ms": 0.040545, "sol_ms": 0.0267},
    {"batch": 1, "seq": 4096, "baseline_ms": 0.04112, "sol_ms": 0.0267},
    {"batch": 1, "seq": 256, "baseline_ms": 0.0192805, "sol_ms": 0.002},
    {"batch": 2, "seq": 1024, "baseline_ms": 0.0393605, "sol_ms": 0.0135},
    {"batch": 1, "seq": 1024, "baseline_ms": 0.0271685, "sol_ms": 0.007},
]
assert [(w["batch"], w["seq"]) for w in OFFICIAL_WORKLOADS] == SHAPES
# Candidate keys match the directory numbering under experiments/. 
DEFAULT_CANDIDATES = {
    "root_current": ROOT.parent / "submission.py",
    "triton01": ROOT / "01_triton_w16_maxnreg32" / "submission.py",
    "cute02": ROOT / "02_cute_dsl_copy128" / "submission.py",
    "cuda03": ROOT / "03_cuda_graph_replay" / "submission.py",
    "cuda04": ROOT / "04_cuda_tma_bulk_smem" / "submission.py",
}


@triton.jit
def official_kernel(hidden, residual, weight, output, eps, hidden_size: tl.constexpr, block_size: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, block_size)
    mask = cols < hidden_size
    offset = row * hidden_size + cols
    hidden_bf16 = tl.load(hidden + offset, mask=mask, other=0.0)
    residual_bf16 = tl.load(residual + offset, mask=mask, other=0.0)
    x_bf16 = (hidden_bf16.to(tl.float32) + residual_bf16.to(tl.float32)).to(tl.bfloat16)
    x = x_bf16.to(tl.float32)
    inv_rms = tl.rsqrt(tl.sum(x * x, axis=0) / hidden_size + eps)
    normalized = (x * inv_rms).to(tl.bfloat16)
    w = tl.load(weight + cols, mask=mask, other=1.0)
    tl.store(output + offset, (normalized.to(tl.float32) * w.to(tl.float32)).to(tl.bfloat16), mask=mask)


def load_module(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def reference(hidden, residual, weight):
    x = residual + hidden
    x_fp32 = x.float()
    return weight * (x_fp32 * torch.rsqrt(x_fp32.square().mean(-1, keepdim=True) + EPS)).to(torch.bfloat16)


def aggregate(values):
    return {
        "sum_ms": sum(values),
        "arithmetic_mean_ms": statistics.mean(values),
        "geometric_mean_ms": math.exp(sum(math.log(v) for v in values) / len(values)),
        "harmonic_mean_ms": len(values) / sum(1.0 / v for v in values),
        "median_shape_ms": statistics.median(values),
        "per_shape_medians_ms": values,
    }


def sol_score(time_ms, baseline_ms, sol_ms):
    gap = baseline_ms - sol_ms
    if gap <= 0:
        raise ValueError(f"invalid official workload times: baseline={baseline_ms}, sol={sol_ms}")
    raw = gap / ((time_ms - sol_ms) + gap)
    return max(0.0, min(1.0, raw))


def official_score(values, bias_ms=0.0):
    scores = []
    adjusted = []
    for value, workload in zip(values, OFFICIAL_WORKLOADS):
        t = max(workload["sol_ms"], value - bias_ms)
        adjusted.append(t)
        scores.append(sol_score(t, workload["baseline_ms"], workload["sol_ms"]))
    return {
        "official_sol_score": statistics.mean(scores),
        "per_shape_official_scores": scores,
        "official_adjusted_medians_ms": adjusted,
    }


def linear_fit_latency(values, bias_ms=0.0):
    rows = [batch * seq for batch, seq in SHAPES]
    y_us = [max(0.0, value - bias_ms) * 1000.0 for value in values]
    x_mean = statistics.mean(rows)
    y_mean = statistics.mean(y_us)
    denom = sum((x - x_mean) ** 2 for x in rows)
    slope_us_per_row = sum((x - x_mean) * (y - y_mean) for x, y in zip(rows, y_us)) / denom
    intercept_us = y_mean - slope_us_per_row * x_mean
    predicted = [intercept_us + slope_us_per_row * x for x in rows]
    residuals = [y - p for y, p in zip(y_us, predicted)]
    bandwidth_tb_s = BYTES_PER_ROW / (slope_us_per_row * 1_000_000.0) if slope_us_per_row > 0 else float("inf")
    return {
        "fixed_overhead_fit": {
            "intercept_us": intercept_us,
            "slope_us_per_row": slope_us_per_row,
            "bandwidth_tb_s": bandwidth_tb_s,
            "residuals_us": residuals,
        },
        "residual_fixed_us_at_7tbps": [
            y - (row * BYTES_PER_ROW / 7_000_000.0) for row, y in zip(rows, y_us)
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configs",
        default="official,triton01,cute02",
        help="Comma-separated config names. Names must be official or keys in DEFAULT_CANDIDATES.",
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--seed", type=int, default=260)
    parser.add_argument("--no-flush-l2", action="store_false", dest="flush_l2")
    parser.add_argument("--anchor", default="cute02")
    parser.add_argument("--anchor-upload-latency-ms", type=float, default=0.018832)
    parser.add_argument("--harness-bias-us", type=float, default=HARNESS_BIAS_MS * 1000.0)
    parser.add_argument("--allow-input-mutation", action="store_true")
    parser.set_defaults(flush_l2=True)
    args = parser.parse_args()

    configs = tuple(name.strip() for name in args.configs.split(",") if name.strip())
    unknown = [name for name in configs if name != "official" and name not in DEFAULT_CANDIDATES]
    if unknown:
        raise ValueError(f"unknown configs: {unknown}")

    modules = {name: load_module(name, DEFAULT_CANDIDATES[name]) for name in configs if name != "official"}
    run_arities = {name: len(inspect.signature(module.run).parameters) for name, module in modules.items()}

    def launch(name, hidden, residual, weight, output=None):
        if name in modules:
            if run_arities[name] >= 5:
                if output is None:
                    output = torch.empty_like(hidden)
                modules[name].run(hidden, residual, weight, EPS, output)
                return output
            return modules[name].run(hidden, residual, weight, EPS)
        output = torch.empty_like(hidden)
        rows = hidden.numel() // 8192
        official_kernel[(rows,)](hidden, residual, weight, output, EPS, hidden_size=8192, block_size=8192)
        return output

    def timed(name, hidden, residual, weight, cache):
        output = torch.empty_like(hidden) if name in run_arities and run_arities[name] >= 5 else None
        for _ in range(args.warmup):
            if cache is not None:
                cache.zero_()
            launch(name, hidden, residual, weight, output)
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(args.repeats)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(args.repeats)]
        for start, end in zip(starts, ends):
            if cache is not None:
                cache.zero_()
            start.record()
            launch(name, hidden, residual, weight, output)
            end.record()
        torch.cuda.synchronize()
        return statistics.median(start.elapsed_time(end) for start, end in zip(starts, ends))

    assert torch.cuda.get_device_capability() == (10, 0)
    torch.manual_seed(args.seed)
    cache = None
    if args.flush_l2:
        cache = torch.empty(torch.cuda.get_device_properties(0).L2_cache_size * 2, dtype=torch.int8, device="cuda")

    per_shape = {name: [] for name in configs}
    round_totals = {name: [0.0] * args.rounds for name in configs}
    passed = {name: 0 for name in configs}

    print(
        "CONTRACT "
        + json.dumps(
            {
                "hidden_size": 8192,
                "dtype": "bfloat16",
                "warmup": args.warmup,
                "repeats": args.repeats,
                "rounds": args.rounds,
                "configs": configs,
                "flush_l2": args.flush_l2,
                "aggregate_primary": "official_sol_score_bias_corrected",
                "anchor": args.anchor,
                "anchor_upload_latency_ms": args.anchor_upload_latency_ms,
                "harness_bias_us": args.harness_bias_us,
                "official_workloads": OFFICIAL_WORKLOADS,
            }
        ),
        flush=True,
    )

    for shape_index, (batch, seq) in enumerate(SHAPES):
        shape = (batch, seq, 8192)
        hidden = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
        residual = torch.randn_like(hidden)
        residual_before = residual.clone()
        weight = torch.randn(8192, dtype=torch.bfloat16, device="cuda")
        expected = reference(hidden, residual, weight)
        correctness = {}

        for name in configs:
            actual = launch(name, hidden, residual, weight)
            torch.cuda.synchronize()
            diff = (actual.float() - expected.float()).abs()
            matched = (diff <= 1e-2 + 1e-2 * expected.float().abs()).float().mean().item()
            residual_unchanged = bool(torch.equal(residual, residual_before))
            correctness[name] = {
                "matched_ratio": matched,
                "max_abs": diff.max().item(),
                "residual_unchanged": residual_unchanged,
            }
            assert matched >= 0.99 and (args.allow_input_mutation or residual_unchanged) and not torch.isnan(actual).any(), (
                name,
                shape,
                correctness[name],
            )
            passed[name] += 1

        samples = {name: [0.0] * args.rounds for name in configs}
        for round_index in range(args.rounds):
            offset = (shape_index + round_index) % len(configs)
            order = configs[offset:] + configs[:offset]
            for name in order:
                value = timed(name, hidden, residual, weight, cache)
                samples[name][round_index] = value
                round_totals[name][round_index] += value

        medians = {name: statistics.median(samples[name]) for name in configs}
        for name in configs:
            per_shape[name].append(medians[name])
        winner = min(medians, key=medians.get)
        print(
            "WORKLOAD "
            + json.dumps(
                {
                    "batch": batch,
                    "seq": seq,
                    "rows": batch * seq,
                    "correctness": correctness,
                    "samples_ms": samples,
                    "medians_ms": medians,
                    "winner": winner,
                }
            ),
            flush=True,
        )
        del hidden, residual, residual_before, weight, expected
        torch.cuda.empty_cache()

    summary = {}
    for name in configs:
        values = per_shape[name]
        item = aggregate(values)
        item.update(official_score(values, bias_ms=0.0))
        biased = official_score(values, bias_ms=args.harness_bias_us / 1000.0)
        item["official_sol_score_bias_corrected"] = biased["official_sol_score"]
        item["per_shape_official_scores_bias_corrected"] = biased["per_shape_official_scores"]
        item["official_bias_corrected_medians_ms"] = biased["official_adjusted_medians_ms"]
        item.update(linear_fit_latency(values, bias_ms=args.harness_bias_us / 1000.0))
        item["passed"] = passed[name]
        item["round_sum_ms"] = round_totals[name]
        item["median_round_sum_ms"] = statistics.median(round_totals[name])
        summary[name] = item

    if args.anchor in summary:
        scale = args.anchor_upload_latency_ms / summary[args.anchor]["harmonic_mean_ms"]
        for item in summary.values():
            item["calibrated_latency_ms"] = item["harmonic_mean_ms"] * scale
        summary["_calibration"] = {
            "anchor": args.anchor,
            "anchor_local_harmonic_mean_ms": summary[args.anchor]["harmonic_mean_ms"],
            "anchor_upload_latency_ms": args.anchor_upload_latency_ms,
            "scale": scale,
        }

    if args.anchor in summary:
        anchor_harm = summary[args.anchor]["harmonic_mean_ms"]
        for name, item in summary.items():
            if not name.startswith("_"):
                item["speedup_vs_anchor_harmonic"] = anchor_harm / item["harmonic_mean_ms"]

    print("SCORE_SUMMARY " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
