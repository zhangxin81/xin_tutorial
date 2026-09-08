#!/usr/bin/env python3
"""GPU concurrency benchmark for small-model / large-GPU experiments.

Modes:
  baseline      one host worker + default CUDA stream
  multistream   one process + N host workers + N CUDA streams
  multiprocess  N spawned processes on one GPU; start NVIDIA MPS externally

The script reports end-to-end latency from request submission to completion.
It intentionally does not hide queueing time. Use Nsight Systems/Compute for
kernel overlap and SM-resource evidence.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import queue
import random
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--mode", choices=["baseline", "multistream", "multiprocess"], required=True)
    p.add_argument("--model", required=True, help="Hugging Face model path or model id")
    p.add_argument("--task", choices=["encoder", "causal_lm_forward", "generate"], default="encoder")
    p.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="bf16")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--new-tokens", type=int, default=1)
    p.add_argument("--requests", type=int, default=200)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--concurrency", type=int, default=1, help="streams for multistream; processes for multiprocess")
    p.add_argument("--arrival", choices=["closed_loop", "burst", "poisson"], default="closed_loop")
    p.add_argument("--qps", type=float, default=100.0, help="mean arrival rate for poisson mode")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--compile", action="store_true", help="apply torch.compile; keep shapes static")
    p.add_argument("--consistency-check", action="store_true",
                   help="after the measured run, compare concurrent outputs across streams against a serial reference (baseline/multistream only)")
    p.add_argument("--output", default="result.json")
    return p.parse_args()


def percentile(values: List[float], q: float) -> float:
    if not values:
        return math.nan
    xs = sorted(values)
    rank = (len(xs) - 1) * q
    lo, hi = math.floor(rank), math.ceil(rank)
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - rank) + xs[hi] * (rank - lo)


def dtype_from_name(torch: Any, name: str) -> Any:
    return {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[name]


def load_model_and_input(args: argparse.Namespace) -> Tuple[Any, Dict[str, Any]]:
    import torch
    from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

    torch.cuda.set_device(args.device)
    dtype = dtype_from_name(torch, args.dtype)
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    vocab_size = int(getattr(config, "vocab_size", 32000))

    cls = AutoModel if args.task == "encoder" else AutoModelForCausalLM
    model = cls.from_pretrained(
        args.model,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
        low_cpu_mem_usage=True,
    ).eval().to(args.device)

    if args.compile:
        model = torch.compile(model, mode="reduce-overhead", fullgraph=False)

    g = torch.Generator(device=args.device)
    g.manual_seed(args.seed)
    input_ids = torch.randint(
        low=0,
        high=max(2, vocab_size),
        size=(args.batch_size, args.seq_len),
        generator=g,
        device=args.device,
        dtype=torch.long,
    )
    attention_mask = torch.ones_like(input_ids)
    torch.cuda.synchronize(args.device)
    return model, {"input_ids": input_ids, "attention_mask": attention_mask}


def run_model(model: Any, inputs: Dict[str, Any], args: argparse.Namespace) -> Any:
    if args.task == "generate":
        return model.generate(
            **inputs,
            max_new_tokens=args.new_tokens,
            min_new_tokens=args.new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=getattr(model.config, "eos_token_id", 0),
        )
    return model(**inputs, use_cache=False) if args.task == "causal_lm_forward" else model(**inputs)


def warmup(model: Any, inputs: Dict[str, Any], args: argparse.Namespace) -> None:
    import torch

    with torch.inference_mode():
        for _ in range(args.warmup):
            _ = run_model(model, inputs, args)
    torch.cuda.synchronize(args.device)


@dataclass
class Result:
    mode: str
    requests: int
    concurrency: int
    wall_s: float
    throughput_rps: float
    latency_ms_p50: float
    latency_ms_p95: float
    latency_ms_p99: float
    latency_ms_max: float
    latency_ms_mean: float
    gpu_service_ms_p50: float
    gpu_service_ms_p99: float
    metadata: Dict[str, Any]


def summarize(args: argparse.Namespace, lat_ms: List[float], gpu_ms: List[float], wall_s: float) -> Result:
    import torch

    props = torch.cuda.get_device_properties(args.device)
    metadata = {
        "model": args.model,
        "task": args.task,
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "new_tokens": args.new_tokens,
        "arrival": args.arrival,
        "qps": args.qps,
        "compile": args.compile,
        "gpu_name": props.name,
        "gpu_total_memory_bytes": props.total_memory,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "mps_pipe_directory": os.getenv("CUDA_MPS_PIPE_DIRECTORY"),
        "mps_active_thread_percentage": os.getenv("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"),
    }
    return Result(
        mode=args.mode,
        requests=len(lat_ms),
        concurrency=args.concurrency,
        wall_s=wall_s,
        throughput_rps=len(lat_ms) / wall_s,
        latency_ms_p50=percentile(lat_ms, 0.50),
        latency_ms_p95=percentile(lat_ms, 0.95),
        latency_ms_p99=percentile(lat_ms, 0.99),
        latency_ms_max=max(lat_ms),
        latency_ms_mean=statistics.fmean(lat_ms),
        gpu_service_ms_p50=percentile(gpu_ms, 0.50),
        gpu_service_ms_p99=percentile(gpu_ms, 0.99),
        metadata=metadata,
    )


def _iter_tensors(obj: Any):
    """Yield every tensor inside a model output (ModelOutput/dict/tuple/tree)."""
    import torch

    if torch.is_tensor(obj):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _iter_tensors(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _iter_tensors(v)


def consistency_check(model: Any, inputs: Dict[str, Any], args: argparse.Namespace) -> Dict[str, float]:
    """Re-run `concurrency` requests truly in parallel across streams and
    compare every output tensor against a serial reference run.

    Read-only inference should reproduce the serial result up to kernel
    nondeterminism; a structural gap means the model (or a custom op) keeps
    mutable internal state and is not safe for the multistream mode.
    Non-floating outputs (e.g. generated token ids) contribute the number of
    mismatched elements to max_abs_error.
    """
    import torch

    with torch.inference_mode():
        refs = [t.detach().clone() for t in _iter_tensors(run_model(model, inputs, args))]
    torch.cuda.synchronize(args.device)

    stream_count = max(1, args.concurrency)
    streams = [torch.cuda.Stream(device=args.device) for _ in range(stream_count)]
    events = [torch.cuda.Event() for _ in range(stream_count)]
    barrier = threading.Barrier(stream_count)
    outputs: List[Any] = [None] * stream_count

    def one(idx: int) -> None:
        with torch.cuda.stream(streams[idx]), torch.inference_mode():
            barrier.wait()
            outputs[idx] = run_model(model, inputs, args)
            events[idx].record(streams[idx])

    threads = [threading.Thread(target=one, args=(i,)) for i in range(stream_count)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for e in events:
        e.synchronize()

    max_abs = 0.0
    max_rel = 0.0
    for out in outputs:
        for got, ref in zip(_iter_tensors(out), refs):
            if got.dtype.is_floating_point:
                diff = (got.float() - ref.float()).abs().max()
                denom = ref.float().abs().max()
                rel = float(diff / denom) if float(denom) > 0 else 0.0
            else:
                diff = torch.tensor(float((got != ref).sum()))
                rel = 0.0
            max_abs = max(max_abs, float(diff))
            max_rel = max(max_rel, rel)
    print(f"consistency: streams={stream_count} max_abs_error={max_abs:.3e} max_rel_error={max_rel:.3e}")
    return {"consistency_max_abs_error": max_abs, "consistency_max_rel_error": max_rel}


def thread_mode(args: argparse.Namespace) -> Result:
    import torch

    model, inputs = load_model_and_input(args)
    warmup(model, inputs, args)

    stream_count = 1 if args.mode == "baseline" else args.concurrency
    streams = [torch.cuda.default_stream(args.device)] if stream_count == 1 else [
        torch.cuda.Stream(device=args.device) for _ in range(stream_count)
    ]
    available: "queue.Queue[int]" = queue.Queue()
    for i in range(stream_count):
        available.put(i)

    def one_request(submitted_at: float, request_id: int) -> Tuple[float, float]:
        idx = available.get()
        stream = streams[idx]
        try:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            torch.cuda.nvtx.range_push(f"request_{request_id}_stream_{idx}")
            with torch.cuda.stream(stream), torch.inference_mode():
                start.record(stream)
                output = run_model(model, inputs, args)
                end.record(stream)
            end.synchronize()
            del output
            torch.cuda.nvtx.range_pop()
            completed_at = time.perf_counter()
            return (completed_at - submitted_at) * 1000.0, start.elapsed_time(end)
        finally:
            available.put(idx)

    workers = 1 if args.mode == "baseline" else args.concurrency
    lat_ms: List[float] = []
    gpu_ms: List[float] = []
    t0 = time.perf_counter()

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="cuda-request") as ex:
        if args.arrival == "closed_loop":
            # Keep at most `workers` requests in flight.
            pending = {}
            next_id = 0
            for _ in range(min(workers, args.requests)):
                submitted = time.perf_counter()
                pending[ex.submit(one_request, submitted, next_id)] = next_id
                next_id += 1
            while pending:
                future = next(as_completed(pending))
                pending.pop(future)
                e2e, service = future.result()
                lat_ms.append(e2e)
                gpu_ms.append(service)
                if next_id < args.requests:
                    submitted = time.perf_counter()
                    pending[ex.submit(one_request, submitted, next_id)] = next_id
                    next_id += 1
        else:
            futures = []
            for request_id in range(args.requests):
                submitted = time.perf_counter()
                futures.append(ex.submit(one_request, submitted, request_id))
                if args.arrival == "poisson" and request_id + 1 < args.requests:
                    time.sleep(random.expovariate(args.qps))
            for future in as_completed(futures):
                e2e, service = future.result()
                lat_ms.append(e2e)
                gpu_ms.append(service)

    torch.cuda.synchronize(args.device)
    result = summarize(args, lat_ms, gpu_ms, time.perf_counter() - t0)
    if args.consistency_check:
        result.metadata.update(consistency_check(model, inputs, args))
    return result


def child_process(rank: int, args_dict: Dict[str, Any], barrier: Any, result_queue: Any) -> None:
    try:
        args = argparse.Namespace(**args_dict)
        import torch

        torch.cuda.set_device(args.device)
        model, inputs = load_model_and_input(args)
        warmup(model, inputs, args)
        count = args.requests // args.concurrency + (1 if rank < args.requests % args.concurrency else 0)
        barrier.wait()
        lat_ms, gpu_ms = [], []
        for request_id in range(count):
            submitted = time.perf_counter()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            torch.cuda.nvtx.range_push(f"proc_{rank}_request_{request_id}")
            with torch.inference_mode():
                start.record()
                output = run_model(model, inputs, args)
                end.record()
            end.synchronize()
            del output
            torch.cuda.nvtx.range_pop()
            lat_ms.append((time.perf_counter() - submitted) * 1000.0)
            gpu_ms.append(start.elapsed_time(end))
        result_queue.put({"rank": rank, "lat_ms": lat_ms, "gpu_ms": gpu_ms})
    except BaseException as exc:
        result_queue.put({"rank": rank, "error": repr(exc)})


def multiprocess_mode(args: argparse.Namespace) -> Result:
    if args.arrival != "closed_loop":
        raise ValueError("multiprocess mode currently supports --arrival closed_loop only")
    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(args.concurrency + 1)
    result_queue = ctx.Queue()
    args_dict = vars(args).copy()
    procs = [ctx.Process(target=child_process, args=(rank, args_dict, barrier, result_queue)) for rank in range(args.concurrency)]
    for p in procs:
        p.start()
    barrier.wait()  # all models are loaded and warmed up
    t0 = time.perf_counter()
    messages = [result_queue.get() for _ in procs]
    wall_s = time.perf_counter() - t0
    for p in procs:
        p.join()
    errors = [m for m in messages if "error" in m]
    if errors:
        raise RuntimeError(f"child process failures: {errors}")
    lat_ms = [x for m in messages for x in m["lat_ms"]]
    gpu_ms = [x for m in messages for x in m["gpu_ms"]]
    return summarize(args, lat_ms, gpu_ms, wall_s)


def main() -> None:
    args = parse_args()
    if args.concurrency < 1 or args.requests < 1:
        raise ValueError("concurrency and requests must be positive")
    random.seed(args.seed)
    if args.mode == "multiprocess" and args.consistency_check:
        print("note: --consistency-check applies to baseline/multistream; each "
              "multiprocess child loads its own model, skipping the check")
        args.consistency_check = False
    result = multiprocess_mode(args) if args.mode == "multiprocess" else thread_mode(args)
    payload = asdict(result)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
