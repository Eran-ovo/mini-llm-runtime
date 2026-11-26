#!/usr/bin/env python3
"""Measure mini-llm-runtime and vLLM on identical pretokenized bursts and arrival traces."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from contextlib import ExitStack, contextmanager, redirect_stdout
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path

import torch


@dataclass(frozen=True)
class RequestSpec:
    input_length: int
    output_length: int
    arrival_ms: int = 0


@dataclass(frozen=True)
class Workload:
    requests: tuple[RequestSpec, ...]
    shape: tuple[int, int, int] | None = None

    def describe(self) -> dict:
        if self.shape is not None:
            return {"shape": self.shape}
        return {"name": "late-ragged-prefill", "requests": [asdict(spec) for spec in self.requests]}


def make_cases(shapes: list[tuple[int, int, int]], mixed: bool) -> list[Workload]:
    cases = [
        Workload((RequestSpec(inputs, outputs),) * count, (count, inputs, outputs))
        for count, inputs, outputs in shapes
    ]
    if mixed:
        cases.append(Workload(
            (RequestSpec(64, 48),) * 4
            + tuple(RequestSpec(length, 32, 60) for length in (128, 512, 1024))
        ))
    return cases


def parse_case(value: str) -> tuple[int, int, int]:
    try:
        sizes = tuple(int(part) for part in value.split("x"))
        if len(sizes) == 3 and min(sizes) > 0:
            return sizes
    except ValueError:
        pass
    raise argparse.ArgumentTypeError("expected REQUESTSxINPUT_TOKENSxOUTPUT_TOKENS")


def percentile(samples: list[float], quantile: float) -> float:
    ordered = sorted(samples)
    rank = (len(ordered) - 1) * quantile
    lower = math.floor(rank)
    upper = math.ceil(rank)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)


def benchmark_metadata(model: Path, args: argparse.Namespace) -> dict:
    repo = Path(__file__).resolve().parents[1]

    def get(*command: str) -> str | None:
        try:
            return subprocess.check_output(command, cwd=repo, stderr=subprocess.DEVNULL, text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    dirty_status = get("git", "status", "--porcelain")
    return {
        "model": str(model),
        "seed": args.seed,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "max_running_requests": args.max_running_requests,
        "max_batch_tokens": args.max_batch_tokens,
        "block_size": 16,
        "kv_blocks_override": args.kv_blocks,
        "ttft_slo_ms": args.ttft_slo_ms,
        "itl_slo_ms": args.itl_slo_ms,
        "git_commit": get("git", "rev-parse", "HEAD"),
        "git_dirty": bool(dirty_status) if dirty_status is not None else None,
        "gpu_snapshot": get(
            "nvidia-smi", "--query-gpu=temperature.gpu,clocks.sm,power.draw",
            "--format=csv,noheader,nounits",
        ),
    }


def measure(submit, step, workload: Workload, prompts: list[list[int]], args: argparse.Namespace) -> dict:
    tokens: list[list[int]] = [[] for _ in prompts]
    timestamps: list[list[int]] = [[] for _ in prompts]
    submitted_at = []

    started = time.perf_counter_ns()
    arrivals = [started + spec.arrival_ms * 1_000_000 for spec in workload.requests]
    initial = [i for i, spec in enumerate(workload.requests) if spec.arrival_ms == 0]
    late_during_initial_decode = 0
    next_request = active = finished_count = 0
    while finished_count < len(prompts):
        now = time.perf_counter_ns()
        while next_request < len(prompts) and arrivals[next_request] <= now:
            spec = workload.requests[next_request]
            if spec.arrival_ms and any(
                0 < len(tokens[i]) < workload.requests[i].output_length for i in initial
            ):
                late_during_initial_decode += 1
            submitted_at.append(time.perf_counter_ns())
            submit(str(next_request), prompts[next_request], spec.output_length)
            next_request += 1
            active += 1
        if not active:
            time.sleep(max(0, arrivals[next_request] - time.perf_counter_ns()) / 1e9)
            continue
        emitted = step()
        ready = time.perf_counter_ns()
        for request_id, delta, finished in emitted:
            index = int(request_id)
            tokens[index].extend(delta)
            timestamps[index].extend([ready] * len(delta))
            if finished:
                if len(tokens[index]) != workload.requests[index].output_length:
                    raise RuntimeError(f"unexpected number of tokens for model request {request_id}")
                finished_count += 1
                active -= 1

    elapsed_ns = max(events[-1] for events in timestamps) - started
    ttft_ms = [(events[0] - arrival) / 1e6 for events, arrival in zip(timestamps, arrivals, strict=True)]
    tpot_ms = [(events[-1] - events[0]) / ((len(events) - 1) * 1e6)
               for events in timestamps if len(events) > 1]
    request_gaps = [
        [(right - left) / 1e6 for left, right in zip(events, events[1:])]
        for events in timestamps
    ]
    itl_ms = [gap for gaps in request_gaps for gap in gaps]
    sample = {
        "output_tokens_per_second": sum(map(len, tokens)) * 1e9 / elapsed_ns,
        "ttft_p50_ms": percentile(ttft_ms, 0.50),
        "ttft_p95_ms": percentile(ttft_ms, 0.95),
        "tpot_p50_ms": percentile(tpot_ms, 0.50) if tpot_ms else None,
        "tpot_p95_ms": percentile(tpot_ms, 0.95) if tpot_ms else None,
        "itl_p95_ms": percentile(itl_ms, 0.95) if itl_ms else None,
        "tokens_sha256": hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
    }
    if workload.shape is None:
        late_ttft = [latency for latency, spec in zip(ttft_ms, workload.requests, strict=True)
                     if spec.arrival_ms > 0]
        initial_gaps = [gap for gaps, spec in zip(request_gaps, workload.requests, strict=True)
                        if spec.arrival_ms == 0 for gap in gaps]
        sample.update(
            late_ttft_p95_ms=percentile(late_ttft, 0.95),
            initial_itl_p95_ms=percentile(initial_gaps, 0.95),
            initial_itl_max_ms=max(initial_gaps),
            late_requests_during_initial_decode=late_during_initial_decode,
            dispatch_lag_p95_ms=percentile(
                [(sent - arrival) / 1e6 for sent, arrival in zip(submitted_at, arrivals, strict=True)],
                0.95,
            ),
        )
    if args.ttft_slo_ms is not None:
        within_slo = sum(
            ttft <= args.ttft_slo_ms and max(gaps, default=0) <= args.itl_slo_ms
            for ttft, gaps in zip(ttft_ms, request_gaps, strict=True)
        )
        sample["goodput_requests_per_second"] = within_slo * 1e9 / elapsed_ns
    if args.record_tokens:
        sample["output_token_ids"] = tokens
    return sample


def load_mini(model: Path, args: argparse.Namespace):
    from mini_llm_runtime.block_admission import PagedBlockAdmissionController
    from mini_llm_runtime.engine import ContinuousBatchEngine
    from mini_llm_runtime.paged_kv_manager import PagedKVCacheManager
    from mini_llm_runtime.qwen_loader import load_qwen_checkpoint
    from mini_llm_runtime.qwen_model_runner import QwenPrefillRunner
    from mini_llm_runtime.scheduler import RequestScheduler

    config, weights = load_qwen_checkpoint(model, device="cuda:0", dtype=torch.float16)
    runner_options = {"decode_attention_backend": "paged_cuda"}
    if args.mini_cuda_graph:
        runner_options["enable_decode_cuda_graph"] = True
    if args.mini_prefill_cuda_graph:
        runner_options["enable_prefill_cuda_graph"] = True
    runner = QwenPrefillRunner(config, weights, **runner_options)
    block_size = 16
    max_blocks = args.kv_blocks or max(
        sum(sorted(
            (math.ceil((spec.input_length + spec.output_length - 1) / block_size)
             for spec in workload.requests),
            reverse=True,
        )[:args.max_running_requests])
        for workload in args.workloads
    )

    manager = PagedKVCacheManager(
        total_blocks=max_blocks,
        block_size=block_size,
        num_layers=config.num_hidden_layers,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        dtype=torch.float16,
        device=weights.embedding.device,
    )

    def new_trial(prefill_backend):
        if manager.allocator.free_count != max_blocks:
            raise RuntimeError("previous mini trial left KV blocks allocated")
        admission = PagedBlockAdmissionController(manager)
        scheduler = RequestScheduler(
            max_running_requests=args.max_running_requests,
            max_batch_tokens=args.max_batch_tokens,
            admission_callback=admission.try_admit,
        )
        engine = ContinuousBatchEngine(
            scheduler=scheduler, runner=runner, admission=admission,
            prefill_attention_backend=prefill_backend,
        )

        def submit(request_id, prompt, output_length):
            engine.submit(request_id, prompt, max_new_tokens=output_length)

        def step():
            result = engine.step()
            if result is None:
                raise RuntimeError("mini engine has unfinished requests and made no progress")
            finished = set(result.update.finished_request_ids)
            return [
                (request_id, (token,), request_id in finished)
                for request_id, token in result.update.emitted_tokens
            ]

        return submit, step

    return new_trial, {
        "kv_blocks": max_blocks,
        "mini_cuda_graph": args.mini_cuda_graph,
        "mini_prefill_cuda_graph": args.mini_prefill_cuda_graph,
    }


def load_vllm(model: Path, args: argparse.Namespace):
    import vllm
    from vllm import EngineArgs, LLMEngine, SamplingParams
    from vllm.sampling_params import RequestOutputKind

    engine = LLMEngine.from_engine_args(
        EngineArgs(
            model=str(model),
            dtype="float16",
            skip_tokenizer_init=True,
            generation_config="vllm",
            max_model_len=max(
                spec.input_length + spec.output_length
                for workload in args.workloads for spec in workload.requests
            ),
            max_num_seqs=args.max_running_requests,
            max_num_batched_tokens=args.max_batch_tokens,
            block_size=16,
            gpu_memory_utilization=args.gpu_memory_utilization,
            num_gpu_blocks_override=args.kv_blocks,
            enable_prefix_caching=False,
            disable_log_stats=True,
            enforce_eager=args.vllm_eager,
        )
    )
    next_trial = 0

    def new_trial(_prefill_backend):
        nonlocal next_trial
        prefix = f"{next_trial}-"
        next_trial += 1

        def submit(request_id, prompt, output_length):
            params = SamplingParams(
                temperature=0.0,
                max_tokens=output_length,
                ignore_eos=True,
                detokenize=False,
                output_kind=RequestOutputKind.DELTA,
            )
            engine.add_request(prefix + request_id, {"prompt_token_ids": prompt}, params)

        def step():
            if not engine.has_unfinished_requests():
                raise RuntimeError("vLLM stopped before the trial reached its token budget")
            return [
                (output.request_id.removeprefix(prefix), output.outputs[0].token_ids, output.finished)
                for output in engine.step()
            ]

        return submit, step

    return new_trial, {
        "vllm_version": vllm.__version__,
        "vllm_engine": type(engine).__module__,
        "vllm_eager": args.vllm_eager,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "kv_blocks_override": args.kv_blocks,
    }


def make_workloads(model: Path, args: argparse.Namespace):
    # Keep input IDs within the low, non-special part of the Qwen vocabulary.
    vocab = min(1000, json.loads((model / "config.json").read_text())["vocab_size"])
    workloads = []
    for workload in args.workloads:
        rng = random.Random(args.seed)
        workloads.append([[rng.randrange(3, vocab) for _ in range(spec.input_length)]
                          for spec in workload.requests])
    return workloads


def summarize(samples: list[dict]) -> dict:
    return {
        name: percentile([sample[name] for sample in samples], 0.50)
        for name, value in samples[0].items()
        if isinstance(value, (int, float))
    }


def serve_trials(model: Path, args: argparse.Namespace) -> None:
    """Run one backend over a local line-based protocol, outside the timing window."""
    workloads = make_workloads(model, args)
    with redirect_stdout(sys.stderr):
        load = load_mini if args.backend == "mini" else load_vllm
        new_trial, settings = load(model, args)
    print(json.dumps({
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        **settings,
    }), flush=True)
    for line in sys.stdin:
        index, prefill_backend = json.loads(line)
        with redirect_stdout(sys.stderr):
            sample = measure(*new_trial(prefill_backend), args.workloads[index], workloads[index], args)
        print(json.dumps(sample), flush=True)


@contextmanager
def trial_worker(interpreter: str, backend: str, args: argparse.Namespace):
    command = [
        interpreter, str(Path(__file__).resolve()), "--worker", "--backend", backend,
        "--model", str(args.model.resolve()), "--output", str(args.output),
        "--max-running-requests", str(args.max_running_requests),
        "--max-batch-tokens", str(args.max_batch_tokens),
        "--seed", str(args.seed), "--gpu-memory-utilization", str(args.gpu_memory_utilization),
    ]
    for shape in args.case:
        command.extend(("--case", "x".join(map(str, shape))))
    if args.kv_blocks is not None:
        command.extend(("--kv-blocks", str(args.kv_blocks)))
    if args.ttft_slo_ms is not None:
        command.extend(("--ttft-slo-ms", str(args.ttft_slo_ms), "--itl-slo-ms", str(args.itl_slo_ms)))
    for enabled, flag in (
        (args.mini_cuda_graph, "--mini-cuda-graph"),
        (args.mini_prefill_cuda_graph, "--mini-prefill-cuda-graph"),
        (args.vllm_eager, "--vllm-eager"),
        (args.mixed, "--mixed"),
        (args.record_tokens, "--record-tokens"),
    ):
        if enabled:
            command.append(flag)
    env = dict(os.environ)
    env.update(VLLM_USE_V1="1", VLLM_ENABLE_V1_MULTIPROCESSING="0", VLLM_NO_USAGE_STATS="1")
    process = subprocess.Popen(
        command, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
    )
    assert process.stdin is not None and process.stdout is not None

    def receive():
        line = process.stdout.readline()
        if not line:
            raise RuntimeError(f"{backend} benchmark worker terminated")
        return json.loads(line)

    def run(index: int, prefill_backend: str = "auto"):
        process.stdin.write(json.dumps((index, prefill_backend)) + "\n")
        process.stdin.flush()
        return receive()

    try:
        yield run, receive()
    finally:
        process.stdin.close()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.terminate()
            process.wait(timeout=10)
        process.stdout.close()


def run_interleaved(model: Path, args: argparse.Namespace) -> None:
    cases = []
    metadata = benchmark_metadata(model, args)
    with ExitStack() as stack:
        run_mini, mini_settings = stack.enter_context(trial_worker(sys.executable, "mini", args))
        runs = {"mini": partial(run_mini, prefill_backend=args.mini_prefill_backend)}
        environments = {"mini": {**mini_settings, "prefill_backend": args.mini_prefill_backend}}
        for backend in args.compare_mini_prefill_backend:
            name = f"mini:{backend}"
            runs[name] = partial(run_mini, prefill_backend=backend)
            environments[name] = {**mini_settings, "prefill_backend": backend}
        if args.interleave_with is not None:
            runs["vllm"], environments["vllm"] = stack.enter_context(
                trial_worker(str(args.interleave_with.absolute()), "vllm", args)
            )
        backends = tuple(runs)
        for index, (workload, prompts) in enumerate(
            zip(args.workloads, make_workloads(model, args), strict=True)
        ):
            for _ in range(args.warmup):
                for run in runs.values():
                    run(index)
            samples: dict[str, list[dict]] = {backend: [] for backend in backends}
            orders = []
            for trial_index in range(args.repeats):
                offset = trial_index % len(backends)
                order = backends[offset:] + backends[:offset]
                if len(backends) > 2 and trial_index // len(backends) % 2:
                    order = order[::-1]
                orders.append(order)
                for backend in order:
                    samples[backend].append(runs[backend](index))
            medians = {backend: summarize(rows) for backend, rows in samples.items()}
            cases.append({
                **workload.describe(),
                "prompt_ids_sha256": hashlib.sha256(json.dumps(prompts).encode()).hexdigest(),
                "order": orders,
                "medians": medians,
                "samples": samples,
            })
            print(json.dumps({**workload.describe(), "medians": medians}), flush=True)
    result = {
        "backend": "interleaved",
        **metadata,
        "environments": environments,
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("mini", "vllm"), required=True)
    parser.add_argument("--model", type=Path, required=True, help="Local Qwen2 checkpoint directory")
    parser.add_argument("--case", type=parse_case, action="append", metavar="REQUESTSxINPUTxOUTPUT")
    parser.add_argument("--mixed", action="store_true",
                        help="Also run a timed trace with short Decode and late ragged Prefill")
    parser.add_argument("--max-running-requests", type=int, default=8)
    parser.add_argument("--max-batch-tokens", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--vllm-eager", action="store_true")
    parser.add_argument("--mini-cuda-graph", action="store_true")
    parser.add_argument("--mini-prefill-cuda-graph", action="store_true")
    parser.add_argument("--mini-prefill-backend", default="auto")
    parser.add_argument("--compare-mini-prefill-backend", action="append", default=[],
                        help="Interleave another mini Prefill backend using the same ModelRunner")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--kv-blocks", type=int,
                        help="Force both backends to use this many 16-token KV blocks")
    parser.add_argument("--ttft-slo-ms", type=float)
    parser.add_argument("--itl-slo-ms", type=float)
    parser.add_argument("--record-tokens", action="store_true",
                        help="Include generated token IDs in the JSON for parity diagnostics")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interleave-with", type=Path, metavar="VLLM_PYTHON",
                        help="Alternate mini and vLLM trials on one GPU using separate interpreters")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.case = args.case or [(4, 32, 32), (8, 256, 32), (1, 2048, 32)]
    args.workloads = make_cases(args.case, args.mixed)
    requests = [spec for workload in args.workloads for spec in workload.requests]
    if not args.model.is_dir() or not (args.model / "config.json").is_file():
        parser.error("--model must be a local checkpoint directory")
    if args.warmup < 1 or args.repeats < 1 or args.max_running_requests < 1:
        parser.error("warmup, repeats and max running requests must be positive")
    if args.max_batch_tokens < max(args.max_running_requests, *(spec.input_length for spec in requests)):
        parser.error("token budget must cover the longest prompt and the running-request limit")
    if args.kv_blocks is not None and args.kv_blocks < max(
        math.ceil((spec.input_length + spec.output_length) / 16) for spec in requests
    ):
        parser.error("--kv-blocks must cover the longest request")
    slo_values = (args.ttft_slo_ms, args.itl_slo_ms)
    if any(value is not None for value in slo_values) and not all(
        value is not None and math.isfinite(value) and value > 0 for value in slo_values
    ):
        parser.error("--ttft-slo-ms and --itl-slo-ms must both be positive")
    if args.backend == "mini" and not args.worker:
        from mini_llm_runtime.engine import PREFILL_ATTENTION_BACKEND_CHOICES

        selected = [args.mini_prefill_backend, *args.compare_mini_prefill_backend]
        if len(set(selected)) != len(selected) or any(
            backend not in PREFILL_ATTENTION_BACKEND_CHOICES for backend in selected
        ):
            parser.error(f"choose distinct mini Prefill backends from {PREFILL_ATTENTION_BACKEND_CHOICES}")

    model = args.model.resolve()
    if args.worker:
        serve_trials(model, args)
        return
    if args.interleave_with is not None or args.compare_mini_prefill_backend:
        if args.backend != "mini":
            parser.error("interleaved comparisons require --backend mini")
        if args.interleave_with is not None and not args.interleave_with.is_file():
            parser.error("--interleave-with requires a vLLM Python executable")
        run_interleaved(model, args)
        return

    metadata = benchmark_metadata(model, args)
    new_trial, settings = (load_mini if args.backend == "mini" else load_vllm)(model, args)
    cases = []
    for workload, prompts in zip(
        args.workloads, make_workloads(model, args), strict=True
    ):
        def run():
            return measure(*new_trial(args.mini_prefill_backend), workload, prompts, args)

        for _ in range(args.warmup):
            run()
        samples = [run() for _ in range(args.repeats)]
        metrics = summarize(samples)
        case = {
            **workload.describe(),
            "prompt_ids_sha256": hashlib.sha256(json.dumps(prompts).encode()).hexdigest(),
            "median": metrics,
            "samples": samples,
        }
        cases.append(case)
        print(json.dumps({**workload.describe(), **metrics}), flush=True)

    result = {
        "backend": args.backend,
        **metadata,
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "dtype": "float16",
        "prefill_backend": args.mini_prefill_backend if args.backend == "mini" else None,
        **settings,
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
