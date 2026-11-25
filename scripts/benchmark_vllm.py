#!/usr/bin/env python3
"""Measure mini-llm-runtime and vLLM with identical, pretokenized burst workloads."""

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
from pathlib import Path

import torch


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


def measure(submit, step, prompts: list[list[int]], output_length: int) -> dict:
    tokens: list[list[int]] = [[] for _ in prompts]
    timestamps: list[list[int]] = [[] for _ in prompts]

    started = time.perf_counter_ns()
    for request_id, prompt in enumerate(prompts):
        submit(str(request_id), prompt, output_length)
    remaining = len(prompts)
    while remaining:
        emitted = step()
        ready = time.perf_counter_ns()
        for request_id, delta, finished in emitted:
            index = int(request_id)
            tokens[index].extend(delta)
            timestamps[index].extend([ready] * len(delta))
            if finished:
                if len(tokens[index]) != output_length:
                    raise RuntimeError(f"unexpected number of tokens for model request {request_id}")
                remaining -= 1

    elapsed_ns = max(events[-1] for events in timestamps) - started
    ttft_ms = [(events[0] - started) / 1e6 for events in timestamps]
    tpot_ms = (
        [(events[-1] - events[0]) / ((output_length - 1) * 1e6) for events in timestamps]
        if output_length > 1
        else []
    )
    itl_ms = [
        (right - left) / 1e6
        for events in timestamps
        for left, right in zip(events, events[1:])
    ]
    return {
        "output_tokens_per_second": len(prompts) * output_length * 1e9 / elapsed_ns,
        "ttft_p50_ms": percentile(ttft_ms, 0.50),
        "ttft_p95_ms": percentile(ttft_ms, 0.95),
        "tpot_p50_ms": percentile(tpot_ms, 0.50) if tpot_ms else None,
        "tpot_p95_ms": percentile(tpot_ms, 0.95) if tpot_ms else None,
        "itl_p95_ms": percentile(itl_ms, 0.95) if itl_ms else None,
        "tokens_sha256": hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
    }


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
    max_blocks = max(
        min(requests, args.max_running_requests) * math.ceil((inputs + outputs - 1) / block_size)
        for requests, inputs, outputs in args.case
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

    def new_trial():
        if manager.allocator.free_count != max_blocks:
            raise RuntimeError("previous mini trial left KV blocks allocated")
        admission = PagedBlockAdmissionController(manager)
        scheduler = RequestScheduler(
            max_running_requests=args.max_running_requests,
            max_batch_tokens=args.max_batch_tokens,
            admission_callback=admission.try_admit,
        )
        engine = ContinuousBatchEngine(scheduler=scheduler, runner=runner, admission=admission)

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
            max_model_len=max(inputs + outputs for _, inputs, outputs in args.case),
            max_num_seqs=args.max_running_requests,
            max_num_batched_tokens=args.max_batch_tokens,
            block_size=16,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enable_prefix_caching=False,
            disable_log_stats=True,
            enforce_eager=args.vllm_eager,
        )
    )
    next_trial = 0

    def new_trial():
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
    }


def make_workloads(model: Path, args: argparse.Namespace):
    # Keep input IDs within the low, non-special part of the Qwen vocabulary.
    vocab = min(1000, json.loads((model / "config.json").read_text())["vocab_size"])
    workloads = []
    for request_count, input_length, _ in args.case:
        rng = random.Random(args.seed)
        workloads.append([[rng.randrange(3, vocab) for _ in range(input_length)]
                          for _ in range(request_count)])
    return workloads


def summarize(samples: list[dict]) -> dict:
    return {
        name: percentile([sample[name] for sample in samples], 0.50)
        for name, value in samples[0].items()
        if name != "tokens_sha256" and value is not None
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
        index = int(line)
        with redirect_stdout(sys.stderr):
            sample = measure(*new_trial(), workloads[index], args.case[index][2])
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
    for enabled, flag in (
        (args.mini_cuda_graph, "--mini-cuda-graph"),
        (args.mini_prefill_cuda_graph, "--mini-prefill-cuda-graph"),
        (args.vllm_eager, "--vllm-eager"),
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

    def run(index: int):
        process.stdin.write(f"{index}\n")
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
    interpreters = {"mini": sys.executable, "vllm": str(args.interleave_with.absolute())}
    cases = []
    with ExitStack() as stack:
        workers = {
            backend: stack.enter_context(trial_worker(interpreter, backend, args))
            for backend, interpreter in interpreters.items()
        }
        for index, (shape, prompts) in enumerate(
            zip(args.case, make_workloads(model, args), strict=True)
        ):
            for _ in range(args.warmup):
                for run, _ in workers.values():
                    run(index)
            samples: dict[str, list[dict]] = {"mini": [], "vllm": []}
            orders = []
            for trial_index in range(args.repeats):
                order = ("mini", "vllm") if trial_index % 2 == 0 else ("vllm", "mini")
                orders.append(order)
                for backend in order:
                    samples[backend].append(workers[backend][0](index))
            medians = {backend: summarize(rows) for backend, rows in samples.items()}
            cases.append({
                "shape": shape,
                "prompt_ids_sha256": hashlib.sha256(json.dumps(prompts).encode()).hexdigest(),
                "order": orders,
                "medians": medians,
                "samples": samples,
            })
            print(json.dumps({"shape": shape, "medians": medians}), flush=True)
    result = {
        "backend": "interleaved",
        "model": str(model),
        "seed": args.seed,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "max_running_requests": args.max_running_requests,
        "max_batch_tokens": args.max_batch_tokens,
        "block_size": 16,
        "environments": {backend: stats for backend, (_, stats) in workers.items()},
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("mini", "vllm"), required=True)
    parser.add_argument("--model", type=Path, required=True, help="Local Qwen2 checkpoint directory")
    parser.add_argument("--case", type=parse_case, action="append", metavar="REQUESTSxINPUTxOUTPUT")
    parser.add_argument("--max-running-requests", type=int, default=8)
    parser.add_argument("--max-batch-tokens", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--vllm-eager", action="store_true")
    parser.add_argument("--mini-cuda-graph", action="store_true")
    parser.add_argument("--mini-prefill-cuda-graph", action="store_true")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interleave-with", type=Path, metavar="VLLM_PYTHON",
                        help="Alternate mini and vLLM trials on one GPU using separate interpreters")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.case = args.case or [(4, 32, 32), (8, 256, 32), (1, 2048, 32)]
    if not args.model.is_dir() or not (args.model / "config.json").is_file():
        parser.error("--model must be a local checkpoint directory")
    if args.warmup < 1 or args.repeats < 1 or args.max_running_requests < 1:
        parser.error("warmup, repeats and max running requests must be positive")
    if args.max_batch_tokens < max(args.max_running_requests, *(case[1] for case in args.case)):
        parser.error("token budget must cover the longest prompt and the running-request limit")

    model = args.model.resolve()
    if args.worker:
        serve_trials(model, args)
        return
    if args.interleave_with is not None:
        if args.backend != "mini" or not args.interleave_with.is_file():
            parser.error("--interleave-with requires --backend mini and a vLLM Python executable")
        run_interleaved(model, args)
        return

    new_trial, settings = (load_mini if args.backend == "mini" else load_vllm)(model, args)
    cases = []
    for (request_count, input_length, output_length), prompts in zip(
        args.case, make_workloads(model, args), strict=True
    ):
        for _ in range(args.warmup):
            measure(*new_trial(), prompts, output_length)
        samples = [
            measure(*new_trial(), prompts, output_length)
            for _ in range(args.repeats)
        ]
        metrics = summarize(samples)
        case = {
            "shape": [request_count, input_length, output_length],
            "prompt_ids_sha256": hashlib.sha256(json.dumps(prompts).encode()).hexdigest(),
            "median": metrics,
            "samples": samples,
        }
        cases.append(case)
        print(json.dumps({"shape": case["shape"], **metrics}), flush=True)

    result = {
        "backend": args.backend,
        "model": str(model),
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "dtype": "float16",
        "max_running_requests": args.max_running_requests,
        "max_batch_tokens": args.max_batch_tokens,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "seed": args.seed,
        "block_size": 16,
        **settings,
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
