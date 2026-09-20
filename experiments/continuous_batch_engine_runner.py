#!/usr/bin/env python3
"""真实 Qwen 最小 Continuous Batching Engine 动态加入/离开演示。"""

from __future__ import annotations

import argparse
import gc

import torch
from huggingface_hub import snapshot_download

from mini_llm_runtime.block_admission import PagedBlockAdmissionController
from mini_llm_runtime.engine import ContinuousBatchEngine
from mini_llm_runtime.paged_kv_manager import PagedKVCacheManager
from mini_llm_runtime.qwen_loader import load_qwen_checkpoint
from mini_llm_runtime.qwen_model_runner import QwenPrefillRunner
from mini_llm_runtime.scheduler import RequestScheduler


REQUESTS = {
    "A": ("你好，GPU", 3),
    "B": ("CUDA 是什么？", 1),
    "C": ("请简要解释 KV Cache。", 2),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--block-size", type=int, default=4)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def hf_greedy_tokens(model, input_ids: torch.Tensor, count: int) -> tuple[int, ...]:
    """显式分离 HF Prefill/Decode，避免用 generate 隐藏 reference 数据流。"""
    generated: list[int] = []
    with torch.inference_mode():
        output = model(input_ids=input_ids, use_cache=True, return_dict=True)
        token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
        cache = output.past_key_values
        generated.append(int(token.item()))
        for _ in range(count - 1):
            output = model(
                input_ids=token,
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
            token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
            cache = output.past_key_values
            generated.append(int(token.item()))
    return tuple(generated)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("该实验需要 CUDA")
    if args.block_size <= 0:
        raise SystemExit("--block-size 必须 > 0")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_dir = snapshot_download(args.model, local_files_only=args.local_files_only)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    encoded = {
        request_id: tokenizer(prompt, return_tensors="pt")["input_ids"].cuda()
        for request_id, (prompt, _) in REQUESTS.items()
    }

    hf_model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        dtype=torch.float16,
        attn_implementation="eager",
        local_files_only=True,
    ).cuda().eval()
    reference = {
        request_id: hf_greedy_tokens(hf_model, encoded[request_id], max_new_tokens)
        for request_id, (_, max_new_tokens) in REQUESTS.items()
    }
    del hf_model
    gc.collect()
    torch.cuda.empty_cache()

    config, weights = load_qwen_checkpoint(
        model_dir, device="cuda", dtype=torch.float16
    )
    runner = QwenPrefillRunner(
        config, weights, decode_attention_backend="paged_cuda"
    )
    manager = PagedKVCacheManager(
        total_blocks=16,
        block_size=args.block_size,
        num_layers=config.num_hidden_layers,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        dtype=weights.embedding.dtype,
        device=weights.embedding.device,
    )
    admission = PagedBlockAdmissionController(manager)
    scheduler = RequestScheduler(
        max_running_requests=3,
        max_batch_tokens=64,
        admission_callback=admission.try_admit,
    )
    engine = ContinuousBatchEngine(
        scheduler=scheduler, runner=runner, admission=admission
    )

    def submit(request_id: str) -> None:
        _, max_new_tokens = REQUESTS[request_id]
        scheduler.submit(
            request_id,
            encoded[request_id][0].cpu().tolist(),
            max_new_tokens=max_new_tokens,
        )

    submit("A")
    submit("B")
    step_records: list[dict[str, object]] = []
    first = engine.step()
    if first is None:
        raise SystemExit("首个 Engine step 不应为空")
    step_records.append(
        {
            "step": first.batch.step_index,
            "prefill": first.batch.prefill_request_ids,
            "decode": first.batch.decode_request_ids,
            "finished": first.update.finished_request_ids,
        }
    )

    # 模拟 B 完成后、GPU step 边界到达的新请求 C。
    submit("C")
    while scheduler.has_unfinished_requests:
        result = engine.step()
        if result is None:
            raise SystemExit("存在未完成请求，但 block budget 无法取得进展")
        step_records.append(
            {
                "step": result.batch.step_index,
                "prefill": result.batch.prefill_request_ids,
                "decode": result.batch.decode_request_ids,
                "finished": result.update.finished_request_ids,
            }
        )

    print("===== Continuous Batch Engine Steps =====")
    for record in step_records:
        print(
            f"step={record['step']} prefill={record['prefill']} "
            f"decode={record['decode']} finished={record['finished']}"
        )

    print("\n===== Generated Tokens vs Hugging Face =====")
    failed: list[str] = []
    for request_id in ("A", "B", "C"):
        actual = scheduler.get_request(request_id).generated_token_ids
        expected = reference[request_id]
        match = actual == expected
        print(
            f"{request_id}: match={match} ids={actual} "
            f"text={tokenizer.decode(actual)!r}"
        )
        if not match:
            failed.append(request_id)

    all_released = (
        manager.request_ids == ()
        and admission.reservations == ()
        and manager.allocator.free_count == manager.allocator.total_blocks
    )
    print(f"\nall KV blocks released = {all_released}")
    if failed or not all_released:
        raise SystemExit(
            f"Engine 对拍失败：token_mismatch={failed}, released={all_released}"
        )
    print("全部通过：请求动态加入/离开、token 写回与 block 生命周期正确。")


if __name__ == "__main__":
    main()
