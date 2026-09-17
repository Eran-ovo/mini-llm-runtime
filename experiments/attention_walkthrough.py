#!/usr/bin/env python3
"""用真实 Qwen2.5 第 0 层观察 Prefill、Decode、GQA、RoPE 与 KV Cache。

这是教学脚本，不是 benchmark：forward hook 和 output_attentions 都会引入额外开销。
"""

from __future__ import annotations

import argparse
from typing import Any

import torch


def to_head_layout(raw: torch.Tensor, num_heads: int, head_dim: int) -> torch.Tensor:
    """把 Projection 输出 [B, S, H*D] 转换为 Attention 布局 [B, H, S, D]。"""
    if raw.ndim != 3:
        raise ValueError("Projection 输出必须是 [batch, sequence, hidden]")
    batch, sequence, width = raw.shape
    if width != num_heads * head_dim:
        raise ValueError(
            f"最后一维 {width} != num_heads({num_heads}) * head_dim({head_dim})"
        )
    return raw.reshape(batch, sequence, num_heads, head_dim).transpose(1, 2)


def query_to_kv_head(
    query_head: int, num_query_heads: int, num_kv_heads: int
) -> int:
    """返回 GQA 中某个 Query Head 共享的 KV Head。"""
    if num_query_heads % num_kv_heads != 0:
        raise ValueError("num_query_heads 必须能被 num_kv_heads 整除")
    if not 0 <= query_head < num_query_heads:
        raise ValueError("query_head 越界")
    return query_head // (num_query_heads // num_kv_heads)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--prompt", default="你好，GPU")
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="只使用 Hugging Face 本地缓存，不访问网络",
    )
    return parser.parse_args()


def _shape(tensor: torch.Tensor) -> str:
    return str(list(tensor.shape))


def _print_stage(
    name: str,
    raw: dict[str, torch.Tensor],
    attention: torch.Tensor,
    cache_layer: Any,
    num_query_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> None:
    q = to_head_layout(raw["q"], num_query_heads, head_dim)
    k = to_head_layout(raw["k"], num_kv_heads, head_dim)
    v = to_head_layout(raw["v"], num_kv_heads, head_dim)

    # Cache 中最后 raw sequence 个位置，正好对应本次 forward 新产生的 K/V。
    new_length = k.shape[2]
    cached_k = cache_layer.keys[:, :, -new_length:, :]
    cached_v = cache_layer.values[:, :, -new_length:, :]
    k_rope_delta = (cached_k.float() - k.float()).abs().max().item()
    v_delta = (cached_v.float() - v.float()).abs().max().item()

    last_probability_sum = attention[0, 0, -1].float().sum().item()
    last_probabilities = attention[0, 0, -1].float().cpu().tolist()
    head0_matrix = attention[0, 0].float().cpu().tolist()

    print(f"\n===== {name} =====")
    print(f"raw q_proj output : {_shape(raw['q'])}")
    print(f"raw k_proj output : {_shape(raw['k'])}")
    print(f"raw v_proj output : {_shape(raw['v'])}")
    print(f"Q head layout     : {_shape(q)}")
    print(f"K head layout     : {_shape(k)}")
    print(f"V head layout     : {_shape(v)}")
    print(f"Attention         : {_shape(attention)}")
    print(f"Layer-0 K Cache   : {_shape(cache_layer.keys)}")
    print(f"Layer-0 V Cache   : {_shape(cache_layer.values)}")
    print(f"max|cached K - raw K| = {k_rope_delta:.8f}  (K 经过 RoPE)")
    print(f"max|cached V - raw V| = {v_delta:.8f}  (V 不经过 RoPE)")
    print(f"Head-0 最后一个 Query 的概率和 = {last_probability_sum:.8f}")
    print(f"Head-0 最后一个 Query 的 Attention 概率 = {last_probabilities}")
    if attention.shape[-1] <= 16:
        print("Head-0 完整 Attention 矩阵（每行是一个 Query）：")
        for row in head0_matrix:
            print("  " + " ".join(f"{value:8.5f}" for value in row))
    else:
        print("序列长度超过 16，省略完整 Attention 矩阵以避免刷屏")


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("该实验需要 CUDA")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=args.local_files_only
    )
    # eager attention 才能直接返回 Attention probability；因此本脚本不能用于计时。
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float16,
        attn_implementation="eager",
        local_files_only=args.local_files_only,
    ).to(device).eval()

    config = model.config
    num_query_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_dim = config.hidden_size // num_query_heads
    group_size = num_query_heads // num_kv_heads

    print("===== 模型配置 =====")
    print(f"hidden_size       = {config.hidden_size}")
    print(f"num_query_heads   = {num_query_heads}")
    print(f"num_kv_heads      = {num_kv_heads}")
    print(f"head_dim          = {head_dim}")
    print(f"GQA group_size    = {group_size}")
    for q_head in range(num_query_heads):
        print(
            f"query head {q_head:2d} -> kv head "
            f"{query_to_kv_head(q_head, num_query_heads, num_kv_heads)}"
        )

    layer0_attention = model.model.layers[0].self_attn
    captured: dict[str, torch.Tensor] = {}

    def capture(name: str):
        def hook(_module: Any, _inputs: Any, output: torch.Tensor) -> None:
            # 这里只保存第 0 层的三个小 Projection 输出；detach 防止保留计算图。
            captured[name] = output.detach()

        return hook

    handles = [
        layer0_attention.q_proj.register_forward_hook(capture("q")),
        layer0_attention.k_proj.register_forward_hook(capture("k")),
        layer0_attention.v_proj.register_forward_hook(capture("v")),
    ]

    try:
        encoded = tokenizer(args.prompt, return_tensors="pt").to(device)
        prompt_length = encoded["input_ids"].shape[1]
        print(f"\nprompt            = {args.prompt!r}")
        print(f"prompt token ids  = {encoded['input_ids'][0].tolist()}")
        print(f"prompt length     = {prompt_length}")

        with torch.inference_mode():
            prefill = model(
                **encoded,
                use_cache=True,
                output_attentions=True,
                return_dict=True,
            )

        cache = prefill.past_key_values
        prefill_raw = dict(captured)
        _print_stage(
            "Prefill",
            prefill_raw,
            prefill.attentions[0],
            cache.layers[0],
            num_query_heads,
            num_kv_heads,
            head_dim,
        )
        assert cache.get_seq_length() == prompt_length

        # Prefill 最后位置直接产生第一个新 token；它不是由额外 Decode 产生的。
        next_token = torch.argmax(
            prefill.logits[:, -1, :], dim=-1, keepdim=True
        )
        print(f"Prefill 预测 token = {next_token.item()}")
        print(f"Prefill 预测文本  = {tokenizer.decode(next_token[0])!r}")

        # DynamicCache 会原地增长，先复制历史前缀，才能检查 Decode 是否只追加数据。
        old_key_prefix = cache.layers[0].keys.detach().clone()
        old_cache_length = cache.get_seq_length()
        captured.clear()
        decode_mask = torch.cat(
            [encoded["attention_mask"], torch.ones_like(next_token)], dim=1
        )

        with torch.inference_mode():
            decode = model(
                input_ids=next_token,
                attention_mask=decode_mask,
                past_key_values=cache,
                use_cache=True,
                output_attentions=True,
                return_dict=True,
            )

        _print_stage(
            "第一次 Decode",
            captured,
            decode.attentions[0],
            cache.layers[0],
            num_query_heads,
            num_kv_heads,
            head_dim,
        )
        new_cache_length = cache.get_seq_length()
        history_unchanged = torch.equal(
            old_key_prefix, cache.layers[0].keys[:, :, :old_cache_length, :]
        )
        print(f"Decode 前 Cache length = {old_cache_length}")
        print(f"Decode 后 Cache length = {new_cache_length}")
        print(f"历史 K Cache 保持不变   = {history_unchanged}")
        print("\n结论：Decode 只为新 token 计算 Q/K/V，并把新 K/V 追加到每层 Cache。")

        assert new_cache_length == old_cache_length + 1
        assert history_unchanged
        assert decode.attentions[0].shape[-2:] == (1, new_cache_length)
    finally:
        for handle in handles:
            handle.remove()


if __name__ == "__main__":
    main()
