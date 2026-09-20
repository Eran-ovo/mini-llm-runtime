from experiments.continuous_batch_engine_runner import evaluate_gate


def make_valid_gate_input() -> dict:
    return {
        "step_records": [
            {
                "step": 0,
                "prefill": ("A", "B"),
                "decode": (),
                "released_blocks": {"B": (2,)},
                "active_cache_after_step": {
                    "A": {"block_ids": (0, 1), "token_count": 3},
                },
            },
            {
                "step": 1,
                "prefill": ("C",),
                "decode": ("A",),
                "released_blocks": {},
                "active_cache_after_step": {
                    "A": {"block_ids": (0, 1), "token_count": 4},
                    "C": {"block_ids": (2, 3), "token_count": 7},
                },
            },
            {
                "step": 2,
                "prefill": (),
                "decode": ("A", "C"),
                "released_blocks": {"A": (0, 1), "C": (2, 3)},
                "active_cache_after_step": {},
            },
        ],
        "token_comparisons": {
            "A": {"match": True},
            "B": {"match": True},
            "C": {"match": True},
        },
        "block_size": 4,
        "all_resources_released": True,
    }


def test_correctness_gate_requires_all_runtime_paths() -> None:
    result = evaluate_gate(**make_valid_gate_input())
    assert result["passed"]
    assert all(result["checks"].values())
    assert result["reused_block_ids"] == (2,)


def test_correctness_gate_fails_shared_token_mismatch() -> None:
    inputs = make_valid_gate_input()
    inputs["token_comparisons"]["C"]["match"] = False
    result = evaluate_gate(**inputs)
    assert not result["passed"]
    assert not result["checks"]["all_token_sequences_match_hf"]


def test_correctness_gate_fails_when_batched_decode_path_is_not_exercised() -> None:
    inputs = make_valid_gate_input()
    inputs["step_records"][2]["decode"] = ("A",)
    result = evaluate_gate(**inputs)
    assert not result["passed"]
    assert not result["checks"]["has_batched_decode"]


def test_correctness_gate_requires_observed_logical_block_crossing() -> None:
    inputs = make_valid_gate_input()
    for record in inputs["step_records"]:
        for cache in record["active_cache_after_step"].values():
            # 即使预留了两个物理 block，逻辑 token 未超过 block_size 也不算命中边界。
            cache["token_count"] = min(cache["token_count"], 4)
    result = evaluate_gate(**inputs)
    assert not result["passed"]
    assert not result["checks"][
        "observes_cache_sequence_crossing_block_boundary"
    ]
