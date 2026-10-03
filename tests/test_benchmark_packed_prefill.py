import pytest
import torch

from scripts.benchmark_packed_prefill import (
    build_layered_correctness,
    compare_kv_diagnostic,
    compare_model_logits,
    compare_packed_storage_exact,
    construct_prompts,
)


BASE = ((1, 2), (3, 4, 5))


def test_construct_prompts_preserves_original_without_length_override() -> None:
    assert construct_prompts(BASE) == BASE


def test_construct_prompts_supports_equal_and_ragged_lengths() -> None:
    assert construct_prompts(BASE, fixed_prompt_length=4) == (
        (1, 2, 1, 2),
        (3, 4, 5, 3),
    )
    # 请求数可以多于模板数；每条请求仍独立使用自己的指定长度。
    assert construct_prompts(BASE, prompt_lengths=(1, 4, 3)) == (
        (1,),
        (3, 4, 5, 3),
        (1, 2, 1),
    )


@pytest.mark.parametrize(
    ("base", "fixed", "lengths"),
    [
        ((), None, None),
        (((1,), ()), None, None),
        (BASE, 0, None),
        (BASE, None, ()),
        (BASE, None, (1, 0)),
        (BASE, 2, (2,)),
    ],
)
def test_construct_prompts_rejects_invalid_shapes(base, fixed, lengths) -> None:
    with pytest.raises(ValueError):
        construct_prompts(
            base, fixed_prompt_length=fixed, prompt_lengths=lengths
        )


class FakeManager:
    def __init__(self, key: torch.Tensor, value: torch.Tensor) -> None:
        self.key = key
        self.value = value

    def gather(self, request_id: str) -> tuple[torch.Tensor, torch.Tensor]:
        assert request_id == "request-0"
        return self.key, self.value


def test_layered_gate_keeps_fp16_kv_drift_as_diagnostic() -> None:
    reference_logits = torch.tensor([[2.0, 1.0]])
    packed_logits = torch.tensor([[1.999, 1.001]])
    model = compare_model_logits(reference_logits, packed_logits)
    serial = FakeManager(torch.ones(2), torch.ones(2))
    packed = FakeManager(torch.ones(2), torch.tensor([1.0, 1.1]))
    diagnostics = compare_kv_diagnostic(serial, packed, ("request-0",))
    exact_write = compare_packed_storage_exact(
        packed,
        FakeManager(packed.key.clone(), packed.value.clone()),
        ("request-0",),
        packed_logits,
        packed_logits.clone(),
    )
    gate = build_layered_correctness(
        {"packed_vectorized": model},
        {"masked_scalar_vs_vectorized": exact_write},
        {"packed_vectorized": diagnostics},
    )
    assert model["passed"]
    assert exact_write["passed"]
    assert not diagnostics["kv_within_legacy_1pct_diagnostic"]
    assert gate["schema_version"] == 2
    assert gate["passed"]


def test_layered_gate_rejects_physical_kv_or_logit_mismatch() -> None:
    logits = torch.tensor([[2.0, 1.0]])
    scalar = FakeManager(torch.ones(2), torch.ones(2))
    wrong = FakeManager(torch.ones(2), torch.tensor([1.0, 1.1]))
    storage = compare_packed_storage_exact(
        scalar, wrong, ("request-0",), logits, logits.clone()
    )
    assert not storage["passed"]
    assert storage["first_kv_mismatch"] == "request-0:value"
    gate = build_layered_correctness(
        {"packed": compare_model_logits(logits, logits.clone())},
        {"masked_scalar_vs_vectorized": storage},
        {"packed": compare_kv_diagnostic(scalar, scalar, ("request-0",))},
    )
    assert not gate["passed"]


def test_layered_gate_rejects_nonfinite_or_changed_greedy_logits() -> None:
    reference = torch.tensor([[2.0, 1.0]])
    changed = compare_model_logits(reference, torch.tensor([[1.0, 2.0]]))
    too_far = compare_model_logits(reference, torch.tensor([[2.02, 1.02]]))
    nonfinite = compare_model_logits(reference, torch.tensor([[float("nan"), 1.0]]))
    assert not changed["passed"] and not changed["tokens_match"]
    assert not too_far["passed"] and too_far["tokens_match"]
    assert not nonfinite["passed"] and nonfinite["logits_relative_l2"] is None
    assert not build_layered_correctness({"changed": changed}, {}, {})["passed"]
    assert not build_layered_correctness({}, {}, {})["passed"]


def test_layered_gate_rejects_nonfinite_kv() -> None:
    clean = FakeManager(torch.ones(2), torch.ones(2))
    nonfinite = FakeManager(torch.ones(2), torch.tensor([float("nan"), 1.0]))
    diagnostic = compare_kv_diagnostic(clean, nonfinite, ("request-0",))
    assert not diagnostic["finite"]
    assert not build_layered_correctness(
        {"packed": {"passed": True}}, {}, {"packed": diagnostic}
    )["passed"]
