import json
from pathlib import Path

import pytest

from scripts.run_release_evaluation import (
    StepSpec,
    build_steps,
    bundle_files,
    require_clean_repo,
    verify_release_artifacts,
    run_step,
)


def test_release_steps_keep_formal_measurement_parameters(tmp_path: Path) -> None:
    steps = build_steps(
        python="/venv/python",
        output_dir=tmp_path,
        local_files_only=True,
    )
    assert [item.name for item in steps] == [
        "pytest",
        "continuous_batch_correctness",
        "static_vs_continuous_benchmark",
        "mixed_prefill_tail_benchmark",
    ]
    commands = {item.name: item.command for item in steps}
    assert "--local-files-only" in commands["continuous_batch_correctness"]
    batching = commands["static_vs_continuous_benchmark"]
    assert batching[batching.index("--warmup") + 1] == "3"
    assert batching[batching.index("--repeats") + 1] == "10"
    mixed = commands["mixed_prefill_tail_benchmark"]
    assert mixed[mixed.index("--warmup") + 1] == "3"
    assert mixed[mixed.index("--repeats") + 1] == "12"


def test_bundle_files_are_sorted_hashed_and_exclude_manifest(tmp_path: Path) -> None:
    (tmp_path / "z.txt").write_text("z", encoding="utf-8")
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    (tmp_path / "bundle_manifest.json").write_text("{}", encoding="utf-8")
    records = bundle_files(tmp_path)
    assert [item["path"] for item in records] == ["a.txt", "z.txt"]
    assert all(len(item["sha256"]) == 64 for item in records)


def write_result(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_release_verification_requires_clean_matching_artifacts(
    tmp_path: Path,
) -> None:
    commit = "a" * 40
    environment = {"git_commit": commit, "git_dirty": False}
    write_result(
        tmp_path / "continuous_batch_correctness/result.json",
        {"gate": {"passed": True}, "environment_after": environment},
    )
    write_result(
        tmp_path / "static_vs_continuous/result.json",
        {
            "correctness": {"all_policy_token_sequences_equal": True},
            "environment_after": environment,
        },
    )
    write_result(
        tmp_path / "mixed_prefill_tail/result.json",
        {
            "correctness": {"all_budget_token_sequences_equal": True},
            "environment_after": environment,
        },
    )
    checks = verify_release_artifacts(tmp_path, commit)
    assert all(checks.values())

    dirty = json.loads(
        (tmp_path / "mixed_prefill_tail/result.json").read_text(encoding="utf-8")
    )
    dirty["environment_after"]["git_dirty"] = True
    write_result(tmp_path / "mixed_prefill_tail/result.json", dirty)
    with pytest.raises(RuntimeError, match="mixed_tree_clean"):
        verify_release_artifacts(tmp_path, commit)


def test_require_clean_repo_rejects_dirty_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    responses = iter(("b" * 40, " M README.md"))
    monkeypatch.setattr(
        "scripts.run_release_evaluation.run_git",
        lambda *_args: next(responses),
    )
    with pytest.raises(RuntimeError, match="clean worktree"):
        require_clean_repo(tmp_path, "b" * 40)


def test_failed_step_returns_auditable_record(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    step = StepSpec(
        "failure",
        (
            "/bin/sh",
            "-c",
            "printf failure-output; exit 7",
        ),
    )
    record = run_step(step, tmp_path, log_dir)
    assert record["returncode"] == 7
    assert record["log"] == "logs/failure.log"
    assert (log_dir / "failure.log").read_text() == "failure-output"
