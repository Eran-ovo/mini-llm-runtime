#!/usr/bin/env python3
"""在临时 clean worktree 中运行 v1.0 release correctness 与正式 benchmark。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class StepSpec:
    name: str
    command: tuple[str, ...]


class ReleaseStepError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_git(repo_root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=repo_root, text=True, stderr=subprocess.STDOUT
    ).strip()


def require_clean_repo(repo_root: Path, expected_commit: str) -> None:
    actual_commit = run_git(repo_root, "rev-parse", "HEAD")
    if actual_commit != expected_commit:
        raise RuntimeError(
            f"clean worktree commit 不匹配：actual={actual_commit}, "
            f"expected={expected_commit}"
        )
    dirty = run_git(repo_root, "status", "--porcelain")
    if dirty:
        raise RuntimeError(f"release evaluation 要求 clean worktree：\n{dirty}")


def build_steps(
    *,
    python: str,
    output_dir: Path,
    local_files_only: bool,
) -> tuple[StepSpec, ...]:
    local_flag = ("--local-files-only",) if local_files_only else ()
    return (
        StepSpec("pytest", (python, "-m", "pytest", "-q")),
        StepSpec(
            "continuous_batch_correctness",
            (
                python,
                "-m",
                "experiments.continuous_batch_engine_runner",
                *local_flag,
                "--output-dir",
                str(output_dir / "continuous_batch_correctness"),
            ),
        ),
        StepSpec(
            "static_vs_continuous_benchmark",
            (
                python,
                "scripts/benchmark_batching_policy.py",
                *local_flag,
                "--request-count",
                "8",
                "--generation-lengths",
                "2,4,8,12",
                "--max-running-requests",
                "4",
                "--max-batch-tokens",
                "64",
                "--block-size",
                "16",
                "--warmup",
                "3",
                "--repeats",
                "10",
                "--output-dir",
                str(output_dir / "static_vs_continuous"),
            ),
        ),
        StepSpec(
            "mixed_prefill_tail_benchmark",
            (
                python,
                "scripts/benchmark_mixed_prefill_budget.py",
                *local_flag,
                "--request-count",
                "8",
                "--generation-lengths",
                "2,4,8,12",
                "--max-running-requests",
                "4",
                "--max-batch-tokens",
                "64",
                "--mixed-prefill-budgets",
                "none,8,4",
                "--block-size",
                "16",
                "--warmup",
                "3",
                "--repeats",
                "12",
                "--output-dir",
                str(output_dir / "mixed_prefill_tail"),
            ),
        ),
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bundle_files(output_dir: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(item for item in output_dir.rglob("*") if item.is_file()):
        if path.name == "bundle_manifest.json":
            continue
        records.append(
            {
                "path": str(path.relative_to(output_dir)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_release_artifacts(output_dir: Path, commit: str) -> dict[str, bool]:
    correctness = load_json(
        output_dir / "continuous_batch_correctness" / "result.json"
    )
    batching = load_json(output_dir / "static_vs_continuous" / "result.json")
    mixed = load_json(output_dir / "mixed_prefill_tail" / "result.json")
    checks = {
        "continuous_batch_gate_passed": correctness["gate"]["passed"] is True,
        "continuous_batch_commit_matches": correctness["environment_after"][
            "git_commit"
        ]
        == commit,
        "continuous_batch_tree_clean": correctness["environment_after"][
            "git_dirty"
        ]
        is False,
        "batching_tokens_match": batching["correctness"][
            "all_policy_token_sequences_equal"
        ]
        is True,
        "batching_commit_matches": batching["environment_after"]["git_commit"]
        == commit,
        "batching_tree_clean": batching["environment_after"]["git_dirty"] is False,
        "mixed_tokens_match": mixed["correctness"][
            "all_budget_token_sequences_equal"
        ]
        is True,
        "mixed_commit_matches": mixed["environment_after"]["git_commit"]
        == commit,
        "mixed_tree_clean": mixed["environment_after"]["git_dirty"] is False,
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"release artifact verification 失败：{failed}")
    return checks


def run_step(step: StepSpec, repo_root: Path, log_dir: Path) -> dict[str, Any]:
    log_path = log_dir / f"{step.name}.log"
    started = time.perf_counter_ns()
    print(f"[release] START {step.name}: {shlex.join(step.command)}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            step.command,
            cwd=repo_root,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    elapsed_ns = time.perf_counter_ns() - started
    record = {
        "name": step.name,
        "command": step.command,
        "returncode": completed.returncode,
        "elapsed_ns": elapsed_ns,
        "log": str(log_path.relative_to(log_dir.parent)),
    }
    print(
        f"[release] END {step.name}: rc={completed.returncode}, "
        f"elapsed={elapsed_ns / 1e9:.2f}s",
        flush=True,
    )
    return record


def write_manifest(output_dir: Path, manifest: dict[str, Any]) -> None:
    manifest["files"] = bundle_files(output_dir)
    (output_dir / "bundle_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def run_inside_clean_worktree(args: argparse.Namespace) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise SystemExit(f"输出目录已存在，拒绝覆盖：{output_dir}")
    output_dir.mkdir(parents=True)
    log_dir = output_dir / "logs"
    log_dir.mkdir()

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "classification": "formal_release_candidate",
        "passed": False,
        "source_commit": args.expected_commit,
        "started_at_utc": utc_now(),
        "completed_at_utc": None,
        "python_executable": sys.executable,
        "torch_cuda_arch_list": os.environ.get("TORCH_CUDA_ARCH_LIST"),
        "local_files_only": not args.allow_download,
        "import_provenance": {},
        "steps": [],
        "verification": {},
        "error": None,
    }

    try:
        require_clean_repo(repo_root, args.expected_commit)
        import mini_llm_runtime
        from mini_llm_runtime.environment import collect_environment

        package_file = Path(mini_llm_runtime.__file__).resolve()
        expected_source = (repo_root / "src").resolve()
        if not package_file.is_relative_to(expected_source):
            raise RuntimeError(
                f"mini_llm_runtime 导入来源错误：{package_file}，"
                f"expected under {expected_source}"
            )
        manifest["import_provenance"] = {
            "mini_llm_runtime": str(package_file),
            "expected_source_root": str(expected_source),
        }
        (output_dir / "environment_before.json").write_text(
            json.dumps(collect_environment(repo_root), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        steps = build_steps(
            python=sys.executable,
            output_dir=output_dir,
            local_files_only=not args.allow_download,
        )
        for step in steps:
            record = run_step(step, repo_root, log_dir)
            manifest["steps"].append(record)
            if record["returncode"] != 0:
                raise ReleaseStepError(
                    f"step {step.name} 失败，returncode={record['returncode']}，"
                    f"详见 {output_dir / record['log']}"
                )

        manifest["verification"] = verify_release_artifacts(
            output_dir, args.expected_commit
        )
        require_clean_repo(repo_root, args.expected_commit)
        (output_dir / "environment_after.json").write_text(
            json.dumps(collect_environment(repo_root), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        manifest["passed"] = True
    except Exception as error:
        manifest["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        manifest["completed_at_utc"] = utc_now()
        write_manifest(output_dir, manifest)


def launch_isolated(args: argparse.Namespace) -> int:
    source_root = Path(__file__).resolve().parents[1]
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise SystemExit(f"输出目录已存在，拒绝覆盖：{output_dir}")
    commit = run_git(source_root, "rev-parse", "HEAD")

    with tempfile.TemporaryDirectory(prefix="mini-llm-release-") as temp_parent:
        worktree = Path(temp_parent) / "repo"
        subprocess.run(
            ["git", "worktree", "add", "--detach", str(worktree), commit],
            cwd=source_root,
            check=True,
        )
        try:
            env = os.environ.copy()
            clean_paths = [str(worktree / "src"), str(worktree)]
            if env.get("PYTHONPATH"):
                clean_paths.append(env["PYTHONPATH"])
            env["PYTHONPATH"] = os.pathsep.join(clean_paths)
            env["TORCH_CUDA_ARCH_LIST"] = args.cuda_arch
            command = [
                sys.executable,
                str(worktree / "scripts" / "run_release_evaluation.py"),
                "--inside-clean-worktree",
                "--expected-commit",
                commit,
                "--output-dir",
                str(output_dir),
                "--cuda-arch",
                args.cuda_arch,
            ]
            if args.allow_download:
                command.append("--allow-download")
            print(f"[release] detached worktree: {worktree}", flush=True)
            print(f"[release] source commit: {commit}", flush=True)
            completed = subprocess.run(command, env=env, check=False)
            return completed.returncode
        finally:
            subprocess.run(
                ["git", "worktree", "remove", "--force", str(worktree)],
                cwd=source_root,
                check=True,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cuda-arch", default="8.6")
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="允许模型下载；默认强制使用本地 Hugging Face cache",
    )
    parser.add_argument(
        "--inside-clean-worktree", action="store_true", help=argparse.SUPPRESS
    )
    parser.add_argument("--expected-commit", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.inside_clean_worktree:
        if not args.expected_commit:
            raise SystemExit("内部模式缺少 --expected-commit")
        run_inside_clean_worktree(args)
        return
    raise SystemExit(launch_isolated(args))


if __name__ == "__main__":
    main()
