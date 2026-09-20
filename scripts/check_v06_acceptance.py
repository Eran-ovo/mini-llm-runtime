#!/usr/bin/env python3
"""验证 v0.6 requirement evidence 与本地正式 artifact 白名单。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_inside(repo_root: Path, relative_path: str) -> Path:
    path = (repo_root / relative_path).resolve()
    if not path.is_relative_to(repo_root.resolve()):
        raise ValueError(f"manifest 路径越出仓库：{relative_path}")
    return path


def lookup_path(document: Any, keys: list[str]) -> Any:
    current = document
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            raise KeyError(".".join(keys))
        current = current[key]
    return current


def validate_manifest(manifest: dict[str, Any], repo_root: Path) -> list[str]:
    errors: list[str] = []
    if manifest.get("schema_version") != 1:
        errors.append("manifest schema_version 必须为 1")
    if manifest.get("decision") != "frozen":
        errors.append("milestone decision 必须为 frozen")

    requirements = manifest.get("requirements", [])
    if not isinstance(requirements, list) or not requirements:
        errors.append("manifest 至少需要一个 requirement")
        requirements = []
    requirement_ids = [item.get("id") for item in requirements]
    if len(requirement_ids) != len(set(requirement_ids)):
        errors.append("requirement id 不能重复")

    for requirement in requirements:
        requirement_id = requirement.get("id", "<missing-id>")
        for group in ("implementation", "tests", "docs"):
            paths = requirement.get(group, [])
            if not paths:
                errors.append(f"requirement {requirement_id} 缺少 {group} evidence")
            for relative_path in paths:
                try:
                    path = resolve_inside(repo_root, relative_path)
                except ValueError as error:
                    errors.append(str(error))
                    continue
                if not path.is_file():
                    errors.append(
                        f"requirement {requirement_id} evidence 不存在：{relative_path}"
                    )

    artifacts = manifest.get("artifacts", [])
    if not isinstance(artifacts, list) or not artifacts:
        errors.append("manifest 至少需要一个 artifact")
        artifacts = []
    artifact_ids = [item.get("id") for item in artifacts]
    if len(artifact_ids) != len(set(artifact_ids)):
        errors.append("artifact id 不能重复")
    allowed_classifications = {
        "formal_benchmark",
        "formal_benchmark_repeat",
        "formal_correctness",
        "profiler_evidence",
    }

    for artifact in artifacts:
        artifact_id = artifact.get("id", "<missing-id>")
        classification = artifact.get("classification")
        if classification not in allowed_classifications:
            errors.append(
                f"artifact {artifact_id} classification 非法：{classification!r}"
            )
        relative_path = artifact.get("path")
        if not isinstance(relative_path, str):
            errors.append(f"artifact {artifact_id} 缺少 path")
            continue
        if "smoke" in Path(relative_path).parts:
            errors.append(f"artifact {artifact_id} 不能把 smoke 结果列为正式证据")
        try:
            path = resolve_inside(repo_root, relative_path)
        except ValueError as error:
            errors.append(str(error))
            continue
        if not path.is_file():
            errors.append(f"artifact {artifact_id} 不存在：{relative_path}")
            continue

        expected_sha = artifact.get("sha256")
        if expected_sha and sha256_file(path) != expected_sha:
            errors.append(f"artifact {artifact_id} SHA-256 不匹配：{relative_path}")

        for required_path in artifact.get("required_files", []):
            try:
                required = resolve_inside(repo_root, required_path)
            except ValueError as error:
                errors.append(str(error))
                continue
            if not required.is_file():
                errors.append(
                    f"artifact {artifact_id} 配套文件不存在：{required_path}"
                )

        assertions = artifact.get("assertions", [])
        if not assertions:
            continue
        if path.suffix != ".json":
            errors.append(f"artifact {artifact_id} 非 JSON，不能执行 assertions")
            continue
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"artifact {artifact_id} JSON 无法读取：{error}")
            continue
        for assertion in assertions:
            keys = assertion.get("path")
            if not isinstance(keys, list) or not all(
                isinstance(key, str) for key in keys
            ):
                errors.append(f"artifact {artifact_id} assertion path 非法")
                continue
            try:
                actual = lookup_path(document, keys)
            except KeyError:
                errors.append(
                    f"artifact {artifact_id} 缺少字段：{'.'.join(keys)}"
                )
                continue
            expected = assertion.get("equals")
            if actual != expected:
                errors.append(
                    f"artifact {artifact_id} 字段 {'.'.join(keys)}："
                    f"actual={actual!r}, expected={expected!r}"
                )
    return errors


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=repo_root / "milestones" / "v0.6.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    errors = validate_manifest(manifest, repo_root)
    if errors:
        print(f"v0.6 acceptance: FAILED ({len(errors)} errors)")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)

    print(f"v0.6 acceptance: PASSED ({manifest['milestone']})")
    print(f"requirements: {len(manifest['requirements'])}")
    print(f"whitelisted artifacts: {len(manifest['artifacts'])}")
    print("known limitations:")
    for limitation in manifest["known_limitations"]:
        print(f"- {limitation}")


if __name__ == "__main__":
    main()
