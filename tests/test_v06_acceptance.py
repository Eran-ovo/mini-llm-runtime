import hashlib
import json
from pathlib import Path

from scripts.check_v06_acceptance import lookup_path, validate_manifest


def write_fixture(root: Path, relative_path: str, content: str) -> str:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return hashlib.sha256(content.encode()).hexdigest()


def make_manifest(root: Path) -> dict:
    for path in ("src/code.py", "tests/test_code.py", "docs/design.md"):
        write_fixture(root, path, "evidence")
    artifact = {"gate": {"passed": True}}
    artifact_text = json.dumps(artifact)
    artifact_sha = write_fixture(root, "results/result.json", artifact_text)
    write_fixture(root, "results/report.md", "report")
    return {
        "schema_version": 1,
        "milestone": "v0.6-test",
        "decision": "frozen",
        "requirements": [
            {
                "id": "feature",
                "implementation": ["src/code.py"],
                "tests": ["tests/test_code.py"],
                "docs": ["docs/design.md"],
            }
        ],
        "artifacts": [
            {
                "id": "correctness",
                "classification": "formal_correctness",
                "path": "results/result.json",
                "sha256": artifact_sha,
                "required_files": ["results/report.md"],
                "assertions": [
                    {"path": ["gate", "passed"], "equals": True}
                ],
            }
        ],
        "known_limitations": [],
    }


def test_acceptance_manifest_passes_complete_evidence(tmp_path: Path) -> None:
    assert validate_manifest(make_manifest(tmp_path), tmp_path) == []


def test_acceptance_manifest_detects_tampering_and_failed_assertion(
    tmp_path: Path,
) -> None:
    manifest = make_manifest(tmp_path)
    (tmp_path / "results/result.json").write_text(
        json.dumps({"gate": {"passed": False}}), encoding="utf-8"
    )
    errors = validate_manifest(manifest, tmp_path)
    assert any("SHA-256" in error for error in errors)
    assert any("actual=False" in error for error in errors)


def test_acceptance_manifest_detects_missing_evidence(tmp_path: Path) -> None:
    manifest = make_manifest(tmp_path)
    (tmp_path / "docs/design.md").unlink()
    errors = validate_manifest(manifest, tmp_path)
    assert any("evidence 不存在" in error for error in errors)


def test_lookup_path_rejects_missing_nested_key() -> None:
    try:
        lookup_path({"a": {}}, ["a", "b"])
    except KeyError as error:
        assert error.args == ("a.b",)
    else:
        raise AssertionError("缺失字段必须抛出 KeyError")


def test_acceptance_manifest_rejects_empty_or_smoke_evidence(tmp_path: Path) -> None:
    empty = {
        "schema_version": 1,
        "decision": "frozen",
        "requirements": [],
        "artifacts": [],
    }
    errors = validate_manifest(empty, tmp_path)
    assert any("至少需要一个 requirement" in error for error in errors)
    assert any("至少需要一个 artifact" in error for error in errors)

    manifest = make_manifest(tmp_path)
    artifact = manifest["artifacts"][0]
    artifact["path"] = "results/smoke/result.json"
    errors = validate_manifest(manifest, tmp_path)
    assert any("smoke 结果" in error for error in errors)
