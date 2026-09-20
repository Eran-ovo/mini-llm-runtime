import re
from html.parser import HTMLParser
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def local_target(source: Path, raw_target: str) -> Path | None:
    target = raw_target.split("#", 1)[0]
    if not target or "://" in target or target.startswith("mailto:"):
        return None
    return (source.parent / target).resolve()


def test_core_markdown_local_links_exist() -> None:
    sources = (
        REPO_ROOT / "README.md",
        REPO_ROOT / "docs/architecture.md",
        REPO_ROOT / "docs/evidence_index.md",
    )
    missing = []
    for source in sources:
        text = source.read_text(encoding="utf-8")
        for label, target in re.findall(r"\[([^]]+)\]\(([^)]+)\)", text):
            path = local_target(source, target)
            if path is not None and not path.exists():
                missing.append((str(source.relative_to(REPO_ROOT)), label, target))
    assert missing == []


class LinkCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        if tag != "a":
            return
        for name, value in attrs:
            if name == "href" and value is not None:
                self.hrefs.append(value)


def test_runtime_flow_links_exist_and_status_is_current() -> None:
    source = REPO_ROOT / "docs/vllm_runtime_flow.html"
    text = source.read_text(encoding="utf-8")
    parser = LinkCollector()
    parser.feed(text)
    missing = []
    for target in parser.hrefs:
        path = local_target(source, target)
        if path is not None and not path.exists():
            missing.append(target)
    assert missing == []

    stale_claims = (
        "Scheduler → ModelRunner 自动 Engine 编排尚未接通",
        "Engine 编排层仍是缺失的最后桥梁",
        "仓库还没有 Engine 自动拆分",
        "尚无稳定入口",
    )
    assert all(claim not in text for claim in stale_claims)
    assert "ContinuousBatchEngine" in text
    assert "../src/mini_llm_runtime/engine.py" in text
