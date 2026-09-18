"""Contracts for the shipped agent docs bundle."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

PACKAGE = "hypatorch"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCES = PROJECT_ROOT / "src" if (PROJECT_ROOT / "src").is_dir() else PROJECT_ROOT
BUNDLE = SOURCES / PACKAGE / "agent_docs"
RELATIVE_LINK = re.compile(r"\]\((?!https?://|#)([^)]+)\)")


def test_links_resolve_inside_the_bundle():
    for document in BUNDLE.rglob("*.md"):
        for target in RELATIVE_LINK.findall(document.read_text()):
            resolved = (document.parent / target.split("#", 1)[0]).resolve()
            assert resolved.exists(), f"{document}: {target}"
            assert resolved.is_relative_to(BUNDLE), f"{document}: {target}"


def test_every_document_says_when_to_read_it():
    assert (BUNDLE / "OVERVIEW.md").is_file()

    for document in BUNDLE.rglob("*.md"):
        head = document.read_text().split("\n\n")[:2]
        assert any(part.startswith("Use this when ") for part in head), document


def test_bundle_is_declared_as_package_data():
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as handle:
        project = tomllib.load(handle)

    package_data = project.get("tool", {}).get("setuptools", {}).get("package-data", {})
    assert package_data.get(PACKAGE) == ["agent_docs/*", "agent_docs/**/*"]
