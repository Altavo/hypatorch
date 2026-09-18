"""Contracts for the shipped agent docs bundle."""

from __future__ import annotations

import inspect
import json
import os
import re
import tomllib
from importlib import import_module
from pathlib import Path

PACKAGE = "hypatorch"

# Everything a config can name. A change here means agent_docs/ may now be wrong.
DOCUMENTED_SURFACE: list[str] = [
    "hypatorch.core.Model",
    "hypatorch.core.Scheduler",
    "hypatorch.train.Trainer",
    "hypatorch.assessments.HypaAssessment",
    "hypatorch.assessments.MaskedAssessment",
    "hypatorch.losses.MAE_Loss",
    "hypatorch.losses.MMAE_Loss",
    "hypatorch.losses.MSE_Loss",
    "hypatorch.losses.MMSE_Loss",
    "hypatorch.logger.ConsoleLogger",
    "hypatorch.logger.WandbLogger",
    "hypatorch.logger.MLflowLogger",
]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCES = PROJECT_ROOT / "src" if (PROJECT_ROOT / "src").is_dir() else PROJECT_ROOT
BUNDLE = SOURCES / PACKAGE / "agent_docs"
SNAPSHOT = Path(__file__).with_name("surface.json")
RELATIVE_LINK = re.compile(r"\]\((?!https?://|#)([^)]+)\)")


def _surface_names(target) -> list[str]:
    """The names a config writes: a model's fields, or a callable's parameters."""
    fields = getattr(target, "model_fields", None)
    if isinstance(fields, dict):
        return sorted(fields)

    subject = target if inspect.isroutine(target) else target.__init__
    parameters = inspect.signature(subject).parameters
    return sorted(
        name
        for name, parameter in parameters.items()
        if name not in {"self", "cls"}
        and parameter.kind
        not in {parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD}
    )


def _current_surface() -> dict[str, list[str]]:
    surface = {}
    for dotted in DOCUMENTED_SURFACE:
        module_name, _, attribute = dotted.rpartition(".")
        surface[dotted] = _surface_names(getattr(import_module(module_name), attribute))
    return surface


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


def test_documented_surface_is_unchanged():
    current = _current_surface()
    if os.environ.get("UPDATE_SURFACE"):
        SNAPSHOT.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")

    repin = f"re-pin with UPDATE_SURFACE=1 pytest {Path(__file__).name}"
    assert SNAPSHOT.is_file(), f"no surface snapshot yet: {repin}"

    pinned = json.loads(SNAPSHOT.read_text())
    moved = {
        name: {
            "pinned": pinned.get(name),
            "current": current.get(name),
        }
        for name in sorted(set(pinned) | set(current))
        if pinned.get(name) != current.get(name)
    }
    assert not moved, (
        "the configurable surface changed, so agent_docs/ may now be wrong. "
        f"Update the affected page, then {repin}.\n"
        + json.dumps(moved, indent=2)
    )
