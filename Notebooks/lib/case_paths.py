"""Resolve case ``metrics/`` and ``figures/`` dirs without fragile local variables."""

from __future__ import annotations

from pathlib import Path


def resolve_case_dir(family: str, case: str) -> Path:
    """Return ``Notebooks/{family}/{case}``, creating it only if found under a parent."""
    for base in [Path.cwd().resolve(), *Path.cwd().resolve().parents]:
        case_dir = base / "Notebooks" / family / case
        if case_dir.is_dir():
            return case_dir
    case_dir = Path.cwd().resolve() / "Notebooks" / family / case
    case_dir.mkdir(parents=True, exist_ok=True)
    return case_dir


def resolve_metrics_dir(family: str, case: str) -> Path:
    metrics_dir = resolve_case_dir(family, case) / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    return metrics_dir


def resolve_figures_dir(family: str, case: str) -> Path:
    figures_dir = resolve_case_dir(family, case) / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir
