"""Reporting helpers for CSV outputs and concise console summaries."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def ensure_parent_dir(path: Path) -> None:
    """Create parent directories for an output file if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)


def save_results(results: pd.DataFrame, output_csv: Path) -> None:
    """Write full metric table to CSV."""
    ensure_parent_dir(output_csv)
    results.to_csv(output_csv, index=False)


def build_composite_summary(results: pd.DataFrame) -> pd.DataFrame:
    """Extract composite score rows for quick model comparisons."""
    subset = results[results["Metric"].str.contains("COMPOSITE", na=False)].copy()
    columns = ["Model", "Prompt", "Metric", "Value", "CI_Low", "CI_High", "N", "Note"]
    existing_cols = [col for col in columns if col in subset.columns]
    return subset[existing_cols].reset_index(drop=True)
