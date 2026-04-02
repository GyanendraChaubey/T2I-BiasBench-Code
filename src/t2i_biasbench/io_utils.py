"""Dataset loading and normalization utilities."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from .config import DatasetConfig
from .text_utils import clean_text, detect_prompt


def _normalize_col(name: str) -> str:
    cleaned = str(name).strip().lower()
    cleaned = re.sub(r"[\s_\-]+", "", cleaned)
    return cleaned


def _resolve_column(df: pd.DataFrame, spec: int | str, kind: str) -> str:
    columns = list(df.columns)

    if isinstance(spec, int):
        if spec < 0 or spec >= len(columns):
            raise ValueError(f"{kind}_column index {spec} is out of range for columns {columns}")
        return columns[spec]

    if isinstance(spec, str) and spec.isdigit():
        idx = int(spec)
        if 0 <= idx < len(columns):
            return columns[idx]

    if spec in columns:
        return spec  # exact name match

    wanted = _normalize_col(spec)
    normalized = {_normalize_col(col): col for col in columns}
    if wanted in normalized:
        return normalized[wanted]

    raise ValueError(f"Could not resolve {kind}_column='{spec}'. Available columns: {columns}")


def load_caption_dataset(config: DatasetConfig) -> pd.DataFrame:
    """Load and normalize one caption dataset according to config."""
    csv_path = Path(config.input_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    try:
        raw = pd.read_csv(csv_path)
    except Exception:
        raw = pd.read_csv(csv_path, engine="python")

    image_col = _resolve_column(raw, config.image_column, "image")
    caption_col = _resolve_column(raw, config.caption_column, "caption")

    frame = raw[[image_col, caption_col]].copy()
    frame.columns = ["image", "caption"]
    frame = frame.dropna(subset=["image", "caption"])

    frame["image"] = frame["image"].astype(str).str.strip()
    frame["caption"] = frame["caption"].astype(str).str.strip()
    frame["cap"] = frame["caption"].apply(clean_text)
    frame["prompt"] = frame["image"].apply(lambda x: detect_prompt(x, config.prompt_rules))

    return frame
