"""Configuration models and YAML loaders for reproducible runs."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .constants import DEFAULT_PROMPT_RULES


ColumnSpec = int | str


@dataclass
class DatasetConfig:
    """Single-model experiment configuration."""

    model_name: str
    input_csv: Path
    output_csv: Path
    image_column: ColumnSpec = "Image"
    caption_column: ColumnSpec = "Caption"
    prompt_rules: dict[str, list[str]] = field(default_factory=lambda: deepcopy(DEFAULT_PROMPT_RULES))


@dataclass
class StudyConfig:
    """Multi-model study orchestration configuration."""

    run_configs: list[Path]
    combined_output: Path
    composite_output: Path


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping at top-level in {path}")
    return data


def _resolve_path(path_like: str | Path, base_dir: Path) -> Path:
    candidate = Path(path_like)
    if candidate.is_absolute():
        return candidate
    return (base_dir / candidate).resolve()


def _validate_prompt_rules(raw_rules: Any) -> dict[str, list[str]]:
    if raw_rules is None:
        return deepcopy(DEFAULT_PROMPT_RULES)
    if not isinstance(raw_rules, dict):
        raise ValueError("prompt_rules must be a mapping of prompt -> keyword list")

    parsed: dict[str, list[str]] = {}
    for prompt, keywords in raw_rules.items():
        if isinstance(keywords, str):
            parsed[str(prompt)] = [keywords.lower()]
            continue
        if not isinstance(keywords, list):
            raise ValueError(f"prompt_rules['{prompt}'] must be list[str]")
        parsed[str(prompt)] = [str(keyword).lower() for keyword in keywords]
    return parsed


def load_dataset_config(config_path: str | Path) -> DatasetConfig:
    """Load one dataset config YAML file."""
    path = Path(config_path).resolve()
    data = _load_yaml(path)
    base = path.parent

    try:
        model_name = str(data["model_name"])
        input_csv = _resolve_path(data["input_csv"], base)
        output_csv = _resolve_path(data["output_csv"], base)
    except KeyError as exc:
        raise ValueError(f"Missing required key: {exc.args[0]}") from exc

    prompt_rules = _validate_prompt_rules(data.get("prompt_rules"))

    return DatasetConfig(
        model_name=model_name,
        input_csv=input_csv,
        output_csv=output_csv,
        image_column=data.get("image_column", "Image"),
        caption_column=data.get("caption_column", "Caption"),
        prompt_rules=prompt_rules,
    )


def load_study_config(config_path: str | Path) -> StudyConfig:
    """Load a study config that references multiple dataset config files."""
    path = Path(config_path).resolve()
    data = _load_yaml(path)
    base = path.parent

    runs = data.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("study config must include a non-empty 'runs' list")

    run_configs: list[Path] = []
    for run in runs:
        if isinstance(run, dict) and "config" in run:
            run_configs.append(_resolve_path(run["config"], base))
        elif isinstance(run, str):
            run_configs.append(_resolve_path(run, base))
        else:
            raise ValueError("Each run must be either a path string or {config: <path>}")

    combined_output = _resolve_path(data.get("combined_output", "outputs/study_combined_metrics.csv"), base)
    composite_output = _resolve_path(data.get("composite_output", "outputs/study_composite_summary.csv"), base)

    return StudyConfig(
        run_configs=run_configs,
        combined_output=combined_output,
        composite_output=composite_output,
    )
