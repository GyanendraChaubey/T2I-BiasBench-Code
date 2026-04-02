"""Command-line entrypoint for modular T2I-BiasBench experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .config import load_dataset_config, load_study_config
from .evaluate import evaluate_model
from .io_utils import load_caption_dataset
from .reporting import build_composite_summary, save_results


def run_single(config_path: str | Path, bootstrap_samples: int = 0, seed: int = 42) -> tuple[pd.DataFrame, Path]:
    """Run one configured model evaluation and save results."""
    config = load_dataset_config(config_path)
    frame = load_caption_dataset(config)
    results = evaluate_model(
        frame,
        model_name=config.model_name,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )

    save_results(results, config.output_csv)
    return results, config.output_csv


def run_study(config_path: str | Path, bootstrap_samples: int = 0, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, Path, Path]:
    """Run all dataset configs declared in a study YAML file."""
    study = load_study_config(config_path)

    all_results: list[pd.DataFrame] = []
    for run_config in study.run_configs:
        run_df, run_output = run_single(run_config, bootstrap_samples=bootstrap_samples, seed=seed)
        print(f"[run] saved: {run_output}")
        all_results.append(run_df)

    combined = pd.concat(all_results, axis=0, ignore_index=True)
    composite = build_composite_summary(combined)

    save_results(combined, study.combined_output)
    save_results(composite, study.composite_output)

    return combined, composite, study.combined_output, study.composite_output


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="t2i-biasbench",
        description="Modular fairness/bias evaluation for text-to-image caption datasets.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run one dataset config")
    run_parser.add_argument("--config", required=True, help="Path to dataset YAML config")
    run_parser.add_argument("--bootstrap-samples", type=int, default=0, help="Bootstrap samples for confidence intervals")
    run_parser.add_argument("--seed", type=int, default=42, help="Random seed for bootstrap")

    study_parser = subparsers.add_parser("study", help="Run multiple dataset configs")
    study_parser.add_argument("--config", required=True, help="Path to study YAML config")
    study_parser.add_argument("--bootstrap-samples", type=int, default=0, help="Bootstrap samples for confidence intervals")
    study_parser.add_argument("--seed", type=int, default=42, help="Random seed for bootstrap")

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "run":
        results, output_path = run_single(args.config, bootstrap_samples=args.bootstrap_samples, seed=args.seed)
        print(f"[run] rows={len(results)}")
        print(f"[run] saved: {output_path}")
        print("\n[run] composite summary")
        print(build_composite_summary(results).to_string(index=False))
        return

    combined, composite, combined_out, composite_out = run_study(
        args.config,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    print(f"[study] total rows={len(combined)}")
    print(f"[study] saved combined: {combined_out}")
    print(f"[study] saved composite: {composite_out}")
    print("\n[study] composite summary")
    print(composite.to_string(index=False))


if __name__ == "__main__":
    main()
