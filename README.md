# T2I-BiasBench (Modular Research Pipeline)

This repository was refactored from notebook-only scripts into a **modular, reproducible research codebase** for text-to-image fairness and bias evaluation.

## What this refactor adds

- Config-driven experiments (YAML per model/dataset)
- Reusable `src/` package with separated responsibilities
- Single CLI for one-run or full-study execution
- Optional bootstrap confidence intervals for scalar metrics
- Standardized CSV outputs for paper tables/figures
- Basic unit tests for metric sanity checks

## Package layout

```text
src/t2i_biasbench/
  config.py        # Dataset/study config loaders
  io_utils.py      # Robust CSV ingestion + column resolution
  text_utils.py    # text normalization and prompt detection
  extractors.py    # gender/ethnicity/skin/animal/insect extractors
  metrics.py       # core metric formulas + bootstrap CI
  evaluate.py      # prompt-wise evaluation orchestration (13 metrics)
  reporting.py     # output helpers
  cli.py           # command-line entrypoint

configs/
  datasets/*.yaml  # model-specific dataset settings
  study.yaml       # multi-model orchestration config
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Run a single model

```bash
python -m t2i_biasbench.cli run --config configs/datasets/bksdm.yaml
```

With bootstrap confidence intervals (research mode):

```bash
python -m t2i_biasbench.cli run \
  --config configs/datasets/bksdm.yaml \
  --bootstrap-samples 1000 \
  --seed 42
```

## Run all models together

```bash
python -m t2i_biasbench.cli study --config configs/study.yaml
```

This writes:

- Per-model outputs declared in each dataset config
- Combined study table: `outputs/study_combined_metrics_modular.csv`
- Composite summary table: `outputs/study_composite_summary_modular.csv`

## Implemented metrics (13)

Original metrics:

1. Representation Parity
2. Parity Difference
3. Bias Amplification
4. Shannon Entropy
5. KL Divergence from Uniform
6. CAS (Contextual Association Score)
7. Composite Bias Score

Extended metrics:

8. GMR (Grounded Missing Rate)
9. IEMR (Implicit Element Missing Rate)
10. Hallucination Score
11. Vendi Score (lexical proxy)
12. CLIP Proxy Score
13. Cultural Accuracy Ratio

## Conference-readiness checklist

- Fix random seeds (`--seed`) in all reported runs
- Report confidence intervals for key scalar metrics
- Keep dataset/model configs version-controlled in `configs/`
- Export all final tables from `outputs/` and reference exact config files in appendix
- Add ablations by cloning a dataset config and adjusting lexicons/rules

## Notes

- Existing notebooks are preserved; this pipeline is the reproducible path for paper experiments.
- `configs/datasets/sd_template.yaml` is included as a template when `SD.csv` is available.
