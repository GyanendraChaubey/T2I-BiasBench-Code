# T2I-BiasBench-Code

T2I-BiasBench-Code is a **modular, reproducible research codebase** for text-to-image fairness and bias evaluation.

## What this repository provides

- Config-driven experiments (YAML per model/dataset)
- Reusable `src/` package with separated responsibilities
- Single CLI for one-run or full-study execution
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
