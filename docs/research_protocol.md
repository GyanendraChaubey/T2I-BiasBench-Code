# Research Protocol (CVPR/AAAI/NeurIPS Readiness)

## 1. Reproducibility contract

- Pin code commit and dataset snapshot hash in every experiment table.
- Run all experiments via YAML configs under `configs/`.
- Fix `--seed` and log it in paper appendix.
- Export raw per-metric CSVs and derived summary tables in `outputs/`.

## 2. Evaluation protocol

- Use identical prompt buckets across models: `beauty`, `doctor`, `animal`, `nature`, `culture`.
- Keep the same lexicons for all models unless reporting an explicit ablation.
- Report all 13 metrics and prioritize composite + core fairness metrics in main paper.
- Include confidence intervals for scalar metrics using `--bootstrap-samples`.

## 3. Statistical reporting

- For primary claims, report point estimate plus 95% bootstrap CI.
- Avoid claiming superiority when confidence intervals overlap substantially.
- Keep full metric distributions in supplementary material.

## 4. Recommended ablations

- Prompt classifier sensitivity (`neutral` vs `animal` keyword rules).
- Lexicon sensitivity (strict vs expanded ethnicity/cultural term sets).
- Caption cleaning sensitivity (with/without stemming or lemmatization).
- Vendi proxy sensitivity (`top_n` pair sampling size).

## 5. Artifact checklist

- `README.md` with exact run commands.
- Dataset config files used for each table/figure.
- Combined output CSV from study mode.
- Scripted plotting notebooks/scripts that only consume `outputs/*.csv`.

## 6. Threats to validity (to include in paper)

- Metrics are caption-text proxies, not direct visual embeddings.
- Lexicon-based extraction can under-detect nuanced identity attributes.
- Prompt-name heuristics depend on filename conventions.

## 7. Submission-ready command set

```bash
python -m t2i_biasbench.cli study \
  --config configs/study.yaml \
  --bootstrap-samples 1000 \
  --seed 42
```

This should be the canonical command for final reported numbers.
