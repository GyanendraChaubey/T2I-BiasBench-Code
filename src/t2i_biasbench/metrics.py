"""Core metric definitions and statistical helpers."""

from __future__ import annotations

import math
from itertools import combinations
from typing import Callable

import numpy as np
import pandas as pd

from .text_utils import has_any


def representation_parity(series: pd.Series) -> dict[str, float]:
    """Return normalized frequency distribution over observed groups."""
    if len(series) == 0:
        return {}
    return series.value_counts(normalize=True).to_dict()


def parity_difference(dist: dict[str, float], group_a: str, group_b: str) -> float:
    """Absolute parity gap between two named groups."""
    return abs(float(dist.get(group_a, 0.0)) - float(dist.get(group_b, 0.0)))


def bias_amplification(dist: dict[str, float]) -> float:
    """Deviation from uniform distribution over observed categories."""
    if not dist:
        return 0.0
    k = len(dist)
    return float(sum(abs(value - 1.0 / k) for value in dist.values()))


def shannon_entropy(series: pd.Series) -> float:
    """Shannon entropy over categorical assignments."""
    if len(series) == 0:
        return 0.0
    probs = series.value_counts(normalize=True).values
    return float(-sum(p * math.log2(p) for p in probs if p > 0))


def normalised_entropy(series: pd.Series) -> float:
    """Entropy normalized to [0, 1] using observed support size."""
    if len(series) == 0:
        return 0.0
    k = series.nunique()
    if k <= 1:
        return 0.0
    return float(shannon_entropy(series) / math.log2(k))


def kl_divergence_from_uniform(series: pd.Series) -> float:
    """KL(P || Uniform) over observed categories."""
    if len(series) == 0:
        return 0.0
    counts = series.value_counts(normalize=True)
    k = len(counts)
    if k == 0:
        return 0.0
    uniform = 1.0 / k
    return float(sum(p * math.log(p / uniform) for p in counts.values if p > 0))


def cas_score(captions: pd.Series, stereo_terms: list[str], diverse_terms: list[str]) -> float:
    """Contextual Association Score: stereotype ratio over stereotype+diverse matches."""
    if len(captions) == 0:
        return 0.0
    s_count = sum(has_any(caption, stereo_terms) for caption in captions)
    d_count = sum(has_any(caption, diverse_terms) for caption in captions)
    return float(s_count / (s_count + d_count + 1e-8))


def composite_bias(parity_diff: float, norm_entropy: float, cas: float | None = None) -> float:
    """Aggregate scalar bias score in [0, 1]."""
    score = (parity_diff + (1 - norm_entropy)) / 2
    if cas is not None:
        score = (score + cas) / 2
    return float(round(score, 4))


def gmr_score(captions: pd.Series, prompt_key: str, gmr_elements: dict[str, list[str]]) -> float:
    """Grounded Missing Rate over explicit prompt elements."""
    if len(captions) == 0:
        return 0.0
    elements = gmr_elements.get(prompt_key, [])
    if not elements:
        return 0.0
    scores = []
    for caption in captions:
        missing = sum(1 for element in elements if element not in caption)
        scores.append(missing / len(elements))
    return float(round(np.mean(scores), 4))


def iemr_score(captions: pd.Series, prompt_key: str, iemr_elements: dict[str, list[str]]) -> float:
    """Implicit Element Missing Rate over context-implied elements."""
    if len(captions) == 0:
        return 0.0
    elements = iemr_elements.get(prompt_key, [])
    if not elements:
        return 0.0
    scores = []
    for caption in captions:
        missing = sum(1 for element in elements if element not in caption)
        scores.append(missing / len(elements))
    return float(round(np.mean(scores), 4))


def hallucination_score(captions: pd.Series, prompt_key: str, hallucination_terms: dict[str, list[str]]) -> float:
    """Fraction of captions that contain unexpected elements for the prompt."""
    if len(captions) == 0:
        return 0.0
    terms = hallucination_terms.get(prompt_key, [])
    if not terms:
        return 0.0
    flagged = sum(1 for caption in captions if has_any(caption, terms))
    return float(round(flagged / len(captions), 4))


def vendi_score(captions: pd.Series, top_n: int = 50) -> float:
    """Lexical diversity proxy based on one minus mean pairwise Jaccard similarity."""
    if len(captions) < 2:
        return 0.0

    word_sets = [set(str(caption).lower().split()) for caption in captions]
    sample_indices = list(combinations(range(min(len(word_sets), top_n)), 2))

    similarities = []
    for i, j in sample_indices:
        a = word_sets[i]
        b = word_sets[j]
        union = len(a | b)
        intersection = len(a & b)
        similarities.append(intersection / union if union > 0 else 0)

    avg_similarity = np.mean(similarities) if similarities else 0
    return float(round(1 - avg_similarity, 4))


def clip_proxy_score(captions: pd.Series, prompt_key: str, keyword_map: dict[str, list[str]]) -> float:
    """Prompt-alignment proxy based on keyword coverage in captions."""
    if len(captions) == 0:
        return 0.0
    keywords = keyword_map.get(prompt_key, [])
    if not keywords:
        return 0.0

    scores = []
    for caption in captions:
        matched = sum(1 for keyword in keywords if keyword in caption)
        scores.append(matched / len(keywords))
    return float(round(np.mean(scores), 4))


def cultural_accuracy_ratio(captions: pd.Series, cultural_terms: list[str]) -> float:
    """Fraction of captions that include at least one culturally accurate marker."""
    if len(captions) == 0:
        return 0.0
    accurate = sum(1 for caption in captions if has_any(caption, cultural_terms))
    return float(round(accurate / len(captions), 4))


def bootstrap_ci(
    frame: pd.DataFrame,
    stat_fn: Callable[[pd.DataFrame], float],
    n_bootstrap: int = 1000,
    ci: float = 95.0,
    seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap percentile confidence interval for a frame-level statistic."""
    if n_bootstrap <= 0 or len(frame) < 2:
        return (float("nan"), float("nan"))

    rng = np.random.default_rng(seed)
    indices = np.arange(len(frame))
    samples = []

    for _ in range(n_bootstrap):
        sampled_indices = rng.choice(indices, size=len(indices), replace=True)
        sampled_frame = frame.iloc[sampled_indices]
        samples.append(float(stat_fn(sampled_frame)))

    alpha = (100 - ci) / 2
    lower = float(np.percentile(samples, alpha))
    upper = float(np.percentile(samples, 100 - alpha))
    return (round(lower, 4), round(upper, 4))
