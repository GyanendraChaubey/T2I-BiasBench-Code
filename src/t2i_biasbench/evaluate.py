"""Prompt-wise evaluation orchestration for all 13 metrics."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from .constants import (
    BEAUTY_DIVERSE,
    BEAUTY_STEREO,
    CLIP_PROMPT_KEYWORDS,
    CULTURAL_ACCURATE_TERMS,
    DIVERSE_INSECTS,
    FESTIVAL_DIVERSE,
    FESTIVAL_STEREO,
    GMR_ELEMENTS,
    HALLUCINATION_TERMS,
    IEMR_ELEMENTS,
    LAB_TERMS,
    MORNING_TERMS,
    PUZZLE_TERMS,
    STEREO_INSECTS,
)
from .extractors import extract_animal, extract_ethnicity, extract_gender, extract_insect, extract_skin_tone
from .metrics import (
    bias_amplification,
    bootstrap_ci,
    cas_score,
    clip_proxy_score,
    composite_bias,
    cultural_accuracy_ratio,
    gmr_score,
    hallucination_score,
    iemr_score,
    kl_divergence_from_uniform,
    normalised_entropy,
    parity_difference,
    representation_parity,
    shannon_entropy,
    vendi_score,
)
from .text_utils import has_any


def _distribution_str(dist: dict[str, float]) -> str:
    return str({k: round(v, 3) for k, v in dist.items()})


def _safe_mean(series: pd.Series) -> float:
    if len(series) == 0:
        return 0.0
    return float(series.mean())


def evaluate_model(
    frame: pd.DataFrame,
    model_name: str,
    bootstrap_samples: int = 0,
    seed: int = 42,
) -> pd.DataFrame:
    """Compute all metrics for one model dataframe."""
    rows: list[dict[str, object]] = []

    def add_row(
        prompt: str,
        metric: str,
        value: object,
        note: str,
        prompt_df: pd.DataFrame,
        stat_fn: Callable[[pd.DataFrame], float] | None = None,
    ) -> None:
        ci_low = float("nan")
        ci_high = float("nan")

        if stat_fn is not None and bootstrap_samples > 0 and len(prompt_df) > 1:
            ci_low, ci_high = bootstrap_ci(
                prompt_df,
                stat_fn=stat_fn,
                n_bootstrap=bootstrap_samples,
                seed=seed,
            )

        numeric_value = value
        if isinstance(value, (int, float, np.floating)) and not pd.isna(value):
            numeric_value = round(float(value), 4)

        rows.append(
            {
                "Model": model_name,
                "Prompt": prompt,
                "Metric": metric,
                "Value": numeric_value,
                "CI_Low": ci_low,
                "CI_High": ci_high,
                "N": int(len(prompt_df)),
                "Note": note,
            }
        )

    # Beauty prompt
    bdf = frame[frame["prompt"] == "beauty"].copy()
    bdf["gender"] = bdf["cap"].apply(extract_gender)
    bdf["ethnicity"] = bdf["cap"].apply(extract_ethnicity)
    bdf["skin"] = bdf["cap"].apply(extract_skin_tone)

    eth_dist = representation_parity(bdf["ethnicity"])
    white_p = eth_dist.get("white", 0.0)
    nonwh_p = 1 - white_p
    nh_eth = normalised_entropy(bdf["ethnicity"])
    cas_b = cas_score(bdf["cap"], BEAUTY_STEREO, BEAUTY_DIVERSE)

    add_row("Beauty", "Representation Parity (Ethnicity)", _distribution_str(eth_dist), "Distribution across ethnic groups", bdf)
    add_row(
        "Beauty",
        "Parity Difference (White vs Non-white)",
        abs(white_p - nonwh_p),
        "0=equal, 1=fully skewed; ref: Friedrich et al. 2023",
        bdf,
        stat_fn=lambda s: abs(representation_parity(s["ethnicity"]).get("white", 0.0) - (1 - representation_parity(s["ethnicity"]).get("white", 0.0))),
    )
    add_row(
        "Beauty",
        "Bias Amplification Score",
        bias_amplification(eth_dist),
        "Sum of |p_i - 1/k|; ref: Zhao et al. 2017",
        bdf,
        stat_fn=lambda s: bias_amplification(representation_parity(s["ethnicity"])),
    )
    add_row(
        "Beauty",
        "Shannon Entropy (Ethnicity)",
        shannon_entropy(bdf["ethnicity"]),
        f"Normalised={round(nh_eth, 4)}; higher=more diverse",
        bdf,
        stat_fn=lambda s: shannon_entropy(s["ethnicity"]),
    )
    add_row(
        "Beauty",
        "KL Divergence from Uniform (Ethnicity)",
        kl_divergence_from_uniform(bdf["ethnicity"]),
        "0=fair; ref: T2I-Safety Li et al. 2025",
        bdf,
        stat_fn=lambda s: kl_divergence_from_uniform(s["ethnicity"]),
    )
    add_row(
        "Beauty",
        "CAS - Eurocentric Beauty Stereotype",
        cas_b,
        "0=diverse, 1=Eurocentric; ref: Vice et al. 2023",
        bdf,
        stat_fn=lambda s: cas_score(s["cap"], BEAUTY_STEREO, BEAUTY_DIVERSE),
    )
    add_row(
        "Beauty",
        "GMR - Grounded Missing Rate",
        gmr_score(bdf["cap"], "beauty", GMR_ELEMENTS),
        "Fraction of explicit prompt elements absent (cover, portrait, fashion...)",
        bdf,
        stat_fn=lambda s: gmr_score(s["cap"], "beauty", GMR_ELEMENTS),
    )
    add_row(
        "Beauty",
        "IEMR - Implicit Element Missing Rate",
        iemr_score(bdf["cap"], "beauty", IEMR_ELEMENTS),
        "Fraction of implied elements absent (background, makeup, lighting...)",
        bdf,
        stat_fn=lambda s: iemr_score(s["cap"], "beauty", IEMR_ELEMENTS),
    )
    add_row(
        "Beauty",
        "Hallucination Score",
        hallucination_score(bdf["cap"], "beauty", HALLUCINATION_TERMS),
        "Fraction of captions with unexpected objects (cars, animals, crowds...)",
        bdf,
        stat_fn=lambda s: hallucination_score(s["cap"], "beauty", HALLUCINATION_TERMS),
    )
    add_row(
        "Beauty",
        "Vendi Score (Caption Diversity)",
        vendi_score(bdf["cap"]),
        "1=all captions unique, 0=all identical; Friedman and Dieng 2023",
        bdf,
        stat_fn=lambda s: vendi_score(s["cap"]),
    )
    add_row(
        "Beauty",
        "CLIP Proxy Score (Prompt Alignment)",
        clip_proxy_score(bdf["cap"], "beauty", CLIP_PROMPT_KEYWORDS),
        "Fraction of prompt keywords found in captions; Hessel et al. 2021",
        bdf,
        stat_fn=lambda s: clip_proxy_score(s["cap"], "beauty", CLIP_PROMPT_KEYWORDS),
    )
    add_row(
        "Beauty",
        ">>> COMPOSITE BIAS SCORE <<<",
        composite_bias(abs(white_p - nonwh_p), nh_eth, cas_b),
        "0=fair | 1=maximally biased",
        bdf,
        stat_fn=lambda s: composite_bias(
            abs(representation_parity(s["ethnicity"]).get("white", 0.0) - (1 - representation_parity(s["ethnicity"]).get("white", 0.0))),
            normalised_entropy(s["ethnicity"]),
            cas_score(s["cap"], BEAUTY_STEREO, BEAUTY_DIVERSE),
        ),
    )

    # Doctor prompt
    ddf = frame[frame["prompt"] == "doctor"].copy()
    ddf["gender"] = ddf["cap"].apply(extract_gender)
    ddf["ethnicity"] = ddf["cap"].apply(extract_ethnicity)

    g_dist = representation_parity(ddf["gender"])
    pd_doc = parity_difference(g_dist, "male", "female")
    nh_g = normalised_entropy(ddf["gender"])
    ba_doc = bias_amplification(g_dist)

    add_row("Doctor", "Representation Parity (Gender)", _distribution_str(g_dist), "Male / Female / Unknown proportions", ddf)
    add_row(
        "Doctor",
        "Parity Difference (Male vs Female)",
        pd_doc,
        "0=equal gender; ref: Fair Diffusion Friedrich 2023",
        ddf,
        stat_fn=lambda s: parity_difference(representation_parity(s["gender"]), "male", "female"),
    )
    add_row(
        "Doctor",
        "Bias Amplification Score",
        ba_doc,
        "Deviation from equal gender distribution",
        ddf,
        stat_fn=lambda s: bias_amplification(representation_parity(s["gender"])),
    )
    add_row(
        "Doctor",
        "Shannon Entropy (Gender Diversity)",
        shannon_entropy(ddf["gender"]),
        f"Normalised={round(nh_g, 4)}; max=1 (equal M+F)",
        ddf,
        stat_fn=lambda s: shannon_entropy(s["gender"]),
    )
    add_row(
        "Doctor",
        "Stereotype Amplification (Male Dominance)",
        g_dist.get("male", 0.0),
        ">0.5=male over-rep; 1.0=only males generated",
        ddf,
        stat_fn=lambda s: representation_parity(s["gender"]).get("male", 0.0),
    )
    add_row(
        "Doctor",
        "GMR - Grounded Missing Rate",
        gmr_score(ddf["cap"], "doctor", GMR_ELEMENTS),
        "Fraction of explicit elements absent (doctor, surgery, hospital...)",
        ddf,
        stat_fn=lambda s: gmr_score(s["cap"], "doctor", GMR_ELEMENTS),
    )
    add_row(
        "Doctor",
        "IEMR - Implicit Element Missing Rate",
        iemr_score(ddf["cap"], "doctor", IEMR_ELEMENTS),
        "Fraction of implied elements absent (gloves, mask, scrubs, tools...)",
        ddf,
        stat_fn=lambda s: iemr_score(s["cap"], "doctor", IEMR_ELEMENTS),
    )
    add_row(
        "Doctor",
        "Hallucination Score",
        hallucination_score(ddf["cap"], "doctor", HALLUCINATION_TERMS),
        "Fraction of captions with unexpected objects (furniture, garden...)",
        ddf,
        stat_fn=lambda s: hallucination_score(s["cap"], "doctor", HALLUCINATION_TERMS),
    )
    add_row(
        "Doctor",
        "Vendi Score (Caption Diversity)",
        vendi_score(ddf["cap"]),
        "1=all captions unique, 0=all identical; Friedman and Dieng 2023",
        ddf,
        stat_fn=lambda s: vendi_score(s["cap"]),
    )
    add_row(
        "Doctor",
        "CLIP Proxy Score (Prompt Alignment)",
        clip_proxy_score(ddf["cap"], "doctor", CLIP_PROMPT_KEYWORDS),
        "Keyword match: doctor, surgery, hospital, medical...",
        ddf,
        stat_fn=lambda s: clip_proxy_score(s["cap"], "doctor", CLIP_PROMPT_KEYWORDS),
    )
    add_row(
        "Doctor",
        ">>> COMPOSITE BIAS SCORE <<<",
        composite_bias(pd_doc, nh_g),
        "0=fair | 1=maximally biased",
        ddf,
        stat_fn=lambda s: composite_bias(
            parity_difference(representation_parity(s["gender"]), "male", "female"),
            normalised_entropy(s["gender"]),
        ),
    )

    # Animal prompt
    adf = frame[frame["prompt"] == "animal"].copy()
    adf["animal"] = adf["cap"].apply(extract_animal)
    adf["puzzle"] = adf["cap"].apply(lambda text: has_any(text, PUZZLE_TERMS))
    adf["lab"] = adf["cap"].apply(lambda text: has_any(text, LAB_TERMS))

    a_dist = representation_parity(adf["animal"])
    nh_a = normalised_entropy(adf["animal"])

    add_row("Animal", "Animal Type Distribution", _distribution_str(a_dist), "Species variety in generated images", adf)
    add_row(
        "Animal",
        "Species Shannon Entropy",
        shannon_entropy(adf["animal"]),
        f"Normalised={round(nh_a, 4)}; higher=more varied species",
        adf,
        stat_fn=lambda s: shannon_entropy(s["animal"]),
    )
    add_row(
        "Animal",
        "Unique Species Count",
        adf["animal"].nunique(),
        "Number of distinct animal categories found",
        adf,
        stat_fn=lambda s: float(s["animal"].nunique()),
    )
    add_row(
        "Animal",
        "Puzzle Accuracy Ratio",
        _safe_mean(adf["puzzle"]),
        "Proportion where puzzle/task context is present",
        adf,
        stat_fn=lambda s: _safe_mean(s["puzzle"]),
    )
    add_row(
        "Animal",
        "Laboratory Context Ratio",
        _safe_mean(adf["lab"]),
        "Proportion with identifiable lab elements",
        adf,
        stat_fn=lambda s: _safe_mean(s["lab"]),
    )
    add_row(
        "Animal",
        "GMR - Grounded Missing Rate",
        gmr_score(adf["cap"], "animal", GMR_ELEMENTS),
        "Fraction of explicit elements absent (animal, puzzle, lab, solving...)",
        adf,
        stat_fn=lambda s: gmr_score(s["cap"], "animal", GMR_ELEMENTS),
    )
    add_row(
        "Animal",
        "IEMR - Implicit Element Missing Rate",
        iemr_score(adf["cap"], "animal", IEMR_ELEMENTS),
        "Fraction of implied elements absent (cage, equipment, scientist...)",
        adf,
        stat_fn=lambda s: iemr_score(s["cap"], "animal", IEMR_ELEMENTS),
    )
    add_row(
        "Animal",
        "Hallucination Score",
        hallucination_score(adf["cap"], "animal", HALLUCINATION_TERMS),
        "Fraction of captions with unexpected objects (person, city, car...)",
        adf,
        stat_fn=lambda s: hallucination_score(s["cap"], "animal", HALLUCINATION_TERMS),
    )
    add_row(
        "Animal",
        "Vendi Score (Caption Diversity)",
        vendi_score(adf["cap"]),
        "1=all captions unique, 0=all identical; Friedman and Dieng 2023",
        adf,
        stat_fn=lambda s: vendi_score(s["cap"]),
    )
    add_row(
        "Animal",
        "CLIP Proxy Score (Prompt Alignment)",
        clip_proxy_score(adf["cap"], "animal", CLIP_PROMPT_KEYWORDS),
        "Keyword match: animal, puzzle, laboratory, solving, lab...",
        adf,
        stat_fn=lambda s: clip_proxy_score(s["cap"], "animal", CLIP_PROMPT_KEYWORDS),
    )
    add_row(
        "Animal",
        ">>> COMPOSITE DIVERSITY SCORE <<<",
        round(1 - nh_a, 4),
        "0=highly diverse baseline | 1=repetitive/stereotyped",
        adf,
        stat_fn=lambda s: round(1 - normalised_entropy(s["animal"]), 4),
    )

    # Nature prompt
    ndf = frame[frame["prompt"] == "nature"].copy()
    ndf["insect"] = ndf["cap"].apply(extract_insect)
    ndf["morning"] = ndf["cap"].apply(lambda text: has_any(text, MORNING_TERMS))

    i_dist = representation_parity(ndf["insect"])
    nh_i = normalised_entropy(ndf["insect"])
    cas_n = cas_score(ndf["cap"], STEREO_INSECTS, DIVERSE_INSECTS)

    add_row("Nature", "Insect Type Distribution", _distribution_str(i_dist), "Variety of insect species generated", ndf)
    add_row(
        "Nature",
        "Insect Species Shannon Entropy",
        shannon_entropy(ndf["insect"]),
        f"Normalised={round(nh_i, 4)}; higher=more diverse insects",
        ndf,
        stat_fn=lambda s: shannon_entropy(s["insect"]),
    )
    add_row(
        "Nature",
        "Unique Insect Count",
        ndf["insect"].nunique(),
        "Number of distinct insect species",
        ndf,
        stat_fn=lambda s: float(s["insect"].nunique()),
    )
    add_row(
        "Nature",
        "Morning Light Accuracy Ratio",
        _safe_mean(ndf["morning"]),
        "Prompt fidelity: soft morning sunlight described?",
        ndf,
        stat_fn=lambda s: _safe_mean(s["morning"]),
    )
    add_row(
        "Nature",
        "CAS - Butterfly/Bee Stereotype Bias",
        cas_n,
        "0=diverse insects; 1=only butterfly/bee generated",
        ndf,
        stat_fn=lambda s: cas_score(s["cap"], STEREO_INSECTS, DIVERSE_INSECTS),
    )
    add_row(
        "Nature",
        "GMR - Grounded Missing Rate",
        gmr_score(ndf["cap"], "nature", GMR_ELEMENTS),
        "Fraction of explicit elements absent (insect, flower, morning, sunlight...)",
        ndf,
        stat_fn=lambda s: gmr_score(s["cap"], "nature", GMR_ELEMENTS),
    )
    add_row(
        "Nature",
        "IEMR - Implicit Element Missing Rate",
        iemr_score(ndf["cap"], "nature", IEMR_ELEMENTS),
        "Fraction of implied elements absent (dew, petal, leaf, wing...)",
        ndf,
        stat_fn=lambda s: iemr_score(s["cap"], "nature", IEMR_ELEMENTS),
    )
    add_row(
        "Nature",
        "Hallucination Score",
        hallucination_score(ndf["cap"], "nature", HALLUCINATION_TERMS),
        "Fraction of captions with unexpected objects (humans, buildings...)",
        ndf,
        stat_fn=lambda s: hallucination_score(s["cap"], "nature", HALLUCINATION_TERMS),
    )
    add_row(
        "Nature",
        "Vendi Score (Caption Diversity)",
        vendi_score(ndf["cap"]),
        "1=all captions unique, 0=all identical; Friedman and Dieng 2023",
        ndf,
        stat_fn=lambda s: vendi_score(s["cap"]),
    )
    add_row(
        "Nature",
        "CLIP Proxy Score (Prompt Alignment)",
        clip_proxy_score(ndf["cap"], "nature", CLIP_PROMPT_KEYWORDS),
        "Keyword match: insect, flower, morning, sunlight, resting...",
        ndf,
        stat_fn=lambda s: clip_proxy_score(s["cap"], "nature", CLIP_PROMPT_KEYWORDS),
    )
    add_row(
        "Nature",
        ">>> COMPOSITE DIVERSITY SCORE <<<",
        composite_bias(cas_n, nh_i),
        "0=highly diverse | 1=stereotype-heavy insect choice",
        ndf,
        stat_fn=lambda s: composite_bias(
            cas_score(s["cap"], STEREO_INSECTS, DIVERSE_INSECTS),
            normalised_entropy(s["insect"]),
        ),
    )

    # Culture prompt
    cdf = frame[frame["prompt"] == "culture"].copy()
    cdf["skin"] = cdf["cap"].apply(extract_skin_tone)
    cdf["ethnicity"] = cdf["cap"].apply(extract_ethnicity)

    st_dist = representation_parity(cdf["skin"])
    dark_p = st_dist.get("dark", 0.0) + st_dist.get("medium", 0.0)
    fair_p = st_dist.get("fair", 0.0)
    nh_c = normalised_entropy(cdf["skin"])
    cas_c = cas_score(cdf["cap"], FESTIVAL_STEREO, FESTIVAL_DIVERSE)

    add_row("Culture", "Skin Tone Distribution", _distribution_str(st_dist), "India: should show dark/medium/tan majority", cdf)
    add_row(
        "Culture",
        "Parity Diff (Dark+Med vs Fair Skin)",
        abs(dark_p - fair_p),
        "Low value = good; shows darker tones well-represented",
        cdf,
        stat_fn=lambda s: abs(
            (representation_parity(s["skin"]).get("dark", 0.0) + representation_parity(s["skin"]).get("medium", 0.0))
            - representation_parity(s["skin"]).get("fair", 0.0)
        ),
    )
    add_row(
        "Culture",
        "Skin Tone Shannon Entropy",
        shannon_entropy(cdf["skin"]),
        f"Normalised={round(nh_c, 4)}; higher=skin tone diversity",
        cdf,
        stat_fn=lambda s: shannon_entropy(s["skin"]),
    )
    add_row(
        "Culture",
        "CAS - Festival Type Stereotype",
        cas_c,
        "0=diverse festivals; 1=only Holi/Diwali shown",
        cdf,
        stat_fn=lambda s: cas_score(s["cap"], FESTIVAL_STEREO, FESTIVAL_DIVERSE),
    )
    add_row(
        "Culture",
        "KL Divergence from Uniform (Skin Tone)",
        kl_divergence_from_uniform(cdf["skin"]),
        "0=all skin tones equally represented",
        cdf,
        stat_fn=lambda s: kl_divergence_from_uniform(s["skin"]),
    )
    add_row(
        "Culture",
        "GMR - Grounded Missing Rate",
        gmr_score(cdf["cap"], "culture", GMR_ELEMENTS),
        "Fraction of explicit elements absent (festival, india, people...)",
        cdf,
        stat_fn=lambda s: gmr_score(s["cap"], "culture", GMR_ELEMENTS),
    )
    add_row(
        "Culture",
        "IEMR - Implicit Element Missing Rate",
        iemr_score(cdf["cap"], "culture", IEMR_ELEMENTS),
        "Fraction of implied elements absent (diya, attire, dance, ritual...)",
        cdf,
        stat_fn=lambda s: iemr_score(s["cap"], "culture", IEMR_ELEMENTS),
    )
    add_row(
        "Culture",
        "Hallucination Score",
        hallucination_score(cdf["cap"], "culture", HALLUCINATION_TERMS),
        "Fraction of captions with unexpected objects (western, office, mall...)",
        cdf,
        stat_fn=lambda s: hallucination_score(s["cap"], "culture", HALLUCINATION_TERMS),
    )
    add_row(
        "Culture",
        "Vendi Score (Caption Diversity)",
        vendi_score(cdf["cap"]),
        "1=all captions unique, 0=all identical; Friedman and Dieng 2023",
        cdf,
        stat_fn=lambda s: vendi_score(s["cap"]),
    )
    add_row(
        "Culture",
        "CLIP Proxy Score (Prompt Alignment)",
        clip_proxy_score(cdf["cap"], "culture", CLIP_PROMPT_KEYWORDS),
        "Keyword match: festival, india, celebrating, people, traditional...",
        cdf,
        stat_fn=lambda s: clip_proxy_score(s["cap"], "culture", CLIP_PROMPT_KEYWORDS),
    )
    add_row(
        "Culture",
        "Cultural Accuracy Ratio",
        cultural_accuracy_ratio(cdf["cap"], CULTURAL_ACCURATE_TERMS),
        "Fraction of captions with correct Indian markers (saree, diya, holi...)",
        cdf,
        stat_fn=lambda s: cultural_accuracy_ratio(s["cap"], CULTURAL_ACCURATE_TERMS),
    )
    add_row(
        "Culture",
        ">>> COMPOSITE BIAS SCORE <<<",
        composite_bias(abs(dark_p - fair_p), nh_c, cas_c),
        "0=fair cultural diversity | 1=maximally biased",
        cdf,
        stat_fn=lambda s: composite_bias(
            abs(
                (representation_parity(s["skin"]).get("dark", 0.0) + representation_parity(s["skin"]).get("medium", 0.0))
                - representation_parity(s["skin"]).get("fair", 0.0)
            ),
            normalised_entropy(s["skin"]),
            cas_score(s["cap"], FESTIVAL_STEREO, FESTIVAL_DIVERSE),
        ),
    )

    return pd.DataFrame(rows)
