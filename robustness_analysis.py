from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import hypergeom
from sklearn.metrics import cohen_kappa_score

from hea_forecast.elemental_features import add_elemental_features
from hea_forecast.temporal import expand_element_pairs

METRIC_PATTERNS = {
    "overpotential_mv": re.compile(
        r"(?:overpotential(?:s)?(?:\s+of)?|η)\s*(?:is|was|of|=|:)??\s*"
        r"(\d{2,4}(?:\.\d+)?)(?:\s*(?:and|,)\s*\d{2,4}(?:\.\d+)?)?\s*mV",
        re.IGNORECASE,
    ),
    "tafel_mv_dec": re.compile(
        r"(?:Tafel\s+slope(?:s)?(?:\s+of)?|Tafel)\s*(?:is|was|of|=|:)??\s*(\d{1,4}(?:\.\d+)?)\s*mV\s*(?:dec(?:ade)?(?:\s*[-−–^]?\s*1)?|/dec)",
        re.IGNORECASE,
    ),
    "current_density_ma_cm2": re.compile(
        r"[−–-]?\s*(\d+(?:\.\d+)?)(?:\s*(?:and|,)\s*[−–-]?\s*\d+(?:\.\d+)?)?"
        r"\s*mA\s*(?:cm|cm²|cm2)\s*[-−–^]?\s*2",
        re.IGNORECASE,
    ),
    "half_wave_potential_v": re.compile(
        r"(?:half[- ]wave\s+potential|E\s*1/2|E½)\s*(?:is|was|of|=|:)??\s*(-?\d+(?:\.\d+)?)\s*V",
        re.IGNORECASE,
    ),
    "mass_activity": re.compile(
        r"(?:mass\s+activity(?:\s+of)?|activity)\s*(?:is|was|of|=|:)??\s*(\d+(?:\.\d+)?)\s*(A|mA)\s*(?:mg|g)\s*[-−–^]?\s*1",
        re.IGNORECASE,
    ),
    "stability_hours": re.compile(
        r"(?:stable|stability|durability|operation|operating|maintain(?:ed|s)?)\D{0,45}(\d+(?:\.\d+)?)\s*(?:h|hours?)\b",
        re.IGNORECASE,
    ),
}


def _truth(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Unlike a percentile bootstrap on a very small Top-K set, this interval does
    not collapse to an implausibly narrow range when only a few hits are
    observed.
    """
    if trials <= 0:
        return (math.nan, math.nan)
    proportion = successes / trials
    denominator = 1 + (z**2 / trials)
    centre = (proportion + z**2 / (2 * trials)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / trials + z**2 / (4 * trials**2)
        )
        / denominator
    )
    return (max(0.0, centre - radius), min(1.0, centre + radius))


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    """Return Benjamini-Hochberg adjusted p values in original order."""
    if not p_values:
        return []
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 1.0
    for reverse_rank, index in enumerate(order[::-1], start=1):
        rank = len(values) - reverse_rank + 1
        running = min(running, values[index] * len(values) / rank)
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def build_endpoint_robustness(
    ranked_system_csv: str | Path,
    output_dir: str | Path,
    k_values: tuple[int, ...] = (10, 20, 50, 100),
) -> dict[str, Any]:
    """Create endpoint and HEA-definition sensitivity tables.

    The primary endpoint is fixed as peer-reviewed exact element-set
    Precision@10 using the full candidate universe (>=4 elements).  Other K
    values, the preprint-inclusive scope, and the conventional >=5-element HEA
    definition are explicitly secondary or sensitivity analyses.
    """
    ranked = pd.read_csv(ranked_system_csv, low_memory=False)
    ranked = ranked.sort_values("ensemble_score", ascending=False, kind="stable")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    scopes = [
        ("peer_reviewed_articles_only", "human_hit_peer_reviewed_articles_only"),
        ("all_primary_and_preprint", "human_hit_all_primary_and_preprint"),
    ]
    for minimum_elements in (4, 5):
        subset = ranked.loc[ranked["element_count"] >= minimum_elements].reset_index(drop=True)
        for scope, label_column in scopes:
            labels = subset[label_column].astype(int)
            positives = int(labels.sum())
            prevalence = positives / len(subset) if len(subset) else math.nan
            for k in k_values:
                actual_k = min(k, len(subset))
                hits = int(labels.iloc[:actual_k].sum())
                precision = hits / actual_k if actual_k else math.nan
                ci_low, ci_high = wilson_interval(hits, actual_k)
                p_value = (
                    float(hypergeom.sf(hits - 1, len(subset), positives, actual_k))
                    if positives and hits
                    else 1.0
                )
                rows.append(
                    {
                        "minimum_elements": minimum_elements,
                        "scope": scope,
                        "k": k,
                        "candidate_count": len(subset),
                        "positives": positives,
                        "prevalence": prevalence,
                        "hits_at_k": hits,
                        "precision_at_k": precision,
                        "precision_wilson_95_low": ci_low,
                        "precision_wilson_95_high": ci_high,
                        "enrichment_at_k": precision / prevalence if prevalence else math.nan,
                        "hypergeometric_p": p_value,
                        "endpoint_role": (
                            "primary"
                            if minimum_elements == 4
                            and scope == "peer_reviewed_articles_only"
                            and k == 10
                            else "secondary_or_sensitivity"
                        ),
                    }
                )
    table = pd.DataFrame(rows)
    secondary_mask = table["endpoint_role"].eq("secondary_or_sensitivity")
    table["bh_q_across_secondary_tests"] = math.nan
    table.loc[secondary_mask, "bh_q_across_secondary_tests"] = benjamini_hochberg(
        table.loc[secondary_mask, "hypergeometric_p"].tolist()
    )
    table.to_csv(destination / "endpoint_and_hea_definition_sensitivity.csv", index=False)

    primary = table.loc[table["endpoint_role"].eq("primary")].iloc[0].to_dict()
    five_element = table.loc[
        (table["minimum_elements"].eq(5))
        & (table["scope"].eq("peer_reviewed_articles_only"))
        & (table["k"].eq(10))
    ].iloc[0].to_dict()
    summary = {
        "entity_definition": "canonical element set; stoichiometric coefficients are not modeled",
        "primary_endpoint": primary,
        "five_or_more_element_sensitivity": five_element,
        "multiplicity_policy": (
            "The primary endpoint is interpreted without multiplicity adjustment. "
            "All other K, evidence-scope, and HEA-definition tests are exploratory; "
            "Benjamini-Hochberg q values are supplied for transparency."
        ),
        "unresolved_historical_record": {
            "annotation_id": "A00096",
            "system_id": "Co-Cr-Fe-Mn-Ni",
            "reaction_id": "OER",
            "year": 2020,
            "status": "excluded from model-ready data because independent adjudication was incomplete",
            "prospective_2026_effect": "none: the frozen 2026 candidate table and labels are unchanged",
        },
    }
    (destination / "endpoint_robustness_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def prepare_2026_review_package(
    validation_csv: str | Path,
    future_dir: str | Path,
    output_dir: str | Path,
    sample_fraction: float = 0.30,
    random_state: int = 20260916,
) -> dict[str, Any]:
    """Create a transparent machine triage and a blinded second-reviewer sample."""
    validation = pd.read_csv(validation_csv, low_memory=False).fillna("")
    future_dir = Path(future_dir)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    system_rank = pd.read_csv(future_dir / "future_candidates_system.csv")
    system_rank = system_rank.sort_values("ensemble_score", ascending=False).reset_index(drop=True)
    system_rank["forecast_system_rank"] = np.arange(1, len(system_rank) + 1)
    system_rank = system_rank[
        ["system_id", "reaction_id", "forecast_system_rank", "ensemble_score"]
    ]

    pair_rank = pd.read_csv(future_dir / "future_candidates_element_pair.csv")
    pair_rank = pair_rank.sort_values("ensemble_score", ascending=False).reset_index(drop=True)
    pair_rank["forecast_pair_rank"] = np.arange(1, len(pair_rank) + 1)
    pair_rank = pair_rank[["system_id", "reaction_id", "forecast_pair_rank"]]
    expanded = expand_element_pairs(validation)
    pair_best = expanded.merge(pair_rank, on=["system_id", "reaction_id"], how="left")
    pair_best = pair_best.groupby("annotation_id", as_index=False)["forecast_pair_rank"].min()

    review = validation.merge(system_rank, on=["system_id", "reaction_id"], how="left")
    review = review.merge(pair_best, on="annotation_id", how="left")
    title_support = review["formula_in_title"].map(_truth) & review["reaction_in_title"].map(_truth)
    high = review["extraction_confidence"].astype(str).str.lower().eq("high")
    review["auto_evidence_tier"] = np.select(
        [high & title_support, title_support | high], ["strict", "moderate"], default="screening"
    )
    review["auto_material_suggestion"] = np.where(
        review["material_class"].eq("metallic_hea") & review["system_id"].ne(""),
        "likely_yes",
        "manual_check",
    )
    review["auto_reaction_suggestion"] = np.where(
        review["reaction_id"].ne("") & review["reaction_in_title"].map(_truth),
        "likely_yes",
        "manual_check",
    )
    review["auto_target_suggestion"] = np.where(title_support, "likely_yes", "manual_check")
    review["is_frozen_system_candidate"] = review["forecast_system_rank"].notna()
    review["is_frozen_pair_candidate"] = review["forecast_pair_rank"].notna()
    review["review_priority"] = np.select(
        [
            review["is_frozen_system_candidate"],
            review["is_frozen_pair_candidate"],
            review["auto_evidence_tier"].eq("strict"),
        ],
        ["P1_system_hit", "P2_pair_hit", "P3_strict_evidence"],
        default="P4_screening",
    )
    review["reviewer2_material_valid"] = ""
    review["reviewer2_reaction_valid"] = ""
    review["reviewer2_target_catalyst"] = ""
    review["reviewer2_corrected_system_id"] = ""
    review["reviewer2_corrected_reaction_id"] = ""
    review["reviewer2_evidence_sentence"] = ""
    review["reviewer2_notes"] = ""
    review["reviewer2_status"] = "pending"
    review = review.sort_values(
        ["review_priority", "forecast_system_rank", "forecast_pair_rank", "annotation_id"],
        na_position="last",
    )
    review.to_csv(destination / "openalex_2026_machine_triage.csv", index=False)

    # Include every predicted hit, then sample enough non-hits to reach at least 30%.
    mandatory = review[review["is_frozen_system_candidate"] | review["is_frozen_pair_candidate"]]
    target_n = max(len(mandatory), math.ceil(len(review) * sample_fraction))
    remaining = review.loc[~review["annotation_id"].isin(mandatory["annotation_id"])]
    extra_n = min(max(target_n - len(mandatory), 0), len(remaining))
    extra = remaining.sample(n=extra_n, random_state=random_state) if extra_n else remaining.head(0)
    sample = pd.concat([mandatory, extra], ignore_index=True).drop_duplicates("annotation_id")
    sample = sample.sample(frac=1, random_state=random_state).reset_index(drop=True)
    sample["blinded_row_id"] = [f"R2-{index:03d}" for index in range(1, len(sample) + 1)]
    blind_columns = [
        "blinded_row_id",
        "annotation_id",
        "paper_id",
        "doi",
        "publication_date",
        "display_name",
        "composition_raw",
        "system_id",
        "reaction_id",
        "material_class",
        "evidence_context",
        "reviewer2_material_valid",
        "reviewer2_reaction_valid",
        "reviewer2_target_catalyst",
        "reviewer2_corrected_system_id",
        "reviewer2_corrected_reaction_id",
        "reviewer2_evidence_sentence",
        "reviewer2_notes",
        "reviewer2_status",
    ]
    sample[blind_columns].to_csv(destination / "second_reviewer_blinded_sample.csv", index=False)

    strict = review[review["auto_evidence_tier"].eq("strict")].copy()
    strict.to_csv(destination / "strict_machine_evidence_subset.csv", index=False)
    summary = {
        "validation_rows": len(review),
        "strict_machine_evidence_rows": len(strict),
        "system_candidate_matches": int(review["is_frozen_system_candidate"].sum()),
        "pair_candidate_matches": int(review["is_frozen_pair_candidate"].sum()),
        "second_reviewer_sample_rows": len(sample),
        "human_review_complete": False,
        "warning": "Machine triage is not an independent human review and must not be reported as one.",
    }
    (destination / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def reviewer_agreement(review_csv: str | Path, output_json: str | Path) -> dict[str, Any]:
    """Calculate raw agreement and Cohen's kappa after reviewer 2 fills the sample."""
    frame = pd.read_csv(review_csv, low_memory=False).fillna("")
    metrics: dict[str, Any] = {"rows": len(frame), "fields": {}}
    for first, second in [
        ("reviewer_material_valid", "reviewer2_material_valid"),
        ("reviewer_reaction_valid", "reviewer2_reaction_valid"),
        ("reviewer_target_catalyst", "reviewer2_target_catalyst"),
    ]:
        if first not in frame or second not in frame:
            continue
        valid = frame[first].ne("") & frame[second].ne("")
        subset = (
            frame.loc[valid, [first, second]].astype(str).apply(lambda c: c.str.lower().str.strip())
        )
        if subset.empty:
            metrics["fields"][second] = {"n": 0, "agreement": None, "cohen_kappa": None}
            continue
        agreement = float((subset[first] == subset[second]).mean())
        labels = sorted(set(subset[first]) | set(subset[second]))
        p_e = sum(
            (subset[first].eq(label).mean() * subset[second].eq(label).mean()) for label in labels
        )
        kappa = (agreement - p_e) / (1 - p_e) if p_e < 1 else 1.0
        metrics["fields"][second] = {
            "n": len(subset),
            "agreement": agreement,
            "cohen_kappa": float(kappa),
        }
    destination = Path(output_json)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def _jaccard(left: str, right: str) -> float:
    a, b = set(str(left).split("-")), set(str(right).split("-"))
    return len(a & b) / len(a | b) if a | b else 0.0


def _extract_metrics(text: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, pattern in METRIC_PATTERNS.items():
        match = pattern.search(text or "")
        result[name] = float(match.group(1)) if match else np.nan
    result["performance_evidence_found"] = any(pd.notna(value) for value in result.values())
    return result


def build_multicriteria_ranking(
    future_csv: str | Path,
    discovery_events_csv: str | Path,
    validation_csv: str | Path,
    validation_works_csv: str | Path,
    output_csv: str | Path,
    top_n: int = 50,
) -> dict[str, Any]:
    """Rank candidates by forecast, chemical plausibility, novelty, and cost-risk proxy."""
    candidates = (
        pd.read_csv(future_csv, low_memory=False)
        .sort_values("ensemble_score", ascending=False)
        .head(top_n)
        .copy()
    )
    events = pd.read_csv(discovery_events_csv, low_memory=False)
    validation = pd.read_csv(validation_csv, low_memory=False).fillna("")
    works = pd.read_csv(validation_works_csv, low_memory=False).fillna("")
    candidates = add_elemental_features(candidates)

    reaction_systems = {
        reaction: group["system_id"].unique().tolist()
        for reaction, group in events.groupby("reaction_id")
    }
    candidates["composition_novelty"] = [
        1
        - max(
            (_jaccard(system, known) for known in reaction_systems.get(reaction, [])), default=0.0
        )
        for system, reaction in zip(
            candidates["system_id"], candidates["reaction_id"], strict=False
        )
    ]
    # Transparent feasibility proxy: lower atomic-size/electronegativity mismatch is favored.
    size = candidates["chem_atomic_size_mismatch"].fillna(
        candidates["chem_atomic_size_mismatch"].median()
    )
    en_std = candidates["chem_electronegativity_std"].fillna(
        candidates["chem_electronegativity_std"].median()
    )
    size_score = (1 - (size / 10.0)).clip(0, 1)
    en_score = (1 - (en_std / 0.5)).clip(0, 1)
    candidates["synthesis_feasibility_proxy"] = 0.65 * size_score + 0.35 * en_score

    # Cost-risk is a reproducible proxy, not a market-price estimate.
    candidates["cost_risk_proxy"] = (
        0.7 * candidates["chem_platinum_group_fraction"] + 0.3 * candidates["chem_noble_fraction"]
    ).clip(0, 1)
    candidates["affordability_proxy"] = 1 - candidates["cost_risk_proxy"]
    candidates["actionability_score"] = (
        0.45 * candidates["ensemble_score"]
        + 0.20 * candidates["composition_novelty"]
        + 0.20 * candidates["synthesis_feasibility_proxy"]
        + 0.15 * candidates["affordability_proxy"]
    )

    full_text = (
        works.set_index("id")
        .apply(lambda row: f"{row.get('display_name', '')} {row.get('abstract', '')}", axis=1)
        .to_dict()
    )
    validation["performance_text"] = [
        full_text.get(paper, evidence)
        for paper, evidence in zip(
            validation["paper_id"], validation["evidence_context"], strict=False
        )
    ]
    perf_rows = []
    for _, row in validation.iterrows():
        perf_rows.append(
            {
                "system_id": row["system_id"],
                "reaction_id": row["reaction_id"],
                "validation_doi": row["doi"],
                "validation_title": row["display_name"],
                **_extract_metrics(row["performance_text"]),
            }
        )
    perf = pd.DataFrame(perf_rows)
    perf = perf.sort_values("performance_evidence_found", ascending=False).drop_duplicates(
        ["system_id", "reaction_id"]
    )
    candidates = candidates.merge(perf, on=["system_id", "reaction_id"], how="left")
    candidates["prospective_2026_match"] = candidates["validation_title"].notna()
    candidates = candidates.sort_values("actionability_score", ascending=False).reset_index(
        drop=True
    )
    candidates["actionability_rank"] = np.arange(1, len(candidates) + 1)
    destination = Path(output_csv)
    destination.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(destination, index=False)
    summary = {
        "candidate_rows": len(candidates),
        "prospective_matches": int(candidates["prospective_2026_match"].sum()),
        "performance_evidence_rows": int(candidates["performance_evidence_found"].eq(True).sum()),
        "score_weights": {
            "forecast": 0.45,
            "novelty": 0.20,
            "synthesis_feasibility_proxy": 0.20,
            "affordability_proxy": 0.15,
        },
        "warning": "Feasibility and cost are screening proxies; they are not phase-stability calculations or market quotations.",
    }
    destination.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def evaluate_completed_human_review(
    review_csv: str | Path,
    triage_csv: str | Path,
    future_dir: str | Path,
    works_csv: str | Path,
    output_dir: str | Path,
    permutations: int = 10_000,
    random_state: int = 20260916,
) -> dict[str, Any]:
    """Evaluate frozen forecasts using completed human review.

    The review sample was designed to include every machine-matched frozen candidate,
    so accepted matches can be projected back to the full frozen ranking.  This does
    not turn unreviewed non-matches into reviewed literature records.
    """
    review = pd.read_csv(review_csv, low_memory=False).fillna("")
    triage = pd.read_csv(triage_csv, low_memory=False).fillna("")
    works = pd.read_csv(works_csv, low_memory=False).fillna("")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    accepted = review.loc[
        review["reviewer2_material_valid"].eq("yes")
        & review["reviewer2_reaction_valid"].eq("yes")
        & review["reviewer2_target_catalyst"].eq("yes")
    ].copy()
    accepted = accepted.merge(
        works[["id", "work_type", "source_name", "abstract"]],
        left_on="paper_id",
        right_on="id",
        how="left",
    )
    accepted["peer_reviewed_article"] = accepted["work_type"].eq("article")
    accepted.to_csv(destination / "accepted_reviewed_records.csv", index=False)

    # Machine-human agreement is reported separately from human-human reliability.
    comparison = review.merge(
        triage[
            [
                "annotation_id",
                "auto_evidence_tier",
                "auto_material_suggestion",
                "auto_reaction_suggestion",
                "auto_target_suggestion",
            ]
        ],
        on="annotation_id",
        how="left",
    )
    comparison["human_accept"] = (
        comparison["reviewer2_material_valid"].eq("yes")
        & comparison["reviewer2_reaction_valid"].eq("yes")
        & comparison["reviewer2_target_catalyst"].eq("yes")
    )
    comparison["machine_strict_accept"] = comparison["auto_evidence_tier"].eq("strict")
    agreement = {
        "sample_rows": len(comparison),
        "human_accepted": int(comparison["human_accept"].sum()),
        "machine_strict_accepted": int(comparison["machine_strict_accept"].sum()),
        "binary_raw_agreement": float(
            comparison["human_accept"].eq(comparison["machine_strict_accept"]).mean()
        ),
        "binary_cohen_kappa": float(
            cohen_kappa_score(comparison["human_accept"], comparison["machine_strict_accept"])
        ),
        "interpretation": "machine-human agreement, not human-human inter-rater reliability",
    }
    (destination / "machine_human_agreement.json").write_text(
        json.dumps(agreement, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    rng = np.random.default_rng(random_state)
    all_metrics: list[dict[str, Any]] = []
    for entity_level in ["system", "element_pair"]:
        candidates = pd.read_csv(
            Path(future_dir) / f"future_candidates_{entity_level}.csv"
        ).sort_values("ensemble_score", ascending=False, kind="stable")
        candidates = candidates.reset_index(drop=True)
        candidates["rank"] = np.arange(1, len(candidates) + 1)
        accepted_level = (
            expand_element_pairs(accepted) if entity_level == "element_pair" else accepted
        )
        for scope, source in [
            ("all_primary_and_preprint", accepted_level),
            (
                "peer_reviewed_articles_only",
                accepted_level.loc[accepted_level["peer_reviewed_article"]],
            ),
        ]:
            valid_pairs = set(zip(source["system_id"], source["reaction_id"], strict=False))
            label_column = f"human_hit_{scope}"
            candidates[label_column] = [
                int((system, reaction) in valid_pairs)
                for system, reaction in zip(
                    candidates["system_id"], candidates["reaction_id"], strict=False
                )
            ]
            n_candidates = len(candidates)
            positives = int(candidates[label_column].sum())
            prevalence = positives / n_candidates if n_candidates else 0.0
            for k in [10, 20, 50, 100]:
                actual_k = min(k, n_candidates)
                top_labels = candidates.head(actual_k)[label_column].to_numpy(dtype=int)
                hits = int(top_labels.sum())
                precision = hits / actual_k if actual_k else 0.0
                enrichment = precision / prevalence if prevalence else np.nan
                p_hypergeom = (
                    float(hypergeom.sf(hits - 1, n_candidates, positives, actual_k))
                    if positives and hits
                    else 1.0
                )
                permutation_hits = np.empty(permutations, dtype=int)
                full_labels = candidates[label_column].to_numpy(dtype=int)
                for index in range(permutations):
                    permutation_hits[index] = rng.permutation(full_labels)[:actual_k].sum()
                p_permutation = float(
                    (1 + np.count_nonzero(permutation_hits >= hits)) / (permutations + 1)
                )
                precision_boot = np.empty(permutations, dtype=float)
                prevalence_boot = np.empty(permutations, dtype=float)
                for index in range(permutations):
                    precision_boot[index] = rng.choice(top_labels, actual_k, replace=True).mean()
                    prevalence_boot[index] = rng.choice(
                        full_labels, n_candidates, replace=True
                    ).mean()
                valid = prevalence_boot > 0
                enrichment_boot = precision_boot[valid] / prevalence_boot[valid]
                all_metrics.append(
                    {
                        "entity_level": entity_level,
                        "scope": scope,
                        "k": k,
                        "candidate_count": n_candidates,
                        "positives": positives,
                        "prevalence": prevalence,
                        "hits_at_k": hits,
                        "precision_at_k": precision,
                        "enrichment_at_k": enrichment,
                        "precision_ci_low": float(np.quantile(precision_boot, 0.025)),
                        "precision_ci_high": float(np.quantile(precision_boot, 0.975)),
                        "enrichment_ci_low": float(np.quantile(enrichment_boot, 0.025)),
                        "enrichment_ci_high": float(np.quantile(enrichment_boot, 0.975)),
                        "hypergeometric_p": p_hypergeom,
                        "permutation_p": p_permutation,
                    }
                )
        candidates.to_csv(destination / f"ranked_with_human_hits_{entity_level}.csv", index=False)
        matched = candidates.loc[candidates["human_hit_all_primary_and_preprint"].eq(1)].merge(
            accepted_level,
            on=["system_id", "reaction_id"],
            how="left",
        )
        matched.to_csv(destination / f"accepted_matches_{entity_level}.csv", index=False)
    metrics = pd.DataFrame(all_metrics)
    metrics.to_csv(destination / "topk_significance.csv", index=False)
    summary = {
        "review_rows": len(review),
        "complete_rows": int(review["reviewer2_status"].eq("complete").sum()),
        "human_accepted_records": len(accepted),
        "human_rejected_or_uncertain_records": len(review) - len(accepted),
        "machine_human_agreement": agreement,
        "warning": "Cohen's kappa here compares machine triage with one human reviewer; a second independent human label set is required for inter-rater reliability.",
    }
    (destination / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def build_human_evidence_cards(
    accepted_matches_csv: str | Path,
    output_csv: str | Path,
    output_markdown: str | Path,
) -> dict[str, Any]:
    matches = pd.read_csv(accepted_matches_csv, low_memory=False).fillna("")
    rows: list[dict[str, Any]] = []
    for _, row in matches.sort_values("rank").iterrows():
        text = f"{row.get('display_name', '')} {row.get('abstract', '')}"
        metrics = _extract_metrics(text)
        rows.append(
            {
                "rank": int(row["rank"]),
                "system_id": row["system_id"],
                "reaction_id": row["reaction_id"],
                "doi": row.get("doi", ""),
                "publication_date": row.get("publication_date", ""),
                "work_type": row.get("work_type", ""),
                "source_name": row.get("source_name", ""),
                "display_name": row.get("display_name", ""),
                "reviewer_evidence": row.get("reviewer2_evidence_sentence", ""),
                "reviewer_notes": row.get("reviewer2_notes", ""),
                **metrics,
            }
        )
    cards = pd.DataFrame(rows).drop_duplicates(["system_id", "reaction_id"])
    csv_path = Path(output_csv)
    md_path = Path(output_markdown)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    cards.to_csv(csv_path, index=False)
    sections = ["# 人工确认的2026年前瞻命中证据卡", ""]
    for _, row in cards.iterrows():
        sections.extend(
            [
                f"## Rank {int(row['rank'])}: {row['system_id']} / {row['reaction_id']}",
                "",
                f"- DOI：{row['doi']}",
                f"- 发表日期：{row['publication_date']}",
                f"- 类型：{row['work_type']}",
                f"- 期刊/来源：{row['source_name']}",
                f"- 论文：{row['display_name']}",
                f"- 审核证据：{row['reviewer_evidence']}",
                f"- 审核说明：{row['reviewer_notes']}",
                f"- 过电位（mV，摘要自动提取）：{row['overpotential_mv']}",
                f"- Tafel斜率（mV dec⁻¹）：{row['tafel_mv_dec']}",
                f"- 稳定性（h）：{row['stability_hours']}",
                "",
            ]
        )
    md_path.write_text("\n".join(sections), encoding="utf-8")
    return {"cards": len(cards), "csv": str(csv_path), "markdown": str(md_path)}
