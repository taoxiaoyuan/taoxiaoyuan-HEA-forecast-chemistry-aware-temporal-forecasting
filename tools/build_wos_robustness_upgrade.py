"""Quantify how potential WoS historical omissions affect the locked endpoint.

This is a pre-review stress test. Every machine-detected historical pair is
treated as a qualifying prior edge in the conservative scenario. The output is
therefore a lower-bound sensitivity analysis, not a replacement human label.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import hypergeom


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "wos_robustness_upgrade_2026"
RANKING = ROOT / "results" / "human_validation_2026" / "ranked_with_human_hits_system.csv"
KNOWN = ROOT / "results" / "human_validation_2026" / "accepted_matches_system.csv"
PRIOR = ROOT / "results" / "wos_independent_validation_2026" / "wos_potential_pre2026_edges_blinded_review.csv"
AUTO_SUMMARY = ROOT / "results" / "wos_independent_validation_2026" / "automatic_validation_summary.json"


def wilson(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if trials <= 0:
        return math.nan, math.nan
    p = successes / trials
    denominator = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denominator
    radius = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denominator
    return max(0.0, centre - radius), min(1.0, centre + radius)


def metrics(labels: pd.Series, scenario: str, ks: tuple[int, ...] = (10, 20, 50, 100, 280)) -> list[dict]:
    y = labels.astype(int).reset_index(drop=True)
    n = len(y)
    positives = int(y.sum())
    prevalence = positives / n if n else math.nan
    rows: list[dict] = []
    rng = np.random.default_rng(20261001)
    for requested_k in ks:
        k = min(requested_k, n)
        hits = int(y.iloc[:k].sum())
        precision = hits / k if k else math.nan
        low, high = wilson(hits, k)
        p_exact = float(hypergeom.sf(hits - 1, n, positives, k)) if hits and positives else 1.0
        if positives and k:
            draws = rng.hypergeometric(positives, n - positives, k, size=10000)
            p_perm = float((1 + np.count_nonzero(draws >= hits)) / 10001)
        else:
            p_perm = 1.0
        rows.append(
            {
                "scenario": scenario,
                "k": requested_k,
                "actual_k": k,
                "candidate_count": n,
                "positives": positives,
                "prevalence": prevalence,
                "hits_at_k": hits,
                "precision_at_k": precision,
                "precision_wilson_95_low": low,
                "precision_wilson_95_high": high,
                "enrichment_at_k": precision / prevalence if prevalence else math.nan,
                "hypergeometric_p": p_exact,
                "permutation_p_10000": p_perm,
            }
        )
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ranking = pd.read_csv(RANKING, low_memory=False).sort_values("rank").reset_index(drop=True)
    known = pd.read_csv(KNOWN, dtype=str, keep_default_na=False)
    prior = pd.read_csv(PRIOR, dtype=str, keep_default_na=False)
    automatic = json.loads(AUTO_SUMMARY.read_text(encoding="utf-8"))

    challenged_pairs = prior[["system_id", "reaction_id"]].drop_duplicates()
    challenged_pairs["machine_detected_potential_prior_edge"] = True
    annotated = ranking.merge(challenged_pairs, on=["system_id", "reaction_id"], how="left")
    annotated["machine_detected_potential_prior_edge"] = annotated[
        "machine_detected_potential_prior_edge"
    ].map(lambda value: bool(value) if pd.notna(value) else False)
    annotated["conservative_peer_reviewed_label"] = annotated[
        "human_hit_peer_reviewed_articles_only"
    ].astype(int)
    annotated.loc[
        annotated["machine_detected_potential_prior_edge"],
        "conservative_peer_reviewed_label",
    ] = 0
    annotated["conservative_all_primary_label"] = annotated[
        "human_hit_all_primary_and_preprint"
    ].astype(int)
    annotated.loc[
        annotated["machine_detected_potential_prior_edge"],
        "conservative_all_primary_label",
    ] = 0
    annotated.to_csv(OUT / "ranking_with_pre_review_wos_flags.csv", index=False)

    rows = []
    rows.extend(metrics(annotated["human_hit_peer_reviewed_articles_only"], "locked_original_peer_reviewed"))
    rows.extend(metrics(annotated["conservative_peer_reviewed_label"], "conservative_all_machine_prior_pairs_valid"))
    rows.extend(metrics(annotated["human_hit_all_primary_and_preprint"], "locked_original_including_preprint"))
    rows.extend(metrics(annotated["conservative_all_primary_label"], "conservative_including_preprint"))
    metric_table = pd.DataFrame(rows)
    metric_table.to_csv(OUT / "pre_review_endpoint_sensitivity.csv", index=False)

    prior_grouped = (
        prior.groupby(["system_id", "reaction_id"], as_index=False)
        .agg(
            potential_prior_records=("prior_blind_id", "nunique"),
            earliest_potential_prior_date=("publication_date", "min"),
            potential_prior_dois=("doi", lambda x: "|".join(sorted(set(v for v in x if v)))),
            potential_prior_titles=("display_name", lambda x: " || ".join(sorted(set(v for v in x if v)))),
        )
    )
    hit_impact = known.merge(prior_grouped, on=["system_id", "reaction_id"], how="left")
    hit_impact["historical_challenge_status"] = np.where(
        hit_impact["potential_prior_records"].notna(),
        "potential_prior_edge_requires_human_confirmation",
        "no_machine_detected_prior_edge_in_wos_export",
    )
    hit_impact.to_csv(OUT / "impact_on_original_locked_hits.csv", index=False)

    original_top10 = metric_table[
        metric_table["scenario"].eq("locked_original_peer_reviewed") & metric_table["k"].eq(10)
    ].iloc[0]
    conservative_top10 = metric_table[
        metric_table["scenario"].eq("conservative_all_machine_prior_pairs_valid") & metric_table["k"].eq(10)
    ].iloc[0]
    original_total = int(annotated["human_hit_peer_reviewed_articles_only"].sum())
    conservative_total = int(annotated["conservative_peer_reviewed_label"].sum())
    challenged_original_hits = hit_impact[
        hit_impact["peer_reviewed_article"].str.lower().eq("true")
        & hit_impact["potential_prior_records"].notna()
    ]
    summary = {
        "status": "pre_review_conservative_stress_test_complete",
        "interpretation": "Lower bound only; machine-detected prior edges have not yet been human adjudicated.",
        "automatic_cross_database_checks": {
            "historical_wos_to_openalex_overlap": automatic["historical_wos_to_openalex_overlap"],
            "historical_event_identity_recall": automatic["historical_event_identity_recall"],
            "historical_event_exact_year_agreement": automatic["historical_event_exact_year_agreement"],
            "outcome_window_wos_to_openalex_overlap": automatic["outcome_window_wos_to_openalex_overlap"],
        },
        "machine_challenged_candidate_pairs": int(len(challenged_pairs)),
        "machine_challenged_original_peer_reviewed_hits": int(len(challenged_original_hits)),
        "original_peer_reviewed_positives": original_total,
        "conservative_peer_reviewed_positives": conservative_total,
        "original_top10_hits": int(original_top10["hits_at_k"]),
        "original_top10_precision": float(original_top10["precision_at_k"]),
        "conservative_top10_hits": int(conservative_top10["hits_at_k"]),
        "conservative_top10_precision": float(conservative_top10["precision_at_k"]),
        "next_required_step": "Complete both blinded WoS review files, then run finalize_wos_independent_validation.py.",
    }
    (OUT / "pre_review_stress_test_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report = f"""# WoS cross-database robustness upgrade

## What is confirmed automatically

- Historical WoS-to-OpenAlex work overlap: {automatic['historical_wos_to_openalex_overlap']:.1%}.
- Frozen first-report events found by identity in WoS: {automatic['historical_event_identity_recall']:.1%}.
- Exact first-report-year agreement among matched events: {automatic['historical_event_exact_year_agreement']:.1%}; agreement within one year: {automatic['historical_event_within_one_year_agreement']:.1%}.
- WoS-to-OpenAlex work overlap in the locked 2026 date window: {automatic['outcome_window_wos_to_openalex_overlap']:.1%}.

## Pre-review lower-bound stress test

The original locked endpoint contains {original_total} peer-reviewed positives, including {int(original_top10['hits_at_k'])} in the Top 10. Machine extraction from the independent WoS historical export found possible pre-2026 edges for {len(challenged_pairs)} frozen candidate pairs. If every one of those possible prior edges is conservatively treated as valid, {len(challenged_original_hits)} original peer-reviewed hits are removed, leaving {conservative_total}; Top-10 hits fall from {int(original_top10['hits_at_k'])} to {int(conservative_top10['hits_at_k'])}.

This is intentionally a worst-case sensitivity bound. It must not be described as the final WoS result until the blinded historical records have been adjudicated. Its value is to show reviewers that the project tests its central conclusion against historical-index incompleteness instead of assuming that absence from one database means non-existence.
"""
    (OUT / "robustness_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
