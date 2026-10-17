from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
from scipy.stats import hypergeom


ROOT = Path(__file__).resolve().parents[1]
WOS = ROOT / "results" / "wos_independent_validation_2026" / "final"
OUT = ROOT / "results" / "third_version_2026"


def wilson(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if trials <= 0:
        return math.nan, math.nan
    p = successes / trials
    denominator = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denominator
    radius = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denominator
    return max(0.0, centre - radius), min(1.0, centre + radius)


def metrics(endpoint: str, universe: pd.DataFrame, label: str, rank_column: str) -> list[dict]:
    ordered = universe.sort_values(rank_column).reset_index(drop=True)
    n = len(ordered)
    positives = int(ordered[label].sum())
    prevalence = positives / n
    rows: list[dict] = []
    for requested_k in (10, 20, 50, 100, n):
        k = min(requested_k, n)
        hits = int(ordered[label].iloc[:k].sum())
        precision = hits / k
        low, high = wilson(hits, k)
        rows.append(
            {
                "endpoint": endpoint,
                "candidate_count": n,
                "positives": positives,
                "prevalence": prevalence,
                "k": requested_k,
                "actual_k": k,
                "hits_at_k": hits,
                "precision_at_k": precision,
                "precision_wilson_95_low": low,
                "precision_wilson_95_high": high,
                "enrichment_at_k": precision / prevalence if prevalence else math.nan,
                "hypergeometric_p": (
                    float(hypergeom.sf(hits - 1, n, positives, k)) if hits else 1.0
                ),
            }
        )
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    adjudicated = pd.read_csv(WOS / "wos_outcome_adjudicated.csv", low_memory=False)
    frozen = pd.read_csv(WOS / "frozen_ranking_with_wos_labels.csv", low_memory=False)
    eligible = pd.read_csv(WOS / "wos_eligible_reranked_candidate_universe.csv", low_memory=False)

    observed = adjudicated[
        adjudicated["outcome_scientifically_valid"].eq(True)
        & adjudicated["peer_reviewed_article"].eq(True)
        & adjudicated["rank"].notna()
    ].copy()
    observed = observed.sort_values(["rank", "publication_date"]).drop_duplicates(
        ["system_id", "reaction_id"], keep="first"
    )
    observed["outcome_class"] = observed["confirmed_prior_edge"].map(
        {True: "renewed_or_continued_attention", False: "genuinely_novel_relation"}
    )
    public_columns = [
        "blind_id", "wos_record_id", "doi", "publication_date", "system_id",
        "reaction_id", "rank", "confirmed_prior_edge", "qualifying_new_edge",
        "outcome_class", "reviewer_id",
    ]
    observed[public_columns].to_csv(OUT / "dual_endpoint_observed_relations.csv", index=False)

    activity_pairs = observed[["system_id", "reaction_id"]].copy()
    activity_pairs["future_research_activity"] = 1
    activity_universe = frozen.merge(
        activity_pairs, on=["system_id", "reaction_id"], how="left", validate="one_to_one"
    )
    activity_universe["future_research_activity"] = (
        activity_universe["future_research_activity"].fillna(0).astype(int)
    )

    novelty_universe = eligible.copy()
    novelty_universe["genuinely_novel_relation"] = novelty_universe[
        "wos_independent_hit"
    ].astype(int)

    rows = []
    rows.extend(metrics("future_research_activity", activity_universe, "future_research_activity", "rank"))
    rows.extend(metrics("genuinely_novel_relation", novelty_universe, "genuinely_novel_relation", "wos_eligible_rank"))
    metric_table = pd.DataFrame(rows)
    metric_table.to_csv(OUT / "dual_endpoint_topk_metrics.csv", index=False)

    activity_universe[[
        "rank", "system_id", "reaction_id", "ensemble_score", "future_research_activity",
        "confirmed_prior_edge", "wos_independent_hit",
    ]].to_csv(OUT / "frozen_280_dual_endpoint_labels.csv", index=False)

    activity_top10 = metric_table[
        metric_table["endpoint"].eq("future_research_activity") & metric_table["k"].eq(10)
    ].iloc[0]
    activity_top50 = metric_table[
        metric_table["endpoint"].eq("future_research_activity") & metric_table["k"].eq(50)
    ].iloc[0]
    novelty_top10 = metric_table[
        metric_table["endpoint"].eq("genuinely_novel_relation") & metric_table["k"].eq(10)
    ].iloc[0]
    novelty_top50 = metric_table[
        metric_table["endpoint"].eq("genuinely_novel_relation") & metric_table["k"].eq(50)
    ].iloc[0]
    summary = {
        "analysis_status": "completed_frozen_score_dual_endpoint_analysis",
        "interpretation": "post_audit_secondary_endpoint_with_unchanged_scores",
        "frozen_candidates": int(len(activity_universe)),
        "future_research_activity_pairs": int(activity_universe["future_research_activity"].sum()),
        "genuinely_novel_pairs": int((observed["outcome_class"] == "genuinely_novel_relation").sum()),
        "renewed_or_continued_pairs": int((observed["outcome_class"] == "renewed_or_continued_attention").sum()),
        "activity_top10_hits": int(activity_top10["hits_at_k"]),
        "activity_top10_precision": float(activity_top10["precision_at_k"]),
        "activity_top10_enrichment": float(activity_top10["enrichment_at_k"]),
        "activity_top10_hypergeometric_p": float(activity_top10["hypergeometric_p"]),
        "activity_top50_hits": int(activity_top50["hits_at_k"]),
        "activity_top50_precision": float(activity_top50["precision_at_k"]),
        "activity_top50_enrichment": float(activity_top50["enrichment_at_k"]),
        "activity_top50_hypergeometric_p": float(activity_top50["hypergeometric_p"]),
        "novelty_top10_hits": int(novelty_top10["hits_at_k"]),
        "novelty_top10_enrichment": float(novelty_top10["enrichment_at_k"]),
        "novelty_top10_hypergeometric_p": float(novelty_top10["hypergeometric_p"]),
        "novelty_top50_hits": int(novelty_top50["hits_at_k"]),
        "novelty_top50_enrichment": float(novelty_top50["enrichment_at_k"]),
        "novelty_top50_hypergeometric_p": float(novelty_top50["hypergeometric_p"]),
    }
    (OUT / "dual_endpoint_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
