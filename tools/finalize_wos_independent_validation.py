"""Finalize the WoS audit after the two blinded review files are completed."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
from scipy.stats import hypergeom


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results" / "wos_independent_validation_2026"
OUT = ROOT / "results" / "wos_independent_validation_2026" / "final"
OUTCOME_REVIEW = BASE / "wos_locked_candidate_blinded_review.csv"
PRIOR_REVIEW = BASE / "wos_potential_pre2026_edges_blinded_review.csv"
PRIVATE_KEY = BASE / "wos_locked_candidate_review_key_private.csv"
WOS_WORKS = BASE / "wos_outcome_standardized.csv"
RANKING = ROOT / "results" / "human_validation_2026" / "ranked_with_human_hits_system.csv"


def read_review_csv(path: Path) -> pd.DataFrame:
    """Read reviewer CSVs saved by common English or Chinese Windows locales."""
    errors: list[str] = []
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return pd.read_csv(path, dtype=str, keep_default_na=False, encoding=encoding)
        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}: {exc}")
    raise UnicodeDecodeError("review_csv", b"", 0, 1, "; ".join(errors))


def yes(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"yes", "y", "1", "true"})


def nonempty(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().ne("")


def wilson(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if trials <= 0:
        return math.nan, math.nan
    p = successes / trials
    denominator = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denominator
    radius = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denominator
    return max(0.0, centre - radius), min(1.0, centre + radius)


def incomplete_status(outcome: pd.DataFrame, prior: pd.DataFrame) -> dict:
    outcome_fields = [
        "metallic_hea_valid_yes_no_uncertain",
        "target_catalyst_yes_no_uncertain",
        "reaction_valid_yes_no_uncertain",
        "reviewer_id",
        "review_status",
    ]
    prior_fields = [
        "same_exact_element_set_yes_no_uncertain",
        "same_target_reaction_yes_no_uncertain",
        "target_catalyst_yes_no_uncertain",
        "qualifying_prior_edge_yes_no_uncertain",
        "reviewer_id",
        "review_status",
    ]
    incomplete_outcome = ~outcome[outcome_fields].apply(nonempty).all(axis=1)
    incomplete_prior = ~prior[prior_fields].apply(nonempty).all(axis=1)
    return {
        "status": "awaiting_blinded_review",
        "outcome_rows": int(len(outcome)),
        "outcome_rows_incomplete": int(incomplete_outcome.sum()),
        "historical_rows": int(len(prior)),
        "historical_rows_incomplete": int(incomplete_prior.sum()),
        "outcome_file": str(OUTCOME_REVIEW.resolve()),
        "historical_file": str(PRIOR_REVIEW.resolve()),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    outcome = read_review_csv(OUTCOME_REVIEW)
    prior = read_review_csv(PRIOR_REVIEW)
    status = incomplete_status(outcome, prior)
    if status["outcome_rows_incomplete"] or status["historical_rows_incomplete"]:
        (OUT / "finalization_status.json").write_text(
            json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(status, ensure_ascii=False, indent=2))
        return

    outcome_valid = (
        yes(outcome["metallic_hea_valid_yes_no_uncertain"])
        & yes(outcome["target_catalyst_yes_no_uncertain"])
        & yes(outcome["reaction_valid_yes_no_uncertain"])
    )
    prior_valid = (
        yes(prior["same_exact_element_set_yes_no_uncertain"])
        & yes(prior["same_target_reaction_yes_no_uncertain"])
        & yes(prior["target_catalyst_yes_no_uncertain"])
        & yes(prior["qualifying_prior_edge_yes_no_uncertain"])
    )
    prior_pairs = prior.loc[prior_valid, ["system_id", "reaction_id"]].drop_duplicates()
    prior_pairs["confirmed_prior_edge"] = True
    confirmed_prior_rows = prior.loc[prior_valid].copy()
    confirmed_prior_rows = confirmed_prior_rows.sort_values("publication_date")
    confirmed_prior_events = confirmed_prior_rows.drop_duplicates(
        subset=["system_id", "reaction_id"], keep="first"
    )[[
        "wos_record_id", "doi", "publication_date", "display_name",
        "system_id", "reaction_id", "material_class", "evidence_context",
        "evidence_quote_or_pointer", "reviewer_id",
    ]]
    confirmed_prior_events["event_source"] = "independent_wos_audit"
    confirmed_prior_events.to_csv(
        OUT / "confirmed_wos_historical_event_additions.csv", index=False
    )
    confirmed_prior_events[[
        "wos_record_id", "doi", "publication_date", "system_id",
        "reaction_id", "material_class", "reviewer_id", "event_source",
    ]].to_csv(OUT / "confirmed_wos_historical_event_additions_public.csv", index=False)

    corrected_system = nonempty(outcome["corrected_system_id"])
    corrected_reaction = nonempty(outcome["corrected_reaction_id"])
    outcome.loc[corrected_system, "system_id"] = outcome.loc[
        corrected_system, "corrected_system_id"
    ]
    outcome.loc[corrected_reaction, "reaction_id"] = outcome.loc[
        corrected_reaction, "corrected_reaction_id"
    ]
    outcome["outcome_scientifically_valid"] = outcome_valid.to_numpy()

    key = pd.read_csv(PRIVATE_KEY, dtype=str, keep_default_na=False)
    works = pd.read_csv(WOS_WORKS, dtype=str, keep_default_na=False)[["id", "work_type", "source_name"]]
    ranking = pd.read_csv(RANKING, low_memory=False).sort_values("rank").reset_index(drop=True)
    rank_lookup = ranking[["system_id", "reaction_id", "rank"]].drop_duplicates()
    adjudicated = outcome.merge(
        key[["blind_id", "openalex_work_match_method"]], on="blind_id", how="left", validate="one_to_one"
    ).merge(works, left_on="wos_record_id", right_on="id", how="left", validate="many_to_one").merge(
        rank_lookup, on=["system_id", "reaction_id"], how="left", validate="many_to_one"
    )
    adjudicated = adjudicated.merge(prior_pairs, on=["system_id", "reaction_id"], how="left")
    adjudicated["confirmed_prior_edge"] = adjudicated["confirmed_prior_edge"].map(
        lambda value: bool(value) if pd.notna(value) else False
    )
    adjudicated["qualifying_new_edge"] = (
        adjudicated["outcome_scientifically_valid"] & ~adjudicated["confirmed_prior_edge"]
    )
    adjudicated["peer_reviewed_article"] = adjudicated["work_type"].str.lower().str.startswith("article")
    adjudicated.to_csv(OUT / "wos_outcome_adjudicated.csv", index=False)
    prior.assign(qualifying_prior_edge_final=prior_valid.to_numpy()).to_csv(
        OUT / "wos_historical_prior_edges_adjudicated.csv", index=False
    )
    outcome_public_columns = [
        "blind_id", "wos_record_id", "doi", "publication_date", "system_id",
        "reaction_id", "metallic_hea_valid_yes_no_uncertain",
        "target_catalyst_yes_no_uncertain", "reaction_valid_yes_no_uncertain",
        "corrected_system_id", "corrected_reaction_id", "reviewer_id",
        "review_status", "notes",
    ]
    outcome[outcome_public_columns].assign(
        outcome_scientifically_valid=outcome_valid.to_numpy()
    ).to_csv(OUT / "wos_outcome_blinded_review_public.csv", index=False)
    prior_public_columns = [
        "prior_blind_id", "wos_record_id", "doi", "publication_date", "system_id",
        "reaction_id", "same_exact_element_set_yes_no_uncertain",
        "same_target_reaction_yes_no_uncertain", "target_catalyst_yes_no_uncertain",
        "qualifying_prior_edge_yes_no_uncertain", "reviewer_id", "review_status", "notes",
    ]
    prior[prior_public_columns].assign(
        qualifying_prior_edge_final=prior_valid.to_numpy()
    ).to_csv(OUT / "wos_historical_blinded_review_public.csv", index=False)

    all_new_edges = adjudicated[
        adjudicated["qualifying_new_edge"] & adjudicated["peer_reviewed_article"]
    ].sort_values(["rank", "publication_date"])
    all_new_edges = all_new_edges.drop_duplicates(
        subset=["system_id", "reaction_id"], keep="first"
    )
    all_new_edges.to_csv(OUT / "wos_all_adjudicated_new_edges.csv", index=False)
    hits = all_new_edges[all_new_edges["rank"].notna()].copy()
    hits.to_csv(OUT / "wos_independent_peer_reviewed_hits.csv", index=False)
    all_new_edges[all_new_edges["rank"].isna()].to_csv(
        OUT / "wos_new_edges_outside_frozen_universe.csv", index=False
    )

    hit_pairs = hits[["system_id", "reaction_id"]].drop_duplicates()
    hit_pairs["wos_independent_hit"] = 1
    evaluated = ranking.merge(hit_pairs, on=["system_id", "reaction_id"], how="left")
    evaluated["wos_independent_hit"] = evaluated["wos_independent_hit"].fillna(0).astype(int)
    evaluated = evaluated.merge(prior_pairs, on=["system_id", "reaction_id"], how="left")
    evaluated["confirmed_prior_edge"] = evaluated["confirmed_prior_edge"].map(
        lambda value: bool(value) if pd.notna(value) else False
    )
    evaluated["wos_eligible_candidate"] = ~evaluated["confirmed_prior_edge"]
    eligible = evaluated[evaluated["wos_eligible_candidate"]].copy().reset_index(drop=True)
    eligible["wos_eligible_rank"] = range(1, len(eligible) + 1)
    eligible_five = eligible[eligible["element_count"].astype(int) >= 5].copy().reset_index(drop=True)
    eligible_five["wos_eligible_rank"] = range(1, len(eligible_five) + 1)
    evaluated.to_csv(OUT / "frozen_ranking_with_wos_labels.csv", index=False)
    eligible.to_csv(OUT / "wos_eligible_reranked_candidate_universe.csv", index=False)
    eligible_five.to_csv(
        OUT / "wos_eligible_five_element_candidate_universe.csv", index=False
    )

    rows = []
    for universe_name, universe in [
        ("frozen_280_audit", evaluated),
        ("wos_history_eligible_reranked", eligible),
        ("wos_history_eligible_five_or_more", eligible_five),
    ]:
        n = len(universe)
        positives = int(universe["wos_independent_hit"].sum())
        prevalence = positives / n if n else math.nan
        for requested_k in (10, 20, 50, 100, 280):
            k = min(requested_k, n)
            count = int(universe["wos_independent_hit"].iloc[:k].sum())
            precision = count / k if k else math.nan
            low, high = wilson(count, k)
            rows.append(
                {
                    "candidate_universe": universe_name,
                    "k": requested_k,
                    "actual_k": k,
                    "candidate_count": n,
                    "positives": positives,
                    "prevalence": prevalence,
                    "hits_at_k": count,
                    "precision_at_k": precision,
                    "precision_wilson_95_low": low,
                    "precision_wilson_95_high": high,
                    "enrichment_at_k": precision / prevalence if prevalence else math.nan,
                    "hypergeometric_p": float(hypergeom.sf(count - 1, n, positives, k)) if positives and count else 1.0,
                }
            )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT / "wos_independent_topk_metrics.csv", index=False)

    top10 = metrics[
        metrics["candidate_universe"].eq("wos_history_eligible_reranked")
        & metrics["k"].eq(10)
    ].iloc[0]
    positives = int(eligible["wos_independent_hit"].sum())
    final_status = {
        "status": "completed_external_corpus_audit",
        "outcome_review_rows": int(len(outcome)),
        "scientifically_valid_outcome_rows": int(outcome_valid.sum()),
        "confirmed_historical_prior_edge_rows": int(prior_valid.sum()),
        "confirmed_historical_prior_pairs": int(len(prior_pairs)),
        "wos_eligible_candidate_count": int(len(eligible)),
        "wos_eligible_five_or_more_candidate_count": int(len(eligible_five)),
        "independent_peer_reviewed_new_edge_pairs": positives,
        "additional_new_edge_pairs_outside_frozen_universe": int(
            all_new_edges["rank"].isna().sum()
        ),
        "top10_hits": int(top10["hits_at_k"]),
        "precision_at_10": float(top10["precision_at_k"]),
        "top10_enrichment": None if pd.isna(top10["enrichment_at_k"]) else float(top10["enrichment_at_k"]),
        "top10_hypergeometric_p": float(top10["hypergeometric_p"]),
    }
    (OUT / "finalization_status.json").write_text(
        json.dumps(final_status, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    analysis_summary = f"""### Independent Web of Science sensitivity analysis

An independently exported Web of Science Core Collection corpus was retrieved on 1 October 2026 and screened with the frozen element-set and reaction definitions. After blinded adjudication of {len(outcome)} candidate outcome rows and {len(prior)} possible historical conflicts, {len(prior_pairs)} element-set–reaction pairs were confirmed as already reported before 2026 and removed from the WoS-eligible candidate universe. The independent endpoint contained {positives} qualifying peer-reviewed new links among {len(eligible)} eligible candidates while preserving the original score order. {int(top10['hits_at_k'])} occurred in the reranked Top 10 (Precision@10 = {float(top10['precision_at_k']):.3f}; enrichment = {float(top10['enrichment_at_k']):.2f} when defined; exact hypergeometric p = {float(top10['hypergeometric_p']):.4f}). A separate frozen-280 audit is reported without reranking. This analysis distinguishes database coverage from scientific eligibility and uses the earliest indexed public date to prevent 2025 early-access papers from being counted as 2026 discoveries.
"""
    (OUT / "independent_index_analysis_summary.md").write_text(
        analysis_summary, encoding="utf-8"
    )
    print(json.dumps(final_status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
