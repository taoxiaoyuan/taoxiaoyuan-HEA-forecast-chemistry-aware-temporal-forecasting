from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from hea_forecast.elemental_features import add_elemental_features


def _precision_at_k(labels: np.ndarray, scores: np.ndarray, k: int) -> float:
    actual = min(k, len(labels))
    if actual == 0:
        return 0.0
    order = np.argsort(-scores, kind="stable")[:actual]
    return float(labels[order].mean())


def historical_randomisation_audit(
    prediction_dirs: dict[str, str | Path],
    output_csv: str | Path,
    permutations: int = 2_000,
    seed: int = 20260926,
) -> dict[str, Any]:
    """Test whether historical rankings beat exchangeable-label null models.

    This is deliberately a post-model diagnostic: scores are held fixed and
    labels are permuted within each temporal test window.  It therefore tests
    ranking information beyond prevalence without retraining on future data.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for entity_level, directory in prediction_dirs.items():
        for path in sorted(Path(directory).glob("predictions_*.csv")):
            frame = pd.read_csv(path)
            if frame.empty or frame["label"].nunique() < 2:
                continue
            labels = frame["label"].astype(int).to_numpy()
            scores = pd.to_numeric(frame["score"], errors="coerce").fillna(0).to_numpy()
            observed_ap = float(average_precision_score(labels, scores))
            observed_p20 = _precision_at_k(labels, scores, 20)
            null_ap = np.empty(permutations, dtype=float)
            null_p20 = np.empty(permutations, dtype=float)
            for index in range(permutations):
                permuted = rng.permutation(labels)
                null_ap[index] = average_precision_score(permuted, scores)
                null_p20[index] = _precision_at_k(permuted, scores, 20)
            parts = path.stem.split("_", 2)
            cutoff = int(parts[1])
            model = parts[2]
            rows.append(
                {
                    "entity_level": entity_level,
                    "cutoff_year": cutoff,
                    "model": model,
                    "candidates": len(frame),
                    "positives": int(labels.sum()),
                    "prevalence": float(labels.mean()),
                    "observed_pr_auc": observed_ap,
                    "null_pr_auc_mean": float(null_ap.mean()),
                    "null_pr_auc_95_low": float(np.quantile(null_ap, 0.025)),
                    "null_pr_auc_95_high": float(np.quantile(null_ap, 0.975)),
                    "pr_auc_empirical_p": float(
                        (1 + np.count_nonzero(null_ap >= observed_ap)) / (permutations + 1)
                    ),
                    "observed_precision_at_20": observed_p20,
                    "null_precision_at_20_mean": float(null_p20.mean()),
                    "precision_at_20_empirical_p": float(
                        (1 + np.count_nonzero(null_p20 >= observed_p20))
                        / (permutations + 1)
                    ),
                    "permutations": permutations,
                    "seed": seed,
                }
            )
    destination = Path(output_csv)
    destination.parent.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows).sort_values(
        ["entity_level", "cutoff_year", "model"], kind="stable"
    )
    result.to_csv(destination, index=False)
    summary = {
        "rows": len(result),
        "permutations_per_row": permutations,
        "seed": seed,
        "significant_pr_auc_rows_0_05": int(result["pr_auc_empirical_p"].lt(0.05).sum()),
        "interpretation": (
            "Fixed historical scores were compared with labels permuted inside each test window; "
            "this is a ranking-information negative control, not an independent-corpus validation."
        ),
    }
    destination.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def build_candidate_chemistry_screen(
    future_csv: str | Path,
    output_csv: str | Path,
) -> dict[str, Any]:
    """Attach non-DFT composition, supply and transparent feasibility proxies."""
    frame = pd.read_csv(future_csv, low_memory=False)
    frame = add_elemental_features(frame)
    size = frame["chem_atomic_size_mismatch"].fillna(
        frame["chem_atomic_size_mismatch"].median()
    )
    electronegativity = frame["chem_electronegativity_std"].fillna(
        frame["chem_electronegativity_std"].median()
    )
    frame["empirical_solid_solution_proxy"] = (
        0.65 * (1 - size / 10).clip(0, 1)
        + 0.35 * (1 - electronegativity / 0.5).clip(0, 1)
    )
    frame["supply_burden_proxy"] = (
        0.7 * frame["screen_platinum_group_fraction"]
        + 0.3 * frame["screen_critical_transition_fraction"]
    ).clip(0, 1)
    frame["abundance_proxy"] = frame["screen_abundant_3d_fraction"]
    frame["chemistry_screen_score"] = (
        0.45 * frame["ensemble_score"]
        + 0.30 * frame["empirical_solid_solution_proxy"]
        + 0.15 * frame["abundance_proxy"]
        + 0.10 * (1 - frame["supply_burden_proxy"])
    )
    frame = frame.sort_values("chemistry_screen_score", ascending=False, kind="stable")
    frame["chemistry_screen_rank"] = np.arange(1, len(frame) + 1)
    destination = Path(output_csv)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    summary = {
        "candidate_rows": len(frame),
        "descriptor_scope": "equiatomic composition-only and versioned screening groups",
        "warning": (
            "The screen is not DFT, CALPHAD, a phase-stability calculation, a live market-price "
            "estimate, or evidence of catalytic performance."
        ),
    }
    destination.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def prepare_expert_blind_study(
    future_csv: str | Path,
    output_dir: str | Path,
    per_stratum: int = 10,
    seed: int = 20260926,
) -> dict[str, Any]:
    """Create a score-blinded, stratified expert-ranking package and private key."""
    frame = pd.read_csv(future_csv, low_memory=False).sort_values(
        "ensemble_score", ascending=False, kind="stable"
    )
    frame["locked_rank"] = np.arange(1, len(frame) + 1)
    n = len(frame)
    strata = {
        "high": frame.head(max(per_stratum, int(np.ceil(n * 0.10)))).head(per_stratum),
        "middle": frame.iloc[max(0, n // 2 - per_stratum // 2) :][:per_stratum],
        "low": frame.tail(max(per_stratum, int(np.ceil(n * 0.10)))).tail(per_stratum),
    }
    selected = pd.concat(
        [subset.assign(sampling_stratum=name) for name, subset in strata.items()],
        ignore_index=True,
    ).drop_duplicates(["system_id", "reaction_id"])
    rng = np.random.default_rng(seed)
    selected = selected.iloc[rng.permutation(len(selected))].reset_index(drop=True)
    selected["candidate_code"] = [f"E{index:03d}" for index in range(1, len(selected) + 1)]
    public = selected[["candidate_code", "system_id", "reaction_id"]].copy()
    for column in [
        "chemical_plausibility_1_to_5",
        "synthesis_feasibility_1_to_5",
        "research_value_1_to_5",
        "future_report_likelihood_1_to_5",
        "overall_priority_1_to_5",
        "confidence_1_to_5",
        "exclude_yes_no",
        "reasoning_notes",
        "reviewer_id",
        "review_date",
    ]:
        public[column] = ""
    key_columns = [
        "candidate_code",
        "system_id",
        "reaction_id",
        "locked_rank",
        "ensemble_score",
        "sampling_stratum",
    ]
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    public.to_csv(destination / "expert_blind_review_template.csv", index=False)
    selected[key_columns].to_csv(destination / "PRIVATE_expert_study_key.csv", index=False)
    protocol = """# Blinded expert-comparison protocol

1. Recruit at least three HEA/electrocatalysis researchers who did not create the locked ranking.
2. Give reviewers only `expert_blind_review_template.csv`; do not share the private key, model
   scores, strata, literature outcomes, or other reviewers' responses.
3. Reviewers score every row independently using the six 1--5 fields and document exclusions.
4. Freeze and hash completed files before unblinding.
5. Primary expert endpoint: mean overall-priority score. Compare AI, the median expert, and a
   prespecified AI--expert combination against subsequently observed literature outcomes.
6. Report inter-rater agreement and retain disagreements; do not redefine the locked AI endpoint.

This package prepares the study but contains no expert results.
"""
    (destination / "expert_blind_study_protocol.md").write_text(protocol, encoding="utf-8")
    summary = {
        "candidates": len(selected),
        "per_stratum_requested": per_stratum,
        "seed": seed,
        "status": "template_only_no_expert_results",
    }
    (destination / "expert_study_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def prepare_independent_corpus_validation(
    future_csv: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Create a complete intake table for WoS/Scopus/Lens independent validation."""
    candidates = pd.read_csv(future_csv, low_memory=False).sort_values(
        "ensemble_score", ascending=False, kind="stable"
    )
    candidates["locked_rank"] = np.arange(1, len(candidates) + 1)
    intake_columns = [
        "external_database",
        "database_query_version",
        "retrieval_date",
        "external_record_id",
        "doi",
        "title",
        "publication_date",
        "document_type",
        "candidate_system_id",
        "candidate_reaction_id",
        "metallic_hea_valid_yes_no_uncertain",
        "target_catalyst_yes_no_uncertain",
        "reaction_valid_yes_no_uncertain",
        "first_report_in_external_corpus_yes_no_uncertain",
        "evidence_quote_or_pointer",
        "reviewer_id",
        "adjudication_status",
        "notes",
    ]
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=intake_columns).to_csv(
        destination / "external_corpus_records_template.csv", index=False
    )
    candidates[
        ["locked_rank", "system_id", "reaction_id", "ensemble_score"]
    ].rename(
        columns={"system_id": "candidate_system_id", "reaction_id": "candidate_reaction_id"}
    ).to_csv(destination / "locked_candidate_key.csv", index=False)
    protocol = """# Independent discovery-corpus validation protocol

## Eligible sources
Use Web of Science, Scopus, or Lens as the primary independent discovery corpus. Crossref may
verify DOI/title identity but is not, by itself, an independent discovery-corpus test.

## Locked analysis
- Do not retrain or rerank the 280 candidates.
- Freeze the database name, complete query, retrieval date, and export before outcome review.
- Deduplicate journal/preprint versions by DOI and normalized title.
- Apply the existing metallic-HEA, target-catalyst, reaction, and first-report rules.
- Have a second reviewer independently assess every putative Top-50 match and all ambiguous rows.

## Primary replication endpoint
Peer-reviewed exact element-set--reaction Precision@10 in the unchanged candidate universe.
Also report Precision@20/50, prevalence, enrichment, Wilson intervals, and exact hypergeometric
tests. Differences from OpenAlex first-report dates must be listed rather than silently replaced.

The supplied CSV is an empty intake template; no independent-corpus result is claimed yet.
"""
    (destination / "independent_corpus_validation_protocol.md").write_text(
        protocol, encoding="utf-8"
    )
    summary = {
        "locked_candidates": len(candidates),
        "status": "protocol_and_empty_intake_only",
        "acceptable_primary_sources": ["Web of Science", "Scopus", "Lens"],
        "crossref_role": "identity verification only",
    }
    (destination / "external_validation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary
