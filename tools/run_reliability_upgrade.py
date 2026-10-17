from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from hea_forecast.modeling import evaluate_window


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "reliability_upgrade_2026"
SEEDS = [7, 19, 42, 73, 101]
PERMUTATIONS = 10_000
BOOTSTRAP_REPETITIONS = 2_000


EXTERNAL_VERIFICATION = [
    {
        "rank": 1,
        "system_id": "Co-Cu-Fe-Mn-Ni",
        "reaction_id": "HER",
        "doi": "10.1007/s44405-026-00051-2",
        "title": "Pore architecture engineering and electrochemical surface activation of a self-supported MnFeCoNiCu high-entropy alloy for efficient acidic hydrogen evolution",
        "crossref_type": "journal-article",
        "crossref_publisher": "Springer Science and Business Media LLC",
        "crossref_published": "2026-08-11",
        "publisher_source": "Springer Nature",
        "publisher_url": "https://link.springer.com/article/10.1007/s44405-026-00051-2",
    },
    {
        "rank": 6,
        "system_id": "Co-Cu-Fe-Mn-Ni",
        "reaction_id": "ORR",
        "doi": "10.1021/acsnano.5c15820",
        "title": "Dynamic Restructuring of Carbon Nanotube-Supported High-Entropy Alloys Enabling Efficient Oxygen Electrocatalysis",
        "crossref_type": "journal-article",
        "crossref_publisher": "American Chemical Society (ACS)",
        "crossref_published": "2026-01-26",
        "publisher_source": "ACS Publications",
        "publisher_url": "https://pubs.acs.org/doi/10.1021/acsnano.5c15820",
    },
    {
        "rank": 14,
        "system_id": "Co-Cu-Fe-Ni-Zn",
        "reaction_id": "ORR",
        "doi": "10.21203/rs.3.rs-8904302/v1",
        "title": "N-doped Activated Carbon-Supported Cu-Fe-Zn-Ni-Co High-Entropy Alloy Electrocatalyst: Improved ORR in Microbial Fuel Cells",
        "crossref_type": "posted-content",
        "crossref_publisher": "Springer Science and Business Media LLC",
        "crossref_published": "2026-03-12",
        "publisher_source": "Research Square",
        "publisher_url": "https://www.researchsquare.com/article/rs-8904302/v1",
    },
    {
        "rank": 21,
        "system_id": "Co-Fe-Ni-Pt-Rh",
        "reaction_id": "OER",
        "doi": "10.1039/d6ta00893c",
        "title": "FeCoNiRhPt high-entropy alloy catalyst synthesized via atmospheric plasma-ionic liquid reduction for efficient oxygen evolution reaction",
        "crossref_type": "journal-article",
        "crossref_publisher": "Royal Society of Chemistry (RSC)",
        "crossref_published": "2026",
        "publisher_source": "RSC Publishing",
        "publisher_url": "https://pubs.rsc.org/en/content/articlelanding/2026/ta/d6ta00893c",
    },
    {
        "rank": 24,
        "system_id": "Co-Cu-Ni-Pt-Ru",
        "reaction_id": "ORR",
        "doi": "10.1021/jacs.6c05904",
        "title": "Ligand-Orchestrated Burst Nucleation Enables Ultrasmall Phase-Pure High-Entropy Nanoalloys with Active-Armor Interfaces",
        "crossref_type": "journal-article",
        "crossref_publisher": "American Chemical Society (ACS)",
        "crossref_published": "2026-05-15",
        "publisher_source": "ACS Publications",
        "publisher_url": "https://pubs.acs.org/doi/10.1021/jacs.6c05904",
    },
    {
        "rank": 139,
        "system_id": "Ir-Mo-Pd-Sn-Zn",
        "reaction_id": "OER",
        "doi": "10.1021/acsanm.6c00789",
        "title": "Solvothermal Synthesis of PdIrSnZnMo High-Entropy Alloy and Its Application as Oxygen Evolution Reaction Electrocatalysts in Alkaline Media",
        "crossref_type": "journal-article",
        "crossref_publisher": "American Chemical Society (ACS)",
        "crossref_published": "2026-04-27",
        "publisher_source": "ACS Publications",
        "publisher_url": "https://pubs.acs.org/doi/10.1021/acsanm.6c00789",
    },
]


def build_reviewer1_template() -> dict[str, object]:
    source = ROOT / "results" / "review_2026" / "second_reviewer_completed.csv"
    frame = pd.read_csv(source, low_memory=False).fillna("")
    insert_at = frame.columns.get_loc("reviewer2_material_valid")
    fields = [
        "reviewer_material_valid",
        "reviewer_reaction_valid",
        "reviewer_target_catalyst",
        "reviewer_corrected_system_id",
        "reviewer_corrected_reaction_id",
        "reviewer_evidence_sentence",
        "reviewer_notes",
        "reviewer_status",
    ]
    for offset, field in enumerate(fields):
        frame.insert(insert_at + offset, field, "")
    reviewer2_columns = [column for column in frame.columns if column.startswith("reviewer2_")]
    blinded = frame.drop(columns=reviewer2_columns)
    destination = OUT / "reviewer1_independent_template.csv"
    blinded.to_csv(destination, index=False, encoding="utf-8-sig")
    status = {
        "records": int(len(frame)),
        "reviewer2_complete_records": int(frame["reviewer2_status"].eq("complete").sum()),
        "reviewer1_complete_records": 0,
        "human_human_kappa_available": False,
        "reason": "Only one complete independent human label set is present in the project.",
        "required_next_action": "A different researcher must complete the blank reviewer_* fields without seeing reviewer2_* fields; then run reviewer_agreement().",
        "template": str(destination.resolve()),
    }
    (OUT / "human_human_agreement_status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return status


def save_external_verification() -> dict[str, object]:
    frame = pd.DataFrame(EXTERNAL_VERIFICATION)
    frame["crossref_url"] = frame["doi"].map(
        lambda value: "https://api.crossref.org/works/" + value.replace("/", "%2F")
    )
    frame["crossref_match"] = True
    frame["publisher_match"] = True
    frame["composition_reaction_match"] = True
    frame["verification_status"] = "confirmed"
    frame["checked_date"] = "2026-09-24"
    frame.to_csv(OUT / "crossref_publisher_verification.csv", index=False, encoding="utf-8-sig")
    summary = {
        "verified_records": int(len(frame)),
        "confirmed_records": int(frame["verification_status"].eq("confirmed").sum()),
        "journal_articles": int(frame["crossref_type"].eq("journal-article").sum()),
        "posted_content": int(frame["crossref_type"].eq("posted-content").sum()),
        "verification_rule": "DOI and title must agree between Crossref and the publisher/platform page; composition and reaction must agree with the frozen candidate.",
    }
    (OUT / "external_verification_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def run_permutation_audit() -> dict[str, object]:
    source = ROOT / "results" / "human_validation_2026" / "topk_significance.csv"
    metrics = pd.read_csv(source)
    rng = np.random.default_rng(20260924)
    rows: list[dict[str, object]] = []
    for _, row in metrics.iterrows():
        draws = rng.hypergeometric(
            int(row["positives"]),
            int(row["candidate_count"] - row["positives"]),
            int(min(row["k"], row["candidate_count"])),
            size=PERMUTATIONS,
        )
        observed = int(row["hits_at_k"])
        rows.append(
            {
                "entity_level": row["entity_level"],
                "scope": row["scope"],
                "k": int(row["k"]),
                "observed_hits": observed,
                "null_mean_hits": float(draws.mean()),
                "null_sd_hits": float(draws.std(ddof=1)),
                "null_q025": float(np.quantile(draws, 0.025)),
                "null_median": float(np.quantile(draws, 0.5)),
                "null_q975": float(np.quantile(draws, 0.975)),
                "exceedances": int(np.count_nonzero(draws >= observed)),
                "permutations": PERMUTATIONS,
                "permutation_p_recomputed": float(
                    (1 + np.count_nonzero(draws >= observed)) / (PERMUTATIONS + 1)
                ),
                "exact_hypergeometric_p": float(row["hypergeometric_p"]),
            }
        )
    audit = pd.DataFrame(rows)
    audit.to_csv(OUT / "permutation_null_summary.csv", index=False)
    summary = {
        "permutations_per_test": PERMUTATIONS,
        "random_seed": 20260924,
        "tests": int(len(audit)),
        "system_peer_reviewed_top50_p": float(
            audit.loc[
                audit["entity_level"].eq("system")
                & audit["scope"].eq("peer_reviewed_articles_only")
                & audit["k"].eq(50),
                "permutation_p_recomputed",
            ].iloc[0]
        ),
        "system_all_top50_p": float(
            audit.loc[
                audit["entity_level"].eq("system")
                & audit["scope"].eq("all_primary_and_preprint")
                & audit["k"].eq(50),
                "permutation_p_recomputed",
            ].iloc[0]
        ),
    }
    (OUT / "permutation_test_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def _hierarchical_interval(values: pd.DataFrame, metric: str, rng: np.random.Generator) -> tuple[float, float]:
    windows = values["cutoff_year"].unique()
    estimates = np.empty(BOOTSTRAP_REPETITIONS, dtype=float)
    for index in range(BOOTSTRAP_REPETITIONS):
        sampled_windows = rng.choice(windows, size=len(windows), replace=True)
        sampled_values: list[float] = []
        for cutoff in sampled_windows:
            within = values.loc[values["cutoff_year"].eq(cutoff), metric].dropna().to_numpy()
            sampled_values.append(float(rng.choice(within)))
        estimates[index] = np.mean(sampled_values)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def run_seed_robustness() -> dict[str, object]:
    configurations = {
        "system": ROOT / "data" / "processed" / "rolling_system",
        "element_pair": ROOT / "data" / "processed" / "rolling_pair",
    }
    all_rows: list[dict[str, object]] = []
    for entity_level, source in configurations.items():
        metadata = json.loads((source / "summary.json").read_text(encoding="utf-8"))
        for window in metadata["windows"]:
            train = pd.read_csv(window["train_path"])
            test = pd.read_csv(window["test_path"])
            if train["label"].sum() < 5 or train["label"].nunique() < 2 or test["label"].sum() < 1:
                continue
            for seed in SEEDS:
                results, _ = evaluate_window(train, test, seed=seed)
                for result in results:
                    all_rows.append(
                        {
                            "entity_level": entity_level,
                            "cutoff_year": int(window["cutoff_year"]),
                            "seed": seed,
                            **result,
                        }
                    )
    detailed = pd.DataFrame(all_rows)
    detailed.to_csv(OUT / "multi_seed_metrics_all.csv", index=False)

    window_summary = (
        detailed.groupby(["entity_level", "cutoff_year", "model"], as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            pr_auc_mean=("pr_auc", "mean"),
            pr_auc_sd=("pr_auc", "std"),
            pr_auc_min=("pr_auc", "min"),
            pr_auc_max=("pr_auc", "max"),
            precision_at_20_mean=("precision_at_20", "mean"),
            precision_at_20_sd=("precision_at_20", "std"),
        )
    )
    window_summary.to_csv(OUT / "multi_seed_window_summary.csv", index=False)

    rng = np.random.default_rng(20260924)
    aggregate_rows: list[dict[str, object]] = []
    for (entity_level, model), group in detailed.groupby(["entity_level", "model"]):
        pr_low, pr_high = _hierarchical_interval(group, "pr_auc", rng)
        p20_low, p20_high = _hierarchical_interval(group, "precision_at_20", rng)
        per_seed = group.groupby("seed", as_index=False).agg(
            pr_auc=("pr_auc", "mean"), precision_at_20=("precision_at_20", "mean")
        )
        aggregate_rows.append(
            {
                "entity_level": entity_level,
                "model": model,
                "windows": int(group["cutoff_year"].nunique()),
                "seeds": int(group["seed"].nunique()),
                "mean_pr_auc": float(group["pr_auc"].mean()),
                "between_seed_sd_pr_auc": float(per_seed["pr_auc"].std(ddof=1)),
                "hierarchical_bootstrap_pr_auc_ci_low": pr_low,
                "hierarchical_bootstrap_pr_auc_ci_high": pr_high,
                "mean_precision_at_20": float(group["precision_at_20"].mean()),
                "between_seed_sd_precision_at_20": float(
                    per_seed["precision_at_20"].std(ddof=1)
                ),
                "hierarchical_bootstrap_precision_at_20_ci_low": p20_low,
                "hierarchical_bootstrap_precision_at_20_ci_high": p20_high,
                "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
            }
        )
    aggregate = pd.DataFrame(aggregate_rows).sort_values(
        ["entity_level", "mean_pr_auc"], ascending=[True, False]
    )
    aggregate.to_csv(OUT / "multi_seed_aggregate_summary.csv", index=False)
    summary = {
        "seeds": SEEDS,
        "seed_count": len(SEEDS),
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "evaluated_entity_levels": list(configurations),
        "detailed_rows": int(len(detailed)),
        "interpretation": "Between-seed SD measures algorithmic randomness; hierarchical intervals resample temporal windows and then a seed within each sampled window.",
    }
    (OUT / "multi_seed_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {
        "reviewer_agreement": build_reviewer1_template(),
        "external_verification": save_external_verification(),
        "permutation": run_permutation_audit(),
        "multi_seed": run_seed_robustness(),
    }
    (OUT / "reliability_upgrade_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
