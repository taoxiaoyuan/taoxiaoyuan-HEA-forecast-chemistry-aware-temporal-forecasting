from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hea_forecast.config import load_config
from hea_forecast.modeling import _metrics, evaluate_window
from hea_forecast.temporal import (
    build_candidate_table,
    build_historical_training_table,
    expand_element_pairs,
)

PREDICTION_PATTERN = re.compile(r"predictions_(\d{4})_(.+)\.csv$")


def _prediction_files(result_dir: str | Path) -> list[tuple[int, str, Path]]:
    found: list[tuple[int, str, Path]] = []
    for path in Path(result_dir).glob("predictions_*.csv"):
        match = PREDICTION_PATTERN.match(path.name)
        if match:
            found.append((int(match.group(1)), match.group(2), path))
    return sorted(found)


def discovery_lead_times(
    result_dir: str | Path,
    output_csv: str | Path,
    top_fraction: float = 0.10,
) -> dict[str, Any]:
    rows: list[pd.DataFrame] = []
    top_prediction_counts: dict[str, int] = {}
    hit_counts: dict[str, int] = {}
    for cutoff, model, path in _prediction_files(result_dir):
        frame = pd.read_csv(path)
        frame = frame.sort_values("score", ascending=False, kind="stable").reset_index(drop=True)
        frame["rank"] = np.arange(1, len(frame) + 1)
        frame["rank_percentile"] = frame["rank"] / len(frame)
        top_prediction_counts[model] = top_prediction_counts.get(model, 0) + int(
            frame["rank_percentile"].le(top_fraction).sum()
        )
        successful = frame.loc[
            frame["label"].eq(1)
            & frame["first_edge_year"].notna()
            & frame["rank_percentile"].le(top_fraction)
        ].copy()
        if successful.empty:
            continue
        hit_counts[model] = hit_counts.get(model, 0) + len(successful)
        successful["cutoff_year"] = cutoff
        successful["model"] = model
        successful["lead_time_years"] = successful["first_edge_year"] - cutoff
        rows.append(successful)
    combined = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not combined.empty:
        combined = combined.sort_values(
            ["model", "system_id", "reaction_id", "cutoff_year"], kind="stable"
        )
        combined = combined.drop_duplicates(["model", "system_id", "reaction_id"], keep="first")
    destination = Path(output_csv)
    destination.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(destination, index=False)
    summary = {
        "top_fraction": top_fraction,
        "successful_predictions": len(combined),
        "by_model": {},
    }
    if not combined.empty:
        for model, group in combined.groupby("model"):
            summary["by_model"][model] = {
                "count": len(group),
                "top_predictions": top_prediction_counts.get(model, 0),
                "hits_across_windows": hit_counts.get(model, 0),
                "top_fraction_precision": (
                    hit_counts.get(model, 0) / top_prediction_counts[model]
                    if top_prediction_counts.get(model, 0)
                    else 0.0
                ),
                "median_lead_time_years": float(group["lead_time_years"].median()),
                "mean_lead_time_years": float(group["lead_time_years"].mean()),
            }
    destination.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def case_analysis(
    result_dir: str | Path,
    output_csv: str | Path,
    cutoff_year: int = 2022,
    per_group: int = 10,
) -> dict[str, Any]:
    result_path = Path(result_dir)
    metrics = pd.read_csv(result_path / "metrics.csv")
    eligible = metrics.loc[metrics["cutoff_year"].eq(cutoff_year) & metrics["model"].ne("random")]
    if eligible.empty:
        raise ValueError(f"No evaluated models for cutoff {cutoff_year}")
    best_model = str(eligible.sort_values("pr_auc", ascending=False).iloc[0]["model"])
    frame = pd.read_csv(result_path / f"predictions_{cutoff_year}_{best_model}.csv")
    frame = frame.sort_values("score", ascending=False, kind="stable").reset_index(drop=True)
    frame["rank"] = np.arange(1, len(frame) + 1)
    true_positive = frame.loc[frame["label"].eq(1)].head(per_group).copy()
    true_positive["case_type"] = "successful_prediction"
    false_positive = frame.loc[frame["label"].eq(0)].head(per_group).copy()
    false_positive["case_type"] = "high_rank_unobserved"
    missed = frame.loc[frame["label"].eq(1)].tail(per_group).copy()
    missed["case_type"] = "missed_positive"
    cases = pd.concat([true_positive, false_positive, missed], ignore_index=True)
    destination = Path(output_csv)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cases.to_csv(destination, index=False)
    return {
        "cutoff_year": cutoff_year,
        "model": best_model,
        "case_rows": len(cases),
        "output": str(destination.resolve()),
    }


def label_sensitivity(
    result_dir: str | Path,
    output_csv: str | Path,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for cutoff, model, path in _prediction_files(result_dir):
        frame = pd.read_csv(path)
        scenarios = {"strict": frame}
        late_event = frame["first_edge_year"].notna() & frame["first_edge_year"].gt(
            frame["horizon_end"]
        )
        scenarios["exclude_late_discoveries"] = frame.loc[~late_event].copy()
        scenarios["minimum_five_elements"] = frame.loc[frame["element_count"].ge(5)].copy()
        for scenario, subset in scenarios.items():
            if subset.empty:
                continue
            metrics = _metrics(subset["label"].astype(int).to_numpy(), subset["score"].to_numpy())
            rows.append(
                {
                    "cutoff_year": cutoff,
                    "model": model,
                    "scenario": scenario,
                    "candidates": len(subset),
                    "positives": int(subset["label"].sum()),
                    **metrics,
                }
            )
    destination = Path(output_csv)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(destination, index=False)
    return {"rows": len(rows), "output": str(destination.resolve())}


def rank_future_candidates(
    input_csv: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
    cutoff_year: int = 2025,
    horizon_end: int = 2028,
) -> dict[str, Any]:
    evidence = pd.read_csv(input_csv, low_memory=False)
    config = load_config(config_path)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"cutoff_year": cutoff_year, "horizon_end": horizon_end}
    for entity_level in ["system", "element_pair"]:
        level_evidence = (
            expand_element_pairs(evidence) if entity_level == "element_pair" else evidence
        )
        train = build_historical_training_table(
            level_evidence, list(config["reactions"]), cutoff_year
        )
        candidates = build_candidate_table(
            level_evidence,
            list(config["reactions"]),
            cutoff_year,
            cutoff_year + 1,
            horizon_end,
        )
        _, predictions = evaluate_window(train, candidates)
        ranked = candidates.copy()
        rank_scores: list[str] = []
        for model, prediction in predictions.items():
            if model == "random":
                continue
            score_column = f"score_{model}"
            scores = prediction.set_index(["system_id", "reaction_id"])["score"]
            ranked[score_column] = [
                scores.loc[(system, reaction)]
                for system, reaction in zip(
                    ranked["system_id"], ranked["reaction_id"], strict=False
                )
            ]
            percentile_column = f"percentile_{model}"
            ranked[percentile_column] = ranked[score_column].rank(pct=True, method="average")
            rank_scores.append(percentile_column)
        ranked["ensemble_score"] = ranked[rank_scores].mean(axis=1)
        ranked = ranked.sort_values("ensemble_score", ascending=False, kind="stable")
        ranked.to_csv(destination / f"future_candidates_{entity_level}.csv", index=False)
        ranked.head(50).to_csv(
            destination / f"future_candidates_{entity_level}_top50.csv", index=False
        )
        summary[entity_level] = {
            "candidates": len(ranked),
            "models": len(rank_scores),
            "top_output": str(
                (destination / f"future_candidates_{entity_level}_top50.csv").resolve()
            ),
        }
    (destination / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def validate_future_candidates(
    future_dir: str | Path,
    validation_csv: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Match frozen 2025 rankings to machine-extracted, still-unreviewed 2026 events."""
    validation = pd.read_csv(validation_csv, low_memory=False).fillna("")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "status": "provisional_machine_extraction",
        "validation_rows": len(validation),
    }
    for entity_level in ["system", "element_pair"]:
        events = expand_element_pairs(validation) if entity_level == "element_pair" else validation
        events = events.drop_duplicates(["system_id", "reaction_id"])
        high_confidence_events = events.loc[
            events.get("extraction_confidence", "").astype(str).str.lower().eq("high")
        ]
        candidates = pd.read_csv(
            Path(future_dir) / f"future_candidates_{entity_level}.csv"
        ).reset_index(drop=True)
        candidates["rank"] = np.arange(1, len(candidates) + 1)
        event_pairs = set(zip(events["system_id"], events["reaction_id"], strict=False))
        high_confidence_pairs = set(
            zip(
                high_confidence_events["system_id"],
                high_confidence_events["reaction_id"],
                strict=False,
            )
        )
        candidates["provisional_2026_hit"] = [
            int((system, reaction) in event_pairs)
            for system, reaction in zip(
                candidates["system_id"], candidates["reaction_id"], strict=False
            )
        ]
        candidates["high_confidence_2026_hit"] = [
            int((system, reaction) in high_confidence_pairs)
            for system, reaction in zip(
                candidates["system_id"], candidates["reaction_id"], strict=False
            )
        ]
        matched = candidates.loc[candidates["provisional_2026_hit"].eq(1)].merge(
            events[
                [
                    column
                    for column in [
                        "system_id",
                        "reaction_id",
                        "paper_id",
                        "doi",
                        "publication_date",
                        "display_name",
                        "extraction_confidence",
                    ]
                    if column in events.columns
                ]
            ],
            on=["system_id", "reaction_id"],
            how="left",
        )
        matched.to_csv(destination / f"provisional_matches_{entity_level}.csv", index=False)
        all_ranked_path = destination / f"ranked_with_2026_hits_{entity_level}.csv"
        candidates.to_csv(all_ranked_path, index=False)
        top_metrics = {}
        for k in [10, 20, 50, 100]:
            actual_k = min(k, len(candidates))
            hits = int(candidates.head(actual_k)["provisional_2026_hit"].sum())
            top_metrics[f"precision_at_{k}"] = hits / actual_k if actual_k else 0.0
            top_metrics[f"hits_at_{k}"] = hits
            high_hits = int(candidates.head(actual_k)["high_confidence_2026_hit"].sum())
            top_metrics[f"high_confidence_precision_at_{k}"] = (
                high_hits / actual_k if actual_k else 0.0
            )
            top_metrics[f"high_confidence_hits_at_{k}"] = high_hits
        summary[entity_level] = {
            "candidate_count": len(candidates),
            "provisional_hits": int(candidates["provisional_2026_hit"].sum()),
            "provisional_prevalence": float(candidates["provisional_2026_hit"].mean()),
            "high_confidence_hits": int(candidates["high_confidence_2026_hit"].sum()),
            "high_confidence_prevalence": float(candidates["high_confidence_2026_hit"].mean()),
            **top_metrics,
        }
    (destination / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary
