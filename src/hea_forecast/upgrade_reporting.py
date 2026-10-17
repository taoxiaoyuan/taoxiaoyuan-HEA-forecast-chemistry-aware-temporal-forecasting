from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _aggregate_metrics(path: str | Path, level: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = pd.read_csv(path)
    metrics["entity_level"] = level
    summary = (
        metrics.groupby("model", as_index=False)
        .agg(
            windows=("cutoff_year", "nunique"),
            mean_pr_auc=("pr_auc", "mean"),
            sd_pr_auc=("pr_auc", "std"),
            median_precision_at_20=("precision_at_20", "median"),
            mean_brier=("brier_score", "mean"),
            mean_ece=("expected_calibration_error", "mean"),
        )
        .sort_values("mean_pr_auc", ascending=False)
    )
    summary.insert(0, "entity_level", level)
    return metrics, summary


def build_upgrade_report_tables(
    rolling_system_metrics: str | Path,
    rolling_pair_metrics: str | Path,
    future_system_csv: str | Path,
    strict_matches_csv: str | Path,
    actionability_csv: str | Path,
    output_dir: str | Path,
) -> dict[str, object]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    system, system_summary = _aggregate_metrics(rolling_system_metrics, "system")
    pair, pair_summary = _aggregate_metrics(rolling_pair_metrics, "element_pair")
    rolling = pd.concat([system, pair], ignore_index=True)
    aggregate = pd.concat([system_summary, pair_summary], ignore_index=True)
    rolling.to_csv(destination / "rolling_metrics_all.csv", index=False)
    aggregate.to_csv(destination / "rolling_metrics_summary.csv", index=False)

    future = pd.read_csv(future_system_csv)
    percentile_cols = [column for column in future if column.startswith("percentile_")]
    future["model_percentile_mean"] = future[percentile_cols].mean(axis=1)
    future["model_percentile_sd"] = future[percentile_cols].std(axis=1)
    future["models_in_top_decile"] = future[percentile_cols].ge(0.90).sum(axis=1)
    future["ranking_confidence"] = pd.cut(
        future["model_percentile_sd"],
        bins=[-np.inf, 0.10, 0.20, np.inf],
        labels=["high", "moderate", "low"],
    ).astype(str)
    future.sort_values("ensemble_score", ascending=False).head(50).to_csv(
        destination / "future_top50_with_uncertainty.csv", index=False
    )

    strict = pd.read_csv(strict_matches_csv, low_memory=False)
    strict = strict.sort_values("rank").copy()
    strict["lead_time_status"] = "prospective match in 2026"
    case_columns = [
        "rank",
        "system_id",
        "reaction_id",
        "ensemble_score",
        "doi",
        "publication_date",
        "display_name",
        "extraction_confidence",
        "lead_time_status",
    ]
    strict[[column for column in case_columns if column in strict]].to_csv(
        destination / "prospective_success_cases.csv", index=False
    )

    actionability = pd.read_csv(actionability_csv)
    actionability.head(20).to_csv(destination / "actionable_top20.csv", index=False)

    # Figure: rolling PR-AUC with prevalence represented by the random baseline.
    selected = [
        "random",
        "popularity",
        "logistic_topology",
        "random_forest_topology",
        "logistic_text",
        "random_forest_chemistry",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for axis, (level, frame) in zip(
        axes, [("Exact composition", system), ("Element pair", pair)], strict=False
    ):
        for model in selected:
            subset = frame.loc[frame["model"].eq(model)].sort_values("cutoff_year")
            if not subset.empty:
                axis.plot(
                    subset["cutoff_year"],
                    subset["pr_auc"],
                    marker="o",
                    label=model.replace("_", " "),
                )
        axis.set_title(level)
        axis.set_xlabel("Training cutoff year")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("PR-AUC in the following year")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.savefig(destination / "rolling_one_year_pr_auc.png", dpi=300)
    fig.savefig(destination / "rolling_one_year_pr_auc.pdf")
    plt.close(fig)

    # Figure: actionable candidates, preserving the component scores.
    top = actionability.head(15).sort_values("actionability_score")
    labels = top["system_id"] + " / " + top["reaction_id"]
    fig, axis = plt.subplots(figsize=(9, 6))
    axis.barh(labels, top["actionability_score"], color="#2A788E")
    axis.scatter(top["ensemble_score"], labels, color="#F28E2B", label="Forecast score", zorder=3)
    axis.set_xlabel("Score (0–1)")
    axis.set_title("Top actionable HEA–reaction candidates")
    axis.legend(frameon=False, loc="lower right")
    axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(destination / "actionable_top15.png", dpi=300)
    fig.savefig(destination / "actionable_top15.pdf")
    plt.close(fig)

    summary = {
        "rolling_system_windows_evaluated": int(system["cutoff_year"].nunique()),
        "rolling_pair_windows_evaluated": int(pair["cutoff_year"].nunique()),
        "top_system_model_mean_pr_auc": system_summary.iloc[0]["model"],
        "top_system_mean_pr_auc": float(system_summary.iloc[0]["mean_pr_auc"]),
        "top_pair_model_mean_pr_auc": pair_summary.iloc[0]["model"],
        "top_pair_mean_pr_auc": float(pair_summary.iloc[0]["mean_pr_auc"]),
        "strict_prospective_system_cases": len(strict),
    }
    (destination / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary
