"""Build robust rolling-window summaries and an untuned rank ensemble.

The script uses only frozen prediction files. It does not refit a model, change
the candidate universe, or inspect the locked 2026 outcomes.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "temporal_robustness_upgrade_2026"
ENTITY_DIRS = {
    "Exact element set": ROOT / "results" / "rolling_system",
    "Element pair": ROOT / "results" / "rolling_pair",
}
ENSEMBLE_MEMBERS = (
    "logistic_topology",
    "popularity",
    "random_forest_chemistry",
)
KEYS = ("system_id", "reaction_id")


def load_prediction(entity: str, year: int, model: str) -> pd.DataFrame:
    path = ENTITY_DIRS[entity] / f"predictions_{year}_{model}.csv"
    frame = pd.read_csv(path)
    required = {*KEYS, "label", "score"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} lacks columns: {sorted(missing)}")
    if frame.duplicated(list(KEYS)).any():
        raise ValueError(f"Duplicate candidate keys in {path}")
    return frame[[*KEYS, "label", "score"]].copy()


def metric_row(entity: str, year: int, model: str, frame: pd.DataFrame) -> dict:
    labels = frame["label"].astype(int).to_numpy()
    scores = frame["score"].astype(float).to_numpy()
    prevalence = float(labels.mean())
    ap = float(average_precision_score(labels, scores))
    return {
        "entity_level": entity,
        "cutoff_year": int(year),
        "model": model,
        "candidates": int(len(frame)),
        "positives": int(labels.sum()),
        "prevalence": prevalence,
        "pr_auc": ap,
        "prevalence_normalized_lift": ap / prevalence,
    }


def summarize_windows(rows: pd.DataFrame) -> pd.DataFrame:
    records: list[dict] = []
    for (entity, model), group in rows.groupby(["entity_level", "model"], sort=False):
        values = group.sort_values("cutoff_year")["pr_auc"].to_numpy(dtype=float)
        lifts = group["prevalence_normalized_lift"].to_numpy(dtype=float)
        loo = [float(np.mean(np.delete(values, i))) for i in range(len(values))]
        records.append(
            {
                "entity_level": entity,
                "model": model,
                "windows": len(values),
                "mean_pr_auc": float(np.mean(values)),
                "sample_sd_pr_auc": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "median_pr_auc": float(np.median(values)),
                "q1_pr_auc": float(np.quantile(values, 0.25)),
                "q3_pr_auc": float(np.quantile(values, 0.75)),
                "iqr_pr_auc": float(np.quantile(values, 0.75) - np.quantile(values, 0.25)),
                "geometric_mean_lift": float(math.exp(np.mean(np.log(lifts)))),
                "windows_above_prevalence": int((group["pr_auc"] > group["prevalence"]).sum()),
                "loo_mean_min": min(loo),
                "loo_mean_max": max(loo),
            }
        )
    return pd.DataFrame(records)


def build_ensemble(entity: str, years: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    window_rows: list[dict] = []
    pooled_rows: list[pd.DataFrame] = []
    for year in years:
        merged: pd.DataFrame | None = None
        for model in ENSEMBLE_MEMBERS:
            frame = load_prediction(entity, year, model).rename(columns={"score": f"score_{model}"})
            if merged is None:
                merged = frame
            else:
                merged = merged.merge(
                    frame.drop(columns="label"), on=list(KEYS), how="inner", validate="one_to_one"
                )
                original = load_prediction(entity, year, model)[[*KEYS, "label"]]
                check = merged[[*KEYS, "label"]].merge(
                    original, on=list(KEYS), suffixes=("_base", "_new"), validate="one_to_one"
                )
                if not (check["label_base"] == check["label_new"]).all():
                    raise ValueError(f"Label disagreement for {entity}, {year}, {model}")
        assert merged is not None
        rank_cols = []
        for model in ENSEMBLE_MEMBERS:
            name = f"rank_{model}"
            merged[name] = merged[f"score_{model}"].rank(method="average", pct=True)
            rank_cols.append(name)
        merged["score"] = merged[rank_cols].mean(axis=1)
        row = metric_row(entity, year, "fixed_equal_rank_ensemble", merged)
        window_rows.append(row)
        merged["entity_level"] = entity
        merged["cutoff_year"] = year
        pooled_rows.append(merged[["entity_level", "cutoff_year", *KEYS, "label", "score"]])
    return pd.DataFrame(window_rows), pd.concat(pooled_rows, ignore_index=True)


def pooled_metrics(entity: str, model: str, frames: list[pd.DataFrame]) -> dict:
    pooled = pd.concat(frames, ignore_index=True)
    pooled["within_window_rank"] = pooled.groupby("cutoff_year")["score"].rank(
        method="average", pct=True
    )
    labels = pooled["label"].astype(int).to_numpy()
    prevalence = float(labels.mean())
    raw_ap = float(average_precision_score(labels, pooled["score"]))
    rank_ap = float(average_precision_score(labels, pooled["within_window_rank"]))
    return {
        "entity_level": entity,
        "model": model,
        "windows": int(pooled["cutoff_year"].nunique()),
        "candidates": int(len(pooled)),
        "positives": int(labels.sum()),
        "prevalence": prevalence,
        "raw_pooled_pr_auc": raw_ap,
        "rank_normalized_pooled_pr_auc": rank_ap,
        "rank_normalized_pooled_lift": rank_ap / prevalence,
    }


def create_figure(all_windows: pd.DataFrame, path: Path) -> None:
    selected = all_windows[all_windows["model"].isin(["logistic_topology", "fixed_equal_rank_ensemble"])]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.1), sharey=False)
    palette = {"logistic_topology": "#32658f", "fixed_equal_rank_ensemble": "#c46b33"}
    labels = {"logistic_topology": "Topology LR", "fixed_equal_rank_ensemble": "Fixed rank ensemble"}
    for ax, entity in zip(axes, ENTITY_DIRS):
        part = selected[selected["entity_level"] == entity]
        for model in ("logistic_topology", "fixed_equal_rank_ensemble"):
            series = part[part["model"] == model].sort_values("cutoff_year")
            ax.plot(
                series["cutoff_year"], series["pr_auc"], marker="o", linewidth=1.8,
                markersize=5, color=palette[model], label=labels[model]
            )
        prevalence = part[part["model"] == "logistic_topology"].sort_values("cutoff_year")
        ax.plot(
            prevalence["cutoff_year"], prevalence["prevalence"], color="#777777",
            linewidth=1.2, linestyle="--", label="Window prevalence"
        )
        ax.set_title(entity)
        ax.set_xlabel("Cutoff year")
        ax.set_ylabel("PR-AUC")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#d8d8d8", linewidth=0.6)
        ax.set_xticks(sorted(part["cutoff_year"].unique()))
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=450, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    metric_frames = {entity: pd.read_csv(directory / "metrics.csv") for entity, directory in ENTITY_DIRS.items()}
    years_by_entity: dict[str, list[int]] = {}
    baseline_rows: list[dict] = []
    ensemble_windows: list[pd.DataFrame] = []
    ensemble_pooled: dict[str, pd.DataFrame] = {}
    pooled_records: list[dict] = []

    for entity, metrics in metric_frames.items():
        years = sorted(metrics.loc[metrics["model"] == "logistic_topology", "cutoff_year"].astype(int).unique())
        years_by_entity[entity] = years
        for model in ("logistic_topology", "popularity", "random_forest_chemistry"):
            frames = []
            for year in years:
                frame = load_prediction(entity, year, model)
                baseline_rows.append(metric_row(entity, year, model, frame))
                pooled = frame.copy()
                pooled["cutoff_year"] = year
                frames.append(pooled)
            pooled_records.append(pooled_metrics(entity, model, frames))
        window_table, pooled = build_ensemble(entity, years)
        ensemble_windows.append(window_table)
        ensemble_pooled[entity] = pooled
        pooled_records.append(
            pooled_metrics(
                entity,
                "fixed_equal_rank_ensemble",
                [group.copy() for _, group in pooled.groupby("cutoff_year")],
            )
        )

    windows = pd.concat([pd.DataFrame(baseline_rows), *ensemble_windows], ignore_index=True)
    popularity = windows[windows["model"] == "popularity"][
        ["entity_level", "cutoff_year", "pr_auc"]
    ].rename(columns={"pr_auc": "popularity_pr_auc"})
    windows = windows.merge(popularity, on=["entity_level", "cutoff_year"], how="left", validate="many_to_one")
    windows["beats_popularity"] = windows["pr_auc"] > windows["popularity_pr_auc"]
    windows["beats_prevalence"] = windows["pr_auc"] > windows["prevalence"]
    windows.to_csv(OUT / "window_robustness_metrics.csv", index=False)

    summary = summarize_windows(windows)
    win_counts = windows.groupby(["entity_level", "model"], as_index=False).agg(
        windows_beating_popularity=("beats_popularity", "sum"),
        total_windows=("cutoff_year", "count"),
    )
    summary = summary.merge(win_counts, on=["entity_level", "model"], how="left", validate="one_to_one")
    summary.to_csv(OUT / "model_robust_summary.csv", index=False)
    pd.DataFrame(pooled_records).to_csv(OUT / "pooled_out_of_time_metrics.csv", index=False)
    pd.concat(ensemble_windows, ignore_index=True).to_csv(
        OUT / "fixed_equal_rank_ensemble_window_metrics.csv", index=False
    )

    create_figure(windows, OUT / "figure_temporal_robustness")

    focus = summary[summary["model"].isin(["logistic_topology", "fixed_equal_rank_ensemble"])]
    payload = {
        "analysis_status": "completed_from_frozen_rolling_predictions",
        "ensemble_status": "exploratory_untuned_secondary_analysis",
        "ensemble_members": list(ENSEMBLE_MEMBERS),
        "years_by_entity": years_by_entity,
        "focus_summary": focus.to_dict(orient="records"),
        "pooled_summary": [r for r in pooled_records if r["model"] in {"logistic_topology", "fixed_equal_rank_ensemble"}],
    }
    (OUT / "summary.json").write_text(
        json.dumps(payload, indent=2, default=lambda value: value.item()), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
