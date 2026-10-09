from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from hea_forecast.config import load_config
from hea_forecast.elemental_features import add_elemental_features
from hea_forecast.modeling import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    _cluster_bootstrap_intervals,
    _metrics,
    _preprocessor,
)
from hea_forecast.temporal import (
    _prepare_evidence,
    build_candidate_table,
    expand_element_pairs,
)


def build_person_period_table(
    evidence: pd.DataFrame,
    reactions: list[str],
    cutoff_year: int,
    allow_pending: bool = False,
) -> pd.DataFrame:
    """Create one-year risk intervals while retaining right-censored candidates."""
    prepared = _prepare_evidence(evidence, allow_pending)
    if prepared.empty:
        return pd.DataFrame()
    first_year = int(prepared["publication_year"].min())
    intervals: list[pd.DataFrame] = []
    for event_year in range(first_year + 1, cutoff_year + 1):
        table = build_candidate_table(
            evidence=evidence,
            reactions=reactions,
            cutoff_year=event_year - 1,
            horizon_start=event_year,
            horizon_end=event_year,
            allow_pending=allow_pending,
        )
        if table.empty:
            continue
        table["risk_year"] = event_year
        table["years_since_corpus_start"] = event_year - first_year
        intervals.append(table)
    return pd.concat(intervals, ignore_index=True) if intervals else pd.DataFrame()


def _harrell_concordance(
    first_edge_year: pd.Series, score: np.ndarray, cutoff_year: int, horizon_end: int
) -> float | None:
    event_year = pd.to_numeric(first_edge_year, errors="coerce").to_numpy(dtype=float)
    observed = np.isfinite(event_year) & (event_year > cutoff_year) & (event_year <= horizon_end)
    comparable = 0
    concordant = 0.0
    for left in range(len(score)):
        if not observed[left]:
            continue
        for right in range(len(score)):
            if left == right:
                continue
            right_later = not observed[right] or event_year[right] > event_year[left]
            if not right_later:
                continue
            comparable += 1
            if score[left] > score[right]:
                concordant += 1
            elif score[left] == score[right]:
                concordant += 0.5
    return float(concordant / comparable) if comparable else None


def _integrated_brier(
    first_edge_year: pd.Series,
    annual_hazard: np.ndarray,
    cutoff_year: int,
    horizon_end: int,
) -> float:
    event_year = pd.to_numeric(first_edge_year, errors="coerce").to_numpy(dtype=float)
    values: list[float] = []
    for step, year in enumerate(range(cutoff_year + 1, horizon_end + 1), start=1):
        observed_by_year = np.isfinite(event_year) & (event_year <= year)
        cumulative_risk = 1 - np.power(1 - annual_hazard, step)
        values.append(float(np.mean((observed_by_year.astype(float) - cumulative_risk) ** 2)))
    return float(np.mean(values))


def evaluate_survival_window(
    evidence: pd.DataFrame,
    reactions: list[str],
    cutoff_year: int,
    horizon_start: int,
    horizon_end: int,
    allow_pending: bool = False,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], dict[str, pd.DataFrame]]:
    train = build_person_period_table(evidence, reactions, cutoff_year, allow_pending)
    test = build_candidate_table(
        evidence,
        reactions,
        cutoff_year,
        horizon_start,
        horizon_end,
        allow_pending,
    )
    if train.empty or test.empty or train["label"].sum() < 5:
        return [], {}
    train = add_elemental_features(train)
    test = add_elemental_features(test)
    chemical_features = [
        column
        for column in train.columns
        if column == "chemical_element_count" or column.startswith(("chem_", "rxn_"))
    ]
    numeric_features = NUMERIC_FEATURES + chemical_features + ["years_since_corpus_start"]
    test["years_since_corpus_start"] = cutoff_year - int(
        _prepare_evidence(evidence, allow_pending)["publication_year"].min()
    )
    features = numeric_features + CATEGORICAL_FEATURES
    y_train = train["label"].astype(int).to_numpy()
    y_test = test["label"].astype(int).to_numpy()
    models = {
        "discrete_hazard_logistic": Pipeline(
            [
                ("features", _preprocessor(numeric_features)),
                (
                    "model",
                    LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed),
                ),
            ]
        ),
        "discrete_hazard_forest": Pipeline(
            [
                ("features", _preprocessor(numeric_features)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=500,
                        min_samples_leaf=3,
                        class_weight="balanced_subsample",
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }
    horizon = horizon_end - cutoff_year
    groups = test["system_id"].astype(str).to_numpy()
    results: list[dict[str, Any]] = []
    predictions: dict[str, pd.DataFrame] = {}
    for model_index, (name, model) in enumerate(models.items()):
        model.fit(train[features], y_train)
        annual_hazard = np.clip(model.predict_proba(test[features])[:, 1], 1e-6, 1 - 1e-6)
        cumulative_risk = 1 - np.power(1 - annual_hazard, horizon)
        results.append(
            {
                "model": name,
                "person_period_rows": len(train),
                "person_period_events": int(y_train.sum()),
                "concordance_index": _harrell_concordance(
                    test["first_edge_year"], cumulative_risk, cutoff_year, horizon_end
                ),
                "integrated_brier_score": _integrated_brier(
                    test["first_edge_year"], annual_hazard, cutoff_year, horizon_end
                ),
                **_metrics(y_test, cumulative_risk),
                **_cluster_bootstrap_intervals(y_test, cumulative_risk, groups, seed + model_index),
            }
        )
        ranked = test.copy()
        ranked["annual_hazard"] = annual_hazard
        ranked["score"] = cumulative_risk
        predictions[name] = ranked.sort_values("score", ascending=False, kind="stable")
    return results, predictions


def evaluate_survival_from_config(
    input_csv: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
    entity_level: str = "system",
) -> dict[str, Any]:
    config = load_config(config_path)
    evidence = pd.read_csv(input_csv, low_memory=False)
    if entity_level == "element_pair":
        evidence = expand_element_pairs(evidence)
    elif entity_level != "system":
        raise ValueError("entity_level must be 'system' or 'element_pair'")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    all_results: list[dict[str, Any]] = []
    skipped: list[int] = []
    for window in config["temporal_backtests"]:
        results, predictions = evaluate_survival_window(
            evidence,
            list(config["reactions"]),
            window["cutoff_year"],
            window["horizon_start"],
            window["horizon_end"],
        )
        if not results:
            skipped.append(window["cutoff_year"])
            continue
        for result in results:
            all_results.append({"cutoff_year": window["cutoff_year"], **result})
        for model_name, ranked in predictions.items():
            ranked.to_csv(
                destination / f"predictions_{window['cutoff_year']}_{model_name}.csv",
                index=False,
            )
    pd.DataFrame(all_results).to_csv(destination / "metrics.csv", index=False)
    summary = {
        "entity_level": entity_level,
        "evaluated_windows": len({row["cutoff_year"] for row in all_results}),
        "skipped_cutoffs": skipped,
    }
    (destination / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary
