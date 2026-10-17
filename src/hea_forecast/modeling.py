from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    ndcg_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from hea_forecast.elemental_features import add_elemental_features

NUMERIC_FEATURES = [
    "system_age",
    "system_paper_count",
    "system_recency",
    "system_reaction_degree",
    "reaction_system_count",
    "element_count",
]
CATEGORICAL_FEATURES = ["reaction_id"]


def _expected_calibration_error(
    y_true: np.ndarray, probability: np.ndarray, bins: int = 10
) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for left, right in pairwise(edges):
        include_right = right == 1.0
        mask = (probability >= left) & (
            (probability <= right) if include_right else (probability < right)
        )
        if mask.any():
            error += mask.mean() * abs(float(y_true[mask].mean()) - float(probability[mask].mean()))
    return float(error if len(y_true) else np.nan)


def _metrics(y_true: np.ndarray, score: np.ndarray) -> dict[str, float | None]:
    probability = np.clip(np.asarray(score, dtype=float), 1e-6, 1 - 1e-6)
    order = np.argsort(-score, kind="stable")
    ranked = y_true[order]
    prevalence = float(y_true.mean())
    result: dict[str, float | None] = {
        "roc_auc": float(roc_auc_score(y_true, score)) if len(np.unique(y_true)) == 2 else None,
        "pr_auc": float(average_precision_score(y_true, score)) if y_true.sum() > 0 else None,
        "ndcg": float(ndcg_score(y_true.reshape(1, -1), score.reshape(1, -1)))
        if y_true.sum() > 0
        else None,
        "mrr": None,
        "prevalence": prevalence,
        "brier_score": float(brier_score_loss(y_true, probability)),
        "log_loss": float(log_loss(y_true, probability, labels=[0, 1])),
        "expected_calibration_error": _expected_calibration_error(y_true, probability),
    }
    positive_ranks = np.flatnonzero(ranked == 1)
    if positive_ranks.size:
        result["mrr"] = float(1.0 / (positive_ranks[0] + 1))
    for k in (10, 20, 50, 100):
        actual_k = min(k, len(ranked))
        top = ranked[:actual_k]
        precision = float(top.mean()) if actual_k else 0.0
        result[f"precision_at_{k}"] = precision
        result[f"recall_at_{k}"] = float(top.sum() / y_true.sum()) if y_true.sum() else 0.0
        result[f"enrichment_at_{k}"] = precision / prevalence if prevalence else None
    return result


def _cluster_bootstrap_intervals(
    y_true: np.ndarray,
    score: np.ndarray,
    groups: np.ndarray,
    seed: int,
    repetitions: int = 500,
) -> dict[str, float | None]:
    """Bootstrap by material entity to preserve within-system dependence."""
    unique_groups = np.unique(groups)
    empty_result = {
        "pr_auc_ci_low": None,
        "pr_auc_ci_high": None,
        "precision_at_20_ci_low": None,
        "precision_at_20_ci_high": None,
    }
    if unique_groups.size < 2:
        return empty_result
    group_rows = {group: np.flatnonzero(groups == group) for group in unique_groups}
    rng = np.random.default_rng(seed)
    pr_values: list[float] = []
    p20_values: list[float] = []
    for _ in range(repetitions):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        sampled_rows = np.concatenate([group_rows[group] for group in sampled_groups])
        sampled_y = y_true[sampled_rows]
        sampled_score = score[sampled_rows]
        if sampled_y.sum() == 0:
            continue
        metrics = _metrics(sampled_y, sampled_score)
        if metrics["pr_auc"] is not None:
            pr_values.append(float(metrics["pr_auc"]))
        p20_values.append(float(metrics["precision_at_20"]))

    def interval(values: list[float]) -> tuple[float | None, float | None]:
        if len(values) < max(20, repetitions // 5):
            return None, None
        low, high = np.quantile(values, [0.025, 0.975])
        return float(low), float(high)

    pr_low, pr_high = interval(pr_values)
    p20_low, p20_high = interval(p20_values)
    return {
        "pr_auc_ci_low": pr_low,
        "pr_auc_ci_high": pr_high,
        "precision_at_20_ci_low": p20_low,
        "precision_at_20_ci_high": p20_high,
    }


def _preprocessor(numeric_features: list[str]) -> ColumnTransformer:
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical = OneHotEncoder(handle_unknown="ignore")
    return ColumnTransformer(
        [("numeric", numeric, numeric_features), ("reaction", categorical, CATEGORICAL_FEATURES)]
    )


def _text_preprocessor(numeric_features: list[str] | None = None) -> ColumnTransformer:
    transformers: list[tuple[str, Any, Any]] = [
        (
            "text",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                min_df=2,
                max_features=3000,
                sublinear_tf=True,
            ),
            "text_document",
        ),
        ("reaction", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ]
    if numeric_features:
        numeric = Pipeline(
            [("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
        )
        transformers.insert(0, ("numeric", numeric, numeric_features))
    return ColumnTransformer(transformers)


def _model_specs(
    chemical_features: list[str], seed: int = 42, include_text: bool = True
) -> dict[str, tuple[Any, list[str]]]:
    hybrid_features = NUMERIC_FEATURES + chemical_features
    models: dict[str, tuple[Any, list[str]]] = {
        "logistic_topology": (
            Pipeline(
                [
                    ("features", _preprocessor(NUMERIC_FEATURES)),
                    (
                        "model",
                        LogisticRegression(
                            class_weight="balanced", max_iter=2000, random_state=seed
                        ),
                    ),
                ]
            ),
            NUMERIC_FEATURES,
        ),
        "random_forest_topology": (
            Pipeline(
                [
                    ("features", _preprocessor(NUMERIC_FEATURES)),
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=500,
                            min_samples_leaf=2,
                            class_weight="balanced_subsample",
                            random_state=seed,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
            NUMERIC_FEATURES,
        ),
        "random_forest_chemistry": (
            Pipeline(
                [
                    ("features", _preprocessor(chemical_features)),
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=500,
                            min_samples_leaf=2,
                            class_weight="balanced_subsample",
                            random_state=seed,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
            chemical_features,
        ),
        "random_forest_hybrid": (
            Pipeline(
                [
                    ("features", _preprocessor(hybrid_features)),
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=500,
                            min_samples_leaf=2,
                            class_weight="balanced_subsample",
                            random_state=seed,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
            hybrid_features,
        ),
    }
    if not include_text:
        return models
    models["logistic_text"] = (
        Pipeline(
            [
                ("features", _text_preprocessor()),
                (
                    "model",
                    LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed),
                ),
            ]
        ),
        [],
    )
    models["logistic_text_hybrid"] = (
        Pipeline(
            [
                ("features", _text_preprocessor(hybrid_features)),
                (
                    "model",
                    LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed),
                ),
            ]
        ),
        hybrid_features,
    )
    return models


def evaluate_window(
    train: pd.DataFrame, test: pd.DataFrame, seed: int = 42
) -> tuple[list[dict[str, Any]], dict[str, pd.DataFrame]]:
    train = add_elemental_features(train)
    test = add_elemental_features(test)
    chemical_features = [
        column
        for column in train.columns
        if column == "chemical_element_count" or column.startswith("chem_")
    ]
    y_train = train["label"].astype(int).to_numpy()
    y_test = test["label"].astype(int).to_numpy()
    results: list[dict[str, Any]] = []
    predictions: dict[str, pd.DataFrame] = {}
    groups = test["system_id"].astype(str).to_numpy()

    train_popularity = (train["system_paper_count"].astype(float) + 1) * (
        train["reaction_system_count"].astype(float) + 1
    )
    test_popularity = (test["system_paper_count"].astype(float) + 1) * (
        test["reaction_system_count"].astype(float) + 1
    )
    popularity_calibrator = LogisticRegression(class_weight="balanced", random_state=seed)
    popularity_calibrator.fit(np.log1p(train_popularity).to_numpy().reshape(-1, 1), y_train)
    popularity = popularity_calibrator.predict_proba(
        np.log1p(test_popularity).to_numpy().reshape(-1, 1)
    )[:, 1]
    random_scores = np.random.default_rng(seed).random(len(test))
    baseline_scores = {"random": random_scores, "popularity": popularity}

    for model_index, (name, scores) in enumerate(baseline_scores.items()):
        results.append(
            {
                "model": name,
                **_metrics(y_test, scores),
                **_cluster_bootstrap_intervals(y_test, scores, groups, seed=seed + model_index),
            }
        )
        ranked = test.copy()
        ranked["score"] = scores
        predictions[name] = ranked.sort_values("score", ascending=False, kind="stable")

    include_text = (
        "text_document" in train.columns and train["text_document"].fillna("").ne("").any()
    )
    for model_index, (name, (model, numeric_features)) in enumerate(
        _model_specs(chemical_features, seed, include_text).items(), start=len(baseline_scores)
    ):
        features = numeric_features + CATEGORICAL_FEATURES
        if name.startswith("logistic_text"):
            features += ["text_document"]
        model.fit(train[features], y_train)
        scores = model.predict_proba(test[features])[:, 1]
        results.append(
            {
                "model": name,
                **_metrics(y_test, scores),
                **_cluster_bootstrap_intervals(y_test, scores, groups, seed=seed + model_index),
            }
        )
        ranked = test.copy()
        ranked["score"] = scores
        predictions[name] = ranked.sort_values("score", ascending=False, kind="stable")

    if include_text:
        lsa_model = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        lowercase=True,
                        ngram_range=(1, 2),
                        min_df=2,
                        max_features=3000,
                        sublinear_tf=True,
                    ),
                ),
                ("embedding", TruncatedSVD(n_components=16, random_state=seed)),
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed),
                ),
            ]
        )
        train_documents = (
            train["text_document"].fillna("").astype(str)
            + " TARGET_REACTION_"
            + train["reaction_id"].astype(str)
        )
        test_documents = (
            test["text_document"].fillna("").astype(str)
            + " TARGET_REACTION_"
            + test["reaction_id"].astype(str)
        )
        lsa_model.fit(train_documents, y_train)
        scores = lsa_model.predict_proba(test_documents)[:, 1]
        results.append(
            {
                "model": "lsa_text_embedding",
                **_metrics(y_test, scores),
                **_cluster_bootstrap_intervals(y_test, scores, groups, seed=seed + len(results)),
            }
        )
        ranked = test.copy()
        ranked["score"] = scores
        predictions["lsa_text_embedding"] = ranked.sort_values(
            "score", ascending=False, kind="stable"
        )

    pu_features = NUMERIC_FEATURES + chemical_features + CATEGORICAL_FEATURES
    pu_model = Pipeline(
        [
            ("features", _preprocessor(NUMERIC_FEATURES + chemical_features)),
            (
                "model",
                LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed),
            ),
        ]
    )
    positive_count = int(y_train.sum())
    negative_count = int((1 - y_train).sum())
    folds = min(5, positive_count, negative_count)
    if folds >= 2:
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        observed_probability = cross_val_predict(
            pu_model,
            train[pu_features],
            y_train,
            cv=cv,
            method="predict_proba",
            n_jobs=1,
        )[:, 1]
        propensity = float(np.clip(observed_probability[y_train == 1].mean(), 0.05, 1.0))
        pu_model.fit(train[pu_features], y_train)
        scores = np.clip(pu_model.predict_proba(test[pu_features])[:, 1] / propensity, 0, 1)
        results.append(
            {
                "model": "pu_logistic_hybrid",
                "pu_label_propensity": propensity,
                **_metrics(y_test, scores),
                **_cluster_bootstrap_intervals(y_test, scores, groups, seed=seed + len(results)),
            }
        )
        ranked = test.copy()
        ranked["score"] = scores
        predictions["pu_logistic_hybrid"] = ranked.sort_values(
            "score", ascending=False, kind="stable"
        )
    return results, predictions


def evaluate_backtest_directory(
    backtest_dir: str | Path,
    output_dir: str | Path,
    minimum_train_positives: int = 5,
) -> dict[str, Any]:
    source = Path(backtest_dir)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    all_results: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for window in metadata["windows"]:
        train = pd.read_csv(window["train_path"])
        test = pd.read_csv(window["test_path"])
        reason: str | None = None
        if train.empty or test.empty:
            reason = "empty_train_or_test"
        elif train["label"].sum() < minimum_train_positives:
            reason = "insufficient_train_positives"
        elif train["label"].nunique() < 2:
            reason = "single_class_train"
        elif test["label"].sum() < 1:
            reason = "no_test_positives"
        if reason:
            skipped.append({"cutoff_year": window["cutoff_year"], "reason": reason})
            continue

        results, predictions = evaluate_window(train, test)
        for result in results:
            all_results.append({"cutoff_year": window["cutoff_year"], **result})
        for model_name, ranked in predictions.items():
            ranked.to_csv(
                destination / f"predictions_{window['cutoff_year']}_{model_name}.csv", index=False
            )

    results_frame = pd.DataFrame(all_results)
    results_frame.to_csv(destination / "metrics.csv", index=False)
    summary = {
        "entity_level": metadata.get("entity_level", "system"),
        "evaluated_windows": len({result["cutoff_year"] for result in all_results}),
        "skipped_windows": len(skipped),
        "skipped": skipped,
        "metrics_path": str((destination / "metrics.csv").resolve()),
    }
    with (destination / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
    return summary
