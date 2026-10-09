from __future__ import annotations

import json
import re
from itertools import combinations, product
from pathlib import Path
from typing import Any

import pandas as pd

from hea_forecast.config import load_config

VERIFIED_STATUSES = {"accepted", "corrected", "adjudicated"}
CANDIDATE_COLUMNS = [
    "cutoff_year",
    "horizon_start",
    "horizon_end",
    "system_id",
    "reaction_id",
    "label",
    "first_edge_year",
    "system_first_year",
    "system_age",
    "system_paper_count",
    "system_recency",
    "system_reaction_degree",
    "reaction_system_count",
    "element_count",
    "text_document",
]

REACTION_TEXT_PATTERN = re.compile(
    r"\b(?:HER|OER|ORR|CO2RR|hydrogen evolution(?: reaction)?|"
    r"oxygen evolution(?: reaction)?|oxygen reduction(?: reaction)?|"
    r"carbon dioxide reduction(?: reaction)?|CO2 reduction(?: reaction)?|"
    r"water oxidation|water reduction)\b",
    flags=re.IGNORECASE,
)


def _system_documents(records: pd.DataFrame) -> dict[str, str]:
    """Build cutoff-safe system documents with reaction labels redacted."""
    text_columns = [
        column
        for column in ["display_name", "evidence_context", "abstract"]
        if column in records.columns
    ]
    if not text_columns:
        return {}
    documents: dict[str, str] = {}
    for system_id, group in records.groupby("system_id", sort=False):
        snippets: list[str] = []
        seen: set[str] = set()
        for column in text_columns:
            for value in group[column].fillna("").astype(str):
                cleaned = REACTION_TEXT_PATTERN.sub(" reaction ", value)
                cleaned = " ".join(cleaned.split())
                if cleaned and cleaned not in seen:
                    snippets.append(cleaned)
                    seen.add(cleaned)
        documents[str(system_id)] = " ".join(snippets)
    return documents


def expand_element_pairs(frame: pd.DataFrame) -> pd.DataFrame:
    """Represent every HEA system by its unordered elemental pairs."""
    rows: list[pd.Series] = []
    for _, row in frame.iterrows():
        elements = sorted(set(str(row["system_id"]).split("-")))
        for left, right in combinations(elements, 2):
            expanded = row.copy()
            expanded["system_id"] = f"{left}-{right}"
            rows.append(expanded)
    if not rows:
        return frame.iloc[0:0].copy()
    reaction_column = "reaction_ids" if "reaction_ids" in frame.columns else "reaction_id"
    return pd.DataFrame(rows).drop_duplicates(
        subset=["paper_id", "publication_date", "system_id", reaction_column]
    )


def _prepare_evidence(frame: pd.DataFrame, allow_pending: bool) -> pd.DataFrame:
    data = frame.copy().fillna("")
    required = {"paper_id", "publication_date", "system_id"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing temporal evidence fields: {sorted(missing)}")
    if "reaction_ids" not in data.columns:
        if "reaction_id" not in data.columns:
            raise ValueError("Temporal evidence requires reaction_ids or reaction_id")
        data["reaction_ids"] = data["reaction_id"]
    if "corrected_system_id" in data.columns:
        data["system_id"] = data["corrected_system_id"].where(
            data["corrected_system_id"].ne(""), data["system_id"]
        )
    if "corrected_reaction_id" in data.columns:
        data["reaction_ids"] = data["corrected_reaction_id"].where(
            data["corrected_reaction_id"].ne(""), data["reaction_ids"]
        )
    if not allow_pending:
        status_column = (
            "manual_status"
            if "manual_status" in data.columns
            else "adjudication_status"
            if "adjudication_status" in data.columns
            else None
        )
        if status_column is None:
            raise ValueError("Verified backtests require a manual or adjudication status")
        data = data.loc[data[status_column].isin(VERIFIED_STATUSES)].copy()
    data["publication_year"] = pd.to_datetime(
        data["publication_date"], errors="coerce"
    ).dt.year.astype("Int64")
    data = data.loc[data["publication_year"].notna() & data["system_id"].ne("")].copy()
    return data


def build_candidate_table(
    evidence: pd.DataFrame,
    reactions: list[str],
    cutoff_year: int,
    horizon_start: int,
    horizon_end: int,
    allow_pending: bool = False,
) -> pd.DataFrame:
    """Build an all-candidate warm-start table using only cutoff-safe features."""
    data = _prepare_evidence(evidence, allow_pending)
    if data.empty:
        return pd.DataFrame(columns=CANDIDATE_COLUMNS)

    systems = (
        data.groupby("system_id", as_index=False)
        .agg(
            system_first_year=("publication_year", "min"),
            system_last_year_all=("publication_year", "max"),
        )
        .loc[lambda value: value["system_first_year"] <= cutoff_year]
    )
    observed_systems = systems["system_id"].tolist()

    edges = data.loc[data["reaction_ids"].ne("")].copy()
    edges["reaction_id"] = edges["reaction_ids"].str.split("|")
    edges = edges.explode("reaction_id", ignore_index=True)
    edges = edges.loc[edges["reaction_id"].isin(reactions)]
    first_edges = edges.groupby(["system_id", "reaction_id"], as_index=False).agg(
        first_edge_year=("publication_year", "min"), edge_paper_count=("paper_id", "nunique")
    )

    historical_edges = first_edges.loc[first_edges["first_edge_year"] <= cutoff_year]
    historical_pairs = set(
        zip(historical_edges["system_id"], historical_edges["reaction_id"], strict=False)
    )
    first_edge_lookup = {
        (row.system_id, row.reaction_id): int(row.first_edge_year)
        for row in first_edges.itertuples()
    }

    cutoff_records = data.loc[data["publication_year"] <= cutoff_year]
    document_lookup = _system_documents(cutoff_records)
    system_stats = cutoff_records.groupby("system_id").agg(
        system_paper_count=("paper_id", "nunique"),
        system_last_year=("publication_year", "max"),
    )
    degree_lookup = historical_edges.groupby("system_id")["reaction_id"].nunique().to_dict()
    reaction_popularity = historical_edges.groupby("reaction_id")["system_id"].nunique().to_dict()

    rows: list[dict[str, Any]] = []
    first_year_lookup = systems.set_index("system_id")["system_first_year"].to_dict()
    for system_id, reaction_id in product(observed_systems, reactions):
        if (system_id, reaction_id) in historical_pairs:
            continue
        first_edge_year = first_edge_lookup.get((system_id, reaction_id))
        system_last_year = int(system_stats.loc[system_id, "system_last_year"])
        rows.append(
            {
                "cutoff_year": cutoff_year,
                "horizon_start": horizon_start,
                "horizon_end": horizon_end,
                "system_id": system_id,
                "reaction_id": reaction_id,
                "label": int(
                    first_edge_year is not None and horizon_start <= first_edge_year <= horizon_end
                ),
                "first_edge_year": first_edge_year,
                "system_first_year": int(first_year_lookup[system_id]),
                "system_age": cutoff_year - int(first_year_lookup[system_id]),
                "system_paper_count": int(system_stats.loc[system_id, "system_paper_count"]),
                "system_recency": cutoff_year - system_last_year,
                "system_reaction_degree": int(degree_lookup.get(system_id, 0)),
                "reaction_system_count": int(reaction_popularity.get(reaction_id, 0)),
                "element_count": len(system_id.split("-")),
                "text_document": document_lookup.get(system_id, ""),
            }
        )
    return pd.DataFrame(rows, columns=CANDIDATE_COLUMNS)


def build_historical_training_table(
    evidence: pd.DataFrame,
    reactions: list[str],
    cutoff_year: int,
    allow_pending: bool = False,
) -> pd.DataFrame:
    """Reconstruct cutoff-era links for upstream-compatible supervision.

    Degree and reaction-popularity features remove the target edge for positive
    examples so the label is not directly encoded in its own graph features.
    """
    data = _prepare_evidence(evidence, allow_pending)
    data = data.loc[data["publication_year"] <= cutoff_year].copy()
    if data.empty:
        return pd.DataFrame(columns=CANDIDATE_COLUMNS)

    systems = data.groupby("system_id", as_index=False).agg(
        system_first_year=("publication_year", "min"),
        system_last_year=("publication_year", "max"),
        system_paper_count=("paper_id", "nunique"),
    )
    edges = data.loc[data["reaction_ids"].ne("")].copy()
    edges["reaction_id"] = edges["reaction_ids"].str.split("|")
    edges = edges.explode("reaction_id", ignore_index=True)
    edges = edges.loc[edges["reaction_id"].isin(reactions)]
    first_edges = edges.groupby(["system_id", "reaction_id"], as_index=False).agg(
        first_edge_year=("publication_year", "min")
    )
    existing = set(zip(first_edges["system_id"], first_edges["reaction_id"], strict=False))
    first_edge_lookup = {
        (row.system_id, row.reaction_id): int(row.first_edge_year)
        for row in first_edges.itertuples()
    }
    degree_lookup = first_edges.groupby("system_id")["reaction_id"].nunique().to_dict()
    reaction_popularity = first_edges.groupby("reaction_id")["system_id"].nunique().to_dict()
    system_lookup = systems.set_index("system_id").to_dict("index")
    document_lookup = _system_documents(data)

    rows: list[dict[str, Any]] = []
    for system_id, reaction_id in product(systems["system_id"], reactions):
        label = int((system_id, reaction_id) in existing)
        stats = system_lookup[system_id]
        rows.append(
            {
                "cutoff_year": cutoff_year,
                "horizon_start": cutoff_year,
                "horizon_end": cutoff_year,
                "system_id": system_id,
                "reaction_id": reaction_id,
                "label": label,
                "first_edge_year": first_edge_lookup.get((system_id, reaction_id)),
                "system_first_year": int(stats["system_first_year"]),
                "system_age": cutoff_year - int(stats["system_first_year"]),
                "system_paper_count": int(stats["system_paper_count"]),
                "system_recency": cutoff_year - int(stats["system_last_year"]),
                "system_reaction_degree": max(0, int(degree_lookup.get(system_id, 0)) - label),
                "reaction_system_count": max(
                    0, int(reaction_popularity.get(reaction_id, 0)) - label
                ),
                "element_count": len(system_id.split("-")),
                "text_document": document_lookup.get(system_id, ""),
            }
        )
    return pd.DataFrame(rows, columns=CANDIDATE_COLUMNS)


def build_backtests_from_config(
    input_csv: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
    allow_pending: bool = False,
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
    windows: list[dict[str, Any]] = []
    total_candidates = 0
    total_positives = 0
    for window in config["temporal_backtests"]:
        train_cutoff = window["cutoff_year"] - 3
        train_table = build_candidate_table(
            evidence=evidence,
            reactions=list(config["reactions"]),
            cutoff_year=train_cutoff,
            horizon_start=train_cutoff + 1,
            horizon_end=window["cutoff_year"],
            allow_pending=allow_pending,
        )
        test_table = build_candidate_table(
            evidence=evidence,
            reactions=list(config["reactions"]),
            cutoff_year=window["cutoff_year"],
            horizon_start=window["horizon_start"],
            horizon_end=window["horizon_end"],
            allow_pending=allow_pending,
        )
        stem = f"{entity_level}_backtest_{window['cutoff_year']}_{window['horizon_end']}"
        train_path = destination / f"{stem}.train.csv"
        test_path = destination / f"{stem}.test.csv"
        train_table.to_csv(train_path, index=False)
        test_table.to_csv(test_path, index=False)
        train_positives = int(train_table["label"].sum()) if not train_table.empty else 0
        test_positives = int(test_table["label"].sum()) if not test_table.empty else 0
        total_candidates += len(test_table)
        total_positives += test_positives
        windows.append(
            {
                **window,
                "train_cutoff_year": train_cutoff,
                "train_horizon_start": train_cutoff + 1,
                "train_horizon_end": window["cutoff_year"],
                "train_candidates": len(train_table),
                "train_positives": train_positives,
                "test_candidates": len(test_table),
                "test_positives": test_positives,
                "test_positive_rate": test_positives / len(test_table) if len(test_table) else 0.0,
                "train_path": str(train_path.resolve()),
                "test_path": str(test_path.resolve()),
            }
        )
    summary = {
        "allow_pending": allow_pending,
        "entity_level": entity_level,
        "windows": windows,
        "total_candidates": total_candidates,
        "total_positives": total_positives,
    }
    with (destination / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
    return summary


def build_historical_backtests_from_config(
    input_csv: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
    allow_pending: bool = False,
    entity_level: str = "system",
) -> dict[str, Any]:
    """Build same-cutoff historical supervision and strictly future tests."""
    config = load_config(config_path)
    evidence = pd.read_csv(input_csv, low_memory=False)
    if entity_level == "element_pair":
        evidence = expand_element_pairs(evidence)
    elif entity_level != "system":
        raise ValueError("entity_level must be 'system' or 'element_pair'")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    windows: list[dict[str, Any]] = []
    total_candidates = 0
    total_positives = 0
    for window in config["temporal_backtests"]:
        cutoff = window["cutoff_year"]
        train_table = build_historical_training_table(
            evidence=evidence,
            reactions=list(config["reactions"]),
            cutoff_year=cutoff,
            allow_pending=allow_pending,
        )
        test_table = build_candidate_table(
            evidence=evidence,
            reactions=list(config["reactions"]),
            cutoff_year=cutoff,
            horizon_start=window["horizon_start"],
            horizon_end=window["horizon_end"],
            allow_pending=allow_pending,
        )
        stem = f"{entity_level}_historical_{cutoff}_{window['horizon_end']}"
        train_path = destination / f"{stem}.train.csv"
        test_path = destination / f"{stem}.test.csv"
        train_table.to_csv(train_path, index=False)
        test_table.to_csv(test_path, index=False)
        train_positives = int(train_table["label"].sum()) if not train_table.empty else 0
        test_positives = int(test_table["label"].sum()) if not test_table.empty else 0
        total_candidates += len(test_table)
        total_positives += test_positives
        windows.append(
            {
                **window,
                "training_mode": "historical_link_reconstruction",
                "train_candidates": len(train_table),
                "train_positives": train_positives,
                "test_candidates": len(test_table),
                "test_positives": test_positives,
                "test_positive_rate": test_positives / len(test_table) if len(test_table) else 0.0,
                "train_path": str(train_path.resolve()),
                "test_path": str(test_path.resolve()),
            }
        )
    summary = {
        "allow_pending": allow_pending,
        "entity_level": entity_level,
        "training_mode": "historical_link_reconstruction",
        "windows": windows,
        "total_candidates": total_candidates,
        "total_positives": total_positives,
    }
    with (destination / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
    return summary
