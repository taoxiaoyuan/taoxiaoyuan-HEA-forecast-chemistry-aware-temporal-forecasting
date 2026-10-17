from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from hea_forecast.config import load_config

TEXT_COLUMNS = ("display_name", "title", "abstract", "concepts", "llama_concepts")


def _joined_text(frame: pd.DataFrame) -> pd.Series:
    available = [name for name in TEXT_COLUMNS if name in frame.columns]
    if not available:
        raise ValueError(f"None of the expected text columns are present: {TEXT_COLUMNS}")
    return frame[available].fillna("").astype(str).agg(" ".join, axis=1).str.lower()


def _literal_pattern(terms: list[str]) -> str:
    ordered = sorted(
        {term.lower().strip() for term in terms if term.strip()}, key=len, reverse=True
    )
    return "|".join(rf"(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9])" for term in ordered)


def _matched_terms(text: str, terms: list[str]) -> list[str]:
    return [
        term for term in terms if re.search(_literal_pattern([term]), text, flags=re.IGNORECASE)
    ]


def audit_corpus(
    input_csv: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Measure whether a literature table can support the configured HEA study."""
    source = Path(input_csv)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    config = load_config(config_path)
    frame = pd.read_csv(source, low_memory=False)
    text = _joined_text(frame)

    aliases = list(config["material_aliases"])
    reaction_terms = {
        reaction: [reaction, *synonyms] for reaction, synonyms in config["reactions"].items()
    }
    hea_pattern = _literal_pattern(aliases)
    frame["matched_hea_aliases"] = [
        json.dumps(_matched_terms(value, aliases), ensure_ascii=False) for value in text
    ]
    frame["is_hea_candidate"] = text.str.contains(hea_pattern, regex=True, na=False)

    reaction_flags: list[str] = []
    matched_reaction_values: list[str] = []
    for value in text:
        matches = [
            reaction
            for reaction, terms in reaction_terms.items()
            if re.search(_literal_pattern(terms), value, flags=re.IGNORECASE)
        ]
        reaction_flags.append("|".join(matches))
        matched_reaction_values.append(json.dumps(matches, ensure_ascii=False))
    frame["reaction_ids"] = reaction_flags
    frame["matched_reactions"] = matched_reaction_values
    frame["is_hea_reaction_candidate"] = frame["is_hea_candidate"] & frame["reaction_ids"].ne("")

    date_column = "publication_date" if "publication_date" in frame.columns else None
    if date_column:
        dates = pd.to_datetime(frame[date_column], errors="coerce")
        frame["publication_year"] = dates.dt.year.astype("Int64")
    elif "publication_year" not in frame.columns:
        frame["publication_year"] = pd.Series(pd.NA, index=frame.index, dtype="Int64")

    candidates = frame.loc[frame["is_hea_candidate"]].copy()
    candidates.to_csv(destination / "hea_candidates.csv", index=False)

    reaction_counts = {
        reaction: int(
            (
                frame["is_hea_candidate"]
                & frame["reaction_ids"]
                .str.split("|")
                .apply(lambda values, reaction=reaction: reaction in values)
            ).sum()
        )
        for reaction in config["reactions"]
    }
    yearly = (
        frame.loc[frame["is_hea_candidate"]]
        .groupby("publication_year", dropna=False)
        .size()
        .sort_index()
    )
    summary: dict[str, Any] = {
        "source": str(source.resolve()),
        "records": len(frame),
        "hea_candidates": int(frame["is_hea_candidate"].sum()),
        "hea_reaction_candidates": int(frame["is_hea_reaction_candidate"].sum()),
        "reaction_candidate_counts": reaction_counts,
        "publication_year_min": int(frame["publication_year"].min())
        if frame["publication_year"].notna().any()
        else None,
        "publication_year_max": int(frame["publication_year"].max())
        if frame["publication_year"].notna().any()
        else None,
        "hea_candidates_by_year": {
            str(int(year)): int(count) for year, count in yearly.items() if pd.notna(year)
        },
        "is_demo_sufficient": bool(
            frame["is_hea_reaction_candidate"].sum()
            >= config["quality_gates"]["minimum_valid_records"]
        ),
    }
    with (destination / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
    return summary
