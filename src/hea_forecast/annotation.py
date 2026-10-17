from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

VALID_DECISIONS = {"", "yes", "no", "uncertain"}
VALID_STATUSES = {"pending", "accepted", "rejected", "needs_adjudication"}


def create_annotation_batch(
    input_csv: str | Path,
    output_csv: str | Path,
    include_no_reaction: bool = False,
) -> dict[str, Any]:
    frame = pd.read_csv(input_csv, low_memory=False).fillna("")
    required = {"paper_id", "system_id", "reaction_ids", "display_name"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing annotation fields: {sorted(missing)}")

    frame["reaction_id"] = frame["reaction_ids"].map(
        lambda value: str(value).split("|") if value else [""]
    )
    frame = frame.explode("reaction_id", ignore_index=True)
    if not include_no_reaction:
        frame = frame.loc[frame["reaction_id"].ne("")].copy()

    confidence_rank = {"high": 0, "medium": 1, "low": 2}
    frame["_priority"] = frame.get("extraction_confidence", "").map(confidence_rank).fillna(3)
    frame = frame.sort_values(
        ["_priority", "publication_date", "paper_id", "system_id", "reaction_id"],
        kind="stable",
    ).drop(columns=["_priority"])
    frame.insert(0, "annotation_id", [f"A{i:05d}" for i in range(1, len(frame) + 1)])

    reviewer_columns = {
        "reviewer_material_valid": "",
        "reviewer_reaction_valid": "",
        "reviewer_target_catalyst": "",
        "corrected_system_id": "",
        "corrected_reaction_id": "",
        "reviewer_notes": "",
        "adjudication_status": "pending",
    }
    for column, default in reviewer_columns.items():
        frame[column] = default

    preferred = [
        "annotation_id",
        "paper_id",
        "doi",
        "publication_date",
        "display_name",
        "composition_raw",
        "system_id",
        "reaction_id",
        "material_class",
        "extraction_confidence",
        "formula_in_title",
        "reaction_in_title",
        "evidence_context",
        *reviewer_columns,
    ]
    frame = frame[[column for column in preferred if column in frame.columns]]
    destination = Path(output_csv)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False, encoding="utf-8-sig")
    summary = {
        "annotation_rows": len(frame),
        "unique_papers": int(frame["paper_id"].nunique()),
        "unique_systems": int(frame["system_id"].nunique()),
        "reaction_counts": {
            str(reaction): int(count)
            for reaction, count in frame.groupby("reaction_id").size().items()
        },
    }
    with destination.with_suffix(".summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
    return summary


def compile_adjudicated_events(input_csv: str | Path, output_csv: str | Path) -> dict[str, Any]:
    """Validate reviewer decisions and compile earliest accepted discovery events."""
    frame = pd.read_csv(input_csv, dtype=str, keep_default_na=False)
    decision_columns = [
        "reviewer_material_valid",
        "reviewer_reaction_valid",
        "reviewer_target_catalyst",
    ]
    required = {
        "paper_id",
        "publication_date",
        "system_id",
        "reaction_id",
        "adjudication_status",
        "corrected_system_id",
        "corrected_reaction_id",
        *decision_columns,
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing adjudication fields: {sorted(missing)}")

    for column in decision_columns:
        invalid = sorted(set(frame[column].str.lower()) - VALID_DECISIONS)
        if invalid:
            raise ValueError(f"Invalid values in {column}: {invalid}")
        frame[column] = frame[column].str.lower()
    frame["adjudication_status"] = frame["adjudication_status"].str.lower()
    invalid_statuses = sorted(set(frame["adjudication_status"]) - VALID_STATUSES)
    if invalid_statuses:
        raise ValueError(f"Invalid adjudication statuses: {invalid_statuses}")

    accepted = frame["adjudication_status"].eq("accepted")
    all_yes = frame[decision_columns].eq("yes").all(axis=1)
    corrected_reaction = (
        frame["reviewer_material_valid"].eq("yes")
        & frame["reviewer_target_catalyst"].eq("yes")
        & frame["reviewer_reaction_valid"].eq("no")
        & frame["corrected_reaction_id"].ne("")
    )
    inconsistent = accepted & ~(all_yes | corrected_reaction)
    if inconsistent.any():
        ids = frame.loc[inconsistent, "annotation_id"].tolist() if "annotation_id" in frame else []
        raise ValueError(
            f"Accepted rows require three 'yes' decisions or a valid corrected reaction: {ids}"
        )

    accepted_frame = frame.loc[accepted].copy()
    accepted_frame["system_id"] = accepted_frame["corrected_system_id"].where(
        accepted_frame["corrected_system_id"].ne(""), accepted_frame["system_id"]
    )
    accepted_frame["reaction_id"] = accepted_frame["corrected_reaction_id"].where(
        accepted_frame["corrected_reaction_id"].ne(""), accepted_frame["reaction_id"]
    )
    accepted_frame["publication_date"] = pd.to_datetime(
        accepted_frame["publication_date"], errors="coerce"
    )
    if accepted_frame["publication_date"].isna().any():
        raise ValueError("Accepted rows require a valid publication_date")

    accepted_frame = accepted_frame.sort_values(["publication_date", "paper_id"], kind="stable")
    events = accepted_frame.drop_duplicates(["system_id", "reaction_id"], keep="first")
    columns = [
        column
        for column in [
            "paper_id",
            "doi",
            "publication_date",
            "system_id",
            "reaction_id",
            "material_class",
            "annotation_id",
        ]
        if column in events.columns
    ]
    destination = Path(output_csv)
    destination.parent.mkdir(parents=True, exist_ok=True)
    events[columns].to_csv(destination, index=False, encoding="utf-8-sig")
    summary = {
        "annotation_rows": len(frame),
        "accepted_rows": int(accepted.sum()),
        "rejected_rows": int(frame["adjudication_status"].eq("rejected").sum()),
        "needs_adjudication_rows": int(frame["adjudication_status"].eq("needs_adjudication").sum()),
        "pending_rows": int(frame["adjudication_status"].eq("pending").sum()),
        "unique_discovery_events": len(events),
        "output": str(destination.resolve()),
    }
    with destination.with_suffix(".summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
    return summary
