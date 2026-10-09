from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd

from hea_forecast.openalex import canonical_doi


def title_fingerprint(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return re.sub(r"[^a-z0-9]+", "", normalized)


def deduplicate_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Group DOI duplicates and equal-title versions while retaining provenance."""
    data = frame.copy()
    title_column = "display_name" if "display_name" in data.columns else "title"
    if title_column not in data.columns:
        raise ValueError("Input must contain display_name or title")
    data["_canonical_doi"] = data.get("doi", "").fillna("").map(canonical_doi)
    data["_title_fingerprint"] = data[title_column].fillna("").map(title_fingerprint)
    data["_date"] = pd.to_datetime(data.get("publication_date"), errors="coerce")
    data = data.sort_values(["_date", "id"], na_position="last", kind="stable")

    # Equal normalized titles connect preprint/journal records even when their DOIs differ.
    data["version_group_id"] = data["_title_fingerprint"]
    empty_title = data["version_group_id"].eq("")
    data.loc[empty_title, "version_group_id"] = data.loc[empty_title, "id"].astype(str)

    output_rows: list[pd.Series] = []
    for _, group in data.groupby("version_group_id", sort=False, dropna=False):
        representative = group.iloc[0].copy()
        representative["publication_date"] = (
            group["_date"].min().date().isoformat() if group["_date"].notna().any() else ""
        )
        representative["version_ids"] = json.dumps(
            sorted(set(group["id"].dropna().astype(str))), ensure_ascii=False
        )
        representative["version_dois"] = json.dumps(
            sorted({doi for doi in group["_canonical_doi"] if doi}), ensure_ascii=False
        )
        representative["version_count"] = len(group)
        output_rows.append(representative)

    output = pd.DataFrame(output_rows).drop(
        columns=["_canonical_doi", "_title_fingerprint", "_date"], errors="ignore"
    )
    return output.reset_index(drop=True)


def deduplicate_corpus(input_csv: str | Path, output_csv: str | Path) -> dict[str, Any]:
    source = pd.read_csv(input_csv, low_memory=False)
    output = deduplicate_frame(source)
    destination = Path(output_csv)
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(destination, index=False)
    summary = {
        "input_records": len(source),
        "deduplicated_records": len(output),
        "merged_records": len(source) - len(output),
        "multi_version_groups": int(output["version_count"].gt(1).sum()),
    }
    with destination.with_suffix(".summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
    return summary
