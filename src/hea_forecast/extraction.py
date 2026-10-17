from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from hea_forecast.audit import TEXT_COLUMNS, _literal_pattern
from hea_forecast.chemistry import extract_formula_candidates
from hea_forecast.config import load_config


def _material_class(text: str) -> str:
    lowered = text.lower()
    class_terms = {
        "high_entropy_oxide": ("high entropy oxide", "high-entropy oxide"),
        "high_entropy_sulfide": ("high entropy sulfide", "high-entropy sulfide"),
        "high_entropy_phosphide": ("high entropy phosphide", "high-entropy phosphide"),
        "high_entropy_hydroxide": ("high entropy hydroxide", "high-entropy hydroxide"),
    }
    for material_class, terms in class_terms.items():
        if any(term in lowered for term in terms):
            return material_class
    return "metallic_hea"


def _reaction_ids(text: str, reactions: dict[str, list[str]]) -> list[str]:
    return [
        reaction
        for reaction, synonyms in reactions.items()
        if re.search(_literal_pattern([reaction, *synonyms]), text, flags=re.IGNORECASE)
    ]


def _evidence_context(text: str, mention: str, radius: int = 240) -> str:
    index = text.casefold().find(mention.strip("()").casefold())
    if index < 0:
        return text[: radius * 2].strip()
    start = max(0, index - radius)
    end = min(len(text), index + len(mention) + radius)
    return re.sub(r"\s+", " ", text[start:end]).strip()


def extract_dataset(
    input_csv: str | Path,
    config_path: str | Path,
    output_csv: str | Path,
) -> dict[str, Any]:
    """Extract candidate formula-reaction evidence from an already retrieved corpus."""
    config = load_config(config_path)
    frame = pd.read_csv(input_csv, low_memory=False)
    available_text = [column for column in TEXT_COLUMNS if column in frame.columns]
    if not available_text:
        raise ValueError("Input has no supported title, abstract, or concept column")
    joined = frame[available_text].fillna("").astype(str).agg(" ".join, axis=1)
    entity_columns = [
        column for column in ("display_name", "title", "abstract") if column in frame.columns
    ]
    entity_text = frame[entity_columns].fillna("").astype(str).agg(" ".join, axis=1)
    aliases = config["material_aliases"]
    hea_mask = joined.str.contains(_literal_pattern(aliases), case=False, regex=True, na=False)
    minimum_metals = config["inclusion"]["minimum_metal_elements"]

    evidence: list[dict[str, Any]] = []
    for row_index in frame.index[hea_mask]:
        text = joined.loc[row_index]
        evidence_text = entity_text.loc[row_index]
        candidates = extract_formula_candidates(
            evidence_text, minimum_metal_elements=minimum_metals
        )
        reactions = _reaction_ids(text, config["reactions"])
        base = frame.loc[row_index]
        title = str(base.get("display_name", base.get("title", "")) or "")
        title_reactions = _reaction_ids(title, config["reactions"])
        for candidate in candidates:
            formula_in_title = candidate.raw.strip("()").casefold() in title.casefold()
            reaction_in_title = bool(set(reactions) & set(title_reactions))
            auto_confidence = (
                "high"
                if formula_in_title and reaction_in_title
                else "medium"
                if reactions
                else "low"
            )
            evidence.append(
                {
                    "paper_id": base.get("id", ""),
                    "doi": base.get("doi", ""),
                    "publication_date": base.get("publication_date", ""),
                    "display_name": base.get("display_name", base.get("title", "")),
                    "composition_raw": candidate.raw,
                    "system_id": candidate.system_id,
                    "elements": json.dumps(candidate.elements),
                    "metal_count": candidate.metal_count,
                    "material_class": _material_class(text),
                    "reaction_ids": "|".join(reactions),
                    "formula_in_title": formula_in_title,
                    "reaction_in_title": reaction_in_title,
                    "evidence_context": _evidence_context(evidence_text, candidate.raw),
                    "catalyst_role": "pending",
                    "extraction_confidence": auto_confidence,
                    "manual_status": "pending",
                }
            )

    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    evidence_frame = pd.DataFrame(evidence)
    evidence_frame.to_csv(output, index=False)
    summary = {
        "input_records": len(frame),
        "hea_records": int(hea_mask.sum()),
        "evidence_rows": len(evidence_frame),
        "unique_systems": int(evidence_frame["system_id"].nunique())
        if not evidence_frame.empty
        else 0,
        "records_with_reaction": int(evidence_frame["reaction_ids"].ne("").sum())
        if not evidence_frame.empty
        else 0,
    }
    with output.with_suffix(".summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
    return summary
