from pathlib import Path

import pandas as pd
import pytest

from hea_forecast.annotation import compile_adjudicated_events


def _row(paper: str, date: str, status: str = "accepted") -> dict[str, str]:
    return {
        "annotation_id": paper,
        "paper_id": paper,
        "doi": "",
        "publication_date": date,
        "system_id": "Co-Fe-Ni-Pt",
        "reaction_id": "HER",
        "material_class": "metallic_hea",
        "reviewer_material_valid": "yes",
        "reviewer_reaction_valid": "yes",
        "reviewer_target_catalyst": "yes",
        "corrected_system_id": "",
        "corrected_reaction_id": "",
        "adjudication_status": status,
    }


def test_compile_keeps_earliest_event(tmp_path: Path) -> None:
    source = tmp_path / "annotations.csv"
    output = tmp_path / "events.csv"
    pd.DataFrame([_row("later", "2021-01-01"), _row("early", "2020-01-01")]).to_csv(
        source, index=False
    )
    summary = compile_adjudicated_events(source, output)
    events = pd.read_csv(output)
    assert summary["unique_discovery_events"] == 1
    assert events.loc[0, "paper_id"] == "early"


def test_compile_rejects_inconsistent_acceptance(tmp_path: Path) -> None:
    source = tmp_path / "annotations.csv"
    output = tmp_path / "events.csv"
    row = _row("bad", "2020-01-01")
    row["reviewer_material_valid"] = "no"
    pd.DataFrame([row]).to_csv(source, index=False)
    with pytest.raises(ValueError, match="three 'yes'"):
        compile_adjudicated_events(source, output)


def test_compile_accepts_reviewed_reaction_correction(tmp_path: Path) -> None:
    source = tmp_path / "annotations.csv"
    output = tmp_path / "events.csv"
    row = _row("corrected", "2020-01-01")
    row["reviewer_reaction_valid"] = "no"
    row["corrected_reaction_id"] = "OER"
    pd.DataFrame([row]).to_csv(source, index=False)
    summary = compile_adjudicated_events(source, output)
    events = pd.read_csv(output)
    assert summary["accepted_rows"] == 1
    assert events.loc[0, "reaction_id"] == "OER"
