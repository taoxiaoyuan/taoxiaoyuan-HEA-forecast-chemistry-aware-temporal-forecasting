import json
from pathlib import Path

import pandas as pd

from hea_forecast.audit import audit_corpus


def test_audit_finds_hea_reaction_candidate(tmp_path: Path) -> None:
    source = tmp_path / "works.csv"
    pd.DataFrame(
        [
            {
                "id": "W1",
                "display_name": "CoFeNiCrMn high-entropy alloy for oxygen evolution",
                "abstract": "The catalyst is evaluated for the OER.",
                "publication_date": "2021-04-03",
            },
            {
                "id": "W2",
                "display_name": "Binary alloy mechanics",
                "abstract": "A mechanical study.",
                "publication_date": "2020-01-01",
            },
        ]
    ).to_csv(source, index=False)

    output = tmp_path / "audit"
    summary = audit_corpus(source, Path("configs/project.yaml"), output)

    assert summary["hea_candidates"] == 1
    assert summary["hea_reaction_candidates"] == 1
    assert summary["reaction_candidate_counts"]["OER"] == 1
    saved = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert saved["publication_year_max"] == 2021


def test_short_reaction_acronym_does_not_match_inside_word(tmp_path: Path) -> None:
    source = tmp_path / "works.csv"
    pd.DataFrame(
        [
            {
                "id": "W1",
                "display_name": "Whether a high entropy alloy is stable",
                "abstract": "A mechanical investigation without catalysis.",
                "publication_date": "2020-01-01",
            }
        ]
    ).to_csv(source, index=False)

    summary = audit_corpus(source, Path("configs/project.yaml"), tmp_path / "audit")

    assert summary["hea_candidates"] == 1
    assert summary["hea_reaction_candidates"] == 0
    assert summary["reaction_candidate_counts"]["HER"] == 0
