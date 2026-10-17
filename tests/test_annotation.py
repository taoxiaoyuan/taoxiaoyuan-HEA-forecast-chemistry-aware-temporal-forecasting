from pathlib import Path

import pandas as pd

from hea_forecast.annotation import create_annotation_batch


def test_annotation_batch_explodes_reactions(tmp_path: Path) -> None:
    source = tmp_path / "evidence.csv"
    pd.DataFrame(
        [
            {
                "paper_id": "W1",
                "doi": "10.1/x",
                "publication_date": "2021-01-01",
                "display_name": "HEA for water splitting",
                "composition_raw": "CoCrFeMnNi",
                "system_id": "Co-Cr-Fe-Mn-Ni",
                "reaction_ids": "HER|OER",
                "material_class": "metallic_hea",
                "extraction_confidence": "high",
                "formula_in_title": True,
                "reaction_in_title": True,
                "evidence_context": "evidence",
            }
        ]
    ).to_csv(source, index=False)

    destination = tmp_path / "batch.csv"
    summary = create_annotation_batch(source, destination)
    batch = pd.read_csv(destination)
    assert summary["annotation_rows"] == 2
    assert set(batch["reaction_id"]) == {"HER", "OER"}
    assert set(batch["adjudication_status"]) == {"pending"}
