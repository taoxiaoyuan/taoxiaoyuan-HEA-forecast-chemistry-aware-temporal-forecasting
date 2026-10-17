import pandas as pd

from hea_forecast.temporal import build_candidate_table, expand_element_pairs


def test_backtest_uses_only_existing_systems_and_future_first_edges() -> None:
    evidence = pd.DataFrame(
        [
            {
                "paper_id": "P1",
                "publication_date": "2018-01-01",
                "system_id": "Co-Cr-Fe-Mn-Ni",
                "reaction_ids": "HER",
                "manual_status": "accepted",
            },
            {
                "paper_id": "P2",
                "publication_date": "2021-01-01",
                "system_id": "Co-Cr-Fe-Mn-Ni",
                "reaction_ids": "OER",
                "manual_status": "accepted",
            },
            {
                "paper_id": "P3",
                "publication_date": "2021-01-01",
                "system_id": "Ag-Au-Cu-Pd-Pt",
                "reaction_ids": "ORR",
                "manual_status": "accepted",
            },
        ]
    )
    table = build_candidate_table(
        evidence,
        reactions=["HER", "OER", "ORR"],
        cutoff_year=2018,
        horizon_start=2019,
        horizon_end=2021,
    )
    assert set(table["system_id"]) == {"Co-Cr-Fe-Mn-Ni"}
    assert "HER" not in set(table["reaction_id"])
    oer = table.loc[table["reaction_id"].eq("OER")].iloc[0]
    assert oer["label"] == 1
    assert oer["system_paper_count"] == 1


def test_unverified_evidence_is_excluded_by_default() -> None:
    evidence = pd.DataFrame(
        [
            {
                "paper_id": "P1",
                "publication_date": "2018-01-01",
                "system_id": "Co-Cr-Fe-Mn-Ni",
                "reaction_ids": "HER",
                "manual_status": "pending",
            }
        ]
    )
    table = build_candidate_table(
        evidence,
        reactions=["HER", "OER"],
        cutoff_year=2018,
        horizon_start=2019,
        horizon_end=2021,
    )
    assert table.empty


def test_expand_element_pairs() -> None:
    evidence = pd.DataFrame(
        [
            {
                "paper_id": "P1",
                "publication_date": "2020-01-01",
                "system_id": "Co-Fe-Ni",
                "reaction_ids": "HER",
            }
        ]
    )
    expanded = expand_element_pairs(evidence)
    assert set(expanded["system_id"]) == {"Co-Fe", "Co-Ni", "Fe-Ni"}


def test_reviewed_single_reaction_rows_are_supported() -> None:
    evidence = pd.DataFrame(
        [
            {
                "paper_id": "P1",
                "publication_date": "2020-01-01",
                "system_id": "Co-Fe-Ni-Pt",
                "reaction_id": "HER",
                "adjudication_status": "accepted",
            },
            {
                "paper_id": "P2",
                "publication_date": "2021-01-01",
                "system_id": "Co-Fe-Ni-Pt",
                "reaction_id": "OER",
                "adjudication_status": "rejected",
            },
        ]
    )
    table = build_candidate_table(evidence, ["HER", "OER"], 2020, 2021, 2022)
    assert len(table) == 1
    assert table.iloc[0]["reaction_id"] == "OER"
    assert table.iloc[0]["label"] == 0


def test_candidate_text_is_cutoff_safe_and_reaction_redacted() -> None:
    evidence = pd.DataFrame(
        [
            {
                "paper_id": "P1",
                "publication_date": "2020-01-01",
                "system_id": "Co-Fe-Ni-Pt",
                "reaction_id": "HER",
                "adjudication_status": "accepted",
                "display_name": "CoFeNiPt for hydrogen evolution reaction",
            }
        ]
    )
    table = build_candidate_table(evidence, ["HER", "OER"], 2020, 2021, 2022)
    assert "hydrogen evolution" not in table.iloc[0]["text_document"].lower()
    assert "cofenipt" in table.iloc[0]["text_document"].lower()
