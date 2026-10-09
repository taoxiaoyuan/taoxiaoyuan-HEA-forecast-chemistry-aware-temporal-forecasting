import pandas as pd

from hea_forecast.survival import build_person_period_table


def test_person_period_table_keeps_at_risk_pairs_until_event() -> None:
    evidence = pd.DataFrame(
        [
            {
                "paper_id": "P1",
                "publication_date": "2019-01-01",
                "system_id": "Co-Fe-Ni-Pt",
                "reaction_id": "HER",
                "adjudication_status": "accepted",
            },
            {
                "paper_id": "P2",
                "publication_date": "2021-01-01",
                "system_id": "Co-Fe-Ni-Pt",
                "reaction_id": "OER",
                "adjudication_status": "accepted",
            },
        ]
    )
    table = build_person_period_table(evidence, ["HER", "OER"], cutoff_year=2021)
    oer = table.loc[table["reaction_id"].eq("OER")]
    assert set(oer["risk_year"]) == {2020, 2021}
    assert oer.loc[oer["risk_year"].eq(2021), "label"].iloc[0] == 1
