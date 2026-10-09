import pandas as pd

from hea_forecast.temporal import build_historical_training_table


def test_historical_positive_removes_target_edge_from_degree() -> None:
    evidence = pd.DataFrame(
        [
            {
                "paper_id": "p1",
                "publication_date": "2020-01-01",
                "system_id": "Co-Fe-Ni-Pt",
                "reaction_ids": "HER|OER",
            }
        ]
    )
    table = build_historical_training_table(
        evidence, ["HER", "OER", "ORR"], cutoff_year=2020, allow_pending=True
    )
    her = table.loc[table["reaction_id"].eq("HER")].iloc[0]
    orr = table.loc[table["reaction_id"].eq("ORR")].iloc[0]
    assert her["label"] == 1
    assert her["system_reaction_degree"] == 1
    assert her["reaction_system_count"] == 0
    assert orr["label"] == 0
    assert orr["system_reaction_degree"] == 2
