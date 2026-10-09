from __future__ import annotations

import pandas as pd

from hea_forecast.no_dft_upgrade import (
    historical_randomisation_audit,
    prepare_expert_blind_study,
    prepare_independent_corpus_validation,
)


def _future_frame() -> pd.DataFrame:
    rows = []
    for index in range(30):
        rows.append(
            {
                "system_id": f"Co-Fe-Ni-Pt-X{index}",
                "reaction_id": "HER",
                "ensemble_score": 1 - index / 30,
            }
        )
    return pd.DataFrame(rows)


def test_prepare_expert_study_hides_scores(tmp_path) -> None:
    source = tmp_path / "future.csv"
    _future_frame().to_csv(source, index=False)
    result = prepare_expert_blind_study(source, tmp_path / "expert", per_stratum=3)
    public = pd.read_csv(tmp_path / "expert" / "expert_blind_review_template.csv")
    private = pd.read_csv(tmp_path / "expert" / "PRIVATE_expert_study_key.csv")
    assert result["candidates"] == 9
    assert "ensemble_score" not in public.columns
    assert "ensemble_score" in private.columns


def test_prepare_external_validation_is_empty_and_locked(tmp_path) -> None:
    source = tmp_path / "future.csv"
    _future_frame().to_csv(source, index=False)
    result = prepare_independent_corpus_validation(source, tmp_path / "external")
    intake = pd.read_csv(tmp_path / "external" / "external_corpus_records_template.csv")
    key = pd.read_csv(tmp_path / "external" / "locked_candidate_key.csv")
    assert result["locked_candidates"] == 30
    assert intake.empty
    assert len(key) == 30


def test_randomisation_audit_writes_empirical_p_values(tmp_path) -> None:
    directory = tmp_path / "predictions"
    directory.mkdir()
    pd.DataFrame(
        {
            "label": [1, 1, 0, 0, 0, 0],
            "score": [0.9, 0.8, 0.4, 0.3, 0.2, 0.1],
        }
    ).to_csv(directory / "predictions_2024_demo.csv", index=False)
    output = tmp_path / "audit.csv"
    result = historical_randomisation_audit(
        {"system": directory}, output, permutations=100, seed=7
    )
    frame = pd.read_csv(output)
    assert result["rows"] == 1
    assert 0 < frame.loc[0, "pr_auc_empirical_p"] <= 1
