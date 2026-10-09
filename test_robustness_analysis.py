from __future__ import annotations

import pandas as pd

from hea_forecast.robustness_analysis import (
    _extract_metrics,
    benjamini_hochberg,
    reviewer_agreement,
    wilson_interval,
)


def test_extract_metrics_finds_common_electrocatalysis_values() -> None:
    text = "The catalyst needs an overpotential of 210 mV at 10 mA cm-2 and has a Tafel slope of 58 mV dec-1, remaining stable for 40 h."
    result = _extract_metrics(text)
    assert result["overpotential_mv"] == 210
    assert result["current_density_ma_cm2"] == 10
    assert result["tafel_mv_dec"] == 58
    assert result["stability_hours"] == 40
    assert result["performance_evidence_found"]


def test_extract_metrics_handles_paired_values() -> None:
    text = "overpotentials of 76.7 and 173.8 mV at − 10 and − 50 mA cm − 2"
    result = _extract_metrics(text)
    assert result["overpotential_mv"] == 76.7
    assert result["current_density_ma_cm2"] == 10


def test_reviewer_agreement_reports_perfect_match(tmp_path) -> None:
    source = tmp_path / "reviews.csv"
    output = tmp_path / "agreement.json"
    pd.DataFrame(
        {
            "reviewer_material_valid": ["yes", "no", "yes"],
            "reviewer2_material_valid": ["yes", "no", "yes"],
        }
    ).to_csv(source, index=False)
    result = reviewer_agreement(source, output)
    assert result["fields"]["reviewer2_material_valid"]["agreement"] == 1.0
    assert result["fields"]["reviewer2_material_valid"]["cohen_kappa"] == 1.0
    assert output.exists()


def test_wilson_interval_contains_observed_precision() -> None:
    low, high = wilson_interval(2, 10)
    assert low < 0.2 < high
    assert round(low, 3) == 0.057
    assert round(high, 3) == 0.510


def test_benjamini_hochberg_is_monotone_by_sorted_p_value() -> None:
    p_values = [0.04, 0.001, 0.02, 0.5]
    adjusted = benjamini_hochberg(p_values)
    ordered = sorted(zip(p_values, adjusted))
    assert [q for _, q in ordered] == sorted(q for _, q in ordered)
    assert all(p <= q <= 1 for p, q in zip(p_values, adjusted, strict=False))
