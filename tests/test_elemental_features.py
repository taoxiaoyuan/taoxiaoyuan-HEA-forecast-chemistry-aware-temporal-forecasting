import pytest

pymatgen = pytest.importorskip("pymatgen")

import pandas as pd

from hea_forecast.elemental_features import add_elemental_features, describe_elements


def test_describe_elements_is_order_invariant() -> None:
    first = describe_elements(["Co", "Fe", "Ni"])
    second = describe_elements(["Ni", "Co", "Fe"])
    assert first == second
    assert first["chemical_element_count"] == 3
    assert first["chem_atomic_number_range"] > 0
    assert first["chem_vec_mean"] > 0
    assert first["chem_configurational_entropy_j_mol_k"] > 0
    assert first["screen_abundant_3d_fraction"] == 1.0


def test_reaction_specific_features_change_with_reaction() -> None:
    frame = pd.DataFrame(
        [
            {"system_id": "Co-Fe-Ni-Pt", "reaction_id": "HER"},
            {"system_id": "Co-Fe-Ni-Pt", "reaction_id": "CO2RR"},
        ]
    )
    enriched = add_elemental_features(frame)
    assert (
        enriched.loc[0, "rxn_active_family_fraction"]
        != enriched.loc[1, "rxn_active_family_fraction"]
    )
