from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

PROPERTY_NAMES = (
    "atomic_number",
    "atomic_mass",
    "electronegativity",
    "atomic_radius",
    "row",
    "group",
)

GAS_CONSTANT_J_MOL_K = 8.314462618

# Versioned, transparent screening groups.  These are not live price data and
# must not be interpreted as a formal critical-minerals assessment.
SUPPLY_SCREENING_GROUPS = {
    "platinum_group": {"Ru", "Rh", "Pd", "Os", "Ir", "Pt"},
    "critical_transition": {"Co", "Cr", "Mn", "Ni", "V", "W"},
    "refractory": {"Cr", "Hf", "Mo", "Nb", "Re", "Ta", "Ti", "V", "W", "Zr"},
    "abundant_3d": {"Cr", "Mn", "Fe", "Co", "Ni", "Cu"},
}

ELEMENT_FAMILIES = {
    "noble": {"Ru", "Rh", "Pd", "Ag", "Os", "Ir", "Pt", "Au"},
    "platinum_group": {"Ru", "Rh", "Pd", "Os", "Ir", "Pt"},
    "late_transition": {"Fe", "Co", "Ni", "Cu", "Ru", "Rh", "Pd", "Ag", "Os", "Ir", "Pt", "Au"},
    "first_row_transition": {"Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn"},
    "oxophilic": {"Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Zr", "Nb", "Mo", "Hf", "Ta", "W"},
    "her_family": {"Ni", "Mo", "W", "Ru", "Rh", "Pd", "Ir", "Pt"},
    "oer_family": {"Fe", "Co", "Ni", "Ru", "Ir", "Mn"},
    "orr_family": {"Fe", "Co", "Ni", "Cu", "Pd", "Ag", "Pt", "Au"},
    "co2rr_family": {"Cu", "Ag", "Au", "Sn", "Bi", "In", "Pd"},
}


def _element_values(symbol: str) -> dict[str, float]:
    try:
        from pymatgen.core import Element
    except ImportError as error:  # pragma: no cover - exercised only without optional dependency
        raise RuntimeError(
            "Chemical descriptors require the optional 'chemistry' dependencies"
        ) from error

    element = Element(symbol)
    radius = float(element.atomic_radius) if element.atomic_radius is not None else np.nan
    electronegativity = float(element.X) if element.X is not None else np.nan
    return {
        "atomic_number": float(element.Z),
        "atomic_mass": float(element.atomic_mass),
        "electronegativity": electronegativity,
        "atomic_radius": radius,
        "row": float(element.row),
        "group": float(element.group),
    }


def describe_elements(elements: Iterable[str]) -> dict[str, float]:
    """Aggregate composition-only properties without using future literature."""
    unique = sorted(set(elements))
    values = [_element_values(symbol) for symbol in unique]
    descriptors: dict[str, float] = {"chemical_element_count": float(len(unique))}
    for property_name in PROPERTY_NAMES:
        array = np.asarray([value[property_name] for value in values], dtype=float)
        valid = array[~np.isnan(array)]
        if valid.size:
            descriptors[f"chem_{property_name}_mean"] = float(valid.mean())
            descriptors[f"chem_{property_name}_std"] = float(valid.std(ddof=0))
            descriptors[f"chem_{property_name}_range"] = float(valid.max() - valid.min())
        else:
            descriptors[f"chem_{property_name}_mean"] = np.nan
            descriptors[f"chem_{property_name}_std"] = np.nan
            descriptors[f"chem_{property_name}_range"] = np.nan
    radius_mean = descriptors["chem_atomic_radius_mean"]
    radius_std = descriptors["chem_atomic_radius_std"]
    descriptors["chem_atomic_size_mismatch"] = (
        float(100 * radius_std / radius_mean)
        if radius_mean and not np.isnan(radius_mean)
        else np.nan
    )
    element_set = set(unique)
    denominator = len(element_set) or 1

    # Equiatomic configurational entropy and a transparent VEC proxy are useful
    # composition-level descriptors when stoichiometric coefficients are not
    # available.  For transition metals, the periodic-table group number is the
    # conventional VEC proxy; for main-group metals, the outer-shell count is
    # used (group - 10).
    vec_values: list[float] = []
    for symbol in unique:
        group = _element_values(symbol)["group"]
        if 3 <= group <= 12:
            vec_values.append(group)
        elif 13 <= group <= 18:
            vec_values.append(group - 10)
        elif group <= 2:
            vec_values.append(group)
    descriptors["chem_vec_mean"] = float(np.mean(vec_values)) if vec_values else np.nan
    descriptors["chem_vec_std"] = float(np.std(vec_values)) if vec_values else np.nan
    descriptors["chem_configurational_entropy_j_mol_k"] = (
        float(GAS_CONSTANT_J_MOL_K * np.log(len(unique))) if unique else 0.0
    )
    for family, members in ELEMENT_FAMILIES.items():
        descriptors[f"chem_{family}_fraction"] = len(element_set & members) / denominator
    for family, members in SUPPLY_SCREENING_GROUPS.items():
        descriptors[f"screen_{family}_fraction"] = len(element_set & members) / denominator
    return descriptors


def _reaction_specific_features(elements: Iterable[str], reaction_id: str) -> dict[str, float]:
    element_set = set(elements)
    denominator = len(element_set) or 1
    reaction = str(reaction_id).upper()
    family_name = {
        "HER": "her_family",
        "OER": "oer_family",
        "ORR": "orr_family",
        "CO2RR": "co2rr_family",
    }.get(reaction)
    active_fraction = (
        len(element_set & ELEMENT_FAMILIES[family_name]) / denominator if family_name else 0.0
    )
    return {
        "rxn_active_family_fraction": active_fraction,
        "rxn_noble_fraction": len(element_set & ELEMENT_FAMILIES["noble"]) / denominator,
        "rxn_oxophilic_fraction": len(element_set & ELEMENT_FAMILIES["oxophilic"]) / denominator,
        "rxn_first_row_fraction": len(element_set & ELEMENT_FAMILIES["first_row_transition"])
        / denominator,
    }


def add_elemental_features(frame: pd.DataFrame, system_column: str = "system_id") -> pd.DataFrame:
    """Attach deterministic elemental descriptors to a candidate table."""
    systems = frame[system_column].fillna("").astype(str)
    unique_descriptors = {
        system: describe_elements(system.split("-")) for system in systems.unique() if system
    }
    descriptors = pd.DataFrame(
        [unique_descriptors.get(system, {}) for system in systems], index=frame.index
    )
    if "reaction_id" in frame.columns:
        reaction_features = pd.DataFrame(
            [
                _reaction_specific_features(system.split("-"), reaction)
                for system, reaction in zip(systems, frame["reaction_id"], strict=False)
            ],
            index=frame.index,
        )
        descriptors = pd.concat([descriptors, reaction_features], axis=1)
    return pd.concat([frame.copy(), descriptors], axis=1)
