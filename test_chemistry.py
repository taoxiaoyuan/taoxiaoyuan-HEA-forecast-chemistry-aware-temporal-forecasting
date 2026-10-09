from hea_forecast.chemistry import extract_formula_candidates, parse_formula_token


def test_parse_compact_and_hyphenated_hea_formula() -> None:
    compact = parse_formula_token("PtCoNiFeCu")
    hyphenated = parse_formula_token("Pt-Co-Ni-Fe-Cu")
    assert compact is not None
    assert hyphenated is not None
    assert compact.system_id == "Co-Cu-Fe-Ni-Pt"
    assert hyphenated.system_id == compact.system_id


def test_parse_stoichiometric_formula() -> None:
    candidate = parse_formula_token("Al0.3CoCrFeNi")
    assert candidate is not None
    assert candidate.system_id == "Al-Co-Cr-Fe-Ni"


def test_reject_prose_and_binary_compounds() -> None:
    assert parse_formula_token("HighEntropyAlloy") is None
    assert parse_formula_token("TiO2") is None


def test_extracts_formula_before_based_suffix() -> None:
    candidates = extract_formula_candidates("CoCrFeMnNi-based high entropy alloy")
    assert [candidate.system_id for candidate in candidates] == ["Co-Cr-Fe-Mn-Ni"]
