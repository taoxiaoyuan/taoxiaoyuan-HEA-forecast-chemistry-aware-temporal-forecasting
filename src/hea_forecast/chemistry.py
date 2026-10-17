from __future__ import annotations

import re
from dataclasses import dataclass

ELEMENT_SYMBOLS = {
    "Ac",
    "Ag",
    "Al",
    "Am",
    "Ar",
    "As",
    "At",
    "Au",
    "B",
    "Ba",
    "Be",
    "Bh",
    "Bi",
    "Bk",
    "Br",
    "C",
    "Ca",
    "Cd",
    "Ce",
    "Cf",
    "Cl",
    "Cm",
    "Cn",
    "Co",
    "Cr",
    "Cs",
    "Cu",
    "Db",
    "Ds",
    "Dy",
    "Er",
    "Es",
    "Eu",
    "F",
    "Fe",
    "Fl",
    "Fm",
    "Fr",
    "Ga",
    "Gd",
    "Ge",
    "H",
    "He",
    "Hf",
    "Hg",
    "Ho",
    "Hs",
    "I",
    "In",
    "Ir",
    "K",
    "Kr",
    "La",
    "Li",
    "Lr",
    "Lu",
    "Lv",
    "Mc",
    "Md",
    "Mg",
    "Mn",
    "Mo",
    "Mt",
    "N",
    "Na",
    "Nb",
    "Nd",
    "Ne",
    "Nh",
    "Ni",
    "No",
    "Np",
    "O",
    "Og",
    "Os",
    "P",
    "Pa",
    "Pb",
    "Pd",
    "Pm",
    "Po",
    "Pr",
    "Pt",
    "Pu",
    "Ra",
    "Rb",
    "Re",
    "Rf",
    "Rg",
    "Rh",
    "Rn",
    "Ru",
    "S",
    "Sb",
    "Sc",
    "Se",
    "Sg",
    "Si",
    "Sm",
    "Sn",
    "Sr",
    "Ta",
    "Tb",
    "Tc",
    "Te",
    "Th",
    "Ti",
    "Tl",
    "Tm",
    "Ts",
    "U",
    "V",
    "W",
    "Xe",
    "Y",
    "Yb",
    "Zn",
    "Zr",
}

# Constituents counted toward the metallic-HEA threshold. B and Si may remain in
# the normalized system when present, but do not count as metallic constituents.
NONMETAL_OR_METALLOID = {
    "H",
    "He",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Sb",
    "Te",
    "I",
    "Xe",
    "At",
    "Rn",
    "Ts",
    "Og",
}

SUBSCRIPT_TRANSLATION = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
FORMULA_PART = re.compile(r"([A-Z][a-z]?)(\d+(?:\.\d+)?)?")
TOKEN_PATTERN = re.compile(r"(?<![A-Za-z0-9])\(?[A-Z][A-Za-z0-9.₀-₉()\-–—−]{3,}(?![A-Za-z0-9])")


@dataclass(frozen=True)
class FormulaCandidate:
    raw: str
    elements: tuple[str, ...]
    metal_count: int

    @property
    def system_id(self) -> str:
        return "-".join(sorted(set(self.elements)))


def parse_formula_token(token: str, minimum_metal_elements: int = 4) -> FormulaCandidate | None:
    """Parse a compact or hyphenated element formula into a normalized system."""
    raw = token.strip(".,;:[]{}")
    raw = re.sub(r"[-–—−](?:based|type|derived)$", "", raw, flags=re.IGNORECASE)
    normalized = raw.translate(SUBSCRIPT_TRANSLATION).strip("()")
    normalized = re.sub(r"[-–—−]", "", normalized)
    normalized = re.sub(r"\d+(?:\.\d+)?$", "", normalized) if ")" in raw else normalized
    parts = list(FORMULA_PART.finditer(normalized))
    if not parts or "".join(match.group(0) for match in parts) != normalized:
        return None
    elements = tuple(match.group(1) for match in parts)
    if any(element not in ELEMENT_SYMBOLS for element in elements):
        return None
    unique_elements = tuple(dict.fromkeys(elements))
    metal_count = sum(element not in NONMETAL_OR_METALLOID for element in set(elements))
    if metal_count < minimum_metal_elements:
        return None
    return FormulaCandidate(raw=raw, elements=unique_elements, metal_count=metal_count)


def extract_formula_candidates(
    text: str, minimum_metal_elements: int = 4
) -> list[FormulaCandidate]:
    """Extract unique compact HEA-like formula candidates from text."""
    candidates: dict[str, FormulaCandidate] = {}
    for match in TOKEN_PATTERN.finditer(text or ""):
        candidate = parse_formula_token(match.group(0), minimum_metal_elements)
        if candidate is not None:
            candidates.setdefault(candidate.system_id, candidate)
    return list(candidates.values())
