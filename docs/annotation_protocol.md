# Annotation protocol

## Purpose

Each row determines whether a paper creates a dated edge between one normalized
HEA system and one electrocatalytic reaction. Review the title and evidence
context first; consult the abstract or paper only when those fields are
insufficient.

## Reviewer fields

Use only `yes`, `no`, or `uncertain` in the first three fields.

- `reviewer_material_valid`: `yes` when the extracted composition is genuinely
  studied as a high-entropy or multi-principal-element material. Exclude a list
  of separately tested metals, conventional dilute alloys, and background-only
  mentions.
- `reviewer_reaction_valid`: `yes` when the assigned reaction is actually
  investigated in the paper. Do not infer a reaction from generic terms such as
  water splitting when the half-reaction is not identifiable.
- `reviewer_target_catalyst`: `yes` when the extracted material is an evaluated
  catalyst, electrode, or catalytic precursor central to the work. Exclude
  comparison materials and passing mentions.
- `corrected_system_id`: leave blank when correct. Otherwise provide unique
  element symbols in alphabetical order separated by hyphens, for example
  `Co-Cu-Fe-Ni-Pt`.
- `corrected_reaction_id`: leave blank when correct. Otherwise use one of
  `HER`, `OER`, `ORR`, or `CO2RR`.
- `reviewer_notes`: briefly record exclusion or correction evidence.
- `adjudication_status`: use `accepted`, `rejected`, or `needs_adjudication`.

## Acceptance rule

A row is accepted only when all three validity fields are `yes`. If a formula or
reaction is wrong but can be corrected unambiguously, enter the correction and
accept the row. Mark conflicting, incomplete, or interpretation-dependent
records as `needs_adjudication`; never force them into a binary label.

## Date and duplicate rule

The discovery date is the earliest verifiable public date. A preprint and its
journal version count as one event. A later paper using the same system-reaction
pair is supporting evidence, not a new discovery edge.

## Difficult cases

- Oxides, phosphides, nitrides, and other high-entropy compounds are retained
  only when their multi-element sublattice is identifiable; keep the material
  class recorded separately.
- Supported HEAs are valid when the HEA is the active catalyst rather than merely
  a substrate or additive.
- Bifunctional water-splitting papers may create both HER and OER edges only when
  both half-reactions are experimentally evaluated.
- Reviews, perspectives, bibliometric papers, and computational screening
  without a newly studied catalyst do not establish experimental discovery
  edges. Computational events may later be modeled as a separate relation.

## Quality control

Independently double-review all future-window positives and a stratified 20%
sample of negatives. Report Cohen's kappa for the three binary decisions.
Resolve disagreements before creating temporal labels. Keep the untouched
machine extraction and both reviewer versions for auditability.
