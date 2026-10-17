# Data dictionary

## Publication table

| Field | Meaning |
|---|---|
| `paper_id` | Stable internal record identifier |
| `openalex_id` | OpenAlex work identifier |
| `doi` | Canonical lowercase DOI when available |
| `title` | Publication title |
| `abstract` | Abstract used for local extraction, subject to redistribution rules |
| `publication_date` | Resolved event date used for temporal analysis |
| `publication_year` | Year derived from the resolved event date |
| `source_dataset` | Upstream Figshare or incremental OpenAlex batch |
| `retrieved_at` | UTC retrieval timestamp |
| `is_review` | Review-document indicator |
| `is_retracted` | Retraction indicator |

## Material-reaction evidence table

| Field | Meaning |
|---|---|
| `paper_id` | Publication foreign key |
| `composition_raw` | Verbatim material mention |
| `system_id` | Sorted unique element set, e.g. `Co-Cu-Fe-Ni-Pt` |
| `composition_id` | Normalized stoichiometric identifier when known |
| `material_class` | Metallic HEA or separately tracked high-entropy compound class |
| `reaction_id` | Canonical reaction label |
| `catalyst_role` | Target, comparator, support, precursor, or uncertain |
| `evidence_sentence` | Local context supporting the extracted relation |
| `extraction_confidence` | Calibrated automated confidence |
| `manual_status` | Pending, accepted, corrected, rejected, or adjudicated |

