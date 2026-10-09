from hea_forecast.openalex import canonical_doi, reconstruct_abstract


def test_reconstruct_abstract_orders_tokens() -> None:
    inverted = {"entropy": [2], "high": [0], "alloy": [3], "is": [1]}
    assert reconstruct_abstract(inverted) == "high is entropy alloy"


def test_canonical_doi_removes_url_prefix() -> None:
    assert canonical_doi("https://doi.org/10.1234/ABC") == "10.1234/abc"
    assert canonical_doi(None) == ""
