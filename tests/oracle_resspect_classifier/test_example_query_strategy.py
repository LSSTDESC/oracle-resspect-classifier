from oracle_resspect_classifier.example_query_strategy import ExampleQueryStrategy


def test_example_query_strategy():
    """Test for basic adherence to the QueryStrategy API."""
    eqs = ExampleQueryStrategy(queryable_ids=[1, 2, 3], test_ids=[4, 5, 6])

    assert hasattr(eqs, "queryable_ids")
    assert hasattr(eqs, "test_ids")
    assert hasattr(eqs, "batch")
    assert hasattr(eqs, "query_threshold")
    assert hasattr(eqs, "screen")
    assert hasattr(eqs, "sample")
    assert callable(eqs.sample)
