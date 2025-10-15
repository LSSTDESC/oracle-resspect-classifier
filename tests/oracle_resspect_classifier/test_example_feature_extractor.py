from oracle_resspect_classifier.example_feature_extractor import ExampleFeatureExtractor


def test_expected_attributes() -> None:
    """Verify the expected attributes of the `ExampleFeatureExtractor` class"""

    fe = ExampleFeatureExtractor
    assert hasattr(fe, "feature_names")
    assert isinstance(fe.feature_names, list)

    assert hasattr(fe, "id_column")
    assert isinstance(fe.id_column, str)

    assert hasattr(fe, "label_column")
    assert isinstance(fe.label_column, str)

    assert hasattr(fe, "non_anomaly_classes")
    assert isinstance(fe.non_anomaly_classes, list)
