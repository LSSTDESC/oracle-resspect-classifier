import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src/')))

from oracle_resspect_classifier.elasticc2_oracle_feature_extractor import ELAsTiCC2_ORACLEFeatureExtractor

def test_expected_attributes() -> None:
    """Verify the expected attributes of the `ExampleFeatureExtractor` class"""

    fe = ELAsTiCC2_ORACLEFeatureExtractor()
    assert hasattr(fe, "feature_names")
    assert isinstance(fe.feature_names, list)

    assert hasattr(fe, "id_column")
    assert isinstance(fe.id_column, str)

    assert hasattr(fe, "label_column")
    assert isinstance(fe.label_column, str)

    assert hasattr(fe, "non_anomaly_classes")
    assert isinstance(fe.non_anomaly_classes, list)
    
    assert getattr(fe, "num_features") == 24
