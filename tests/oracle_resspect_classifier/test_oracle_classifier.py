import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src/')))

from oracle_resspect_classifier.oracle_classifier import ExampleClassifier


def test_example_classifier():
    """Test for basic adherence to the ResspectClassifier API."""
    ec = ExampleClassifier()

    assert hasattr(ec, "classifier")

def test_my_classifier():
    """Test for basic adherence to the partial sklearn classifier API."""
    ec = ExampleClassifier()

    assert callable(ec.classifier.fit)
    assert callable(ec.classifier.predict)
    assert callable(ec.classifier.predict_proba)
