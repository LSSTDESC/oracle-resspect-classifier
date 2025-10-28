from resspect.classifiers import ResspectClassifier
from oracle.pretrained.ELAsTiCC import ORACLE1_ELAsTiCC, ORACLE_Taxonomy
from astropy.table import Table
import numpy as np

class OracleResspectClassifier(ResspectClassifier):
    """Example of an externally defined classifier for RESSPECT. The API for the
    subclass of ResspectClassifier itself is very simple. However, the classifier
    that is assigned to `self.classifier` has a more substantial expected API,
    based on the scikit-learn API for classifiers.

    The MyClassifier class shows the methods that are expected to be implemented
    by the classifier."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.classifier = OracleClassifier(**self.kwargs)


class OracleClassifier:
    """Example of a classifier. Note that the expected API mirrors a portion of
    the scikit-learn classifier API.
    """

    def __init__(self, **kwargs):
        """It is better to define __init__ with the explicitly required input
        parameters instead of `**kwargs`."""
        self.model = ORACLE1_ELAsTiCC()
        self.taxonomy = ORACLE_Taxonomy()

    def fit(self, train_features: list, train_labels: list) -> None:
        """Fit the classifier to the training data. Not that there is no return
        value, it is only expected to fit the classifier to the data.

        Parameters
        ----------
        train_features : array-like
            The features used for training, [n_samples, m_features].
        train_labels : array-like
            The training labels, [n_samples].
        """
        pass

    # NOTE: this function was originally written with type signature (self, list) -> list; had to be adapted for ORACLE but may require changes elsewhere
    def predict(self, test_dataframe: Table) -> dict:
        """Predict the class labels for the test data.

        Parameters
        ----------
        test_features : array-like
            The features used for testing, [n_samples, m_features].

        Returns
        -------
        predictions : array-like
            The predicted class labels, [n_samples].
        """
        return self.model.predict(test_dataframe)

    def predict_proba(self, test_dataframe: Table) -> dict:
        """Predict the class probabilities for the test data.

        Parameters
        ----------
        test_features : array-like
            The features used for testing, [n_samples, m_features].

        Returns
        -------
        probabilities : array-like
            The predicted class probabilities, [n_samples, n_classes].
        """
        return self.model.score(test_dataframe)
