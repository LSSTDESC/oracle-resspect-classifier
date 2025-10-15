from resspect.classifiers import ResspectClassifier


class ExampleClassifier(ResspectClassifier):
    """Example of an externally defined classifier for RESSPECT. The API for the
    subclass of ResspectClassifier itself is very simple. However, the classifier
    that is assigned to `self.classifier` has a more substantial expected API,
    based on the scikit-learn API for classifiers.

    The MyClassifier class shows the methods that are expected to be implemented
    by the classifier."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.classifier = MyClassifier(**self.kwargs)


class MyClassifier:
    """Example of a classifier. Note that the expected API mirrors a portion of
    the scikit-learn classifier API.
    """

    def __init__(self, **kwargs):
        """It is better to define __init__ with the explicitly required input
        parameters instead of `**kwargs`."""
        pass

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

    def predict(self, test_features: list) -> list:
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
        pass

    def predict_proba(self, test_features: list) -> list:
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
        pass
