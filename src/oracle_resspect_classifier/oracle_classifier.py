from resspect.classifiers import ResspectClassifier
from oracle.pretrained.ELAsTiCC import ORACLE1_ELAsTiCC, ORACLE_Taxonomy
from astropy.table import Table
from numpy.typing import ArrayLike
import oracle.train
import numpy as np
import pandas as pd

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

    def __init__(self,
                 dir: str,
                 weights_dir: str,
                 num_epochs: int = oracle.train.default_num_epochs,
                 batch_size: int = oracle.train.default_batch_size,
                 lr: float = oracle.train.default_learning_rate,
                 max_n_per_class: int | None = oracle.train.default_max_n_per_class,
                 alpha: float = oracle.train.default_alpha,
                 gamma: float = oracle.train.default_gamma):
        """It is better to define __init__ with the explicitly required input
        parameters instead of `**kwargs`."""
        # hyperparameters
        self.num_epochs = num_epochs
        self.batch_size = batch_size,
        self.lr = lr
        self.max_n_per_class = max_n_per_class
        self.alpha = alpha
        self.gamma = gamma
        self.dir = dir
        self.model_type = "ELAsTiCC"
        self.weights_dir = weights_dir
        
        # model objects
        self.model = ORACLE1_ELAsTiCC(model_dir=self.weights_dir)
        self.taxonomy = ORACLE_Taxonomy()

    def load_classifier(self, pretrained_weights_path: str):
        # in the current oracle API, there is no way to instantiate the model without pretrained weights,
        # so the self.model object is already loaded and there isn't anything to do
        # for now this method just exists so that RESSPECT doesn't error out
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
        hyperparams = {
            "num_epochs": self.num_epochs,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "max_n_per_class": self.max_n_per_class,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "dir": self.dir,
            "model": self.model_type,
            "load_weights": self.load_weights
        }
        oracle.train.run_training_loop(args=hyperparams)
        
        # TODO: fetch updated model from wandb

    # NOTE: this function was originally written with type signature (self, list) -> list; had to be adapted for ORACLE but may require changes elsewhere
    def predict(self, test_features: ArrayLike) -> dict:
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
        if type(test_features) == pd.DataFrame:
            test_dataframe = Table.from_pandas(test_features)
            
            # TODO: explode the table, add the static features as metadata and just keep time series features in the actual table
            # so that the format works with what the oracle API expects
        else:
            test_dataframe = Table(test_features)
            
        test_dataframe.pprint(max_width=-1)
        
        return self.model.predict(test_dataframe)

    def predict_proba(self, test_features: list) -> dict:
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
        test_dataframe = Table(test_features)
        return self.model.score(test_dataframe)
