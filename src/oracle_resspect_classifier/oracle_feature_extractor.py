import numpy as np
import itertools

from resspect.feature_extractors.light_curve import LightCurve


class ORACLEFeatureExtractor(LightCurve):
    """A minimal example of an external feature extractor class."""

    # Static feature names that will be extracted from a light curve.
    static_feature_names = [
        'ra',
        'decl',
        'mwebv',
        'mwebv_err',
        'z_final',
        'z_final_err',
    ]
    
    ts_feature_names = [
        'midpointtai',
        'filtername',
        'psflux',
        'psfluxerr',
        'photflag',
    ]

    # The name of the id column of the data. e.g. 'ID', 'obj_id', etc.
    id_column = "diaobject_id"

    # The name of the label column. e.g. 'type', 'label', 'sntype', etc.
    # This is the column where the class label is stored.
    label_column = "type"

    # The names of classes that are NOT anomalies. e.g. ['Ia', 'Normal', etc.]
    non_anomaly_classes = ['Ia']    # at the moment, we're using ORACLE as a binary Ia vs. non-Ia classifier, will be expanded to more classes later on

    def __init__(self):
        super().__init__()
        self.num_features = len(ORACLEFeatureExtractor.static_feature_names) + len(ORACLEFeatureExtractor.ts_feature_names)
        self.features = None

    @classmethod
    def get_features(cls, filters: list) -> list[str]:
        """
        A class method that returns the list of features that will be extracted
        from the light curve.

        Often the feature list is a cross product of the feature name and the filters.
        In this example we return only the feature name list.

        Returns
        -------
        feature_names: list[str]
            List of feature names that will be extracted from the light curve.
        """
        return cls.ts_feature_names + cls.static_feature_names

    @classmethod
    def get_metadata_columns(cls) -> list[str]:
        """
        A class method that returns the metadata columns for the feature extractor.
        Depending on how dynamic the metadata columns are, this method can be
        a hard-coded list or a dynamically generated list.

        Returns
        -------
        metadata_columns: list[str]
            List of metadata columns for the feature extractor.
        """
        return [cls.id_column, "redshift", cls.label_column, "sncode"]

    # TODO: To be implemented
    @classmethod
    def get_feature_header(cls, filters: list[str], **kwargs) -> list[str]:
        """
        A class method that returns the full list column names for an output file.
        This includes the metadata columns and the feature columns.

        Returns
        -------
        header: list[str]
            List of column names for the output file.
        """

        # One way to return this is to concatenate the metadata columns and the feature columns
        return ORACLEFeatureExtractor.get_metadata_header(**kwargs) + ORACLEFeatureExtractor.get_features(
            filters
        )
        
    @classmethod
    def _get_features_per_filter(cls, features: list, filters: list) -> list[str]:
        """Simple function to get all possible combinations of features and filters.
        Will replace the '*' in the feature name with the filter name.

        i.e. features = ['example_*'], filters = ['r', 'g']. Returns ['example_r', 'example_g']

        Parameters
        ----------
        features : list[str]
            List of features where each '*' in the name will be replaced with the filter name.
        filters : list[str]
            List of filters to replace the '*' in the feature names.

        Returns
        -------
        list[str]
            List of features with the '*' replaced by the filter name.
        """

        return [pair[0].replace('*', pair[1]) for pair in itertools.product(features, filters)]

    # TODO: To be implemented
    def fit_all(self) -> np.ndarray:
        """
        Extracts features for all light curves in the dataset.

        Returns
        -------
        features: np.ndarray
            Features extracted from the light curves.
        """
        # Implement feature extraction here
        self.features = self._example_extraction_function()
        return self.features

    # TODO: To be implemented
    def _example_extraction_function(self):
        # Just for demo purposes
        pass

    # TODO: To be implemented
    def get_features_to_write(self):
        """
        Implement this method to return the features that will be persisted to disk.
        The base `LightCurve` class has a simple implementation, but you can
        override it here.

        The base `LightCurve` class implementation will return `features_list`:
         features_list = [
            self.id,
            self.redshift,
            self.sntype,
            self.sncode,
            self.sample]
        features_list.extend(self.features)
        """

        return super().get_features_to_write()
