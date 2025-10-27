import numpy as np

from oracle_resspect_classifier.oracle_feature_extractor import ORACLEFeatureExtractor

class ELAsTiCC2_ORACLEFeatureExtractor(ORACLEFeatureExtractor):
    
    host_galaxy_features = [
        'hostgal_zphot',
        'hostgal_zphot_err',
        'hostgal_zspec',
        'hostgal_zspec_err',
        'hostgal_ra',
        'hostgal_dec',
        'hostgal_snsep',
        'hostgal_ellipticity',
        'hostgal_mag_*'
    ]
    
    lsst_filters = ['u', 'g', 'r', 'i', 'z', 'y']
    
    TOM_scaling_const = 10 ** ((31.4 - 27.5) / 2.5)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # total number of features (adding on to the features already present in self.features from ORACLEFeatureExtractor)
        # currently subtracting one from the host_galaxy_features list length to account for the hostgal magnitudes in different bands
        self.num_features += len(ELAsTiCC2_ORACLEFeatureExtractor.lsst_filters) + (len(ELAsTiCC2_ORACLEFeatureExtractor.host_galaxy_features) - 1)
        
    @classmethod
    def get_features(cls, filters: list) -> list[str]:
        return super().get_features(filters) + cls._get_host_features()
    
    @classmethod
    def _get_host_features(cls) -> list[str]:
        host_features = []
        host_features.extend(ELAsTiCC2_ORACLEFeatureExtractor.host_galaxy_features[:-1])
        
        host_features.extend(
            cls._get_features_per_filter(list(ELAsTiCC2_ORACLEFeatureExtractor.host_galaxy_features[-1]),
                                         ELAsTiCC2_ORACLEFeatureExtractor.lsst_filters)
        )
        
        return host_features
    
    def fit_all(self) -> np.ndarray:
        lc = self.photometry
        
        return np.array([])