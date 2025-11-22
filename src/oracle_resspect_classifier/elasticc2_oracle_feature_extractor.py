import numpy as np
import pandas as pd
import polars as pl
from oracle.custom_datasets.ELAsTiCC import time_dependent_feature_list

from oracle_resspect_classifier.oracle_feature_extractor import ORACLEFeatureExtractor
from oracle_resspect_classifier.plot_elasticc_lightcurves import ELAsTiCC_plotter

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
    
    # maps time-series flux feature names in TOM/fastDB to the time-series feature names that ORACLE expects
    ts_feature_map = {
        'MJD': 'midpointtai',
        'midpointtai': 'midpointtai',
        'BAND': 'filtername',
        'FLUXCAL': 'psflux',
        'FLUXCALERR': 'psfluxerr' 
    }
    
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
    
    @classmethod
    def _plot_sample_lc(cls, lc: pd.DataFrame):
        x_ts = np.array(lc)
        plotter = ELAsTiCC_plotter()
        plotter.get_lc_plots(x_ts)
    
    def fit(self, plot_samples=False) -> pd.DataFrame:
        lc = self.photometry if type(self.photometry) == pd.DataFrame else pd.Series()
        
        if plot_samples and type(lc) == pd.DataFrame:
            self._plot_sample_lc(lc)
        
        time_series_features = pd.Series(0, index=time_dependent_feature_list)
        for feature_name in time_dependent_feature_list:
            time_series_features[feature_name] = lc[ELAsTiCC2_ORACLEFeatureExtractor.ts_feature_map[feature_name]]
        
        host_feature_names = self._get_host_features()
        static_features = pd.Series(0, index=ORACLEFeatureExtractor.static_feature_names + host_feature_names)
        
        for feature_name in (ORACLEFeatureExtractor.static_feature_names + 
                                             host_feature_names):
            static_features[feature_name] = obj_data[feature_name]
        
        return pd.merge(time_series_features, static_features, on='diaobject_id')

    def fit_all(self, obj_data=None, parquet_path='TOM_days_storage/TOM_training_features.csv', plot_samples=False):
        # write features to intermediate parquet file that ORACLE can ingest
        features_df = self.fit(obj_data, plot_samples=plot_samples)
        self.features = features_df
        
        features_df = pl.from_pandas(features_df)
        features_df.write_csv(parquet_path)