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
        'midpointtai': 'MJD',
        'filtername': 'BAND',
        'psflux': 'FLUXCAL',
        'psfluxerr': 'FLUXCALERR',
        'photflag': 'PHOTFLAG',
    }
    
    static_feature_map_parquet = {
        'ra': 'RA',
        'decl': 'DEC',
        'mwebv': 'MWEBV',
        'mwebv_err': 'MWEBV_ERR',
        'vpec': 'VPEC',
        'vpec_err': 'VPEC_ERR',
        'z_final': 'REDSHIFT_HELIO',
        'z_final_err': 'REDSHIFT_HELIO_ERR',
        'hostgal_zphot': 'HOSTGAL_PHOTOZ',
        'hostgal_zphot_err': 'HOSTGAL_PHOTOZ_ERR',
        'hostgal_zspec': 'HOSTGAL_SPECZ',
        'hostgal_zspec_err': 'HOSTGAL_SPECZ_ERR',
        'hostgal_ra': 'HOSTGAL_RA',
        'hostgal_dec': 'HOSTGAL_DEC',
        'hostgal_snsep': 'HOSTGAL_SNSEP',
        'hostgal_ellipticity': 'HOSTGAL_ELLIPTICITY',
        'hostgal_mag_u': 'HOSTGAL_MAG_u',
        'hostgal_mag_g': 'HOSTGAL_MAG_g',
        'hostgal_mag_r': 'HOSTGAL_MAG_r',
        'hostgal_mag_i': 'HOSTGAL_MAG_i',
        'hostgal_mag_z': 'HOSTGAL_MAG_z',
        'hostgal_mag_y': 'HOSTGAL_MAG_Y',
    }
    
    TOM_scaling_const = 10 ** ((31.4 - 27.5) / 2.5)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # total number of features (adding on to the features already present in self.features from ORACLEFeatureExtractor)
        # currently subtracting one from the host_galaxy_features list length to account for the hostgal magnitudes in different bands
        self.num_features += len(ELAsTiCC2_ORACLEFeatureExtractor.lsst_filters) + (len(ELAsTiCC2_ORACLEFeatureExtractor.host_galaxy_features) - 1)
        
    @classmethod
    def get_features(cls, filters: list) -> list[str]:
        return cls._get_ts_features() + cls._get_static_features()
    
    @classmethod
    def get_feature_header(cls, filters: list[str], **kwargs) -> list[str]:
        return super().get_metadata_columns() + ELAsTiCC2_ORACLEFeatureExtractor.get_features(ELAsTiCC2_ORACLEFeatureExtractor.lsst_filters)
    
    @classmethod
    def _get_static_features(cls) -> list[str]:
        return list(map(
            lambda x: ELAsTiCC2_ORACLEFeatureExtractor.static_feature_map_parquet.get(x, ""),
            cls.static_feature_names + cls._get_host_features()))
    
    @classmethod
    def _get_ts_features(cls) -> list[str]:
        return list(map(
            lambda x: ELAsTiCC2_ORACLEFeatureExtractor.ts_feature_map.get(x, ""),
            cls.ts_feature_names))
    
    @classmethod
    def _get_host_features(cls) -> list[str]:
        host_features = []
        host_features.extend(ELAsTiCC2_ORACLEFeatureExtractor.host_galaxy_features[:-1])
        
        host_features.extend(
            cls._get_features_per_filter([ELAsTiCC2_ORACLEFeatureExtractor.host_galaxy_features[-1]],
                                         ELAsTiCC2_ORACLEFeatureExtractor.lsst_filters)
        )
        
        return host_features
        
    @classmethod
    def _plot_sample_lc(cls, lc: pd.DataFrame):
        '''
        This is a helper function for visualizing light curves. It takes the astropy table of time series features,
        and returns the RGB image tensor for the plot, which can then be visualized with `plt.show()`
        '''
        # Only treat time-dependent columns as iterables, others as scalars
        if isinstance(lc, pd.DataFrame):
            try:
                if lc.shape[0] == 1:
                    # For time-dependent columns, convert the single row's cell to array
                    arrays = [np.asarray(lc[col].iloc[0]) if col in time_dependent_feature_list else np.array([lc[col].iloc[0]]) for col in lc.columns]
                    # Only stack time-dependent columns
                    x_ts = np.column_stack([np.asarray(lc[col].iloc[0]) for col in time_dependent_feature_list])
                else:
                    x_ts = np.asarray(lc[time_dependent_feature_list])
            except Exception:
                x_ts = np.array(lc[time_dependent_feature_list])
        else:
            x_ts = np.array(lc)

        plotter = ELAsTiCC_plotter()
        img = plotter.get_lc_plots(x_ts)

        # Display inline in notebooks: convert returned image tensor/array and show with matplotlib
        try:
            import matplotlib.pyplot as plt
            try:
                import torch
            except Exception:
                torch = None

            if torch is not None and isinstance(img, torch.Tensor):
                img_np = img.permute(1, 2, 0).numpy().astype(int)
            else:
                img_np = np.asarray(img)

            plt.figure(figsize=(4, 4))
            plt.imshow(img_np)
            plt.axis('off')
            plt.show()
        except Exception:
            # If matplotlib isn't available or display fails, silently continue.
            pass

        return img
    
    def fit(self, plot_samples=False) -> pd.DataFrame:
        lc = self.photometry if type(self.photometry) == pd.DataFrame else pd.Series()
        
        if plot_samples and type(lc) == pd.DataFrame:
            self._plot_sample_lc(lc)
        
        time_series_features = pd.DataFrame(columns=time_dependent_feature_list)
        float64_features = set(['FLUXCAL', 'FLUXCALERR', 'MJD'])
        for feature_name in time_dependent_feature_list:
            # ensure we store plain Python lists (not pandas.Series) so Arrow/type
            # inference used by polars/pyarrow won't try to interpret a Series object
            if feature_name in float64_features:
                time_series_features[feature_name] = [lc[feature_name].astype("float64").tolist()]
            else:
                time_series_features[feature_name] = [lc[feature_name].tolist()]
        
        static_feature_list = self._get_static_features()
        # Build a single-row DataFrame for static features so values appear
        # on the same row as time-series features (avoid empty-frame alignment)
        static_values = {}
        for feature_name in static_feature_list:
            # use .get to avoid KeyError if feature missing
            static_values[feature_name] = self.additional_info.get(feature_name, None)
        static_features = pd.DataFrame([static_values], columns=static_feature_list)
        return pd.concat([time_series_features.reset_index(drop=True), static_features.reset_index(drop=True)], axis=1)

    def fit_all(self, parquet_path='../TOM_training_features.parquet', plot_samples=False):
        # write features to intermediate parquet file that ORACLE can ingest
        features_df = self.fit(plot_samples=plot_samples)
        self.features = features_df.iloc[0].tolist()

    def get_features_to_write(self):
        features_list = [self.id, self.redshift, self.sntype, self.sncode]
        features_list.extend(self.features)
        return features_list