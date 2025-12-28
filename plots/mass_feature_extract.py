import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '../src')))

import numpy as np
import polars as pl
import pandas as pd
from resspect.fit_lightcurves import fit, fit_TOM

from oracle.constants import ELAsTiCC_to_Astrophysical_mappings

from oracle_resspect_classifier.elasticc2_oracle_feature_extractor import ELAsTiCC2_ORACLEFeatureExtractor
from oracle_resspect_classifier.oracle_classifier import OracleClassifier

nersc_parquet_path = '/global/cfs/cdirs/desc-td/ELASTICC2_TRAIN02_parquet/'
# class_parquet_names = os.listdir(nersc_parquet_path)
class_parquet_names = ELAsTiCC_to_Astrophysical_mappings.keys()
class_parquet_names = [file_name + '.parquet' for file_name in class_parquet_names]

additional_features = ELAsTiCC2_ORACLEFeatureExtractor._get_static_features()
feature_extraction_method = 'oracle_resspect_classifier.elasticc2_oracle_feature_extractor.ELAsTiCC2_ORACLEFeatureExtractor'
per_class_object_counts = 1000

intermediate_parquet_path = '/pscratch/sd/a/arjun15/intermediate_TOM_training_features.parquet'
final_parquet_path = '/pscratch/sd/a/arjun15/parquets/final_TOM_training_features'

sncode_to_class = {
    10: 'SNIa', 110: 'SNIa', 25: 'SNIb/c', 37: 'SNII', 12: 'SNIax', 11: 'SN91bg', 50: 'KN', 82: 'M-dwarf Flare', 84: 'Dwarf Novae', 88: 'uLens', 
    40: 'SLSN', 42: 'TDE', 45: 'ILOT', 46: 'CART', 59: 'PISN', 90: 'Cepheid', 80: 'RR Lyrae', 91: 'Delta Scuti', 83: 'EB', 60: 'AGN', 87: 'uLens',
    32: 'SNII', 31: 'SNII', 35: 'SNII', 36: 'SNII', 21: 'SNIb/c', 20: 'SNIb/c', 72: 'SLSN', 27: 'SNIb/c', 26: 'SNIb/c', 146: 'CART', 150: 'KN',
    142: 'TDE', 127: 'SNIb/c', 151: 'KN', 189: 'uLens', 125: 'SNIb/c', 120: 'SNIb/c', 132: 'SNII', 111: 'SNIa', 145: 'ILOT', 160: 'AGN', 137: 'SNII',
    121: 'SNIb/c', 130: 'SNII', 136: 'SNII', 131: 'SNII', 140: 'SLSN', 159: 'PISN'
}

def get_phot_from_parquet(parquet_rows):    
    
    data = []
    for idx, obj in enumerate(parquet_rows.iter_rows(named=True)):        
        phot_d = {}
        phot_d['objectid'] = int(obj['SNID'])
        phot_d['sncode'] = obj['SNTYPE']
        # phot_d['sncode'] = class_to_sncode[obj['ELASTICC_class']]
        phot_d['redshift'] = obj['REDSHIFT_FINAL']
        phot_d['RA'] = obj['RA']
        phot_d['DEC'] = obj['DEC']
        
        phot_d['photometry'] = {}
        phot_d['photometry']['BAND'] = obj['BAND']
        phot_d['photometry']['MJD'] = obj['MJD']
        phot_d['photometry']['FLUXCAL'] = obj['FLUXCAL']
        phot_d['photometry']['FLUXCALERR'] = obj['FLUXCALERR']
        phot_d['photometry']['PHOTFLAG'] = obj['PHOTFLAG']
        
        phot_d['additional_info'] = {}
        
        for feature in additional_features:
            phot_d[feature] = obj[feature]
        
        data.append(phot_d)
        
    return data

def sort_by_mjd(row):
    zipped = list(zip(row["MJD"], row["FLUXCAL"], row["FLUXCALERR"], row["ZEROPT"], row["PHOTFLAG"], row["BAND"]))
    sorted_zip = sorted(zipped, key=lambda x: x[0])  # sort by MJD
    mjd, fluxcal, fluxcalerr, zeropt, photflag, band = zip(*sorted_zip)
    return {
        "sorted_MJD": list(mjd),
        "sorted_FLUXCAL": list(fluxcal),
        "sorted_FLUXCALERR": list(fluxcalerr),
        "sorted_ZEROPT": list(zeropt),
        "sorted_PHOTFLAG": list(photflag),
        "sorted_BAND": list(band),
    }

def sort_pandas_by_mjd(df):
    """Apply MJD sorting to pandas dataframe list columns."""
    for col in ['MJD', 'FLUXCAL', 'FLUXCALERR', 'ZEROPT', 'PHOTFLAG', 'BAND']:
        if col not in df.columns:
            continue
    
    sorted_data = []
    for idx, row in df.iterrows():
        zipped = list(zip(row["MJD"], row["FLUXCAL"], row["FLUXCALERR"], row["PHOTFLAG"], row["BAND"]))
        sorted_zip = sorted(zipped, key=lambda x: x[0])
        mjd, fluxcal, fluxcalerr, photflag, band = zip(*sorted_zip)
        
        row_dict = row.to_dict()
        row_dict["MJD"] = list(mjd)
        row_dict["FLUXCAL"] = list(fluxcal)
        row_dict["FLUXCALERR"] = list(fluxcalerr)
        row_dict["PHOTFLAG"] = list(photflag)
        row_dict["BAND"] = list(band)
        sorted_data.append(row_dict)
    
    return pd.DataFrame(sorted_data)

def main():
    for obj_type in class_parquet_names:        
        print(f'Processing data for class {obj_type}...')
        
        input_parquet = pl.scan_parquet(os.path.join(nersc_parquet_path, obj_type)).fetch(per_class_object_counts)
        sorted_input_parquet = input_parquet.with_columns([
            pl.struct(["MJD", "FLUXCAL", "FLUXCALERR", "ZEROPT", "PHOTFLAG", "BAND"]).map_elements(sort_by_mjd).alias("sorted")
        ]).unnest("sorted")
        
        data_dic = get_phot_from_parquet(sorted_input_parquet)
        del input_parquet, sorted_input_parquet
        
        # perform feature extraction for that specific class
        fit(
            data_dic,
            output_features_file = intermediate_parquet_path,
            feature_extractor = feature_extraction_method,
            filters = 'LSST',
            additional_info = additional_features,
            # one_code = gentypes
        )
        
        data = pd.read_parquet(intermediate_parquet_path)
        data['orig_sample'] = 'train'
        data['type'] = np.where((data['sncode'] == 10) | (data['sncode'] == 110), 'Ia', 'non-Ia')
        # data['ELASTICC_class'] = sncode_to_class[int(data['sncode'].iloc[0])]
        data['ELASTICC_class'] = obj_type[:-8]
        
        # Apply sorting to the final dataframe
        data = sort_pandas_by_mjd(data)
        
        save_path = final_parquet_path + '_' + obj_type
        
        data.to_parquet(save_path, index=False)
        
if __name__ == '__main__':
    main()