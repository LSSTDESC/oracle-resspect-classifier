import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '../src')))

import random
import numpy as np
import polars as pl
import pandas as pd
from typing import Literal, Dict
from astropy.table import Table
from resspect.fit_lightcurves import fit

from oracle_resspect_classifier.elasticc2_oracle_feature_extractor import ELAsTiCC2_ORACLEFeatureExtractor
from oracle_resspect_classifier.oracle_classifier import OracleClassifier

nersc_parquet_files = '/global/cfs/cdirs/desc-td/ELASTICC2_TRAIN02_parquet'
feature_extraction_method = 'oracle_resspect_classifier.elasticc2_oracle_feature_extractor.ELAsTiCC2_ORACLEFeatureExtractor'
sncode_to_class = {
    10: 'SNIa', 110: 'SNIa', 25: 'SNIb/c', 37: 'SNII', 12: 'SNIax', 11: 'SNI91bg', 50: 'KN', 82: 'M-dwarf Flare', 84: 'Dwarf Novae', 88: 'uLens', 
    40: 'SLSN', 42: 'TDE', 45: 'ILOT', 46: 'CART', 59: 'PISN', 90: 'Cepheid', 80: 'RR Lyrae', 91: 'Delta Scuti', 83: 'EB', 60: 'AGN', 87: 'uLens',
    32: 'SNII', 31: 'SNII', 35: 'SNII', 36: 'SNII', 21: 'SNIb/c', 20: 'SNIb/c', 72: 'SLSN', 27: 'SNIb/c', 26: 'SNIb/c', 146: 'CART', 150: 'KN',
    142: 'TDE', 127: 'SNIb/c', 151: 'KN', 189: 'uLens', 125: 'SNIb/c', 120: 'SNIb/c', 132: 'SNII', 111: 'SNIa', 145: 'ILOT', 160: 'AGN', 137: 'SNII',
    121: 'SNIb/c', 130: 'SNII', 136: 'SNII', 131: 'SNII', 140: 'SLSN', 159: 'PISN', 111: 'SNI91bg', 126: 'SNIb/c', 112: 'SNIax', 135: 'SNII',
}
additional_features = ELAsTiCC2_ORACLEFeatureExtractor._get_static_features()

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

def polars_row_to_astropy_table(row, time_dependent_features=None, prefer_numpy=True):
    """
    Convert a single Polars row (or dict-like) into an Astropy Table where each row
    corresponds to one observation (per time-step). Non-list values are stored in
    the returned Table.meta dictionary.

    Parameters
    - row: a single-row object returned from Polars (e.g. `parquet[i]`) or a mapping/dict
    - time_dependent_features: optional list of column names to treat as time-dependent lists;
      if None the function will infer list-like columns by checking for Python `list` instances
    - prefer_numpy: if True, table columns will contain numpy arrays where possible

    Returns
    - astropy.table.Table with one row per observation and metadata in `Table.meta`
    """

    # Normalize input to a plain dict
    if hasattr(row, "to_dict"):
        try:
            row_dict = row.to_dict()
        except Exception:
            # Polars sometimes returns a single-row DataFrame; try converting via dict comprehension
            row_dict = {c: row[c] for c in row.columns}
    elif isinstance(row, dict):
        row_dict = row.copy()
    else:
        # Generic fallback for mapping-like objects
        try:
            keys = getattr(row, "keys", None)
            if callable(keys):
                row_dict = {k: row[k] for k in row.keys()}
            else:
                row_dict = dict(row)
        except Exception:
            raise TypeError("Unsupported row type for conversion to dict")

    # Detect list-like columns by checking for plain Python lists (guaranteed by user)
    if time_dependent_features is None:
        time_cols = [k for k, v in row_dict.items() if isinstance(v, np.ndarray)]
    else:
        time_cols = list(time_dependent_features)

    # Build arrays for each time-dependent column and track lengths
    arrays = {}
    lengths = set()
    for col in time_cols:
        v = row_dict.get(col)
        arr = np.asarray(v)
        arr = arr.ravel()
        arrays[col] = arr
        lengths.add(arr.shape[0])

    # If lengths differ, pad numeric arrays with np.nan and object arrays with None
    if len(lengths) == 0:
        # No time-dependent columns found -> return empty table with metadata
        tbl = Table()
        tbl.meta.update({k: (v.item() if isinstance(v, np.generic) else v) for k, v in row_dict.items()})
        return tbl

    if len(lengths) > 1:
        maxlen = max(lengths)
        for col, arr in arrays.items():
            if arr.shape[0] < maxlen:
                if np.issubdtype(arr.dtype, np.number):
                    pad = np.full((maxlen - arr.shape[0],), np.nan, dtype=arr.dtype)
                else:
                    pad = np.full((maxlen - arr.shape[0],), None, dtype=object)
                arrays[col] = np.concatenate([arr, pad])

    # Create Astropy Table and populate columns
    tbl = Table()
    for col, arr in arrays.items():
        if prefer_numpy:
            tbl[col] = np.asarray(arr)
        else:
            tbl[col] = list(arr)

    # Store non-time-dependent scalar values as metadata (convert numpy scalars to Python types)
    meta = {}
    for k, v in row_dict.items():
        if k in time_cols:
            continue
        if isinstance(v, np.generic):
            meta[k] = v.item()
        else:
            meta[k] = v

    tbl.meta.update(meta)
    return tbl

# # # PAIR OF FUNCTIONS TO GENERATE CUSTOM DATASETS FROM THE PARQUET FILES

# Helper function to partition objects into different sets
def partition_dataset_by_snid(
        per_class_object_counts: int,
        train_split: float = 0.6,
        val_split: float = 0.2,
        test_split: float = 0.1,
        pool_split: float = 0.1,
        random_seed: int = 42):
    """
    Partitions all available objects into train/val/test/pool sets based on SNID.
    
    Args:
        per_class_object_counts: Total objects per class (will be split across all sets)
        train_split: Fraction for training set
        val_split: Fraction for validation set
        test_split: Fraction for test set
        pool_split: Fraction for pool (unlabeled) set
        random_seed: Seed for reproducibility
    
    Returns:
        Dict mapping 'train', 'val', 'test', 'pool' to dict of {class_name: [snids]}
    """
    random.seed(random_seed)
    
    class_parquet_names = os.listdir(nersc_parquet_files)
    partitions = {'train': {}, 'val': {}, 'test': {}, 'pool': {}}
    
    for class_name in class_parquet_names:
        obj_data_path = os.path.join(nersc_parquet_files, class_name)
        # Load all available SNIDs for this class
        parquet = pl.scan_parquet(obj_data_path).select(['SNID']).fetch(per_class_object_counts)
        snids = parquet['SNID'].to_list()
        
        # Shuffle SNIDs deterministically
        random.shuffle(snids)
        
        # Calculate split indices
        n = len(snids)
        train_idx = int(n * train_split)
        val_idx = train_idx + int(n * val_split)
        test_idx = val_idx + int(n * test_split)
        
        # Partition SNIDs
        partitions['train'][class_name] = snids[:train_idx]
        partitions['val'][class_name] = snids[train_idx:val_idx]
        partitions['test'][class_name] = snids[val_idx:test_idx]
        partitions['pool'][class_name] = snids[test_idx:]
        
        print(f"{class_name}: train={len(partitions['train'][class_name])}, "
              f"val={len(partitions['val'][class_name])}, "
              f"test={len(partitions['test'][class_name])}, "
              f"pool={len(partitions['pool'][class_name])}")
    
    return partitions

# Updated generate_dataset function that accepts pre-partitioned SNIDs
def generate_dataset(
        dataset_type: Literal["train", "val", "test", "pool"],
        snid_partitions: Dict[str, list],  # Dict of {class_name: [snids]} for this dataset
        final_dataset_file: str):
    """
    Generates a dataset from pre-partitioned SNIDs with per-class feature extraction.
    Processes each class individually, then concatenates results.
    
    Args:
        dataset_type: Type of dataset being created
        snid_partitions: Dict mapping class names to lists of SNIDs for this dataset
        final_dataset_file: Path to save final processed dataset
    """
    all_class_data = []
    additional_features = ELAsTiCC2_ORACLEFeatureExtractor._get_static_features()
    
    for class_name, snids in snid_partitions.items():
        if not snids:  # Skip empty classes
            print(f"Skipping {class_name} (no objects in this partition)")
            continue
        
        print(f"\nProcessing {class_name} ({len(snids)} objects)...")
        
        # Load class parquet and filter to SNIDs in this partition
        obj_data_path = os.path.join(nersc_parquet_files, class_name)
        class_parquet = pl.scan_parquet(obj_data_path).filter(
            pl.col('SNID').is_in(snids)
        ).collect()
        
        # Feature extract for this class
        data_dic = get_phot_from_parquet(class_parquet)
        intermediate_features_file = f'./datasets/intermediate_{dataset_type}_{class_name}.parquet'
        
        fit(
            data_dic,
            output_features_file = intermediate_features_file,
            feature_extractor = feature_extraction_method,
            filters = 'LSST',
            additional_info = additional_features,
        )
        
        # Read, process, and sort this class's data
        class_data = pd.read_parquet(intermediate_features_file)
        class_data['orig_sample'] = dataset_type
        class_data['type'] = np.where((class_data['sncode'] == 10) | (class_data['sncode'] == 110), 'Ia', 'non-Ia')
        class_data['ELASTICC_class'] = sncode_to_class[int(class_data['sncode'].iloc[0])]
        
        # Sort by MJD for this class
        class_data = sort_pandas_by_mjd(class_data)
        
        all_class_data.append(class_data)
        
        # Clean up intermediate file
        os.remove(intermediate_features_file)
        print(f"✓ Processed {class_name} with {len(class_data)} objects")
    
    if not all_class_data:
        print(f"Warning: No objects found for {dataset_type} set")
        return
    
    # Concatenate all classes together
    final_data = pd.concat(all_class_data, ignore_index=True)
    final_data.to_parquet(final_dataset_file, index=False)
    
    print(f"\n✓ Saved {dataset_type} set with {len(final_data)} total objects to {final_dataset_file}")

if __name__ == '__main__':
    # # Example usage (uncomment to run):
    # row = data.iloc[0]         # or example_input[0]
    # tbl = polars_row_to_astropy_table(row)
    # print(tbl.meta)
    # print(tbl)
    pass