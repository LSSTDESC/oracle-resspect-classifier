import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '../src')))

import numpy as np
import polars as pl
import pandas as pd
from resspect.fit_lightcurves import fit, fit_TOM
from oracle.custom_datasets.ELAsTiCC import ELAsTiCC_LC_Dataset, truncate_ELAsTiCC_light_curve_fractionally, custom_collate_ELAsTiCC
from functools import partial
import torch
from torch.utils.data import DataLoader
import csv
from tqdm import tqdm
import matplotlib.pyplot as plt

from oracle.constants import ELAsTiCC_to_Astrophysical_mappings

from oracle_resspect_classifier.elasticc2_oracle_feature_extractor import ELAsTiCC2_ORACLEFeatureExtractor
from oracle_resspect_classifier.oracle_classifier import OracleClassifier

torch.set_default_device('cpu')

nersc_parquet_path = '/global/cfs/cdirs/desc-td/ELASTICC2_TRAIN02_parquet/'
# class_parquet_names = os.listdir(nersc_parquet_path)
class_parquet_names = ELAsTiCC_to_Astrophysical_mappings.keys()
class_parquet_names = [file_name + '.parquet' for file_name in class_parquet_names]

additional_features = ELAsTiCC2_ORACLEFeatureExtractor._get_static_features()
feature_extraction_method = 'oracle_resspect_classifier.elasticc2_oracle_feature_extractor.ELAsTiCC2_ORACLEFeatureExtractor'
per_class_object_counts = 1000
batch_size = 100

intermediate_parquet_path = 'intermediate_TOM_training_features.parquet'
final_parquet_path = 'final_TOM_training_features.parquet'
output_csv_file = 'classification_results_orig_classes.csv'

classifier_test = OracleClassifier(dir='../', weights_dir='/pscratch/sd/a/arjun15/')

sncode_to_class = {
    10: 'SNIa', 110: 'SNIa', 25: 'SNIb/c', 37: 'SNII', 12: 'SNIax', 11: 'SNI91bg', 50: 'KN', 82: 'M-dwarf Flare', 84: 'Dwarf Novae', 88: 'uLens', 
    40: 'SLSN', 42: 'TDE', 45: 'ILOT', 46: 'CART', 59: 'PISN', 90: 'Cepheid', 80: 'RR Lyrae', 91: 'Delta Scuti', 83: 'EB', 60: 'AGN', 87: 'uLens',
    32: 'SNII', 31: 'SNII', 35: 'SNII', 36: 'SNII', 21: 'SNIb/c', 20: 'SNIb/c', 72: 'SLSN', 27: 'SNIb/c', 26: 'SNIb/c', 146: 'CART', 150: 'KN',
    142: 'TDE', 127: 'SNIb/c', 151: 'KN', 189: 'uLens', 125: 'SNIb/c', 120: 'SNIb/c', 132: 'SNII', 111: 'SNIa', 145: 'ILOT', 160: 'AGN', 137: 'SNII',
    121: 'SNIb/c', 130: 'SNII', 136: 'SNII', 131: 'SNII', 140: 'SLSN', 159: 'PISN', 111: 'SNI91bg'
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
    total_preds = []
    total_corrects = []
    
    for obj_type in class_parquet_names:
        print(f'Processing data for class {obj_type}...')
        
        input_parquet = pl.scan_parquet(os.path.join(nersc_parquet_path, obj_type)).fetch(per_class_object_counts)
        data_dic = get_phot_from_parquet(input_parquet)
        del input_parquet
        
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
        data['ELASTICC_class'] = sncode_to_class[int(data['sncode'].iloc[0])]
        
        # Apply sorting to the final dataframe
        data = sort_pandas_by_mjd(data)
        
        data.to_parquet(final_parquet_path, index=False)
        
        oracle_class_name = data['ELASTICC_class'][0]
        
        print(data.iloc[0]) # printing out a sample from that class just for manual inspection
        
        # Sample 5 SNIDs for the class
        sample_snids = data['SNID'].sample(5, random_state=42).tolist()
        
        # Get indices of the sampled SNIDs
        sample_indices = data[data['SNID'].isin(sample_snids)].index.tolist()
                
        test_dataloaders = []
        # fractions_list = np.linspace(0.1, 1, 10).round(decimals=1)
        fractions_list = [1]
        generator = torch.Generator(device='cpu')

        for f in fractions_list:
            test_data = ELAsTiCC_LC_Dataset(
                parquet_file_path=final_parquet_path,
                max_n_per_class=None,
                include_lc_plots=True,
            )
            test_data.transform = partial(truncate_ELAsTiCC_light_curve_fractionally, f=f)
            test_dl = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_ELAsTiCC, generator=generator, pin_memory=True)
            test_dataloaders.append(test_dl)

        classifier_test.model.eval()
        classifier_test.model.to('cpu')

        predicted_classes = []

        print(f'\nRunning inference on {obj_type} objects...')
        for frac, fractional_dl in zip(fractions_list, test_dataloaders):
            for index, batch in enumerate(tqdm(fractional_dl)):
                batch = {k: v.to('cpu') if torch.is_tensor(v) else v for k, v in batch.items()}
                        
                prediction_df = classifier_test.model.predict_class_probabilities_df(batch)
                predicted_classes.extend(list(prediction_df.iloc[:, 7:].idxmax(axis=1)))
                total_preds.extend(list(prediction_df.iloc[:, 7:].idxmax(axis=1)))
                total_corrects.extend([oracle_class_name] * batch_size)

        print(predicted_classes[-5:])
        print(total_preds[-5:])
        print(total_corrects[-5:])
        num_correct_preds = predicted_classes.count(oracle_class_name)
        
        # Get predicted classes for the samples
        sample_preds = [predicted_classes[i] for i in sample_indices]

        print('Writing results to output csv file...')
        with open(output_csv_file, 'a', newline='') as output_csv:
            writer = csv.writer(output_csv)
            results = [oracle_class_name, per_class_object_counts, num_correct_preds] + sample_snids + sample_preds
            writer.writerow(results)
            
        print('\n')
    
    # Generate confusion matrix using sklearn
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    
    print(np.unique(total_corrects))
    print(np.unique(total_preds))
    
    print('Generating confusion matrix...')
    cm = confusion_matrix(total_corrects, total_preds, labels=sorted(set(total_corrects) | set(total_preds)))
    
    # Normalize the confusion matrix by row (to get percentages)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create a larger figure
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Create and display the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=sorted(set(total_corrects) | set(total_preds)))
    disp.plot(cmap='Blues', values_format='.2f', ax=ax)
    
    # Set text color based on background intensity
    # Make text black for light backgrounds (values < 0.5) and white for dark backgrounds (values >= 0.5)
    for i in range(cm_normalized.shape[0]):
        for j in range(cm_normalized.shape[1]):
            text = disp.text_[i, j]
            if cm_normalized[i, j] < 0.5:
                text.set_color('black')
            else:
                text.set_color('white')
    
    plt.title('Classification Confusion Matrix (Normalized)', fontsize=16)
    plt.xticks(rotation=90, fontsize=12)
    plt.yticks(fontsize=12)
    disp.ax_.xaxis.label.set_fontsize(12)
    disp.ax_.yaxis.label.set_fontsize(12)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
    print('Confusion matrix saved as confusion_matrix.png')
    
        
if __name__ == '__main__':
    main()