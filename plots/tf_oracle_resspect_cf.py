import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '../src')))

import numpy as np
import polars as pl
import pandas as pd
from resspect.fit_lightcurves import fit, fit_TOM
from functools import partial
import csv

from astropy.table import Table
from astropy import units as u
from astropy.coordinates import SkyCoord
from collections import OrderedDict
from astroOracle.taxonomy import get_classification_labels, get_astrophysical_class, plot_colored_tree
from astroOracle.LSST_Source import LSST_Source

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from astroOracle.pretrained_models import ORACLE
from astroOracle.dataloader import LSSTSourceDataSet
import astroOracle.taxonomy as taxonomy

from typing import List
from tqdm import tqdm
from astroOracle.dataloader import get_static_features, ts_length, ts_flag_value
from tensorflow.keras.utils import pad_sequences
import pandas as pd

nersc_parquet_path = '/global/cfs/cdirs/desc-td/ELASTICC2_TRAIN02_parquet/'
# class_parquet_names = os.listdir(nersc_parquet_path)
# class_parquet_names = ELAsTiCC_to_Astrophysical_mappings.keys()
class_parquet_names = ['SNII-NMF', 'SNIc-Templates', 'CART', 'EB', 'SNIc+HostXT_V19', 'd-Sct', 'SNIb-Templates', 'SNIIb+HostXT_V19',
                       'SNIcBL+HostXT_V19', 'CLAGN', 'PISN', 'Cepheid', 'TDE', 'SNIa-91bg', 'SLSN-I+host', 'SNIIn-MOSFIT', 'SNII+HostXT_V19',
                       'SLSN-I_no_host', 'SNII-Templates', 'SNIax', 'SNIa-SALT3', 'KN_K17', 'SNIIn+HostXT_V19', 'dwarf-nova', 
                       'uLens-Binary', 'RRL', 'Mdwarf-flare', 'ILOT', 'KN_B19', 'uLens-Single-GenLens', 'SNIb+HostXT_V19', 'uLens-Single_PyLIMA']

per_class_object_counts = 1000

final_parquet_path = '/pscratch/sd/a/arjun15/parquets/final_TOM_training_features'
output_csv_file = 'tf_cf_classification_results_orig_classes.csv'

class ParquetDataSet(LSSTSourceDataSet):
    def __init__(self, file):
        """
        Arguments:
            path (string): Parquet file.
            transform (callable, optional): Optional transform to be applied on a sample.
        """

        # print(f'Loading parquet dataset: {path}', flush=True)

        # self.path = path
        self.parquet = pl.read_parquet(file)
        self.num_sample = self.parquet.shape[0]

        print(f"Number of sources: {self.num_sample}")
    
    def get_item(self, idx):
        
        row = self.parquet[idx]
        source = LSST_Source(row)
        table = source.get_event_table()

        astrophysical_class = taxonomy.get_astrophysical_class(source.ELASTICC_class)
        _, class_labels = taxonomy.get_classification_labels(astrophysical_class)
        class_labels = np.array(class_labels)
        snid = source.SNID

        return source, class_labels, snid

    def get_item_from_snid(self, snid):
        row = self.parquet.filter(pl.col('SNID') == snid)
        # row = self.pandas.loc[self.pandas['SNID'] == snid]
        # print(row)
        source = LSST_Source(row)
        table = source.get_event_table()

        astrophysical_class = taxonomy.get_astrophysical_class(source.ELASTICC_class)
        _, class_labels = taxonomy.get_classification_labels(astrophysical_class)
        class_labels = np.array(class_labels)
        snid = source.SNID

        return source, class_labels, snid

class nonscaledORACLE(ORACLE):
    def prep_dataframes(self, x_ts_list:List[pd.DataFrame]):

        # Assert that columns names are correct

        augmented_arrays = []

        for ind in range(len(x_ts_list)):

            df = x_ts_list[ind]

            # Scale the flux and flux error values
            # df['scaled_FLUXCAL'] = df['FLUXCAL'] / flux_scaling_const
            # df['scaled_FLUXCALERR'] = df['FLUXCALERR']/ flux_scaling_const

            # Subtract off the time of first observation and divide by scale factor
            # df['scaled_time_since_first_obs'] = df['MJD'] / time_scaling_const

            # Remove saturations
            # saturation_mask = np.where((df['PHOTFLAG'] & 1024) == 0)[0]
            # df = df.iloc[saturation_mask].copy()

            # 1 if it was a detection, zero otherwise
            # df.loc[:,'detection_flag'] = np.where((df['PHOTFLAG'] & 4096 != 0), 1, 0)

            # Encode pass band information correctly 
            # df['band_label'] = [pb_wavelengths[pb] for pb in df['BAND']]
            
            # df = df[['scaled_time_since_first_obs', 'detection_flag', 'scaled_FLUXCAL', 'scaled_FLUXCALERR', 'band_label']]
            
            # Truncate array if too long
            arr = df.to_numpy()
            if arr.shape[0]>ts_length:
                arr = arr[:ts_length, :]

            augmented_arrays.append(arr)
            
        augmented_arrays = pad_sequences(augmented_arrays, maxlen=ts_length,  dtype='float32', padding='post', value=ts_flag_value)

        return augmented_arrays

def main():    
    total_preds = []
    total_corrects = []
    
    for obj_type in class_parquet_names:
        print(f'Processing data for class {obj_type}...')
        
        read_path = final_parquet_path + '_' + obj_type + '.parquet'
                
        oracle_class_name = taxonomy.class_map[obj_type]
        
        # print(data.iloc[0]) # printing out a sample from that class just for manual inspection

        ds = ParquetDataSet(read_path)

        model = nonscaledORACLE(model_path='/global/homes/a/arjun15/ELAsTiCC-Classification/models/lsst_alpha_0.5/best_model.h5')
        
        predicted_classes = []

        print(f'\nRunning inference on {obj_type} objects...')
        for i in tqdm(range(ds.get_len())):
            source, class_labels, snid = ds.get_item(i)
            table = source.get_event_table()
            pred = model.predict_classes([table.to_pandas()], [table.meta])[0]
            # if source.SNID == 487573:
            #     print(f'============Predicted class for SNID {source.SNID} = {pred}============================')
            #     table.write('values.ecsv', overwrite=True)
            #     print(table.meta)
            #     return
            
            predicted_classes.append(pred)
            
            # for plotting confusion matrix later on
            total_preds.append(pred)
            total_corrects.append(oracle_class_name)
            
        num_correct_preds = predicted_classes.count(oracle_class_name)
        
        print('Writing results to output csv file...')
        with open(output_csv_file, 'a', newline='') as output_csv:
            writer = csv.writer(output_csv)
            results = [obj_type, per_class_object_counts, num_correct_preds]
            writer.writerow(results)
            
        print('\n')
    
    # Generate confusion matrix using sklearn
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
        
    print('Generating confusion matrix...')
    cm = confusion_matrix(total_corrects, total_preds)
    
    # Create and display the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(total_corrects))
    disp.plot(cmap='Blues', values_format='d')
    plt.title('Classification Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
    print('Confusion matrix saved as confusion_matrix.png')
    

if __name__ == '__main__':
    main()