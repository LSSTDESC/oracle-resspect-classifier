from resspect.tom_client import TomClient
import os
import pandas as pd
import itertools

from oracle_resspect_classifier.elasticc2_oracle_feature_extractor import ELAsTiCC2_ORACLEFeatureExtractor

url = os.environ.get('TOM_URL', "https://desc-tom-2.lbl.gov")

username = os.environ.get('TOM_USERNAME', 'USER')
passwordfile = os.environ.get('TOM_PASSWORDFILE', 'FILEPATH')

def get_phot(tom: TomClient, obj_df: pd.DataFrame):    
    ids = obj_df['diaobject_id'].tolist()
    
    # using these object ids, load in static and time series data for ORACLE
    static = tom.post('db/runsqlquery/',
                  json={'query': '''SELECT diaobject_id, ra, decl, mwebv, mwebv_err, z_final, z_final_err, hostgal_zphot, hostgal_zphot_err,
                  hostgal_zspec, hostgal_zspec_err, hostgal_ra, hostgal_dec, hostgal_snsep, hostgal_ellipticity, hostgal_mag_u,
                  hostgal_mag_g, hostgal_mag_r, hostgal_mag_i, hostgal_mag_z, hostgal_mag_y FROM elasticc2_ppdbdiaobject WHERE diaobject_id IN (%s) ORDER BY diaobject_id;''' % (', '.join(str(id) for id in ids)),
                       'subdict': {}})
    static_data = static.json() if static.status_code == 200 else {'status': static.status_code}
    
    assert static_data['status'] == 'ok', 'Failed to retrieve static data'

    ts = tom.post('db/runsqlquery/',
                 json={'query': 'SELECT diaobject_id, midpointtai, filtername, psflux, psfluxerr FROM elasticc2_ppdbdiaforcedsource WHERE diaobject_id IN (%s) ORDER BY diaobject_id;' % (', '.join(str(id) for id in ids)),
                      'subdict': {}})
    ts_data = ts.json() if ts.status_code == 200 else {'status': ts.status_code}
    
    assert ts_data['status'] == 'ok', 'Failed to retrieve time series data'
    
    # for each object, sort all observations by MJD
    ts_data['rows'].sort(key=lambda obs: obs['diaobject_id'])
    grouped_ts_data = {snid: list(obj) for snid, obj in itertools.groupby(ts_data['rows'], key=lambda obs: obs['diaobject_id'])}
    
    for observation in grouped_ts_data.values():
        observation.sort(key=lambda obs: obs['midpointtai'])
        
def main():
    tom = TomClient(url=url, username=username, passwordfile=passwordfile)