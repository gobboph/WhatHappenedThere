'''
(c) 2017 Roberto Gobbetti
'''

import pandas as pd
import numpy as np

def find_anom(ts, tolerance=2.0, window=30, threshold=1000):
    '''
    Function to detect anomalies
    
    :ts: pandas.Series
    :tolerance: \sigma deviation from mean after which I call for an anomaly
    :window: rolling window
    :threshold: an absolute threshold. This is useful if I am only interested in events that are big in an absolute sense
    
    :return: _ anom: pandas.Series with first days of anomaly sequence, with value
             _ norm: pandas.Series with non-anomalous days, with value
             _ anom_all: pandas.Series with all anomalies, with value
    '''

    anom_all = pd.Series()
    anom     = pd.Series()
    norm     = ts[:window]
    

    for i in range(len(ts.index)):
        
        roll_mean = norm[-30:].mean()
        roll_std  = norm[-30:].std()
        
        if (ts[ts.index[i]] > roll_mean + tolerance * roll_std) and (ts[ts.index[i]] >= threshold):
            anom_all = anom_all.set_value(ts.index[i], ts[ts.index[i]])
            if ts.index[i-1] not in anom_all.index:
                anom = anom.set_value(ts.index[i], ts[ts.index[i]])
        elif ts[ts.index[i]] < roll_mean - tolerance * roll_std:
            anom_all = anom_all.set_value(ts.index[i], ts[ts.index[i]])
            if ts.index[i-1] not in anom_all.index:                
                anom = anom.set_value(ts.index[i], ts[ts.index[i]])
        else:
            norm = norm.set_value(ts.index[i],ts[ts.index[i]])
        
    return anom, norm, anom_all


