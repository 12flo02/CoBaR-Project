from . import findOnPeriods
import numpy as np

def findStimulationData(raw_pretarsi_data, raw_metadata):
    '''
    Extract data where the stimulation was ON
    
    raw_pretarsi_data
        Raw xy data for each pretarsus, size Lx12 for each fly
    
    raw_metadata
        Corresponding metadata, size Lx6 for each fly
    
    Returns
    ------
    stim_data
        XY stimulation data for each pretarsus, size lx12 for each fly, l < L
    '''
    stim_data = {}
    
    for key, data in raw_pretarsi_data.items():
        on_idxs = findOnPeriods(key, raw_metadata)
        
        stim_data[key] = data[on_idxs,:]
        
    return stim_data