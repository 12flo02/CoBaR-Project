import numpy as np
import pickle
import pandas as pd


def load_data(xp):
    '''
    Load data from an experiment (xp)
    
    Returns
    ------
    gen_dict
        General info on data (on/off periods, collisions...)
    
    data
        Raw data
    
    metadata
        Raw metadata
    '''
    
    # Load gendict
    genDict = np.load(f'../CoBar-Dataset/{xp}/U3_f/genotype_dict.npy', allow_pickle=True).item()
    
    # Load data
    with open(f'../CoBar-Dataset/{xp}/U3_f/{xp}_U3_f_trackingData.pkl', 'rb') as f:
        data = pickle.load(f)
    
    print(f'{xp} - Data dimension: {data.shape}')
    
    # Extract metadata
    metadata = np.array([list(item) for item in data.index.values])
    print(f'{xp} - Metadata dimension: {metadata.shape}')
    
    return genDict, data, metadata