import copy
import numpy as np


def normalizeFrames(wavelet_data):   
    normalized_wavelet_data = copy.deepcopy(wavelet_data)
    
    for data in normalized_wavelet_data.values():
        for t in range(data.shape[0]):
            data[t,:] = data[t,:]/(data[t,:].sum())
    
    return normalized_wavelet_data