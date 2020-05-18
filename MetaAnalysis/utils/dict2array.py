import numpy as np
from scipy import signal


def dict2array(stim_data, nCoords=2, groupByFly = False, decimate = False):
    '''
    Build array from dictionary data containg stimulation xy positions for each fly
    
    stim_data
        Dictionary of "on" xy data for each fly
    
    nCoords
        Parameter to adjust size of matrix
        nCoords = 2 if no wavelet transformation is applied
        nCoords = 40 otherwise
    
    groupByFly
        If true, resulting dataset is grouped by fly. Dimensions: nFlies x (nFrames * nDimensions)
        Otherwise, group dataset as: (nFlies * nFrames) * nDimensions
    
    Returns
    stim_array
        Transformation of stim_data into a numpy array: 
        if groupByFly: size Nx(l*K)
        else: size (N*l)xK
        
        N: number of flies
        l: number of on-stimulation frames
        K: number of features
    '''
    
    nLegs = 6
    min_nFrames = stim_data[min(stim_data, key=lambda x: stim_data[x].shape[0])].shape[0]
    
    stim_array = np.zeros((len(stim_data), min_nFrames*nLegs*nCoords))
    
    for i, (key, data) in enumerate(stim_data.items()):
        stim_array[i,:] = data[:min_nFrames,:].flatten()
       
    if not(groupByFly):
        # Dataset dimension: (N*L)*K
        stim_array = stim_array.reshape(len(stim_data)*min_nFrames, nLegs*nCoords)
        
        if decimate:
            stim_array = signal.decimate(stim_array, 2, axis=0, ftype='iir')
        
    return stim_array, min_nFrames