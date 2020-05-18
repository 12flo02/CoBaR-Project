import numpy as np
from scipy import signal


def findWavelets(stim_data):
    '''
    Apply wavelet transformation
    
    We used 20 frequencies from 1 to 40 Hz, 40 being the Nyquist frequency (Frame rate: 80 fps)
    
    Parameters
    ------
    stim_data
        A dictionary with raw data for each fly
        key = experiment + fly index (according to tracking video)
    
    Returns
    ------
    wavelet_data
        A dictionary with wavelet transform of data for each fly
    '''
    
    wavelet_data = {}

    n_scales = 20 # Number of frequencies used for wavelet transformation
    fps = 80 # Frame rate
    f_min = 1 # Minimum frequency for wavelet transform
    f_max = fps/2 # Nyquist frequency

    for (key, data) in stim_data.items():
        wavelet_data[key] = np.zeros((list(data.shape) + [n_scales]))

        # Wavelet-transformation for each feature
        for i in range(data.shape[1]):
            # Take amplitude of wavelet
            sig = abs(signal.cwt(data[:,i], signal.morlet2, np.geomspace(f_min, f_max, n_scales)).T)

            wavelet_data[key][:,i,:] = sig
        
        wavelet_data[key] = wavelet_data[key].reshape(wavelet_data[key].shape[0],
                                                      wavelet_data[key].shape[1]*wavelet_data[key].shape[2])
    return wavelet_data