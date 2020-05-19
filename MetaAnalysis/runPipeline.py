import numpy as np
from sklearn.decomposition import PCA
from cluster import runTSNE
from utils import findStimulationData, dict2array
from wavelet import findWavelets, normalizeFrames


def runPipeline(raw_pretarsi_data, raw_metadata, n_trial_data, 
                groupByFly=False, decimate=False, pca=False):
    print('Wavelet transformation...')
    wavelet_data = findWavelets(raw_pretarsi_data)
    print('Wavelet transform: Done')

    normalized_wavelet_data = normalizeFrames(wavelet_data)
    print('Frame normalization: Done')
    
    stim_data = findStimulationData(normalized_wavelet_data, raw_metadata)
    print(f'Stimulation data - number of flies: {len(stim_data)}')
    print(f'Dimension of an observation {stim_data[next(iter(stim_data))].shape} \n')
    
    normalized_wavelet_array, min_nFrames = dict2array(stim_data, nCoords=40, 
                                                   groupByFly=groupByFly, decimate=decimate)
    print(f'Conversion to array - data shape: {normalized_wavelet_array.shape} \n')
    
    if pca:
        print('PCA decomposition...')
        normalized_wavelet_array_array = PCA(n_components=0.9, random_state=42).fit_transform(normalized_wavelet_array)
        print(f'Number of features kept to explain 90% of variance: {normalized_wavelet_array.shape[1]} \n')
        
    print('Dimensionality reduction by TSNE...')
    embedded_array = runTSNE(normalized_wavelet_array, groupByFly = groupByFly)
    np.save('embedded.npy', embedded_array)
    print(f'TSNE - data shape: {embedded_array.shape} \n')
    
    return embedded_array, min_nFrames