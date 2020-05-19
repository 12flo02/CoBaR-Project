import os
import numpy as np
from cluster import buildColorMap, buildJointPlot, buildKDEPlots
from utils import get_data_per_fly_per_xp
from runPipeline import runPipeline


def run(embedded_file = None, groupByFly=False, decimate=False, pca=False):
    '''
    embedded_file
        Filename of a .npy array containing results from TSNE
        TSNE is the longest step of the pipeline, so user can pre-load TSNE results if available
    
    average
        If true, xy data was averaged between flies of a same strain before wavelet transform
    
    groupByFly
        If true, data is of dimension Nx(L*K), N = nb of flies, L = nb of frames, K = nb of features
        If true, it does not make sense to plot kernel density estimation as there are far less data points
        If false, data is of dimension (N*L)xK
    '''
    
    # Retrieve folders for transgenic strains
    transgenics = os.listdir('../CoBar-Dataset')
    transgenics.remove('PR') # Discard control from meta-analysis
    
    raw_pretarsi_data, raw_metadata, _, n_trial_data = get_data_per_fly_per_xp(transgenics)
    
    if embedded_file is None:
        # Run processing pipeline
        embedded_array, min_nFrames = runPipeline(raw_pretarsi_data, raw_metadata, n_trial_data,
                                     groupByFly=groupByFly, decimate=decimate, pca=pca)
    else:
        embedded_array = np.load(embedded_file)
        min_nFrames = 704  # Minimum number of frames found during a stimulation
        
    # Build color maps
    classes, strains, unique_classes = buildColorMap(n_trial_data, min_nFrames, groupByFly=groupByFly)
    
    # Build joint scatter plot
    buildJointPlot(embedded_array, n_trial_data, classes, strains, unique_classes)

    if not(groupByFly):
        # Build kernel density estimation plots
        buildKDEPlots(embedded_array, n_trial_data, min_nFrames)