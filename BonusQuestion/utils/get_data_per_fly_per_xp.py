from .load_data import load_data
import numpy as np
import pandas as pd

def get_data_per_fly_per_xp(transgenics):  
    '''
    Extract data for each fly in each experiment
    
    N: number of flies
    L: number of frames
    
    Returns
    ------
    dict_pretarsi_data
        A dictionary containing xy data for each pretarsus for each fly, size Lx12 for each fly
        key = experiment + fly index (according to tracking video)
    
    dict_metadata
        Corresponding metadata, size Lx6 for each fly
        key = experiment + fly index (according to tracking video)
    
    dict_pos_data
        Centroid xy data, size Lx2 for each fly
        key = experiment + fly index (according to tracking video)
    
    n_trial_data
        Get number of flies in each experiment
        key = experiment
    '''
    
    dict_pretarsi_data = {}
    dict_metadata = {}
    dict_pos_data = {}
    n_trial_data = {}
    
    stim_col = 1 # Metadata column with stimulation info ('on'/'off')
    xp_col = 3   # Metadata column with experiment info (time of experiment)
    fly_col = 4  # Metadata column with fly info (fly 0, 1 or 2) 
    
    for strain in transgenics:
        # Load data for given strain
        genDict, data, metadata = load_data(strain)
        
        # Extract pretarsi data
        pretarsi = ["LFclaw", "LHclaw", "LMclaw", "RFclaw", "RHclaw", "RMclaw"]
        pos = ["posx", "posy"]
        orientation = ["orientation"]
        pretarsi_data = data[pretarsi]   
        pos_data = data["center"][pos + orientation]
    
        # Gather all possible experiments and maximum number of flies
        xps = np.unique(metadata[:,xp_col])
        flies = np.unique(metadata[:,fly_col])
        
        n_trials = len(xps)*len(flies)

        for xp in xps:
            # Extract rows corresponding to current experiment
            xp_idx = np.where(metadata[:,xp_col] == xp)[0]
            
            # Extract corresponding metadata, pretarsi data and positional data
            xp_metadata = metadata[xp_idx]
            xp_pretarsi_data = pretarsi_data.iloc[xp_idx]
            xp_pos_data = pos_data.iloc[xp_idx]

            for fly in flies:
                # Extract rows corresponding to current fly
                fly_idx = np.where(xp_metadata[:,fly_col] == fly)[0]
                
                # Extract corresponding metadata and data for current fly
                xp_fly_metadata = xp_metadata[fly_idx]
                xp_fly_pretarsi_data = xp_pretarsi_data.iloc[fly_idx]
                xp_fly_pos_data = xp_pos_data.iloc[fly_idx]
                
                xp_fly_metadata = np.append(xp_fly_metadata, np.array(range(len(xp_fly_metadata))).reshape(-1,1), axis=1)
                
                
                # Sort timestamps, and re-arrange fly data for time stamps
                # Order = 'off0', 'on0', 'off1', 'on1', 'off2', 'on2', 'off3'
                if not(xp_fly_pretarsi_data.empty):
                    
                    dict_metadata[xp+fly] = np.array(sorted(xp_fly_metadata, key=lambda x: (int(x[stim_col][-1]), x[stim_col])))
                    idx_sort = np.array(list(map(int, dict_metadata[xp+fly][:,-1])))
                    dict_pretarsi_data[xp + fly] = np.array(xp_fly_pretarsi_data)[idx_sort,:] * 38/832                    
                    dict_pos_data[xp + fly] = np.array(xp_fly_pos_data)[idx_sort,:]
                    
                    # Only convert x, y pos positions to mm
                    dict_pos_data[xp + fly][:,:2] *= 38/832
                else:
                    n_trials -= 1
            
        n_trial_data[strain] = n_trials
        
        print(f'{strain}: {n_trials} trials')
    
    return dict_pretarsi_data, dict_metadata, dict_pos_data, n_trial_data