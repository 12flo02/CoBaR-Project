import numpy as np

def findOnPeriods(key, raw_metadata, display=False):
    '''
    Find On-Stimulation periods for a given experiment
    
    Parameters
    ------
    key
        Experiment key
    
    raw_metadata
        Raw metadata for each fly in each experiment
    
    display
        If true, displays on-stimulation intervals
    
    Returns
    ------
    numpy.ndarray
        All frames where the stimulation was ON
    '''
    
    metadata = raw_metadata[key]
    on_periods = ['on0', 'on1', 'on2']

    on_intervals = []

    for p in on_periods:
        start_period = np.where(metadata[:,1] == p)[0][0]
        end_period = np.where(metadata[:,1] == p)[0][-1]
        on_intervals.extend(list(range(start_period, end_period)))
        if display:
            print(f'{p}: {[start_period, end_period]}')
    return np.array(on_intervals)