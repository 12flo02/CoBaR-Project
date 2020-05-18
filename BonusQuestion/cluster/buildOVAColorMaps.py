import numpy as np


def buildOVAColorMaps(n_trial_data, classes, min_nFrames, groupByFly = False):

    ova_classes = []
    ranges = []

    start = 0

    for i, val in enumerate(n_trial_data.values()):
        if groupByFly:
            nPoints = val
        else:
            nPoints = min_nFrames*val
        
        range_i = range(start, start+nPoints)
        
        ova_class = [classes[j,:] if j in range_i else np.zeros((3,)) for j in range(len(classes))]
        ova_classes.append(ova_class)

        start += nPoints

        ranges.append(range_i)
    
    return np.array(ova_classes), ranges