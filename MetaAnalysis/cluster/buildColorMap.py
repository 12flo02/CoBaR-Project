import seaborn as sns
import numpy as np


def buildColorMap(n_trial_data, min_nFrames, groupByFly = False):
    # Extract classes for each fly (1 = MDN, 2 = SS01049...)
    classes = [] # List of colours attributed to each strain for later plotting
    strains = [] # Strain corresponding to each fly
    unique_classes = [] # Unique version of 'classes'
    
    if groupByFly:
        nObservationsPerStrain = list(n_trial_data.values())
    else:
        nObservationsPerStrain = [val*min_nFrames for val in n_trial_data.values()]

    for i, key in enumerate(n_trial_data.keys()):
        unique_classes.append(sns.color_palette()[i])

        for j in range(nObservationsPerStrain[i]):
            classes.append(sns.color_palette()[i])
            strains.append(key)
    
    classes = np.array(classes)
    strains = np.array(strains)
    unique_classes = np.array(unique_classes)    
    
    return classes, strains, unique_classes