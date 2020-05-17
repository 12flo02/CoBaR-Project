from .buildJointPlot import buildJointPlot


def buildStrainJointPlots(embedded_array, n_trial_data, ova_classes, strains, ranges):
    for i, key in enumerate(n_trial_data.keys()):
        print(f'{i+1}/{len(list(n_trial_data.keys()))}')
        
        strain_embedding = embedded_array[ranges[i],:]
        ova_class = ova_classes[i,ranges[i],:]
        strain = strains[ranges[i]]
        
        buildJointPlot(strain_embedding, n_trial_data, ova_class, strain, None, usingFullData=False, fname=key)
        