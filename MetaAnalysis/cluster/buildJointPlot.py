import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def buildJointPlot(embedded_array, n_trial_data, classes, strains, unique_classes, usingFullData=True, fname='full'):
    '''
    Create bivariate plot for TSNE visualization
    
    embedded_array
        2-dimensional array output from TSNE
       
    n_trial_data
        Dictionary with key = strain, value = number of flies recorded for this strain
    
    classes
        List of colors applied to each fly
    
    strains
        array containg the corresponding strain to each fly
    
    unique_classes
        List of colors applied to each strain
    
    usingFullData
        If true, displays the legend for the scatter plot (as there would be multiple strains)
        If only one strain is to be displayed, use False (the name of the used strain will appear in the title)
    
    fname
        Filename to save the plot
        "full" is left by default for the plot of the whole data
        For singular strain plots, the suggested filename is the name of the strain (e.g. 'MDN.png' for MDN)
    '''
    
    g = sns.JointGrid(embedded_array[:,0], embedded_array[:,1])
    g.plot_joint(sns.kdeplot)

    #Clear the axes containing the scatter plot
    g.ax_joint.cla()
    g.ax_marg_x.set_axis_off()
    g.ax_marg_y.set_axis_off()

    # set the current axis to be the joint plot's axis
    plt.sca(g.ax_joint)

    sc = plt.scatter(embedded_array[:,0], embedded_array[:,1], c = classes, label = strains)
    
    if usingFullData:
        lp = lambda i: plt.plot([], color = unique_classes[i], mec="none",
                                label=list(n_trial_data.keys())[i], ls="", marker="o")[0]

        handles = [lp(i) for i in np.arange(len(np.unique(strains)))]
        plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', handles=handles, ncol=2, prop={'size': 8})
    else:
        plt.title(fname)
   
    plt.show()
    plt.savefig(fname + '.png')