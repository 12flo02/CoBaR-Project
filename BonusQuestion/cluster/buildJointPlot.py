import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def buildJointPlot(embedded_array, n_trial_data, classes, strains, unique_classes, usingFullData=True, fname='full'):
    g = sns.jointplot(embedded_array[:,0], embedded_array[:,1], kind="kde")

    #Clear the axes containing the scatter plot
    g.ax_joint.cla()
    g.ax_marg_x.set_axis_off()
    g.ax_marg_y.set_axis_off()

    # set the current axis to be the joint plot's axis
    plt.sca(g.ax_joint)

    # plt.scatter takes a 'c' keyword for color
    # you can also pass an array of floats and use the 'cmap' keyword to
    # convert them into a colormap

    sc = plt.scatter(embedded_array[:,0], embedded_array[:,1], c = classes, label = strains)
    
    if usingFullData:
        lp = lambda i: plt.plot([], color = unique_classes[i], mec="none",
                                label=list(n_trial_data.keys())[i], ls="", marker="o")[0]

        handles = [lp(i) for i in np.arange(len(np.unique(strains)))]
        plt.legend(handles=handles, ncol=2, prop={'size': 8})
    else:
        plt.title(fname)
    plt.tight_layout()
    plt.savefig(fname + '.png')