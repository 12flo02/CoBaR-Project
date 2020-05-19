import matplotlib.pyplot as plt
import seaborn as sns


def buildKDEPlots(embedded_array, n_trial_data, min_nFrames):
    start = 0
    for i, (strain, nFlies) in enumerate(n_trial_data.items()):
        print(f'{i+1}/{len(n_trial_data)}')
        
        nPoints = min_nFrames*nFlies
        
        range_i = range(start, start+nPoints)
        
        # Get TSNE data for current strain
        embedded_strain = embedded_array[range_i,:]
        
        # Create figure
        plt.figure()
        # Plot general joint distribution
        g = sns.JointGrid(embedded_array[:,0], embedded_array[:,1])
        g.plot_joint(sns.kdeplot)

        # Show distribution of given strain
        sns.kdeplot(embedded_strain[:,0], embedded_strain[:,1], shade=True, cmap="Reds")

        # Save figure
        g.fig.suptitle(strain)
        plt.savefig('kde_' + strain + '.png')
        
        # Update range of points to be taken for next strain
        start += nPoints