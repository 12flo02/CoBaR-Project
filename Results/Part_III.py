import pickle
import pandas as pd
import numpy as np
import math
import scipy.fftpack
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import savgol_filter
from scipy import signal, fftpack, stats
from matplotlib import animation
from matplotlib.transforms import (Bbox, TransformedBbox, blended_transform_factory)
from mpl_toolkits.axes_grid1.inset_locator import (BboxPatch, BboxConnector, BboxConnectorPatch)
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms


plt.rc('lines', linewidth=1.0)
plt.rc('font', size=8.0)


class Environment():
    ### environment variables 
    ### convert the pixel width (832) in [mm] --> environment = 38[mm]x38[mm]
    def __init__(self, bool_save = False):
        self.enivronment_size = 38
        self.nb_pixel = 832
        self.position_convert = self.enivronment_size/self.nb_pixel
        self.stim_legend = ["stimulation off", "stimulation on"]
        self.color_plot = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple', 'tab:brown']
        self.color_claw = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']
        self.color_stim = ['r', 'b']
        self.figure_size = [8, 6]
        self.dpi = 600
        self.bool_save = bool_save
        self.plot_lim = [-38, 38]
        
    
class Fly_Experiment():
    
    def __init__(self, name):
        self.name = name
        
    def general_info(self, simulation = 'MDN', folder = '200206_110534_s1a10_p3-4', index_folder = 0, frame_split = [],\
                     nb_fly = 0, frame_per_fly = 0, frame_per_period = None, \
                     frame_per_period_tot = None, frame_frequency = 80): #, x_pos = None, y_pos = None) :
        
        self.simulation = simulation
        self.folder = folder
        self.split_name_fodler = self.folder[:13]
        self.frame_split = frame_split
        self.index_folder = index_folder
        self.nb_fly = nb_fly
        self.frame_per_fly = frame_per_fly
        self.frame_per_period = frame_per_period
        self.frame_per_period_tot = frame_per_period_tot
        self.total_frame = frame_per_period_tot[-1]
        self.frame_frequency = frame_frequency

    def index_order(self, index):
        self.index = index

    def position_order(self, x_pos, y_pos, orientation = None):
        self.x_pos = x_pos
        self.y_pos = y_pos
        if orientation.any() : 
            self.orientation_order(orientation)
        
    def orientation_order(self, orientation):
        self.orientation = orientation
        
    def position_n_order(self, x_pos_n, y_pos_n, orientation_n = None):
        self.x_pos_n = x_pos_n
        self.y_pos_n = y_pos_n
        if orientation_n.any() :
            self.orientation_n_order(orientation_n)
        
    def orientation_n_order(self, orientation_n):
        self.orientation_n = orientation_n

        
    def position(self, x_pos, y_pos):
        self.x = x_pos
        self.y = y_pos
        
        
#%%
def general_data(first_layer) :
    '''
    Find the correct path and return the metadata dictonnary and the data
    '''

    #find the right path for the .pkl and .npy 
    simulation = ['MDN', 'PR', 'SS01049', 'SS01054', 'SS01540', 'SS02111', 'SS02279', 'SS02377', 'SS02608', 'SS02617']
    
    data_path = ['../CoBar-Dataset/']
    data_path.append(simulation[first_layer])
    data_path.append('/U3_f/')
    data_path.append(simulation[first_layer])
    data_path.append('_U3_f_trackingData.pkl')
    data_path = ''.join(data_path)
    "output : '../CoBar-Dataset/MDN/U3_f/MDN_U3_f_trackingData.pkl'"
    
    gen_path = ['../CoBar-Dataset/']
    gen_path.append(simulation[first_layer])
    gen_path.append('/U3_f/genotype_dict.npy')
    gen_path = ''.join(gen_path)
    "output : '../CoBar-Dataset/MDN/U3_f/genotype_dict.npy'"
    

    genDict = np.load(gen_path, allow_pickle=True).item()
    
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
            
    ### store the name of the different folder in all_folder
    all_folder = []
    
    for key, _ in genDict.items() :
        all_folder.append(key)
    """ output : all_folder = ['200206_110534_s1a10_p3-4', '200206_160327_s4a9_p3-4', '200206_105311_s1a9_p3-4', '200206_153954_s4a10_p3-4']"""
  
    return genDict, data, all_folder, simulation[first_layer]

#%%
def experiment_propeties(genDict, data, all_folder, simulation, second_layer):  
    '''
    
    Parameters
    ----------
     genDict : dict
        Metadata of all experiment from a specific transgenic strain.
    data : Dataframe
        Position of the flies at each time step.
    all_folder : list
        Name of all the folder from a sptecific transgenic strain. 
        (ex : "200206_110534_s1a10_p3-4")
    simulation : str
        Nome of the transgenic strain. 
        (ex : "MDN")
    second_layer : int
        Number between 0 and 4, extract the data of this folder.
        (ex : "all_folder[second_layer]")

    Returns
    -------
    experiment : Fly _Experiment
        store te general info of an experiment.
        (ex : experiment.folder, experiment.nb_fly, experiment.total_frame, ...)

    '''
    
    ### pick one specific folder
    folder = all_folder[second_layer]
    split_name_fodler = folder[:13]
    """output : '200206_110534'"""
    
    ### Find the first element of the folder (experiment) in the DataFrame
    index_folder = np.where(np.array(data.reset_index().iloc[:,3]) == split_name_fodler)[0][0]
     
    ###  extract the nuber of frames per video and per sequence on/off
    frame_split = genDict[folder]['stimulation_paradigm']
    frame_per_period = []
    frame_per_period_tot = [0]
    third_layer = [] 
    
    for key, _ in genDict[folder].items() :
            third_layer.append(key) 
            """ third_layer[2] is always the first fly, could be fly1 if fly0 doesn't exist"""
        
    for _, split in enumerate(frame_split) :
        frame_per_period.append(genDict[folder][third_layer[2]][split]['nb frames'])
        frame_per_period_tot.append(sum(frame_per_period))
    """ output : frame_per_period     = [38, 234, 793, 240, 798, 240, 47]
        output : frame_per_period_tot = [0, 38, 272, 1065, 1305, 2103, 2343, 2390]"""
    
    
    ### number of fly and frame per fly
    nb_fly = len(genDict[folder].keys()) - 2
    frame_per_fly = frame_per_period_tot[-1]    
    frame_frequency = genDict[all_folder[second_layer]]['fps']
    
    ### insert all values in a class
    experiment = Fly_Experiment(second_layer)
    experiment.general_info(simulation, folder, index_folder, frame_split, nb_fly, \
                            frame_per_fly, frame_per_period, \
                            frame_per_period_tot, frame_frequency)
        
    return experiment

#%%
def any_coordinates(experiment, x_coordinate, y_coordinate):
    '''
    Extract the desired coordinates from the data
    '''
        
    x_pos = x_coordinate.values[experiment.index] * environment.position_convert
    y_pos = y_coordinate.values[experiment.index] * environment.position_convert
       
    return x_pos, y_pos


def angle_coordinates(experiment, theta_coordinate):
    '''
    Extract the desired coordinates from the data
    '''
    
    theta_pos = theta_coordinate.values[experiment.index]
    
    return theta_pos


def adjust_angular_displacement(dtheta):
    for i in range(dtheta.shape[0]):
        for j in range(dtheta.shape[1]):
            if dtheta[i][j] > 300:
                dtheta[i][j] = dtheta[i][j] - 360
            elif dtheta[i][j] < -300 :
                dtheta[i][j] = dtheta[i][j] + 360
            
    return dtheta

def pos2vel(x, fps=80, angular=False, number_fly = 3):
    '''
    Convert positions to velocities, and apply moving-average filter
    '''
    dx = np.array([x[:, i] - x[:, i-1] for i in range(1, x.shape[1])]).T
    
    if angular:
        dx = adjust_angular_displacement(dx)
        # Convert to radians
        dx *= np.pi/180
        
    dx = np.hstack((np.zeros((number_fly,1)), dx))
    velocity = np.zeros(x.shape)
    
    for i in range(len(x)):
        velocity[i,:] =  SMA(dx[i,:]*fps)
        
    return velocity


def SMA_pos(x, angular=False, number_fly = 3):
    '''
    Calculate the backward difference and the Moving-average filter
    '''
    
    return pos2vel(x, fps=1, angular=angular, number_fly=number_fly)

    
def cart2pol(x,y):
    '''
    Convert Cartesian to Polar coordinates
    '''
    rho = np.sqrt(x**2+y**2)
    phi = np.arctan2(y, x)*180/np.pi
    return (rho, phi)

def SMA(x, window=5):
    '''
    Moving-average with convolution
    '''
    return np.convolve(x, np.ones((window,))/window, mode='same')

def minimal_frame(all_experiment, sequence_on = [1, 3, 5]):
    '''
    find the number of minimum frame for the same sequence
    '''
                
    min_frame_sequence_on = np.Inf * np.ones(len(sequence_on))
    for count, i in enumerate(sequence_on):    
        for j in range(0,len(all_experiment)):
            exp = all_experiment[j]
            if  exp.frame_per_period[i] < min_frame_sequence_on[count]:
                min_frame_sequence_on[count] = exp.frame_per_period[i]
                
    
    min_frame = np.Inf  
    min_frame_period = np.Inf * np.ones(7)
    min_frame_period_tot = np.zeros(8)
    for j in range(0,len(all_experiment)):
        exp = all_experiment[j]
        if  exp.total_frame < min_frame:
            min_frame = exp.total_frame
        for i in range(0, len(exp.frame_per_period)):
            if exp.frame_per_period[i] < min_frame_period[i]:
                min_frame_period[i] = exp.frame_per_period[i]
                
                
    for j, frame in enumerate(min_frame_period):
            min_frame_period_tot[j+1] = min_frame_period_tot[j] + frame
                        
                      
    min_frame = min_frame.astype(int) 
    min_frame_period = min_frame_period.astype(int)
    min_frame_period_tot = min_frame_period_tot.astype(int)
    min_frame_sequence_on = min_frame_sequence_on.astype(int)  
    
    return min_frame, min_frame_period, min_frame_period_tot, min_frame_sequence_on

#%%
def experiment_center_pos(genDict, data, all_folder, simulation, second_layer):
    '''
    

    Parameters
    ----------
    genDict : dict
        Metadata of all experiment from a specific transgenic strain.
    data : Dataframe
        Position of the flies at each time step.
    all_folder : list
        Name of all the folder from a sptecific transgenic strain. 
        (ex : "200206_110534_s1a10_p3-4")
    simulation : str
        Nome of the transgenic strain. 
        (ex : "MDN")
    second_layer : int
        Number between 0 and 4, extract the data of this folder.
        (ex : "all_folder[second_layer]")

    Returns
    -------
    experiment : Fly_Experiment
        Store all the data of this experiment.
        (ex: experiment.pos_x, experiment.pos_y, ...)

    '''
    
    """index correspond to the right index of all flies and all sequences for a specific experiment"""
    index = np.array([])
    experiment = experiment_propeties(genDict, data,
                                      all_folder, simulation, second_layer)  
            
    for i, frame in enumerate(experiment.frame_per_period): 

        idx_tmp = []
        
        #find the index of the starting "on/off" period for each fly
        for j in range(0,experiment.nb_fly):
            
            index_fly = experiment.frame_per_fly*j
            index_stim = np.where(np.array(data.reset_index().iloc[(experiment.index_folder + index_fly): ,1])\
                                  == experiment.frame_split[i])[0][0]
            index_stim += experiment.index_folder + index_fly
            idx_tmp = np.append(idx_tmp, np.arange(index_stim, (index_stim + frame)))
            
        idx_tmp = idx_tmp.reshape(experiment.nb_fly, frame)
        index = np.hstack([index, idx_tmp]) if index.size else idx_tmp
        index = index.astype(int)
    
    experiment.index_order(index)
    
    x_pos, y_pos = any_coordinates(experiment, data.center.posx, data.center.posy)   
    orientation = angle_coordinates(experiment, data.center.orientation)
    
    x_pos_n, y_pos_n = any_coordinates(experiment, data.center.posx_n, data.center.posy_n)    
    orientation_n = angle_coordinates(experiment, data.center.orientation_n)
    
    experiment.position_order(x_pos, y_pos, orientation)  
    experiment.position_n_order(x_pos_n, y_pos_n, orientation_n)
            
    return experiment

#%%
def plot_x_position_over_time_Sen_like(genDict, all_experiment, all_folder, std = False):
    
    '''

    Parameters
    ----------
    genDict : dict
        Metadata of all experiment from a specific transgenic strain.
    all_experiment : list of Fly_Experiment object
        The data of all the experiment from a specific transgenic strain.
    all_folder : list
        Name of all the folder from a sptecific transgenic strain. 
        (ex : "200206_110534_s1a10_p3-4")
    std : bool, optional
        if True, plot the standard deviation. The default is False.

    Returns
    -------
    None.

    '''

    which_pos = ["x position", "y position"]
    color = ["0.8", "0.5"]
    
    ax = []
    x_pos_all = np.array([])
    y_pos_all = np.array([])
    min_frame, min_frame_period, min_frame_period_tot, _ = minimal_frame(all_experiment)
       
    legend = []
    for i in range(0, len(all_folder)):
        legend.append(all_folder[i][7:13])
        
    genDict_key = []
    for key, value in genDict[all_experiment[0].folder].items() :       
        genDict_key.append(key)
        
  
    """ Define the figure"""
    fig = plt.figure(str(all_experiment[0].simulation) +  " : x and y position over time" + str(std),
                     constrained_layout=False,
                     figsize = environment.figure_size,
                     dpi = environment.dpi)
    
    fig.suptitle(str(all_experiment[0].simulation) + " : x and y position over time")      
    gs = gridspec.GridSpec(2, 1, figure=fig) 
    

    """ Plot the x and y position of all the fly of one specific transfgenic strain"""
    for l in range(0,2):
        
        ax = fig.add_subplot(2, 1, l+1) 
        #apply correct legend
        ax.plot(-5, -5, c=color[std], label = "Raw data")
        ax.plot(-5, -5, c= "0", label = "Mean data")
        if std == True :
            ax.plot(-5, -5, c= "0.9", label = "Standard deviation")
            
        ax.legend(loc=1, fontsize = 6)

        #plot x and y coordinates
        for i in range(0,len(all_experiment)) :
            exp = all_experiment[i]
            time = np.arange(0,min_frame)/exp.frame_frequency
            x_pos = np.array([])
            y_pos = np.array([])
            
            for j in range(0, len(exp.frame_per_period)):
                start_index = exp.frame_per_period_tot[j]
                x_pos = np.hstack([x_pos, exp.x_pos_n[:, start_index : start_index + min_frame_period[j]]]) \
                        if x_pos.size else exp.x_pos_n[:, start_index : start_index + min_frame_period[j]]
                y_pos = np.hstack([y_pos, exp.y_pos_n[:, start_index : start_index + min_frame_period[j]]]) \
                        if y_pos.size else exp.y_pos_n[:, start_index : start_index + min_frame_period[j]]
                            
            x_pos_all = np.vstack([x_pos_all, x_pos]) if x_pos_all.size else x_pos
            y_pos_all = np.vstack([y_pos_all, y_pos]) if y_pos_all.size else y_pos
                      
            for k in range(0, exp.nb_fly):
                                                   
                # x coordinate
                if l == 0 :
                    ax.plot(time[:x_pos.shape[1]], x_pos[k,:], c=color[std])
                    ax.set_xticks([])
                    
                # y coordinate                
                else:
                    ax.plot(time[:y_pos.shape[1]], y_pos[k,:], c=color[std])
                    ax.set_xlabel("time [s]")  
                    
        """ Plot the mean over all the trajectories """
        if std == True : 
            if l == 0:
                ax.errorbar(time[:x_pos_all.shape[1]], 
                            np.mean(x_pos_all, axis=0), 
                            yerr=np.std(x_pos_all, axis=0), ecolor = "0.9", c="0")
            else:
                ax.errorbar(time[:y_pos_all.shape[1]], 
                            np.mean(y_pos_all, axis=0), 
                            yerr=np.std(y_pos_all, axis=0), ecolor = "0.9", c="0")
        else:
            if l == 0:
                
                ax.plot(time[:x_pos_all.shape[1]], np.mean(x_pos_all, axis=0), c="0")
            else:
                ax.plot(time[:y_pos_all.shape[1]], np.mean(y_pos_all, axis=0), c="0")
                
        ax.set_ylabel(str(which_pos[l]) + " [mm]")
        ax.set_xlim(time[0], time[-1])
        ax.set_ylim(-35, 35)
        
           
        """ vertical line to separate each sequence"""            
        for j in range(0, len(exp.frame_per_period)):
            ax.axvline(min_frame_period_tot[j]/exp.frame_frequency, ymin = -30, ymax = 30, c='r', ls='--')

    if environment.bool_save == True:
        if std == True :
            fig.savefig(str(exp.simulation) + " x and y position over time std.eps")
        else:
            fig.savefig(str(exp.simulation) + " x and y position over time.eps")
    else:
        fig.show()
              
    return

#%%
def plot_all_y_position_over_time_Sen_like(genDict, all_experiment, all_folder, std = True, iteration=0):
    
    '''
    
    Parameters
    ----------
    genDict : dict
        Metadata of all experiment from a specific transgenic strain.
    all_experiment : list of Fly_Experiment object
        The data of all the experiment from a specific transgenic strain.
    all_folder : list
        Name of all the folder from a sptecific transgenic strain. 
        (ex : "200206_110534_s1a10_p3-4")
    std : bool, optional
        if True, plot the standard deviation. The default is False.
    iteration : int, optional
        choose the stransgeic strain to plot.
        (0 : MDN, 1 : SS02377, 2 : PR)

    Returns
    -------
    None.

    '''       
    l = iteration
    color = ["0.8", "0.5"]
    title = ["MDN", "SS02377", "PR"]
       
    ax = []
    x_pos_all = np.array([])
    y_pos_all = np.array([])
    min_frame, min_frame_period, min_frame_period_tot, _ = minimal_frame(all_experiment)
    
    legend = []
    for i in range(0, len(all_folder)):
        legend.append(all_folder[i][7:13])
        
    genDict_key = []
    for key, value in genDict[all_experiment[0].folder].items() :       
        genDict_key.append(key)

    """ Define the figure"""
    fig = plt.figure("x and y position over time" + str(std),
                     constrained_layout=False,
                     figsize = environment.figure_size,
                     dpi = environment.dpi)
    
    gs = gridspec.GridSpec(3, 1, figure=fig)       
    ax = fig.add_subplot(3, 1, l+1) 
    
    #apply correct legend
    ax.plot(-5, -5, c=color[std], label = "Raw data")
    ax.plot(-5, -5, c= "0", label = "Mean data")
    if std == True :
        ax.plot(-5, -5, c= "0.9", label = "Standard deviation")
    
    ax.legend(loc=1, fontsize = 6)
    ax.set_title(str(title[l]))

    #plot x and y coordinates
    for i in range(0,len(all_experiment)) :
        exp = all_experiment[i]
        time = np.arange(0,min_frame)/exp.frame_frequency
        x_pos = np.array([])
        y_pos = np.array([])
        for j in range(0, len(exp.frame_per_period)):
            start_index = exp.frame_per_period_tot[j]
            x_pos = np.hstack([x_pos, exp.x_pos_n[:, start_index : start_index + min_frame_period[j]]]) \
                    if x_pos.size else exp.x_pos_n[:, start_index : start_index + min_frame_period[j]]
            y_pos = np.hstack([y_pos, exp.y_pos_n[:, start_index : start_index + min_frame_period[j]]]) \
                    if y_pos.size else exp.y_pos_n[:, start_index : start_index + min_frame_period[j]]
                        
        x_pos_all = np.vstack([x_pos_all, x_pos]) if x_pos_all.size else x_pos
        y_pos_all = np.vstack([y_pos_all, y_pos]) if y_pos_all.size else y_pos
      
        for k in range(0, exp.nb_fly):
            # y coordinate                
            ax.plot(time[:y_pos.shape[1]], y_pos[k,:], c=color[std])
            # hide the xaxis if the the bottom subplot
            if l == 2 :
                ax.set_xlabel("time [s]")  
            else:
                ax.xaxis.set_visible(False)
                
    if std == True : 
        ax.errorbar(time[:y_pos_all.shape[1]], 
                    np.mean(y_pos_all, axis=0), 
                    yerr=np.std(y_pos_all, axis=0), ecolor = "0.9", c="0")
    else:
        ax.plot(time[:y_pos_all.shape[1]], np.mean(y_pos_all, axis=0), c="0")
            
    ax.set_ylabel("y position [mm]")
    ax.set_xlim(time[0], time[-1])
    ax.set_ylim(-35, 35)
    
     
    """ vertical line to separate each sequence"""            
    # ax.axvline(0, ymin = -30, ymax = 30, c='r', ls='--')
    for j in range(0, len(exp.frame_per_period)):
        ax.axvline(min_frame_period_tot[j]/exp.frame_frequency, ymin = -30, ymax = 30, c='r', ls='--')
    
    fig.tight_layout()
    
       
    if environment.bool_save == True:
        if std == True :
            fig.savefig("all y position over time std.eps")
            fig.savefig("all y position over time std.png")
        else:
            fig.savefig(str("all y position over time.eps"))
    else:
        fig.show()
              
    return

#%%
def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of `x` and `y`

    Parameters
    ----------
    x, y : array_like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    Returns
    -------
    matplotlib.patches.Ellipse
    
    Other parameters
    ----------------
    kwargs : `~matplotlib.patches.Patch` properties
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y, rowvar=False)
    
    cov = np.clip(cov, 1e-8, cov.max())
    
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    
    return ax.add_patch(ellipse)


#%%
def plot_xy_position_Sen_like(genDict, all_experiment, all_folder, std = True):
    '''
    
    Parameters
    ----------
    genDict : dict
        Metadata of all experiment from a specific transgenic strain.
    all_experiment : list of Fly_Experiment object
        The data of all the experiment from a specific transgenic strain.
    all_folder : list
        Name of all the folder from a sptecific transgenic strain. 
        (ex : "200206_110534_s1a10_p3-4")
    std : bool, optional
        if True, plot the standard deviation. The default is False.

    Returns
    -------
    None.

    '''
    position = np.array([])    
    color = ['#F3B4B4', '#BCBFF5']
    color_mean = ['#FC1D1D', '#121DB3'] #rouge et bleu
    sequence_on = [1, 3, 5] # sequence 0n0, 0n1, 0n2

    legend = []
    for i in range(0, len(all_folder)):
        legend.append(all_folder[i][7:13])
    
    genDict_key = []
    for key, value in genDict[all_experiment[0].folder].items() :
        genDict_key.append(key)
        
    stim_key = genDict[all_experiment[0].folder]['stimulation_paradigm']    
    _, _, _, min_frame = minimal_frame(all_experiment)
    
    
    """ PLOT ALL THE TRAJECTORIES"""
    fig = plt.figure(str(all_experiment[0].simulation) + " : xy position over time ",
                     figsize = environment.figure_size,
                     dpi = environment.dpi)
    # plt.title(str(all_experiment[0].simulation) + " : xy position over time " +  str(stim_key[i]))
    
    ax = fig. add_subplot(111)
    
    #plot x and y coordinates
    for count, i in enumerate(sequence_on):  
        
        x_pos_all = np.array([])
        y_pos_all = np.array([])
        dir_all = np.array([])
        
        if count == 0 :
            # bullshit point for the legend
            ax.plot(0, 0, c=color[0], label = "< 0 mm/s (raw data)")
            ax.plot(0, 0, c=color[1], label = "> 0 mm/s (raw data)")            
            ax.plot(0, 0, c=color_mean[0], label = "< 0 mm/s (mean data)")
            ax.plot(0, 0, c=color_mean[1], label = "> 0 mm/s (mean data)")
            if std == True :
                ax.plot(0, 0, c="0.9", label = "standard deviation")
            ax.legend()
        
        """ iterate over all the flies """
        for j in range(0,len(all_experiment)) :
            exp = all_experiment[j]
            for k in range(0, exp.nb_fly) :
                
                direction, _ = get_speed_direction(genDict, all_experiment, all_folder, 
                                                fly_number = k, experiment = j, sequence = i)
                start_index = exp.frame_per_period_tot[i]
                
                direction = np.heaviside(direction[:min_frame[count]], 0)                 
                x_pos = exp.x_pos_n[k, start_index : start_index + min_frame[count]]
                y_pos = exp.y_pos_n[k, start_index : start_index + min_frame[count]]
                position = np.vstack([x_pos, y_pos]).T
                
                for l, (start, stop) in enumerate(zip(position[:-1], position[1:])):
                    x, y = zip(start, stop)    
                    ax.plot(x, y, color=color[int(direction[l])])
                    
                x_pos_all = np.vstack([x_pos_all, x_pos]) if x_pos_all.size else x_pos
                y_pos_all = np.vstack([y_pos_all, y_pos]) if y_pos_all.size else y_pos
                dir_all = np.vstack([dir_all, direction]) if dir_all.size else direction
                
       
    x_pos_mean = np.mean(x_pos_all, axis=0)
    y_pos_mean = np.mean(y_pos_all, axis=0)
    x_pos_std = np.std(x_pos_all, axis=0)
    y_pos_std = np.std(y_pos_all ,axis=0)
    position_all = np.vstack([x_pos_mean, y_pos_mean]).T
    dir_mean = np.mean(dir_all, axis=0)
    dir_mean = np.heaviside(-0.5 + dir_mean, 1)
          
    # apply the right color depending on the instantaneous speed of the fly 
    for l, (start, stop) in enumerate(zip(position_all[:-1], position_all[1:])):
        x, y = zip(start, stop)
        ax.plot(x, y, color=color_mean[int(dir_mean[l])], zorder = 2)
        
    if std == True:
        confidence_ellipse(x_pos_mean, y_pos_mean, ax, edgecolor='0.5', n_std=3.0)

               
    ax.axvline(0, ymin = -50, ymax = 50, c='0.9', ls='--')
    ax.axhline(0, xmin = -50, xmax = 50, c='0.9', ls='--')                                                        
   
    ax.set_xlabel("x position [mm]")
    ax.set_ylabel("y position [mm]")  
    ax.axis("equal") 
# =============================================================================
#     ax.set_xlim(-38, 38)
#     ax.set_ylim(-38, 38)             
# =============================================================================
    ax.set_ylim(-5,8)
    ax.set_xlim(-5,5)  
        
    if environment.bool_save == True:
        if std == True :
            plt.savefig(str(all_experiment[0].simulation) + " xy position over time.png")
            plt.savefig(str(all_experiment[0].simulation) + " xy position over time.eps")   
    else:
        plt.show()
       
   
    return

#%%
def plot_all_xy_position_Sen_like(genDict, all_experiment, all_folder, std = False, iteration = 0, all_sequence = False):
    '''

    Parameters
    ----------
    genDict : dict
        Metadata of all experiment from a specific transgenic strain.
    all_experiment : list of Fly_Experiment object
        The data of all the experiment from a specific transgenic strain.
    all_folder : list
        Name of all the folder from a sptecific transgenic strain. 
        (ex : "200206_110534_s1a10_p3-4")
    std : bool, optional
        if True, plot the standard deviation. The default is False.
    iteration : int, optional
        choose the stransgeic strain to plot.
        (0 : MDN, 1 : SS02377, 2 : PR)

    Returns
    -------
    None.

    '''

    position = np.array([])   
    color = ['#F3B4B4', '#BCBFF5']
    color_mean = ['#FC1D1D', '#121DB3'] #rouge et bleu
    sequence = [4,5] #plot the sequence off2 and on2
    sequence_on = [5]


    legend = []
    for i in range(0, len(all_folder)):
        legend.append(all_folder[i][7:13])
    
    genDict_key = []
    for key, value in genDict[all_experiment[0].folder].items() :
        genDict_key.append(key)
        
    _, _, _, min_frame = minimal_frame(all_experiment, sequence_on = sequence_on)
    
    
    """ PLOT ALL THE TRAJECTORIES"""
    fig = plt.figure("xy position over time " + str(std),
                     figsize = 2*np.array(environment.figure_size),
                     dpi = 2*np.array(environment.dpi))

    gs = gridspec.GridSpec(2, 3, figure=fig) 

        
    #plot x and y coordinates
    for count, i in enumerate(sequence):  
        x_pos_all = np.array([])
        y_pos_all = np.array([])
        dir_all = np.array([])
        ax = fig.add_subplot(gs[iteration + 3*count]) 
              
        # bullshit point for the legend
        if iteration == 2 and count == 0 :
            ax.plot(0, 0, c=color[0], label = "< 0 mm/s (raw data)")
            ax.plot(0, 0, c=color[1], label = "> 0 mm/s (raw data)")            
            ax.plot(0, 0, c=color_mean[0], label = "< 0 mm/s (mean data)")
            ax.plot(0, 0, c=color_mean[1], label = "> 0 mm/s (mean data)")
            if std == True :
                ax.plot(0, 0, c="0.5", label = "standard deviation")
            ax.legend()
        
        for j in range(0,len(all_experiment)) :
            exp = all_experiment[j]
            for k in range(0, exp.nb_fly) :
                
                direction, _ = get_speed_direction(genDict, all_experiment, all_folder, 
                                                fly_number = k, experiment = j, sequence = i)
                start_index = exp.frame_per_period_tot[i]
                
                direction = np.heaviside(direction[:min_frame[0]], 0)                 
                x_pos = exp.x_pos_n[k, start_index : start_index + min_frame[0]]
                y_pos = exp.y_pos_n[k, start_index : start_index + min_frame[0]]
                position = np.vstack([x_pos, y_pos]).T
                
                for l, (start, stop) in enumerate(zip(position[:-1], position[1:])):
                    x, y = zip(start, stop)    
                    ax.plot(x, y, color=color[int(direction[l])])
                    
                x_pos_all = np.vstack([x_pos_all, x_pos]) if x_pos_all.size else x_pos
                y_pos_all = np.vstack([y_pos_all, y_pos]) if y_pos_all.size else y_pos
                dir_all = np.vstack([dir_all, direction]) if dir_all.size else direction
                
       
        x_pos_mean = np.mean(x_pos_all, axis=0)
        y_pos_mean = np.mean(y_pos_all, axis=0)
        position_all = np.vstack([x_pos_mean, y_pos_mean]).T
        dir_mean = np.mean(dir_all, axis=0)
        dir_mean = np.heaviside(-0.5 + dir_mean, 1)
        
               
        for l, (start, stop) in enumerate(zip(position_all[:-1], position_all[1:])):
            x, y = zip(start, stop) 
            ax.plot(x, y, color=color_mean[int(dir_mean[l])])
        if std == True:
            confidence_ellipse(x_pos_mean, y_pos_mean, ax, edgecolor='0.5', n_std=3.0)
              
                
                
        
        ax.axvline(0, ymin = -50, ymax = 50, c='0.9', ls='--')
        ax.axhline(0, xmin = -50, xmax = 50, c='0.9', ls='--')                                            
        ax.set_title("%s %s" %(str(all_experiment[0].simulation), str(environment.stim_legend[count])))
        ax.axis("equal") 
        ax.set_xlim(-38, 38)
        ax.set_ylim(-38, 38)             

    """ Plot the labels on the big subplot only"""
    if iteration == 2 :
        fig.add_subplot(111, frameon=False)
        plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        plt.ylabel("y position [mm]")
        plt.xlabel("x position [mm]")
        for count, i in enumerate(sequence):
            if environment.bool_save == True:
                 plt.savefig("xy_position_on_off_std_" + str(std) + ".eps")
                 plt.savefig("xy_position_on_off_std_" + str(std) + ".jpg")
            else:
                plt.show()
                   
    return


#%%
def plot_speed_Sen_like(genDict, all_experiment, all_folder, size = 51, polynomial = 3, frame_before = 30):
    
    '''

    Parameters
    ----------
    genDict : dict
        Metadata of all experiment from a specific transgenic strain.
    all_experiment : list of Fly_Experiment object
        The data of all the experiment from a specific transgenic strain.
    all_folder : list
        Name of all the folder from a sptecific transgenic strain. 
        (ex : "200206_110534_s1a10_p3-4")
    size : int, optional
        window size for the savignol filter. The default is 51.
    polynomial : int, optional
        polynomial order for the savignol filter. The default is 3.
    frame_before : int, optional
        number of frame ploted before and after the stimulation. The default is 30.

    Returns
    -------
    None.

    '''

    sequence_on = [1, 3, 5]
    
    genDict_key = []
    stim_key = genDict[all_experiment[0].folder]['stimulation_paradigm']
    
    for key, value in genDict[all_experiment[0].folder].items() :
        genDict_key.append(key)
           
    _, _, _, min_frame = minimal_frame(all_experiment)
            
    for count, i in enumerate(sequence_on):  
        
        """ PLOT ALL THE TRAJECTORIES"""
        plt.figure(str(all_experiment[0].simulation) + " speed " + str(stim_key[i]),
                   # figsize = environment.figure_size,
                   figsize = [8,3],
                   dpi = environment.dpi)
        # plt.title(str(all_experiment[0].simulation) + " speed " + str(stim_key[i]))
        
        plt.plot(-50, -50, c = "0.5", label = "Raw data")
        plt.plot(-50, -50, c= "0", label = "Mean data")
        plt.plot(-50, -50, c= "0.9", label = "Standard deviation")
        plt.legend(loc=4)
        
        total_velocity = np.array([])
        
        for j in range(0,len(all_experiment)):
            exp = all_experiment[j]
            time = np.arange(0,int(exp.total_frame))/exp.frame_frequency
            stim_time = np.array(exp.frame_per_period_tot[1:])/exp.frame_frequency
              
            start_index = exp.frame_per_period_tot[i] - frame_before
            stop_index = start_index + min_frame[count] + 2*frame_before

            for k in range(0, exp.nb_fly) :
                _, velocity = get_speed_direction(genDict, all_experiment, all_folder, 
                                                  fly_number = k, experiment = j)
                
                v_hat = savgol_filter(velocity, size, polynomial)
                
                plt.plot(time[start_index : stop_index],
                         v_hat[start_index : stop_index],
                         c="0.5")
                
                total_velocity = np.vstack([total_velocity, v_hat[start_index : stop_index]]) \
                                            if total_velocity.size else v_hat[start_index : stop_index]

        vel_mean = np.mean(total_velocity, axis=0)
        vel_std = np.std(total_velocity, axis=0)
        
        """ plot the mean over all the trajectories and the standard deviation"""
        plt.errorbar(time[start_index : stop_index],
                     vel_mean, 
                     yerr=vel_std, 
                     ecolor = "0.9", c="0", linewidth = 2)
        
        plt.axhline(0, xmin = -50, xmax = 50, c='r', ls='--')
        plt.axvline(0, ymin = -50, ymax = 50, c='r', ls='--')
        for l in range(0, len(exp.frame_per_period)):
            plt.axvline(stim_time[l], ymin = -50, ymax = 50, c='r', ls='--')

        plt.xlabel("time [s]")
        plt.ylabel("velociy [mm/s]")
        plt.ylim(-25, 20)
        plt.xlim(start_index/exp.frame_frequency,stop_index/exp.frame_frequency)
        
        
        if environment.bool_save == True:
             plt.savefig(str(all_experiment[0].simulation) + "_speed_" + str(stim_key[i]) +  ".eps")
             plt.savefig(str(all_experiment[0].simulation) + "_speed_" + str(stim_key[i]) +  ".png")
        else:
            plt.show()
   
    
    
    return

#%%
def plot_all_speed_Sen_like(genDict, all_experiment, all_folder, size = 51, polynomial = 3, frame_before = 30, iteration = 0):
    '''
    
    Parameters
    ----------
    genDict : dict
        Metadata of all experiment from a specific transgenic strain.
    all_experiment : list of Fly_Experiment object
        The data of all the experiment from a specific transgenic strain.
    all_folder : list
        Name of all the folder from a sptecific transgenic strain. 
        (ex : "200206_110534_s1a10_p3-4")
    size : int, optional
        window size for the savignol filter. The default is 51.
    polynomial : int, optional
        polynomial order for the savignol filter. The default is 3.
    frame_before : int, optional
        number of frame ploted before and after the stimulation. The default is 30.
     iteration : int, optional
        choose the stransgeic strain to plot.
        (0 : MDN, 1 : SS02377, 2 : PR)

    Returns
    -------
    None.

    '''
 
    sequence_on = [1, 3, 5]
    _, _, _, min_frame = minimal_frame(all_experiment, sequence_on = sequence_on)
    
    #vert clair, vert foncé, rouge clair, rouge foncé, bleu clair, bleu foncé
    color = ["#ABDCB2", "#008912", "#FFCACA", "#FF0000", "#A8ABFF", "#0009FF"]
    
    genDict_key = []
    for key, value in genDict[all_experiment[0].folder].items() :
        genDict_key.append(key)
    stim_key = genDict[all_experiment[0].folder]['stimulation_paradigm']

            
    for count, i in enumerate(sequence_on):  
        
        """ PLOT ALL THE TRAJECTORIES"""
        plt.figure(" speed " + str(stim_key[i]),
                   # figsize = environment.figure_size,
                   figsize = [8,3],
                   dpi = environment.dpi)
        # plt.title(" speed " + str(stim_key[i]))
        
        plt.plot(-50, -50, c=color[2*iteration + 1], label = "Mean data " + str(all_experiment[0].simulation))
        plt.legend(loc=1)
        
        total_velocity = np.array([])
        
        for j in range(0,len(all_experiment)):
            
            exp = all_experiment[j]
            time = np.arange(0,int(exp.total_frame))/exp.frame_frequency
            stim_time = np.array(exp.frame_per_period_tot[1:])/exp.frame_frequency
              
            start_index = exp.frame_per_period_tot[i] - frame_before
            stop_index = start_index + min_frame[count] + 2*frame_before

            for k in range(0, exp.nb_fly) :
                _, velocity = get_speed_direction(genDict, all_experiment, all_folder, 
                                                  fly_number = k, experiment = j)
                
                v_hat = savgol_filter(velocity, size, polynomial)                
                total_velocity = np.vstack([total_velocity, v_hat[start_index : stop_index]]) \
                                            if total_velocity.size else v_hat[start_index : stop_index]

        vel_mean = np.mean(total_velocity, axis=0)
        vel_std = np.std(total_velocity, axis=0)
        
        """ plot the mean over all the trajectories and the standard deviation"""
        plt.errorbar(time[start_index : stop_index],
                     vel_mean, 
                     yerr=vel_std, 
                     ecolor = color[2*iteration], 
                     c=color[2*iteration + 1], 
                     linewidth = 2)
        
        if iteration == 0 :
            plt.axhline(0, xmin= 0, xmax = 30, c='k', ls='--')
            plt.axvline(0, ymin = -50, ymax = 50, c='k', ls='--')
            for l in range(0, len(exp.frame_per_period)):
                plt.axvline(stim_time[l], ymin = -50, ymax = 50, c='k', ls='--')

        plt.xlabel("time [s]")
        plt.ylabel("velociy [mm/s]")
        plt.ylim(-25, 25)
        
        plt.xlim(start_index/exp.frame_frequency,stop_index/exp.frame_frequency)
        
    if iteration == 2 :
        for count, i in enumerate(sequence_on):
            if environment.bool_save == True:
                 plt.savefig(" speed_" + str(stim_key[i]) +  ".eps")
                 plt.savefig(" speed_" + str(stim_key[i]) +  ".jpg")
            else:
                plt.show()
    
    return

#%%
def plot_angular_speed_Sen_like(genDict, all_experiment, all_folder, size = 31, polynomial = 3, frame_before = 30):
    '''
    
    Parameters
    ----------
    genDict : dict
        Metadata of all experiment from a specific transgenic strain.
    all_experiment : list of Fly_Experiment object
        The data of all the experiment from a specific transgenic strain.
    all_folder : list
        Name of all the folder from a sptecific transgenic strain. 
        (ex : "200206_110534_s1a10_p3-4")
    size : int, optional
        window size for the savignol filter. The default is 51.
    polynomial : int, optional
        polynomial order for the savignol filter. The default is 3.
    frame_before : int, optional
        number of frame ploted before and after the stimulation. The default is 30.

    Returns
    -------
    None.

    '''
    
    
    
    genDict_key = []
    for key, value in genDict[all_experiment[0].folder].items() :
        genDict_key.append(key)
    stim_key = genDict[all_experiment[0].folder]['stimulation_paradigm']

    sequence_on = [1, 3, 5]
    _, _, _, min_frame = minimal_frame(all_experiment, sequence_on = sequence_on)
            
    for count, i in enumerate(sequence_on):  
        
        """ PLOT ALL THE TRAJECTORIES"""
        plt.figure(str(all_experiment[0].simulation) + " angular speed " + str(stim_key[i]),
                   figsize = environment.figure_size,
                   dpi = environment.dpi)
        plt.title(str(all_experiment[0].simulation) + " angular speed " + str(stim_key[i]))
        
        #apply the coorect legend
        plt.plot(-50, -50, c = "0.5", label = "Raw data")
        plt.plot(-50, -50, c= "0", label = "Mean data")
        plt.plot(-50, -50, c= "0.9", label = "Standard deviation")
        plt.legend(loc=1)
        
        total_angular_velocity = np.array([])
        
        for j in range(0,len(all_experiment)):

            exp = all_experiment[j]
            time = np.arange(0,int(exp.total_frame))/exp.frame_frequency
            stim_time = np.array(exp.frame_per_period_tot[1:])/exp.frame_frequency
            
            theta = exp.orientation
            v_theta = pos2vel(theta, angular = True, number_fly = exp.nb_fly)            
            angular_velocity = abs(v_theta)
            v_angualar_hat = savgol_filter(angular_velocity, size, polynomial)
                       
            start_index = exp.frame_per_period_tot[i] - frame_before
            stop_index = start_index + min_frame[count] + 2*frame_before

            for k in range(0, exp.nb_fly) :

                plt.plot(time[start_index : stop_index],
                         v_angualar_hat[k, start_index : stop_index],
                         c="0.5")
                
                total_angular_velocity = np.vstack([total_angular_velocity, v_angualar_hat[k, start_index : stop_index]]) \
                                         if total_angular_velocity.size else v_angualar_hat[k, start_index : stop_index]

        vel_mean = np.mean(total_angular_velocity, axis=0)
        vel_std = np.std(total_angular_velocity, axis=0)
        
        """ plot the mean over all the trajectories and the standard deviation"""
        plt.errorbar(time[start_index : stop_index],
                     vel_mean, 
                     yerr=vel_std, 
                     ecolor = "0.9", c="0", linewidth = 2)
        
        plt.axhline(0, xmin=0, xmax=30, c='r', ls='--')
        plt.axvline(0, ymin = -50, ymax = 50, c='r', ls='--')
        for l in range(0, len(exp.frame_per_period)):
            plt.axvline(stim_time[l], ymin = -50, ymax = 50, c='r', ls='--')

        plt.xlabel("time [s]")
        plt.ylabel("angular velociy [rad/s]")
        plt.ylim(-1, 10)
        
        plt.xlim(start_index/exp.frame_frequency,stop_index/exp.frame_frequency)
        
        
        if environment.bool_save == True:
             plt.savefig(str(all_experiment[0].simulation) + " angular speed " + str(stim_key[i]) +  ".eps")
        else:
            plt.show()
    
    return


#%%
def plot_all_angular_speed_Sen_like(genDict, all_experiment, all_folder, size = 31, polynomial = 3, frame_before = 30, iteration = 0):
    '''
    
    Parameters
    ----------
    genDict : dict
        Metadata of all experiment from a specific transgenic strain.
    all_experiment : list of Fly_Experiment object
        The data of all the experiment from a specific transgenic strain.
    all_folder : list
        Name of all the folder from a sptecific transgenic strain. 
        (ex : "200206_110534_s1a10_p3-4")
    size : int, optional
        window size for the savignol filter. The default is 51.
    polynomial : int, optional
        polynomial order for the savignol filter. The default is 3.
    frame_before : int, optional
        number of frame ploted before and after the stimulation. The default is 30.
    iteration : int, optional
        choose the stransgeic strain to plot.
        (0 : MDN, 1 : SS02377, 2 : PR)

    Returns
    -------
    None.

    '''
    
    color = ["#ABDCB2", "#008912", "#FFCACA", "#FF0000", "#A8ABFF", "#0009FF", "#EEA9E7", "#F518DF"]
    sequence_on = [1, 3, 5]    
    _, _, _, min_frame = minimal_frame(all_experiment, sequence_on = sequence_on)
    
    genDict_key = []
    for key, value in genDict[all_experiment[0].folder].items() :
        genDict_key.append(key)
    stim_key = genDict[all_experiment[0].folder]['stimulation_paradigm']

            
    for count, i in enumerate(sequence_on):  
        
        """ PLOT ALL THE TRAJECTORIES"""
        plt.figure("angular speed " + str(stim_key[i]),
                   figsize = [8,3],
                   # figsize = environment.figure_size,
                   dpi = environment.dpi)
       #  plt.title("angular speed " + str(stim_key[i]))
        
        plt.plot(-50, -50, c=color[2*iteration + 1], label = "Mean data " + str(all_experiment[0].simulation))
        plt.legend(loc=1)
        
        total_angular_velocity = np.array([])
        
        for j in range(0,len(all_experiment)):

            exp = all_experiment[j]
            theta = exp.orientation
            v_theta = pos2vel(theta, angular = True, number_fly = exp.nb_fly)
                                              
            time = np.arange(0,int(exp.total_frame))/exp.frame_frequency
            stim_time = np.array(exp.frame_per_period_tot[1:])/exp.frame_frequency
             
            angular_velocity = abs(v_theta)
            v_hat = savgol_filter(angular_velocity, size, polynomial)
                       
            start_index = exp.frame_per_period_tot[i] - frame_before
            stop_index = start_index + min_frame[count] + 2*frame_before

            for k in range(0, exp.nb_fly) :
                
                total_angular_velocity = np.vstack([total_angular_velocity, v_hat[k, start_index : stop_index]]) \
                                         if total_angular_velocity.size else v_hat[k, start_index : stop_index]

        vel_mean = np.mean(total_angular_velocity, axis=0)
        vel_std = np.std(total_angular_velocity, axis=0)
        
        """ plot the mean over all the trajectories and the standard deviation"""
        plt.errorbar(time[start_index : stop_index],
                     vel_mean, 
                     yerr=vel_std, 
                     ecolor = color[2*iteration], 
                     c=color[2*iteration + 1], 
                     linewidth = 2)
        
        if iteration == 0 :
            plt.axhline(y=0, xmin=0, xmax=30, c='k', ls='--')
            plt.axvline(0, ymin = -50, ymax = 50, c='k', ls='--')
            for l in range(0, len(exp.frame_per_period)):
                plt.axvline(stim_time[l], ymin = -50, ymax = 50, c='k', ls='--')

        plt.xlabel("time [s]")
        plt.ylabel("angular velociy [rad/s]")
        plt.ylim(-1, 10)
        
        plt.xlim(start_index/exp.frame_frequency,stop_index/exp.frame_frequency)
        
        
    if iteration == 2 :
        for count, i in enumerate(sequence_on):
            if environment.bool_save == True:
                 plt.savefig("angular_speed_" + str(stim_key[i]) +  ".eps")
                 plt.savefig("angular_speed_" + str(stim_key[i]) +  ".jpg")
            else:
                plt.show()    
    return


#%%
def get_speed_direction(genDict, all_experiment, all_folder, fly_number = 0, experiment = 0, sequence = None):
    '''

    Parameters
    ----------
    genDict : dict
        Metadata of all experiment from a specific transgenic strain.
    all_experiment : list of Fly_Experiment object
        The data of all the experiment from a specific transgenic strain.
    all_folder : list
        Name of all the folder from a sptecific transgenic strain. 
        (ex : "200206_110534_s1a10_p3-4")
    fly_number : int, optional
        The fly to analyse. The default is 0.
    experiment : int, optional
        The experiment to anaylse. The default is 0.      
    sequence : int, optional
        The sequence to analyse. The default is None.
        (ex : 0 --> off0
              1 --> on1 ...)

    Returns
    -------
    direction : TYPE
        DESCRIPTION.
    velocity : TYPE
        DESCRIPTION.

    '''
    
    k = fly_number
    
    genDict_key = []   
    for key, value in genDict[all_experiment[0].folder].items() :
        genDict_key.append(key)
    
    exp = all_experiment[experiment]
    
    if sequence == None :
       delta_x = SMA_pos(exp.x_pos)[k, :]  
       delta_y = SMA_pos(exp.y_pos)[k, :] 
       theta = exp.orientation[k, :]
                     
    else:   
        delta_x = SMA_pos(exp.x_pos[:, exp.frame_per_period_tot[sequence] : exp.frame_per_period_tot[sequence + 1]])[k, :]   
        delta_y = SMA_pos(exp.y_pos[:, exp.frame_per_period_tot[sequence] : exp.frame_per_period_tot[sequence + 1]])[k, :] 
        theta = exp.orientation[k, exp.frame_per_period_tot[sequence] : exp.frame_per_period_tot[sequence + 1]]
                    
    direction = np.zeros_like(delta_x)
    theta_effective = np.zeros_like(delta_x)
    
    for m in range(0, delta_x.shape[0]):
        theta_effective[m] = math.atan2(delta_x[m], (-delta_y[m])) * 360/(2*math.pi)
        
        """ find the direction in order to know if the fly is going forard or backward""" 
        if 90 <= theta[m] <= 270:
            if (theta[m] - 90) < (theta_effective[m] % 360) <= (theta[m] + 90):
                direction[m] = 1
            else:
                direction[m] = -1
                
        elif theta[m] < 90 :
            if (theta[m] - 90) < theta_effective[m] <= (theta[m] + 90):
                direction[m] = 1
            else:
                direction[m] = -1
            
        else:
            if (theta[m] - 360 - 90) < theta_effective[m] <= ((theta[m] + 90) % 360):
                direction[m] = 1
            else:
                direction[m] = -1 

        delta_pos = (delta_x**2 + delta_y**2)**0.5 
        velocity = direction * delta_pos *  exp.frame_frequency
        
   
    return direction, velocity


#%%
def joint_normalized(exp, x_pos_all, y_pos_all, x_pos_joint, y_pos_joint) :
    
    
    x_pos_all = np.hstack([x_pos_all, x_pos_joint]) if x_pos_all.size else x_pos_joint
    y_pos_all = np.hstack([y_pos_all, y_pos_joint]) if y_pos_all.size else y_pos_joint


    return x_pos_all, y_pos_all


#%%
def frequency_cycle(genDict, data, all_experiment, all_folder):
    '''
    

    Parameters
    ----------
    genDict : dict
        Metadata of all experiment from a specific transgenic strain.
    data : Dataframe
        Position of the flies at each time step.
    all_experiment : list of Fly_Experiment object
        The data of all the experiment from a specific transgenic strain.
    all_folder : list
        Name of all the folder from a sptecific transgenic strain. 
        (ex : "200206_110534_s1a10_p3-4")
    experiment : int, optional
        The experiment to anaylse. The default is 0.      



    Returns
    -------
    None.

    '''
    
    
    legend = []
    genDict_key = []
    legend = ["Fore Leg", "Middle Leg", "Hind Leg"]
    legend_short = ["LF", "LM", "LH", "RF", "RM", "RH"]

    
    for key, value in genDict[all_experiment[0].folder].items() :
        genDict_key.append(key)
     
    
    for i in range(0,len(all_experiment)) :
    # for i in range(experiment,experiment+1) :
        exp = all_experiment[i]      

        x_pos_all = np.array([])
        y_pos_all = np.array([])
        stim_time = np.array(exp.frame_per_period_tot[1:])/exp.frame_frequency
                    
        """ extracting all the joint positions """
        x_pos_LF, y_pos_LF = any_coordinates(exp, data.LFclaw.x, data.LFclaw.y)
        x_pos_all, y_pos_all = joint_normalized(exp, x_pos_all, y_pos_all, x_pos_LF, y_pos_LF)            
        
        x_pos_LM, y_pos_LM = any_coordinates(exp, data.LMclaw.x, data.LMclaw.y)
        x_pos_all, y_pos_all = joint_normalized(exp, x_pos_all, y_pos_all, x_pos_LM, y_pos_LM)
        
        x_pos_LH, y_pos_LH = any_coordinates(exp, data.LHclaw.x, data.LHclaw.y)
        x_pos_all, y_pos_all = joint_normalized(exp, x_pos_all, y_pos_all, x_pos_LH, y_pos_LH)
        
        x_pos_RF, y_pos_RF = any_coordinates(exp, data.RFclaw.x, data.RFclaw.y)
        x_pos_all, y_pos_all = joint_normalized(exp, x_pos_all, y_pos_all, x_pos_RF, y_pos_RF)
        
        x_pos_RM, y_pos_RM = any_coordinates(exp, data.RMclaw.x, data.RMclaw.y)
        x_pos_all, y_pos_all = joint_normalized(exp, x_pos_all, y_pos_all, x_pos_RM, y_pos_RM)
           
        x_pos_RH, y_pos_RH = any_coordinates(exp, data.RHclaw.x, data.RHclaw.y)
        x_pos_all, y_pos_all = joint_normalized(exp, x_pos_all, y_pos_all, x_pos_RH, y_pos_RH)

        for k in range(0, exp.nb_fly): 
            Sxx_total_Fore = np.array([])
            Sxx_total_Middle = np.array([])
            Sxx_total_Hind = np.array([])
            Sxx_total = [Sxx_total_Fore, Sxx_total_Middle, Sxx_total_Hind]
            
            for l in range(0,3) :
                for j in range(0,2) :
                
                    start_index = (l+3*j)*exp.frame_per_fly
                    stop_index = (l+3*j+1)*exp.frame_per_fly
        
                    f, t, Sxx = signal.spectrogram(y_pos_all[k, start_index: stop_index], exp.frame_frequency, nperseg=40)                
                    Sxx_total[l] = np.dstack([Sxx_total[l], Sxx]) if Sxx_total[l].size else Sxx           
                
            fig = plt.figure(str(exp.simulation) + "_" + str(exp.folder[7:13]) + "_" + str(genDict_key[2+k]) + " : Mean Claw gait frequency ",
                            figsize = environment.figure_size,
                            # figsize = [8,4],
                            dpi = environment.dpi)
            # plt.suptitle(str(exp.simulation) + " : Mean Claw gait frequency ")
    
            for l in range(0,3):
                ax = fig.add_subplot(3,1,l+1)
                        
                ax.pcolormesh(t, f, np.mean(Sxx_total[l], axis = 2))
                ax.yaxis.set_label_position("right")
                ax.set_ylabel("%s" %(legend[l]))
                if l < 2 :
                    ax.set_xticks([])
                    
                for j in range(0, len(exp.frame_per_period) - 1):
                    ax.axvline(stim_time[j], ymin = 0, ymax = 30, c='r', ls='--')
            
            fig.add_subplot(111, frameon=False)
            plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            plt.ylabel("Frequency [Hz]")
            plt.xlabel("Time [s]")
            plt.tight_layout()
    
            """ plot or save """
            if environment.bool_save == True:
                plt.savefig(str(exp.simulation) + "_" + str(exp.folder[7:13]) + "_" + str(genDict_key[2+k]) + "_Mean_Claw_frequency.png")
                plt.savefig(str(exp.simulation) + "_" + str(exp.folder[7:13]) + "_" + str(genDict_key[2+k]) + "_Mean_Claw_frequency.eps")
            else:
                plt.show()


    return
#%%
def mean_frequency_cycle(genDict, data, all_experiment, all_folder):
    '''
    

    Parameters
    ----------
    genDict : dict
        Metadata of all experiment from a specific transgenic strain.
    data : Dataframe
        Position of the flies at each time step.
    all_experiment : list of Fly_Experiment object
        The data of all the experiment from a specific transgenic strain.
    all_folder : list
        Name of all the folder from a sptecific transgenic strain. 
        (ex : "200206_110534_s1a10_p3-4")
    experiment : int, optional
        The experiment to anaylse. The default is 0.      



    Returns
    -------
    None.

    '''
    
    
    legend = []
    genDict_key = []
    legend = ["Fore Leg", "Middle Leg", "Hind Leg"]
    legend_short = ["LF", "LM", "LH", "RF", "RM", "RH"]

    
    for key, value in genDict[all_experiment[0].folder].items() :
        genDict_key.append(key)
     
        
    Sxx_total_Fore = np.array([])
    Sxx_total_Middle = np.array([])
    Sxx_total_Hind = np.array([])
    Sxx_total = [Sxx_total_Fore, Sxx_total_Middle, Sxx_total_Hind]
    
    for i in range(0,len(all_experiment)) :
    # for i in range(experiment,experiment+1) :
        exp = all_experiment[i] 
       

        x_pos_all = np.array([])
        y_pos_all = np.array([])
        stim_time = np.array(exp.frame_per_period_tot[1:])/exp.frame_frequency
                    
        """ extracting all the joint positions """
        x_pos_LF, y_pos_LF = any_coordinates(exp, data.LFclaw.x, data.LFclaw.y)
        x_pos_all, y_pos_all = joint_normalized(exp, x_pos_all, y_pos_all, x_pos_LF, y_pos_LF)            
        
        x_pos_LM, y_pos_LM = any_coordinates(exp, data.LMclaw.x, data.LMclaw.y)
        x_pos_all, y_pos_all = joint_normalized(exp, x_pos_all, y_pos_all, x_pos_LM, y_pos_LM)
        
        x_pos_LH, y_pos_LH = any_coordinates(exp, data.LHclaw.x, data.LHclaw.y)
        x_pos_all, y_pos_all = joint_normalized(exp, x_pos_all, y_pos_all, x_pos_LH, y_pos_LH)
        
        x_pos_RF, y_pos_RF = any_coordinates(exp, data.RFclaw.x, data.RFclaw.y)
        x_pos_all, y_pos_all = joint_normalized(exp, x_pos_all, y_pos_all, x_pos_RF, y_pos_RF)
        
        x_pos_RM, y_pos_RM = any_coordinates(exp, data.RMclaw.x, data.RMclaw.y)
        x_pos_all, y_pos_all = joint_normalized(exp, x_pos_all, y_pos_all, x_pos_RM, y_pos_RM)
           
        x_pos_RH, y_pos_RH = any_coordinates(exp, data.RHclaw.x, data.RHclaw.y)
        x_pos_all, y_pos_all = joint_normalized(exp, x_pos_all, y_pos_all, x_pos_RH, y_pos_RH)

        for k in range(0, exp.nb_fly):            
            for l in range(0,3) :
                for j in range(0,2) :
                
                    start_index = (l+3*j)*exp.frame_per_fly
                    stop_index = (l+3*j+1)*exp.frame_per_fly
        
                    f, t, Sxx = signal.spectrogram(y_pos_all[k, start_index: stop_index], exp.frame_frequency, nperseg=40)                
                    Sxx_total[l] = np.dstack([Sxx_total[l], Sxx]) if Sxx_total[l].size else Sxx           
                
    fig = plt.figure(str(exp.simulation) + " : Mean Claw gait frequency ",
                    figsize = environment.figure_size,
                    # figsize = [8,4],
                    dpi = environment.dpi)
    # plt.suptitle(str(exp.simulation) + " : Mean Claw gait frequency ")
    
    for l in range(0,3):
        ax = fig.add_subplot(3,1,l+1)
                
        ax.pcolormesh(t, f, np.mean(Sxx_total[l], axis = 2))
        ax.yaxis.set_label_position("right")
        ax.set_ylabel("%s" %(legend[l]))
        if l < 2 :
            ax.set_xticks([])
# =============================================================================
#         ax.set_ylabel('Frequency [Hz]')
#         ax.set_xlabel('Time [s]')
# =============================================================================
            
        ax.set_ylim(0, 25)
        
        for j in range(0, len(exp.frame_per_period) - 1):
            ax.axvline(stim_time[j], ymin = 0, ymax = 30, c='r', ls='--')
     
            
    
    
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.ylabel("Frequency [Hz]")
    plt.xlabel("Time [s]")
    plt.tight_layout()
    
    """ plot or save """
    if environment.bool_save == True:
        plt.savefig(str(exp.simulation) + "_" + str(exp.folder[7:13]) + "_" + str(genDict_key[2+k]) + "_Mean_Claw_frequency.png")
        plt.savefig(str(exp.simulation) + "_" + str(exp.folder[7:13]) + "_" + str(genDict_key[2+k]) + "_Mean_Claw_frequency.eps")
    else:
        plt.show()


    return

#%%
def connect_bbox(bbox1, bbox2,
                 loc1a, loc2a, loc1b, loc2b,
                 prop_lines, prop_patches=None):
    if prop_patches is None:
        prop_patches = {
            **prop_lines,
            "alpha": prop_lines.get("alpha", 1) * 0.2,
        }

    c1 = BboxConnector(bbox1, bbox2, loc1=loc1a, loc2=loc2a, **prop_lines)
    c1.set_clip_on(False)
    c2 = BboxConnector(bbox1, bbox2, loc1=loc1b, loc2=loc2b, **prop_lines)
    c2.set_clip_on(False)

    bbox_patch1 = BboxPatch(bbox1, **prop_patches)
    bbox_patch2 = BboxPatch(bbox2, **prop_patches)

    p = BboxConnectorPatch(bbox1, bbox2,
                           # loc1a=3, loc2a=2, loc1b=4, loc2b=1,
                           loc1a=loc1a, loc2a=loc2a, loc1b=loc1b, loc2b=loc2b,
                           **prop_patches)
    p.set_clip_on(False)

    return c1, c2, bbox_patch1, bbox_patch2, p


def zoom_effect(ax1, ax2, xmin_time, xmax_time, xmin_frame, xmax_frame, **kwargs):
    """
    Connect *ax1* and *ax2*. The *xmin*-to-*xmax* range in both axes will
    be marked.

    Parameters
    ----------
    ax1
        The main axes.
    ax2
        The zoomed axes.
    xmin, xmax
        The limits of the colored area in both plot axes.
    **kwargs
        Arguments passed to the patch constructor.
    """

    trans1 = blended_transform_factory(ax1.transData, ax1.transAxes)
    trans2 = blended_transform_factory(ax2.transData, ax2.transAxes)

    bbox1 = Bbox.from_extents(xmin_time, 0, xmax_time, 1)
    bbox2 = Bbox.from_extents(xmin_frame, 0, xmax_frame, 1)

    mybbox1 = TransformedBbox(bbox1, trans1)
    mybbox2 = TransformedBbox(bbox2, trans2)

    prop_patches = {**kwargs, "ec": "none", "alpha": 0.2}

    c1, c2, bbox_patch1, bbox_patch2, p = connect_bbox(
        mybbox1, mybbox2,
        loc1a=1, loc2a=4, loc1b=2, loc2b=3,
        prop_lines=kwargs, prop_patches=prop_patches)
    
        # loc1a=3, loc2a=2, loc1b=4, loc2b=1,
        # loc1a=4, loc2a=1, loc1b=3, loc2b=2

    ax1.add_patch(bbox_patch1)
    ax2.add_patch(bbox_patch2)
    ax2.add_patch(c1)
    ax2.add_patch(c2)
    ax2.add_patch(p)

    return c1, c2, bbox_patch1, bbox_patch2, p

#%%
def joint_position_over_time_specific(genDict, data, all_experiment, all_folder, fly_number = 2, experiment = 2,
                                      sequence = 1, xmin_time = None, xmax_time = None, markersize = 2, backwards = True):
    '''
    

    Parameters
    ----------
    genDict : dict
        Metadata of all experiment from a specific transgenic strain.
    data : Dataframe
        Position of the flies at each time step.   
    all_experiment : list of Fly_Experiment object
        The data of all the experiment from a specific transgenic strain.
    all_folder : list
        Name of all the folder from a sptecific transgenic strain. 
        (ex : "200206_110534_s1a10_p3-4")
    fly_number : int, optional
        The fly to analyse. The default is 0.
    experiment : int, optional
        The experiment to anaylse. The default is 0.      
    sequence : int, optional
        The sequence to analyse. The default is None.
        (ex : 0 --> off0
              1 --> on1 ...)
    xmin_time : int, optional
        min time for the gait pattern analysis. The default is None.
    xmax_time : int, optional
         max time for the gait pattern analysis. The default is None.
    markersize : int, optional
        size of the black square on the gait pattern plot. The default is 2.
    backwards : bool, optional
        take into accoun the direction of the speed. The default is True.

    Returns
    -------
    None.

    '''
    
    k = fly_number
    j = sequence    
    exp = all_experiment[experiment]
    
    stim_key = genDict[all_experiment[0].folder]['stimulation_paradigm']
    stimulation = stim_key[sequence]
    
    legend = []
    genDict_key = []
    gait_frequency_tot = np.array([])
    duty_factor_tot = np.array([])
    
    legend = ["Left Fore", "Left Middle", "Left Hind", "Right Fore", "Right Middle", "Right Hind"]
    legend_short = ["LF", "LM", "LH", "RF", "RM", "RH"]
    x_plot_interval = [0, 0, -1.5, 0, 0, 1.5]


    
    for key, value in genDict[all_experiment[0].folder].items() :
        genDict_key.append(key)

    #plot x and y coordinates of the Tarus

    x_pos_all = np.array([])
    y_pos_all = np.array([])
    time = np.arange(0,int(exp.total_frame))/exp.frame_frequency
                
    """ extracting all the joint positions """
    x_pos_LF, y_pos_LF = any_coordinates(exp, data.LFclaw.x, data.LFclaw.y)
    x_pos_all, y_pos_all = joint_normalized(exp, x_pos_all, y_pos_all, x_pos_LF, y_pos_LF)            
    
    x_pos_LM, y_pos_LM = any_coordinates(exp, data.LMclaw.x, data.LMclaw.y)
    x_pos_all, y_pos_all = joint_normalized(exp, x_pos_all, y_pos_all, x_pos_LM, y_pos_LM)
    
    x_pos_LH, y_pos_LH = any_coordinates(exp, data.LHclaw.x, data.LHclaw.y)
    x_pos_all, y_pos_all = joint_normalized(exp, x_pos_all, y_pos_all, x_pos_LH, y_pos_LH)
    
    x_pos_RF, y_pos_RF = any_coordinates(exp, data.RFclaw.x, data.RFclaw.y)
    x_pos_all, y_pos_all = joint_normalized(exp, x_pos_all, y_pos_all, x_pos_RF, y_pos_RF)
    
    x_pos_RM, y_pos_RM = any_coordinates(exp, data.RMclaw.x, data.RMclaw.y)
    x_pos_all, y_pos_all = joint_normalized(exp, x_pos_all, y_pos_all, x_pos_RM, y_pos_RM)
       
    x_pos_RH, y_pos_RH = any_coordinates(exp, data.RHclaw.x, data.RHclaw.y)
    x_pos_all, y_pos_all = joint_normalized(exp, x_pos_all, y_pos_all, x_pos_RH, y_pos_RH)
    
        
    fig = plt.figure(str(exp.simulation) + " " + str(exp.folder[7:13]) + " " + str(genDict_key[2+k]) + \
                        " : position and gait pattern " +  str(stim_key[j]),
                        # figsize = environment.figure_size,
                        figsize = [8,4],
                        dpi = environment.dpi)
    
    gs = fig.add_gridspec(2, 3)
    ax_y = fig.add_subplot(gs[1, 1:])
    ax_x = fig.add_subplot(gs[0,0])
    ax_gait = fig.add_subplot(gs[0, 1:])
    ax_legend = fig.add_subplot(gs[1,0])
    
    gait_pattern = np.zeros((11*6, exp.frame_per_period[j]))
    y_legend_postion = 11*np.arange(6) + 5
    
    
    for l in range(0,6) :
        start_index = l*exp.frame_per_fly
        
        """ Y POSITION"""
        ax_y.plot(time[: exp.frame_per_period[j]], 
                  y_pos_all[k, start_index + exp.frame_per_period_tot[j] : start_index + exp.frame_per_period_tot[j+1]])
        
        """ X POSITION"""
        ax_x.plot(x_pos_all[k, start_index + exp.frame_per_period_tot[j] : start_index + exp.frame_per_period_tot[j+1]] + x_plot_interval[l],
                 time[: exp.frame_per_period[j]])
        
        """ LEGEND"""
# =============================================================================
#         ax_legend.plot(-1, -1)
# =============================================================================
        
        """ GAIT PATTERN"""
        SMA_speed = pos2vel(y_pos_all[:, start_index + exp.frame_per_period_tot[j] : start_index + exp.frame_per_period_tot[j+1]])[k,:] 
        # SMA_speed_threshold = np.array([0 if -1.5 < speed < 1.5 else speed for speed in SMA_speed])
        SMA_speed_threshold = SMA_speed
        
        ### find the direction of the fly, forward speed or backward speed
        direction, _ = get_speed_direction(genDict, all_experiment, all_folder, 
                                        fly_number = fly_number, 
                                        experiment = experiment, 
                                        sequence = sequence)
        
        if backwards == True :
            # apply -1 to have the stance phase in black and the swing phase in white 
            speed_sign = np.heaviside(direction*SMA_speed_threshold, 1) 
        else:             
            speed_sign = np.heaviside(SMA_speed_threshold, 1)
        speed_sign = np.repeat(speed_sign.reshape((1, len(speed_sign))), 11 - 1, axis = 0)
        gait_pattern[11*l : 11*(l+1) - 1, :] = speed_sign 
        
    
    if xmin_time != None :
        xmin_time = int(xmin_time*exp.frame_frequency)/exp.frame_frequency
        if xmax_time != None:
            xmax_time = int(xmax_time*exp.frame_frequency)/exp.frame_frequency
        else:
            xmax_time = time[exp.frame_per_period[j]]
        xmin_frame = 0
        xmax_frame = (xmax_time - xmin_time)*exp.frame_frequency
        zoom_effect(ax_y, ax_gait, xmin_time, xmax_time, xmin_frame, xmax_frame)
        ax_gait.spy(gait_pattern[:, int(xmin_time*exp.frame_frequency) : int(xmax_time*exp.frame_frequency)], \
                                 markersize=markersize, aspect = "auto", c= "k")
        
        for i in range(0,6):
            sequence = gait_pattern[11*i, int(xmin_time*exp.frame_frequency) : int(xmax_time*exp.frame_frequency)]
            duty_factor = np.count_nonzero(sequence)/len(sequence)
            duty_factor_tot = np.append(duty_factor_tot, duty_factor)           
            
            
            # count the occurence of 1 --> count the number of time you switch from swing leg to stance leg
            # cycle/s
            edge_detector = np.hstack((0, [sequence[j+1] - sequence[j] for j in range(0, len(sequence)-1)]))
            gait_frequency = exp.frame_frequency * np.count_nonzero(edge_detector == 1)/len(sequence)
            gait_frequency_tot = np.append(gait_frequency_tot, gait_frequency)
            
            print("%s duty factor = %.2f \t gait frequency  = %.2f [cycle/s]" %(legend_short[i], duty_factor, gait_frequency))
    
    else:
        ax_gait.spy(gait_pattern, markersize=markersize, aspect = "auto", c='k')
        
        for i in range(0,6):
            sequence = gait_pattern[11*i, :]
            duty_factor = np.count_nonzero(sequence)/len(sequence)
            duty_factor_tot = np.append(duty_factor_tot, duty_factor)
            
            # count the occurence of 1 --> count the number of time you switch from swing leg to stance leg
            # cycle/s
            edge_detector = np.hstack((0, [sequence[j+1] - sequence[j] for j in range(0, len(sequence)-1)]))
            gait_frequency = exp.frame_frequency * np.count_nonzero(edge_detector == 1)/len(sequence)
            gait_frequency_tot = np.append(gait_frequency_tot, gait_frequency)
            
            print("%s duty factor = %.2f \t gait frequency = %.2f [cycle/s]" %(legend_short[i], duty_factor, gait_frequency))

    print("\nAverage duty factor = %.2f" %(np.mean(duty_factor_tot)))  
    print("Average gait frequency = %.2f" %(np.mean(gait_frequency_tot)))

        
    ax_y.set_ylim(5.5,-0.5)
    ax_y.set_xlim(0, time[exp.frame_per_period[j]])
    ax_y.set_xlabel("time [s]")
    ax_y.set_yticks([])
                                  
    ax_x.set_ylabel("time [s]")
    ax_x.set_xticks([])
    ax_x.set_xlim(-0.5 ,6.5)
    ax_x.set_ylim(0, time[exp.frame_per_period[j]])
    

    ax_legend.axis('off')
# =============================================================================
#     ax_legend.legend(legend, fontsize = 12)
# =============================================================================

        
    ax_gait.set_xticks([])
    
    plt.sca(ax_gait)
    plt.yticks(y_legend_postion, legend_short)
 
# =============================================================================
#     plt.suptitle(str(exp.simulation) + " " + str(exp.folder[7:13]) + " " + str(genDict_key[2+k]) + \
#                  " : position and gait pattern " +  str(stim_key[j]))
# =============================================================================
        
    """ plot or save """
    if environment.bool_save == True:
        if backwards == True:
            fig.savefig(str(exp.simulation) + " " + str(exp.folder[7:13]) + " " + str(genDict_key[2+k]) + \
                 " position and gait pattern " +  str(stim_key[j]) + " backwards.png")
            fig.savefig(str(exp.simulation) + "_" + str(exp.folder[7:13]) + "_" + str(genDict_key[2+k]) + \
                 " position and gait pattern_" +  str(stim_key[j]) + "_backwards.eps")
        else:
            fig.savefig(str(exp.simulation) + " " + str(exp.folder[7:13]) + " " + str(genDict_key[2+k]) + \
                     " position and gait pattern " +  str(stim_key[j]) + ".png")
    else:
        fig.show()

    return

#%%
########### THIS IS HOW TO CHOOSE THE DESIRED DIRECTORY
#1st level folder choose a number between 0 and 9
    """ ['MDN', 'PR', 'SS01049, 'SS01054', 'SS01540', 'SS02111', 'SS02279', 'SS02377', 'SS02608', 'SS02617]
    note :  first layer name is saved as self.simulation in Fly_Experiment class"""


#2nd layer folder choose a number between 0 and 3
    """example for the MDN folder :
   ['200206_110534_s1a10_p3-4', '200206_160327_s4a9_p3-4', '200206_105311_s1a9_p3-4', '200206_153954_s4a10_p3-4']"""

    """ normalized = 0 --> plot x_pos
    normalized = 1 --> plot x_pos_n"""

    """ to save all the figure, bool_save = True"""

global environment


first_layer = 7
second_layer = 0
normalized = 0
bool_save = True
all_experiment = []


environment = Environment(bool_save = bool_save)
genDict, data, all_folder, simulation = general_data(first_layer)

for i in range(0,4):
    second_layer = i
    experiment = experiment_center_pos(genDict, data, all_folder, simulation, second_layer)
    all_experiment.append(experiment)
    
joint_position_over_time_specific(genDict, data, all_experiment, all_folder, fly_number = 2, experiment = 2,
                                  sequence = 1, xmin_time = None, xmax_time = None, markersize = 2, backwards = True)
   
frequency_cycle(genDict, data, all_experiment, all_folder)    
plot_x_position_over_time_Sen_like(genDict, all_experiment, all_folder, std = True)    
plot_angular_speed_Sen_like(genDict, all_experiment, all_folder, size = 31, polynomial = 3, frame_before = 30)
plot_speed_Sen_like(genDict, all_experiment, all_folder, size = 51, polynomial = 3, frame_before = 30)

""" This is for all the velocity plot"""

bool_save = False
all_experiment = []
strain = [1, 0, 7] 
environment = Environment(bool_save = bool_save)

for j, st in enumerate(strain):
    all_experiment = []
    first_layer = st
    genDict, data, all_folder, simulation = general_data(first_layer)

    for i in range(0,4):
        second_layer = i
        experiment = experiment_center_pos(genDict, data, all_folder, simulation, second_layer)
        all_experiment.append(experiment)
        
    stim_key = genDict[all_experiment[0].folder]['stimulation_paradigm']
    all_frequency_cycle(genDict, data, all_experiment, all_folder, iteration = j)
    plot_all_xy_position_Sen_like(genDict, all_experiment, all_folder, std = True, iteration = j)
    plot_all_y_position_over_time_Sen_like(genDict, all_experiment, all_folder, std = True, iteration=j)
    plot_all_angular_speed_Sen_like(genDict, all_experiment, all_folder, size = 31, polynomial = 3, frame_before = 30, iteration = j)
    plot_all_speed_Sen_like(genDict, all_experiment, all_folder, size = 51, polynomial = 3, frame_before = 30, iteration = j)


