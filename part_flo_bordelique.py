
import pickle
import pandas as pd
import numpy as np
import math
import scipy.fftpack
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from tqdm import tqdm
from matplotlib import animation
from matplotlib.transforms import (Bbox, TransformedBbox, blended_transform_factory)
from mpl_toolkits.axes_grid1.inset_locator import (BboxPatch, BboxConnector, BboxConnectorPatch)


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
        self.dpi = 300
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
# =============================================================================
#     first_layer = 0 
#     second_layer = 0
# =============================================================================
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
    
    "there is an error in SS02608 '200212_155556_s3a8_p3-4' some posy data are far away from reality "
    if first_layer == 8 : 
        for i in range(0,9):
            wrong_data = 7638
            true_data = wrong_data - 1
            data[('center', 'posy')][wrong_data + i] = data[('center', 'posy')][true_data]
            
            
    ### store the name of the different folder in all_folder
    all_folder = []
    
    for key, _ in genDict.items() :
        all_folder.append(key)
    """ output : all_folder = ['200206_110534_s1a10_p3-4', '200206_160327_s4a9_p3-4', '200206_105311_s1a9_p3-4', '200206_153954_s4a10_p3-4']"""
  
    return genDict, data, all_folder, simulation[first_layer]

 

#%%
def experiment_propeties(genDict, data, environment, all_folder, simulation, second_layer):      
   
    ### pick one specific folder
    folder = all_folder[second_layer]
    split_name_fodler = folder[:13]
    """output : '200206_110534'"""
    
    ### Find the first element of the folder (experiment) in the DataFrame
    index_folder = np.where(np.array(data.reset_index().iloc[:,3]) == split_name_fodler)[0][0]
    
    # =============================================================================
    # data2 = data.reset_index()
    # data3 = data2.iloc[:,3]
    # data3 = np.array(data3)
    # np.where(data3 =="200206_160327")[0][0]
    # =============================================================================
    
    
    ###  extract the nuber of frames per video and per sequence on/off
    frame_split = genDict[folder]['stimulation_paradigm']
    frame_per_period = []
    frame_per_period_tot = [0]
    third_layer = [] 
    
    for key, _ in genDict[folder].items() :
            third_layer.append(key) 
            "third_layer[2] is always the first fly, could be fly1 if fly0 doesn't exist"
        
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
        
    x_pos = x_coordinate.values[experiment.index] * environment.position_convert
    y_pos = y_coordinate.values[experiment.index] * environment.position_convert
       
    return x_pos, y_pos

#%%
def angle_coordinates(experiment, theta_coordinate):
    
    theta_pos = theta_coordinate.values[experiment.index]
    
    return theta_pos

#%%

def pos2vel(x, fps=80):
    '''
    Convert positions to velocities, and apply moving-average filter
    '''

    dx = np.array([x[:, i] - x[:, i-1] for i in range(1, x.shape[1])]).T
    dx = np.hstack((np.zeros((3,1)), dx))
    velocity = np.zeros(x.shape)
    
    for i in range(len(x)):
        velocity[i,:] =  SMA(dx[i,:]*fps)
        
    return velocity

def SMA_pos(x):
    
   return pos2vel(x, fps=1)
    
    
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
#%%
def experiment_center_pos(genDict, data, environment, all_folder, simulation, second_layer):
    
    """index correspond to the right index of all flies and all sequences for a specific experiment"""
    index = np.array([])
    experiment = experiment_propeties(genDict, data, environment, 
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
def plot_one_experiment_trajectories(genDict, experiment):
    
    exp = experiment   
    genDict_key = []
    for key, value in genDict[exp.folder].items() :
        genDict_key.append(key)
    

    #plot x and y coordinates
    for i in range(0,exp.nb_fly) :
        plt.figure(str(exp.simulation) + " " + str(exp.folder[7:13]) + " " + str(genDict_key[2+i]) + " : xy position",
                   figsize = environment.figure_size,
                   dpi = environment.dpi)
        plt.title(str(exp.simulation) + " " + str(exp.folder[7:13]) + " " + str(genDict_key[2+i]) + " : xy position")

        for j in range(1, len(exp.frame_per_period_tot)):  
            
            """ simulation off --> blue
                simulation on --> red """
            plt.plot(exp.x_pos[i, (exp.frame_per_period_tot[j-1]):exp.frame_per_period_tot[j]], 
                     exp.y_pos[i, (exp.frame_per_period_tot[j-1]):exp.frame_per_period_tot[j]], 
                     c=environment.color_stim[(j%2)])           
        
        """ plot the start and end point """
        plt.plot(exp.x_pos[i,0], exp.y_pos[i,0], c='k', marker='X')
        plt.plot(exp.x_pos[i,-1], exp.y_pos[i,-1], c='k', marker='o')

        plt.xlabel("x position [mm]")
        plt.ylabel("y position [mm]")
        plt.xlim(0, 38)
        plt.ylim(0, 38)
        plt.legend(environment.stim_legend)

        """ plot or save """
        if environment.bool_save == True:
            plt.savefig(str(exp.simulation) + " " + str(exp.folder[7:13]) + " " + str(genDict_key[2+i]) + " xy position.png")
        else:
            plt.show()
  
    return 
#%%
def plot_one_fly_trajectories(genDict, data, all_experiment, normalized = 0, fly_number = 0):
       
    """ to extract x and y position of any data.keys(), 
        example : Leye.x, Rantenna.y"""
    # =============================================================================
    #     x_pos, y_pos = any_coordinates(experiment, data.center.posx, data.center.posy)
    #     experiment.position_order(x_pos, y_pos)
    # =============================================================================

    
    k = fly_number
    genDict_key = []
    for key, value in genDict[all_experiment[0].folder].items() :
        genDict_key.append(key)
    
    
    #plot x and y coordinates
    for i in range(0,len(all_experiment)) :
        
        exp = all_experiment[i]
        
        """ take position or position_n """
        if normalized == 0 :
            x_pos = exp.x_pos
            y_pos = exp.y_pos
        else :
            x_pos = exp.x_pos_n
            y_pos = exp.y_pos_n
                
        plt.figure(str(exp.simulation) + " " + str(genDict_key[2+k]) + " " + str(exp.folder[7:13]) + " xy position",
                   figsize = environment.figure_size,
                   dpi = environment.dpi)
        plt.title(str(exp.simulation) + " " + str(genDict_key[2+k]) + " " + str(exp.folder[7:13]) + " xy position")
        
        
        for j in range(1, len(exp.frame_per_period_tot)):  

            """ simulation off --> blue
                simulation on --> red """
            plt.plot(x_pos[k, (exp.frame_per_period_tot[j-1]):exp.frame_per_period_tot[j]], 
                     y_pos[k, (exp.frame_per_period_tot[j-1]):exp.frame_per_period_tot[j]], 
                     c=environment.color_stim[(j%2)]) 
        
                
        if normalized == 0 :                       
            plt.plot(x_pos[k,-1], y_pos[k,-1], c='k', marker='o')
            
        plt.plot(x_pos[k,0], y_pos[k,0], c='k', marker='X')                 
        plt.xlabel("x position [mm]")
        plt.ylabel("y position [mm]")
        plt.legend(environment.stim_legend)
        
        if normalized == 0:
            plt.xlim(0, 38)
            plt.ylim(0, 38)
        else:
            plt.xlim(-38,38)
            plt.ylim(-38,38)
        
        
        if environment.bool_save == True:
            plt.savefig(str(exp.simulation) + " " + str(genDict_key[2+k]) + " " + str(exp.folder[7:13]) + " xy position.png")
        else:
            plt.show()
        
    return
    

#%%
def plot_x_position_over_time(genDict, data, all_experiment, all_folder, fly_number = 0):
       
    """ to extract x and y position of any data.keys(), 
        example : Leye.x, Rantenna.y"""
# =============================================================================
#     x_pos, y_pos = any_coordinates(experiment, data.center.posx_n, data.center.posy_n)
#     experiment.position_order(x_pos, y_pos)
# =============================================================================

    
    k = fly_number
    color = environment.color_plot
    which_pos = ["x position", "y position"]
    
    legend = []
    ax = []
    
    for i in range(0, len(all_folder)):
        legend.append(all_folder[i][7:13])
        
    genDict_key = []
    for key, value in genDict[all_experiment[0].folder].items() :
        
        genDict_key.append(key)
    
    fig = plt.figure(str(all_experiment[0].simulation) + " " + str(genDict_key[2+k]) + " : x and y position over time",
                     constrained_layout=False,
                     figsize = environment.figure_size,
                     dpi = environment.dpi)
    
    fig.suptitle(str(all_experiment[0].simulation) + " " + str(genDict_key[2+k]) + " : x and y position over time")

    gs = gridspec.GridSpec(2, 1, figure=fig) 
    
    for l in range(0,2):
        
        ax = fig.add_subplot(2, 1, l+1)      

       #plot x and y coordinates
        for i in range(0,len(all_experiment)) :
            exp = all_experiment[i]
            time = np.arange(0,exp.total_frame)/exp.frame_frequency
            stim_time = np.array(exp.frame_per_period_tot[1:])/exp.frame_frequency
            
                        
            # x coordinate
            if l == 0 :
                ax.plot(time, exp.x_pos_n[k, :], c=color[i])
                ax.set_xticks([])
                
            # y coordinate                
            else:
                ax.plot(time, exp.y_pos_n[k, :], c=color[i])
                ax.set_xlabel("time [s]")                
    
        ax.legend(legend, loc=3)        
        ax.set_ylabel(str(which_pos[l]) + " [mm]")
        ax.set_xlim(time[0], time[-1])
        ax.set_ylim(-35, 35)
        
           
        """ vertical line to separate each sequence"""            
        ax.axvline(0, ymin = -30, ymax = 30, c='r', ls='--')
        for j in range(0, len(exp.frame_per_period)):
            ax.axvline(stim_time[j], ymin = -30, ymax = 30, c='r', ls='--')
    
    # fig.tight_layout()
            
    if environment.bool_save == True:
        fig.savefig(str(exp.simulation) + " " + str(genDict_key[2+k]) + " x and y position over time.png")
    else:
        fig.show()
              
    return

#%%
def plot_x_position_over_time_like_Sen(genDict, data, all_experiment, all_folder, fly_number = 0):
       

    k = fly_number
    color = environment.color_plot
    which_pos = ["x position", "y position"]
    
    legend = []
    ax = []
    x_pos_all = np.array([])
    y_pos_all = np.array([])
    
    for i in range(0, len(all_folder)):
        legend.append(all_folder[i][7:13])
        
    genDict_key = []
    for key, value in genDict[all_experiment[0].folder].items() :       
        genDict_key.append(key)
        
    """ find the number of frame minimum for the same sequence"""   
    sequence_on = [1, 3, 5]
    
    min_frame = np.Inf  
    min_frame_period = np.Inf * np.ones(7)
    min_frame_period_tot = np.zeros(8)
    for j in range(0,len(all_experiment)):
        exp = all_experiment[j]
        if  exp.total_frame < min_frame:
            min_frame = exp.total_frame
        # print("experiment : %d" %(j))
        for i in range(0, len(exp.frame_per_period)):
            # print("frame %d " %(exp.frame_per_period[i]))
            if exp.frame_per_period[i] < min_frame_period[i]:
                min_frame_period[i] = exp.frame_per_period[i]
                
    for j, frame in enumerate(min_frame_period):
            min_frame_period_tot[j+1] = min_frame_period_tot[j] + frame
                        
                       
    min_frame = min_frame.astype(int) 
    # min_frame_period = np.insert(min_frame_period, 0, 0)
    min_frame_period = min_frame_period.astype(int)
    min_frame_period_tot = min_frame_period_tot.astype(int)
    print(min_frame)
    print(min_frame_period)
    print(min_frame_period_tot)
    
    
    fig = plt.figure(str(all_experiment[0].simulation) +  " : x and y position over time",
                     constrained_layout=False,
                     figsize = environment.figure_size,
                     dpi = environment.dpi)
    
    fig.suptitle(str(all_experiment[0].simulation) + " : x and y position over time")
    

    gs = gridspec.GridSpec(2, 1, figure=fig) 
    
    for l in range(0,2):
        
        ax = fig.add_subplot(2, 1, l+1) 
        ax.plot(-5, -5, c="0.8", label = "Raw data")
        ax.plot(-5, -5, c= "0", label = "Mean data")
        ax.legend(loc=1, fontsize = 6)

       #plot x and y coordinates
        for i in range(0,len(all_experiment)) :
            exp = all_experiment[i]
            time = np.arange(0,min_frame)/exp.frame_frequency
            stim_time = np.array(exp.frame_per_period_tot[1:min_frame])/exp.frame_frequency
            x_pos = np.array([])
            y_pos = np.array([])
            for j in range(0, len(exp.frame_per_period)):
                start_index = exp.frame_per_period_tot[j]
                # print("start index : %d" %(start_index))
                x_pos = np.hstack([x_pos, exp.x_pos_n[:, start_index : start_index + min_frame_period[j]]]) \
                        if x_pos.size else exp.x_pos_n[:, start_index : start_index + min_frame_period[j]]
                y_pos = np.hstack([y_pos, exp.y_pos_n[:, start_index : start_index + min_frame_period[j]]]) \
                        if y_pos.size else exp.y_pos_n[:, start_index : start_index + min_frame_period[j]]
                            
            x_pos_all = np.vstack([x_pos_all, x_pos]) if x_pos_all.size else x_pos
            y_pos_all = np.vstack([y_pos_all, y_pos]) if y_pos_all.size else y_pos
# =============================================================================
#             x_pos_all = np.vstack([x_pos_all, exp.x_pos_n[:, : min_frame]]) if x_pos_all.size else  exp.x_pos_n[:, : min_frame]
#             y_pos_all = np.vstack([y_pos_all, exp.y_pos_n[:, : min_frame]]) if y_pos_all.size else  exp.y_pos_n[:, : min_frame]
# =============================================================================
            
            for k in range(0, exp.nb_fly):
                
                                    
                # x coordinate
                if l == 0 :
                    print(x_pos.shape[1])
                    print(x_pos.shape)
                    ax.plot(time[:x_pos.shape[1]], x_pos[k,:], c="0.5")
                    ax.set_xticks([])
                    
                # y coordinate                
                else:
                    ax.plot(time[:y_pos.shape[1]], y_pos[k,:], c="0.5")
                    ax.set_xlabel("time [s]")  
                    

        
        if l == 0:
            ax.errorbar(time[:x_pos_all.shape[1]], 
                        np.mean(x_pos_all, axis=0), 
                        yerr=np.std(x_pos_all, axis=0), ecolor = "0.9", c="0")
            # ax.plot(time[:x_pos_all.shape[1]], np.mean(x_pos_all, axis=0), c="0")
        else:
            ax.errorbar(time[:y_pos_all.shape[1]], 
                        np.mean(y_pos_all, axis=0), 
                        yerr=np.std(y_pos_all, axis=0), ecolor = "0.9", c="0")
            # ax.plot(time[:y_pos_all.shape[1]], np.mean(y_pos_all, axis=0), c="0")
        ax.set_ylabel(str(which_pos[l]) + " [mm]")
        ax.set_xlim(time[0], time[-1])
        ax.set_ylim(-35, 35)
        
        
        
           
        """ vertical line to separate each sequence"""            
        # ax.axvline(0, ymin = -30, ymax = 30, c='r', ls='--')
        for j in range(0, len(exp.frame_per_period)):
            ax.axvline(min_frame_period_tot[j]/exp.frame_frequency, ymin = -30, ymax = 30, c='r', ls='--')
    
    # fig.tight_layout()
            
    if environment.bool_save == True:
        fig.savefig(str(exp.simulation) + " x and y position over time.png")
    else:
        fig.show()
              
    return


#%%
def plot_xy_position_Sen_like(genDict, data, all_experiment, all_folder):
       
    """ to extract x and y position of any data.keys(), 
        example : Leye.x, Rantenna.y"""
# =============================================================================
#     x_pos, y_pos = any_coordinates(experiment, data.center.posx_n, data.center.posy_n)
#     experiment.position_order(x_pos, y_pos)
# =============================================================================
    position = np.array([])
    legend = []
    color = ['#E58525', '#15BC26']
    color_mean = ['#FC1D1D', '#121DB3'] 

    
    for i in range(0, len(all_folder)):
        legend.append(all_folder[i][7:13])
    
    genDict_key = []
    for key, value in genDict[all_experiment[0].folder].items() :
        genDict_key.append(key)
        
    stim_key = genDict[all_experiment[0].folder]['stimulation_paradigm']
    
    """ find the number of frame minimum for the same sequence"""   
    sequence_on = [1, 3, 5]
    
    min_frame = np.Inf * np.ones(3)
    for count, i in enumerate(sequence_on):    
        for j in range(0,len(all_experiment)):
            exp = all_experiment[j]
            for k in range(0, exp.nb_fly) :
                if  exp.frame_per_period[i] < min_frame[count]:
                    min_frame[count] = exp.frame_per_period[i]
    
    min_frame = min_frame.astype(int) 
    
    #plot x and y coordinates
    for count, i in enumerate(sequence_on):  
        
        x_pos_all = np.array([])
        y_pos_all = np.array([])
        dir_all = np.array([])
        
        """ PLOT ALL THE TRAJECTORIES"""
        plt.figure(str(all_experiment[0].simulation) + " : xy position over time " +  str(stim_key[i]),
                   figsize = environment.figure_size,
                   dpi = environment.dpi)
        plt.title(str(all_experiment[0].simulation) + " : xy position over time " +  str(stim_key[i]))
        
        # bullshit point for the legend
        plt.plot(-50, -50, c=color[0], label = "< 0 mm/s (row data)")
        plt.plot(50, 50, c=color[1], label = "> 0 mm/s (row data)")
        plt.plot(-50, -50, c=color_mean[0], label = "< 0 mm/s (mean data)")
        plt.plot(50, 50, c=color_mean[1], label = "> 0 mm/s (mean data)")
        plt.legend()
        
        for j in range(0,len(all_experiment)) :
            exp = all_experiment[j]
            for k in range(0, exp.nb_fly) :
                
                direction = get_speed_direction(genDict, data, all_experiment, all_folder, fly_number = k, experiment = j, sequence = i)
                start_index = exp.frame_per_period_tot[i]
                
                direction = np.heaviside(direction[:min_frame[count]], 0)                 
                x_pos = exp.x_pos_n[k, start_index : start_index + min_frame[count]]
                y_pos = exp.y_pos_n[k, start_index : start_index + min_frame[count]]
                position = np.vstack([x_pos, y_pos]).T
                for l, (start, stop) in enumerate(zip(position[:-1], position[1:])):
                    x, y = zip(start, stop)    
                    plt.plot(x, y, color=color[int(direction[l])])
                    
                x_pos_all = np.vstack([x_pos_all, x_pos]) if x_pos_all.size else x_pos
                y_pos_all = np.vstack([y_pos_all, y_pos]) if y_pos_all.size else y_pos
                dir_all = np.vstack([dir_all, direction]) if dir_all.size else direction
                # print(x_pos_all.shape)
                # print(dir_all.shape)
# =============================================================================
#                 
#         plt.xlabel("x position [mm]")
#         plt.ylabel("y position [mm]")        
#         plt.xlim(-38,38)
#         plt.ylim(-38,38)
#         
#         if environment.bool_save == True:
#              plt.savefig(str(all_experiment[0].simulation) + " xy position over time " +  str(stim_key[i]) + ".png")
#         else:
#             plt.show() 
#             
# =============================================================================
            
            
# =============================================================================
#         """ PLOT THE MEAN TRAJECTORY"""
#         plt.figure(str(all_experiment[0].simulation) + " : mean xy position over time " +  str(stim_key[i]),
#                    figsize = environment.figure_size,
#                    dpi = environment.dpi)
#         plt.title(str(all_experiment[0].simulation) + " : mean xy position over time " +  str(stim_key[i]))
#         
#         # bullshit point for the legend
#         plt.plot(-50, -50, c='r', label = "< 0 mm/s")
#         plt.plot(50, 50, c='b', label = "> 0 mm/s")
#         plt.legend()
#         
# =============================================================================
        x_pos_mean = np.mean(x_pos_all, axis=0)
        y_pos_mean = np.mean(y_pos_all, axis=0)
        x_pos_std = np.std(x_pos_all, axis=0)
        y_pos_std = np.std(y_pos_all ,axis=0)
        position_all = np.vstack([x_pos_mean, y_pos_mean]).T
        dir_mean = np.mean(dir_all, axis=0)
        dir_mean = np.heaviside(-0.5 + dir_mean, 1)
        
        plt.errorbar(x_pos_mean, y_pos_mean, 
                     xerr=x_pos_std,
                     yerr=y_pos_std, 
                     ecolor = "0.8", c="0")
        
        for l, (start, stop) in enumerate(zip(position_all[:-1], position_all[1:])):
                    x, y = zip(start, stop)    
                    # plt.plot(x, y, color=color[int(dir_mean[l])])
# =============================================================================
#                     plt.errorbar(x, y,
#                                  xerr=x_pos_std[l],
#                                  yerr=y_pos_std[l], 
#                                  ecolor = "0.8",
#                                  c=color[int(dir_mean[l])])
# =============================================================================
                                                    
   
        plt.xlabel("x position [mm]")
        plt.ylabel("y position [mm]")                
        plt.xlim(-38,38)
        plt.ylim(-38,38)
        
# =============================================================================
#         if environment.bool_save == True:
#              plt.savefig(str(all_experiment[0].simulation) + " mean xy position over time " +  str(stim_key[i]) + ".png")
#         else:
#             plt.show()
# =============================================================================
            
        if environment.bool_save == True:
             plt.savefig(str(all_experiment[0].simulation) + " xy position over time " +  str(stim_key[i]) + ".png")
        else:
            plt.show()
       
   
    return


#%%
def plot_xy_position_over_time(genDict, data, all_experiment, all_folder, fly_number = 0):
       
    """ to extract x and y position of any data.keys(), 
        example : Leye.x, Rantenna.y"""
# =============================================================================
#     x_pos, y_pos = any_coordinates(experiment, data.center.posx_n, data.center.posy_n)
#     experiment.position_order(x_pos, y_pos)
# =============================================================================
    
    k = fly_number
    color = environment.color_plot 
    legend = []
    
    for i in range(0, len(all_folder)):
        legend.append(all_folder[i][7:13])
    
    genDict_key = []
    for key, value in genDict[all_experiment[0].folder].items() :
        genDict_key.append(key)
        
    stim_key = genDict[all_experiment[0].folder]['stimulation_paradigm']
    
    #plot x and y coordinates
    for j in range(1, len(all_experiment[0].frame_per_period_tot)): 
        plt.figure(str(all_experiment[0].simulation) + " " + str(genDict_key[2+k]) + " : xy position over time " +  str(stim_key[j-1]),
                   figsize = environment.figure_size,
                   dpi = environment.dpi)
        plt.title(str(all_experiment[0].simulation) + " " + str(genDict_key[2+k]) + " : xy position over time " +  str(stim_key[j-1]))
        
        for i in range(0,len(all_experiment)) :
            exp = all_experiment[i]                        
            plt.plot(exp.x_pos_n[k, (exp.frame_per_period_tot[j-1]):exp.frame_per_period_tot[j]], 
                     exp.y_pos_n[k, (exp.frame_per_period_tot[j-1]):exp.frame_per_period_tot[j]], c=color[i])
    
        plt.legend(legend)
        plt.xlabel("x position [mm]")
        plt.ylabel("y position [mm]")
                 
        plt.xlim(-38,38)
        plt.ylim(-38,38)
        
        if environment.bool_save == True:
             plt.savefig(str(all_experiment[0].simulation) + " " + str(genDict_key[2+k]) + " xy position over time " +  str(stim_key[j-1]) + ".png")
        else:
            plt.show()
       
   
    return

#%%
def plot_speed_like_Sen(genDict, data, all_experiment, all_folder, size = 51, polynomial = 3, frame_before = 30):

    total_velocity = np.array([])
    genDict_key = []
    stim_key = genDict[all_experiment[0].folder]['stimulation_paradigm']
    
    for key, value in genDict[all_experiment[0].folder].items() :
        genDict_key.append(key)
    
    # for i in range(0,len(all_experiment)) :
        
    """ find the number of frame minimum for the same sequence"""   
    sequence_on = [1, 3, 5]
    
    min_frame = np.Inf * np.ones(3)
    for count, i in enumerate(sequence_on):    
        for j in range(0,len(all_experiment)):
            exp = all_experiment[j]
            for k in range(0, exp.nb_fly) :
                if  exp.frame_per_period[i] < min_frame[count]:
                    min_frame[count] = exp.frame_per_period[i]
                
    # print(min_frame.astype(int))   
    min_frame = min_frame.astype(int)   
            
    for count, i in enumerate(sequence_on):  
        
        plt.figure(str(all_experiment[0].simulation) + " speed " + str(stim_key[i]),
                   figsize = environment.figure_size,
                   dpi = environment.dpi)
        plt.title(str(all_experiment[0].simulation) + " speed " + str(stim_key[i]))
        
        for j in range(0,len(all_experiment)):

            print("\nExperiment %d" %(j))
            exp = all_experiment[j]
            delta_x = SMA_pos(exp.x_pos)    
            delta_y = SMA_pos(exp.y_pos)
            theta = exp.orientation
                                   
            time = np.arange(0,int(exp.total_frame))/exp.frame_frequency
            stim_time = np.array(exp.frame_per_period_tot[1:])/exp.frame_frequency
            direction = np.zeros_like(delta_x)
            theta_effective = np.zeros_like(delta_x)              
                
            for m in range(0, delta_x.shape[0]):
                for n in range(0, delta_y.shape[1]):
                    theta_effective[m,n] = math.atan2(delta_x[m, n], (-delta_y[m, n])) * 360/(2*math.pi)
                    
                    """ find the direction in order to know if the fly is going forard or backward""" 
                    if 90 <= theta[m,n] <= 270:
                        if (theta[m,n] - 90) < (theta_effective[m, n] % 360) <= (theta[m,n] + 90):
                            direction[m, n] = 1
                        else:
                            direction[m, n] = -1
                            
                    elif theta[m, n] < 90 :
                        if (theta[m,n] - 90) < theta_effective[m, n] <= (theta[m,n] + 90):
                            direction[m, n] = 1
                        else:
                            direction[m, n] = -1
                        
                    else:
                        if (theta[m,n] - 360 - 90) < theta_effective[m, n] <= ((theta[m,n] + 90) % 360):
                            direction[m, n] = 1
                        else:
                            direction[m, n] = -1
                        
               
            delta_pos = (delta_x**2 + delta_y**2)**0.5        
            velocity = direction * delta_pos *  exp.frame_frequency        
            v_hat = savgol_filter(velocity, size, polynomial)
            
            
            start_index = exp.frame_per_period_tot[i] - frame_before
            stop_index = start_index + min_frame[count] + 2*frame_before
            print("start : %d \t stop : %d" %(start_index, stop_index))

            for k in range(0, exp.nb_fly) :
                

# =============================================================================
#                 if i == 1 :
#                     start_index = exp.frame_per_period_tot[i-1]
#                     stop_index = exp.frame_per_period_tot[i+1] + 1*exp.frame_frequency
#                     
#                 elif i == 3:
#                     start_index = exp.frame_per_period_tot[i] - 1*exp.frame_frequency
#                     stop_index = exp.frame_per_period_tot[i+1] + 1*exp.frame_frequency
#                     
#                 elif i == 5 :
#                     start_index = exp.frame_per_period_tot[i] - 1*exp.frame_frequency
#                     stop_index =  exp.frame_per_period_tot[-1]
#                     
#                 else:
#                     start_index = exp.frame_per_period_tot[i-1]
#                     stop_index = exp.frame_per_period_tot[i+1] + 1*exp.frame_frequency  
# =============================================================================
# =============================================================================
#                 if i == 5 :
#                     stop_index =  exp.frame_per_period_tot[-1]
#                 else:
#                     stop_index = exp.frame_per_period_tot[i+1] + 1*exp.frame_frequency    
# =============================================================================
                plt.plot(time[start_index : stop_index],
                         v_hat[k, start_index : stop_index],
                         c="0.8")
                # print("nb frame %s = %d" %(stim_key[i], exp.frame_per_period[i]))
# =============================================================================
#                 total_velocity = np.vstack([total_velocity, v_hat[k, start_index : stop_index]]) \
#                                             if total_velocity.size else v_hat[k, start_index : stop_index]
# 
#         plt.plot(time[start_index : stop_index],
#                  np.mean(total_velocity),
#                         c="k", linewidth = 20)
# =============================================================================

        plt.axvline(0, ymin = -50, ymax = 50, c='r', ls='--')
        for l in range(0, len(exp.frame_per_period)):
            plt.axvline(stim_time[l], ymin = -50, ymax = 50, c='r', ls='--')

        plt.xlabel("time [s]")
        plt.ylabel("velociy [mm/s]")
        plt.ylim(-50,50)
        plt.xlim(start_index/exp.frame_frequency,stop_index/exp.frame_frequency)
        
        
# =============================================================================
#         if environment.bool_save == True:
#              plt.savefig(str(exp.simulation) + " "  + str(exp.folder[7:13]) + " speed.png")
#         else:
#             plt.show()
# =============================================================================
   
    
    
    return
    
    
#%%
def plot_speed_over_time(genDict, data, all_experiment, all_folder, fly_number = 0):
       
    """ to extract x and y position of any data.keys(), 
        example : Leye.x, Rantenna.y"""
# =============================================================================
#     x_pos, y_pos = any_coordinates(experiment, data.center.posx_n, data.center.posy_n)
#     experiment.position_order(x_pos, y_pos)
# =============================================================================
    
    k = fly_number
    color = environment.color_plot
    step = 1
    legend = []
    genDict_key = []
    
    for key, value in genDict[all_experiment[0].folder].items() :
        genDict_key.append(key)
    
    for i in range(0,len(all_experiment)) :
        exp = all_experiment[i]
        
        
        delta_x = SMA_pos(exp.x_pos)    
        delta_y = SMA_pos(exp.y_pos)
        theta = exp.orientation
        
                    
        time = np.arange(0,int(exp.total_frame/step))/exp.frame_frequency
        direction = np.zeros_like(delta_x)
        theta_effective = np.zeros_like(delta_x)
        
        for m in range(0, delta_x.shape[0]):
            for n in range(0, delta_y.shape[1]):
                theta_effective[m,n] = math.atan2(delta_x[m, n], (-delta_y[m, n])) * 360/(2*math.pi)
                
                """ find the direction in order to know if the fly is going forard or backward""" 
                if 90 <= theta[m,n] <= 270:
                    if (theta[m,n] - 90) < (theta_effective[m, n] % 360) <= (theta[m,n] + 90):
                        direction[m, n] = 1
                    else:
                        direction[m, n] = -1
                        
                elif theta[m, n] < 90 :
                    if (theta[m,n] - 90) < theta_effective[m, n] <= (theta[m,n] + 90):
                        direction[m, n] = 1
                    else:
                        direction[m, n] = -1
                    
                else:
                    if (theta[m,n] - 360 - 90) < theta_effective[m, n] <= ((theta[m,n] + 90) % 360):
                        direction[m, n] = 1
                    else:
                        direction[m, n] = -1
                        
                        
                    
                
        delta_pos = (delta_x**2 + delta_y**2)**0.5        
        # convert mm to m
        velocity = direction * delta_pos *  exp.frame_frequency / (step)
        v_hat = savgol_filter(velocity, 51, 3)

        plt.figure(str(exp.simulation) + " " + str(genDict_key[2+k]) + " "  + str(exp.folder[7:13]) + " speed",
                   figsize = environment.figure_size,
                   dpi = environment.dpi)
        plt.title(str(exp.simulation) + " " + str(genDict_key[2+k]) + " "  + str(exp.folder[7:13]) + " speed")
        
        
        
        for j in range (1, len(exp.frame_per_period_tot)): 
            
# =============================================================================
#             plt.plot(time[(int(exp.frame_per_period_tot[j-1]/step)) : int((exp.frame_per_period_tot[j]/step - 1))], 
#                      v_hat[k, (int(exp.frame_per_period_tot[j-1]/step)) : int((exp.frame_per_period_tot[j]/step - 1))], 
#                      c=environment.color_stim[(j%2)])
# =============================================================================
            plt.scatter(time[(int(exp.frame_per_period_tot[j-1]/step)) : int((exp.frame_per_period_tot[j]/step - 1))], 
                        velocity[k, (int(exp.frame_per_period_tot[j-1]/step)) : int((exp.frame_per_period_tot[j]/step - 1))], 
                        c=environment.color_stim[(j%2)], s = 1)

        plt.legend(environment.stim_legend, loc=1)
        plt.xlabel("time [s]")
        plt.ylabel("velociy [mm/s]")
        plt.ylim(-50,50)
        
        if environment.bool_save == True:
             plt.savefig(str(exp.simulation) + " " + str(genDict_key[2+k]) + " "  + str(exp.folder[7:13]) + " speed.png")
        else:
            plt.show()
   
    return

#%%
def get_speed_direction(genDict, data, all_experiment, all_folder, fly_number = 0, experiment = 0, sequence = None):
    
    k = fly_number
    
    color = environment.color_plot
    legend = []
    genDict_key = []
    
    for key, value in genDict[all_experiment[0].folder].items() :
        genDict_key.append(key)
    
    exp = all_experiment[experiment]
    
    if sequence == None :
       delta_x = SMA_pos(exp.x_pos)[k, :]  
       delta_y = SMA_pos(exp.y_pos)[k, :] 
       theta = exp.orientation[k,:]
                     
    else:   
        delta_x = SMA_pos(exp.x_pos[:, exp.frame_per_period_tot[sequence] : exp.frame_per_period_tot[sequence + 1]])[k, :]   
        delta_y = SMA_pos(exp.y_pos[:, exp.frame_per_period_tot[sequence] : exp.frame_per_period_tot[sequence + 1]])[k, :] 
        theta = exp.orientation[k, exp.frame_per_period_tot[sequence] : exp.frame_per_period_tot[sequence + 1]]
        print(theta.shape)
                    
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
    
# =============================================================================
#     for m in range(0, delta_x.shape[0]):
#         for n in range(0, delta_y.shape[1]):
#             theta_effective[m,n] = math.atan2(delta_x[m, n], (-delta_y[m, n])) * 360/(2*math.pi)
#             
#             """ find the direction in order to know if the fly is going forard or backward""" 
#             if 90 <= theta[m,n] <= 270:
#                 if (theta[m,n] - 90) < (theta_effective[m, n] % 360) <= (theta[m,n] + 90):
#                     direction[m, n] = 1
#                 else:
#                     direction[m, n] = -1
#                     
#             elif theta[m, n] < 90 :
#                 if (theta[m,n] - 90) < theta_effective[m, n] <= (theta[m,n] + 90):
#                     direction[m, n] = 1
#                 else:
#                     direction[m, n] = -1
#                 
#             else:
#                 if (theta[m,n] - 360 - 90) < theta_effective[m, n] <= ((theta[m,n] + 90) % 360):
#                     direction[m, n] = 1
#                 else:
#                     direction[m, n] = -1 
# =============================================================================
   
    return direction
    

#%%
def joint_normalized(exp, x_pos_all, y_pos_all, x_pos_joint, y_pos_joint) :
    
# =============================================================================
#     x_pos_joint = x_pos_joint - exp.x_pos
#     y_pos_joint = y_pos_joint - exp.y_pos
# =============================================================================
    
    """ reshape x_pos_all ti have (3, :)"""
    # x_pos_all = np.append(x_pos_all, x_pos_joint)
    # x_pos_all = np.concatenate(x_pos_all, x_pos_joint)
    x_pos_all = np.hstack([x_pos_all, x_pos_joint]) if x_pos_all.size else x_pos_joint
    
    # y_pos_all = np.append(y_pos_all, y_pos_joint)
    # y_pos_all = np.concatenate(y_pos_all, y_pos_joint)
    y_pos_all = np.hstack([y_pos_all, y_pos_joint]) if y_pos_all.size else y_pos_joint

    
    
    return x_pos_all, y_pos_all

#%%
def joint_position_over_time(genDict, data, all_experiment, all_folder, fly_number = 0):
    
# =============================================================================
#     """ to extract x and y position of any data.keys(), 
#         example : Leye.x, Rantenna.y"""
# =============================================================================

    k = fly_number
    color = environment.color_plot
    step = 1
    legend = []
    genDict_key = []
    stim_key = genDict[all_experiment[0].folder]['stimulation_paradigm']
    legend = ["Left Fore", "Left Middle", "Left Hind", "Right Fore", "Right Middle", "Right Hind"]
    # legend = ["Left fft", "Left", "Right fft", "Right"]
    x_plot_interval = [0, 0, -1.5, 0, 0, 1.5]

    
    for key, value in genDict[all_experiment[0].folder].items() :
        genDict_key.append(key)
      
    
    for i in range(0,len(all_experiment)) :
    # for i in range(0,1) :
        exp = all_experiment[i]  

        #plot x and y coordinates of the Tarus

        x_pos_all = np.array([])
        y_pos_all = np.array([])
        time = np.arange(0,int(exp.total_frame/step))/exp.frame_frequency
                    
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
        
# =============================================================================
#         x_pos_LF, y_pos_LF = any_coordinates(exp, data.LFtibiaTarsus.x, data.LFtibiaTarsus.y)
#         x_pos_all, y_pos_all = joint_normalized(exp, x_pos_all, y_pos_all, x_pos_LF, y_pos_LF)            
#         
#         x_pos_LM, y_pos_LM = any_coordinates(exp, data.LMtibiaTarsus.x, data.LMtibiaTarsus.y)
#         x_pos_all, y_pos_all = joint_normalized(exp, x_pos_all, y_pos_all, x_pos_LM, y_pos_LM)
#         
#         x_pos_LH, y_pos_LH = any_coordinates(exp, data.LHtibiaTarsus.x, data.LHtibiaTarsus.y)
#         x_pos_all, y_pos_all = joint_normalized(exp, x_pos_all, y_pos_all, x_pos_LH, y_pos_LH)
#         
#         x_pos_RF, y_pos_RF = any_coordinates(exp, data.RFtibiaTarsus.x, data.RFtibiaTarsus.y)
#         x_pos_all, y_pos_all = joint_normalized(exp, x_pos_all, y_pos_all, x_pos_RF, y_pos_RF)
#         
#         x_pos_RM, y_pos_RM = any_coordinates(exp, data.RMtibiaTarsus.x, data.RMtibiaTarsus.y)
#         x_pos_all, y_pos_all = joint_normalized(exp, x_pos_all, y_pos_all, x_pos_RM, y_pos_RM)
#            
#         x_pos_RH, y_pos_RH = any_coordinates(exp, data.RHtibiaTarsus.x, data.RHtibiaTarsus.y)
#         x_pos_all, y_pos_all = joint_normalized(exp, x_pos_all, y_pos_all, x_pos_RH, y_pos_RH)
# =============================================================================
  

# =============================================================================
#         """ X POSITION"""
#         for j in range(0, len(exp.frame_per_period_tot)-1): 
#             
#             """ plotting the y position"""
#             plt.figure(str(exp.simulation) + " " + str(exp.folder[7:13]) + " " + str(genDict_key[2+k]) + " : Claw x position " +  str(stim_key[j]),
#                        figsize = environment.figure_size,
#                        dpi = environment.dpi)
#             plt.title(str(exp.simulation) + " " + str(exp.folder[7:13]) + " " + str(genDict_key[2+k]) + " : Claw x position " +  str(stim_key[j]))
# 
#             for l in range(0,6) :
#                 start_index = l*exp.frame_per_fly
#                 plt.plot(x_pos_all[k, start_index + exp.frame_per_period_tot[j] : start_index + exp.frame_per_period_tot[j+1]] + x_plot_interval[l],
#                          time[exp.frame_per_period_tot[j] : exp.frame_per_period_tot[j+1]])
#                                        
#             plt.xlabel("x joint position [mm]")
#             plt.ylabel("time")
#             plt.xticks([])
#             plt.xlim(-0.5 ,6.5)
#             plt.legend(legend, loc=1, fontsize = 6)
#             
#             """ plot or save """
#             if environment.bool_save == True:
#                 plt.savefig(str(exp.simulation) + " " + str(exp.folder[7:13]) + " " + str(genDict_key[2+k]) + " Claw x position " +  str(stim_key[j]) + ".png")
#             else:
#                 plt.show()
#             
#             
#             
#             
#         """ plotting the y position all in one"""
#         plt.figure(str(exp.simulation) + " " + str(exp.folder[7:13]) + " " + str(genDict_key[2+k]) + " : Claw x position",
#                    figsize = environment.figure_size,
#                    dpi = environment.dpi)
#         plt.title(str(exp.simulation) + " " + str(exp.folder[7:13]) + " " + str(genDict_key[2+k]) + " : Claw x position")
#             
#         for l in range(0,6) :               
#             plt.plot(x_pos_all[k, l*exp.frame_per_fly : (l+1)*exp.frame_per_fly] + x_plot_interval[l], 
#                      time)
#                                        
#         plt.xlabel("x joint position [mm]")
#         plt.ylabel("time")
#         plt.ylim(0,30)
#         plt.xlim(-0.5 ,6.5)
#         plt.legend(legend, loc=1, fontsize = 6)
#         
#         
#         """ vertical line to separate each sequence"""      
#         stim_time = np.array(exp.frame_per_period_tot[1:])/exp.frame_frequency
#         plt.axhline(0, xmin = -30, xmax = 30, c='k', ls='--')
#         for l in range(0, len(exp.frame_per_period)):
#             plt.axhline(stim_time[l], xmin = -30, xmax = 30, c='k', ls='--')
#         
#         """ plot or save """
#         if environment.bool_save == True:
#             plt.savefig(str(exp.simulation) + " " + str(exp.folder[7:13]) + " " + str(genDict_key[2+k]) + " Claw x position.png")
#         else:
#             plt.show()      
# =============================================================================




        """ Y POSITION"""
        for j in range(0, len(exp.frame_per_period_tot)-1): 
            
            """ plotting the y position"""
            plt.figure(str(exp.simulation) + " " + str(exp.folder[7:13]) + " " + str(genDict_key[2+k]) + " : Claw y position " +  str(stim_key[j]),
                       figsize = environment.figure_size,
                       dpi = environment.dpi)
            plt.title(str(exp.simulation) + " " + str(exp.folder[7:13]) + " " + str(genDict_key[2+k]) + " : Claw y position " +  str(stim_key[j]))

            for l in range(0,6) :
                start_index = l*exp.frame_per_fly
                plt.plot(time[: exp.frame_per_period[j]], 
                         y_pos_all[k, start_index + exp.frame_per_period_tot[j] : start_index + exp.frame_per_period_tot[j+1]])
                                       
            plt.ylabel("y joint position [mm]")
            plt.xlabel("time")
            plt.ylim(5.5,-0.5)
            plt.xlim(0, time[exp.frame_per_period[j]])
            plt.legend(legend, loc=1, fontsize = 6)
            
            """ plot or save """
            if environment.bool_save == True:
                plt.savefig(str(exp.simulation) + " " + str(exp.folder[7:13]) + " " + str(genDict_key[2+k]) + " Claw y position " +  str(stim_key[j]) + ".png")
            else:
                plt.show()
            
           
        """ plotting the y position all in one"""
        plt.figure(str(exp.simulation) + " " + str(exp.folder[7:13]) + " " + str(genDict_key[2+k]) + " : Claw y position",
                   figsize = environment.figure_size,
                   dpi = environment.dpi)
        plt.title(str(exp.simulation) + " " + str(exp.folder[7:13]) + " " + str(genDict_key[2+k]) + " : Claw y position")
            
        for l in range(0,6) :               
            plt.plot(time, y_pos_all[k, l*exp.frame_per_fly : (l+1)*exp.frame_per_fly])
                                       
        plt.ylabel("y joint position [mm]")
        plt.xlabel("time")
        plt.xlim(0,30)
        plt.ylim(5.5,-0.5)
        plt.legend(legend, loc=1, fontsize = 6)
        
        
        """ vertical line to separate each sequence"""      
        stim_time = np.array(exp.frame_per_period_tot[1:])/exp.frame_frequency
        plt.axvline(0, ymin = -30, ymax = 30, c='k', ls='--')
        for l in range(0, len(exp.frame_per_period)):
            plt.axvline(stim_time[l], ymin = -30, ymax = 30, c='k', ls='--')
        
        """ plot or save """
        if environment.bool_save == True:
            plt.savefig(str(exp.simulation) + " " + str(exp.folder[7:13]) + " " + str(genDict_key[2+k]) + " Claw y position.png")
        else:
            plt.show()
        
        
        
        
# =============================================================================
#         """ SPEED OF LIMB"""
#         plt.figure(str(exp.simulation) + " " + str(exp.folder[7:13]) + " " + str(genDict_key[2+k]) + " : Claw y speed",
#                    figsize = environment.figure_size,
#                    dpi = environment.dpi)
#         plt.title(str(exp.simulation) + " " + str(exp.folder[7:13]) + " " + str(genDict_key[2+k]) + " : Claw y speed")
#                 
#         gait_pattern = np.zeros((11*6, exp.frame_per_period[j]))
#         legend_postion = 11*np.arange(6) + int(11/2)
#     
#         for l in range(0,6) :
#             start_index = l*exp.frame_per_fly
# # =============================================================================
# #             delta_pos = np.array([np.subtract(y_pos_all[k, i+1], y_pos_all[k, i]) for i in range(l*exp.frame_per_fly, (l+1)*exp.frame_per_fly - 1)])
# #             SMA_pos = np.array([sum(delta_pos[(i - 5): i]/5) for i in range(5, len(delta_pos))])
# #             SMA_speed = SMA_pos/exp.frame_frequency
# # =============================================================================
#             SMA_speed = pos2vel(y_pos_all[:, start_index + exp.frame_per_period_tot[j] : start_index + exp.frame_per_period_tot[j+1]])[k,:]
#             speed_sign = np.heaviside(SMA_speed, 0)
#             speed_sign = np.repeat(speed_sign.reshape((1, len(speed_sign))), 10, axis = 0)
#             # speed_sign = np.sign(SMA_speed)
#             
#             """ plot the on1 sequence"""
#             gait_pattern[11*l : 11*l + 10, :] = speed_sign
#             
#         # plt.spy(gait_pattern[:, exp.frame_per_period_tot[1]:exp.frame_per_period_tot[2]], markersize=1, aspect = "equal", c='k')
#         plt.spy(gait_pattern, markersize=1, aspect = "equal", c='k')
#         plt.xticks([])
#         plt.yticks(legend_postion, ("LF", "LM", "LH", "RF", "RM", "RH"))
#         plt.show()
# =============================================================================

    
    

    return

#%%
def gait_cycle(genDict, data, all_experiment, all_folder, fly_number = 0):
    
# =============================================================================
#     """ to extract x and y position of any data.keys(), 
#         example : Leye.x, Rantenna.y"""
# =============================================================================

    k = fly_number
    color = environment.color_plot
    legend = []
    genDict_key = []
    stim_key = genDict[all_experiment[0].folder]['stimulation_paradigm']
    legend = ["Left Fore", "Left Middle", "Left Hind", "Right Fore", "Right Middle", "Right Hind"]
    x_plot_interval = [0, -0.5, -1.5, 0, 0.5, 1.5]

    gait_pattern_size = [11, 11, 11, 11, 11, 11, 11]
    gait_interval = [1, 1, 1, 1, 1, 1, 1]
    
    for key, value in genDict[all_experiment[0].folder].items() :
        genDict_key.append(key)
      
    
    # for i in range(0,len(all_experiment)) :
    for i in range(0,1) :
        exp = all_experiment[i] 
        

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

        
        
        # for j in range(0, len(exp.frame_per_period_tot)-1):
        for j in range(2,3):
            """ SPEED OF LIMB"""
            plt.figure(str(exp.simulation) + " " + str(exp.folder[7:13]) + " " + str(genDict_key[2+k]) + " : Claw gait cycle " + str(stim_key[j]),
                       figsize = environment.figure_size,
                       dpi = environment.dpi)
            plt.title(str(exp.simulation) + " " + str(exp.folder[7:13]) + " " + str(genDict_key[2+k]) + " : Claw gait cycle " + str(stim_key[j]))
                    
            gait_pattern = np.zeros((11*6, exp.frame_per_period[j]))
            y_legend_postion = 11*np.arange(6) + 5

            
            # print("j is equal : %d" %(j))
            for l in range(0,6) :
                start_index = l*exp.frame_per_fly
                SMA_speed = pos2vel(y_pos_all[:, start_index + exp.frame_per_period_tot[j] : start_index + exp.frame_per_period_tot[j+1]])[k,:]
                
                
                ### find the direction of the fly, forward speed or backward speed
                direction = get_speed_direction(genDict, data, all_experiment, all_folder, 
                                                fly_number = k, 
                                                experiment = i, 
                                                sequence = j)
# =============================================================================
#                 direction = get_speed_direction(genDict, data, all_experiment, all_folder, 
#                                                 fly_number = k, 
#                                                 experiment = i, 
#                                                 sequence = j)[k, :]
# =============================================================================
                 
                """ vitesse negative --> stance leg"""
                SMA_speed_threshold = [0 if -1.5 < speed < 1.5 else speed for speed in SMA_speed]
                speed_sign = np.heaviside(direction*SMA_speed_threshold, 1) 
                speed_sign = np.repeat(speed_sign.reshape((1, len(speed_sign))), 11 - 1, axis = 0)

                """ plot the on1 sequence"""
                gait_pattern[11*l : 11*(l+1) - 1, :] = speed_sign
            
            if j == 2 or j == 4 :
                plt.spy(gait_pattern[:, :exp.frame_per_period[j-1]], markersize=1, aspect = "equal", c='k')
            else :
                plt.spy(gait_pattern, markersize=1, aspect = "equal", c='k')
# =============================================================================
#                 plt.plot(SMA_speed_thresold)
# =============================================================================

            plt.xticks([])
            plt.yticks(y_legend_postion, ("LF", "LM", "LH", "RF", "RM", "RH"))
            # plt.ylim(-5,5)

            """ plot or save """
            if environment.bool_save == True:
                plt.savefig(str(exp.simulation) + " " + str(exp.folder[7:13]) + " " + str(genDict_key[2+k]) + " Claw gait cycle " + str(stim_key[j]) + ".png")
            else:
                plt.show()

    
    

    return
#%%
def spectral_statistics(y, window = 10, fs = 1):
    """
    Compute mean frequency

    :param y: 1-d signal
    :param fs: sampling frequency [Hz]
    :return: mean frequency
    """
    mean = np.array([])   
    # spec = np.abs(np.fft.rfft(y[i : i + window]))
        
    for i in range(0, len(y) - window):
        spec = np.abs(np.fft.rfft(y[i : i + window]))
        freq = np.fft.rfftfreq(window, d=1/fs)    
        amp = spec/ spec.sum()
        mean = np.append(mean, (freq * amp).sum())
        
    for j in range(window):
        spec = np.abs(np.fft.rfft(y[i+j : -1]))
        freq = np.fft.rfftfreq((window-j), d=1/fs)    
        amp = spec/ spec.sum()
        mean = np.append(mean, (freq * amp).sum())
    
    return mean
#%%
def frequency_cycle2(genDict, data, all_experiment, all_folder, experiment = 0, fly_number = 0, window = 100, speed_threshold = 1.5):
    
    k = fly_number
    color = environment.color_plot
    legend = []
    genDict_key = []
    stim_key = genDict[all_experiment[0].folder]['stimulation_paradigm']
    legend = ["Left Fore", "Left Middle", "Left Hind", "Right Fore", "Right Middle", "Right Hind"]
    legend_short = ["LF", "LM", "LH", "RF", "RM", "RH"]
    x_plot_interval = [0, -0.5, -1.5, 0, 0.5, 1.5]

    gait_pattern_size = [11, 11, 11, 11, 11, 11, 11]
    gait_interval = [1, 1, 1, 1, 1, 1, 1]
    
    for key, value in genDict[all_experiment[0].folder].items() :
        genDict_key.append(key)
      
    
    # for i in range(0,len(all_experiment)) :
    for i in range(experiment,experiment+1) :
        exp = all_experiment[i] 
        

        spectral_freqs = np.array([])
        x_pos_all = np.array([])
        y_pos_all = np.array([])
        time = np.arange(0,int(exp.total_frame))/exp.frame_frequency
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

        
        
        # for j in range(0, len(exp.frame_per_period_tot)-1):
        # for j in range(2,3):
        """ SPEED OF LIMB"""
        plt.figure(str(exp.simulation) + " " + str(exp.folder[7:13]) + " " + str(genDict_key[2+k]) + " : Claw gait cycle ",
                   figsize = environment.figure_size,
                   dpi = environment.dpi)
        plt.title(str(exp.simulation) + " " + str(exp.folder[7:13]) + " " + str(genDict_key[2+k]) + " : Claw gait cycle ")
        
        # print("j is equal : %d" %(j))
        for l in range(0,6) :
        # for l in [0,3] :    
            gait_frequency_tot = np.array([])
            start_index = l*exp.frame_per_fly                
            
            ### find the direction of the fly, forward speed or backward speed
            direction = get_speed_direction(genDict, data, all_experiment, all_folder, 
                                            fly_number = k, 
                                            experiment = i)
            
            
            SMA_speed = pos2vel(y_pos_all[:, l*exp.frame_per_fly  : (l+1)*exp.frame_per_fly ])[k,:]
            SMA_speed = direction*SMA_speed
             
            """ vitesse negative --> stance leg"""
            SMA_speed_threshold = np.array([0 if -1*speed_threshold < speed < speed_threshold else speed for speed in SMA_speed])  
            speed_sign = np.heaviside(direction*SMA_speed_threshold, 1) 
            
            for j in range(0, exp.frame_per_fly - window):
                edge_detector = np.hstack((0, [speed_sign[m+1] - speed_sign[m] for m in range(j, window + j)]))
                first_occ = np.argmax(edge_detector)
                last_occ = len(edge_detector[::-1]) - np.argmax(edge_detector[::-1]) - 1
                if last_occ == first_occ :
                    gait_frequency = 0
                else :
                    gait_frequency = exp.frame_frequency * np.count_nonzero(edge_detector == 1)/(last_occ - first_occ)
                # gait_frequency = exp.frame_frequency * np.count_nonzero(edge_detector == 1)/len(edge_detector)
                gait_frequency_tot = np.append(gait_frequency_tot, gait_frequency)
            
                # print("gait frequency  = %.2f [cycle/s]" %(gait_frequency))
                
            print(gait_frequency_tot.shape)

            plt.plot(time[:-window], gait_frequency_tot)
        plt.ylabel("mean frequency over %d frames" %(window))
        plt.xlabel("time [s]")
        plt.legend(legend_short)
        
        plt.axvline(0, ymin = 0, ymax = 30, c='k', ls='--')
        for j in range(0, len(exp.frame_per_period)):
            plt.axvline(stim_time[j], ymin = 0, ymax = 30, c='k', ls='--')



# =============================================================================
#         """ plot or save """
#         if environment.bool_save == True:
#             plt.savefig(str(exp.simulation) + " " + str(exp.folder[7:13]) + " " + str(genDict_key[2+k]) + " Claw gait cycle.png")
#         else:
#             plt.show()
# 
# =============================================================================
    
    return

#%%
def frequency_cycle(genDict, data, all_experiment, all_folder, experiment = 0, fly_number = 0, window = 80, speed_threshold = 1.5):
    
    k = fly_number
    color = environment.color_plot
    legend = []
    genDict_key = []
    stim_key = genDict[all_experiment[0].folder]['stimulation_paradigm']
    legend = ["Left Fore", "Left Middle", "Left Hind", "Right Fore", "Right Middle", "Right Hind"]
    legend_short = ["LF", "LM", "LH", "RF", "RM", "RH"]
    x_plot_interval = [0, -0.5, -1.5, 0, 0.5, 1.5]

    gait_pattern_size = [11, 11, 11, 11, 11, 11, 11]
    gait_interval = [1, 1, 1, 1, 1, 1, 1]
    
    for key, value in genDict[all_experiment[0].folder].items() :
        genDict_key.append(key)
      
    
    for i in range(0,len(all_experiment)) :
    # for i in range(experiment,experiment+1) :
        exp = all_experiment[i] 
        

        spectral_freqs = np.array([])
        x_pos_all = np.array([])
        y_pos_all = np.array([])
        time = np.arange(0,int(exp.total_frame))/exp.frame_frequency
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
        
            # for j in range(0, len(exp.frame_per_period_tot)-1):
            # for j in range(2,3):
            """ SPEED OF LIMB"""
            plt.figure(str(exp.simulation) + " " + str(exp.folder[7:13]) + " " + str(genDict_key[2+k]) + " : Claw gait cycle ",
                       figsize = environment.figure_size,
                       dpi = environment.dpi)
            plt.title(str(exp.simulation) + " " + str(exp.folder[7:13]) + " " + str(genDict_key[2+k]) + " : Claw gait cycle ")
            
            # print("j is equal : %d" %(j))
            for l in range(0,6) :
            # for l in [0,3] :    
                start_index = l*exp.frame_per_fly                
                
                ### find the direction of the fly, forward speed or backward speed
                direction = get_speed_direction(genDict, data, all_experiment, all_folder, 
                                                fly_number = k, 
                                                experiment = i)
    # =============================================================================
    #             direction = get_speed_direction(genDict, data, all_experiment, all_folder, 
    #                                             fly_number = k, 
    #                                             experiment = i)[k, :]
    # =============================================================================
                
                SMA_speed = pos2vel(y_pos_all[:, l*exp.frame_per_fly  : (l+1)*exp.frame_per_fly ])[k,:]
                SMA_speed = direction*SMA_speed
                
                 
                """ vitesse negative --> stance leg"""
                SMA_speed_threshold = np.array([0 if -1*speed_threshold < speed < speed_threshold else speed for speed in SMA_speed])  
                speed_sign = np.heaviside(direction*SMA_speed_threshold, 1) 
    # =============================================================================
    #             spectral_freqs = spectral_statistics(SMA_speed_threshold, window = window, fs = exp.frame_frequency)
    # =============================================================================
    # =============================================================================
    #             plt.plot(SMA_speed_threshold)
    # =============================================================================
                freqs = spectral_statistics(speed_sign, window = window, fs = exp.frame_frequency)
                spectral_freqs = np.vstack([spectral_freqs, freqs]) if spectral_freqs.size else freqs
                plt.plot(time, freqs)
                # plt.plot(time, spectral_freqs[l, :])
            plt.ylabel("mean frequency over %d frames" %(window))
            plt.xlabel("time [s]")
            plt.legend(legend_short, loc=3)
            
            plt.axvline(0, ymin = 0, ymax = 30, c='k', ls='--')
            for j in range(0, len(exp.frame_per_period)):
                plt.axvline(stim_time[j], ymin = 0, ymax = 30, c='k', ls='--')
    
            plt.ylim(-1,16)
    
    
            """ plot or save """
            if environment.bool_save == True:
                plt.savefig(str(exp.simulation) + " " + str(exp.folder[7:13]) + " " + str(genDict_key[2+k]) + " window " + str(window)+" Claw gait cycle.png")
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
                                      sequence = 1, xmin_time = None, xmax_time = None, markersize = 2, backwards = False):
    
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
      
    
    # for i in range(0,1) :
    # exp = all_experiment[i]  

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
    
    
    
    # for j in range(0, len(exp.frame_per_period_tot)-1): 
        
    fig = plt.figure(str(exp.simulation) + " " + str(exp.folder[7:13]) + " " + str(genDict_key[2+k]) + \
                        " : position and gait pattern " +  str(stim_key[j]),
                        figsize = environment.figure_size,
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
        ax_legend.plot(-1, -1)
        
        """ GAIT PATTERN"""
        SMA_speed = pos2vel(y_pos_all[:, start_index + exp.frame_per_period_tot[j] : start_index + exp.frame_per_period_tot[j+1]])[k,:] 
        # SMA_speed_threshold = np.array([0 if -1.5 < speed < 1.5 else speed for speed in SMA_speed])
        SMA_speed_threshold = SMA_speed
        
        ### find the direction of the fly, forward speed or backward speed
        direction = get_speed_direction(genDict, data, all_experiment, all_folder, 
                                        fly_number = fly_number, 
                                        experiment = experiment, 
                                        sequence = sequence)
# =============================================================================
#         direction = get_speed_direction(genDict, data, all_experiment, all_folder, 
#                                         fly_number = fly_number, 
#                                         experiment = experiment, 
#                                         sequence = sequence)[k, :]
# =============================================================================
        
        if backwards == True :
            # apply -1 to have the stance phase in black and the swing phase in white 
            speed_sign = np.heaviside(direction*SMA_speed_threshold, 1) 
        else:             
            speed_sign = np.heaviside(SMA_speed_threshold, 1)
        speed_sign = np.repeat(speed_sign.reshape((1, len(speed_sign))), 11 - 1, axis = 0)
        
# =============================================================================
#         duty_factor = np.count_nonzero(speed_sign[1,:])/speed_sign.shape[1]
#         print("%s duty factor = %.2f" %(legend_short[l], duty_factor))
# =============================================================================

        

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
    ax_legend.legend(legend, fontsize = 12)

        
    ax_gait.set_xticks([])
    
    plt.sca(ax_gait)
    plt.yticks(y_legend_postion, legend_short)
 
    plt.suptitle(str(exp.simulation) + " " + str(exp.folder[7:13]) + " " + str(genDict_key[2+k]) + \
                 " : position and gait pattern " +  str(stim_key[j]))
        
    """ plot or save """
    if environment.bool_save == True:
        if backwards == True:
            fig.savefig(str(exp.simulation) + " " + str(exp.folder[7:13]) + " " + str(genDict_key[2+k]) + \
                 " position and gait pattern " +  str(stim_key[j]) + " backwards.png")
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
first_layer = 0
second_layer = 0
normalized = 0
bool_save = True
all_experiment = []


environment = Environment(bool_save = bool_save)
genDict, data, all_folder, simulation = general_data(first_layer)


for i in range(0,4):
    second_layer = i
    experiment = experiment_center_pos(genDict, data, environment, all_folder, simulation, second_layer)
    all_experiment.append(experiment)
    
# frequency_cycle(genDict, data, all_experiment, all_folder, experiment = 0, fly_number = 0, window = 100, speed_threshold = 1.5)
    
# plot_xy_position_Sen_like(genDict, data, all_experiment, all_folder)
# plot_x_position_over_time_like_Sen(genDict, data, all_experiment, all_folder, fly_number = 0)
# plot_speed_like_Sen(genDict, data, all_experiment, all_folder, size = 51, polynomial = 3)    
# frequency_cycle(genDict, data, all_experiment, all_folder, fly_number = 0)
# joint_position_over_time_specific(genDict, data, all_experiment, all_folder, fly_number = 2, experiment = 2,sequence = 1, xmin_time = 0.3, xmax_time = 2.3)

# =============================================================================
# for k in tqdm(range(0,3)):
#     joint_position_over_time(genDict, data, all_experiment, all_folder, fly_number = k)
#     gait_cycle(genDict, data, all_experiment, all_folder, fly_number = k)
#     plt.close("all")
#     
#     plot_one_fly_trajectories(genDict, data, all_experiment, normalized = normalized, fly_number = k)
#     plot_x_position_over_time(genDict, data, all_experiment, all_folder, fly_number = k)
#     plot_xy_position_over_time(genDict, data, all_experiment, all_folder, fly_number = k)
#     plot_speed_over_time(genDict, data, all_experiment, all_folder, fly_number = k)
#     plt.close("all")
# =============================================================================

