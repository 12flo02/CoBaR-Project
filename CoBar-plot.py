
import pickle
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from tqdm import tqdm

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
    

# =============================================================================
#     k = fly_number
#     color = environment.color_plot
#     which_pos = ["x position", "y position"]
#     
#     legend = []
#     
#     for l in range(0,2):
#         for i in range(0, len(all_folder)):
#             legend.append(all_folder[i][7:13])
#         
#         genDict_key = []
#         for key, value in genDict[all_experiment[0].folder].items() :
#             genDict_key.append(key)
#         
#         
#         #plot x and y coordinates
#         for i in range(0,len(all_experiment)) :
#             exp = all_experiment[i]
#             time = np.arange(0,exp.total_frame)/exp.frame_frequency
#             stim_time = np.array(exp.frame_per_period_tot[1:])/exp.frame_frequency
#             
#             
#             plt.figure(str(exp.simulation) + " " + str(genDict_key[2+k]) + " : " + str(which_pos[l]) + " over time",
#                        figsize = environment.figure_size,
#                        dpi = environment.dpi)
#             plt.title(str(exp.simulation) + " " + str(genDict_key[2+k]) + " : " + str(which_pos[l]) + " over time")
#             
#             # x coordinate
#             if l == 0 :
#                 plt.plot(time, exp.x_pos_n[k, :], c=color[i])
#                 
#             # y coordinate                
#             else:
#                 plt.plot(time, exp.y_pos_n[k, :], c=color[i])
#                 
#     
#     
#         plt.figure(str(exp.simulation) + " " + str(genDict_key[2+k]) + " : " + str(which_pos[l]) + " over time")
#         plt.legend(legend, loc=1)
#         plt.xlabel("time [s]")
#         plt.ylabel(str(which_pos[l]) + " [mm]")
#         plt.xlim(time[0], time[-1])
#             
#         """ vertical line to separate each sequence"""            
#         plt.axvline(0, ymin = -30, ymax = 30, c='r', ls='--')
#         for j in range(0, len(exp.frame_per_period)):
#             plt.axvline(stim_time[j], ymin = -30, ymax = 30, c='r', ls='--')
#             
#             
#         if environment.bool_save == True:
#             print("saving fig")
#             print(str(exp.simulation) + " " + str(genDict_key[2+k]) + " " + str(which_pos[l]) + " over time.png")
#             plt.savefig(str(exp.simulation) + " " + str(genDict_key[2+k]) + " " + str(which_pos[l]) + " over time.png")
#         else:
#             plt.show()
# =============================================================================
          
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
        
        
        delta_x = np.array([exp.x_pos[:, step*n] - exp.x_pos[:, step*(n-1)] \
                            for n in range(1, int(exp.frame_per_period_tot[-1]/step))]).T
            
        delta_y = np.array([exp.y_pos[:, n] - exp.y_pos[:, n-1] \
                            for n in range(1, int(exp.frame_per_period_tot[-1]/step))]).T
            
        theta = np.array([exp.orientation[:, step*n] \
                          for n in range(0, int(exp.frame_per_period_tot[-1]/step) - 1)]).T    
                    
        time = np.arange(0,int(exp.total_frame/step) - 1)/exp.frame_frequency
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
        velocity = direction * delta_pos *  exp.frame_frequency / (step*100)

        plt.figure(str(exp.simulation) + " " + str(genDict_key[2+k]) + " "  + str(exp.folder[7:13]) + " speed",
                   figsize = environment.figure_size,
                   dpi = environment.dpi)
        plt.title(str(exp.simulation) + " " + str(genDict_key[2+k]) + " "  + str(exp.folder[7:13]) + " speed")
        
        
        
        for j in range (1, len(exp.frame_per_period_tot)): 
            
            plt.scatter(time[(int(exp.frame_per_period_tot[j-1]/step)) : int((exp.frame_per_period_tot[j]/step - 1))], 
                        velocity[k, (int(exp.frame_per_period_tot[j-1]/step)) : int((exp.frame_per_period_tot[j]/step - 1))], 
                        c=environment.color_stim[(j%2)], s = 1)

        plt.legend(environment.stim_legend, loc=1)
        plt.xlabel("time [s]")
        plt.ylabel("velociy [m/s]")
        plt.ylim(-0.5,0.5)
        
        if environment.bool_save == True:
             plt.savefig(str(exp.simulation) + " " + str(genDict_key[2+k]) + " "  + str(exp.folder[7:13]) + " speed.png")
        else:
            plt.show()
   
    return

#%%
def orientation_over_time(genDict, data, all_experiment, all_folder, fly_number = 0):
    
    

#%%
# =============================================================================
# def joint_position_over_time(genDict, data, all_experiment, all_folder, fly_number = 0):
#     
# # =============================================================================
# #     """ to extract x and y position of any data.keys(), 
# #         example : Leye.x, Rantenna.y"""
# # =============================================================================
#         
#     x_pos, y_pos = any_coordinates(experiment, data.center.posx_n, data.center.posy_n)
#     experiment.position_order(x_pos, y_pos)
# 
# 
#     
#     
#     return
# =============================================================================

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


# plot_one_experiment_trajectories(genDict, all_experiment[0])
# plot_x_position_over_time(genDict, data, all_experiment, all_folder, fly_number = 0)

for k in tqdm(range(0,3)):
    plot_one_fly_trajectories(genDict, data, all_experiment, normalized = normalized, fly_number = k)
    plot_x_position_over_time(genDict, data, all_experiment, all_folder, fly_number = k)
    plot_xy_position_over_time(genDict, data, all_experiment, all_folder, fly_number = k)
    plot_speed_over_time(genDict, data, all_experiment, all_folder, fly_number = k)
    # plt.close("all")








