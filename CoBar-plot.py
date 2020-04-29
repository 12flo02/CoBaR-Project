
import pickle
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

plt.rc('lines', linewidth=1.0)
plt.rc('font', size=8.0)


class Environment:
    ### environment variables 
    ### convert the pixel width (832) in [mm] --> environment = 38[mm]x38[mm]
    def __init__(self):
        self.enivronment_size = 38
        self.nb_pixel = 832
        self.position_convert = self.enivronment_size/self.nb_pixel
        self.stim_legend = ["stimulation on", "stimulation off"]
        self.color_plot = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple', 'tab:brown']
        
    
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

    def position_order(self, x_pos, y_pos):
        self.x_pos = x_pos
        self.y_pos = y_pos
        
    def position_n_order(self, x_pos_n, y_pos_n):
        self.x_pos_n = x_pos_n
        self.y_pos_n = y_pos_n
        
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
    
    data_path = ['CoBar-Dataset/']
    data_path.append(simulation[first_layer])
    data_path.append('/U3_f/')
    data_path.append(simulation[first_layer])
    data_path.append('_U3_f_trackingData.pkl')
    data_path = ''.join(data_path)
    "output : 'CoBar-Dataset/MDN/U3_f/MDN_U3_f_trackingData.pkl'"
    
    gen_path = ['CoBar-Dataset/']
    gen_path.append(simulation[first_layer])
    gen_path.append('/U3_f/genotype_dict.npy')
    gen_path = ''.join(gen_path)
    "output : 'CoBar-Dataset/MDN/U3_f/genotype_dict.npy'"
    

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
        
    # print(experiment.__dict__.keys())                           

    return experiment


#%%
def any_coordinates(experiment, x_coordinate, y_coordinate):
    
    
    x_pos = x_coordinate.values[experiment.index] * environment.position_convert
    y_pos = y_coordinate.values[experiment.index] * environment.position_convert
       
    
# =============================================================================
#     data.center.posx.values[all_experiment[0].index].shape
#     data.center.posy.values[all_experiment[0].index].shape
# =============================================================================

    return x_pos, y_pos
  
  
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
            # print(data.index[index_stim])
            idx_tmp = np.append(idx_tmp, np.arange(index_stim, (index_stim + frame)))
            
        idx_tmp = idx_tmp.reshape(experiment.nb_fly, frame)
        index = np.hstack([index, idx_tmp]) if index.size else idx_tmp
        index = index.astype(int)
    
    experiment.index_order(index)
    
    x_pos, y_pos = any_coordinates(experiment, data.center.posx, data.center.posy)
    experiment.position_order(x_pos, y_pos)
    
    x_pos_n, y_pos_n = any_coordinates(experiment, data.center.posx_n, data.center.posy_n)        
    experiment.position_n_order(x_pos_n, y_pos_n)  
    
    # print(experiment.__dict__.keys())
        
    return experiment


    
#%%           
def plot_one_trajectory(genDict, experiment):
    
    exp = experiment   
    genDict_key = []
    for key, value in genDict[exp.folder].items() :
        genDict_key.append(key)
    

    #plot x and y coordinates
    for i in range(0,exp.nb_fly) :
        plt.figure(str(exp.simulation) + " " + str(exp.folder[7:13]) + " " + str(genDict_key[2+i]) + " : xy position")
        plt.title(str(exp.simulation) + " " + str(exp.folder[7:13]) + " " + str(genDict_key[2+i]) + " : xy position")

        for j in range(1, len(exp.frame_per_period)):  
 
            #the "stimulation off" period
            # take the previous point too to have a continued line
            if (j % 2 == 0) :
                plt.plot(exp.x_pos[i, (exp.frame_per_period_tot[j-1]):exp.frame_per_period_tot[j]], 
                         exp.y_pos[i, (exp.frame_per_period_tot[j-1]):exp.frame_per_period_tot[j]], c='b')
            #the "stimulation on" period
            else :
                plt.plot(exp.x_pos[i, (exp.frame_per_period_tot[j-1]):exp.frame_per_period_tot[j]], 
                         exp.y_pos[i, (exp.frame_per_period_tot[j-1]):exp.frame_per_period_tot[j]], c='r')
                
            plt.plot(exp.x_pos[i,0], exp.y_pos[i,0], c='k', marker='x')
            plt.plot(exp.x_pos[i,-1], exp.y_pos[i,-1], c='k', marker='o')

        # plt.legend(stim_legend)
        plt.xlabel("x position [mm]")
        plt.ylabel("y position [mm]")
        # plt.x_lim()

    # =============================================================================
    # plt.legend(genDict_key[2:])
    # =============================================================================
    

    
    # =============================================================================
    # #plot the velocity versus time
    # plt.figure("x coordinate versus time")
    # plt.title("x coordinate versus time")
    # 
    # for i in range(0,nb_fly) :
    # 
    #     plt.subplot(nb_fly, 1, i+1)
    #     plt.plot(time_vector, x_pos[i,:])
    #     plt.legend(str(genDict_key[i+2]))
    #     
    # # plt.legend(genDict_key[2:])
    # plt.xlabel("time [s]")
    # plt.ylabel("x position [mm]")
    # =============================================================================
  
    return 
#%%
def plot_trajectories(genDict, data, all_experiment, fly_number = 0, normalized = 0):
       
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
        plt.figure(str(exp.simulation) + " " + str(exp.folder[7:13]) + " " + str(genDict_key[2+k]) + " : xy position")
        plt.title(str(exp.simulation) + " " + str(exp.folder[7:13]) + " " + str(genDict_key[2+k]) + " : xy position")
        
        if normalized == 0 :
            x_pos = exp.x_pos
            y_pos = exp.y_pos
        else :
            x_pos = exp.x_pos_n
            y_pos = exp.y_pos_n
        
        for j in range(1, len(exp.frame_per_period_tot)):  

            #the "off" period
            # take the previous point too to have a continued line
            if (j % 2 == 0) :
                plt.plot(x_pos[k, (exp.frame_per_period_tot[j-1]):exp.frame_per_period_tot[j]], 
                         y_pos[k, (exp.frame_per_period_tot[j-1]):exp.frame_per_period_tot[j]], c='b')
            #the "on" period
            else :
                plt.plot(x_pos[k, (exp.frame_per_period_tot[j-1]):exp.frame_per_period_tot[j]], 
                         y_pos[k, (exp.frame_per_period_tot[j-1]):exp.frame_per_period_tot[j]], c='r')
            
            plt.plot(x_pos[k,0], y_pos[k,0], c='k', marker='x')
            
            if normalized == 0 :
                plt.plot(x_pos[k,-1], y_pos[k,-1], c='k', marker='o')

        # plt.legend(stim_legend)
        plt.xlabel("x position [mm]")
        plt.ylabel("y position [mm]")
        # plt.x_lim()
   
    return
    

#%%
def plot_x_distance_over_time(genDict, data, all_experiment, all_folder, fly_number = 0):
       
    """ to extract x and y position of any data.keys(), 
        example : Leye.x, Rantenna.y"""
# =============================================================================
#     x_pos, y_pos = any_coordinates(experiment, data.center.posx_n, data.center.posy_n)
#     experiment.position_order(x_pos, y_pos)
# =============================================================================
    
    k = fly_number
    color = environment.color_plot
    
    legend = []
    for l in range(0,2):
        for i in range(0, len(all_folder)):
            legend.append(all_folder[i][7:13])
        
        genDict_key = []
        for key, value in genDict[all_experiment[0].folder].items() :
            genDict_key.append(key)
        
        
        #plot x and y coordinates
        for i in range(0,len(all_experiment)) :
            exp = all_experiment[i]
            time = np.arange(0,exp.total_frame)/exp.frame_frequency
            stim_time = np.array(exp.frame_per_period_tot[1:])/exp.frame_frequency
            
            # x coordinate
            if l == 0 :
                plt.figure(str(exp.simulation) + " " + str(genDict_key[2+k]) + " : x position over time")
                plt.title(str(exp.simulation) + " " + str(genDict_key[2+k]) + " : x position over time")
                # plt.title(str(genDict_key[2+k]) + " exp : " + str(exp.folder[:13]))            
                plt.plot(time, exp.x_pos_n[k, :], c=color[i])
             
            # y coordinate
            else :
                plt.figure(str(exp.simulation) + " " + str(genDict_key[2+k]) + " : y position over time")
                plt.title(str(exp.simulation) + " " + str(genDict_key[2+k]) + " : y position over time")
                # plt.title(str(genDict_key[2+k]) + " exp : " + str(exp.folder[:13]))            
                plt.plot(time, exp.y_pos_n[k, :], c=color[i])
                
    # =============================================================================
    #         
    #         for j in range(0, len(exp.frame_per_period)):  
    #             plt.plot(time[(exp.frame_per_period_tot[j-1]):exp.frame_per_period_tot[j]], 
    #                      exp.x_pos_n[k, (exp.frame_per_period_tot[j-1]):exp.frame_per_period_tot[j]], c=color[i])
    # =============================================================================
    
        # x coordinate
        if l == 0 :
            plt.figure(str(exp.simulation) + " " + str(genDict_key[2+k]) + " : x position over time")
            plt.legend(legend)
            plt.xlabel("time [s]")
            plt.ylabel("x position [mm]")
        
        # y coordinate
        else :
            plt.figure(str(exp.simulation) + " " + str(genDict_key[2+k]) + " : y position over time")
            plt.legend(legend)
            plt.xlabel("time [s]")
            plt.ylabel("y position [mm]")
        
        plt.axvline(0, ymin = -30, ymax = 30, c='r', ls='--')
        for j in range(0, len(exp.frame_per_period)):
            plt.axvline(stim_time[j], ymin = -30, ymax = 30, c='r', ls='--')
        
        plt.xlim(time[0], time[-1])
   
    return

#%%
def plot_xy_distance_over_time(genDict, data, all_experiment, all_folder, fly_number = 0):
       
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
        plt.figure(str(all_experiment[0].simulation) + " " + str(genDict_key[2+k]) + " : xy position over time " +  str(stim_key[j-1]))
        plt.title(str(all_experiment[0].simulation) + " " + str(genDict_key[2+k]) + " : xy position over time " +  str(stim_key[j-1]))
        
        # for i in range(0,1) :        
        for i in range(0,len(all_experiment)) :
            exp = all_experiment[i]                        
            plt.plot(exp.x_pos_n[k, (exp.frame_per_period_tot[j-1]):exp.frame_per_period_tot[j]], 
                     exp.y_pos_n[k, (exp.frame_per_period_tot[j-1]):exp.frame_per_period_tot[j]], c=color[i])
        # print(stim_time)
    
        plt.legend(legend)
        plt.xlabel("x position [mm]")
        plt.ylabel("y position [mm]")
# =============================================================================
#         plt.xlim(-30, 30)
#         plt.ylim(-30, 30)
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
    legend = []
    genDict_key = []
    
    for key, value in genDict[all_experiment[0].folder].items() :
        genDict_key.append(key)
    
    for i in range(0,len(all_experiment)) :
        exp = all_experiment[i]
        
        ############# FAUX LA VITESSE NE SE CALCLE PAS COMME CA
        absolute_pos = (exp.x_pos_n**2 + exp.y_pos_n**2)**0.5
        time = np.arange(0,exp.total_frame)/exp.frame_frequency
        stim_time = np.array(exp.frame_per_period_tot[1:])/exp.frame_frequency
        
        
        plt.figure(str(exp.simulation) + " " + str(exp.folder[7:13]) + " " + str(genDict_key[2+k]) + " : speed")
        plt.title(str(exp.simulation) + " " + str(exp.folder[7:13]) + " " + str(genDict_key[2+k]) + " : speed")
        
        
        
        for j in range (1, len(exp.frame_per_period_tot)): 
            
            delta_pos = np.array([absolute_pos[:, n] - absolute_pos[:, n-1] \
                                  for n in range(exp.frame_per_period_tot[j-1], \
                                                 exp.frame_per_period_tot[j] - 1)])
                
            speed = (delta_pos.T)/exp.frame_frequency
             
            #the "off" period
            if (j % 2 == 0) :
                plt.scatter(time[(exp.frame_per_period_tot[j-1] + 1) : (exp.frame_per_period_tot[j] - 1)], 
                         speed[k, 1:], c='r', s = 1)
            #the "on" period
            else :
                plt.scatter(time[(exp.frame_per_period_tot[j-1] + 1) : (exp.frame_per_period_tot[j] - 1)], 
                         speed[k, 1:], c='b', s = 1)

        # plt.legend(stim_legend)
        plt.xlabel("time [s]")
        plt.ylabel("velociy [m/s]")
        # plt.x_lim()

   
    return

#%%
########### THIS IS HOW TO CHOOSE THE DESIRED DIRECTORY
#1st level folder choose a number between 0 and 9
""" ['MDN', 'PR', 'SS01049, 'SS01054', 'SS01540', 'SS02111', 'SS02279', 'SS02377', 'SS02608', 'SS02617]
    note :  first layer name is saved as self.simulation in Fly_Experiment class"""


#2nd layer folder choose a number between 0 and 3
"""example for the MDN folder :
   ['200206_110534_s1a10_p3-4', '200206_160327_s4a9_p3-4', '200206_105311_s1a9_p3-4', '200206_153954_s4a10_p3-4']"""
 
global environment
first_layer = 0
second_layer = 0
all_experiment = []


environment = Environment()
genDict, data, all_folder, simulation = general_data(first_layer)


for i in range(0,4):
    second_layer = i
    experiment = experiment_center_pos(genDict, data, environment, all_folder, simulation, second_layer)
    all_experiment.append(experiment)

""" normalized = 0 --> plot x_pos
    normalized = 1 --> plot x_pos_n"""
    
plot_trajectories(genDict, data, all_experiment, normalized = 1)
plot_x_distance_over_time(genDict, data, all_experiment, all_folder, fly_number = 0)
plot_xy_distance_over_time(genDict, data, all_experiment, all_folder, fly_number = 0)
plot_speed_over_time(genDict, data, all_experiment, all_folder, fly_number = 0)



# plot_one_trajectory(genDict, all_experiment[0])
print("aaa")
# =============================================================================
# second_layer = 0
# plot_trajectories(first_layer, second_layer)
# =============================================================================
# =============================================================================
# 
# for i in range(0,4):
#     second_layer = i    
#     plot_trajectories(first_layer, second_layer)
# =============================================================================





