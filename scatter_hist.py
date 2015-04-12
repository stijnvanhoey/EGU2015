import numpy as np
#import scipy as sp
import pylab as p
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pandas as pd
import hydropy as hp

from matplotlib.colors import Normalize
from matplotlib.transforms import offset_copy
from matplotlib.ticker import MaxNLocator
from pylab import *

from model_evaluation_mc import ObjectiveFunctions
from matplotlib.ticker import MaxNLocator, LinearLocator, NullLocator
from mpl_toolkits.axes_grid import make_axes_locatable

p.rc('mathtext', default='regular')
               

def translate_cmap(yourchoice):
    """
    Convert readable colormap to a matplotlib cmap entity
    """
    cmap_transposer = {'red_yellow_green': 'RdYlGn_r', 'yellow_orange_brown' : 'YlOrBr',
                       'blue_green' : 'winter'}
    return cmap_transposer[yourchoice]


def _define_thresholdrange(measured, modelled, objective_function='SSE'):
                            
        
    temp = ObjectiveFunctions(measured, modelled)  
    objective= np.vstack((temp.SSE(), temp.RMSE(), temp.RRMSE(), temp.NSE()))
    
        
    if objective_function == "SSE":
        selected_of = objective[0]
        print("Objective_function = SSE")
    elif objective_function == "RMSE":
        selected_of = objective[1]
        print("Objective_function = RMSE")
    elif objective_function == "RRMSE":
        selected_of = objective[2]
        print("Objective_function = RRMSE")
    else:
        selected_of = objective[3]
        print("Objective_function = NSE")
     
   
    return  selected_of
     
    
def create_scatterhist(all_parameters, parameter1, parameter2, measured, modelled, 
                        pars_names,
                        objective_function='SSE', 
                        colormaps = "red_yellow_green",
                        threshold=0.005, 
                        *args,  **kwargs):
                            
    '''
    Function to create the interactive scatterplot. Behavioural is always taken
    as lower as the given threshold.
    Parameters
    ----------
    all_parameters : Nxp np.array
        A number of parameter (p) value combinations used as model input. Length N
        corresponds to the number of simulations
        
    parameter1, parameter 2: {parametername : value}
        Parametername of the parameters that the user wants to use for 
        model evaluation
        
    modelled: Nxk np.array
        simulation outputs for all parameterscombinations (N). Length k is the length 
        of the timeserie
        
    measured: 1xk np.array
        observations. Length k is the length 
        of the timeserie
        
    parsnames: {parametername : value}
        dict with all parameters (p) used the model
        
    objective_function : Nx1 np.array
        The user is allowed to choose between different objective functions
        (SSE, RMSE, RRMSE)
        
                
    colormaps : 
        Colormap of the scatterplot 
        
     treshold : 
        Value between 0 and 1. Parametersets for which the scaled value of the
        objective function < threshold are retained (these parametersets are behavioral).
            
    *args,  **kwargs: args
        arguments given to the scatter plot
        example: s=15, marker='o', edgecolors= 'k', facecolor = 'white'
    ...
    '''
     
    # Extract values from pd dataframes    
    all_parameters=all_parameters.values
    measured=measured.values
    modelled=modelled.values
    selected_of = _define_thresholdrange(measured, modelled, objective_function)
    
    
    #Translate color into a colormap
    colormaps = translate_cmap(colormaps)
    
    #Selected parameters & objective function 
    parameters = np.vstack((all_parameters[:,parameter1], all_parameters[:,parameter2])).T
    scaled_of = np.array((selected_of-selected_of.min())/(selected_of.max()-selected_of.min()))
    
    #calculation of the real threshold value from the relative one
    real_threshold = threshold*(selected_of.max() - selected_of.min())+selected_of.min()
    print("Current threshold = " + str(real_threshold))   
       
    #check if selected parameter are not the same
    if parameter1 == parameter2:
        raise Exception('Select two different parameters')
          
    #check if parameter and of are really arrays
    if not isinstance(parameters, np.ndarray):
        raise Exception('parameters need to be numpy ndarray')
    if not isinstance(scaled_of, np.ndarray):
        raise Exception('objective function need to be numpy ndarray')

    # check if objective function is of size 1xN     
    scaled_of = np.atleast_2d(scaled_of).T
    if (len(scaled_of.shape) != 2 ):
       raise Exception("Objective function need to be of size (1, N) got %s instead" %(scaled_of.shape))
   
    # check that SSE row length is equal to parameters
    if not parameters.shape[0] == scaled_of.shape[0]:
        raise Exception("None corresponding size of parameters and OF!")

    # Check if threshold is in range of SSE values
    if threshold < 0 or threshold > 1:
        raise Exception("Threshold outside objective function ranges")

    # Select behavioural parameter sets with of lower as threshold
    search=np.where(scaled_of < threshold)
    behav_par = parameters[search[0]]
    behav_obj = selected_of[search[0]].T 
    print("Number of behavioural parametersets = " + str(behav_obj.shape[0]) + " out of " + str(parameters.shape[0]))
        
    if not behav_par.size > 0:
        raise Exception('Threshold to severe, no behavioural sets.')
 
    fig, ax_scatter = plt.subplots(figsize=(8,6))
    divider = make_axes_locatable(ax_scatter)
    ax_scatter.set_autoscale_on(True)
        
    # create a new axes with  above the axScatter
    ax_histx = divider.new_vertical(1.5, pad=0.0001, sharex=ax_scatter)

    # create a new axes on the right side of the axScatter
    ax_histy = divider.new_horizontal(1.5, pad=0.0001, sharey=ax_scatter)

    fig.add_axes(ax_histx)
    fig.add_axes(ax_histy)  

    # now determine nice limits by hand:
    xmin = np.min(all_parameters[:,parameter1])
    xmax = np.max(all_parameters[:,parameter1])
    ymin = np.min(all_parameters[:,parameter2])
    ymax = np.max(all_parameters[:,parameter2])
    
    ax_histx.set_xlim( (xmin, xmax) )
    ax_histy.set_ylim( (ymin, ymax) )

    #determine binwidth (pylab examples:scatter_hist.py )
    binwidth = 0.05
    xymax = np.max( [np.max(behav_par[:,0]), np.max(behav_par[:,1])] )
    lim = (int(xymax/binwidth) + 1) * binwidth

    bins = np.arange(-lim, lim + binwidth, binwidth)    
    
    # create scatter & histogram
    sc1 = ax_scatter.scatter(behav_par[:,0], behav_par[:,1], c=behav_obj,
                             edgecolors= 'none', cmap=colormaps, *args,  **kwargs)
    ax_histx.hist(behav_par[:,0], color='0.6', edgecolor='None',  bins = bins)
    ax_histy.hist(behav_par[:,1], orientation='horizontal',  edgecolor='None',
                  color='0.6', bins=bins)
                  
    #determine number of bins scatter              
    majloc1 = MaxNLocator(nbins=5, prune='lower')
    ax_scatter.yaxis.set_major_locator(majloc1)
    majloc2 = MaxNLocator(nbins=5)
    ax_scatter.xaxis.set_major_locator(majloc2)

    ax_scatter.grid(linestyle = 'dashed', color = '0.75',linewidth = 1.)
    ax_scatter.set_axisbelow(True)
    
    
    #ax_histx.set_axisbelow(True)


    plt.setp(ax_histx.get_xticklabels() + ax_histy.get_yticklabels(),
            visible=False)
    plt.setp(ax_histx.get_yticklabels() + ax_histy.get_xticklabels(),
            visible=False)
    ax_histy.set_xticks([])
    ax_histx.set_yticks([])
    ax_histx.xaxis.set_ticks_position('bottom')
    ax_histy.yaxis.set_ticks_position('left')

    ax_histy.spines['right'].set_color('none')
    ax_histy.spines['top'].set_color('none')
    ax_histy.spines['bottom'].set_color('none')
    ax_histy.spines['left'].set_color('none')

    ax_histx.spines['top'].set_color('none')
    ax_histx.spines['right'].set_color('none')
    ax_histx.spines['left'].set_color('none')
    ax_histx.spines['bottom'].set_color('none')
    ax_scatter.spines['top'].set_color('none')
    ax_scatter.spines['right'].set_color('none')


   # x and y label of parameter names
    ax_scatter.set_xlabel(pars_names[parameter1], horizontalalignment ='center', verticalalignment ='center')    
    ax_scatter.set_ylabel(pars_names[parameter2], horizontalalignment ='center', verticalalignment ='center')    
  
    # Colorbar
    cbar = fig.colorbar(sc1, ax=ax_scatter, cmap=colormaps,
                        orientation='vertical')
    cbar.ax.set_ylabel('Value objective function')
    
    behav_outputs=modelled[search[0]]
    
    return fig, ax_scatter, ax_histx, ax_histy, sc1, cbar, behav_outputs

