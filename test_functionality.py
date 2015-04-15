# -*- coding: utf-8 -*-
"""
Created on Thu Jan  8 16:46:39 2015

@author: stvhoey
"""
import numpy as np
import pandas as pd

from scatter_hist_season import create_scatterhist, create_seasonOF
pars =pd.read_csv('data/example2_PDM_parameters.txt',header=0, sep=',', index_col=0)
measured = pd.read_csv('data/example_PDM_measured.txt', header=0, sep='\t', decimal='.', index_col=0)
modelled = pd.read_csv('data/example2_PDM_outputs.txt',header=0, sep=',', index_col=0).T
names=pars.columns

time=np.array(measured.index)
modelled.index = time

 
objective=create_seasonOF(modelled,measured)


a = create_scatterhist(pars, 'b' , 'be', objective,
                        xbinwidth = 0.05, ybinwidth = 0.05,
                        objective_function='SSE', 
                        colormaps = "red_yellow_green",
                        threshold=0.2,
                        season = 'Winter')