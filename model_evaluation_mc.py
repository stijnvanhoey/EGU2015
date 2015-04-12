# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 01:42:34 2015

@author: stvhoey
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class ObjectiveFunctions():
    '''
    Class for deriving different objective functions.
    '''

    def __init__(self, measured, modelled):
        """
        Compare measured and modelled values by a set of objective functions

        Parameters
        -----------
        measured : np.ndarray
            array withh measured values
        modelled : np.ndarray
            array withh measured values
        """

        self.modelled = modelled
        self.measured = measured
        self.scen = self.modelled.shape[0]
        
        if self.modelled.shape[1] != self.measured.shape[1]:
            raise Exception('Modelled and observed timeseries need \
                                 to be of the same length')

        self.residuals = self.measured - self.modelled

    def SSE(self):
        """
        Calculate Sum of Squared Errors (SSE) between measured and modelled values

        Notes
        -----
        We use here the np.ndarray structure, not pd.DataFrames
        """
        OF = np.sum(self.residuals**2, axis=1)
        return OF

    def RMSE(self):
        '''
        Root Mean Square Error

        Parameters
        -----------
        measured : np.array
            numpy array, length N, with measured values
        modelled : np.array
            numpy array, length N, with modelled values

        Notes
        -----
        We use here the np.ndarray structure, not pd.DataFram
        '''
        OF = np.sqrt((self.residuals**2).mean(axis=1))
        return OF

    def RRMSE(self):
        '''
        Relative Root Mean Square Error

        Notes
        -------
        The relative Root Mean Square Error is the Root Mean Square Error
        devided by the mean of the observations.

        See Also
        ---------
        RMSE
        '''
        OF = np.sqrt((self.residuals**2).mean(axis=1))/(self.measured).mean(axis=1)
        return OF
        
    
    def NSE(self):
        """Nash Sutcliffe model efficiency coefficient

        Parameters
        ----------
        x : numpy.array
            1D numpy array to calculate the metric
        y : numpy.array
            1D numpy array to calculate the metric

        Returns
        -------
        Nash Sutcliffe coefficient : float
             Nash Sutcliffe model efficiency coefficient
             """
        one=np.full((1,self.scen), 1).T
        OF = one - (np.sum(self.residuals ** 2, axis=1) / (np.sum((self.measured - np.mean(self.measured))** 2, axis=1)))
        return OF