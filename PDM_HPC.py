# -*- coding: utf-8 -*-
"""
Created on Tue Aug 07 10:27:41 2012

@author: Stijn Van Hoey
stvhoey.vanhoey@ugent.be

Free to use the code, but don't forget who did this for you...
"""

import numpy as np

def linres(n_res,q_init,co,k):  #nog VHM gewijs met cov te doen van vorige tijdstap
    if n_res==1:
        q_init[0]=q_init[0]*np.exp(-1/k) + co*(1 - np.exp(-1/k))
        return q_init
    else:
        q_init[n_res-1]=q_init[n_res-1]* np.exp(-1/k)+linres(n_res-1,q_init,co,k)[n_res-2]*(1 - np.exp(-1/k))
        return q_init

def linresv(n_res,q_init,co,v,k):  #nog VHM gewijs met cov te doen van vorige tijdstap
    if n_res==1:
        q_init[0]=q_init[0]*np.exp(-1/k) + co*(1 - np.exp(-1/k))*v
        return q_init
    else:
        q_init[n_res-1]=q_init[n_res-1]* np.exp(-1/k)+linres(n_res-1,q_init,co,k)[n_res-2]*(1 - np.exp(-1/k))*v
        return q_init


def PDM(Pars,Const,InitCond,Rain,ET):
    '''
    currently for hourly timesteps    
    not perfectly same as output of PDM model, but conceptually the same
    added feature is the possibility of using a series of lin reservoirs for each substream    
    '''

    #Define the parameters
    ##################################
    #Storage model parameters
    Cmax = np.float64(Pars[0])    
    b = np.float64(Pars[1])    
    be = np.float64(Pars[2])   
    bg = 1.   
    kg = np.float64(Pars[3])    
    kb = np.float64(Pars[4])    
    kf =  np.float64(Pars[5])    
    Stau =  np.float64(Pars[6])  

    #derived par
    Smax = Cmax/(b+1)       
    
    #Define the constants
    ##################################
    area = np.float64(Const[0])     #catchment area
#    timestep = np.float64(Const[1])

    totn = Rain.size     #Timesteps (eigenlijk niet nodig als gepast inputs!)
    dt=1.

    #Define the initial conditions
    ##################################
    S = np.float64(InitCond[0])     #Soil moiosture storage
    S3 = np.float64(InitCond[1])     #Soil moiosture storage

    of = np.float64(0.0)*np.ones(2)   
    
    #qr = qo + qi + qg
    v = np.float64(area * 1000.0 / (60.0 * 60.0))    #mm/h to m3/s

    #Define arrays for outputs
    ##################################  
    q_out=np.zeros(totn,np.float64)
    
    #Calculate C* for first timestep    
    Cstert = max(min(Cmax *(1.-(1.-S/Smax)**(1./(b+1))),Cmax),0.0)
   
    #Start dynamic run
    ##################################
    for t in range(totn):
        rain = np.float64(Rain[t])
        pet = np.float64(ET[t])
        
        ETR = pet *(1.-((Smax-S)/Smax)**be)
        pit= rain-ETR
        
        #PARETO
        Cstert = Cmax *(1.-(1.-S/Smax)**(1./(b+1)))
#        Cstertdt = Cstert + max(pit,0.0)*dt

        Cstertdt = max(min(Cstert + max(pit,0.0)*dt,Cmax),0.0)
        FCs = 1. - (1.-Cstert/Cmax)**b
        qsx = max(FCs * pit,0.0)
        Vqd = max(pit*dt - Smax*((1.-Cstert/Cmax)**(b+1) -(1.-Cstertdt/Cmax)**(b+1)),0.0)

        #drainage
        Dr = max(((S - Stau)**bg)/kg,0.0)
        
        pit = rain-ETR -Dr
        
        S = min(max(S + (pit)*dt - Vqd, 0.01),Smax)
#        if S > Smax:
#            S=Smax
#            qsurplus=(Smax-S)*dt
#            print 'surpluske',qsurplus        
#            qsx=qsx+qsurplus
        
        #groundwater reservoir
        S3 = S3 - ((np.exp(-3*dt*kb*(S3**2))-1.)*(Dr-kb*S3**3))/(3*kb*(S3**2))
        qb = v*kb*S3**3
        
        #overland reservoirs
        of = linresv(2,of,qsx,v,kf)
        qall = of[-1] + qb
  
        #Outputs
        q_out[t] = qall

        #get new Cstert value        
        Cstert = Cstertdt
        
    return q_out  

#==============================================================================
#TESTRUN
#==============================================================================
import os
import time

# DATA NETE Rain_Cal_warm_1jan02_31dec05
datapath="D:\Modellen\Version2012\HPC"
Rain=np.loadtxt(os.path.join(datapath,'Rain_Cal_warm_1jan02_31dec05'))
ET=np.loadtxt(os.path.join(datapath,'ET_Cal_warm_1jan02_31dec05'))
Flow_eval=np.loadtxt(os.path.join(datapath,'Flow_Cal_Meas_13aug02_31dec05'))
ID=Rain.size-Flow_eval.size
rain=Rain
evapo=ET

#Define the parameters PDM - Thomas
Cmax = 500.     
b = 0.25
be = 3.1
bg = 1.
kg = 9000.
kb = 0.004
kf =  24.
Stau =  0.0

#Define the constants
area = 362.0
timestep = 1.0

#Pars=[2.250000000000000000e+002, 2.387500000000000200e-001, 2.987499999999999800e+000, 9.375000000000000000e+003, 4.125000000000000200e-003, 2.212500000000000000e+001, 1.250000000000000000e+000]
Pars=[Cmax,b,be,kg,kb,kf,Stau]
Const=[area,timestep]
InitCond=[10.,1.0]

#----------------------------------------------------------------------
# CHECK CALCULATION TIME
#----------------------------------------------------------------------       
c0 = time.clock() # total CPU time spent in the script so far
#----------------------------------------------------------------------  

trp1=PDM(Pars,Const,InitCond,rain,evapo)

#----------------------------------------------------------------------       
##CHECK TIME TO CALCULATE
cpu_time = time.clock() - c0
print 'The elapsed cpu-time for the model run was %f seconds' %cpu_time
#----------------------------------------------------------------------

import matplotlib.pyplot as plt
plt.plot(trp1[ID:])
plt.plot(Flow_eval,'--')

from obj_functies_nodate import *
Evalu=ObjFunct(Flow_eval,trp1[ID:])
OF_all = Evalu.ALL(7)
print OF_all[20]

#==============================================================================
