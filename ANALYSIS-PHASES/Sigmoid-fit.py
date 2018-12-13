#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 17:10:16 2018

@author: anna
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 14:46:24 2018

@author: anna


"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
import pylab
from scipy.optimize import curve_fit
from skimage import measure
import os

side = 'up'
input_dir = "results/"

output_dir = "classification/"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
    
contours_all = []

def sigmoid(x, x0, k, con1, con2):
     y = con1 / (1 + np.exp(-k*(x-x0))) + con2
     return y

def fit_sigmoid_to_snake(X_data, Y_data):

    idx_helper = np.arange(0, len(X_data))
    entry_point = int(idx_helper[Y_data == min(Y_data)])    
    exit_point_helper = max(Y_data[entry_point:])
    exit_point = int(idx_helper[Y_data == exit_point_helper])
    print(entry_point,exit_point)
    xdata = X_data[entry_point:exit_point]
    ydata = Y_data[entry_point:exit_point]
    
    popt, pcov = curve_fit(sigmoid, xdata, ydata, p0=[100,0.01, 0.0001, 0])
    print(popt)
    x = np.linspace(0, 300, 50)
    y = sigmoid(x, *popt)

    pylab.plot(xdata, ydata, 'o', label='data')
    pylab.plot(x,y, label='fit')
    pylab.legend(loc='best')
    pylab.show()


def func(x,a,e):
    e2 = e**2
    f = ((1/(np.pi*2*e2)) * np.exp(-(x-a)**2/(2*e2))) 
    f[np.abs(x-a) > 3*e] = 0
    return f 


e=9


for j in range(1000, 2000, 999):
    n_grid_x=300
    n_grid_y=100
    if side == 'up':
        
        directors_upper = np.load(input_dir + 'directors_upper_tail_'+ str(j) + '.npy')
            
        coords_value_upper = np.load(input_dir + 'coordinates_upper_tail_' + str(j) + '.npy')
            
        resid_value_upper = np.load(input_dir + 'residues_upper_tail_' + str(j) + '.npy')

       # Vxy_time = []
        
        #for i in range(len(directors_upper)):
        resid_U = resid_value_upper
        n = directors_upper
        pos=coords_value_upper
        
        
 #=============================================================================
 #for periodic boundary conditions
 
        coordsx = np.linspace(min(pos[:,0]), max(pos[:,0]), n_grid_x)
        coordsy = np.linspace(min(pos[:,1]), max(pos[:,1]), n_grid_y)
 
        coordsx_help_low =np.linspace(-max(pos[:,0]) + min(pos[:,0]), min(pos[:,0]), n_grid_x)
        coordsy_help_low =np.linspace(-max(pos[:,1]) + min(pos[:,1]), min(pos[:,1]), n_grid_y)
 
        coordsx_help_up =np.linspace(max(pos[:,0]), 2*max(pos[:,0])-min(pos[:,0]), n_grid_x)
        coordsy_help_up =np.linspace(max(pos[:,1]), 2*max(pos[:,1])-min(pos[:,1]), n_grid_y)        
         
        X, Y = np.meshgrid(coordsx, coordsy)
        X_help_low, Y_help_low = np.meshgrid(coordsx_help_low, coordsy_help_low)
        X_help_up, Y_help_up = np.meshgrid(coordsx_help_up, coordsy_help_up)
    
        Vxy = np.zeros((np.shape(X)))
        Vxy_help_1 = np.zeros((np.shape(X)))
        Vxy_help_2 = np.zeros((np.shape(X)))
        Vxy_help_3 = np.zeros((np.shape(X)))
        Vxy_help_4 = np.zeros((np.shape(X)))
        Vxy_help_5 = np.zeros((np.shape(X)))
        Vxy_help_6 = np.zeros((np.shape(X)))
        Vxy_help_7 = np.zeros((np.shape(X)))
        Vxy_help_8 = np.zeros((np.shape(X)))        
 
        for i in np.arange(len(pos)):
 # 
            Vxy += n[i]*func(X, pos[i,0], e)*func(Y, pos[i,1], e)
            Vxy_help_1 += n[i]*func(X_help_low, pos[i,0], e)*func(Y, pos[i,1], e)
            Vxy_help_2 += n[i]*func(X_help_up, pos[i,0], e)*func(Y, pos[i,1], e)
            Vxy_help_3 += n[i]*func(X, pos[i,0], e)*func(Y_help_low, pos[i,1], e)
            Vxy_help_4 += n[i]*func(X, pos[i,0], e)*func(Y_help_up, pos[i,1], e)
            Vxy_help_5 += n[i]*func(X_help_up, pos[i,0], e)*func(Y_help_up, pos[i,1], e)
            Vxy_help_6 += n[i]*func(X_help_low, pos[i,0], e)*func(Y_help_up, pos[i,1], e) 
            Vxy_help_7 += n[i]*func(X_help_low, pos[i,0], e)*func(Y_help_low, pos[i,1], e)     
            Vxy_help_8 += n[i]*func(X_help_up, pos[i,0], e)*func(Y_help_low, pos[i,1], e) 
        Vxy_total= (Vxy + Vxy_help_1 + Vxy_help_2 + Vxy_help_3 + Vxy_help_4 + Vxy_help_5 + Vxy_help_6 + Vxy_help_7 + Vxy_help_8)
        
#===================================================================================
        #to fit the sigmoid
 #       c = np.sum(Vxy_total, axis=0)/np.shape(Vxy_total)[0]
 #       helper_X = np.arange(len(c))
 #       fit_sigmoid_to_snake(helper_X, c)
 #       fig = plt.figure()
  #      ax=fig.gca()
   #     ax.plot(c)
   #     plt.show()
#===================================================================================
#once you have found the fitting values        
        middle = sigmoid(1.41683047e+02, 1.41683047e+02, 1.18885208e-01, 4.32175343e-05, 1.40605983e-05)
# here we assign lipids to the different phases
        
        Vxy_bool=np.zeros((np.shape(X)))
        Vxy_bool[Vxy_total < middle] = 1
        plt.figure()
        plt.imshow(Vxy_bool, interpolation='nearest', cmap=plt.cm.gray) 
        
        phase_belonging = np.zeros((len(pos)))
        range_x = (max(pos[:,0]) - min(pos[:,0])) / n_grid_x    
        range_y = (max(pos[:,1]) - min(pos[:,1])) / n_grid_y

        for i in np.arange(len(pos)):
            
            idx_x = int((pos[i,0] - min(pos[:,0])) / range_x - 1.e-5) #the  - 1.e-5 is because accuracy issue in the /
            idx_y = int((pos[i,1] - min(pos[:,1])) / range_y - 1.e-5) #this - 1.e-5 is because accuracy issue in the /
            #print(idx_x, idx_y)
            phase_belonging[i] = Vxy_bool[idx_y, idx_x]
           
        resid_phases = np.column_stack((resid_U[:,0], phase_belonging))
        np.save(output_dir + 'resid_phases' + str(j) + '.npy', resid_phases)
        
        plt.figure()
        plt.imshow(Vxy_bool, interpolation='nearest', cmap=plt.cm.gray) 
        plt.scatter((pos[:,0] - min(pos[:,0])) / range_x , pos[:,1] - min(pos[:,1]) / range_y, c= phase_belonging[:])


            
            
#        contours_all.append(contours)
#        np.save('contours_upper', contours_all)
#        for m, contour in enumerate(contours):
#            plt.plot(contour[:, 1]+ min(pos[:, 0]), contour[:, 0] + min (pos[:, 1]), linewidth=2)
             
        plt.imshow(Vxy_total, interpolation='nearest', cmap=plt.cm.gray) 
        
