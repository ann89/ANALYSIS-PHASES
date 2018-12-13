#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 13:20:18 2018

@author: anna
"""

import numpy as np
import scipy
import scipy.optimize
import pandas as pd
import matplotlib.pyplot as plt
import MDAnalysis
import os

MDAnalysis.core.flags['use_periodic_selections'] = True

input_dir = "classification/"

output_dir = "Diffusion/"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
    
top = 'membrane.pdb'
traj = 'traj-dt-1ns-mol.xtc' 
#read the vectors
u = MDAnalysis.Universe(top,traj)


# =============================================================================
# ##Check and adapt
# def fit_anomalous_diffusion_data(time_data_array,MSD_data_array,degrees_of_freedom=2):
#     
#     def function_to_fit(time,fractional_diffusion_coefficient,scaling_exponent):
#         coefficient_dictionary = {1:2,2:4,3:6} #dictionary for mapping degrees_of_freedom to coefficient in fitting equation
#         coefficient = coefficient_dictionary[degrees_of_freedom]
#         return coefficient * fractional_diffusion_coefficient * (time ** scaling_exponent) #equation 1 in the Kneller 2011 paper with appropriate coefficient based on degrees of freedom
# 
#     #fit the above function to the data and pull out the resulting parameters
#     optimized_parameter_value_array, estimated_covariance_params_array = scipy.optimize.curve_fit(function_to_fit,time_data_array,MSD_data_array)
#     #generate sample fitting data over the range of time window values 
#     sample_fitting_data_X_values_nanoseconds = np.linspace(time_data_array[0],time_data_array[-1],100)
#     sample_fitting_data_Y_values_Angstroms = function_to_fit(sample_fitting_data_X_values_nanoseconds, *optimized_parameter_value_array)
#     #could then plot the non-linear fit curve in matplotlib with, for example: axis.plot(sample_fitting_data_X_values_nanoseconds,sample_fitting_data_Y_values_Angstroms,color='black')
#     #could plot the original data points alongside the fit (MSD vs time) with, for example: axis.scatter(time_data_array,MSD_data_array,color='black')
# 
#     #extract pertinent values from the scipy curve_fit arrays (D_alpha, alpha, and their standard deviations)
#     parameter_standard_deviation_array = np.sqrt(np.diagonal(estimated_covariance_params_array))
#     fractional_diffusion_coefficient = optimized_parameter_value_array[0]
#     standard_deviation_fractional_diffusion_coefficient = parameter_standard_deviation_array[0]
#     alpha = optimized_parameter_value_array[1]
#     standard_deviation_alpha = parameter_standard_deviation_array[1]
# 
#     return (fractional_diffusion_coefficient, standard_deviation_fractional_diffusion_coefficient, alpha, standard_deviation_alpha,sample_fitting_data_X_values_nanoseconds,sample_fitting_data_Y_values_Angstroms)
# 
# 
# 
# 
# 
def fit_linear_diffusion_data(MSD_data_array,degrees_of_freedom=2):
#     
# 
     coefficient_dictionary = {1:2.,2:4.,3:6.} #dictionary for mapping degrees_of_freedom to coefficient in fitting equation
     coefficient = coefficient_dictionary[degrees_of_freedom]
# 
     x_data_array = np.arange(len(MSD_data_array))
     y_data_array = MSD_data_array
     z = np.polyfit(x_data_array,y_data_array,1) #fit a polynomial of any degree (x,y, degree)
     slope, intercept = z
     diffusion_constant = slope / coefficient
     p = np.poly1d(z)
     #this is actually for the fit:
     sample_fitting_data_X_values_nanoseconds = np.linspace(x_data_array[0],x_data_array[-1],100)
     sample_fitting_data_Y_values_Angstroms = p(sample_fitting_data_X_values_nanoseconds)
     return (diffusion_constant, sample_fitting_data_X_values_nanoseconds, sample_fitting_data_Y_values_Angstroms)
#  
#     # TODO estimate error in D constant (???)
#     #estimate error in D constant using g_msd approach:
#     first_half_x_data, second_half_x_data = np.array_split(x_data_array,2)
#     first_half_y_data, second_half_y_data = np.array_split(y_data_array,2)
#     slope_first_half, intercept_first_half = np.polyfit(first_half_x_data,first_half_y_data,1)
#     slope_second_half, intercept_second_half = np.polyfit(second_half_x_data,second_half_y_data,1)
#     diffusion_constant_error_estimate = abs(slope_first_half - slope_second_half) / coefficient
#     
      #use poly1d object for polynomial calling convenience:

#     # TODO if works
#     #print(p)
#     sample_fitting_data_X_values_nanoseconds = np.linspace(time_data_array[0],time_data_array[-1],100)
#     sample_fitting_data_Y_values_Angstroms = p(sample_fitting_data_X_values_nanoseconds)
# 

#*****
#read the files and store them in a dictionary:
file_dict = {} # Create an empty dict
for i in range(1000, 2001, 999):
    test= np.load(input_dir + 'resid_phases' + str(i) + '.npy')
    test_unique = np.unique(test[:,0], return_index=True) #index
    file_dict[i] = test[test_unique[1]] #fill the dictionary with vectors  

#take only the common resids for all the t
from functools import reduce
common_elements=reduce(np.intersect1d, ([file_dict[ele][:,0] for ele in file_dict]))

#TODO here you can take note of the CHOL flipping and integrate it with Stefan's code

#take only the resids that do not change phase (i.e., always ordered or always disordered)
def get_unique_ele(dic_ele, unique_ele):
    mask = np.isin(dic_ele[:, 0], unique_ele)
    return dic_ele[mask]

common_mols = [get_unique_ele(file_dict[ele], common_elements) for ele in file_dict]
order_time_evo=np.vstack([ele[:, 1] for ele in common_mols] )

list_always_disordered_ele = []
list_always_ordered_ele = []
for i in range(np.shape(order_time_evo)[1]):
    if all(order_time_evo[:, i] == 1):
        list_always_disordered_ele.append(i)
    if all(order_time_evo[:, i] == 0):
        list_always_ordered_ele.append(i)


common_mols_disordered = common_mols[0][list_always_disordered_ele] #list of resid always in the fluid phase
common_mols_ordered = common_mols[0][list_always_ordered_ele] #list of resid always in the ordered phase

#store disordered coordinates
all_xs_dis = []
all_ys_dis = []

all_xs_DAPC_dis = []
all_ys_DAPC_dis =[]

all_xs_CHL_dis = []
all_ys_CHL_dis =[]

all_xs_DSPC_dis = []
all_ys_DSPC_dis =[]
#ordered coordinates
all_xs_ord = []
all_ys_ord = []

all_xs_DAPC_ord = []
all_ys_DAPC_ord =[]

all_xs_CHL_ord = []
all_ys_CHL_ord = []

all_xs_DSPC_ord = []
all_ys_DSPC_ord =[]
#center of mass disordered phase
disordered_CM_x = []
disordered_CM_y = []
#center of mass ordered phase
ordered_CM_x = []
ordered_CM_y = []


for ts in u.trajectory[1000:1010:1]:

#center of mass of the disordered phase
    disordered_CM_str = ' '.join(str(int(x)) for x in common_mols_disordered[:,0])
    disordered_CM_x.append(u.select_atoms('resnum' + ' ' + disordered_CM_str).center_of_mass()[0])
    disordered_CM_y.append(u.select_atoms('resnum' + ' ' + disordered_CM_str).center_of_mass()[1])
#center of mass of the ordered phase
    ordered_CM_str = ' '.join(str(int(x)) for x in common_mols_ordered[:,0])
    ordered_CM_x.append(u.select_atoms('resnum' + ' ' + ordered_CM_str).center_of_mass()[0])
    ordered_CM_y.append(u.select_atoms('resnum' + ' ' + ordered_CM_str).center_of_mass()[1])
#CM of all the single residues in the disordered phase 
    coords_disordered = []
    coords_dis_DAPC = []
    coords_dis_CHL  = []
    coords_dis_DSPC = []
# resname_disordered = []
    for i in np.arange(len(common_mols_disordered)):
        
        residues = u.select_atoms('resnum %i'%(common_mols_disordered[i,0]))
        dis_DAPC = u.select_atoms('resname DAPC and resnum %i'%(common_mols_disordered[i,0]))
        dis_CHL = u.select_atoms('resname CHL and resnum %i'%(common_mols_disordered[i,0]))
        dis_DSPC = u.select_atoms('resname DSPC and resnum %i'%(common_mols_disordered[i,0]))
        coords_disordered.append(list(residues.center_of_mass()))
        if dis_DAPC:
            coords_dis_DAPC.append(list(dis_DAPC.center_of_mass()))

        if dis_CHL:
            coords_dis_CHL.append(list(dis_CHL.center_of_mass()))    

        if dis_DSPC:
            coords_dis_DSPC.append(list(dis_DSPC.center_of_mass())) 
            
            
    coords_disordered = np.array(coords_disordered)
    coords_dis_DAPC = np.array(coords_dis_DAPC)
    coords_dis_CHL = np.array(coords_dis_CHL)
    coords_dis_DSPC = np.array(coords_dis_DSPC)    
    
    x_coords = coords_disordered[:, 0]
    y_coords = coords_disordered[:, 1]
    
    x_DAPC  = coords_dis_DAPC[:, 0]
    y_DAPC  = coords_dis_DAPC[:, 1]
    
    x_CHL  = coords_dis_CHL[:, 0]
    y_CHL  = coords_dis_CHL[:, 1]
    
    x_DSPC  = coords_dis_DSPC[:, 0]
    y_DSPC  = coords_dis_DSPC[:, 1]
   
    all_xs_dis.append(x_coords)
    all_ys_dis.append(y_coords)
    
    all_xs_DAPC_dis.append(x_DAPC)
    all_ys_DAPC_dis.append(y_DAPC)

    all_xs_CHL_dis.append(x_CHL)
    all_ys_CHL_dis.append(y_CHL)

    all_xs_DSPC_dis.append(x_DSPC)
    all_ys_DSPC_dis.append(y_DSPC)    


#CM of the single resid in the ordered phase 
    coords_ordered = []
    coords_ord_DAPC = []
    coords_ord_CHL  = []
    coords_ord_DSPC = []
    for i in np.arange(len(common_mols_ordered)):
        residues = u.select_atoms('resnum %i'%(common_mols_ordered[i,0]))
        ord_DAPC = u.select_atoms('resname DAPC and resnum %i'%(common_mols_ordered[i,0]))
        ord_CHL = u.select_atoms('resname CHL and resnum %i'%(common_mols_ordered[i,0]))
        ord_DSPC = u.select_atoms('resname DSPC and resnum %i'%(common_mols_ordered[i,0]))
        
        coords_ordered.append(list(residues.center_of_mass()))
        if ord_DAPC:
            coords_ord_DAPC.append(list(ord_DAPC.center_of_mass()))
        if ord_CHL:
            coords_ord_CHL.append(list(ord_CHL.center_of_mass()))    
        if ord_DSPC:
            coords_ord_DSPC.append(list(ord_DSPC.center_of_mass()))                 
#        resname_ordered.append(list(residues.residues.resnames))#for the moment I do not use this info   
    coords_ordered = np.array(coords_ordered)
    coords_ord_DAPC = np.array(coords_ord_DAPC)
    coords_ord_CHL = np.array(coords_ord_CHL)
    coords_ord_DSPC = np.array(coords_ord_DSPC) 
    
    x_coords = coords_ordered[:, 0]
    y_coords = coords_ordered[:, 1]
    
    x_DAPC  = coords_ord_DAPC[:, 0]
    y_DAPC  = coords_ord_DAPC[:, 1]
    
    x_CHL  = coords_ord_CHL[:, 0]
    y_CHL  = coords_ord_CHL[:, 1]
    
    x_DSPC  = coords_ord_DSPC[:, 0]
    y_DSPC  = coords_ord_DSPC[:, 1]
    
    all_xs_DAPC_ord.append(x_DAPC)
    all_ys_DAPC_ord.append(y_DAPC)

    all_xs_CHL_ord.append(x_CHL)
    all_ys_CHL_ord.append(y_CHL)

    all_xs_DSPC_ord.append(x_DSPC)
    all_ys_DSPC_ord.append(y_DSPC)    
    
    
    all_xs_ord.append(x_coords)
    all_ys_ord.append(y_coords)
    

#return the vectors  
#CM dis
disordered_CM_x = np.array(disordered_CM_x)
disordered_CM_y = np.array(disordered_CM_y) 
#CM ord
ordered_CM_x =  np.array(ordered_CM_x)
ordered_CM_y =  np.array(ordered_CM_y)

#lipids dis
all_xs_dis = np.array(all_xs_dis)
all_ys_dis = np.array(all_ys_dis)
all_xs_DAPC_dis = np.array(all_xs_DAPC_dis)
all_ys_DAPC_dis = np.array(all_ys_DAPC_dis)
all_xs_CHL_dis = np.array(all_xs_CHL_dis)
all_ys_CHL_dis = np.array(all_ys_CHL_dis)
all_xs_DSPC_dis = np.array(all_xs_DSPC_dis)
all_ys_DSPC_dis = np.array(all_ys_DSPC_dis) 

#lipids ord   
all_xs_ord = np.array(all_xs_ord)
all_ys_ord = np.array(all_ys_ord)
all_xs_DAPC_ord = np.array(all_xs_DAPC_ord)
all_ys_DAPC_ord = np.array(all_ys_DAPC_ord)
all_xs_CHL_ord = np.array(all_xs_CHL_ord)
all_ys_CHL_ord = np.array(all_ys_CHL_ord)
all_xs_DSPC_ord = np.array(all_xs_DSPC_ord)
all_ys_DSPC_ord = np.array(all_ys_DSPC_ord)

#sliceof the array every delay  
def calcMSD_Cstyle(rx, ry, rx_cm, ry_cm):
    N_FR = rx.shape[0]
    N_PA = rx.shape[1]

    sdx = np.zeros(N_FR)
    sdy = np.zeros(N_FR)
    cnt = np.zeros(N_FR)
    for t in range(N_FR):
        for dt in range(1, (N_FR - t)):
            cnt[dt] = cnt[dt] + 1
            for n in range(N_PA):
                sdx[dt] = sdx[dt] + ((rx[t+dt, n] - rx_cm[t+dt]) - (rx[t, n] - rx_cm[t]))**2
                sdy[dt] = sdy[dt] + ((ry[t+dt, n] - ry_cm[t+dt]) - (ry[t, n] - ry_cm[t]))**2
    for t in range(N_FR):
        sdx[t] = sdx[t] / ((N_PA * cnt[t]) if int(cnt[t]) else 1)
        sdy[t] = sdy[t] / ((N_PA * cnt[t]) if int(cnt[t]) else 1)
    return sdx, sdy 

#retrieve the MSD for the different lipids 
MSD_x_dis, MSD_y_dis = calcMSD_Cstyle(all_xs_dis, all_ys_dis, disordered_CM_x, disordered_CM_y)
MSD_DAPC_x_dis, MSD_DAPC_y_dis = calcMSD_Cstyle(all_xs_DAPC_dis, all_ys_DAPC_dis, disordered_CM_x, disordered_CM_y)
MSD_CHL_x_dis, MSD_CHL_y_dis = calcMSD_Cstyle(all_xs_CHL_dis, all_ys_CHL_dis, disordered_CM_x, disordered_CM_y)
MSD_DSPC_x_dis, MSD_DSPC_y_dis = calcMSD_Cstyle(all_xs_DSPC_dis, all_ys_DSPC_dis, disordered_CM_x, disordered_CM_y)

MSD_x_ord, MSD_y_ord = calcMSD_Cstyle(all_xs_ord, all_ys_ord, ordered_CM_x, ordered_CM_y)
MSD_DAPC_x_ord, MSD_DAPC_y_ord = calcMSD_Cstyle(all_xs_DAPC_ord, all_ys_DAPC_ord, ordered_CM_x, ordered_CM_y)
MSD_CHL_x_ord, MSD_CHL_y_ord = calcMSD_Cstyle(all_xs_CHL_ord, all_ys_CHL_ord, ordered_CM_x, ordered_CM_y)
MSD_DSPC_x_ord, MSD_DSPC_y_ord = calcMSD_Cstyle(all_xs_DSPC_ord, all_ys_DSPC_ord, ordered_CM_x, ordered_CM_y)


MSD_dis = MSD_x_dis + MSD_y_dis
MSD_DAPC_dis = MSD_DAPC_x_dis + MSD_DAPC_y_dis
MSD_DSPC_dis = MSD_DSPC_x_dis + MSD_DSPC_y_dis
MSD_CHL_dis = MSD_CHL_x_dis + MSD_CHL_y_dis

MSD_ord =  MSD_x_ord + MSD_y_ord
MSD_CHL_ord = MSD_CHL_x_ord + MSD_CHL_y_ord
MSD_DSPC_ord = MSD_DSPC_x_ord + MSD_DSPC_y_ord


#TODO check if it is OK
D_test_tuple  = fit_linear_diffusion_data(MSD_DSPC_ord)

plt.plot(MSD_dis)
plt.plot(MSD_DAPC_dis)
plt.plot(MSD_CHL_dis)
plt.plot(MSD_DSPC_dis)


plt.plot(MSD_ord)
plt.plot(MSD_CHL_ord)
plt.plot(MSD_DSPC_ord)


#np.save(output_dir + 'test'+ str(ts.frame) + '.npy', coord_o)