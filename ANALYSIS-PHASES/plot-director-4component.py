
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pylab
from skimage import measure
import os



side = 'up' #or 'down' for lower leaflet
input_dir = "results/"
output_dir = "plots/"


#make output folder if not present
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


#read the vectors

for j in range(0,2001,20):
        
    if side == 'up':
        
        directors_upper = np.load(input_dir + 'directors_upper_tail_'+ str(j) + '.npy')
            
        coords_value_upper = np.load(input_dir + 'coordinates_upper_tail_' + str(j) + '.npy')
            
        resid_value_upper = np.load(input_dir + 'residues_upper_tail_' + str(j) + '.npy')

        resid_U = resid_value_upper
        n = directors_upper
        pos=coords_value_upper
 

        
        plt.figure(figsize=(15,5))
        plt.title('SSM/DSPC/DAPC/Chol  Upper leaflet',size=20)
        

        plt.scatter(pos[:, 0],pos[:, 1],vmin=-0.5,vmax=1.0,c=(n[:,0]),s=50,cmap=plt.cm.GnBu)
        cb=plt.colorbar()
        
        sg=np.where(np.logical_and(resid_U>=153, resid_U<=278))[0]  #DSPC
        tg=np.where(np.logical_and(resid_U>=1, resid_U<=152))[0]  #DAPC
        zg=np.where(np.logical_and(resid_U>=279, resid_U<=404))[0]  #SSM
        cg=np.where((resid_U>=405))[0]  #CHL

        plt.scatter(pos[sg,0],pos[sg,1],facecolors='none', edgecolors='#CC0000', s=130, lw=1.5)
        plt.scatter(pos[tg,0],pos[tg,1],facecolors='none', edgecolors='black', s=130, lw=1.5)
        plt.scatter(pos[cg,0],pos[cg,1],facecolors='none', edgecolors='#F5C816', s=130, lw=1.5)
        plt.scatter(pos[zg,0],pos[zg,1],facecolors='none', edgecolors='#FF00FF', s=130, lw=1.5)

                    
        plt.xlim(np.nanmin(pos[:,0])-1,np.nanmax(pos[:,0])+1)
        plt.ylim(np.nanmin(pos[:,1])-1,np.nanmax(pos[:,1])+1)
        cb.set_label(label='n(' r'$\theta$' ')',size=20)
        cb.ax.tick_params(labelsize=16)
        plt.tick_params(axis='x', pad=8)
        plt.tick_params(axis='both', which='major', labelsize=16)
        plt.savefig(output_dir + 'Directors-upper_frame' + str(j) + '.png', dpi=300)
        
    if side == 'down':
    #read the vectors
        directors_lower = np.load(input_dir + 'directors_lower_tail_' + str(j) + '.npy')
            
        coords_value_lower =np.load(input_dir + 'coordinates_lower_tail_' + str(j) + '.npy')
            
        resid_value_lower = np.load(input_dir + 'residues_lower_tail_' + str(j) + '.npy')
            

        resid_U=resid_value_lower
        n = directors_lower
        
        sg=np.where(np.logical_and(resid_U>=658, resid_U<=783))[0]  #DSPC
        tg=np.where(np.logical_and(resid_U>=506, resid_U<=657))[0]  #DAPC
        zg=np.where(np.logical_and(resid_U>=784, resid_U<=909))[0]  #SSM
        cg=np.where(np.logical_or(resid_U>=910,  resid_U<=505))[0]  #CHL
        pos=coords_value_lower
            #print(pos, psi6_value_upper)
        plt.figure(figsize=(15,5))
        plt.title('SSM/DSPC/DAPC/Chol Lower leaflet',size=20)
        plt.scatter(pos[:, 0],pos[:, 1],vmin=-0.5,vmax=1.0,
             c=(n[:, 0]),s=50,cmap=plt.cm.GnBu)
        cb=plt.colorbar()
            
        plt.scatter(pos[sg,0],pos[sg,1],facecolors='none', edgecolors='#CC0000', s=130, lw=1.5)
        plt.scatter(pos[tg,0],pos[tg,1],facecolors='none', edgecolors='black', s=130, lw=1.5)
        plt.scatter(pos[cg,0],pos[cg,1],facecolors='none', edgecolors='#F5C816', s=130, lw=1.5)
        plt.scatter(pos[zg,0],pos[zg,1],facecolors='none', edgecolors='#FF00FF', s=130, lw=1.5)
        plt.xlim(np.nanmin(pos[:,0])-1,np.nanmax(pos[:,0])+1)
        plt.ylim(np.nanmin(pos[:,1])-1,np.nanmax(pos[:,1])+1)
        cb.set_label(label= 'n(' r'$\theta$' ')',size=20)
        cb.ax.tick_params(labelsize=16)
        plt.tick_params(axis='x', pad=8)
        plt.tick_params(axis='both', which='major', labelsize=16)
        plt.savefig(output_dir + 'Directors-lower_frame' + str(j) + '.png', dpi=300)
            
        
