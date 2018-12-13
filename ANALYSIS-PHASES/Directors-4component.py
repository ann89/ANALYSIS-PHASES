# coding: utf-8


import MDAnalysis
import MDAnalysis.lib.NeighborSearch as NS
import numpy as np
#import sys
#import multiprocessing as mp
#from time import time
from numpy.linalg import norm
import os



top  = 'traj-noH.pdb'
traj = 'traj-noH.xtc'
side = 'up'  #sys.argv[1] # "up" for upper leaflet "down" for lower leaflet

u = MDAnalysis.Universe(top,traj)

# norm vector along the z-axis
vref = np.array([0.0,0.0,1.0])

def get_side_coordinates_and_box(u, time_ts):
    """Assign lipids to leaflets, retrieve their coordinates, resIDs and the director order parameter u=3/2cosˆ2(theta)-1/2"""
 
    x, y, z = u.trajectory.ts.triclinic_dimensions[0][0], u.trajectory.ts.triclinic_dimensions[1][1], u.trajectory.ts.triclinic_dimensions[2][2]
    box = np.array([x, y, z])
    
        ### Determining side of the bilayer CHOL belongs to in this frame
        #Lipid Residue names
    lipid1 ='DAPC'
    lipid2 ='DSPC'
    lipid3 ='SSM'
    lipid4 ='CHL'
        
    lpd1_atoms = u.select_atoms('resname %s and name P'%lipid1)
    lpd2_atoms = u.select_atoms('resname %s and name P'%lipid2)
    lpd3_atoms = u.select_atoms('resname %s and name P'%lipid3)  
    lpd4_atoms = u.select_atoms('resname %s and name O2'%lipid4)
    
    num_lpd1 = lpd1_atoms.n_atoms
    num_lpd2 = lpd2_atoms.n_atoms
    num_lpd3 = lpd3_atoms.n_atoms
    
    
    
        # atoms in the upper leaflet as defined by insane.py or the CHARMM-GUI membrane builders
        # select cholesterol headgroups within 1.5 nm of lipid headgroups in the selected leaflet
        # this must be done because CHOL rapidly flip-flops between leaflets in the MARTINI model
        # so we must assign CHOL to each leaflet at every time step, and in large systems
        # with substantial membrane undulations, a simple cut-off in the z-axis just will not cut it
    if side == 'up':
        lpd1i = lpd1_atoms[:int((num_lpd1)/2)]
        lpd2i = lpd2_atoms[:int((num_lpd2)/2)]
        lpd3i = lpd3_atoms[:int((num_lpd3)/2)]
#        print(lpd3i.resnums)
        lipids = lpd1i + lpd2i + lpd3i

        ns_lipids = NS.AtomNeighborSearch(lpd4_atoms)
        lpd4i = ns_lipids.search(lipids,15.0)
        print(lpd4i)
    elif side == 'down':
        lpd1i = lpd1_atoms[int((num_lpd1)/2):]
        lpd2i = lpd2_atoms[int((num_lpd2)/2):]
        lpd3i = lpd3_atoms[int((num_lpd3)/2):]

        lipids = lpd1i + lpd2i + lpd3i
        ns_lipids = NS.AtomNeighborSearch(lpd4_atoms)
        lpd4i = ns_lipids.search(lipids,15.0)
        print(lpd4i)
        #define the cholesterol 
        # ID center of geometry coordinates for cholesterol on indicated bilayer side
    
 
    lpd4_coords = np.zeros((len(lpd4i.resnums),3))
    lpd4_res = np.zeros((len(lpd4i.resnums),1))
    lpd4_u =np.zeros((len(lpd4i.resnums),1)) 
    for i in np.arange(len(lpd4i.resnums)):

        resnum = lpd4i.resnums[i]
        head_CHL = u.select_atoms('resnum %i and (name C1)'%resnum).center_of_geometry() 
        tail_CHL = u.select_atoms('resnum %i and (name C65)'%resnum).center_of_geometry()
        vect_CHL = head_CHL - tail_CHL
        theta_CHL = np.arccos(np.dot(vect_CHL, vref)/(norm(vect_CHL)*norm(vref)))

        u_CHL= 1.5*((pow(np.cos(theta_CHL), 2)))-0.5
        group = u.select_atoms('resnum %i'%resnum)
        group_cog = group.center_of_geometry()
        lpd4_coords[i] = group_cog
        lpd4_res[i] = resnum
        lpd4_u[i] = u_CHL
                
            
# ID coordinates for lipids on indicated bilayer side, renaming variables
# lipid 1
    lpd1_coordsA = np.zeros((len(lpd1i.resnums),3))
    lpd1_coordsB = np.zeros((len(lpd1i.resnums),3))
    lpd1_res = np.zeros((len(lpd1i.resnums),1))
    lpd1_u = np.zeros((len(lpd1i.resnums),1))
    for i in np.arange(len(lpd1i.resnums)):
        resnum_A = lpd1i.resnums[i]
        resnum_B = lpd1i.resnums[i]

        head_lp1_A = u.select_atoms('resnum %i and (name C22)'%resnum_A).center_of_geometry() 
        tail_lp1_A = u.select_atoms('resnum %i and (name 6C21)'%resnum_A).center_of_geometry()
        vect_lp1_A = head_lp1_A - tail_lp1_A
        theta_lp1_A = np.arccos(np.dot(vect_lp1_A, vref)/(norm(vect_lp1_A)*norm(vref)))
        
        head_lp1_B = u.select_atoms('resnum %i and (name C32)'%resnum_B).center_of_geometry() 
        tail_lp1_B = u.select_atoms('resnum %i and (name 6C31)'%resnum_B).center_of_geometry()
        vect_lp1_B = head_lp1_B - tail_lp1_B
        theta_lp1_B = np.arccos(np.dot(vect_lp1_B, vref)/(norm(vect_lp1_B)*norm(vref))) 
             
        theta_lp1 = (theta_lp1_A + theta_lp1_B)/2
        u_lp1= ((3/2)*(pow(np.cos(theta_lp1), 2))) - 0.5

        lpd1_u[i] = u_lp1
        group_lpd1_chainA = u.select_atoms('resnum %i and (name C26 or name C27 or name C28 or name C29 )'%resnum_A)
        group_lpd1_chainB = u.select_atoms('resnum %i and (name C36 or name C37 or name C38 or name C39)'%resnum_B)
        group_cog_lpd1A = group_lpd1_chainA.center_of_geometry()
        group_cog_lpd1B = group_lpd1_chainB.center_of_geometry()
        lpd1_coordsA[i] = group_cog_lpd1A
        lpd1_coordsB[i] = group_cog_lpd1B
        lpd1_res[i] = resnum_A
        
#lipid 2        
    lpd2_coordsA = np.zeros((len(lpd2i.resnums),3))
    lpd2_coordsB = np.zeros((len(lpd2i.resnums),3))
    lpd2_res = np.zeros((len(lpd2i.resnums),1))
    lpd2_u = np.zeros((len(lpd2i.resnums),1))
    for i in np.arange(len(lpd2i.resnums)):
        resnumB_A = lpd2i.resnums[i]
        resnumB_B = lpd2i.resnums[i] 

        head_DAPC_A = u.select_atoms('resnum %i and (name C24)'%resnumB_A).center_of_geometry()
        tail_DAPC_A = u.select_atoms('resnum %i and (name 6C21)'%resnumB_A).center_of_geometry()
       # head_DLIP_A = u.select_atoms('resnum %i and (name C24)'%resnumB_A).center_of_geometry()
       # tail_DLIP_A = u.select_atoms('resnum %i and (name 6C21)'%resnumB_A).center_of_geometry()

        vect_DAPC_A = head_DAPC_A - tail_DAPC_A
        theta_DAPC_A = np.arccos(np.dot(vect_DAPC_A, vref)/(norm(vect_DAPC_A)*norm(vref)))
      #  vect_DLIP_A = head_DLIP_A - tail_DLIP_A
      #  theta_DLIP_A = np.arccos(np.dot(vect_DLIP_A, vref)/(norm(vect_DLIP_A)*norm(vref)))
        
       
        head_DAPC_B = u.select_atoms('resnum %i and (name C34)'%resnumB_B).center_of_geometry() 
        tail_DAPC_B = u.select_atoms('resnum %i and (name 6C31)'%resnumB_B).center_of_geometry()
      #  head_DLIP_B = u.select_atoms('resnum %i and (name C34)'%resnumB_B).center_of_geometry() 
      #  tail_DLIP_B = u.select_atoms('resnum %i and (name 6C31)'%resnumB_B).center_of_geometry()
        
        vect_DAPC_B = head_DAPC_B - tail_DAPC_B
        theta_DAPC_B = np.arccos(np.dot(vect_DAPC_B, vref)/(norm(vect_DAPC_B)*norm(vref))) 
        theta_DAPC = (theta_DAPC_A + theta_DAPC_B)/2          
      #  vect_DLIP_B = head_DLIP_B - tail_DLIP_B
      #  theta_DLIP_B = np.arccos(np.dot(vect_DLIP_B, vref)/(norm(vect_DLIP_B)*norm(vref))) 
      #  theta_DLIP = (theta_DLIP_A + theta_DLIP_B)/2    


        u_DAPC= ((3/2)*(pow(np.cos(theta_DAPC), 2))) - 0.5
        lpd2_u[i] = u_DAPC
        #u_DLIP= ((3/2)*(pow(np.cos(theta_DLIP), 2))) - 0.5
        #lpd2_u[i] = u_DLIP
        group_lpd2_chainA = u.select_atoms('resnum %i and (name C26 or name C27 or name C28 or name C29)'%resnumB_A)
        group_lpd2_chainB = u.select_atoms('resnum %i and (name C36 or name C37 or name C38 or name C39)'%resnumB_B)
        group_cog_lpd2A = group_lpd2_chainA.center_of_geometry()
        lpd2_coordsA[i] = group_cog_lpd2A
        group_cog_lpd2B = group_lpd2_chainB.center_of_geometry()
        lpd2_coordsB[i] = group_cog_lpd2B
        lpd2_res[i] = resnumB_A

 
 # SSM coordinates: lipid 3   
    lpd3_coordsA = np.zeros((len(lpd3i.resnums),3))
    lpd3_coordsB = np.zeros((len(lpd3i.resnums),3))
    lpd3_res = np.zeros((len(lpd3i.resnums),1))
    lpd3_u = np.zeros((len(lpd3i.resnums),1))
    for i in np.arange(len(lpd3i.resnums)):
        resnumC_A = lpd3i.resnums[i]
        resnumC_B = lpd3i.resnums[i]

        head_SSM_A = u.select_atoms('resnum %i and (name C6S)'%resnumC_A).center_of_geometry() 
        tail_SSM_A = u.select_atoms('resnum %i and (name C16S)'%resnumC_A).center_of_geometry()
        vect_SSM_A = head_SSM_A - tail_SSM_A
        theta_SSM_A = np.arccos(np.dot(vect_SSM_A, vref)/(norm(vect_SSM_A)*norm(vref)))
        
        head_SSM_B = u.select_atoms('resnum %i and (name C6F)'%resnumC_B).center_of_geometry() 
        tail_SSM_B = u.select_atoms('resnum %i and (name C16F)'%resnumC_B).center_of_geometry()
        vect_SSM_B = head_SSM_B - tail_SSM_B
        theta_SSM_B = np.arccos(np.dot(vect_SSM_B, vref)/(norm(vect_SSM_B)*norm(vref))) 
             
        theta_SSM = (theta_SSM_A + theta_SSM_B)/2
        u_SSM= ((3/2)*(pow(np.cos(theta_SSM), 2))) - 0.5

        lpd3_u[i] = u_SSM

        group_lpd3_chainA = u.select_atoms('resnum %i and (name C8S or name C9S or name C10S or name C11S )'%resnumC_A)
        group_lpd3_chainB = u.select_atoms('resnum %i and (name C8F or name C9F or name C10F or name C11F)'%resnumC_B)
        group_cog_lpd3A = group_lpd3_chainA.center_of_geometry()
        group_cog_lpd3B = group_lpd3_chainB.center_of_geometry()
        lpd3_coordsA[i] = group_cog_lpd3A
        lpd3_coordsB[i] = group_cog_lpd3B
        lpd3_res[i] = resnumC_A

   
    lpd_coords = np.vstack((lpd1_coordsA,lpd1_coordsB,lpd2_coordsA,lpd2_coordsB, lpd3_coordsA,lpd3_coordsB,lpd4_coords)) #append
    lpd_resids = np.vstack((lpd1_res,lpd1_res, lpd2_res, lpd2_res, lpd3_res, lpd3_res, lpd4_res)) #append
    lpd_us =np.vstack((lpd1_u,lpd1_u, lpd2_u,lpd2_u, lpd3_u, lpd3_u, lpd4_u)) #append
    #print(time_ts)
    file = open("testfile.txt","a") 
    file.write(str(time_ts.frame)+ "\n")
    file.close()
    
    return lpd_coords,lpd_resids,lpd_us, lpd4_res,box


directory = "results/"

#create the directory in the file path if it does not exist
if not os.path.exists(directory):
    os.makedirs(directory)

for ts in u.trajectory[0:20:20]:
	coordinates, residues, directors, cholesterol, box = get_side_coordinates_and_box(u, ts)
	if side == 'up':
		np.save(directory + 'directors_upper_tail_' + str(ts.frame) + '.npy', directors)
		np.save(directory + 'coordinates_upper_tail_' + str(ts.frame) + '.npy', coordinates)
		np.save(directory + 'residues_upper_tail_' + str(ts.frame) + '.npy'  , residues)
		np.save(directory + 'cholesterol_upper_tail_' + str(ts.frame) + '.npy', cholesterol)
        
	elif side == 'down':
		np.save(directory + 'directors_lower_tail_' + str(ts.frame) + '.npy', directors)
		np.save(directory + 'coordinates_lower_tail_' + str(ts.frame) + '.npy', coordinates)
		np.save(directory + 'residues_lower_tail_' + str(ts.frame) + '.npy', residues)
		np.save(directory + 'cholesterol_lower_tail_' + str(ts.frame) + '.npy', cholesterol)

