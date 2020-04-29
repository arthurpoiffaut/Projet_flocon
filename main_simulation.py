# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 15:33:20 2020

@author: alviss
"""

## import general
import numpy as np
import os
import scipy.constants as spc
import matplotlib.pyplot as plt 

## library developer pour le projet 

import Fonction_Globalvar_projet_flocon as FG   # hoooo flack generatoorrrr!
import Image_generator as IG

## pour test de performence 
import cProfile as cp
import time
import timeit as ti
# Constantes :


# valeur de la dimention (m)
delta_dim=10*10**(-6); 

# valeur de la dimention large (m)
delta_dim_large=30*10**(-6)

# valeur de la variation de temps 
delta_t=10*10**(-15); 


#temperature en kelvin
T=258 

#masse d'une molecule d'eau
m=2.988*10**(-26)


#valeur du coe de diff   (m^2/s)
D=2*10**-5 

#ratio des c_sat et c_solide
ratio_c=1*10**(-6) #pas sur ici faudrait revoir c_sat/c_solide

#nu_kin=133; # micrometre/s en fonction aussi
#Fonction qui calcule nu_ki
nu_kin=ratio_c*np.sqrt((spc.k*T)/(2*spc.pi*m))

#concentration aux extremitées
sigma_limit=0.1


#longueur d'une cellule 
Lcell=np.round((np.sqrt(3)/2)*(delta_dim),9)


#valeur du delta tau parametre sans dimention pour les step de temp
delta_tau=(D*delta_t)/(delta_dim)**2


#dimention dim_box[0]=dimx ect
dim_box_fine=np.array([102,102,18]).astype(np.int32) 

dim_box_large=np.array([74,74,74]).astype(np.int32) 

center=(np.floor((dim_box_fine)/2)).astype(np.int32)


#Definition de matrices 

# mat cristal 0 si pas dans le cristal
ice_fine_ini=np.zeros(dim_box_fine)
# mat vapeur varie en fonction de la densité d'eau en état vapeur entre 0 et 1 ?
vap_fine=sigma_limit*(np.ones(dim_box_fine))


vap_large=sigma_limit*(np.ones(dim_box_large))


ice_fine_ini[center[0],center[1],center[2]]=1

#initalisation d un vecteur de parcour avec tout les position voulue
Pos_fine=[]
Pos_limite_fine=[] 
for i1 in range(0,dim_box_fine[0]):
    for i2 in range(0,dim_box_fine[1]):
        for i3 in range(0,dim_box_fine[2]):
            Pos_fine.append(np.array([i1,i2,i3]).astype(np.int32))     
            

Pos_large=[]
Pos_limite_large=[]  
for i1 in range(0,dim_box_large[0]):
    for i2 in range(0,dim_box_large[1]):
        for i3 in range(0,dim_box_large[2]):
            if (i1<19 or i1>53) and (i2<19 or i2>53) and (i3 < 33 or i3 >39):
                Pos_large.append(np.array([i1,i2,i3]).astype(np.int32))
                
            if (i1>=19 and i1<53) and (i2>=19 and i2<53) and (i3 >= 33 and i3<39):
                Pos_limite_large.append(np.array([i1,i2,i3]).astype(np.int32))
       


      
        
        
VOISIN_fine=[]            
for pos in Pos_fine:
    VOISIN_fine.append(FG.voisin(pos))
            

VOISIN_large=[]            
for pos in Pos_large:
    VOISIN_large.append(FG.voisin(pos))
                   



#initialisation de l eta initiale 
Vposini1=FG.voisin(center)


for i1 in range(0,6):
    ice_fine_ini[Vposini1[i1,0],Vposini1[i1,1],Vposini1[i1,2]]=1
    vap_fine[Vposini1[i1,0],Vposini1[i1,1],Vposini1[i1,2]]=2    
    


#Initalisation de l etat de la frontiere 
fronpos_ini=FG.frontiere(ice_fine_ini)

fron_state=np.zeros([np.shape(fronpos_ini)[0],1])

fron_state=np.concatenate((fronpos_ini,fron_state),axis=1).astype(float)


if True:
    #creation des dossier 
    dossier_vap='dossier_vapeur_newcode_1'
    dossier_ice='dossier_glace_newcode_1'
    os.mkdir( dossier_vap)
    os.mkdir(dossier_ice)
    
    
    
    # framev=[]
    # frameice=[]
    framet=[]
    # #framef=[]
    vap_fine_ini=vap_fine
    
    vap_large_ini=vap_large
    
    # frameice.append(ice_ini.copy())
    # framev.append(vap.copy())  #juste pour pas avoir tout commenter à chaque fois que je change de quoi
    
    #sauvegarde des etat initiaux et du compteur d'iteration des frame
    FG.saveframev(0,dossier_vap,vap_fine_ini)
    FG.saveframeice(0,dossier_ice,ice_fine_ini)
    continuer1=True
    i3=0    
    #Boucle Principale
    
    while continuer1:
        print(i3)
        print(np.sum(ice_fine_ini))
        print()
        ice_fine_c=ice_fine_ini
        
        continuer2=True
        i4=0
        t1=time.time()
        #Boucle de la relaxation de la vapeur d'eaux 
        # elle attend que le champ de vapeur d'eau autour du cristal soit stable 
        # avent de la laisser passer à l'étape de croissance
        while  continuer2:
            t3=time.time()
            vap_large_c=FG.relax1_large(vap_large_ini,vap_fine_ini,dim_box_large,sigma_limit,Pos_limite_large,Pos_large,VOISIN_large)
            #vap_c=relax1(vap_ini,ice_c,fron_state,dim_box,mat_voi,sigma_limit,Pos)
            #(vap_large,vap_fine,dim_box_large,sigma_limit,Pos_limite_large,Pos_large,VOISIN_large)
            t4=time.time()
            i4=i4+1
            print()
            print(t4-t3)
            print()
            #relax1(vap,ice,fron_state,dim_box,mat_voi,Pos):
            
            #Condition de sortie de la boucle si le l'erreur relatice est plus
            #basse que 0.01% peut etre changer article 0.005%
            if ((abs((vap_large_c-vap_large_ini)/vap_large_ini))<0.01).all() : #and i4>=10:
                #framev.append(vap_c.copy())
                vap_large_ini=vap_large_c
                continuer2=False
                print('nbr iteration pour convergence large:')
                print(i4)
                print()
                t2=time.time()
                print('temp convergence large:')
                print(t2-t1)
                print()
            else:
                vap_large_ini=vap_large_c
                
        
        continuer3=True
        while  continuer3:
            t3=time.time()
            vap_fine_c=FG.relax1_fine(vap_fine_ini,vap_large_ini,ice_fine_c,fron_state,dim_box_fine,Pos_fine,VOISIN_fine)
            #vap_c=relax1(vap_ini,ice_c,fron_state,dim_box,mat_voi,sigma_limit,Pos)
            t4=time.time()
            i4=i4+1
            print()
            print(t4-t3)
            print()
            #relax1(vap,ice,fron_state,dim_box,mat_voi,Pos):
            
            #Condition de sortie de la boucle si le l'erreur relatice est plus
            #basse que 0.01% peut etre changer article 0.005%
            if ((abs((vap_fine_c-vap_fine_ini)/vap_fine_ini))<0.01).all() : #and i4>=10:
                #framev.append(vap_c.copy())
                vap_fine_ini=vap_fine_c
                continuer3=False
                print('nbr iteration pour convergence fine:')
                print(i4)
                print()
                t2=time.time()
                print('temp convergence fine:')
                print(t2-t1)
                print()
            else:
                vap_fine_ini=vap_fine_c
        
        t1=time.time()
        
        #étape de Croissence des frontiere
        
        t_min=FG.tmin(fron_state,ice_fine_c,vap_fine_c,nu_kin,dim_box_fine,Lcell)  
       # print(t_min)tmin(fron_state,ice,vap,nu_kin,mat_voi,dim_box,Lcell):
        fron_state_c=FG.croissance(ice_fine_c,fron_state,vap_fine_c,delta_dim,dim_box_fine,nu_kin,t_min)
        #(ice,fron_state,vap,delta_dim,mat_voi,dim_box,nu_kin,t_min)
        update=FG.update_fron(fron_state_c,ice_fine_c,dim_box_fine,Lcell)
        # update_fron(fron_state,ice,mat_voi,dim_box,Lcell)
        #(fron_state,ice,mat_voi_con,dim_box,Lcell)
        fron_state=update[0]
        ice_ini=update[1]
        
        #crl !!!!!!!!!!!!!!!!!!!!
        #framef.append(fron_state.copy())
        # frameice.append(ice_ini.copy())
        framet.append(t_min)
        i3=i3+1
        FG.saveframev(i3,vap_fine_ini)
        FG.saveframeice(i3,ice_fine_ini)
        t2=time.time()
        print('temp pour opperation croi et save:')
        print(t2-t1)
        print()
        
        # Conditon de fin de la simulation si un poin de glace est rendu sur la
        # Frontiere de la boit a été modifie pour le moment car on limit la croissence
        # vertical
        if (fron_state==np.array([dim_box_fine[0]-1,dim_box_fine[1]-1,dim_box_fine[2]-1,np.nan])).any() or (fron_state==np.array([1,1,1,np.nan])).any():
            print('fin')
            continuer1=False    


















