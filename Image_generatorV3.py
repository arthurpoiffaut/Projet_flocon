# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 23:56:24 2020

@author: alviss et Alexis
"""



###############################################################################
import numpy as np
import matplotlib.pyplot as plt 
#import scipy as sp
#import scipy.constants as spc
#import time 
#from joblib import Parallel, delayed # truc de multi proceesing
#from scipy.ndimage import convolve
import os
from numba import njit
#import concurrent.futures
#fonction d<afficha
###############################################################################

#plt.rcParams["figure.figsize"] = (20,20)

plt.rcParams['xtick.labelsize']=15
plt.rcParams['ytick.labelsize']=15


def hexplot(matflocon,k):#prend la matrice 2D contenant l'image du flocon et ''l'hexagonifie'' en scatter
    dimx=np.shape(matflocon)[0]
    dimy=np.shape(matflocon)[1] 
    
    neige = np.where(matflocon >= 1)
    sizemax=max(neige[0])
    sizemin=min(neige[0])
    size=(sizemax-sizemin) #largeur du flocon (où il y a glace) en terme de cases
    trusize=size*10#e-6    # 1 case est 10 microns(deltax)

    
    hexagonal = np.empty((2,dimx,dimy)) #array contenant les points hex
    n = np.array([1., -1./np.sqrt(3)]) #vecteurs de la base hexagonale
    m = np.array([1, 1./np.sqrt(3)])
    for i in range(dimx):
        for j in range(dimy):
            hexagonal[:,i,j] = int(i)*n + int(j)*m
    index_flocon = np.where(matflocon >= 0)
    flocon = hexagonal[:,index_flocon[0],index_flocon[1]]
    
    fig = plt.figure()
    plt.scatter(flocon[0,:], flocon[1,:], c =matflocon[index_flocon[0],\
                  index_flocon[1]]-1., s = 750., \
                    cmap = 'Blues', marker = 'H')
    
    #plt.text(131, -17, r'Rayon flocon: ' +str(trusize/2)+' $\mu$m' , {'color': 'C0', 'fontsize': 75}, va="top", ha="right")  #''Rayon du flocon'' en dessous de celui-ci
    plt.xlabel(r'Rayon du flocon: '+str(trusize/2)+' $\mu$m', {'color': 'k', 'fontsize': 75})
        
        
    # maxx=max(hexagonal[0,index_flocon[0],index_flocon[1]])
    # maxy=max(hexagonal[1,index_flocon[0],index_flocon[1]])
    # minx=min(hexagonal[0,index_flocon[0],index_flocon[1]])
    # miny=min(hexagonal[1,index_flocon[0],index_flocon[1]])
    

    #plt.yticks((-18, -9, 0, 9, 18), (r'-300', r'-150', r'0', r'150', r'300'), color='k', size=50)
    #plt.xticks((40, 70, 100, 130, 160), (r'-300', r'-150', r'0', r'150', r'300'), color='k', size=50)
    
    plt.xlim(minx+35,maxx-35)
    plt.ylim(miny+35,maxy-35)
    #
    #
    #ATTENTION s'assurer que le directory existe. Ici, ''...\Saved_data\Simulation_data_ice'' existe
    #
    #
    my_path = os.path.join(path,"Saved_data","Simulation_data_ice","frame"+str(k))
    fig.savefig(str(my_path)+'.png')  
    #plt.axis("off") #Si on veut crop les axes et avoir que l'image
    #plt.show() 
   

def hexplot2(matflocon,k,name,xmm,ymm):
    
    dimx=np.shape(matflocon)[0]
    dimy=np.shape(matflocon)[1]
    
    #rusize=10 # en micron
    
    hexagonal = np.empty((2,dimx,dimy)) #array contenant les points hex
    n = np.array([1., 1./np.sqrt(3.)]) #vecteurs de la base hexagonale
    m = np.array([1., -1./np.sqrt(3.)])
    x=[]
    y=[]
    z=[]
    f='frame'
    # a=0
    # if a==0:
    #     t=0
    #else:
        #t=t+temp[a]
    #matflocon=frameflocon #frameflocon[a][:,:,5]  # pour la vapeur dernier depen du centre
    #matflocon=np.sum(frameflocon[a],axis=2) #pour la glace
    
    for b in range(0,dimx):
        for c in range(0,dimy):
            hexagonal[:,b,c] = int(b)*n + int(c)*m
            x.append( hexagonal[0,b,c])
            y.append( hexagonal[1,b,c])
            z.append(matflocon[b,c])
     
    if True: #si on cheke la glace
        index_flocon = np.where(matflocon>= 1) #depende de ce qu on veu prendre la pour glace
        
    else:#si on regarde la vapeur
        index_flocon = np.where(matflocon>= 0)
         
    xmax=max(hexagonal[0,index_flocon[0],index_flocon[1]])
    # maxy=max(hexagonal[1,index_flocon[0],index_flocon[1]])
    
    xmin=min(hexagonal[0,index_flocon[0],index_flocon[1]])
    # miny=min(hexagonal[1,index_flocon[0],index_flocon[1]])
    rayon=np.round(xmax-np.abs(xmin))*10 # le 10 est pour la taille de la grille (10 micron)


    
    #fig = plt.figure(num=1,figsize=(11*(1/np.sqrt(3)),10*(2/np.sqrt(3))))#num=1 trouver un facon de fermer limage
    
    fig = plt.figure(num=1,figsize=(11,10))

    
    im=plt.hexbin(x,y,z,gridsize=(dimx,dimy-1),edgecolors='k',linewidths=0.2,cmap='viridis',bins='log')#vmax=0.02,
    
    plt.xlim(xmm[0]-13,xmm[1]+13)
    plt.ylim(ymm[0]-5,ymm[1]+5)
    
    
    # a=plt.axis()
    # print(a)
    # plt.xticks(np.arange(0, 1, step=0.2))
    # plt.yticks(np.arange(0, 1, step=0.2))
    

    # plt.xticks(np.linspace(xmm[0]-5,xmm[1]+5,5),labels=[ax[0]])
    # plt.yticks(np.linspace(ymm[0]-5,ymm[1]+5,5),labels=[ax[1]])
    
    plt.axis('off')
    
   # textstr='$=10~[\mu m]$ disence lateral'
    
    #props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    
   # plt.text(1, 1, textstr, fontsize=14,
    #    verticalalignment='top', bbox=props)
    
    
    plt.title(r'Rayon du flocon: '+str(rayon/2)+' $\mu$m', {'color': 'k', 'fontsize': 20})
    # plt.xlabel(r'Rayon du flocon: '+str(trusize/2)+' $\mu$m', {'color': 'k', 'fontsize': 75})
    

    # fig.colorbar(im)
    
    #plt.legend(['A simple line'])
    
    fig.savefig('testfloconhex_'+name+'mx'+str(dimx)+'my'+str(dimy)+f+str(k)+'.png',dpi=300)
    
    #plt.close(all()) #close all?











    
##
##Initialisation
##

#type of data
    
#type_data=True:


debut=398 #normalement 0
fin=400
frame=np.arange(debut, fin)




# flocons=np.load(os.getcwd()+'\\Data_save\\Simulation_data_ice\\frameice'+str(510)+'.npy') #pour linste je veu zoomer sinon fin -1

# matflocon=np.sum(flocons,axis=2)
    


# dimx=np.shape(matflocon)[0]
# dimy=np.shape(matflocon)[1] 
# #sizemax=max(neige[0])
# #sizemin=min(neige[0])
# #size=(sizemax-sizemin) #largeur du flocon (où il y a glace) en terme de cases
# #trusize=size*10#e-6    # 1 case est 10 microns(deltax)
# hexagonal = np.empty((2,dimx,dimy)) #array contenant les points hex
# n = np.array([1, 1/np.sqrt(3)]) #vecteurs de la base hexagonale
# m = np.array([1, -1/np.sqrt(3)])
# for i in range(dimx):
#     for j in range(dimy):
#         hexagonal[:,i,j] = int(i)*n + int(j)*m
        
# index_flocon = np.where(matflocon >= 1)
   
# flocon = hexagonal[:,index_flocon[0],index_flocon[1]]
    
# maxx=np.round(max(hexagonal[0,index_flocon[0],index_flocon[1]]))
# maxy=np.round(max(hexagonal[1,index_flocon[0],index_flocon[1]]))
# minx=np.round(min(hexagonal[0,index_flocon[0],index_flocon[1]]))
# miny=np.round(min(hexagonal[1,index_flocon[0],index_flocon[1]]))
    
# xmm=[minx,maxx]
# ymm=[miny,maxy]
    
# dx=np.round(maxx-minx,2)
# dy=np.round(maxy-miny,2)
        
# axex=np.arange(0,dx,np.round(dx/5))
# axey=np.arange(0,dy,np.round(dy/5))
# ax=[axex,axey]
        


flocons=np.load(os.getcwd()+'\\Data_save\\Simulation_data_ice\\frameice'+str(fin-1)+'.npy') #pour linste je veu zoomer sinon fin -1

matflocon=np.sum(flocons,axis=2)
    


dimx=np.shape(matflocon)[0]
dimy=np.shape(matflocon)[1] 
#sizemax=max(neige[0])
#sizemin=min(neige[0])
#size=(sizemax-sizemin) #largeur du flocon (où il y a glace) en terme de cases
#trusize=size*10#e-6    # 1 case est 10 microns(deltax)
hexagonal = np.empty((2,dimx,dimy)) #array contenant les points hex
n = np.array([1, 1/np.sqrt(3)]) #vecteurs de la base hexagonale
m = np.array([1, -1/np.sqrt(3)])
for i in range(dimx):
    for j in range(dimy):
        hexagonal[:,i,j] = int(i)*n + int(j)*m
        
index_flocon = np.where(matflocon >= 1)
   
flocon = hexagonal[:,index_flocon[0],index_flocon[1]]
    
maxx=np.round(max(hexagonal[0,index_flocon[0],index_flocon[1]]))
maxy=np.round(max(hexagonal[1,index_flocon[0],index_flocon[1]]))
minx=np.round(min(hexagonal[0,index_flocon[0],index_flocon[1]]))
miny=np.round(min(hexagonal[1,index_flocon[0],index_flocon[1]]))
    
xmm=[minx,maxx]
ymm=[miny,maxy]
    
# dx=np.round(maxx-minx,2)
# dy=np.round(maxy-miny,2)
        
# axex=np.arange(0,dx,np.round(dx/5))
# axey=np.arange(0,dy,np.round(dy/5))
# ax=[axex,axey]
        





for k in reversed(range(debut,fin)):
    print('Traitement de l''image # '+str(k))
    path=os.getcwd()
    #
    #
    #ATTENTION s'assurer que le directory existe
    #
    #
    #Récupération de la matrice .npy
    
    if True:    
        # pour la glace
        flocons=np.load(os.getcwd()+'\\Data_save\\Simulation_data_ice\\frameice'+str(k)+'.npy')
        matflocon=np.sum(flocons,axis=2)
    
    else: 
        flocons=np.load(os.getcwd()+'\\Data_save\\Simulation_data_vap\\framev'+str(k)+'.npy')
        dimz=np.shape(flocons)[2]
        centerz=np.round(dimz/2).astype(np.uint16)
        matflocon=flocons[:,:,centerz]
        Pos=np.argwhere(matflocon==2)
        for pos in Pos:
            matflocon[pos[0],pos[1]]=np.nan
    
        
     



    #matflocon=flocons[:,:,6]
    #fig = plt.figure()
    #plt.imshow(matflocon,interpolation='spline16',cmap='viridis') #Pour visualisation de la matrice
    hexplot2(matflocon,k,'hello',xmm,ymm)
    