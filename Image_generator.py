# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 23:56:24 2020

@author: alviss
"""
###############################################################################
import numpy as np
import matplotlib.pyplot as plt 
#import scipy as sp
#import scipy.constants as spc
import time 
#from joblib import Parallel, delayed # truc de multi proceesing
#from scipy.ndimage import convolve
import os
import concurrent.futures
#fonction d<afficha
###############################################################################

plt.rcParams["figure.figsize"] = (30,30)
def hexplot(matflocon,frame):
    dimx=np.shape(matflocon)[0]
    dimy=np.shape(matflocon)[1]
    hexagonal = np.empty((2,dimx,dimy)) #array contenant les points hex
    n = np.array([1., -1./np.sqrt(3.)]) #vecteurs de la base hexagonale
    m = np.array([1., 1./np.sqrt(3.)])
    for i in range(dimx):
        for j in range(dimy):
            hexagonal[:,i,j] = int(i)*n + int(j)*m
    index_flocon = np.where(matflocon >= 1)  #0 sion veut tous les points de la matrice.
    flocon = hexagonal[:,index_flocon[0],index_flocon[1]]
    fig = plt.figure()
    plt.scatter(flocon[0,:], flocon[1,:], c =matflocon[index_flocon[0],\
                 index_flocon[1]]-1., s = 500., \
                marker = 'H') # s est la taille des points scatter
    
        
    plt.axis()
    fig.savefig('testfloconhex_'+'mx'+str(dimx)+'my'+str(dimy)+frame+'.png')
    plt.show()



def hexplot2(frameflocon,dimx,dimy,name):
    #xi=np.linspace(0,dimx-1,dimx)
    #yi=np.linspace(0,dimy-1,dimy)
    hexagonal = np.empty((2,dimx,dimy)) #array contenant les points hex
    n = np.array([1., -1./np.sqrt(3.)]) #vecteurs de la base hexagonale
    m = np.array([1., 1./np.sqrt(3.)])
    x=[]
    y=[]
    z=[]
    f='frame'
    a=0
    if a==0:
        t=0
    #else:
        #t=t+temp[a]
    matflocon=frameflocon #frameflocon[a][:,:,5]  # pour la vapeur dernier depen du centre
        #matflocon=np.sum(frameflocon[a],axis=2) #pour la glace
    for b in range(0,dimx):
        for c in range(0,dimy):
            hexagonal[:,b,c] = int(b)*n + int(c)*m
            x.append( hexagonal[0,b,c])
            y.append( hexagonal[1,b,c])
            z.append(matflocon[b,c])
        
    index_flocon = np.where(matflocon <1) #depende de ce qu on veu prendre
         
    maxx=max(hexagonal[0,index_flocon[0],index_flocon[1]])
    maxy=max(hexagonal[1,index_flocon[0],index_flocon[1]])
    
    minx=min(hexagonal[0,index_flocon[0],index_flocon[1]])
    miny=min(hexagonal[1,index_flocon[0],index_flocon[1]])
        
    fig = plt.figure()
        
    plt.hexbin(x,y,z,gridsize=(dimx-1,dimy-1))
    plt.xlim(minx-5,maxx+5)
    plt.ylim(miny-5,maxy+5)
    fig.savefig('testfloconhex_'+name+'mx'+str(dimx)+'my'+str(dimy)+f+str(t)+'.png')
    
    
    
    
