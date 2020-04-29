# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 16:05:19 2020

@author: alviss
"""

import numpy as np
import scipy.constants as spc
#import time 
from scipy.ndimage import convolve
import os
from numba import njit

#import concurrent.futures
#import multiprocessing as mp

#Definition des variable par défaut
###############################################################################

#Les matrice pour les voisins

#matrice normal
mat_voi=np.array([[-1,-1,0],[0,-1,0],[-1,0,0],[1,0,0],[0,1,0],[1,1,0],[0,0,1],[0,0,-1]]).astype(np.int32)

#matrice pour convolution
mat_voi_con=np.zeros([3,3,3]).astype(np.int32)

mat_voi_con[1,1,0]=1
mat_voi_con[1,1,2]=1

mat_voi_con[:,:,1]=1
mat_voi_con[0,2,1]=0
mat_voi_con[2,0,1]=0




#matrice pour les conexesion entre les grille

mat_conex=np.array([[-1,-1,-1]]).astype(np.int32)
for i1 in range(-1,2):
    for i2 in range(-1,2):
        for i3 in range(-1,2):
            if (i1!=-1 or i2!=-1 or i3!=-1):
                mat_conex=np.concatenate((mat_conex,np.array([[i1,i2,i3]])), axis=0)




# Les parametre dimentionel et physique de la simulation
##############################################################################
# valeur de la dimention (m)
delta_dim=10*10**(-6); 


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



#FONCTION DE BASE
###############################################################################


@njit()
def inbox(pos,dim_box):
    '''
    Fonction qui valide si le point est dans la boîte de croissance
    
    Parameters
    ----------
    pos : array
        le poin voulue  sous forme [x y z]
    dim_box : array
        les dimention de la boit sous forme [dimx dimy dimz]

    Returns
    -------
    in_box : Bolean
        vrai si dans la boite faut sinon

    '''
    
    if (pos >= 0).all() and (pos<dim_box).all():
        in_box = True    
    else:
        in_box = False  
    return in_box


#Fonction qui valide si on est dans la frontiere de la boite
#Entré le poin (pos) et les dimention de la boit(dim_box)
#Sortie un Bolean
@njit()
def frontbox(pos,dim_box):
    '''
    Fonction qui valide si on est dans la frontiere de la boite
    
    Parameters
    ----------
    pos : array
        le poin volue  sous forme [x y z]
    dim_box : array
        les dimention de la boit  sous forme [dimx dimy dimz]

    Returns
    -------
    front_box : Bolean
        vrai si dans la boite faut sinon

    '''
    if (pos==np.array([0,0,0])).any() or (pos==(dim_box-1)).any() :
        front_box = True
    else:
        front_box = False
    return front_box



@njit()
def voisin(pos):
    '''
    Fonction voisin qui renvoit les voisins d'un point

    Parameters
    ----------
    pos : array
        point voulue  sous forme [x y z]
        
    
    
    mat_voi : array
        la matrice des voisin est 8X3 et est d/fini comme un varialbe global
        
    Returns
    -------
    Vpos : array
        Un matrice 8X3 qui contien les position x y z des 8 voisin

    '''
    Vpos=pos+mat_voi
    return Vpos

#enfais je pourai le get avec shape
@njit()
def in_crystal(pos,ice,dim_box):
    '''
    Fonction valide si le point donné est dans le cristal

    Parameters
    ----------
    pos : array
        point voulue sous forme [x y z]
        
    ice :  array matrice 3D
        matrice avec 1 ou il y a de la glace des 0 sinon
        
    dim_box : array
        les dimention de la boit  sous forme [dimx dimy dimz]

    Returns
    -------
    in_ice : Boulean
        Vraie si le poin est de la glace faut sinon

    '''
    
    if (pos >= 0).all() and (pos<dim_box).all():
        if (ice[pos[0],pos[1],pos[2]]==1)==True:
            in_ice=True 
           
        else:
            in_ice=False
            
    else:
        in_ice=False
    return in_ice



def frontiere(ice): 
    
    '''
    Fonction valide si le point donné est dans le cristal

    Parameters
    ----------
    ice : array (matrice 3D)
        matrice avec 1 ou il y a de la glace des 0 sinon
        
    mat_voi_con :  array matrice 3D
        matrice 3X3X3 avec des 1 au positiond es plus proche voisin 0 sinon
        elle est defini comme un varible global plus haut
        

    Returns
    -------
    fronpos : array nX3
        une matrice qui contien dans ces ligne tout les poin fesant parti
        de la frontiere
    '''
    
    fronpos_mat=convolve(ice,mat_voi_con)
    
    fronpos_mat=fronpos_mat-100*ice
    
    fronpos=(np.argwhere(fronpos_mat>0)).astype(np.int32)
    return fronpos


    
    

@njit()
def infron(pos,fronpos):
    '''
    fonction qui valide si on est dans la frontiere

    Parameters
    ----------
    pos : array
       point voulue sous forme [x y z]
       
  fronpos : array nX3 ou nX4 si fron_state
        une matrice qui contien dans ces ligne tout les poin fesant parti
        de la frontiere. Plus la longeur de glace si on donne fron_state

    Returns
    -------
    out: numerical 
        redonne l'inder ou le poin a été trouver, ou le nombre maximal 
        ecrie en int32 si rien est trouver

    '''

    f=(fronpos==np.array([pos[0],pos[1],pos[2],np.nan]))

    p=((np.sum(f,axis=1))==3)

    if p.any():
        pos2=np.argwhere(p).astype(np.int32)[0][0]

        return pos2
    else:
        out=np.iinfo(np.int32).max
        return out





def valapha(pos,dim_box,ice):
    
    '''
    fonction qui donne le bon coefficent de alpha en fonction des voisins

    Parameters
    ----------
    pos : array
       point voulue sous forme [x y z]
       
    dim_box : array
        les dimention de la boit  sous forme [dimx dimy dimz]
        
        
    ice : array (matrice 3D)
        matrice avec 1 ou il y a de la glace des 0 sinon
        
        
    Returns
    -------
    alpha numerical 
        redonne la valeur de l alpha pour le poin donné
    '''
    
    
    Vpos=voisin(pos)
    #print()
    H=0 #nombre de voisin de glace horizotal initialisation
    V=0 #nombre de voisin de glace vertical initialisation
    i1=0
    for pos in Vpos:#voisin horizontaux
        i1=i1+1
        if i1<7:
            if in_crystal(pos,ice,dim_box)==True:
                H=H+1
        else:#voisin verticau
            if in_crystal(pos,ice,dim_box)==True:
                V=V+1
    #On donne la valeur de alpha (on set tu les val dansla fonction ou dans le code je sais pas)
    if  V==1 and H==0:
        alpha=0.0001 # valeur pour vertical (mi a 0 pour le moment pour les testes ''rapide'')
    elif V<=1 and H==1:
        alpha=0.5 # valeur pour horizatal et vertical petit  
    elif H==2 and V==0:
        alpha=0.5 #reduit pour aider la croisence de branche
    elif H==0 and V==0:
        alpha=0
    else:
        alpha=1

    return alpha 
#Note: peut probablement etre amélioré genre juste compter des vrais et des faux ?
#pas sur si besoin de vectoriser pas tend de gain a fair je croi 

# ici u update peut etre en le ver les 2 for et metre une conditon a la place

# jai tester de le metre njit mais je doit enlever le for avec les pos
# et c'est plus lent fac je sais pas


def K(alpha):
    '''
    Fonction qui calcule la constante de drain AVec les varible global 

    Parameters
    ----------
    alpha : numerical
        valeur de alpha
    
    les autre constent sont considere comme global dans ce code
    
    Returns
    -------
    K : numerical
        valeur de K la constent de drain

    '''
    K=alpha*((2*delta_dim)/(np.sqrt(3)*D))*np.sqrt((spc.k*T)/(2*spc.pi*m))
    return K




#Fonction qui calcul la vitess de croissence normal a la surface
#Entré le coeficien de condensation la valeur de la concentration de vapeur supersature et la vitesse nu_kin
#Sortie vitesse normal de croissence de la glace
@njit()
def nun(alpha,Cvap,nu_kin):
    '''
    Fonction qui calcul la vitess de croissence normal a la surface pour 
    information d'un poin donné

    Parameters
    ----------
    alpha : numerical
        Valeur de alpha au poin donner 
    Cvap : numerical
        Valeur de la concentration de vapeur 
    nu_kin : numerical 
        Valeur vitesse de croisence

    Returns
    -------
    nu_n :  numerical
        vitesse de croissence normal à la surface

    '''
   
    nu_n=alpha*Cvap*nu_kin
    return nu_n



@njit()
def delta_L(nu_n,temp):
    '''
    Fonction de la hauteur de croissance 

    Parameters
    ----------
    nu_n : numerical
        Vitesse de croissence normal
        
    temp : numerical
      le temp de croisence
        

    Returns
    -------
    dL : numerical
        la variation de la longeur normal de la cellule de glace

    '''

    dL=nu_n*temp
    if nu_n==0: #eviter le ca derreu de plus proche voisin  
        dL=0

    return dL

    


@njit()
def tcroi(longeur,nu_n):
    '''
    fonction qui donne le temp de croissence

    Parameters
    ----------
    longeur : numerical
        Longeur de glace
    nu_n : numerical
        vitesse de croisence normal

    Returns
    -------
    tc : numerical
        le temp que va prendre la cellule pouyr grandire de L

    '''
    if nu_n==0: # sa deverai etre en temp in fini mais sa cause des problemes avec la division par 0
        tc=0
    else:
        tc=(longeur)/nu_n
    return tc

    
@njit()
def conex_vap_large_fine(pos_large,vap_fine):
    
    #tout est fai en supposent un grille 3 foi plus grosse
    #avec le bonne intersetion [20 ,20 , 34] donc la grille large 
    #[74 74 74] et la fini [102, 102, 18]
    
    dim_box_fine=np.shape(vap_fine) 
    
    #dim_large=np.shape(vap_large) 
    # estimation de la position dans autre referentiel
    
    pos_r=pos_large
    pos_r=(pos_large*3-3*np.array([19,19,33]))
    
    Cvap=0
    
    pos1=(pos_r+mat_conex).astype(np.int32)
    
    i1=0
    for p in pos1:
        if (p >= 0).all() and (p<dim_box_fine).all():
            Cvap=Cvap+vap_fine[p[0],p[1],p[2]]
            i1=i1+1
            #print('oui') peut etre prob de chifre voisin?
        
    Cvap=Cvap/i1
    
    return Cvap
    
    
    
    
@njit()
def conex_vap_fine_large(pos_fine,vap_large):
    
    # dim_box_fine=np.shape(vap_fine) 
    
    #dim_large=np.shape(vap_large) 
    # estimation de la position dans autre referentiel
    
    pos_r=pos_fine
    pos_r=np.round((pos_fine/3+np.array([19,19,33]))).astype(np.int32)
    
    Cvap=vap_large(pos_r[0],pos_r[1],pos_r[2])
    
    
    return Cvap
    
    
    

#Fonction pour les somme de relaxe1
###############################################################################

# def sum_fron():
#     # on prend les voisin
#     Vpos=voisin(pos1,mat_voi)
#     #on calcul le alpha et le K
#     alpha=valapha(pos1,mat_voi,dim_box,ice)
#     Kval=K(alpha)
#     # indice pour savoir si on un voisin horizotal ou vetical
#     i1=0
#     #pour tout les voisins
#     for pos2 in Vpos:
#         i1=i1+1
#         #si le voisin est horizontal donc affect la somme 1+
#         if i1<7:
#             #si le voisin est de glace
#             #condtion de reflexion
#             #la somme prend la vapeur du poin actuel et non du voisin
#             if in_crystal(pos2,ice,dim_box) == True:
#                 sum1=sum1+vap[pos1[0],pos1[1],pos1[2]]
#                 #si le voisin est a l'exterieur de la boit
#                 # condtion de reflexion aussi (pour etre enlever)
#             elif  inbox(pos2,dim_box) == False:
#                 sum1=sum1+vap[pos1[0],pos1[1],pos1[2]]
#                     #sinon le voisin est de la vapeur 
#                     #la somme prend donc la valeur 
#                     #de vapeur du voisin
#             else:
#                 sum1=sum1+vap[pos2[0],pos2[1],pos2[2]]
#                 #si on est vertical on influence la somme 2
#         else:
#             #les meme condtion que plus haut
#             if in_crystal(pos2,ice,dim_box) == True:
#                 sum2=sum2+vap[pos1[0],pos1[1],pos1[2]]
#             elif inbox(pos2,dim_box) == False:
#                 sum2=sum2+vap[pos1[0],pos1[1],pos1[2]]
#             else:
#                 sum2=sum2+vap[pos2[0],pos2[1],pos2[2]]
#                     #une foir les somme fini on écrie la nouvelle vapeur
#                     #dans la matrice vap out avec la K!=0
    
#     return ((2/3)*sum1+sum2)/(Kval+6)


#FONCTION AVENCÉ
###############################################################################


def relax1_fine(vap_fine,vap_large,ice_fine,fron_state,dim_box_fine,Pos_fine,VOISIN_fine):
    '''
    Fonction de relaxation 1, alpha independant de sigma 
    condition aux frotières de la boîte, sont une source constante

    Parameters
    ----------
    vap : array matrice 3D
        Matrcie 3d qui contien a chaque poin les valeur de vapeur d'eau
        
    ice : array matrice 3D
         matrice avec 1 ou il y a de la glace des 0 sinon
         
    fron_state : array matrice nX4
        Matrice des état de la frontiere contien dans ces ligne les postion
        et la longeur de glace dans cellule sous la forme [x y z l]
        
    dim_box : array
         les dimention de la boit  sous forme [dimx dimy dimz]
         
    sigma_limit : numerical 
        valeur de concentration a la frontiere
    Pos :  Array nX3
        liste content tout les poin de la matrice vapeur dans ces ligne
        sous la forme [x y z]
        
    variable global de position de voisin de paratere physique se trouve plus 
    haut
    Returns
    -------
    vap_out : array matrice 3D
           la matrice de la vapeur relaxer(apres diffusion et absorbtion)
           chaque poin contien la concentration de vapeu a se poin sauf
           pour la glace celle si est just toujour laiser a 2 pour etre
           extraite poss prossesing
    '''

    #pmultiprooooccees
    #def relax1_multi(pos):
    #peut metre les valeur local pour certin truc ici voir si sa aide?
    #s=np.shape(vap)
    #vap_r=np.reshape(vap,[s[0]*s[1]*s[2],1,1])
    #ice_r=np.reshape(ice,[s[0]*s[1]*s[2],1,1])
    #np.unravel_index(np.ravel_multi_index((Pos[10321]), vap.shape), vap_r.shape)
    vap_out=np.zeros(np.shape(vap_fine))
        
    for pos1 in Pos_fine:
        itpos=0
        #initialisation des somme 
        sum1=0
        sum2=0
        #si a la frontiere on mes la valeur sigma_limit comme concentration de vapeur
        if frontbox(pos1,dim_box_fine) == True:
                vap_out[pos1[0],pos1[1],pos1[2]]=conex_vap_fine_large(pos1,vap_large) ########## modifie pa sur 
        
        #si on est pas dans le cristal on est de la vapeur donc on rentre ici
        elif in_crystal(pos1,ice_fine,dim_box_fine)== False:
            #Si le poin est dans la frontiere il posede une valeur de K 
            # (constente de drain) qui es t non null on a donc l'équation 
            #suivente
            if (infron(pos1,fron_state)!=np.iinfo(np.int32).max):   
                # on prend les voisin
                Vpos=VOISIN_fine[itpos]
                #on calcul le alpha et le K
                alpha=valapha(pos1,dim_box_fine,ice_fine)
                Kval=K(alpha)
                # indice pour savoir si on un voisin horizotal ou vetical
                i1=0
                #pour tout les voisins
                for pos2 in Vpos:
                    i1=i1+1
                    #si le voisin est horizontal donc affect la somme 1+
                    if i1<7:
                        #si le voisin est de glace
                        #condtion de reflexion
                        #la somme prend la vapeur du poin actuel et non du voisin
                        if in_crystal(pos2,ice_fine,dim_box_fine) == True:
                             sum1=sum1+vap_fine[pos1[0],pos1[1],pos1[2]]
                        #si le voisin est a l'exterieur de la boit
                        # condtion de reflexion aussi (pour etre enlever)
                        elif  inbox(pos2,dim_box_fine) == False:
                            sum1=sum1+vap_fine[pos1[0],pos1[1],pos1[2]]
                        #sinon le voisin est de la vapeur 
                        #la somme prend donc la valeur 
                        #de vapeur du voisin
                        else:
                             sum1=sum1+vap_fine[pos2[0],pos2[1],pos2[2]]
                    #si on est vertical on influence la somme 2
                    else:
                        #les meme condtion que plus haut
                        if in_crystal(pos2,ice_fine,dim_box_fine) == True:
                            sum2=sum2+vap_fine[pos1[0],pos1[1],pos1[2]]
                        elif inbox(pos2,dim_box_fine) == False:
                            sum2=sum2+vap_fine[pos1[0],pos1[1],pos1[2]]
                        else:
                            sum2=sum2+vap_fine[pos2[0],pos2[1],pos2[2]]
                    #une foir les somme fini on écrie la nouvelle vapeur
                    #dans la matrice vap out avec la K!=0
                    vap_out[pos1[0],pos1[1],pos1[2]]=((2/3)*sum1+sum2)/(Kval+6)
                    
                            
            #si on est dans la vapeur mais pas dans la frontiere on les meme
            #regle mais on K=0
            else:
                Vpos=VOISIN_fine[itpos]
                i2=0
                
                for pos3 in Vpos:
                    i2=i2+1
                    if i2<7:
                        if in_crystal(pos3,ice_fine,dim_box_fine) == True:
                            sum1=sum1+ vap_fine[pos1[0],pos1[1],pos1[2]]
                        elif  inbox(pos3,dim_box_fine)==False:
                            sum1=sum1+vap_fine[pos1[0],pos1[1],pos1[2]]
                        else:
                            sum1=sum1+vap_fine[pos3[0],pos3[1],pos3[2]] 
                    else:
                        if in_crystal(pos3,ice_fine,dim_box_fine) == True:
                            sum2=sum2+ vap_fine[pos1[0],pos1[1],pos1[2]]
                        elif inbox(pos3,dim_box_fine) == False:
                            sum2=sum2+vap_fine[pos1[0],pos1[1],pos1[2]]
                        else:
                            sum2=sum2+vap_fine[pos3[0],pos3[1],pos3[2]]
                
                vap_out[pos1[0],pos1[1],pos1[2]]=((2/3)*sum1+sum2)/6
                

        #si le poin est de la glace  on met un element qu on sais qui sera unique
        # a la glace pour pouvoir modifier apres pour image pour le moment 2
        # il faut pas metre de quoi qui se divise pas dut a la condition 
        # de fin de relazation voir la section du code simulation
        else:
            vap_out[pos1[0],pos1[1],pos1[2]]=2#np.nan
        itpos=itpos+1    
    

    return vap_out

def relax1_large(vap_large,vap_fine,dim_box_large,sigma_limit,Pos_limite_large,Pos_large,VOISIN_large):
    '''
    Fonction de relaxation 1, alpha independant de sigma 
    condition aux frotières de la boîte, sont une source constante

    Parameters
    ----------
    vap : array matrice 3D
        Matrcie 3d qui contien a chaque poin les valeur de vapeur d'eau
        
    ice : array matrice 3D
         matrice avec 1 ou il y a de la glace des 0 sinon
         
    fron_state : array matrice nX4
        Matrice des état de la frontiere contien dans ces ligne les postion
        et la longeur de glace dans cellule sous la forme [x y z l]
        
    dim_box : array
         les dimention de la boit  sous forme [dimx dimy dimz]
         
    sigma_limit : numerical 
        valeur de concentration a la frontiere
    Pos :  Array nX3
        liste content tout les poin de la matrice vapeur dans ces ligne
        sous la forme [x y z]
        
    variable global de position de voisin de paratere physique se trouve plus 
    haut
    Returns
    -------
    vap_out : array matrice 3D
           la matrice de la vapeur relaxer(apres diffusion et absorbtion)
           chaque poin contien la concentration de vapeu a se poin sauf
           pour la glace celle si est just toujour laiser a 2 pour etre
           extraite poss prossesing
    '''
    
    vap_out=np.zeros(np.shape(vap_large))
    
    for pos1 in Pos_large:
        #initialisation des somme 
        sum1=0
        sum2=0
        #si a la frontiere on mes la valeur sigma_limit comme concentration de vapeur
        if frontbox(pos1,dim_box_large) == True:
                vap_out[pos1[0],pos1[1],pos1[2]]=sigma_limit      
        
        #si on est pas dans le cristal on est de la vapeur donc on rentre ici
        elif ((np.sum(pos1==Pos_limite_large,axis=1))==3).any():
            vap_out[pos1[0],pos1[1],pos1[2]]=conex_vap_large_fine(pos1,vap_fine)
        else:
                Vpos=voisin(pos1)
                i2=0
                
                for pos3 in Vpos:
                    i2=i2+1
                    if i2<7:
                        if  inbox(pos3,dim_box_large)==False:
                            sum1=sum1+vap_large[pos1[0],pos1[1],pos1[2]]
                        else:
                            sum1=sum1+vap_large[pos3[0],pos3[1],pos3[2]] 
                    else:
                        if inbox(pos3,dim_box_large) == False:
                            sum2=sum2+vap_large[pos1[0],pos1[1],pos1[2]]
                        else:
                            sum2=sum2+vap_large[pos3[0],pos3[1],pos3[2]]
                # print(((2/3)*sum1+sum2)/6)
                vap_out[pos1[0],pos1[1],pos1[2]]=((2/3)*sum1+sum2)/6
                
# on remet tu la vielle fonction avec delta tau????????

    return vap_out





def relax1(vap,ice,fron_state,dim_box,sigma_limit,Pos):
    '''
    Fonction de relaxation 1, alpha independant de sigma 
    condition aux frotières de la boîte, sont une source constante

    Parameters
    ----------
    vap : array matrice 3D
        Matrcie 3d qui contien a chaque poin les valeur de vapeur d'eau
        
    ice : array matrice 3D
         matrice avec 1 ou il y a de la glace des 0 sinon
         
    fron_state : array matrice nX4
        Matrice des état de la frontiere contien dans ces ligne les postion
        et la longeur de glace dans cellule sous la forme [x y z l]
        
    dim_box : array
         les dimention de la boit  sous forme [dimx dimy dimz]
         
    sigma_limit : numerical 
        valeur de concentration a la frontiere
    Pos :  Array nX3
        liste content tout les poin de la matrice vapeur dans ces ligne
        sous la forme [x y z]
        
    variable global de position de voisin de paratere physique se trouve plus 
    haut
    Returns
    -------
    vap_out : array matrice 3D
           la matrice de la vapeur relaxer(apres diffusion et absorbtion)
           chaque poin contien la concentration de vapeu a se poin sauf
           pour la glace celle si est just toujour laiser a 2 pour etre
           extraite poss prossesing
    '''
     
    
    #pmultiprooooccees
    #def relax1_multi(pos):
    #peut metre les valeur local pour certin truc ici voir si sa aide?
    #s=np.shape(vap)
    #vap_r=np.reshape(vap,[s[0]*s[1]*s[2],1,1])
    #ice_r=np.reshape(ice,[s[0]*s[1]*s[2],1,1])
    #np.unravel_index(np.ravel_multi_index((Pos[10321]), vap.shape), vap_r.shape)
    vap_out=np.zeros(np.shape(vap))
    
    for pos1 in Pos:
        #initialisation des somme 
        sum1=0
        sum2=0
        #si a la frontiere on mes la valeur sigma_limit comme concentration de vapeur
        if frontbox(pos1,dim_box) == True:
                vap_out[pos1[0],pos1[1],pos1[2]]=sigma_limit      
        
        #si on est pas dans le cristal on est de la vapeur donc on rentre ici
        elif in_crystal(pos1,ice,dim_box)== False:
            #Si le poin est dans la frontiere il posede une valeur de K 
            # (constente de drain) qui es t non null on a donc l'équation 
            #suivente
            if (infron(pos1,fron_state)!=np.iinfo(np.int32).max):   
                # on prend les voisin
                Vpos=voisin(pos1)
                #on calcul le alpha et le K
                alpha=valapha(pos1,dim_box,ice)
                Kval=K(alpha)
                # indice pour savoir si on un voisin horizotal ou vetical
                i1=0
                #pour tout les voisins
                for pos2 in Vpos:
                    i1=i1+1
                    #si le voisin est horizontal donc affect la somme 1+
                    if i1<7:
                        #si le voisin est de glace
                        #condtion de reflexion
                        #la somme prend la vapeur du poin actuel et non du voisin
                        if in_crystal(pos2,ice,dim_box) == True:
                             sum1=sum1+vap[pos1[0],pos1[1],pos1[2]]
                        #si le voisin est a l'exterieur de la boit
                        # condtion de reflexion aussi (pour etre enlever)
                        elif  inbox(pos2,dim_box) == False:
                            sum1=sum1+vap[pos1[0],pos1[1],pos1[2]]
                        #sinon le voisin est de la vapeur 
                        #la somme prend donc la valeur 
                        #de vapeur du voisin
                        else:
                             sum1=sum1+vap[pos2[0],pos2[1],pos2[2]]
                    #si on est vertical on influence la somme 2
                    else:
                        #les meme condtion que plus haut
                        if in_crystal(pos2,ice,dim_box) == True:
                            sum2=sum2+vap[pos1[0],pos1[1],pos1[2]]
                        elif inbox(pos2,dim_box) == False:
                            sum2=sum2+vap[pos1[0],pos1[1],pos1[2]]
                        else:
                            sum2=sum2+vap[pos2[0],pos2[1],pos2[2]]
                    #une foir les somme fini on écrie la nouvelle vapeur
                    #dans la matrice vap out avec la K!=0
                    vap_out[pos1[0],pos1[1],pos1[2]]=((2/3)*sum1+sum2)/(Kval+6)
                    
                            
            #si on est dans la vapeur mais pas dans la frontiere on les meme
            #regle mais on K=0
            else:
                Vpos=voisin(pos1)
                i2=0
                
                for pos3 in Vpos:
                    i2=i2+1
                    if i2<7:
                        if in_crystal(pos3,ice,dim_box) == True:
                            sum1=sum1+ vap[pos1[0],pos1[1],pos1[2]]
                        elif  inbox(pos3,dim_box)==False:
                            sum1=sum1+vap[pos1[0],pos1[1],pos1[2]]
                        else:
                            sum1=sum1+vap[pos3[0],pos3[1],pos3[2]] 
                    else:
                        if in_crystal(pos3,ice,dim_box) == True:
                            sum2=sum2+ vap[pos1[0],pos1[1],pos1[2]]
                        elif inbox(pos3,dim_box) == False:
                            sum2=sum2+vap[pos1[0],pos1[1],pos1[2]]
                        else:
                            sum2=sum2+vap[pos3[0],pos3[1],pos3[2]]
                
                vap_out[pos1[0],pos1[1],pos1[2]]=((2/3)*sum1+sum2)/6
                

        #si le poin est de la glace  on met un element qu on sais qui sera unique
        # a la glace pour pouvoir modifier apres pour image pour le moment 2
        # il faut pas metre de quoi qui se divise pas dut a la condition 
        # de fin de relazation voir la section du code simulation
        else:
            vap_out[pos1[0],pos1[1],pos1[2]]=2#np.nan
            
    

    return vap_out
                    
#Fonction de relaxation 2, alpha dependant de sigma 
#Condition aux frontières de la boîte sont une source constante??




def tmin(fron_state,ice,vap,nu_kin,dim_box,Lcell):
    '''
    Fonction qui trouve le temps minimal pour que l'une des cellules se remplisse

    Parameters
    ----------
    fron_state : array matrice nX4
         Matrice des état de la frontiere contien dans ces ligne les postion
         et la longeur de glace dans cellule sous la forme [x y z l]
    ice : array matrice 3D
          matrice avec 1 ou il y a de la glace des 0 sinon
    vap : array matrice 3D
         Matrcie 3d qui contien a chaque poin les valeur de vapeur d'eau
    nu_kin : numerical
        vitesse de croissence normal a la surface

    dim_box : array
        les dimention de la boit  sous forme [dimx dimy dimz]
    Lcell : numerical
        longeur reel d'une cellule

    Returns
    -------
    t_min : numerical
        le temp minimal pour que une cellule soit remplie

    '''
    list_tc=[]
    for i1 in range(0,np.shape(fron_state)[0]):
        pos=np.array([fron_state[i1][0],fron_state[i1][1],fron_state[i1][2]]).astype(np.int32)
        longeur0=fron_state[i1][3]
        longeur=Lcell-longeur0
        Cvap=vap[pos[0],pos[1],pos[2]]
        alpha=valapha(pos,dim_box,ice)
        nu_n=nun(alpha,Cvap,nu_kin)
        tc=tcroi(longeur,nu_n)
        if tc!=0:    
            list_tc.append(tc)
        
        
    t_min=min(list_tc)
    return t_min



def croissance(ice,fron_state,vap,delta_dim,dim_box,nu_kin,t_min):
    
    '''
     Fonction qui fais la croissence des frontiere

    Parameters
    ---------- 
    
    ice : array matrice 3D
          matrice avec 1 ou il y a de la glace des 0 sinon
          
    fron_state : array matrice nX4
         Matrice des état de la frontiere contien dans ces ligne les postion
         et la longeur de glace dans cellule sous la forme [x y z l]
   
    vap : array matrice 3D
         Matrcie 3d qui contien a chaque poin les valeur de vapeur d'eau
    
   delta_dim : numerical
        La taille (dx) voir les truc de l'article
        
     dim_box : array
        les dimention de la boit  sous forme [dimx dimy dimz]     
        
    nu_kin : numerical
        vitesse de croissence normal a la surface
        
    t_min : numerical
        le temp minimal pour que une cellule soit remplie

   

    Returns
    -------
    fron_state_c : array matrice nX4
         Matrice des état de la frontier apres le temp t_min de croissence
         de la  cellule sous la forme [x y z l]
        
    
    '''
    fron_state_c=fron_state.copy()
    fron_state_ini=fron_state.copy()
    for i1 in range(0,np.shape(fron_state)[0]):
        pos=np.array([fron_state[i1][0],fron_state[i1][1],fron_state[i1][2]]).astype(np.int32)
        alpha=valapha(pos,dim_box,ice)

        Cvap=vap[pos[0],pos[1],pos[2]]

        nu_n=nun(alpha,Cvap,nu_kin)
        
       
        dL=delta_L(nu_n,t_min)

        fron_state_c[i1][3]=(fron_state_ini[i1][3]+dL)

    return fron_state_c
            
#fron_stat plus sur si les positions de dans c'est un peu con je suis pas sur
    

def update_fron(fron_state,ice,dim_box,Lcell):
    '''
    Fonction qui update la matrice ice et la liste frontiere

    Parameters
    ----------
    fron_state : array matrice nX4
         Matrice des état de la frontiere contien dans ces ligne les postion
         et la longeur de glace dans cellule sous la forme [x y z l]
    ice : array matrice 3D
          matrice avec 1 ou il y a de la glace des 0 sinon

    dim_box : array
        les dimention de la boit  sous forme [dimx dimy dimz]   
        
    Lcell : numerical
        longeur reel d'une cellule

    Returns
    -------
    list
        une liste qui contien la nouvelle frontiere et la nouvelle 
        matrice de glace

    '''
    
    
    
    fron_state_new=np.zeros([1,4])
    ice_new=ice

    for state1 in fron_state:
        
        if np.round(state1[3],9)>=Lcell:
            
            ice_new[int(state1[0]),int(state1[1]),int(state1[2])]=1
                        
    state_new=frontiere(ice_new)

    for state2 in state_new:
        
        #pos=infron(state2,fron_state)
        ind=infron(state2,fron_state)
        pos=np.array([state2[0],state2[1],state2[2],ind]).astype(int)
        # print(np.array([np.concatenate((state2,np.array([0])))]))
        # print(fron_state_new)
        # print(pos)
        # print(np.concatenate((fron_state_new,state2), axis=0))
        if ind != np.iinfo(np.int32).max :
            #print(ind)
            #print(pos)
            # print(np.array([[pos[0],pos[1],pos[2],fron_state[int(pos[3])][3]]]))
            fron_state_new=np.concatenate((fron_state_new,np.array([[pos[0],pos[1],pos[2],fron_state[pos[3]][3]]])), axis=0)
            #print([fronpos[b][0],fronpos[b][1],fronpos[b][2],fron_state[pos[3]][3]])
        else:
            state2=np.array([np.concatenate((state2,np.array([0])))])
            fron_state_new=np.concatenate((fron_state_new,state2), axis=0)
            #append([state2[0],state2[1],state2[2],0])
            
    fron_state_new=np.delete(fron_state_new,0,axis=0)

    return [fron_state_new,ice_new]

#np.concatenate((fronpos, [pos2]), axis=0)

#FONCTION IMAGE
###############################################################################


#sauvagard txt guiaume
#Fonction de sauvegard
###############################################################################
def saveframev(it,dossier,vap):
    '''
    Fonction qui enregistre les matric de vapeur dans en dossier
    
    Parameters
    ----------
    it : numerical
        le numeeaux de l iterration de la variable que l on veu 
        sausvegarder.
        
    dosier : string
        le nom du dossier dans le quel en veut que les données soit
        sauvegarder.
        
    vap : array
        qui contien la matrice de vapeur

    Returns
    -------
    None.

    '''
    #os.mkdir(dossier)
    
    path=os.getcwd()+'\\'+dossier+'\\framev'+str(it)
    np.save(path,vap)
    
def saveframeice(it,dossier,ice):
    '''
    Fonction qui enregistre les matric de glace dans en dossier
    
    Parameters
    ----------
    it : numerical
        le numereaux de l iterration de la variable que l on veu 
        sausvegarder.
        
    dosier : string
        le nom du dossier dans le quel en veut que les données soit
        sauvegarder.
        
     ice : array
        array qui contien la matrice de vapeur

    Returns
    -------
    None.

    '''
    
    #os.mkdir(dossier)
    path=os.getcwd()+'\\'+dossier+'\\frameice'+str(it)
    np.save(path,ice)
    
     


def saveframet(framet,name):
    '''
    Fonction qui enregistre les matric de vapeur dans en dossier
    
    Parameters
    ----------
    it : numerical
        le numeeaux de l iterration de la variable que l on veu 
        sausvegarder.
        
    dosier : string
        le nom du dossier dans le quel en veut que les données soit
        sauvegarder.
        
    framet :
        liste des temp entre tout les iterations
        

    Returns
    -------
    None.

    '''
    # path=os.getcwd()+'\\Data_save\\Simulation_data_sup\\framet'
    path=os.getcwd()+'framet'
    np.save(path,framet)
    #augarde dans un txt
        
##############################################################################
