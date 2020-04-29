# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 13:30:44 2020

@author: plume_2.0
"""
#os mkdir
#Import
#njit voir la doc
import numpy as np
#import matplotlib.pyplot as plt 
#import scipy as sp
import scipy.constants as spc
import time 
#from joblib import Parallel, delayed # truc de multi proceesing
from scipy.ndimage import convolve
import os
#import concurrent.futures
#import multiprocessing as mp
from numba import njit

#FONCTION DE BASE
###############################################################################

#Fonction qui valide si le point est dans la boîte de croissance 
#Entré le poin (pos) et les dimention de la boit(dim_box)
#Sortie un Bolean
@njit()
def inbox(pos,dim_box):
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
    if (pos==np.array([0,0,0])).any() or (pos==(dim_box-1)).any() :
        front_box = True
    else:
        front_box = False
    return front_box


#Fonction voisin qui renvoit les voisins d'un point
#Entré le poin (pos) et la matrice 8X3 avec regle de plus proche voisin hexagonal(mat_voi)
#Sortie une matrice 8X3 avec les postion x y z des plus proche voisin du poin (pos)
@njit()
def voisin(pos,mat_voi):
    Vpos=pos+mat_voi
    #Vpos=np.ones([8,3]).astype(int)
    #Vpos=np.ones([8,3]).astype(int)*pos
    #Vpos=Vpos+mat_voi
    return Vpos

#Fonction valide si le point donné est dans le cristal 
#Entré le poin (pos) la matrice de glace (ice) et les dimention (dim_box)      #enfais je pourai le get avec shape
#sortie boulean
@njit()
def in_crystal(pos,ice,dim_box):
    if (pos >= 0).all() and (pos<dim_box).all():
        if (ice[pos[0],pos[1],pos[2]]==1)==True:
            in_ice=True 
           
        else:
            in_ice=False
            
    else:
        in_ice=False
    return in_ice


#Fonction qui retourne la frontiere en transition 
#Entré matrice de glace (ice) et la matrice 3X3X3 de plus proche voisin fait pour la convolution
#Sortie Matrice NX3 qui donne tout les poin dans la frontiere

def frontiere(ice,mat_voi_con): # se parallilse mallllllllllll fauderai la repencer mais cest du avec le pop pour en lever les doublon
    
    fronpos_mat=convolve(ice,mat_voi_con)
    
    fronpos_mat=fronpos_mat-100*ice
    
    fronpos=np.argwhere(fronpos_mat>0)
    
    return fronpos.astype(np.int32)
    # icepos=np.argwhere(ice==1)
    # fronpos=np.zeros([1,3])
    
    # for pos1 in icepos :
    #     Vpos=voisin(pos1,mat_voi)
    #     for pos2 in Vpos:
    #         if  inbox(pos2,dim_box) and (in_crystal(pos2,ice,dim_box)==False) :
    #             fronpos=np.concatenate((fronpos, [pos2]), axis=0)
                
            # elif  len(fronpos)>0 and ((np.sum(np.sum((fronpos)==pos2,axis=1)==3).any())>=2):
            #     fronpos.pop()
    
    # fronpos=np.unique(fronpos,axis=0)
    # fronpos=np.delete(fronpos,0,axis=0)

    
    
    

#fonction qui valide si on est dans la frontiere
#Entré le poin (pos) avec  matrice NX3 avec les poin de la frontiere
#Sortie 
#si dans la frontiere resort la postion du poin x y z et la position de la ligne ou il se trouve (entre 0 et N-1)
#si pas dans la frontiere redonne un nan
@njit()
def infron(pos,fronpos):

    f=(fronpos==np.array([pos[0],pos[1],pos[2],np.nan]))
    # print(f)
    # print()
    p=((np.sum(f,axis=1))==3)
    # print(p)
    if p.any():
        pos2=np.argwhere(p).astype(np.int32)[0][0]
        # print(type(np.argwhere(p)))
        # print(pos2)
        # print()
        # print(type(pos))
        # print(pos)
        # print()
        #out=np.concatenate((pos,pos2[0]))
        return pos2
    else:
        out=np.iinfo(np.int32).max
        return out


#Fonction qui calcule la constante de drain AVec les varible global 
#entré la valeur du coefficen de condensation alpha le rest son des variable global
#Sortie la valeur de K 
def K(alpha):
    K=alpha*((2*delta_dim)/(np.sqrt(3)*D))*np.sqrt((spc.k*T)/(2*spc.pi*m))
    return K

#fonction qui donne le bon coefficent de alpha en fonction des voisins
#Entré la position (pos) la matrice des voisin la dimention de la boit (dim_box) et la matrice de glace (ice)
#sortie la valeur de alpha (coéficien de condensation) pour le poin donné 

def valapha(pos,mat_voi,dim_box,ice):
    Vpos=voisin(pos,mat_voi)
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



#Fonction qui calcul la vitess de croissence normal a la surface
#Entré le coeficien de condensation la valeur de la concentration de vapeur supersature et la vitesse nu_kin
#Sortie vitesse normal de croissence de la glace
@njit()
def nun(alpha,Cvap,nu_kin):
    #print("Cvap")
    #print(Cvap)
    #print("nu_kin")
    #print(nu_kin)
    #print("alpha")
    #print(alpha)
   
    nu_n=alpha*Cvap*nu_kin
    return nu_n


#Fonction de la hauteur de croissance 
#Entré Vitese normal de croissence normal et une duré
#Sortie la variation de logeur de glace normale à la surface
@njit()
def delta_L(nu_n,temp):

    dL=nu_n*temp
#    if nu_n==0: #pas sur brian mais ca devrait marcher
#        dL=0

    return dL

#fonction qui donne le temp de croissence
#Entré une longeur l et un vitesse normale de croissence
#Sortie le temp reel que va prend la cellule a grandir de la longeur l
@njit()
def tcroi(longeur,nu_n): #longueur restant à croître
    if nu_n==0:
        tc=0
    else:
        tc=(longeur)/nu_n
    return tc

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





#Fonction de relaxation 1, alpha independant de sigma 
#condition aux frotières de la boîte, sont une source constante
 
#Entré l'état du systeme (matrice 3d vapeur, matrice 3d de glace, matrice NX4 de
# l'état des frontiere(position +longeur de glace)) matrice des voisin et les dimention
# la valeur de condenstion stable aux extremité(sigma limit) la liste de tout les poin (Pos)
def relax1(vap,ice,fron_state,dim_box,mat_voi,sigma_limit,Pos):
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
#               # on prend les voisin
                Vpos=voisin(pos1,mat_voi)
                #on calcul le alpha et le K
                alpha=valapha(pos1,mat_voi,dim_box,ice)
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
                Vpos=voisin(pos1,mat_voi)
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




#Fonction qui trouve le temps minimal pour que l'une des cellules se remplisse
#Entré État du systeme avec la longeu reel d'une cellule (L_cell)
#Sortie le temp minimal (exepeter 0) pour qu'aumoin une cellule soit plein
def tmin(fron_state,ice,vap,nu_kin,mat_voi,dim_box,Lcell):
    list_tc=[]
    for i1 in range(0,np.shape(fron_state)[0]):
        pos=np.array([fron_state[i1][0],fron_state[i1][1],fron_state[i1][2]]).astype(np.int32)
        longeur0=fron_state[i1][3]
        longeur=Lcell-longeur0
        Cvap=vap[pos[0],pos[1],pos[2]]
        alpha=valapha(pos,mat_voi,dim_box,ice)
        nu_n=nun(alpha,Cvap,nu_kin)
        tc=tcroi(longeur,nu_n)
        if tc!=0:    
            list_tc.append(tc)
        
        
    t_min=min(list_tc)
    return t_min
#dLcell

#fonction qui fais la croissence des frontiere
#Entré état du systeme et un temp minimal de croisence (tmin) 
#Sortie l'état de la frontiere avec les longeur de glace modifier en fonction du temp donné
def croissance(ice,fron_state,vap,delta_dim,mat_voi,dim_box,nu_kin,t_min):
    fron_state_c=fron_state.copy()
    fron_state_ini=fron_state.copy()
    for i1 in range(0,np.shape(fron_state)[0]):
        pos=np.array([fron_state[i1][0],fron_state[i1][1],fron_state[i1][2]]).astype(np.int32)
        alpha=valapha(pos,mat_voi,dim_box,ice)

        Cvap=vap[pos[0],pos[1],pos[2]]

        nu_n=nun(alpha,Cvap,nu_kin)
        
       
        dL=delta_L(nu_n,t_min)

        fron_state_c[i1][3]=(fron_state_ini[i1][3]+dL)

    return fron_state_c
            
#fron_stat plus sur si les positions de dans c'est un peu con je suis pas sur
    


#Fonction qui update la matrice ice et la liste frontiere
#Entré état du systeme avec la longeur minimal
#Sortie la matrice de glace modifie avec les nouvelle cellule
# et la matrice frontiere actualiser avec les nouveau poin et en 
# retirant ceux qui son devenu de la glace 
def update_fron(fron_state,ice,mat_voi_con,dim_box,Lcell):
    fron_state_new=np.zeros([1,4])
    ice_new=ice

    for state1 in fron_state:
        
        if np.round(state1[3],9)>=Lcell:
            
            ice_new[int(state1[0]),int(state1[1]),int(state1[2])]=1
                        
    state_new=frontiere(ice_new,mat_voi_con)

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



#Fonction de sauvegard
###############################################################################
def saveframev(it,vap):
    path=os.getcwd()+'\\Data_save\\Simulation_data_vap\\framev'+str(it)
    np.save(path,vap)
    
def saveframeice(it,ice):
    path=os.getcwd()+'\\Data_save\\Simulation_data_ice\\framev'+str(it)
    np.save(path,ice)
    #sauvagard txt guiaume

def saveframet(framet):
    path=os.getcwd()+'\\Data_save\\Simulation_data_sup\\framet'
    np.save(path,framet)
    #augarde dans un txt


# INITIALISATION VARIABLE 
###############################################################################


#Initiation de la matrice pour regarder les plus proches voisins

mat_voi=np.array([[-1,-1,0],[0,-1,0],[-1,0,0],[1,0,0],[0,1,0],[1,1,0],[0,0,1],[0,0,-1]]).astype(np.int32)


mat_voi_con=np.zeros([3,3,3]).astype(np.int32)

mat_voi_con[1,1,0]=1
mat_voi_con[1,1,2]=1

mat_voi_con[:,:,1]=1
mat_voi_con[0,2,1]=0
mat_voi_con[2,0,1]=0

#test la difusion simple sans reflexion pour les casse ni au extreme ni dans la frontiere
# mat_voi_con_diff=np.zeros([3,3,3]).astype(np.int32)

# mat_voi_con_diff[1,1,0]=1/6
# mat_voi_con_diff[1,1,2]=1/6

# mat_voi_con_diff[1,1,1]=0

# mat_voi_con_diff[:,:,1]=1/9
# mat_voi_con_diff[0,2,1]=0
# mat_voi_con_diff[2,0,1]=0

#sa marche pas vraiment
# Constantes :


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


#dimention dim_box[0]=dimx ect
dim_box=np.array([50,50,15]).astype(np.int32) 

center=(np.floor((dim_box)/2)).astype(np.int32)


#Definition de matrices 

# mat cristal 0 si pas dans le cristal
ice_ini=np.zeros(dim_box)
# mat vapeur varie en fonction de la densité d'eau en état vapeur entre 0 et 1 ?
vap=sigma_limit*(np.ones(dim_box))

ice_ini[center[0],center[1],center[2]]=1

#initalisation d un vecteur de parcour avec tout les position voulue
Pos=[]
for i1 in range(0,dim_box[0]):
    for i2 in range(0,dim_box[1]):
        for i3 in range(0,dim_box[2]):
            Pos.append(np.array([i1,i2,i3]).astype(np.int32))       






#initialisation de l eta initiale 
Vposini1=voisin(center,mat_voi)
for i1 in range(0,6):
    ice_ini[Vposini1[i1,0],Vposini1[i1,1],Vposini1[i1,2]]=1
    vap[Vposini1[i1,0],Vposini1[i1,1],Vposini1[i1,2]]=2    






#Initalisation de l etat de la frontiere 
fronpos_ini=frontiere(ice_ini,mat_voi_con)

fron_state=np.zeros([np.shape(fronpos_ini)[0],1])

fron_state=np.concatenate((fronpos_ini,fron_state),axis=1).astype(float)
    
# for pos in fronpos_ini:
#     fron_state.append((np.concatenate((pos,np.array([0])),axis=1)).astype(float))





# SIMULATION
###############################################################################
# plt.rcParams["figure.figsize"] = (10,10)
if True:
    # framev=[]
    # frameice=[]
    framet=[]
    # #framef=[]
    vap_ini=vap
    # frameice.append(ice_ini.copy())
    # framev.append(vap.copy())  #juste pour pas avoir tout commenter à chaque fois que je change de quoi
    
    #sauvegarde des etat initiaux et du compteur d'iteration des frame
    saveframev(00,vap_ini)
    saveframeice(00,ice_ini)
    continuer1=True
    i3=0    
    #Boucle Principale
    while continuer1:
        print(i3)
        print(np.sum(ice_ini))
        print()
        ice_c=ice_ini
        
        continuer2=True
        i4=0
        t1=time.time()
        #Boucle de la relaxation de la vapeur d'eaux 
        # elle attend que le champ de vapeur d'eau autour du cristal soit stable 
        # avent de la laisser passer à l'étape de croissance
        while  continuer2:
            t3=time.time()
            vap_c=relax1(vap_ini,ice_c,fron_state,dim_box,mat_voi,sigma_limit,Pos)
            t4=time.time()
            i4=i4+1
            print()
            print(t4-t3)
            print()
            #relax1(vap,ice,fron_state,dim_box,mat_voi,Pos):
            
            #Condition de sortie de la boucle si le l'erreur relatice est plus
            #basse que 0.01% peut etre changer article 0.005%
            if ((abs((vap_c-vap_ini)/vap_ini))<0.01).all() : #and i4>=10:
                #framev.append(vap_c.copy())
                vap_ini=vap_c
                continuer2=False
                print('nbr iteration pour convergence:')
                print(i4)
                print()
                t2=time.time()
                print('temp convergence:')
                print(t2-t1)
                print()
            else:
                vap_ini=vap_c
        
        t1=time.time()
        
        #étape de Croissence des frontiere
        
        t_min=tmin(fron_state,ice_c,vap_c,nu_kin,mat_voi,dim_box,Lcell)  
       # print(t_min)tmin(fron_state,ice,vap,nu_kin,mat_voi,dim_box,Lcell):
        fron_state_c=croissance(ice_c,fron_state,vap_c,delta_dim,mat_voi,dim_box,nu_kin,t_min)
        #(ice,fron_state,vap,delta_dim,mat_voi,dim_box,nu_kin,t_min)
        update=update_fron(fron_state_c,ice_c,mat_voi_con,dim_box,Lcell)
        # update_fron(fron_state,ice,mat_voi,dim_box,Lcell)
        #(fron_state,ice,mat_voi_con,dim_box,Lcell)
        fron_state=update[0]
        ice_ini=update[1]
        
        #crl !!!!!!!!!!!!!!!!!!!!
        #framef.append(fron_state.copy())
        # frameice.append(ice_ini.copy())
        framet.append(t_min)
        i3=i3+1
        saveframev(i3,vap_ini)
        saveframeice(i3,ice_ini)
        t2=time.time()
        print('temp pour opperation croi et save:')
        print(t2-t1)
        print()
        
        # Conditon de fin de la simulation si un poin de glace est rendu sur la
        # Frontiere de la boit a été modifie pour le moment car on limit la croissence
        # vertical
        if (fron_state==np.array([dim_box[0]-1,dim_box[1]-1,dim_box[2]-1,np.nan])).any() or (fron_state==np.array([1,1,1,np.nan])).any():
            print('fin')
            continuer1=False
        

