import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

#%% Définitions des variabes


numberOfFrames = 15 #Détermine le nombre d'image final
cwd = os.getcwd() 
dataPath = r"\Data_save\Simulation_data_ice" # Les matrices de glaces
frametPath =  r"\Data_save\Simulation_data_sup" # le framet
plt.rcParams["figure.figsize"] = (11,10)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["figure.frameon"] = False
#figure.frameon : True 
plt.rcParams["figure.edgecolor"] = 'k'
plt.rcParams["figure.facecolor"] = 'k'

#%%
#La fonction Hexaplot crée les figure à pratir des données
#Puis les enregistrent en images.
def hexplot(matflocon,k,xmm,ymm):
    
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
        
    index_flocon = np.where(matflocon>= 1) #depende de ce qu on veu prendre la pour glace
         
    xmax=max(hexagonal[0,index_flocon[0],index_flocon[1]])
    # maxy=max(hexagonal[1,index_flocon[0],index_flocon[1]])
    
    xmin=min(hexagonal[0,index_flocon[0],index_flocon[1]])
    # miny=min(hexagonal[1,index_flocon[0],index_flocon[1]])
    rayon=np.round(xmax-np.abs(xmin))*10 # le 10 est pour la taille de la grille (10 micron)

    plt.style.use('dark_background')
    
    fig = plt.figure(num=1,figsize=(11,10),frameon= False )#num=1 trouver un facon de fermer limage
    
    

    
    plt.hexbin(x,y,z,gridsize=(dimx,dimy-1),linewidths=0.1,cmap='winter')#linewidths=0)
    
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
    
    
    #plt.title(r'Rayon du flocon: '+str(rayon/2)+' $\mu$m', {'color': 'k', 'fontsize': 20})
    # plt.xlabel(r'Rayon du flocon: '+str(trusize/2)+' $\mu$m', {'color': 'k', 'fontsize': 75})
    

    #plt.legend(['A simple line'])
    
    path = os.getcwd()
    my_path = os.path.join(path,"Data_save","Images",'testfloconhex_'+'mx'+str(dimx)+'my'+str(dimy)+f+str(k)+'.png')
    fig.savefig(str(my_path),dpi=300)
    
    plt.close(fig)
    return dimx,dimy





#%% Main commence ici
#%% Ouvre le ficher et load les matrices de glace et de temps

def getData(cwd,dataPath,frametPath):
    dataFolder = os.listdir(cwd + dataPath)
    data = []
    for i in range(0,len(dataFolder)) :
        img = np.load(cwd + dataPath +"\\"+ "frameice"+str(i)+".npy")
        data.append((img[:,:,:].astype(np.uint8)))
    
    
    frametFolder = os.listdir(cwd + frametPath)
    for filename in frametFolder:  
        if filename == 'framet.npy':
            framet = np.load(cwd + frametPath +"\\"+ filename)
        else:   
            print("ERROR: No framet!!")   
    return data,framet

#Matrices de glace et veteur de temps pour chaque frame
data,framet = getData(cwd,dataPath,frametPath)
#%% Transfrome le vecteur de glace, pour avoir des
#   espacements égaux

def createImages(data,framet,numberOfFrames):
    #determine le nombre de frame que dure chaque image
    frameMultiplier = (framet/min(framet)).astype(np.uint32)
    frameMultiplier = frameMultiplier.tolist()
    frameVec = [None]*len(frameMultiplier)
    
    #créé une vecteur avec les index de frames
    frameVec[0] = 0
    frameVec[1] = frameMultiplier[0]
    for i in range(2,len(frameMultiplier)):
        frameVec[i] = frameVec[i-1] + frameMultiplier[i-1]
    
    #créé un vecteur avec tout les images séprarées
    #par un pas de temps égal
    imagesVec =  [None]*frameVec[-1]  
    for i in range(1,len(frameVec)):
        for j in range(frameVec[i-1],frameVec[i]):
            imagesVec[j] = data[i-1]
            
    #Garde seulement 25 images également séparée
    step = int(len(imagesVec)/numberOfFrames)
    images = []
    for i in range(0,len(imagesVec)):     
        if i % step ==0:
            images.append(imagesVec[i])
    return images,step

#Vecteur avec les données, ainsi que le nombre de frames entre les images
images,step = createImages(data,framet,numberOfFrames)


#%% Utilise Hexaplot pour faire et sauvegarder la figure

def plotFigures(images):        
    # plt.rcParams["figure.figsize"] = (30,30)
    if os.path.exists(os.path.join(cwd,"Data_save","Images")) == False:
        os.mkdir(os.path.join(cwd,"Data_save","Images"))
            
    for k in reversed(range(0,len(images))):
        print('Traitement de l''image # '+str(k)) 
        matflocon=np.sum(images[k],axis=2).astype(bool).astype(np.uint8)
        if k==(len(images)-1):
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
                
                if True:
                    index_flocon = np.where(matflocon >= 1)
                else:
                    index_flocon = np.where(matflocon >= 2)
        
            # flocon = hexagonal[:,index_flocon[0],index_flocon[1]]
        
            maxx=np.round(max(hexagonal[0,index_flocon[0],index_flocon[1]]))
            maxy=np.round(max(hexagonal[1,index_flocon[0],index_flocon[1]]))
            minx=np.round(min(hexagonal[0,index_flocon[0],index_flocon[1]]))
            miny=np.round(min(hexagonal[1,index_flocon[0],index_flocon[1]]))
        
            xmm=[minx,maxx]
            ymm=[miny,maxy]
        
        
       
        # matflocon=images[k][:,:,6]
        plt.figure()
        #plt.imshow(matflocon,interpolation='spline16',cmap='viridis') #Pour visualisation de la matrice
        dimx,dimy = hexplot(matflocon,k,xmm,ymm) 
    return dimx,dimy

dimx,dimy = plotFigures(images)
#%% Créé et sauvegare un video

path = os.path.join(cwd,"Data_save","Images")

def makeVideoFromImages(path,dimx,dimy):
    folder = os.listdir(path)
    img_array = []
    for k in range(0,len(folder)):
        print("Video: frame {0:} of {1:}".format(k+1,len(folder)))
    
        
        my_path = os.path.join(path,'testfloconhex_'+'mx'+str(dimx)+'my'+str(dimy)+'frame'+str(k)+'.png')
        # my_path = os.path.join(path,"frame"+str(k))
        img = cv2.imread(str(my_path))
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)
        
    
    height, width, layers = img_array[0].shape
    size = (width,height)
    
    out = cv2.VideoWriter(cwd+'\\'+'VideoDuFlocon.avi',cv2.VideoWriter_fourcc(*'DIVX'), 17, size)
    
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()       

makeVideoFromImages(path,dimx,dimy)    

#%% Lit les images sauvegarder
#   Pour en faire une liste

del images 
#%%
dataPath = r"\Data_save\Images"

def loadPLottedImages(cwd,dataPath):
    dataFolder = os.listdir(cwd + dataPath)
    images = []
    for k in range(0,len(dataFolder)):
        my_path = os.path.join(cwd+dataPath,'testfloconhex_'+'mx'+str(dimx)+'my'+str(dimy)+'frame'+str(k)+'.png')
        img = cv2.imread(my_path)
        img = img[:,:,:]#[20:2050,250:2350,:]
        # img = img[250:1740,400:2650,:]
        
        images.append((img.astype(np.uint8)))
    
    
    return images


#Images hexplot non process
print('non')
images = loadPLottedImages(cwd,dataPath)
plt.imshow(images[-1])


#%% Traitement d'image pour obetnir l'aire remplie
#  Puis un edge detection pour le perimetre

def preProcessImages(images):
    gray = [None]*(len(images))
    area = [None]*(len(images))
    perimeter = [None]*(len(images))
    
    i=0 # index zéro est différent
    gray[i] = cv2.cvtColor(images[i],cv2.COLOR_BGR2GRAY)
    area[i] = np.where(gray[i]<50,0,gray[i]).astype(bool).astype(np.uint8)*255
    perimeter[i]=cv2.Canny(area[i],1,1).astype(np.uint8)
    # début de la boucle
    for i in range(1,len(images)):
        # on converte en grayscale
        gray[i] = cv2.cvtColor(images[i],cv2.COLOR_BGR2GRAY)
        # On remplie le flocon et tout le reste est 0
        area[i] = np.where(gray[i]<50,0,gray[i]).astype(bool).astype(np.uint8)*255
        # on fait un edge detection
        perimeter[i]  = cv2.Canny(area[i],1,1).astype(np.uint8)
    
    plt.imshow(area[-1],cmap = 'gray')      
    
    plt.subplots(figsize=[40,30])
    plt.subplot(121)
    plt.imshow(area[-1],cmap = 'gray'), plt.xticks([]), plt.yticks([])
    plt.title('Matrice de surface', fontsize=40)
    
    plt.subplot(122)
    plt.imshow(perimeter[-1],cmap = 'gray'), plt.xticks([]), plt.yticks([])
    plt.title('Matrice de périmètre', fontsize=40)
    
    plt.savefig('Images')
    plt.show()
    return area,perimeter

#Matrices d'aires et de perimetre
area,perimeter = preProcessImages(images)

#Conversion en seconde et en metres

    
#%% On calcule le peri et aire en faisant une sommation
#  Puis sauvegarde une matrice pour l'analyse par Dider

def calulatePerimeterAndArea(perimeter,area,timeVec):
    perimeterVec = [None]*len(area)
    areaVec = [None]*len(area)
    
    for i in range(0,len(area)):
        perimeterVec[i] = np.sum(perimeter[i]/255)
        areaVec[i] = np.sum(area[i]/255)
    
 
    pixel2meter = np.sqrt((19*6*np.sqrt(3)*(5*10**(-6))**2)/areaVec[0]) #metres/pixel


    perimeterVec = np.array(perimeterVec)*pixel2meter
    areaVec = np.array(areaVec)*pixel2meter**2   
    
    #Sauvegarde des données pour donner à Didier
    DATA = [timeVec,perimeterVec,areaVec]
    np.save('DATA',DATA)
    # os.rmdir(os.path.join(cwd,"Data_save","Images"))
    
frame2seconds =  min(framet)*step #secondes/frame
timeVec = np.array(range(0,len(images)))*frame2seconds   
    
calulatePerimeterAndArea(perimeter,area,timeVec)
# END

#%%
#################################################
#################################################
#################################################
#################################################




#%% Lecture des matrices par Didier
DATA = np.load('DATA.npy')
[timeVec,perimeterVec,areaVec] = DATA

#%% Analyse des données
perimeterGrowth = [None]*len(images)
areaGrowth = [None]*len(images)
periOverArea  = [None]*len(images)
periOverArea2  = [None]*len(images)
   
for i in range(0,len(images)):   
    periOverArea[i] = (perimeterVec[i])**1/areaVec[i]
    periOverArea2[i] = (perimeterVec[i])**2/areaVec[i]
    if i == 0:
        perimeterGrowth[i] = 0
        areaGrowth[i] = 0
    else:
        perimeterGrowth[i] = (perimeterVec[i]-perimeterVec[i-1])/(timeVec[1]-timeVec[0])
        areaGrowth[i] = (areaVec[i] - areaVec[i-1])/(timeVec[1]-timeVec[0])

#%%
plt.subplots(figsize = [14,10])
plt.subplot(221)
plt.scatter(timeVec , np.array(perimeterGrowth)*(10**6))
plt.title('Croissance en périmètre')
plt.ylabel('Croissance [µm/s]')  
plt.xlabel('Temps [s]') 

plt.subplot(222)
plt.scatter(timeVec, np.array(areaGrowth)*((10**6)**2))
plt.title('Croissance en surface')
plt.ylabel('Croisance [µm^2/s]')  
plt.xlabel('Temps [s]') 
 
plt.subplot(223)
plt.scatter(timeVec , np.array(periOverArea)) 
plt.title('Ratio P/S')
plt.ylabel('Valeur du ratio')  
plt.xlabel('Temps [s]') 


plt.subplot(224)
plt.scatter(timeVec , np.array(periOverArea2))  
plt.title('Ratio P^2/S') 
plt.ylabel('Valeur du ratio')  
plt.xlabel('Temps [s]') 

plt.savefig('Donnees')
plt.show()        