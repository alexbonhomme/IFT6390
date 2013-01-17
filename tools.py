# -*- coding: utf-8 -*-
import numpy as np
import sys
import logging as log
import os
import viola_jones as vj
import logging as log
import pylab

"""
	Importe les images liste "filename" et les convertient en vecteurs
"""
def loadImageData( trainTest="train", categorie="ORL"):

	import Image as im

    # recupere chemins et indices de classes
	(listeLFW, listeORL) = cheminsToLoad(trainTest)

	if categorie=="LFW":
		log.debug("Chargement des données LFW")
		imageList = np.array( listeLFW )
		# Recuperation des valeurs depuis les images avec un recentrage sur les visages
		vj_model = vj.Viola_Jones()
		return vj_model.detections_faces_list(imageList[:,1], imageList[:, 0].astype(int))

	elif categorie=="ORL":
		log.debug("Chargement des données ORL")

		imageList = np.array( listeORL )

		# Recuperation des valeurs depuis les images sans aucune transformation
		faces_list = []
		compute_nb_class = []
		for i in xrange( len(imageList[:,1]) ):
			# calcul du nombre de classes
			if( not imageList[i, 0] in compute_nb_class ):
				compute_nb_class.append( imageList[i, 0] )
			
			img = im.open(imageList[i,1])
			faces_list.append( list(img.getdata()) )

		return np.transpose(faces_list), imageList[:, 0].astype(int), len(compute_nb_class)

"""
        Modifie les couleurs de l'image en nuances de gris
"""
def changeToGreyTot():

	fichier = open("lfwNames.txt")
	listeFiles = fichier.read().split('\n')
	listeFiles = listeFiles[2:-1]
	
	for i in range(len(listeFiles)):
		changeToGrey(listeFiles[i])
		print(i)
		
"""

"""
def changeToGrey(filename):

	import Image
	from pygame import image
	import PIL
	# Recuperation resolution
	image = image.load(filename)
	resolution = (image.get_width(), image.get_height())
	
	# Modification pixels
	img = Image.open(filename)
	pix = img.load()
	for i in range(resolution[0]):
		for j in range(resolution[1]):
			gris = int(round(0.299*pix[i,j][0]+0.587*pix[i,j][1]+0.114*pix[i,j][2]))
			pix[i,j] = (gris,gris,gris)
	img.save(filename)

"""
        Recupere la liste des images de type precise ('lfw' ou 'orl')
"""
def listPictures(indiceSeparation, liste, type="Databases/LFW/lfw"):
		import os, mimetypes, random
		classesTrain = 0
		classesTest = 0
		exemplesTrain = 0
		exemplesTest = 0
		with_extension = 1
 
		path = os.path.join( os.getcwd(), type)
		finalTrain = []
		finalTest = []
		
		if type == "Databases/LFW/lfw" :
			borneMinTest = 2
			borneMaxTest = 50
		elif type ==  "Databases/orl_faces" : 
			borneMinTest = 1
			borneMaxTest = 9

		for root, dirs, files in os.walk(path):
			# get length of path
			if( root is path ) :
				count = len(root)
 
			if(files):
				compteur = 0
				ajoutTrain = 1
				ajoutTest = 1
				random.shuffle(files)
				if len(files) >= ( indiceSeparation + borneMinTest ):
					for f in files:
						if int(with_extension) == 0:
							f = f.split('.')[0]
						if f != "README" and indiceSeparation == 0:
							finalTrain.append(type+root[count:]+"/"+f)	
						elif f != "README" and compteur < indiceSeparation and (root[count+1:] in liste or len(liste) == 0):
							if ajoutTrain == 1:
								classesTrain += 1
							exemplesTrain += 1
							finalTrain.append(type+root[count:]+"/"+f)	
							ajoutTrain = 0
							compteur += 1
						elif f != "README" and compteur >= indiceSeparation and (root[count+1:] in liste or len(liste) == 0):
							if compteur < (indiceSeparation + borneMaxTest) :
								if ajoutTest == 1:
									classesTest += 1
								exemplesTest += 1
								finalTest.append(type+root[count:]+"/"+f)
								ajoutTest = 0
								compteur += 1
					
             
		return (finalTrain, finalTest, classesTrain, classesTest, exemplesTrain, exemplesTest)  

"""
        Recupere la liste des dossiers de type LFW contenant au moins nbMaxImages+1 images, ainsi que le nombre d'images présentes
"""
def constructLfwNamesCurrent(nbMaxImages):
		import os, mimetypes, random

		path = os.path.join( os.getcwd(),"Databases/LFW/lfw")
		liste = []
		dossier = ""

		for root, dirs, files in os.walk(path):
			# get length of path
			if( root is path ):
				count = len(root)
			if(dirs):
				dossier = dirs
			if(files):
				nbImages=0
				if len(files) > nbMaxImages:
					liste.append([root[count+1:],len(files)])
		liste.sort()
		fichier = file("Databases/LFW/lfw-names_current.txt",'w')
		for i in range(len(liste)):
			fichier.write(liste[i][0]+'\t'+str(liste[i][1])+'\n')
             

"""
        Sauvegarde les noms des images dans lfwNames.txt et orlNames.txt
"""
def picturesDictionaryConstruction():
	liste = listPictures(0,[])[0]
	liste.sort()
	fichier=file('./lfwNames.txt','w')
	fichier.write("jpg\n")
	fichier.write(str(len(liste))+'\n')
	for i in range(len(liste)):
		fichier.write(liste[i]+'\n')
	fichier.close()
	liste = listPictures(0,[],"Databases/orl_faces")[0]
	liste.sort()
	fichier = file('./orlNames.txt','w')
	fichier.write("pgm\n")
	fichier.write(str(len(liste))+'\n')
	for i in range(len(liste)):
		fichier.write(liste[i]+'\n')
	fichier.close()

"""
        Construit les fichiers train.txt et test.txt contenant les noms des images utilisées
"""
def trainAndTestConstruction(nbTrain):
	fichier = open("Databases/LFW/lfw-names_current.txt",'r')
	lignes = fichier.read().split('\n')
	lignes = lignes[:-1]  #On supprime le dernier element ''
	fichierTrain = file('trainFile','w')
	fichierTest = file('testFile','w')
	nom = []
	nbMax = []
	for i in range(len(lignes)):
		(mot, nb) = (lignes[i].split('\t')[0], int(lignes[i].split('\t')[1]))
		nbMax.append(nb)
		nom.append(mot)
	
	(listeTrain, listeTest, 
	 classesTrainLFW, classesTestLFW,
	 exemplesTrainLFW, exemplesTestLFW) = listPictures(nbTrain, nom)
	
	(listeTrainORL, listeTestORL,
	 classesTrainORL, classesTestORL,
	 exemplesTrainORL, exemplesTestORL) = listPictures(np.min((nbTrain,9)),[], "Databases/orl_faces")
	
	listeTrain.sort()
	listeTest.sort()
	listeTrainORL.sort()
	listeTestORL.sort()

	fichierTrain.write(str(classesTrainLFW)+' '+str(exemplesTrainLFW)+'\n'+str(classesTrainORL)+' '+str(exemplesTrainORL)+'\n')
	fichierTest.write(str(classesTestLFW)+' '+str(exemplesTestLFW)+'\n'+str(classesTestORL)+' '+str(exemplesTestORL)+'\n')
	
	for i in range(len(nom)):
		fichierTrain.write(nom[i]+' '+str(np.min((nbTrain,nbMax[i])))+'\n')
		nbTest = nbMax[i] - nbTrain
		if nbTest>0:
			fichierTest.write(nom[i]+' '+str(nbTest)+'\n')
	
	for i in range(len(listeTrain)):
		fichierTrain.write(listeTrain[i]+'\n')
	
	for i in range(len(listeTest)):
		fichierTest.write(listeTest[i]+'\n')	
	
	for i in range(len(listeTrainORL)):
		fichierTrain.write(listeTrainORL[i]+'\n')
	
	for i in range(len(listeTestORL)):
		fichierTest.write(listeTestORL[i]+'\n')
	
	fichierTrain.close()
	fichierTest.close()
	return ( classesTrainLFW, classesTrainORL )

"""
	Recupere les chemins depuis trainFile et testFile, ainsi que les classes. Utilisee par loadImageData()
"""
def cheminsToLoad(trainTest="train"):
	fichier = open(trainTest+"File")
	contenu = fichier.read().split('\n')[:-1]
	fichier.close()
	(nbClassesLFW, nbExemplesLFW) = (int(contenu[0].split(' ')[0]), int(contenu[0].split(' ')[1]))
	(nbClassesORL, nbExemplesORL) = (int(contenu[1].split(' ')[0]), int(contenu[1].split(' ')[1]))
	
	
	nbExemplesPClasseORL = nbExemplesORL/nbClassesORL
	nbExemplesPClasseLFW = nbExemplesLFW/nbClassesLFW
	(listeLFW, listeORL) = ([],[])
	listeCheminsLFW = contenu[2+nbClassesLFW:2+nbClassesLFW+nbExemplesLFW]
	listeCheminsORL = contenu[2+nbExemplesLFW+nbClassesLFW:]
	
	classe = 1
	for i in range(len(listeCheminsLFW)):
		listeLFW.append([classe, listeCheminsLFW[i]])
		if (i+1)%(nbExemplesPClasseLFW) == 0:
			classe += 1
	
	classe = 1
	for i in range(len(listeCheminsORL)):
		listeORL.append([classe, listeCheminsORL[i]])
		if (i+1)%(nbExemplesPClasseORL) == 0:
			classe += 1
	return (listeLFW, listeORL)


"""
    Calcul la distance de Minkowski entre un vecteur et une matrice
"""
def minkowski_mat(x, Y, p=2.0, axis=0):
    x = x.reshape( x.shape[0], 1 )
    return np.sum( np.abs( x - Y )**p, axis=axis )**(1.0/p)

# slow algo
def euclide_mat(x, Y):
    out = []
    for p in range(0, int( Y.shape[1] )):
	    deltaSq = 0
	    for i in range(0, int( Y.shape[0] )):
		    delta = x[i] - Y[i, p]
		    deltaSq += delta*delta  # Eclidean distance
	    out.append( np.sqrt(deltaSq) )

    return np.array( out )

"""
    Noyau gaussien
"""
def gaussianKernel(X, x, theta):
    return np.exp( -((X - x)**2) / (theta**2) )

#TODO check this
def softKernelDist( dist, d=2, theta=0.5 ):
    return (1/( ((2*np.pi)**(d/2)) * (theta**d) )) * np.exp((-1/2) * ((dist) / (theta**2)))
  
"""
    Compte le nombre de classe a partir du vecteur de labels/targets
"""
def countClass( targets ):
    m = []
    for i in range(0, int( targets.shape[0] )):
        if( not targets[i] in m ):
            m.append( targets[i] )
    return len(m)

"""
    Non linéarité de type softmax
"""
def softmax(x):
    e = np.exp(x)
    return e / np.sum( e )

def softmaxMat(X):
    out = np.zeros(X.shape)
    for i in xrange(X.shape[1]):
        e = np.exp(X[:,i])
        out[:,i] = e / np.sum( e )
    
    return out

######################################################################################
#
#   Affichage plusieurs courbes sur la même figure
#
#   x: liste des coordonnées en x
#   y: liste des coordonnées en y
#   cType: liste des représentations associées aux courbes
#   legend: liste des légendes associées aux courbes
#   xlim, ylim: bornes en x, y
#   xlabel, ylabel: labels en x, y
#   title: titre de la figure
#   bGrid: active/desactive l'affichage de la grille
#   bDisplay: active/desactive l'affichage des courbes
#   filename: si renseigné, enregistre la figure dans un fichier nommé <filename.png>
#
######################################################################################
def drawCurves(x, y, cType, legend="", xlim="", ylim="", xlabel="", ylabel="", title="", bGrid=True, bDisplay=True, filename=""):
    if len(x) != len(y) != len(cType):
        print "Attention: Les deux listes doivent avoir la même taille."
        return -1
        
    pylab.plot(x, y, cType)
        #pylab.annotate('Min: '+ str(np.min(y[i])),
                       # xy=(np.argmin(y[i]), np.min(y[i])))
                        #arrowprops=dict(arrowstyle='->'))
    pylab.grid(bGrid)
	
	# titre / labels / legende
    if title != "":
	    pylab.title(title)
    if xlabel != "":
	    pylab.xlabel(xlabel)
    if ylabel != "":
	    pylab.ylabel(ylabel)
    if legend != "":
	    pylab.legend(legend)
	
	# bornes
    if xlim != "":
	    pylab.xlim(xlim)
    if ylim != "":
	    pylab.ylim(ylim)
    
    # sauvegarde de la courbe
    if filename != "":
	    filename += '.png'
	    print 'On sauvegarde la figure dans', filename
	    pylab.savefig(filename,format='png')

    # affichage
    if bDisplay:
        pylab.show()

    # close
    pylab.clf()


"""
        Complete la liste de valeurs avec un pas de 1, de sorte à ce que toutes les valeurs soient >1
"""
def completion(liste, nbElements):

	liste.sort()
	compteur = len(liste)
	valeur = liste[0] - 1
	while valeur > 0 and compteur < nbElements :
		liste.append(valeur)
		compteur += 1
		valeur -= 1
	liste.sort()
	valeur = liste[-1] + 1
	while compteur < nbElements :
		liste.append(valeur)
		compteur += 1
		valeur += 1
		

